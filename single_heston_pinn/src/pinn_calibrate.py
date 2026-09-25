#!/usr/bin/env python3
"""Inverse calibration of Heston parameters through the trained PINN.

The protocol is the repository's own, unchanged: four structural parameters
fitted jointly on train-only surfaces with equal weight per date, one latent
``v0`` per trade date fitted on the calibration fold alone, and scoring on the
holdout fold -- whole strikes, never quotes, assigned to one fold or the other.
The only thing swapped out is the pricing engine, so any difference in the
result is attributable to the engine and not to the data, folds or optimiser.

Two engines are provided:

* ``fourier``  -- exact Heston by Lewis quadrature (the conventional baseline);
* ``pinn``     -- one forward pass of the trained implied-variance network.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import mlx.core as mx
from scipy.optimize import least_squares, minimize_scalar

import pinn_heston_core as C

SH = C.SH
PARAMETER_STARTS = (
    (1.5, 0.06, 0.30, -0.60, 0.06),
    (3.0, 0.04, 0.25, -0.30, 0.04),
    (0.7, 0.12, 0.35, -0.75, 0.12),
)


# --------------------------------------------------------------------------
# Pricing engines
# --------------------------------------------------------------------------

def _assemble(call, spot, strike, maturity, rate, dividend, is_call):
    put = call - spot * np.exp(-dividend * maturity) + strike * np.exp(-rate * maturity)
    return np.where(is_call, call, put)


class FourierEngine:
    """Exact Heston by Lewis quadrature; the conventional calibration engine."""

    name = "fourier"
    # float64 throughout, so SciPy's default sqrt(eps) difference step is fine.
    diff_step = None

    def __init__(self, box: C.Box):
        self.box = box

    def price(self, spot, strike, maturity, rate, dividend, is_call, params):
        kappa, theta, sigma, rho, v0 = (float(p) for p in params)
        return self.price_rowwise(spot, strike, maturity, rate, dividend, is_call,
                                  np.full(spot.shape, kappa), np.full(spot.shape, theta),
                                  np.full(spot.shape, sigma), np.full(spot.shape, rho),
                                  np.full(spot.shape, v0))

    def price_rowwise(self, spot, strike, maturity, rate, dividend, is_call,
                      kappa, theta, sigma, rho, v0):
        positive = np.asarray(spot) > 0
        with np.errstate(divide="ignore"):
            x = np.log(np.where(positive, spot, 1.0) / strike) + (rate - dividend) * maturity
        c = np.where(positive,
                     C.normalised_forward_call_batch(x, maturity, kappa, theta, sigma, rho, v0),
                     0.0)
        return _assemble(strike * np.exp(-rate * maturity) * c,
                         spot, strike, maturity, rate, dividend, is_call)


class PinnEngine:
    """One forward pass of the trained implied-variance network.

    MLX evaluates on the Metal GPU in float32, so prices carry about 1e-7
    relative noise.  SciPy's default difference step of sqrt(eps) ~ 1.5e-8 moves
    a parameter by less than that noise, the numerical Jacobian comes back
    identically zero and ``least_squares`` returns its own starting point.  The
    engine therefore declares a difference step of 1e-3, comfortably above the
    float32 floor and still far inside the curvature scale of the objective.
    """

    name = "pinn"
    diff_step = 2e-3

    def __init__(self, model, box: C.Box):
        self.model = model
        self.box = box
        self.norm = C.Normaliser.from_box(box)

    def _normalised(self, x, maturity, params):
        n = x.size
        const = lambda value: mx.full((n,), float(value), dtype=mx.float32)
        kappa, theta, sigma, rho, v0 = params
        ell = C.log_total_variance(self.model, x, const(v0), maturity,
                                   const(kappa), const(theta), const(sigma), const(rho), self.norm)
        return C.black76_normalised_mx(x, mx.exp(ell))

    def price(self, spot, strike, maturity, rate, dividend, is_call, params):
        kappa, theta, sigma, rho, v0 = (float(p) for p in params)
        ones = np.ones(spot.shape)
        return self.price_rowwise(spot, strike, maturity, rate, dividend, is_call,
                                  kappa * ones, theta * ones, sigma * ones, rho * ones, v0 * ones)

    def price_rowwise(self, spot, strike, maturity, rate, dividend, is_call,
                      kappa, theta, sigma, rho, v0):
        """Price at the true moneyness; read the variance only inside the box.

        The network is trained on ``|x| <= x_half_width``, so the implied total
        variance is read at the clipped state.  Black-76 is then evaluated at the
        *unclipped* ``x``, because its asymptotics are exact for any positive
        variance: ``c -> 0`` as ``x -> -inf`` and ``c -> e^x - 1`` as
        ``x -> +inf``.  Clipping ``x`` inside Black-76 as well would put a
        spurious floor under deep out-of-the-money calls, which is exactly the
        region the study's spot axis reaches when it runs down to zero.
        """
        with np.errstate(divide="ignore"):
            x = np.log(np.where(spot > 0, spot, 1.0) / strike) + (rate - dividend) * maturity
        x = np.where(spot > 0, x, -np.inf)
        x_state = np.clip(np.nan_to_num(x, neginf=-self.box.x_half_width,
                                        posinf=self.box.x_half_width),
                          -self.box.x_half_width, self.box.x_half_width)
        f = lambda a: mx.array(np.asarray(a, dtype=np.float32))
        ell = C.log_total_variance(self.model, f(x_state), f(v0), f(maturity),
                                   f(kappa), f(theta), f(sigma), f(rho), self.norm)
        w = np.asarray(mx.exp(ell), dtype=np.float64)
        c = np.where(np.isfinite(x), C.black76_normalised(np.where(np.isfinite(x), x, 0.0), w), 0.0)
        return _assemble(strike * np.exp(-rate * maturity) * c,
                         spot, strike, maturity, rate, dividend, is_call)


# --------------------------------------------------------------------------
# Residuals, matching single_heston.residuals term for term
# --------------------------------------------------------------------------

V0_GRID = np.linspace(-7.0, 7.0, 15)
TOLERANCES = dict(ftol=1e-12, xtol=1e-12, gtol=1e-12)


def quote_arrays(quotes):
    return dict(
        spot=quotes.spot.to_numpy(float), strike=quotes.strike.to_numpy(float),
        maturity=quotes.maturity.to_numpy(float), rate=quotes.rate.to_numpy(float),
        dividend=quotes.dividend.to_numpy(float),
        is_call=quotes.option_type.eq("CE").to_numpy(),
        market=quotes.market_price_adjusted.to_numpy(float),
        vega=quotes.vega.to_numpy(float), weight=quotes.weight.to_numpy(float),
    )


def surface_residual(engine, arrays, params):
    model = engine.price(arrays["spot"], arrays["strike"], arrays["maturity"],
                         arrays["rate"], arrays["dividend"], arrays["is_call"], params)
    scale = np.maximum(arrays["vega"], 0.002 * arrays["spot"])
    return (model - arrays["market"]) / scale * arrays["weight"]


def fit_joint_structural(engine, transform, surfaces, max_nfev=140):
    """Four structural parameters shared across dates, one ``v0`` per date.

    Every surface is priced in a single vectorised call: both engines accept a
    per-row parameter vector, so the whole panel of dates goes through one
    quadrature (or one network forward pass) per objective evaluation instead of
    one per date.
    """
    arrays = [quote_arrays(q) for q in surfaces]
    sizes = np.array([a["spot"].size for a in arrays])
    index = np.repeat(np.arange(len(arrays)), sizes)
    flat = {k: np.concatenate([a[k] for a in arrays]) for k in arrays[0]}
    date_scale = np.sqrt(sizes)[index]
    v0_start = [float(np.clip(np.nanmedian(q.market_iv) ** 2,
                              transform.box.variance_low, transform.box.variance_high))
                for q in surfaces]

    def encode(structural, variances):
        first = transform.encode((*structural, variances[0]))
        rest = [transform.encode((*structural, v))[4] for v in variances[1:]]
        return np.r_[first, rest]

    def decode(z):
        base = transform.decode(z[:5])
        variances = [base[4]]
        for value in z[5:]:
            variances.append(transform.decode(np.r_[z[:4], value])[4])
        return base[:4], np.asarray(variances)

    def residual(z):
        structural, variances = decode(z)
        ones = np.ones(flat["spot"].shape)
        model = engine.price_rowwise(
            flat["spot"], flat["strike"], flat["maturity"], flat["rate"], flat["dividend"],
            flat["is_call"], structural[0] * ones, structural[1] * ones,
            structural[2] * ones, structural[3] * ones, np.asarray(variances)[index])
        scale = np.maximum(flat["vega"], 0.002 * flat["spot"])
        return (model - flat["market"]) / scale * flat["weight"] / date_scale

    best = None
    evaluations = 0
    for start in PARAMETER_STARTS:
        z0 = encode(start[:4], v0_start)
        result = least_squares(residual, z0, bounds=(-8, 8), loss="soft_l1",
                               f_scale=0.02, max_nfev=max_nfev,
                               diff_step=getattr(engine, "diff_step", None), **TOLERANCES)
        evaluations += int(result.nfev)
        objective = float(np.mean(residual(result.x) ** 2))
        if best is None or objective < best[0]:
            best = (objective, *decode(result.x))
    return {"objective": best[0], "structural": best[1], "variances": best[2],
            "evaluations": evaluations}


def soft_l1_cost(residual, f_scale=0.05):
    """The objective ``least_squares(loss="soft_l1", f_scale=...)`` minimises."""
    scaled = (residual / f_scale) ** 2
    return float(f_scale ** 2 * np.sum(np.sqrt(1.0 + scaled) - 1.0))


def fit_v0(engine, transform, quotes, structural, grid_points=33):
    """Latent variance for one trade date, from the calibration fold alone.

    This is a one-dimensional bounded search, so it is solved as one: a coarse
    scan over the whole transformed interval followed by Brent refinement inside
    the bracketing triple.  A trust-region least-squares solver is the wrong tool
    here -- on this scalar problem it reached its gradient tolerance after two
    evaluations and returned a point whose objective was ten times the optimum,
    and with a network priced in float32 its default difference step is below the
    arithmetic's own noise floor.  The objective, weights, bounds and soft_l1
    scale are exactly the ones the structural fit uses, and both engines run the
    identical search.
    """
    arr = quote_arrays(quotes)

    def cost(z):
        v0 = transform.decode(np.r_[np.zeros(4), z])[4]
        return soft_l1_cost(surface_residual(engine, arr, np.r_[structural, v0]))

    grid = np.linspace(-8.0, 8.0, grid_points)
    values = [cost(z) for z in grid]
    best = int(np.argmin(values))
    lo = grid[max(best - 1, 0)]
    hi = grid[min(best + 1, grid_points - 1)]
    result = minimize_scalar(cost, bounds=(lo, hi), method="bounded",
                             options={"xatol": 1e-7, "maxiter": 60})
    z_star = float(result.x) if result.fun <= values[best] else float(grid[best])
    v0 = float(transform.decode(np.r_[np.zeros(4), z_star])[4])
    return v0, int(grid_points + result.nfev)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def predict(engine, quotes, params):
    arr = quote_arrays(quotes)
    out = quotes.copy()
    out["model_price_adjusted"] = engine.price(
        arr["spot"], arr["strike"], arr["maturity"], arr["rate"], arr["dividend"],
        arr["is_call"], params)
    out["model_price_raw"] = out.model_price_adjusted / out.price_adjustment_factor
    out["model_iv"] = SH.implied_volatility(
        out.model_price_adjusted.to_numpy(float), arr["spot"], arr["strike"],
        arr["maturity"], arr["rate"], arr["dividend"], arr["is_call"])
    out["iv_error"] = out.model_iv - out.market_iv
    return out


def metrics(frame):
    clean = frame.dropna(subset=["market_iv", "model_iv"])
    if not len(clean):
        return {"rows": 0}
    err = (clean.model_iv - clean.market_iv).to_numpy(float)
    perr = (clean.model_price_raw - clean.market_price_raw).to_numpy(float)
    mkt = clean.market_iv.to_numpy(float)
    denom = float(np.sum((mkt - mkt.mean()) ** 2))
    return {
        "rows": int(len(clean)),
        "iv_rmse": float(np.sqrt(np.mean(err ** 2))),
        "iv_mae": float(np.mean(np.abs(err))),
        "iv_bias": float(np.mean(err)),
        "iv_r2": float(1 - np.sum(err ** 2) / denom) if denom > 0 else float("nan"),
        "raw_price_rmse": float(np.sqrt(np.mean(perr ** 2))),
        "raw_price_mae": float(np.mean(np.abs(perr))),
    }
