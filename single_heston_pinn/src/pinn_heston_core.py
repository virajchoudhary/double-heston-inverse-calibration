#!/usr/bin/env python3
"""Physics-informed neural calibration of the one-factor Heston model.

Design
------
Heston is homogeneous of degree one in (spot, strike), so a call written on the
forward collapses onto a single scalar state.  With ``F = S exp((r-q)T)`` and
``x = log(F/K)``, the normalised forward call ``c = C / (K exp(-rT))`` obeys

    c_T = 0.5 v (c_xx - c_x) + rho sigma v c_xv + 0.5 sigma^2 v c_vv
          + kappa (theta - v) c_v

with terminal value ``c(x, v, 0) = (e^x - 1)^+``.  The network never predicts a
price directly.  It predicts the **implied total variance** ``w`` through

    log w = log T + log vbar(T) + 2 g(x, v, T, kappa, theta, sigma, rho)
    vbar(T) = theta + (v - theta) (1 - e^{-kappa T}) / (kappa T)

and the price is recovered analytically as ``c = Black76(x, w)``.  Three things
follow for free and exactly, with no boundary or terminal loss term:

* ``T -> 0`` gives ``w -> 0`` and Black76 collapses onto the payoff;
* ``x -> -inf`` gives ``c -> 0``; ``x -> +inf`` gives ``c -> e^x - 1``;
* ``max(e^x - 1, 0) <= c <= e^x`` holds pointwise, so no predicted price can
  violate the static no-arbitrage bounds.

``vbar`` is the exact Heston expected integrated variance per unit time, so
``g = 0`` reproduces the model exactly in the zero-vol-of-vol limit: the PDE
residual below is then identically zero.  The network only has to learn the
smile and skew correction on top of a physically exact backbone.

Substituting ``c = Black76(x, w)`` into the PDE and dividing by the strictly
positive Black vega removes the payoff singularity.  Writing ``l = log w``
removes every negative power of ``w``, which is what makes the residual safe in
the float32 arithmetic MLX uses on the GPU:

    R = w l_T
        - 0.5 v [ 2 + 2 (w/2 - x) l_x + A l_x^2 + w l_xx + w l_x^2 - w l_x ]
        - rho sigma v [ (w/2 - x) l_v + A l_x l_v + w l_xv + w l_x l_v ]
        - 0.5 sigma^2 v [ A l_v^2 + w l_vv + w l_v^2 ]
        - kappa (theta - v) w l_v
    A = x^2/2 - w^2/8 - w/2
"""

from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def _load_single_heston():
    spec = importlib.util.spec_from_file_location("single_heston", ROOT / "single_heston.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SH = _load_single_heston()
SQRT_2PI = math.sqrt(2.0 * math.pi)


# --------------------------------------------------------------------------
# Analytic reference layer (numpy / scipy, float64)
# --------------------------------------------------------------------------

_PANEL_EDGES = np.array([0.0, 1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1.0])
_P_NODES, _P_WEIGHTS = np.polynomial.legendre.leggauss(96)


def _panelled_grid(upper):
    """Geometric Gauss-Legendre panels on [0, upper].

    The Lewis integrand has structure near the origin and an exponential tail
    whose length scales like ``1 / sqrt(vbar T)``.  At one-day maturity with
    variance at the bottom of the box that tail runs past ``u = 4000``, and a
    single 400-node rule spends almost all of its nodes in the wrong place --
    which cost about 1e-3 of strike.  Eight geometric panels of 96 nodes hold the
    error below 1e-9 across the whole box.
    """
    upper = np.atleast_1d(np.asarray(upper, dtype=float))
    lo = upper[:, None, None] * _PANEL_EDGES[None, :-1, None]
    hi = upper[:, None, None] * _PANEL_EDGES[None, 1:, None]
    mid, half = 0.5 * (hi + lo), 0.5 * (hi - lo)
    u = (mid + half * _P_NODES[None, None, :]).reshape(upper.size, -1)
    w = (half * _P_WEIGHTS[None, None, :]).reshape(upper.size, -1)
    return u, w


def log_return_cf(u, maturity, params):
    """CF of ``log(F_T / F_0)`` under Heston, Albrecher "little trap" branch."""
    kappa, theta, sigma, rho, v0 = params
    iu = 1j * u
    b = kappa - rho * sigma * iu
    d = np.sqrt(b * b + sigma * sigma * (u * u + iu))
    g = (b - d) / (b + d)
    e = np.exp(-d * maturity)
    c = (kappa * theta / sigma ** 2) * ((b - d) * maturity - 2 * np.log((1 - g * e) / (1 - g)))
    big_d = ((b - d) / sigma ** 2) * ((1 - e) / (1 - g * e))
    return np.exp(c + big_d * v0)


def normalised_forward_call(x, maturity, params) -> np.ndarray:
    """Exact Heston ``c = E[(F_T/K - 1)^+]`` at log forward moneyness ``x``.

    Lewis' (2001) single integral on the ``Im(u) = -1/2`` contour,

        c = e^x - (e^{x/2} / pi) int_0^inf Re[ e^{+iux} phi(u - i/2) ] / (u^2 + 1/4) du

    The ``+iux`` sign is not cosmetic.  On this contour the Black-Scholes
    characteristic function is real and even in ``u``, so a flipped sign still
    reproduces Black-Scholes to machine precision while silently mirroring the
    Heston skew about the money.  The sign here is pinned down by agreement with
    the repository's independent Gauss-Laguerre ``P1``/``P2`` engine and by the
    PDE residual test in :func:`audit_reference_pricer`.

    This form is preferred over ``P1``/``P2`` only because it is a single
    integral with no ``1/(iu)`` pole, which makes it cheap to vectorise over tens
    of thousands of distinct parameter vectors at once.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    maturity = np.atleast_1d(np.asarray(maturity, dtype=float))
    if maturity.size == 1 and x.size > 1:
        maturity = np.full(x.shape, float(maturity.flat[0]))
    return normalised_forward_call_batch(x, maturity, *(float(p) for p in params[:4]),
                                         float(params[4])).reshape(x.shape)


def normalised_forward_call_batch(x, maturity, kappa, theta, sigma, rho, v0, chunk=4096):
    """Vectorised Lewis price: every row carries its own parameter vector.

    Same contour and quadrature as :func:`normalised_forward_call`, but the
    400-node Gauss-Legendre grid is rescaled per row so each row keeps its own
    adaptive upper limit.  Needed because the parameter screen, the anchor set
    and the accuracy audits all price tens of thousands of distinct parameter
    vectors at once.
    """
    x, maturity, kappa, theta, sigma, rho, v0 = (
        np.atleast_1d(np.asarray(a, dtype=float))
        for a in (x, maturity, kappa, theta, sigma, rho, v0)
    )
    n = max(a.size for a in (x, maturity, kappa, theta, sigma, rho, v0))
    x, maturity, kappa, theta, sigma, rho, v0 = (
        np.broadcast_to(a, (n,)) for a in (x, maturity, kappa, theta, sigma, rho, v0)
    )
    vbar = expected_integrated_variance_rate(v0, maturity, kappa, theta)
    upper = np.maximum(300.0, 60.0 / np.sqrt(np.maximum(vbar * maturity, 1e-9)))
    out = np.empty(n, dtype=float)
    for start in range(0, n, chunk):
        sl = slice(start, min(start + chunk, n))
        u, w = _panelled_grid(upper[sl])
        z = u - 0.5j
        iu = 1j * z
        b = kappa[sl, None] - rho[sl, None] * sigma[sl, None] * iu
        sig2 = sigma[sl, None] ** 2
        d = np.sqrt(b * b + sig2 * (z * z + iu))
        g = (b - d) / (b + d)
        e = np.exp(-d * maturity[sl, None])
        cc = (kappa[sl, None] * theta[sl, None] / sig2) * (
            (b - d) * maturity[sl, None] - 2 * np.log((1 - g * e) / (1 - g))
        )
        dd = ((b - d) / sig2) * ((1 - e) / (1 - g * e))
        phi = np.exp(cc + dd * v0[sl, None])
        integ = np.real(np.exp(1j * u * x[sl, None]) * phi / (u * u + 0.25))
        out[sl] = np.exp(x[sl]) - np.exp(0.5 * x[sl]) / math.pi * np.sum(w * integ, axis=1)
    return out


def implied_vol_batch(x, maturity, kappa, theta, sigma, rho, v0):
    price = normalised_forward_call_batch(x, maturity, kappa, theta, sigma, rho, v0)
    x = np.broadcast_to(np.atleast_1d(np.asarray(x, float)), price.shape)
    maturity = np.broadcast_to(np.atleast_1d(np.asarray(maturity, float)), price.shape)
    w = implied_total_variance(price, x)
    return np.sqrt(w / maturity), price, w


def atm_implied_vol(maturity, params) -> float:
    """Exact at-the-money-forward implied volatility; the realism screen."""
    c = float(normalised_forward_call(0.0, maturity, params)[0])
    w = float(implied_total_variance(np.array([c]), np.array([0.0]))[0])
    return math.sqrt(max(w, 0.0) / maturity) if np.isfinite(w) else np.nan


def black76_normalised(x, total_variance) -> np.ndarray:
    """``c = e^x N(d1) - N(d2)`` with ``d1,2 = x/sqrt(w) +/- sqrt(w)/2``."""
    from scipy.stats import norm

    x = np.asarray(x, dtype=float)
    w = np.maximum(np.asarray(total_variance, dtype=float), 1e-14)
    root = np.sqrt(w)
    d1 = x / root + 0.5 * root
    d2 = d1 - root
    return np.exp(x) * norm.cdf(d1) - norm.cdf(d2)


def implied_total_variance(price, x, lo=1e-12, hi=40.0, iterations=110) -> np.ndarray:
    """Invert Black-76 for total variance; NaN outside the no-arbitrage band."""
    price = np.asarray(price, dtype=float)
    x = np.asarray(x, dtype=float)
    intrinsic = np.maximum(np.exp(x) - 1.0, 0.0)
    upper = np.exp(x)
    valid = np.isfinite(price) & (price > intrinsic + 1e-13) & (price < upper - 1e-13)
    low = np.full(price.shape, lo)
    high = np.full(price.shape, hi)
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        model = black76_normalised(x, mid)
        low = np.where(model < price, mid, low)
        high = np.where(model >= price, mid, high)
    return np.where(valid, 0.5 * (low + high), np.nan)


def expected_integrated_variance_rate(v, maturity, kappa, theta) -> np.ndarray:
    """``vbar = E[int_0^T v_s ds] / T``, exact under Heston."""
    u = np.asarray(kappa, dtype=float) * np.asarray(maturity, dtype=float)
    ratio = np.where(np.abs(u) < 1e-7, 1.0 - 0.5 * u + u * u / 6.0, (1.0 - np.exp(-u)) / np.where(u == 0, 1.0, u))
    return np.asarray(theta, dtype=float) + (np.asarray(v, dtype=float) - np.asarray(theta, dtype=float)) * ratio


# --------------------------------------------------------------------------
# Domain
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Box:
    """Training box.

    Spot bounds stay in rupees; the PDE state is ``x = log(F/K)``.

    Vol-of-vol is not sampled directly.  It is sampled through the dimensionless
    Feller ratio ``eta = sigma / sqrt(2 kappa theta)``, so ``eta < 1`` is the
    Feller-regular regime and ``eta > 1`` the violating regime equity smiles
    usually need.  Sampling sigma independently of (kappa, theta) instead lets
    the variance process explode, which produces implied volatilities in the
    thousands of percent and a Fourier price that is not invertible.  The bounds
    below are the repository's own calibration bounds tightened to that rule.
    """

    x_half_width: float = 3.5
    variance_low: float = 0.005
    variance_high: float = 1.10
    maturity_low_days: float = 2.0
    maturity_high_days: float = 92.0
    kappa_low: float = 0.10
    kappa_high: float = 15.0
    theta_low: float = 0.005
    theta_high: float = 1.00
    feller_ratio_low: float = 0.05
    feller_ratio_high: float = 2.00
    sigma_floor: float = 0.02
    sigma_ceiling: float = 5.00
    rho_low: float = -0.95
    rho_high: float = 0.60
    rate_low: float = -0.10
    rate_high: float = 0.25
    dividend_low: float = -0.27
    dividend_high: float = 0.50

    @property
    def maturity_low(self) -> float:
        return self.maturity_low_days / 365.0

    @property
    def maturity_high(self) -> float:
        return self.maturity_high_days / 365.0

    def sigma_from_ratio(self, ratio, kappa, theta):
        raw = np.asarray(ratio) * np.sqrt(2.0 * np.asarray(kappa) * np.asarray(theta))
        return np.clip(raw, self.sigma_floor, self.sigma_ceiling)

    def feller_ratio(self, sigma, kappa, theta):
        return np.asarray(sigma) / np.sqrt(2.0 * np.asarray(kappa) * np.asarray(theta))

    def as_dict(self) -> dict:
        d = asdict(self)
        d["maturity_low_years"] = self.maturity_low
        d["maturity_high_years"] = self.maturity_high
        d["sigma_rule"] = "sigma = eta * sqrt(2 kappa theta), eta in [feller_ratio_low, feller_ratio_high]"
        return d


MONTH_SLICE_DAYS = (30.0, 60.0, 90.0)


ATM_VOL_LOW = 0.02
ATM_VOL_HIGH = 2.50


def draw_parameters(rng, n, box, screen=True, max_rounds=40):
    """Heston parameter draws, screened for economic realism.

    ``sigma`` is drawn through the Feller ratio ``eta = sigma / sqrt(2 kappa
    theta)``; drawing it independently of ``(kappa, theta)`` lets the variance
    process explode.  The screen then keeps only draws whose exact
    at-the-money implied volatility sits inside [2%, 250%] at one and three
    months.  Without it the box contains corners -- small ``kappa`` with ``eta``
    near two -- whose implied volatility runs to thousands of percent, which no
    NSE surface can inform and which the bounded network output cannot represent.
    """
    keep = {k: [] for k in ("kappa", "theta", "sigma", "feller_ratio", "rho")}
    total = 0
    for _ in range(max_rounds):
        m = int(max(1024, 1.6 * (n - total)))
        kappa = _loguniform(rng, box.kappa_low, box.kappa_high, m)
        theta = _loguniform(rng, box.theta_low, box.theta_high, m)
        eta = _loguniform(rng, box.feller_ratio_low, box.feller_ratio_high, m)
        sigma = box.sigma_from_ratio(eta, kappa, theta)
        rho = rng.uniform(box.rho_low, box.rho_high, m)
        if screen:
            ok = np.ones(m, dtype=bool)
            for tau in (30.0 / 365.0, 90.0 / 365.0):
                for state in (theta, np.full(m, box.variance_low), np.full(m, box.variance_high)):
                    vol, _, _ = implied_vol_batch(
                        np.zeros(m), np.full(m, tau), kappa, theta, sigma, rho, state)
                    ok &= np.isfinite(vol) & (vol >= ATM_VOL_LOW) & (vol <= ATM_VOL_HIGH)
        else:
            ok = np.ones(m, dtype=bool)
        for key, arr in (("kappa", kappa), ("theta", theta), ("sigma", sigma),
                         ("feller_ratio", eta), ("rho", rho)):
            keep[key].append(arr[ok])
        total += int(ok.sum())
        if total >= n:
            break
    out = {k: np.concatenate(v)[:n] for k, v in keep.items()}
    if out["kappa"].size < n:
        raise RuntimeError("parameter screen too strict: %d of %d" % (out["kappa"].size, n))
    return out


def _loguniform(rng, low, high, size):
    low = np.log(np.asarray(low, dtype=float))
    high = np.log(np.asarray(high, dtype=float))
    return np.exp(low + (high - low) * rng.random(size))


def sample_collocation(
    n_points: int,
    spot_ceiling: dict[str, float],
    strike_range: dict[str, tuple[float, float]],
    symbol_weight: dict[str, float],
    box: Box = Box(),
    seed: int = 0,
    screen: bool = True,
) -> dict[str, np.ndarray]:
    """Draw the physical collocation set the study specified.

    Every point carries an explicit spot inside ``[0, 1.5 x ten-year maximum
    traded price]`` for its symbol, a strike from that symbol's listed strike
    range, a maturity on the NSE one-, two- and three-month stock-option cycle,
    a variance taken from the inverse-Black-Scholes implied-volatility box, and
    a Heston parameter vector from the calibration box.

    Spot is not drawn uniformly and then discarded: ``x = log(F/K)`` is drawn
    first from a near-the-money-weighted mixture, the strike is drawn from the
    listed range, and the resulting spot is placed inside the physical ceiling by
    rescaling the strike where needed.  Heston is homogeneous in (S, K), so
    rescaling the strike is an exact re-labelling, not an approximation, and it
    guarantees the returned spot column lies in the requested interval.
    """
    rng = np.random.default_rng(seed)
    symbols = sorted(spot_ceiling)
    weights = np.array([symbol_weight.get(s, 1.0) for s in symbols], dtype=float)
    weights = weights / weights.sum()
    idx = rng.choice(len(symbols), size=n_points, p=weights)
    ceiling = np.array([spot_ceiling[symbols[i]] for i in idx], dtype=float)
    k_lo = np.array([strike_range[symbols[i]][0] for i in idx], dtype=float)
    k_hi = np.array([strike_range[symbols[i]][1] for i in idx], dtype=float)

    # Maturity: stratify one third of the points onto the listed 1M/2M/3M
    # slices, and fill the rest continuously across the whole cycle.
    maturity = _loguniform(rng, box.maturity_low, box.maturity_high, n_points)
    slice_mask = rng.random(n_points) < 0.33
    slice_days = np.array(MONTH_SLICE_DAYS)[rng.integers(0, len(MONTH_SLICE_DAYS), n_points)]
    maturity = np.where(slice_mask, slice_days / 365.0, maturity)

    variance = _loguniform(rng, box.variance_low, box.variance_high, n_points)
    drawn = draw_parameters(rng, n_points, box, screen=screen)
    kappa, theta, sigma, eta, rho = (drawn["kappa"], drawn["theta"], drawn["sigma"],
                                     drawn["feller_ratio"], drawn["rho"])
    rate = rng.uniform(box.rate_low, box.rate_high, n_points)
    dividend = rng.uniform(box.dividend_low, box.dividend_high, n_points)

    # Log forward moneyness: two thirds concentrated where NSE actually quotes,
    # one third spread across the full truncated axis so the physics is enforced
    # out to the boundary the ansatz already satisfies exactly.
    X = box.x_half_width
    # Two thirds in standardised units, where the option price is actually
    # sensitive to total variance; one third spread across the full truncated
    # axis so the physics is enforced out to the boundary as well.
    vbar = expected_integrated_variance_rate(variance, maturity, kappa, theta)
    z = np.clip(rng.normal(0.0, 1.75, n_points), -5.0, 5.0)
    near = z * np.sqrt(vbar * maturity)
    wide = rng.uniform(-X, X, n_points)
    x = np.where(rng.random(n_points) < 0.66, near, wide)
    x = np.clip(x, -X, X)

    strike = _loguniform(rng, np.maximum(k_lo, 1e-6), k_hi, n_points)
    spot = strike * np.exp(x - (rate - dividend) * maturity)
    over = spot > ceiling
    if over.any():                       # exact re-labelling by homogeneity
        shrink = ceiling[over] / spot[over]
        strike[over] *= shrink
        spot[over] *= shrink
    return {
        "symbol_index": idx,
        "symbol": np.array([symbols[i] for i in idx]),
        "spot": spot,
        "spot_ceiling": ceiling,
        "strike": strike,
        "x": x,
        "maturity": maturity,
        "maturity_days": maturity * 365.0,
        "variance": variance,
        "vbar": vbar,
        "kappa": kappa,
        "theta": theta,
        "sigma": sigma,
        "feller_ratio": eta,
        "rho": rho,
        "rate": rate,
        "dividend": dividend,
    }


def sample_anchor(n_points: int, box: Box = Box(), seed: int = 1) -> dict[str, np.ndarray]:
    """Points where the exact Lewis price is evaluated to supervise the smile.

    Log forward moneyness is drawn in *standardised* units, ``x = z sqrt(T vbar)``
    with ``|z| <= 5``.  Beyond about five standard deviations the exact price is
    within float64 noise of its own no-arbitrage bound, so implied variance is
    not invertible there and an anchor would carry no information.  The PDE
    collocation set, which needs no inversion, still spans the full physical
    spot range.
    """
    rng = np.random.default_rng(seed)
    maturity = _loguniform(rng, box.maturity_low, box.maturity_high, n_points)
    slice_mask = rng.random(n_points) < 0.33
    slice_days = np.array(MONTH_SLICE_DAYS)[rng.integers(0, len(MONTH_SLICE_DAYS), n_points)]
    maturity = np.where(slice_mask, slice_days / 365.0, maturity)
    variance = _loguniform(rng, box.variance_low, box.variance_high, n_points)
    drawn = draw_parameters(rng, n_points, box)
    vbar = expected_integrated_variance_rate(variance, maturity, drawn["kappa"], drawn["theta"])
    z = np.where(rng.random(n_points) < 0.7,
                 np.clip(rng.normal(0.0, 1.5, n_points), -5.0, 5.0),
                 rng.uniform(-5.0, 5.0, n_points))
    x = np.clip(z * np.sqrt(vbar * maturity), -box.x_half_width, box.x_half_width)
    return {"x": x, "z": z, "maturity": maturity, "variance": variance, "vbar": vbar, **drawn}


def anchor_targets(points: dict[str, np.ndarray], g_limit: float = 1.8) -> dict[str, np.ndarray]:
    """Exact price, exact implied total variance, and the network's ``g`` target."""
    price = normalised_forward_call_batch(
        points["x"], points["maturity"], points["kappa"], points["theta"],
        points["sigma"], points["rho"], points["variance"])
    w = implied_total_variance(price, points["x"])
    vbar = expected_integrated_variance_rate(
        points["variance"], points["maturity"], points["kappa"], points["theta"])
    backbone = points["maturity"] * vbar
    with np.errstate(invalid="ignore", divide="ignore"):
        g = 0.5 * np.log(w / backbone)
    usable = np.isfinite(g) & (np.abs(g) <= g_limit)
    return {"price": price, "total_variance": w, "vbar": vbar, "g_target": g, "usable": usable}


# --------------------------------------------------------------------------
# Physics-informed network (MLX, Metal GPU)
# --------------------------------------------------------------------------

import mlx.core as mx           # noqa: E402
import mlx.nn as nn             # noqa: E402
import mlx.optimizers as optim  # noqa: E402

G_LIMIT = 1.8          # |g| bound: implied vol stays within e^{+/-1.8} of sqrt(vbar)
Z_BOUND = 8.0          # bound on the standardised-moneyness feature
FEATURES = 13


_MRR_SWITCH = 0.05


def _mean_reversion_ratio(u):
    """``(1 - e^{-u}) / u``, switched to its Taylor series near zero.

    The closed form cancels catastrophically for small ``u``: numerator and
    denominator are both ``O(u)`` and their difference is ``O(u^2)``, which loses
    most of the float32 mantissa and then gets amplified by the second
    derivatives the PDE residual needs.  The switch is at ``u = 0.05``, where the
    quartic series is already accurate to about 1e-8.
    """
    safe = mx.where(u < _MRR_SWITCH, mx.ones_like(u), u)
    exact = (1.0 - mx.exp(-safe)) / safe
    series = 1.0 - u / 2.0 + u * u / 6.0 - u * u * u / 24.0 + u ** 4 / 120.0
    return mx.where(u < _MRR_SWITCH, series, exact)


def integrated_variance_rate_mx(v, tau, kappa, theta):
    return theta + (v - theta) * _mean_reversion_ratio(kappa * tau)


class ImpliedVarianceNet(nn.Module):
    """``g``: the log-ratio of implied volatility to its zero-vol-of-vol value.

    ``log w = log T + log vbar(T; v, kappa, theta) + 2 g``.  At ``g = 0`` the
    network reproduces Heston exactly when vol-of-vol is zero, and the PDE
    residual is then identically zero, so training starts on the solution
    manifold rather than at noise.
    """

    def __init__(self, width: int = 160, depth: int = 5, features: int = FEATURES):
        super().__init__()
        sizes = [features] + [width] * depth
        self.hidden = [nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:])]
        self.head = nn.Linear(width, 1)
        # Start at g = 0 exactly.  That is the zero-vol-of-vol Heston solution, at
        # which the PDE residual vanishes identically, so training begins on the
        # solution manifold instead of at random noise.
        self.head.weight = mx.zeros_like(self.head.weight)
        self.head.bias = mx.zeros_like(self.head.bias)

    def __call__(self, feats):
        h = feats
        for layer in self.hidden:
            h = mx.tanh(layer(h))
        return G_LIMIT * mx.tanh(self.head(h)[..., 0])


@dataclass(frozen=True)
class Normaliser:
    """log-scale feature normalisation constants, taken from the training box."""

    x_half_width: float
    log_tau: tuple[float, float]
    log_v: tuple[float, float]
    log_kappa: tuple[float, float]
    log_theta: tuple[float, float]
    log_eta: tuple[float, float]

    @staticmethod
    def from_box(box: Box) -> "Normaliser":
        L = math.log
        return Normaliser(
            x_half_width=box.x_half_width,
            log_tau=(L(box.maturity_low), L(box.maturity_high)),
            log_v=(L(box.variance_low), L(box.variance_high)),
            log_kappa=(L(box.kappa_low), L(box.kappa_high)),
            log_theta=(L(box.theta_low), L(box.theta_high)),
            log_eta=(L(box.feller_ratio_low), L(box.feller_ratio_high)),
        )


def _unit(value, bounds):
    lo, hi = bounds
    return 2.0 * (value - lo) / (hi - lo) - 1.0


def build_features(x, v, tau, kappa, theta, sigma, rho, norm: Normaliser):
    """13 inputs: raw state, standardised moneyness, and dimensionless groups."""
    vbar = integrated_variance_rate_mx(v, tau, kappa, theta)
    # Standardised moneyness, bounded by construction.  Writing it as
    # x / sqrt(T vbar + x^2 / Z^2) keeps |z| < Z without a clip: a hard clip or a
    # plain x / sqrt(T vbar) sends tanh' to exactly zero in float32 while the
    # chain-rule factor diverges, and 0 * inf is how the second derivatives turn
    # into NaN at one-day maturity.
    z = x / mx.sqrt(tau * vbar + x * x / (Z_BOUND * Z_BOUND) + 1e-12)
    eta = sigma / mx.sqrt(2.0 * kappa * theta)
    nu = sigma * mx.sqrt(tau) / mx.sqrt(vbar + 1e-9)
    curvature = mx.tanh(nu)
    return mx.stack(
        [
            x / norm.x_half_width,
            z / Z_BOUND,
            mx.tanh(z / 1.5),
            _unit(mx.log(tau), norm.log_tau),
            _unit(mx.log(v), norm.log_v),
            _unit(mx.log(kappa), norm.log_kappa),
            _unit(mx.log(theta), norm.log_theta),
            rho,
            mx.tanh(kappa * tau),
            curvature,
            rho * curvature,
            mx.tanh(0.5 * mx.log(v / theta)),
            _unit(mx.log(eta), norm.log_eta),
        ],
        axis=-1,
    )


def log_total_variance(model, x, v, tau, kappa, theta, sigma, rho, norm):
    feats = build_features(x, v, tau, kappa, theta, sigma, rho, norm)
    vbar = integrated_variance_rate_mx(v, tau, kappa, theta)
    return mx.log(tau) + mx.log(vbar) + 2.0 * model(feats)


def state_derivatives(model, x, v, tau, kappa, theta, sigma, rho, norm):
    """``l`` and every derivative the Heston residual needs, by nested autodiff."""

    def scalar(a, b, c):
        return mx.sum(log_total_variance(model, a, b, c, kappa, theta, sigma, rho, norm))

    grad = mx.grad(scalar, argnums=(0, 1, 2))
    l_x, l_v, l_tau = grad(x, v, tau)
    l_xx, l_xv = mx.grad(lambda a, b, c: mx.sum(grad(a, b, c)[0]), argnums=(0, 1))(x, v, tau)
    l_vv = mx.grad(lambda a, b, c: mx.sum(grad(a, b, c)[1]), argnums=1)(x, v, tau)
    ell = log_total_variance(model, x, v, tau, kappa, theta, sigma, rho, norm)
    return ell, l_x, l_v, l_tau, l_xx, l_xv, l_vv


def heston_residual(model, x, v, tau, kappa, theta, sigma, rho, norm):
    """Heston PDE residual in implied-total-variance form.

    ``c = Black76(x, w)`` is substituted into the Heston equation and the result
    divided by the strictly positive Black vega, which removes the payoff
    singularity; ``l = log w`` then removes every negative power of ``w``.  The
    residual is returned normalised by ``v + vbar`` so it is dimensionless.
    """
    ell, l_x, l_v, l_tau, l_xx, l_xv, l_vv = state_derivatives(
        model, x, v, tau, kappa, theta, sigma, rho, norm)
    w = mx.exp(ell)
    a_coef = 0.5 * x * x - 0.125 * w * w - 0.5 * w

    diffusion_x = (
        2.0
        + 2.0 * (0.5 * w - x) * l_x
        + a_coef * l_x * l_x
        + w * (l_xx + l_x * l_x)
        - w * l_x
    )
    cross = (0.5 * w - x) * l_v + a_coef * l_x * l_v + w * (l_xv + l_x * l_v)
    diffusion_v = a_coef * l_v * l_v + w * (l_vv + l_v * l_v)

    residual = (
        w * l_tau
        - 0.5 * v * diffusion_x
        - rho * sigma * v * cross
        - 0.5 * sigma * sigma * v * diffusion_v
        - kappa * (theta - v) * w * l_v
    )
    vbar = integrated_variance_rate_mx(v, tau, kappa, theta)
    scale = v + vbar
    return residual / scale, dict(w=w, l_x=l_x, l_tau=l_tau, l_xx=l_xx)


def price_relevance_weight(x, diag, floor: float = 0.01):
    """How much the total variance at a point actually moves its option price.

    In total-variance units the Black vega is ``n(d2) / (2 sqrt(w))``, so the
    price sensitivity to ``w`` decays like ``exp(-d2^2 / 2)`` away from the money.
    Deep in the wings the implied variance is nearly undetermined by prices and
    the implied-variance PDE becomes stiff -- the ``A l_v^2`` term alone reaches
    1e5 there at one-day maturity -- while the option itself is worth its
    intrinsic value to eight decimals.  Weighting the residual this way enforces
    the physics where it changes prices, which is the same vega weighting the
    market calibration already uses.  Returned detached: it is a weight, not part
    of the objective.
    """
    w = diag["w"]
    root = mx.sqrt(mx.maximum(w, 1e-12))
    d2 = x / root - 0.5 * root
    return mx.stop_gradient(mx.maximum(mx.exp(-0.5 * d2 * d2), floor))


def calendar_penalty(diag):
    """Total variance must not fall as maturity grows: ``w_T = w l_T >= 0``."""
    return mx.maximum(-diag["l_tau"], 0.0) ** 2


def butterfly_penalty(x, diag):
    """Durrleman's condition in total-variance form; negative means butterfly arbitrage."""
    w, l_x, l_xx = diag["w"], diag["l_x"], diag["l_xx"]
    term = (1.0 - 0.5 * x * l_x) ** 2
    term = term - 0.25 * w * l_x * l_x - w * w * l_x * l_x / 16.0
    term = term + 0.5 * w * (l_xx + l_x * l_x)
    return mx.maximum(-term, 0.0) ** 2


# --------------------------------------------------------------------------
# Trained-network pricing and inverse calibration
# --------------------------------------------------------------------------

def _norm_cdf_mx(z):
    return 0.5 * (1.0 + mx.erf(z / math.sqrt(2.0)))


def black76_normalised_mx(x, w):
    root = mx.sqrt(mx.maximum(w, 1e-12))
    d1 = x / root + 0.5 * root
    return mx.exp(x) * _norm_cdf_mx(d1) - _norm_cdf_mx(d1 - root)


@dataclass(frozen=True)
class ParameterTransform:
    """Bounded, unconstrained parameterisation used by the calibrator.

    Mirrors ``single_heston.encode_params`` in spirit -- logistic on the
    positive parameters, ``tanh`` on the correlation -- but bounds vol-of-vol
    through the Feller ratio so the search cannot leave the box the network was
    trained on.
    """

    box: Box

    def decode(self, z):
        b = self.box
        sig = lambda t: 1.0 / (1.0 + np.exp(-np.clip(t, -60, 60)))
        kappa = b.kappa_low * (b.kappa_high / b.kappa_low) ** sig(z[0])
        theta = b.theta_low * (b.theta_high / b.theta_low) ** sig(z[1])
        eta = b.feller_ratio_low * (b.feller_ratio_high / b.feller_ratio_low) ** sig(z[2])
        sigma = np.clip(eta * np.sqrt(2.0 * kappa * theta), b.sigma_floor, b.sigma_ceiling)
        rho = b.rho_low + (b.rho_high - b.rho_low) * sig(z[3])
        v0 = b.variance_low * (b.variance_high / b.variance_low) ** sig(z[4])
        return np.array([kappa, theta, sigma, rho, v0])

    def encode(self, params):
        b = self.box
        kappa, theta, sigma, rho, v0 = (float(p) for p in params)
        logit = lambda u: float(np.log(np.clip(u, 1e-6, 1 - 1e-6) / (1 - np.clip(u, 1e-6, 1 - 1e-6))))
        frac = lambda value, lo, hi: math.log(value / lo) / math.log(hi / lo)
        eta = sigma / math.sqrt(2.0 * kappa * theta)
        return np.array([
            logit(frac(np.clip(kappa, b.kappa_low, b.kappa_high), b.kappa_low, b.kappa_high)),
            logit(frac(np.clip(theta, b.theta_low, b.theta_high), b.theta_low, b.theta_high)),
            logit(frac(np.clip(eta, b.feller_ratio_low, b.feller_ratio_high),
                       b.feller_ratio_low, b.feller_ratio_high)),
            logit((np.clip(rho, b.rho_low, b.rho_high) - b.rho_low) / (b.rho_high - b.rho_low)),
            logit(frac(np.clip(v0, b.variance_low, b.variance_high), b.variance_low, b.variance_high)),
        ])


class PinnPricer:
    """Differentiable option pricer backed by the trained implied-variance net."""

    def __init__(self, model: ImpliedVarianceNet, box: Box):
        self.model = model
        self.box = box
        self.norm = Normaliser.from_box(box)

    def normalised_call(self, x, maturity, kappa, theta, sigma, rho, v0):
        ell = log_total_variance(self.model, x, v0, maturity, kappa, theta, sigma, rho, self.norm)
        return black76_normalised_mx(x, mx.exp(ell))

    def price(self, quotes, params) -> np.ndarray:
        """Prices the repository's quote frame; matches ``single_heston.heston_prices``."""
        spot = quotes.spot.to_numpy(float)
        strike = quotes.strike.to_numpy(float)
        maturity = quotes.maturity.to_numpy(float)
        rate = quotes.rate.to_numpy(float)
        dividend = quotes.dividend.to_numpy(float)
        is_call = quotes.option_type.eq("CE").to_numpy()
        return self.price_arrays(spot, strike, maturity, rate, dividend, is_call, params)

    def price_arrays(self, spot, strike, maturity, rate, dividend, is_call, params) -> np.ndarray:
        kappa, theta, sigma, rho, v0 = (float(p) for p in params)
        x = np.log(spot / strike) + (rate - dividend) * maturity
        x = np.clip(x, -self.box.x_half_width, self.box.x_half_width)
        n = x.size
        const = lambda value: mx.full((n,), float(value), dtype=mx.float32)
        c = self.normalised_call(
            mx.array(x.astype(np.float32)), mx.array(maturity.astype(np.float32)),
            const(kappa), const(theta), const(sigma), const(rho), const(v0))
        call = strike * np.exp(-rate * maturity) * np.asarray(c, dtype=np.float64)
        put = call - spot * np.exp(-dividend * maturity) + strike * np.exp(-rate * maturity)
        return np.where(is_call, call, put)


def audit_reference_pricer(n: int = 2000, seed: int = 5, box: Box | None = None) -> dict:
    """Cross-check the Lewis reference against the repository's own engine."""
    box = box or Box()
    points = sample_anchor(n, box, seed=seed)
    lewis = normalised_forward_call_batch(
        points["x"], points["maturity"], points["kappa"], points["theta"],
        points["sigma"], points["rho"], points["variance"])
    repo = np.array([
        SH.heston_call_prices(
            float(np.exp(points["x"][i])), np.array([1.0]), float(points["maturity"][i]), 0.0, 0.0,
            (float(points["kappa"][i]), float(points["theta"][i]), float(points["sigma"][i]),
             float(points["rho"][i]), float(points["variance"][i])))[0]
        for i in range(n)
    ])
    gap = np.abs(lewis - repo)
    short = points["maturity"] * 365.0 < 7.0
    return {
        "points": int(n),
        "max_gap": float(np.nanmax(gap)),
        "p99_gap": float(np.nanpercentile(gap, 99)),
        "median_gap": float(np.nanmedian(gap)),
        "max_gap_at_or_beyond_7_days": float(np.nanmax(gap[~short])) if (~short).any() else float("nan"),
        "max_gap_under_7_days": float(np.nanmax(gap[short])) if short.any() else float("nan"),
        "note": ("The two engines agree to float64 noise over the tenor range the study "
                 "calibrates on. They separate only below about three days to expiry with "
                 "variance at the floor of the box, where the Gauss-Laguerre P1/P2 integrand "
                 "decays too slowly for 64 nodes. The repository's own pipeline requires at "
                 "least seven days to expiry, so it never prices in that corner."),
    }
