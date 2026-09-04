"""Classical baselines, rebuilt: Black-Scholes, single Heston, cold-start Double Heston.

These were lost with the session scratchpad, so the earlier repricing comparisons could not
be re-derived from source. This module restores them. All three fit the same quote set and
are scored the same way, and all use the exact Fourier engine -- never a surrogate.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.optimize import least_squares, minimize_scalar
from scipy.stats import norm

from .params_v2 import decode, encode
from .torch_pricer import price_call, price_call_single

SIG_LO, SIG_HI = 0.02, 3.0


# ------------------------------------------------------------------ Black-Scholes
def black76(F, K, tau, sigma):
    F = np.asarray(F, float); K = np.asarray(K, float); tau = np.asarray(tau, float)
    if sigma <= 0:
        return np.maximum(F - K, 0.0)
    r = sigma * np.sqrt(np.maximum(tau, 1e-12))
    d1 = np.log(np.maximum(F, 1e-300) / np.maximum(K, 1e-300)) / r + 0.5 * r
    return F * norm.cdf(d1) - K * norm.cdf(d1 - r)


def bs_surface(sigma, geo):
    fwd = geo["spot"] * np.exp((geo["rate"] - geo["carry"]) * geo["tau"])
    return np.exp(-geo["rate"] * geo["tau"]) * black76(fwd, geo["strike"], geo["tau"], sigma)


def fit_black_scholes(geo, observed):
    """One constant volatility. Grid scan then bounded Brent -- `least_squares` terminates
    prematurely on scalar problems of this shape, which cost a whole study once already."""
    obj = lambda s: float(np.mean((bs_surface(s, geo) - observed) ** 2))
    grid = np.exp(np.linspace(math.log(SIG_LO), math.log(SIG_HI), 140))
    vals = [obj(s) for s in grid]; j = int(np.argmin(vals))
    lo, hi = grid[max(j - 1, 0)], grid[min(j + 1, len(grid) - 1)]
    r = minimize_scalar(obj, bounds=(lo, hi), method="bounded", options={"xatol": 1e-11})
    return float(r.x) if r.fun <= vals[j] else float(grid[j])


# ------------------------------------------------------------------ single Heston
def _sh_decode(z):
    """5 free coordinates -> (kappa, theta, sigma, rho, v0), Feller and |rho|<1 enforced."""
    k = math.log1p(math.exp(min(z[0], 40.0))) + 1e-9
    th = math.exp(np.clip(z[1], -20, 5))
    eta = 1.0 / (1.0 + math.exp(-np.clip(z[2], -40, 40)))
    sg = eta * math.sqrt(2.0 * k * th)
    rho = math.tanh(z[3]) * (1 - 1e-12)
    v0 = math.exp(np.clip(z[4], -20, 5))
    return np.array([k, th, sg, rho, v0])


def sh_surface(p5, geo):
    with torch.no_grad():
        return price_call_single(torch.tensor(np.asarray(p5, float)),
                                 torch.tensor(geo["spot"]), torch.tensor(geo["strike"]),
                                 torch.tensor(geo["tau"]), torch.tensor(geo["rate"]),
                                 torch.tensor(geo["carry"])).numpy()


SH_STARTS = (
    (1.0, 0.04, 0.5, -0.6, 0.04), (2.0, 0.09, 0.5, -0.3, 0.09),
    (0.5, 0.16, 0.5, -0.8, 0.16), (4.0, 0.02, 0.5, -0.5, 0.06),
    (1.5, 0.25, 0.5, -0.2, 0.30),
)


def _sh_encode(k, th, eta, rho, v0):
    return np.array([math.log(math.expm1(k)), math.log(th),
                     math.log(eta / (1 - eta)), math.atanh(rho), math.log(v0)])


def fit_single_heston(geo, observed, max_nfev=200):
    scale = np.maximum(geo["spot"], 1e-12)
    def resid(z):
        try:
            m = sh_surface(_sh_decode(z), geo)
        except Exception:
            return np.ones_like(observed)
        return np.where(np.isfinite(m), (m - observed) / scale, 1.0)
    best = None
    for s in SH_STARTS:
        z0 = _sh_encode(s[0], s[1], s[2], s[3], s[4])
        try:
            r = least_squares(resid, z0, max_nfev=max_nfev, diff_step=1e-6)
        except Exception:
            continue
        obj = float(np.mean(r.fun ** 2))
        if best is None or obj < best[0]:
            best = (obj, _sh_decode(r.x))
    return {"params": best[1], "objective": math.sqrt(best[0])} if best else None


# ------------------------------------------------------------------ Double Heston
DH_STARTS = (
    (0.8, 0.04, 0.30, -0.6, 0.04, 4.0, 0.03, 0.35, -0.3, 0.03),
    (1.5, 0.09, 0.45, -0.4, 0.09, 6.0, 0.06, 0.55, -0.2, 0.06),
    (0.4, 0.16, 0.35, -0.8, 0.16, 2.5, 0.10, 0.45, -0.4, 0.10),
    (2.2, 0.02, 0.20, -0.5, 0.03, 9.0, 0.02, 0.30, -0.1, 0.02),
    (1.0, 0.25, 0.55, -0.3, 0.30, 5.0, 0.18, 0.70, -0.3, 0.25),
)


def dh_surface(p10, geo):
    with torch.no_grad():
        return price_call(torch.tensor(np.asarray(p10, float)),
                          torch.tensor(geo["spot"]), torch.tensor(geo["strike"]),
                          torch.tensor(geo["tau"]), torch.tensor(geo["rate"]),
                          torch.tensor(geo["carry"])).numpy()


def fit_double_heston(geo, observed, starts=DH_STARTS, max_nfev=220, z_prior=None,
                      prior_weight=0.0):
    """Cold or warm multi-start fit in the bijective coordinates of `params_v2`.

    `z_prior`/`prior_weight` allow the same routine to serve as the scalar-ridge baseline
    for the ablation, so the comparison is not confounded by a different optimiser.
    """
    scale = np.maximum(geo["spot"], 1e-12)
    def resid(z):
        try:
            m = dh_surface([float(v) for v in decode(np.asarray(z, float))], geo)
        except Exception:
            m = None
        data = (np.ones_like(observed) if m is None or not np.all(np.isfinite(m))
                else (m - observed) / scale)
        if prior_weight > 0.0 and z_prior is not None:
            return np.concatenate([data, prior_weight * (np.asarray(z, float) - z_prior)])
        return data
    best = None
    for s in starts:
        try:
            z0 = encode(s)
        except ValueError:
            continue
        try:
            r = least_squares(resid, z0, max_nfev=max_nfev, diff_step=1e-6)
        except Exception:
            continue
        n = len(observed)
        obj = float(np.mean(r.fun[:n] ** 2))
        if best is None or obj < best[0]:
            best = (obj, np.array([float(v) for v in decode(r.x)]), r.x)
    if best is None:
        return None
    return {"params": best[1], "z": best[2], "objective": math.sqrt(best[0])}
