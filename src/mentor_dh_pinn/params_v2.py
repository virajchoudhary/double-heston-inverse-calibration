"""Unconstrained parameter coordinates that are a BIJECTION onto the engine's model class.

The audited transform (`calibrate.decode`) had a defect: `PARAM_BOX` imposed finite ranges
that the pricing engine does not require, and it clipped silently. Worse, because
`sigma = eta sqrt(2 kappa theta)`, clipping `theta` or `kappa` also moved `sigma` even when
`sigma` was inside its own range. A vector the exact engine prices happily came back altered
in five of ten coordinates with no warning.

This module fixes that. Every bound below is a bound the ENGINE itself enforces
(`src/constraints.py`), and there are no others:

    kappa_slow < kappa_fast                     ordering, resolves the exact factor
                                                permutation symmetry (measured 1.07e-14)
    2 kappa_i theta_i - sigma_i^2 > 0           Feller, per factor -- the engine REFUSES
                                                to price a violating vector
    rho_slow^2 + rho_fast^2 < 1                 joint correlation disk

Coordinates are economically meaningful, per the redesign brief: total and split for both
the instantaneous and long-run variance, mean-reversion speeds ordered and unbounded above,
Feller ratios, and correlations on the open disk.

    z0  exp      -> kappa_slow                  (0, inf)     unbounded, log-scaled
    z1  exp      -> kappa_fast/kappa_slow - 1    (0, inf)     unbounded, log-scaled,
                                                              enforces ordering
    z2  exp      -> theta_total                 (0, inf)     unbounded
    z3  sigmoid  -> theta_slow share            (0, 1)
    z4  exp      -> v0_total                    (0, inf)     unbounded
    z5  sigmoid  -> v0_slow share               (0, 1)
    z6  sigmoid  -> eta_slow = sigma/sqrt(2 k th)   (0, 1)   exactly the Feller-valid range
    z7  sigmoid  -> eta_fast                    (0, 1)
    z8, z9  radial tanh -> (rho_slow, rho_fast) open unit disk, surjective

Surjectivity matters and is tested: any parameter vector the engine accepts has a preimage,
so the model can express it. Nothing legal is unreachable and nothing illegal is reachable.
There is therefore no silent clipping to hide -- out-of-distribution reporting is done
separately, by locating the answer within the TRAINING PRIOR, never by bounding the output.
"""

from __future__ import annotations

import math

import numpy as np
import torch

CANONICAL = ("kappa_slow", "theta_slow", "sigma_slow", "rho_slow", "v0_slow",
             "kappa_fast", "theta_fast", "sigma_fast", "rho_fast", "v0_fast")
_TINY = 1.0e-12


def _softplus(x, lib):
    if lib is torch:
        return torch.nn.functional.softplus(x)
    return np.logaddexp(0.0, x)


_SHARE_EPS = 1.0e-12


def _sig(x, lib):
    return torch.sigmoid(x) if lib is torch else 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def _share(x, lib):
    """A split in (0,1) held strictly off both ends.

    Plain `sigmoid` saturates to exactly 1.0 in float64 for x > ~37, which would make one
    factor's theta or v0 exactly zero and violate positivity and Feller downstream.
    """
    return _SHARE_EPS + (1.0 - 2.0 * _SHARE_EPS) * _sig(x, lib)


def decode(z):
    """R^10 -> the ten canonical parameters. Works for numpy arrays or torch tensors.

    Returns a list in canonical order so it can be stacked either way.
    """
    lib = torch if torch.is_tensor(z) else np
    sqrt = lib.sqrt
    # Mean-reversion speeds in LOG coordinates, and the second factor as a MULTIPLICATIVE
    # gap. A softplus gap is nearly linear in kappa, which gave that coordinate a spread of
    # 13.0 against about 1.2 for every other one -- it then dominated the parameter loss and
    # the model barely beat a constant predictor. Timescales are the natural log-scaled
    # quantity here.
    ks = lib.exp(z[..., 0])
    # the +_TINY keeps the ordering strict when exp(z1) underflows to 0
    kf = ks * (1.0 + lib.exp(z[..., 1]) + _TINY)
    th_tot = lib.exp(z[..., 2])
    a_th = _share(z[..., 3], lib)
    ts, tf = th_tot * a_th, th_tot * (1.0 - a_th)
    v_tot = lib.exp(z[..., 4])
    a_v = _share(z[..., 5], lib)
    vs, vf = v_tot * a_v, v_tot * (1.0 - a_v)
    ss = _share(z[..., 6], lib) * sqrt(2.0 * ks * ts)       # eta in (0,1) => strict Feller
    sf = _share(z[..., 7], lib) * sqrt(2.0 * kf * tf)
    # radial map of R^2 onto the OPEN unit disk: direction from z, radius through tanh.
    # tanh must NOT also be applied per-component -- that caps the reachable radius at
    # tanh(sqrt(2)) = 0.888 and makes legal correlation pairs unreachable.
    a, b = z[..., 8], z[..., 9]
    n = sqrt(a * a + b * b) + _TINY
    # tanh(n) rounds to exactly 1.0 in float64 for n > ~19, which would put the pair ON the
    # disk boundary; the engine requires strictly inside, so hold it off by one ulp-ish.
    r = lib.tanh(n) * (1.0 - 1.0e-12)
    rs, rf = a / n * r, b / n * r
    return [ks, ts, ss, rs, vs, kf, tf, sf, rf, vf]


def encode(params) -> np.ndarray:
    """Canonical ten-vector -> R^10. Exact inverse of `decode` on the engine's valid set."""
    p = {k: float(v) for k, v in zip(CANONICAL, params)}
    # Fail loudly rather than clamp. A vector outside the engine's model class has no
    # preimage here, and silently returning the nearest representable one is exactly the
    # defect this module exists to remove.
    bad = []
    if not p["kappa_slow"] < p["kappa_fast"]:
        bad.append("kappa_slow < kappa_fast")
    for tag in ("slow", "fast"):
        if 2 * p[f"kappa_{tag}"] * p[f"theta_{tag}"] - p[f"sigma_{tag}"] ** 2 <= 0:
            bad.append(f"{tag}-factor Feller gap")
    if p["rho_slow"] ** 2 + p["rho_fast"] ** 2 >= 1.0:
        bad.append("joint correlation disk")
    if min(p["theta_slow"], p["theta_fast"], p["v0_slow"], p["v0_fast"],
           p["sigma_slow"], p["sigma_fast"], p["kappa_slow"]) <= 0:
        bad.append("positivity")
    if bad:
        raise ValueError("vector is outside the Double Heston model class, so it has no "
                         f"latent preimage; violated: {bad}")
    def logit(u):                       # inverse of _share
        u = (min(max(u, 1e-15), 1 - 1e-15) - _SHARE_EPS) / (1.0 - 2.0 * _SHARE_EPS)
        u = min(max(u, 1e-15), 1 - 1e-15)
        return math.log(u / (1 - u))
    z = np.zeros(10)
    z[0] = math.log(p["kappa_slow"])
    z[1] = math.log(max(p["kappa_fast"] / p["kappa_slow"] - 1.0, 1e-12))
    th_tot = p["theta_slow"] + p["theta_fast"]
    z[2] = math.log(th_tot)
    z[3] = logit(p["theta_slow"] / th_tot)
    v_tot = p["v0_slow"] + p["v0_fast"]
    z[4] = math.log(v_tot)
    z[5] = logit(p["v0_slow"] / v_tot)
    z[6] = logit(p["sigma_slow"] / math.sqrt(2 * p["kappa_slow"] * p["theta_slow"]))
    z[7] = logit(p["sigma_fast"] / math.sqrt(2 * p["kappa_fast"] * p["theta_fast"]))
    rs, rf = p["rho_slow"], p["rho_fast"]
    r = math.hypot(rs, rf)
    if r < 1e-15:
        z[8] = z[9] = 0.0
    else:
        n = math.atanh(min(r, 1 - 1e-15))                     # radius preimage
        z[8], z[9] = rs / r * n, rf / r * n
    return z


def to_array(decoded) -> np.ndarray:
    return np.array([float(v) for v in decoded], dtype=float)


def encode_batch(params: np.ndarray) -> np.ndarray:
    """Vectorised `encode` for a (n, 10) array. Same maths, no Python loop."""
    p = np.asarray(params, dtype=float)
    ks, ts, ss, rs, vs, kf, tf, sf, rf, vf = (p[:, i] for i in range(10))
    z = np.zeros_like(p)
    def logit(u):
        u = (np.clip(u, 1e-15, 1 - 1e-15) - _SHARE_EPS) / (1.0 - 2.0 * _SHARE_EPS)
        u = np.clip(u, 1e-15, 1 - 1e-15)
        return np.log(u / (1 - u))
    z[:, 0] = np.log(ks)
    z[:, 1] = np.log(np.maximum(kf / ks - 1.0, 1e-12))
    th = ts + tf; z[:, 2] = np.log(th); z[:, 3] = logit(ts / th)
    v = vs + vf;  z[:, 4] = np.log(v);  z[:, 5] = logit(vs / v)
    z[:, 6] = logit(ss / np.sqrt(2 * ks * ts))
    z[:, 7] = logit(sf / np.sqrt(2 * kf * tf))
    r = np.hypot(rs, rf)
    n = np.arctanh(np.clip(r, 0.0, 1 - 1e-15))
    safe = np.maximum(r, 1e-15)
    z[:, 8] = np.where(r < 1e-15, 0.0, rs / safe * n)
    z[:, 9] = np.where(r < 1e-15, 0.0, rf / safe * n)
    return z
