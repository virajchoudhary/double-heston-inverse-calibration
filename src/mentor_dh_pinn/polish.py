"""Stage two: a prior-regularised local polish of a network's ten-vector.

Why this exists.  The dual networks are trained on noisy surfaces against clean
targets, so what they learn is the *conditional mean* of the parameters given a
noisy surface.  That estimator is shrunk toward the prior, which is exactly why
it beats classical least squares on parameter recovery under noise -- and
exactly why it reprices worse: a shrunk parameter vector does not sit at the
minimum of the pricing residual for the one surface in front of it.

Classical calibration is the opposite estimator: unbiased-ish, unshrunk, and it
absorbs the noise.  Neither endpoint is the right answer.  This routine
interpolates between them by solving

    argmin_z  || (C(decode(z)) - C_obs) / S ||^2  +  lam * eps^2 * || z - z_net ||^2

in the constraint-free coordinates of ``calibrate.decode``, warm-started at the
network's own answer ``z_net``.  Because the prior block is expressed in the
same units as the data block -- ``eps`` is the typical per-quote noise magnitude
for *this* surface, which the network is already told -- ``lam`` is
dimensionless and comparable across surfaces:

    lam = 0        pure warm-started local fit (the two-stage design of
                   Horvath, Muguruza & Tomas, 2021: network as initialiser)
    lam -> inf     the network answer, unmoved

The single ridge scalar is the whole knob.  It is chosen on the validation split
and then frozen; the test split is read once.

Every point the optimiser can reach is structurally valid, because it never
leaves ``decode``'s image: positivity, the slow/fast kappa ordering, both Feller
gaps and the joint correlation disk hold for any real vector in R^10.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import least_squares

from src.double_heston import price_double_heston_call

from .calibrate import decode, encode


def exact_surface(params, geo: dict) -> np.ndarray:
    """Price the geometry with the frozen production engine."""
    p = [float(v) for v in params]
    return np.array([price_double_heston_call(
        float(geo["spot"][i]), float(geo["strike"][i]), float(geo["tau"][i]),
        float(geo["rate"][i]), float(geo["carry"][i]), p, node_count=64)
        for i in range(len(geo["tau"]))])


def prior_scale(observed: np.ndarray, spot: np.ndarray, noise_level: float) -> float:
    """Typical per-quote noise magnitude, in the units of the data residual."""
    return max(float(noise_level), 1.0e-4) * float(np.sqrt(np.mean((observed / spot) ** 2)))


def polish(params_net, observed: np.ndarray, geo: dict, *, lam: float,
           noise_level: float, max_nfev: int = 30, diff_step: float = 1.0e-6) -> dict:
    """Warm-started, prior-regularised least squares from the network's answer."""
    z_net = encode(params_net)
    spot = np.asarray(geo["spot"], dtype=float)
    weight = np.sqrt(max(lam, 0.0)) * prior_scale(observed, spot, noise_level)

    def residual(z: np.ndarray) -> np.ndarray:
        zz = np.asarray(z, dtype=float)
        prior = weight * (zz - z_net)
        try:                       # identical degenerate-corner guard to calibrate.py
            priced = exact_surface(decode(zz), geo)
        except (FloatingPointError, ValueError, ZeroDivisionError, OverflowError):
            return np.concatenate([np.ones_like(spot), prior])
        if not np.all(np.isfinite(priced)):
            return np.concatenate([np.ones_like(spot), prior])
        return np.concatenate([(priced - observed) / spot, prior])

    t0 = time.perf_counter()
    result = least_squares(residual, z_net, max_nfev=max_nfev, diff_step=diff_step,
                           xtol=1e-10, ftol=1e-10, gtol=1e-10)
    params = np.array([float(v) for v in decode(result.x)], dtype=float)
    data_block = residual(result.x)[:len(spot)]
    return {"params": params, "z": result.x,
            "data_rmse": float(np.sqrt(np.mean(data_block ** 2))),
            "nfev": int(result.nfev), "njev": int(getattr(result, "njev", 0) or 0),
            "seconds": time.perf_counter() - t0}


def blend(params_a, params_b) -> np.ndarray:
    """Ensemble two networks in the constraint-free coordinates, so the mean stays valid."""
    z = 0.5 * (encode(params_a) + encode(params_b))
    return np.array([float(v) for v in decode(z)], dtype=float)
