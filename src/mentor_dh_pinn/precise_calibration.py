"""Optional exact-engine polish of neural estimates, without access to truth or holdouts.

This is a hybrid calibrator: the network supplies initial parameters and SciPy optimises
the ten constrained latent coordinates. Convergence of prices is not unique recovery.
"""
from __future__ import annotations

import time
import numpy as np
import torch
from scipy.optimize import least_squares

from .params_v2 import decode
from .torch_pricer import price_call


def exact_polish(geo, observed, initial_z, *, quote_sigma=None, node_count=128,
                 max_nfev=500, latent_scale=None, n_starts=3, seed=71):
    """Fit only supplied calibration quotes; return every start and the best fit.

    With no quote_sigma supplied, errors are measured in units of spot. Strict tolerances
    concern numerical optimisation, not a claim of parameter identifiability. No physical
    bounds beyond the repository's decode map are introduced.
    """
    target = np.asarray(observed, dtype=float)
    z0 = np.asarray(initial_z, dtype=float)
    if target.ndim != 1 or len(target) < 3 or z0.shape != (10,):
        raise ValueError("Need at least three quotes and ten initial latent coordinates")
    arrays = {k: np.asarray(geo[k], dtype=float) for k in ("spot", "strike", "tau", "rate", "carry")}
    if any(v.shape != target.shape or not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("Geometry must match quotes and be finite")
    if any((arrays[k] <= 0).any() for k in ("spot", "strike", "tau")):
        raise ValueError("Spot, strike and maturity must be positive")
    if not np.isfinite(target).all() or not np.isfinite(z0).all():
        raise ValueError("Observed prices and initial parameters must be finite")
    scale = arrays["spot"] if quote_sigma is None else np.asarray(quote_sigma, dtype=float)
    if scale.shape != target.shape or not (np.isfinite(scale) & (scale > 0)).all():
        raise ValueError("Quote scales must be finite and positive")
    zs = np.ones(10) if latent_scale is None else np.asarray(latent_scale, dtype=float)
    if zs.shape != (10,) or not (np.isfinite(zs) & (zs > 0)).all() or n_starts < 1:
        raise ValueError("Need ten positive latent scales and at least one start")
    tensors = [torch.tensor(arrays[k], dtype=torch.float64) for k in arrays]
    target_t, scale_t = (torch.tensor(a, dtype=torch.float64) for a in (target, scale))

    def residual_t(w):
        z = torch.as_tensor(z0) + torch.as_tensor(zs) * w
        return (price_call(torch.stack(decode(z)), *tensors, node_count=node_count) - target_t) / scale_t

    cached = {}
    rejected_evaluations = 0

    def fun(w):
        nonlocal rejected_evaluations
        if "w" not in cached or not np.array_equal(w, cached["w"]):
            t = torch.tensor(w, dtype=torch.float64)
            with torch.no_grad():
                r = residual_t(t).numpy()
            cached.clear(); cached.update(w=w.copy(), r=r)
        r = cached["r"]
        if not np.isfinite(r).all():
            rejected_evaluations += 1
            # A fixed, enormous residual rejects the complete candidate. It is counted;
            # it can never masquerade as a zero-error fit or a smaller quote subset.
            return np.full_like(target, 1e50)
        return r

    def jac(w):
        J = torch.func.jacfwd(residual_t)(torch.tensor(w, dtype=torch.float64)).detach().numpy()
        if not np.isfinite(J).all():
            raise FloatingPointError("Nonfinite exact-pricer Jacobian")
        return J

    rng = np.random.default_rng(seed)
    starts = [np.zeros(10)] + [rng.normal(0, .25, 10) for _ in range(n_starts-1)]
    records, best = [], None
    start_time = time.perf_counter()
    for i, w in enumerate(starts):
        before = time.perf_counter()
        try:
            result = least_squares(fun, w, jac=jac, method="trf", x_scale="jac",
                                   max_nfev=max_nfev, ftol=1e-13, xtol=1e-13, gtol=1e-13)
            residual = fun(result.x)
            finite = bool(np.isfinite(residual).all() and np.max(np.abs(residual)) < 1e49)
            objective = float(np.mean(residual**2)) if finite else float("inf")
            fitted_z = z0 + zs*result.x
            rec = {"start": i, "objective": objective, "nfev": result.nfev,
                   "success": bool(result.success and finite), "status": int(result.status),
                   "message": str(result.message), "z": fitted_z.tolist(),
                   "seconds": time.perf_counter()-before}
            if finite and (best is None or objective < best["objective"]):
                best = rec
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
            rec = {"start": i, "success": False, "message": str(error),
                   "objective": float("inf"), "seconds": time.perf_counter()-before}
        records.append(rec)
    if best is None:
        return {"success": False, "starts": records, "seconds": time.perf_counter()-start_time,
                "rejected_evaluations": rejected_evaluations}
    z = np.asarray(best["z"])
    params = np.asarray(decode(z))
    return {"success": best["success"], "params": params, "z": z,
            "objective": best["objective"], "starts": records,
            "seconds": time.perf_counter()-start_time, "selected_start": best["start"],
            "rejected_evaluations": rejected_evaluations}
