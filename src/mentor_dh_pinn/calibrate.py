"""Bounded multi-start inverse calibration of the ten Double Heston parameters.

Two engines behind one interface:

* ``FourierCalibrator`` -- the frozen production pricer, exact, slow;
* ``PinnCalibrator``    -- the V4 conditioned network, differentiable, so the
  Jacobian of the residual with respect to the ten parameters is obtained by
  autograd instead of by finite differences.

Every candidate the optimiser can reach is structurally valid.  The transform
below builds validity in rather than penalising violations: positivity, the
slow/fast ordering, both Feller gaps and the joint correlation disk all hold for
any real vector in R^10, so no start, step or converged point can leave the
canonical set.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import least_squares

from src.double_heston import price_double_heston_call
from .model_v4 import PARAM_BOX, PARAM_NAMES

CANONICAL = ("kappa_slow", "theta_slow", "sigma_slow", "rho_slow", "v0_slow",
             "kappa_fast", "theta_fast", "sigma_fast", "rho_fast", "v0_fast")
V0_BOX = (1.0e-3, 1.2)
ETA_MAX = 0.995          # sigma = eta * sqrt(2 kappa theta); eta < 1 keeps Feller strict
RHO_RADIUS = 0.97        # joint correlation disk


def _sig(t):
    return 1.0 / (1.0 + torch.exp(-torch.clamp(t, -40, 40))) if torch.is_tensor(t) \
        else 1.0 / (1.0 + np.exp(-np.clip(t, -40, 40)))


def decode(z):
    """R^10 -> a structurally valid canonical ten-vector.  Works for numpy or torch."""
    lib = torch if torch.is_tensor(z) else np
    log = lib.log; exp = lib.exp; sqrt = lib.sqrt
    def logbox(t, lo, hi):
        return lo * exp(_sig(t) * math.log(hi / lo))
    ks = logbox(z[0], *PARAM_BOX["kappa_slow"])
    kf_hi = PARAM_BOX["kappa_fast"][1]
    kf = ks + (kf_hi - ks) * _sig(z[1])                      # ordering by construction
    ts = logbox(z[2], *PARAM_BOX["theta_slow"])
    tf = logbox(z[3], *PARAM_BOX["theta_fast"])
    ss = ETA_MAX * _sig(z[4]) * sqrt(2.0 * ks * ts)          # Feller by construction
    sf = ETA_MAX * _sig(z[5]) * sqrt(2.0 * kf * tf)
    rs_box, rf_box = PARAM_BOX["rho_slow"], PARAM_BOX["rho_fast"]
    rs = 0.5 * (rs_box[0] + rs_box[1]) + 0.5 * (rs_box[1] - rs_box[0]) * lib.tanh(z[6])
    rf = 0.5 * (rf_box[0] + rf_box[1]) + 0.5 * (rf_box[1] - rf_box[0]) * lib.tanh(z[7])
    radius = sqrt(rs * rs + rf * rf) + 1e-12
    scale = lib.minimum(lib.ones_like(radius) if torch.is_tensor(radius) else np.ones_like(radius),
                        RHO_RADIUS / radius)                  # joint disk by construction
    rs, rf = rs * scale, rf * scale
    vs = logbox(z[8], *V0_BOX)
    vf = logbox(z[9], *V0_BOX)
    return [ks, ts, ss, rs, vs, kf, tf, sf, rf, vf]


def encode(params) -> np.ndarray:
    """Approximate inverse of ``decode``; used to place declared starts."""
    ks, ts, ss, rs, vs, kf, tf, sf, rf, vf = (float(v) for v in params)
    def unlogbox(v, lo, hi):
        frac = math.log(max(v, lo * 1.000001) / lo) / math.log(hi / lo)
        frac = min(max(frac, 1e-6), 1 - 1e-6)
        return math.log(frac / (1 - frac))
    def unlin(v, lo, hi):
        frac = min(max((v - lo) / (hi - lo), 1e-6), 1 - 1e-6)
        return math.log(frac / (1 - frac))
    kf_hi = PARAM_BOX["kappa_fast"][1]
    z = np.zeros(10)
    z[0] = unlogbox(ks, *PARAM_BOX["kappa_slow"])
    z[1] = unlin(kf, ks, kf_hi)
    z[2] = unlogbox(ts, *PARAM_BOX["theta_slow"])
    z[3] = unlogbox(tf, *PARAM_BOX["theta_fast"])
    z[4] = unlin(ss / (ETA_MAX * math.sqrt(2 * ks * ts)), 0.0, 1.0)
    z[5] = unlin(sf / (ETA_MAX * math.sqrt(2 * kf * tf)), 0.0, 1.0)
    rb, fb = PARAM_BOX["rho_slow"], PARAM_BOX["rho_fast"]
    z[6] = math.atanh(min(max((rs - 0.5 * (rb[0] + rb[1])) / (0.5 * (rb[1] - rb[0])), -0.999), 0.999))
    z[7] = math.atanh(min(max((rf - 0.5 * (fb[0] + fb[1])) / (0.5 * (fb[1] - fb[0])), -0.999), 0.999))
    z[8] = unlogbox(vs, *V0_BOX)
    z[9] = unlogbox(vf, *V0_BOX)
    return z


@dataclass(frozen=True)
class Surface:
    spot: np.ndarray
    strike: np.ndarray
    tau: np.ndarray
    rate: np.ndarray
    carry: np.ndarray
    price: np.ndarray

    def __len__(self) -> int:
        return int(self.price.shape[0])


class FourierCalibrator:
    name = "fourier"

    def prices(self, surface: Surface, params) -> np.ndarray:
        p = [float(v) for v in params]
        return np.array([price_double_heston_call(
            float(surface.spot[i]), float(surface.strike[i]), float(surface.tau[i]),
            float(surface.rate[i]), float(surface.carry[i]), p, node_count=64)
            for i in range(len(surface))])

    def residual(self, z, surface: Surface) -> np.ndarray:
        # The engine refuses degenerate corners (Little-Heston-Trap denominator,
        # zero log argument) rather than returning a wrong number. To an optimiser
        # that is a bad point, not a crash: hand back a finite, absurd residual so
        # the trust region backtracks. Applied identically to every arm that calls
        # this engine, so no method gets an advantage from it.
        try:
            priced = self.prices(surface, decode(np.asarray(z, dtype=float)))
        except (FloatingPointError, ValueError, ZeroDivisionError, OverflowError):
            return np.ones_like(surface.price, dtype=float)
        if not np.all(np.isfinite(priced)):
            return np.ones_like(surface.price, dtype=float)
        return (priced - surface.price) / surface.spot

    def jacobian(self, z, surface):
        return None      # SciPy falls back to finite differences


class PinnCalibrator:
    name = "pinn"

    def __init__(self, model):
        self.model = model
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _prices_torch(self, surface: Surface, zt: torch.Tensor) -> torch.Tensor:
        n = len(surface)
        vec = decode(zt)
        p = {name: vec[CANONICAL.index(name)].expand(n) for name in PARAM_NAMES}
        t = lambda a: torch.tensor(a, dtype=torch.float64)
        return self.model.price(t(surface.spot), vec[4].expand(n), vec[9].expand(n),
                                t(surface.tau), t(surface.strike), t(surface.rate),
                                t(surface.carry), p)

    def prices(self, surface: Surface, params) -> np.ndarray:
        n = len(surface)
        t = lambda a: torch.tensor(np.asarray(a, dtype=np.float64), dtype=torch.float64)
        p = {name: torch.full((n,), float(params[CANONICAL.index(name)]), dtype=torch.float64)
             for name in PARAM_NAMES}
        with torch.no_grad():
            return self.model.price(t(surface.spot),
                                    torch.full((n,), float(params[4]), dtype=torch.float64),
                                    torch.full((n,), float(params[9]), dtype=torch.float64),
                                    t(surface.tau), t(surface.strike), t(surface.rate),
                                    t(surface.carry), p).numpy()

    def residual(self, z, surface: Surface) -> np.ndarray:
        zt = torch.tensor(np.asarray(z, dtype=float), dtype=torch.float64)
        with torch.no_grad():
            pr = self._prices_torch(surface, zt).numpy()
        return (pr - surface.price) / surface.spot

    def jacobian(self, z, surface: Surface) -> np.ndarray:
        """Exact d(residual)/dz by autograd -- the point of a differentiable pricer."""
        zt = torch.tensor(np.asarray(z, dtype=float), dtype=torch.float64, requires_grad=True)
        jac = torch.autograd.functional.jacobian(
            lambda t: (self._prices_torch(surface, t)
                       - torch.tensor(surface.price, dtype=torch.float64))
                      / torch.tensor(surface.spot, dtype=torch.float64),
            zt, vectorize=True)
        return jac.detach().numpy()


def calibrate(engine, surface: Surface, starts, *, max_nfev: int = 300,
              loss: str = "linear", f_scale: float = 1.0) -> dict:
    """Bounded multi-start least squares. Every start and its outcome is retained."""
    records = []
    t0 = time.perf_counter()
    for i, start in enumerate(starts):
        z0 = encode(start)
        kwargs = dict(max_nfev=max_nfev, loss=loss, f_scale=f_scale,
                      xtol=1e-12, ftol=1e-12, gtol=1e-12)
        if engine.jacobian(z0, surface) is not None:
            kwargs["jac"] = lambda z, s=surface: engine.jacobian(z, s)
        else:
            kwargs["diff_step"] = 1e-6
        result = least_squares(engine.residual, z0, args=(surface,), **kwargs)
        params = np.array([float(v) for v in decode(result.x)])
        residual = engine.residual(result.x, surface)
        records.append({"start_index": i, "params": params,
                        "objective": float(np.sqrt(np.mean(residual ** 2))),
                        "nfev": int(result.nfev), "status": int(result.status),
                        "success": bool(result.success)})
    best = min(records, key=lambda r: (r["objective"], r["start_index"]))
    return {"engine": engine.name, "best": best, "starts": records,
            "seconds": time.perf_counter() - t0}
