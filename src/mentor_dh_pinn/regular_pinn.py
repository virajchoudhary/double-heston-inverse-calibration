"""One-/two-factor extension of the existing regular implied-variance PINN.

Coordinates are ``[x, v1, (v2), tau]``, where ``x=log(F/K)`` and tau is years.
Each structural factor contains ``[kappa, theta, sigma, rho]``. The network
approximates log implied total variance; Black's formula then returns the
normalized forward call ``c=C/(K exp(-r tau))``. Fourier prices are never used
by this module, including its derivatives used for inverse calibration.

The forward-measure PDE is

    c_tau = .5 sum(v_i) (c_xx-c_x)
            + sum(kappa_i (theta_i-v_i) c_vi
                  + rho_i sigma_i v_i c_xvi + .5 sigma_i**2 v_i c_vivi).

Independent CIR factors imply no c_v1v2 term. The residual uses the same
log-total-variance substitution as single_heston_pinn/src/pinn_heston_core.py.
It is divided by Black's derivative with respect to total variance, then by
the sum of current and expected average variance. The analytic backbone is
exact in the zero-vol-of-vol limit at the zero-head initialization.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


def mean_reversion_ratio(u):
    """Stable ratio on the strictly positive kappa*tau training domain."""
    return -mx.expm1(-u) / u


def _unit_log(value, low, high):
    return 2.0 * (mx.log(value) - math.log(low)) / math.log(high / low) - 1.0


def expected_variance(coords, structural):
    """Expected average integrated variance, summed over the CIR factors."""
    variance = coords[..., 1:-1]
    tau = coords[..., -1:]
    kappa, theta = structural[..., 0], structural[..., 1]
    return mx.sum(theta + (variance - theta) * mean_reversion_ratio(kappa * tau), axis=-1)


def black_call(x, total_variance):
    """Normalized Black call; exact payoff at zero total variance.

    Float32 cancellation can affect tiny time values; this is a display price
    helper, not an independent high-precision reference or an IV inversion.
    """
    positive = total_variance > 0
    safe = mx.where(positive, total_variance, mx.ones_like(total_variance))
    root = mx.sqrt(safe)
    d1 = x / root + 0.5 * root
    d2 = d1 - root
    cdf = lambda y: 0.5 * (1.0 + mx.erf(y / math.sqrt(2.0)))
    price = mx.exp(x) * cdf(d1) - cdf(d2)
    return mx.where(positive, price, mx.maximum(mx.exp(x) - 1.0, 0.0))


class RegularVariancePINN(nn.Module):
    """Smooth conditioned PINN, with 13 single / 24 double Heston features.

    Parameter bounds here normalize features; physical feasibility and factor
    ordering are responsibilities of the data sampler and inverse transform.
    There is no dropout: repeated state derivatives must describe one fixed
    smooth price surface. Weight regularization can be applied in the trainer.
    """

    def __init__(self, factors=2, width=160, depth=5, *, tau_min=7.0 / 365.0,
                 tau_max=2.0, x_half_width=3.5, correction_limit=1.8):
        super().__init__()
        if factors not in (1, 2):
            raise ValueError("factors must be 1 or 2")
        if width < 1 or depth < 1 or not 0 < tau_min < tau_max:
            raise ValueError("positive width/depth and 0 < tau_min < tau_max required")
        if x_half_width <= 0 or correction_limit <= 0:
            raise ValueError("x_half_width and correction_limit must be positive")
        self.factors = factors
        self.width, self.depth = width, depth
        self.tau_min, self.tau_max = tau_min, tau_max
        self.x_half_width, self.correction_limit = x_half_width, correction_limit
        self.feature_count = 13 if factors == 1 else 24
        sizes = [self.feature_count] + [width] * depth
        self.hidden = [nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:])]
        self.head = nn.Linear(width, 1)
        self.head.weight = mx.zeros_like(self.head.weight)
        self.head.bias = mx.zeros_like(self.head.bias)

    def features(self, coords, structural):
        if coords.shape[-1] != self.factors + 2 or structural.shape[-2:] != (self.factors, 4):
            raise ValueError("coordinates/structural shape does not match factors")
        x, tau = coords[..., 0], coords[..., -1]
        vbar = expected_variance(coords, structural)
        z = x / mx.sqrt(tau * vbar + x * x / 64.0 + 1e-12)
        zt = mx.tanh(z / 1.5)
        features = [x / self.x_half_width, z / 8.0, zt,
                    _unit_log(tau, self.tau_min, self.tau_max)]
        skew = mx.zeros_like(x)
        # All features are pointwise; summed gradients therefore give rowwise
        # coordinate derivatives without constructing cross-batch Jacobians.
        for i in range(self.factors):
            v = coords[..., i + 1]
            kappa, theta, sigma, rho = [structural[..., i, j] + mx.zeros_like(x) for j in range(4)]
            nu = mx.tanh(sigma * mx.sqrt(tau) / mx.sqrt(vbar + 1e-9))
            eta = sigma / mx.sqrt(2.0 * kappa * theta)
            features.extend([
                _unit_log(v, 0.01, 0.3), _unit_log(kappa, 0.15, 12.0),
                _unit_log(theta, 0.01, 0.3), rho, mx.tanh(kappa * tau),
                nu, rho * nu, mx.tanh(0.5 * mx.log(v / theta)),
                2.0 * mx.log1p(eta) / math.log(4.0) - 1.0,
            ])
            skew = skew + rho * nu
        if self.factors == 2:
            total = mx.sum(coords[..., 1:-1], axis=-1)
            features.extend([2.0 * coords[..., 1] / total - 1.0, zt * skew])
        return mx.stack(features, axis=-1)

    def correction(self, coords, structural):
        hidden = self.features(coords, structural)
        for layer in self.hidden:
            hidden = mx.tanh(layer(hidden))
        return self.correction_limit * mx.tanh(self.head(hidden)[..., 0])

    def log_total_variance(self, coords, structural):
        """Defined for strictly positive tau and variance state inputs."""
        return (mx.log(coords[..., -1]) + mx.log(expected_variance(coords, structural))
                + 2.0 * self.correction(coords, structural))

    def iv(self, coords, structural):
        # Cancelling log(tau) here avoids a needless subtraction for small tau.
        return mx.sqrt(expected_variance(coords, structural)) * mx.exp(self.correction(coords, structural))

    def price(self, coords, structural):
        """Normalized forward call, with the exact payoff when tau=0."""
        tau = coords[..., -1]
        safe_tau = mx.where(tau > 0, tau, mx.ones_like(tau))
        safe_coords = mx.concatenate([coords[..., :-1], safe_tau[..., None]], axis=-1)
        w = mx.exp(self.log_total_variance(safe_coords, structural))
        return black_call(coords[..., 0], mx.where(tau > 0, w, mx.zeros_like(w)))

    def __call__(self, coords, structural):
        return self.iv(coords, structural)


def residual(model, coords, structural):
    """Dimensionless Black-total-variance-vega-divided PDE residual.

    Collocation coordinates require tau>0 and all v_i>0. Returned convexity is
    Durrleman's dimensionless factor; negativity signals a butterfly violation.
    All returned derivatives retain gradients with respect to network weights.
    """
    # Reverse-over-reverse AD; only the Hessian entries in the PDE are built.
    grad = mx.grad(lambda state: mx.sum(model.log_total_variance(state, structural)))
    first = grad(coords)
    second_x = mx.grad(lambda state: mx.sum(grad(state)[..., 0]))(coords)
    ell = model.log_total_variance(coords, structural)
    w = mx.exp(ell)
    x = coords[..., 0]
    lx, lt, lxx = first[...,0], first[...,-1], second_x[...,0]
    a = 0.5 * x * x - 0.125 * w * w - 0.5 * w
    diffusion_x = (2.0 + 2.0 * (0.5 * w - x) * lx + a * lx * lx
                   + w * (lxx + lx * lx) - w * lx)
    total = mx.sum(coords[..., 1:-1], axis=-1)
    value = w * lt - 0.5 * total * diffusion_x
    for i in range(model.factors):
        j = i + 1
        lv, lxv = first[...,j], second_x[...,j]
        lvv = mx.grad(lambda state: mx.sum(grad(state)[..., j]))(coords)[..., j]
        v = coords[..., j]
        kappa, theta, sigma, rho = [structural[..., i, k] for k in range(4)]
        cross = (0.5 * w - x) * lv + a * lx * lv + w * (lxv + lx * lv)
        diffusion_v = a * lv * lv + w * (lvv + lv * lv)
        value = (value - rho * sigma * v * cross - 0.5 * sigma * sigma * v * diffusion_v
                 - kappa * (theta - v) * w * lv)
    scale = total + expected_variance(coords, structural)
    convexity = ((1.0 - 0.5 * x * lx)**2 - 0.25 * w * lx * lx
                 - w * w * lx * lx / 16.0 + 0.5 * w * (lxx + lx * lx))
    return value / scale, {"ell": ell, "w": w, "l_x": lx, "l_tau": lt,
                           "l_xx": lxx, "convexity": convexity, "scale": scale}
