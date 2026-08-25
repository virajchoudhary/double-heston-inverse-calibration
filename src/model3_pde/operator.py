"""Canonical Double Heston pricing-PDE operator.

The state variable ``maturity`` is time to maturity ``tau = T - t``.  The
canonical production characteristic function implies additive independent
variance factors, so there is no variance/variance cross-derivative term.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

PARAMETER_ORDER = (
    "kappa_slow",
    "theta_slow",
    "sigma_slow",
    "rho_slow",
    "v0_slow",
    "kappa_fast",
    "theta_fast",
    "sigma_fast",
    "rho_fast",
    "v0_fast",
)


@dataclass(frozen=True)
class PDEState:
    """Differentiable collocation state in years and spot units."""

    spot: torch.Tensor
    variance_slow: torch.Tensor
    variance_fast: torch.Tensor
    maturity: torch.Tensor

    def __post_init__(self) -> None:
        fields = {
            "spot": self.spot,
            "variance_slow": self.variance_slow,
            "variance_fast": self.variance_fast,
            "maturity": self.maturity,
        }
        if any(value.ndim != 1 for value in fields.values()):
            raise ValueError("PDEState tensors must be one-dimensional")
        shapes = {value.shape for value in fields.values()}
        if len(shapes) != 1:
            raise ValueError("PDEState tensors must have identical shapes")
        if any(not value.is_floating_point() for value in fields.values()):
            raise TypeError("PDEState tensors must use a floating-point dtype")
        if any(value.dtype != torch.float64 for value in fields.values()):
            raise TypeError("PDEState calculations require float64")
        if any(not value.is_leaf or not value.requires_grad for value in fields.values()):
            raise ValueError(
                "spot, both variances, and maturity must all be autograd leaves; "
                "silent zero derivatives are forbidden"
            )
        if not torch.isfinite(self.spot).all() or bool(torch.any(self.spot <= 0)):
            raise ValueError("spot must be finite and strictly positive")
        if (
            not torch.isfinite(self.variance_slow).all()
            or not torch.isfinite(self.variance_fast).all()
            or bool(torch.any(self.variance_slow <= 0))
            or bool(torch.any(self.variance_fast <= 0))
        ):
            raise ValueError("variance states must be finite and strictly positive")
        if not torch.isfinite(self.maturity).all() or bool(torch.any(self.maturity < 0)):
            raise ValueError("maturity must be finite and non-negative")


def _require_same_points(*values: torch.Tensor, name: str) -> None:
    shapes = {tuple(value.shape) for value in values}
    if len(shapes) != 1:
        raise ValueError(f"{name} must have one common shape")


def validate_canonical_parameters(parameters: torch.Tensor) -> None:
    if parameters.ndim != 2 or parameters.shape[1] != len(PARAMETER_ORDER):
        raise ValueError(f"parameters must have shape (points, {len(PARAMETER_ORDER)})")
    if parameters.dtype != torch.float64:
        raise TypeError("parameters must be float64")
    if not torch.isfinite(parameters).all():
        raise ValueError("parameters must be finite")
    if bool(torch.any(parameters[:, [0, 1, 2, 4, 5, 6, 7, 9]] <= 0)):
        raise ValueError("kappa, theta, sigma, and v0 must be strictly positive")
    if bool(torch.any(parameters[:, 0] >= parameters[:, 5])):
        raise ValueError("canonical ordering requires kappa_slow < kappa_fast")
    slow_feller = 2 * parameters[:, 0] * parameters[:, 1] - parameters[:, 2] ** 2
    fast_feller = 2 * parameters[:, 5] * parameters[:, 6] - parameters[:, 7] ** 2
    if bool(torch.any(slow_feller <= 0)) or bool(torch.any(fast_feller <= 0)):
        raise ValueError("both factor Feller gaps must be strictly positive")
    if (
        bool(torch.any(parameters[:, 3] <= -1))
        or bool(torch.any(parameters[:, 3] >= 1))
        or bool(torch.any(parameters[:, 8] <= -1))
        or bool(torch.any(parameters[:, 8] >= 1))
    ):
        raise ValueError("correlations must lie strictly inside (-1, 1)")
    disk = parameters[:, 3] ** 2 + parameters[:, 8] ** 2
    if bool(torch.any(disk >= 1)):
        raise ValueError("correlations must lie strictly inside the joint unit disk")


_validate_canonical_parameters = validate_canonical_parameters


def _derivative(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    gradient = torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )[0]
    if gradient is None:
        raise RuntimeError("autograd returned None; refusing to silently zero a PDE term")
    return gradient


def double_heston_pde_residual(
    prices: torch.Tensor,
    state: PDEState,
    parameters: torch.Tensor,
    *,
    risk_free_rate: torch.Tensor,
    dividend_yield: torch.Tensor,
) -> torch.Tensor:
    """Evaluate ``V_tau minus the backward-pricing generator minus discount``.

    In tau convention the residual is
    ``V_tau - [(r-q) S V_S + factor drifts + diffusion/cross terms - r V]``.
    A solution has residual zero.
    """
    if prices.ndim != 1 or prices.dtype != torch.float64:
        raise ValueError("prices must be a one-dimensional float64 tensor")
    if not torch.isfinite(prices).all():
        raise ValueError("prices must be finite")
    validate_canonical_parameters(parameters)
    _require_same_points(
        prices,
        state.spot,
        state.variance_slow,
        state.variance_fast,
        state.maturity,
        risk_free_rate,
        dividend_yield,
        name="collocation arrays",
    )
    if parameters.shape[0] != prices.shape[0]:
        raise ValueError("parameters must contain one row per collocation point")
    if risk_free_rate.dtype != torch.float64 or dividend_yield.dtype != torch.float64:
        raise TypeError("risk-free rate and dividend yield must be float64")
    if not torch.isfinite(risk_free_rate).all() or not torch.isfinite(dividend_yield).all():
        raise ValueError("rates and carries must be finite")

    kappa_s, theta_s, sigma_s, rho_s, _ = (parameters[:, index] for index in range(5))
    kappa_f, theta_f, sigma_f, rho_f, _ = (parameters[:, index] for index in range(5, 10))

    price_derivative_spot = _derivative(prices, state.spot)
    price_second_spot = _derivative(price_derivative_spot, state.spot)
    price_derivative_tau = _derivative(prices, state.maturity)
    price_derivative_v_slow = _derivative(prices, state.variance_slow)
    price_second_v_slow = _derivative(price_derivative_v_slow, state.variance_slow)
    price_cross_spot_v_slow = _derivative(price_derivative_spot, state.variance_slow)
    price_derivative_v_fast = _derivative(prices, state.variance_fast)
    price_second_v_fast = _derivative(price_derivative_v_fast, state.variance_fast)
    price_cross_spot_v_fast = _derivative(price_derivative_spot, state.variance_fast)

    spot_drift = (risk_free_rate - dividend_yield) * state.spot * price_derivative_spot
    discount = -risk_free_rate * prices
    spot_diffusion = (
        0.5 * (state.variance_slow + state.variance_fast) * state.spot.square() * price_second_spot
    )
    slow_variance_drift = kappa_s * (theta_s - state.variance_slow) * price_derivative_v_slow
    fast_variance_drift = kappa_f * (theta_f - state.variance_fast) * price_derivative_v_fast
    slow_mixed = (
        rho_s * sigma_s * state.variance_slow * state.spot * price_cross_spot_v_slow
    )
    fast_mixed = (
        rho_f * sigma_f * state.variance_fast * state.spot * price_cross_spot_v_fast
    )
    slow_diffusion = 0.5 * sigma_s.square() * state.variance_slow * price_second_v_slow
    fast_diffusion = 0.5 * sigma_f.square() * state.variance_fast * price_second_v_fast
    generator = (
        spot_drift
        + spot_diffusion
        + slow_variance_drift
        + fast_variance_drift
        + slow_mixed
        + fast_mixed
        + slow_diffusion
        + fast_diffusion
    )
    residual = price_derivative_tau - (generator + discount)
    if not torch.isfinite(residual).all():
        raise FloatingPointError("non-finite Double Heston PDE residual")
    return residual
