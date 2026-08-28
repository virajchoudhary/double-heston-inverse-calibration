"""Explicit V1 data, PDE, terminal, and spot-boundary losses."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

import torch

from src.model3_pde.operator import PDEState, double_heston_pde_residual

from .collocation import BoundaryPoints, PDEPoints, TerminalPoints


def _finite_scalar(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim != 0:
        raise ValueError(f"{name} must be scalar")
    if not torch.isfinite(value):
        raise FloatingPointError(f"{name} is non-finite")
    return value


def data_loss(
    predicted_prices: torch.Tensor,
    reference_prices: torch.Tensor,
    spot: torch.Tensor,
) -> torch.Tensor:
    """MSE of ``(prediction-reference)/S`` as required by the protocol."""
    predicted = predicted_prices.reshape(-1)
    reference = reference_prices.reshape(-1)
    spot_values = spot.reshape(-1)
    if predicted.shape != reference.shape or predicted.shape != spot_values.shape:
        raise ValueError("predicted, reference, and spot arrays must align")
    if not torch.isfinite(predicted).all() or not torch.isfinite(reference).all():
        raise ValueError("data loss inputs must be finite")
    if bool(torch.any(spot_values <= 0)):
        raise ValueError("spot must be strictly positive")
    return torch.mean(((predicted - reference) / spot_values).square())


def normalized_data_loss(
    model: torch.nn.Module,
    features: torch.Tensor,
    reference_prices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate model data loss and return loss, predictions, and raw RMSE."""
    predicted = model(features).reshape(-1)
    loss = data_loss(predicted, reference_prices, features[:, 0])
    raw_rmse = torch.sqrt(torch.mean((predicted - reference_prices.reshape(-1)).square()))
    return loss, predicted, raw_rmse


def pde_residual(model: torch.nn.Module, points: PDEPoints) -> torch.Tensor:
    """Return the canonical raw-price forward PDE residual at interior points."""
    prices = model(points.features).reshape(-1)
    state = PDEState(
        spot=points.spot,
        variance_slow=points.variance_slow,
        variance_fast=points.variance_fast,
        maturity=points.tau,
    )
    return double_heston_pde_residual(
        prices,
        state,
        points.parameters,
        risk_free_rate=points.rate,
        dividend_yield=points.carry,
    )


def pde_loss(
    model: torch.nn.Module,
    points: PDEPoints,
    *,
    scale_floor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return scaled PDE MSE and unscaled residuals for RMS logging."""
    if scale_floor <= 0:
        raise ValueError("scale_floor must be positive")
    residual = pde_residual(model, points)
    prices = model(points.features).reshape(-1)
    scale = prices.detach().abs().clamp_min(float(scale_floor))
    loss = torch.mean((residual / scale).square())
    return _finite_scalar(loss, "PDE loss"), residual


def terminal_payoff(spot: torch.Tensor, strike: torch.Tensor) -> torch.Tensor:
    """Undiscounted European CALL payoff at exactly tau=0."""
    return torch.clamp(spot - strike, min=0.0)


def terminal_loss(
    model: torch.nn.Module,
    points: TerminalPoints,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return exact terminal payoff MSE, predictions, and target."""
    predicted = model(points.features).reshape(-1)
    target = terminal_payoff(points.spot, points.strike)
    loss = torch.mean((predicted - target).square())
    return _finite_scalar(loss, "terminal loss"), predicted, target


def low_s_boundary_loss(
    model: torch.nn.Module,
    points: BoundaryPoints,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the independently logged low-S -> 0 boundary MSE."""
    if points.name != "low_s":
        raise ValueError("low_s_boundary_loss requires low_s points")
    predicted = model(points.features).reshape(-1)
    target = torch.zeros_like(predicted)
    loss = torch.mean((predicted - target).square())
    return _finite_scalar(loss, "low-S boundary loss"), predicted, target


def high_s_boundary_loss(
    model: torch.nn.Module,
    points: BoundaryPoints,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the independently logged high-S call asymptote MSE."""
    if points.name != "high_s":
        raise ValueError("high_s_boundary_loss requires high_s points")
    predicted = model(points.features).reshape(-1)
    target = points.target
    loss = torch.mean((predicted - target).square())
    return _finite_scalar(loss, "high-S boundary loss"), predicted, target


@dataclass(frozen=True)
class LossComponents:
    """The five explicit components used by the weighted V1 objective."""

    data: torch.Tensor
    pde: torch.Tensor
    terminal: torch.Tensor
    low_boundary: torch.Tensor
    high_boundary: torch.Tensor

    @property
    def total(self) -> torch.Tensor:
        return weighted_total_loss(self)

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "data": self.data,
            "pde": self.pde,
            "terminal": self.terminal,
            "low_boundary": self.low_boundary,
            "high_boundary": self.high_boundary,
        }


def weighted_total_loss(
    components: LossComponents | Mapping[str, torch.Tensor],
    weights: Mapping[str, float] | None = None,
) -> torch.Tensor:
    """Explicit weighted sum API; V1 defaults every lambda to exactly 1.0."""
    values: Mapping[str, torch.Tensor] = (
        components.as_dict() if isinstance(components, LossComponents) else components
    )
    component_names = ("data", "pde", "terminal", "low_boundary", "high_boundary")
    if set(values) != set(component_names):
        raise ValueError(f"loss components must be exactly {component_names}")
    selected = (
        {"data": 1.0, "pde": 1.0, "boundary": 1.0, "terminal": 1.0}
        if weights is None
        else dict(weights)
    )
    weight_names = ("data", "pde", "boundary", "terminal")
    if set(selected) != set(weight_names):
        raise ValueError(f"loss weights must be exactly {weight_names}")
    for name in component_names:
        if not torch.isfinite(values[name]):
            raise FloatingPointError(f"non-finite {name} loss")
    for name in weight_names:
        if not torch.isfinite(torch.tensor(float(selected[name]), dtype=torch.float64)):
            raise FloatingPointError(f"non-finite {name} weight")
    boundary = values["low_boundary"] + values["high_boundary"]
    total = (
        float(selected["data"]) * values["data"]
        + float(selected["pde"]) * values["pde"]
        + float(selected["boundary"]) * boundary
        + float(selected["terminal"]) * values["terminal"]
    )
    return _finite_scalar(total, "total loss")


# Common spelling aliases for downstream callers.
physics_loss = pde_loss
terminal_payoff_loss = terminal_loss
boundary_low_s_loss = low_s_boundary_loss
boundary_high_s_loss = high_s_boundary_loss

__all__ = [
    "LossComponents",
    "boundary_high_s_loss",
    "boundary_low_s_loss",
    "data_loss",
    "high_s_boundary_loss",
    "low_s_boundary_loss",
    "normalized_data_loss",
    "pde_loss",
    "pde_residual",
    "physics_loss",
    "terminal_loss",
    "terminal_payoff",
    "terminal_payoff_loss",
    "weighted_total_loss",
]
