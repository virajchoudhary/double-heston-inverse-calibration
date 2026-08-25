"""Minimal joint inverse/forward Model 3 skeleton; no research training here."""

from __future__ import annotations

import torch
from torch import nn

from models.pinn_model import DoubleHestonConstraintMap, PhysicsInformedInverseCalibrator
from src.model3_pde.operator import validate_canonical_parameters
from .losses import discounted_arbitrage_bounds
from .operator import PDEState

FORWARD_INPUT_SIZE = 16
FORWARDED_PARAMETER_INDICES = (0, 1, 2, 3, 5, 6, 7, 8)
TERMINAL_SUPPORT_CUTOFF_YEARS = 7.0 / 365.0


def structural_parameter_conditioning(parameters: torch.Tensor) -> torch.Tensor:
    """Exclude inverse-initial v0 fields; PDE state variances supply them."""
    if parameters.ndim != 2 or parameters.shape[1] != 10:
        raise ValueError("parameters must have shape (points, 10)")
    return parameters[:, list(FORWARDED_PARAMETER_INDICES)]


class ConditionalDoubleHestonPriceNetwork(nn.Module):
    """Smooth conditional forward network mapped into European no-arb bounds."""

    def __init__(
        self,
        *,
        hidden_sizes: tuple[int, ...] = (128, 128, 64),
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        if not hidden_sizes or any(size <= 0 for size in hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers")
        layers: list[nn.Module] = []
        previous_size = FORWARD_INPUT_SIZE
        factories = {"tanh": nn.Tanh, "silu": nn.SiLU, "gelu": nn.GELU}
        try:
            factory = factories[activation.lower()]
        except KeyError as error:
            raise ValueError("activation must be tanh, silu, or gelu") from error
        for size in hidden_sizes:
            layers.extend((nn.Linear(previous_size, size), factory()))
            previous_size = size
        layers.append(nn.Linear(previous_size, 1))
        self.network = nn.Sequential(*layers)
        self.double()

    @staticmethod
    def _column(values: torch.Tensor, name: str) -> torch.Tensor:
        if values.ndim == 1:
            return values.unsqueeze(1)
        if values.ndim == 2 and values.shape[1] == 1:
            return values.reshape(-1)
        raise ValueError(f"{name} must contain one scalar per collocation point")

    def forward(
        self,
        state: PDEState,
        *,
        strike: torch.Tensor,
        risk_free_rate: torch.Tensor,
        dividend_yield: torch.Tensor,
        is_call: torch.Tensor,
        parameters: torch.Tensor,
    ) -> torch.Tensor:
        columns = [
            self._column(state.spot.log(), "spot"),
            self._column(strike.log(), "strike"),
            self._column(state.maturity, "maturity"),
            self._column(risk_free_rate, "risk_free_rate"),
            self._column(dividend_yield, "dividend_yield"),
            self._column(state.variance_slow, "variance_slow"),
            self._column(state.variance_fast, "variance_fast"),
            self._column(is_call.to(dtype=torch.float64), "is_call"),
        ]
        if strike.ndim != 1 or strike.shape != state.spot.shape:
            raise ValueError("strike must contain one positive value per collocation point")
        if bool(torch.any(strike <= 0)) or not torch.isfinite(strike).all():
            raise ValueError("strike must be finite and strictly positive")
        if is_call.shape != state.spot.shape or is_call.dtype != torch.bool:
            raise ValueError("is_call must be a boolean tensor aligned with collocation points")
        validate_canonical_parameters(parameters)
        features = torch.cat(
            (*columns, structural_parameter_conditioning(parameters)), dim=1
        )
        if features.shape[1] != FORWARD_INPUT_SIZE:
            raise AssertionError("forward feature width changed without a protocol update")
        raw = self.network(features).reshape(-1)
        lower, upper = discounted_arbitrage_bounds(
            state,
            strike,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            is_call=is_call,
        )
        payoff = torch.where(
            is_call.bool(),
            torch.clamp(state.spot - strike, min=0.0),
            torch.clamp(strike - state.spot, min=0.0),
        )
        bounded_base = lower + (upper - lower) * torch.sigmoid(raw)
        # C2 terminal blending below the shortest supported tenor.  At and
        # above seven days the weight is exactly zero, so the full bounded-base
        # range remains representable on every frozen-data maturity.
        relative_maturity = state.maturity / TERMINAL_SUPPORT_CUTOFF_YEARS
        inside_pre_support = relative_maturity < 1.0
        terminal_weight = torch.where(
            inside_pre_support,
            (1.0 - relative_maturity).clamp_min(0.0) ** 4,
            torch.zeros_like(relative_maturity),
        )
        return bounded_base - (bounded_base - payoff) * terminal_weight


class Model3PDESystem(nn.Module):
    """R2 inverse encoder plus explicit-PDE conditional pricing network."""

    def __init__(self, *, r2_input_size: int = 100) -> None:
        super().__init__()
        self.inverse = PhysicsInformedInverseCalibrator(input_size=r2_input_size)
        self.constraint_map = DoubleHestonConstraintMap()
        self.pricing = ConditionalDoubleHestonPriceNetwork()
        self.double()

    def predict_parameters(self, r2_features: torch.Tensor) -> torch.Tensor:
        if r2_features.ndim != 2 or r2_features.shape[1] != 100:
            raise ValueError("r2_features must have shape (batch, 100)")
        if not r2_features.is_floating_point():
            raise TypeError("r2_features must use a floating-point dtype")
        if not torch.isfinite(r2_features).all():
            raise ValueError("r2_features must be finite")
        # The shared frozen builder stores compact R2 features as float32;
        # Model 3 physics requires an explicit float64 computation boundary.
        return self.inverse(r2_features.to(dtype=torch.float64))

    def predict_prices(
        self,
        state: PDEState,
        *,
        strike: torch.Tensor,
        risk_free_rate: torch.Tensor,
        dividend_yield: torch.Tensor,
        is_call: torch.Tensor,
        parameters: torch.Tensor,
    ) -> torch.Tensor:
        return self.pricing(
            state,
            strike=strike,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            is_call=is_call,
            parameters=parameters,
        )
