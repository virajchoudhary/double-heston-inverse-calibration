"""Deterministic point classes and samplers for the five V1 loss terms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import BaselineConfig, DomainConfig, variance_state_bounds
from .parameter_source import ParameterSource


def _uniform(
    generator: torch.Generator,
    count: int,
    minimum: float,
    maximum: float,
) -> torch.Tensor:
    if count <= 0:
        raise ValueError("point count must be strictly positive")
    draw = torch.rand((count,), generator=generator, dtype=torch.float64)
    return minimum + (maximum - minimum) * draw


def _state_variances(
    generator: torch.Generator,
    count: int,
    domain: DomainConfig,
    source: ParameterSource,
) -> tuple[torch.Tensor, torch.Tensor]:
    slow_minimum, slow_maximum = variance_state_bounds(float(source.vector[1]), domain)
    fast_minimum, fast_maximum = variance_state_bounds(float(source.vector[6]), domain)
    return (
        _uniform(generator, count, slow_minimum, slow_maximum),
        _uniform(generator, count, fast_minimum, fast_maximum),
    )


def _leaf(values: torch.Tensor, device: torch.device | str = "cpu") -> torch.Tensor:
    return values.detach().to(device=device, dtype=torch.float64).clone().requires_grad_(True)


@dataclass(frozen=True)
class PDEPoints:
    """Interior points whose raw network output is differentiated."""

    spot: torch.Tensor
    variance_slow: torch.Tensor
    variance_fast: torch.Tensor
    tau: torch.Tensor
    strike: torch.Tensor
    rate: torch.Tensor
    carry: torch.Tensor
    parameters: torch.Tensor

    def __post_init__(self) -> None:
        fields = (
            self.spot,
            self.variance_slow,
            self.variance_fast,
            self.tau,
            self.strike,
            self.rate,
            self.carry,
        )
        if any(field.ndim != 1 for field in fields):
            raise ValueError("PDE point tensors must be one-dimensional")
        if len({field.shape for field in fields}) != 1:
            raise ValueError("PDE point tensors must have equal shapes")
        if any(field.dtype != torch.float64 for field in fields):
            raise TypeError("PDE points require float64 tensors")
        if any(not field.is_leaf or not field.requires_grad for field in fields):
            raise ValueError("PDE point state tensors must be differentiable leaves")
        if self.parameters.shape != (self.spot.shape[0], 10):
            raise ValueError("PDE parameters must have shape (points, 10)")
        if self.parameters.dtype != torch.float64:
            raise TypeError("PDE parameters require float64")

    @property
    def features(self) -> torch.Tensor:
        return torch.stack(
            (
                self.spot,
                self.variance_slow,
                self.variance_fast,
                self.tau,
                self.strike,
                self.rate,
                self.carry,
            ),
            dim=1,
        )


@dataclass(frozen=True)
class TerminalPoints:
    """Explicit tau=0 points for the exact call payoff loss."""

    spot: torch.Tensor
    variance_slow: torch.Tensor
    variance_fast: torch.Tensor
    tau: torch.Tensor
    strike: torch.Tensor
    rate: torch.Tensor
    carry: torch.Tensor

    def __post_init__(self) -> None:
        fields = self.fields
        if any(field.ndim != 1 for field in fields):
            raise ValueError("terminal point tensors must be one-dimensional")
        if len({field.shape for field in fields}) != 1:
            raise ValueError("terminal point tensors must have equal shapes")
        if any(field.dtype != torch.float64 for field in fields):
            raise TypeError("terminal points require float64 tensors")
        if torch.any(self.tau != 0):
            raise ValueError("terminal points require tau exactly zero")

    @property
    def fields(self) -> tuple[torch.Tensor, ...]:
        return (
            self.spot,
            self.variance_slow,
            self.variance_fast,
            self.tau,
            self.strike,
            self.rate,
            self.carry,
        )

    @property
    def features(self) -> torch.Tensor:
        return torch.stack(self.fields, dim=1)


@dataclass(frozen=True)
class BoundaryPoints:
    """One explicit spot-boundary point class."""

    spot: torch.Tensor
    variance_slow: torch.Tensor
    variance_fast: torch.Tensor
    tau: torch.Tensor
    strike: torch.Tensor
    rate: torch.Tensor
    carry: torch.Tensor
    target: torch.Tensor
    name: str

    def __post_init__(self) -> None:
        fields = self.fields
        if any(field.ndim != 1 for field in fields):
            raise ValueError("boundary point tensors must be one-dimensional")
        if len({field.shape for field in fields}) != 1 or self.target.shape != self.spot.shape:
            raise ValueError("boundary point tensors must have equal shapes")
        if any(field.dtype != torch.float64 for field in fields) or self.target.dtype != torch.float64:
            raise TypeError("boundary points require float64 tensors")
        if self.name not in {"low_s", "high_s"}:
            raise ValueError("boundary name must be low_s or high_s")

    @property
    def fields(self) -> tuple[torch.Tensor, ...]:
        return (
            self.spot,
            self.variance_slow,
            self.variance_fast,
            self.tau,
            self.strike,
            self.rate,
            self.carry,
        )

    @property
    def features(self) -> torch.Tensor:
        return torch.stack(self.fields, dim=1)


class LowSBoundaryPoints(BoundaryPoints):
    """Typed low-spot boundary batch."""


class HighSBoundaryPoints(BoundaryPoints):
    """Typed high-spot boundary batch."""


def _make_generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(int(seed))


def sample_pde_points(
    point_count: int,
    *,
    config: BaselineConfig,
    parameter_source: ParameterSource,
    seed: int,
    device: torch.device | str = "cpu",
) -> PDEPoints:
    """Sample deterministic interior points and state-conditioned parameters."""
    generator = _make_generator(seed)
    domain = config.domain
    spot = _leaf(_uniform(generator, point_count, domain.spot_min, domain.spot_max), device)
    variance_slow, variance_fast = _state_variances(
        generator, point_count, domain, parameter_source
    )
    tau = _leaf(_uniform(generator, point_count, domain.tau_min, domain.tau_max), device)
    moneyness = _uniform(generator, point_count, domain.moneyness_min, domain.moneyness_max)
    strike = _leaf(spot.detach().cpu() * moneyness, device)
    rate = _leaf(_uniform(generator, point_count, domain.rate_min, domain.rate_max), device)
    carry = _leaf(_uniform(generator, point_count, domain.carry_min, domain.carry_max), device)
    conditioned = np.stack(
        [
            parameter_source.parameters_for_state(float(vs), float(vf))
            for vs, vf in zip(variance_slow, variance_fast, strict=True)
        ],
        axis=0,
    )
    return PDEPoints(
        spot=spot,
        variance_slow=_leaf(variance_slow, device),
        variance_fast=_leaf(variance_fast, device),
        tau=tau,
        strike=strike,
        rate=rate,
        carry=carry,
        parameters=torch.from_numpy(conditioned).to(device=device, dtype=torch.float64),
    )


def _sample_terminal_or_boundary_fields(
    point_count: int,
    *,
    config: BaselineConfig,
    parameter_source: ParameterSource,
    seed: int,
    spot_value: float | None = None,
    sample_global_strike: bool = False,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, ...]:
    generator = _make_generator(seed)
    domain = config.domain
    if spot_value is None:
        spot = _uniform(generator, point_count, domain.spot_min, domain.spot_max)
    else:
        spot = torch.full((point_count,), float(spot_value), dtype=torch.float64)
    variance_slow, variance_fast = _state_variances(
        generator, point_count, domain, parameter_source
    )
    tau = _uniform(
        generator,
        point_count,
        domain.tau_min,
        domain.tau_max,
    )
    if not sample_global_strike:
        moneyness = _uniform(generator, point_count, domain.moneyness_min, domain.moneyness_max)
        strike = spot * moneyness
    else:
        strike = _uniform(
            generator,
            point_count,
            domain.spot_min * domain.moneyness_min,
            domain.spot_max * domain.moneyness_max,
        )
    rate = _uniform(generator, point_count, domain.rate_min, domain.rate_max)
    carry = _uniform(generator, point_count, domain.carry_min, domain.carry_max)
    return (
        _leaf(spot, device),
        _leaf(variance_slow, device),
        _leaf(variance_fast, device),
        _leaf(tau, device),
        _leaf(strike, device),
        _leaf(rate, device),
        _leaf(carry, device),
    )


def sample_terminal_points(
    point_count: int,
    *,
    config: BaselineConfig,
    parameter_source: ParameterSource,
    seed: int,
    device: torch.device | str = "cpu",
) -> TerminalPoints:
    fields = _sample_terminal_or_boundary_fields(
        point_count,
        config=config,
        parameter_source=parameter_source,
        seed=seed,
        device=device,
    )
    return TerminalPoints(
        spot=fields[0],
        variance_slow=fields[1],
        variance_fast=fields[2],
        tau=torch.zeros_like(fields[3]),
        strike=fields[4],
        rate=fields[5],
        carry=fields[6],
    )


def sample_low_s_boundary_points(
    point_count: int,
    *,
    config: BaselineConfig,
    parameter_source: ParameterSource,
    seed: int,
    device: torch.device | str = "cpu",
) -> LowSBoundaryPoints:
    fields = _sample_terminal_or_boundary_fields(
        point_count,
        config=config,
        parameter_source=parameter_source,
        seed=seed,
        spot_value=config.domain.boundary_spot_low,
        sample_global_strike=True,
        device=device,
    )
    target = torch.zeros(point_count, dtype=torch.float64, device=device)
    return LowSBoundaryPoints(*fields, target=target, name="low_s")


def sample_high_s_boundary_points(
    point_count: int,
    *,
    config: BaselineConfig,
    parameter_source: ParameterSource,
    seed: int,
    device: torch.device | str = "cpu",
) -> HighSBoundaryPoints:
    fields = _sample_terminal_or_boundary_fields(
        point_count,
        config=config,
        parameter_source=parameter_source,
        seed=seed,
        spot_value=config.domain.boundary_spot_high,
        sample_global_strike=True,
        device=device,
    )
    spot, variance_slow, variance_fast, tau, strike, rate, carry = fields
    target = spot * torch.exp(-carry * tau) - strike * torch.exp(-rate * tau)
    return HighSBoundaryPoints(
        spot,
        variance_slow,
        variance_fast,
        tau,
        strike,
        rate,
        carry,
        target=target,
        name="high_s",
    )


__all__ = [
    "BoundaryPoints",
    "HighSBoundaryPoints",
    "LowSBoundaryPoints",
    "PDEPoints",
    "TerminalPoints",
    "sample_high_s_boundary_points",
    "sample_low_s_boundary_points",
    "sample_pde_points",
    "sample_terminal_points",
]
