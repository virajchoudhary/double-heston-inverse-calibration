"""Physics-informed inverse network for canonical Double Heston recovery."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from src.constants import PARAMETER_COUNT


_POSITIVE_EPS = 1e-6
_FELLER_SIGMA_SAFETY = 0.995
_CORRELATION_RADIUS_SAFETY = 0.995


class DoubleHestonConstraintMap(nn.Module):
    """Map unconstrained outputs into valid canonical Double Heston parameters."""

    def forward(self, unconstrained: torch.Tensor) -> torch.Tensor:
        if unconstrained.ndim < 1 or unconstrained.shape[-1] != PARAMETER_COUNT:
            raise ValueError(f"unconstrained must have final dimension {PARAMETER_COUNT}")
        if not torch.is_floating_point(unconstrained):
            raise TypeError("unconstrained must use a floating-point dtype")

        raw = unconstrained
        positive = _positive
        kappa_slow = positive(raw[..., 0])
        theta_slow = positive(raw[..., 1])
        sigma_slow = _sigma_with_feller_margin(kappa_slow, theta_slow, raw[..., 2])
        kappa_fast = kappa_slow + positive(raw[..., 5])
        theta_fast = positive(raw[..., 6])
        sigma_fast = _sigma_with_feller_margin(kappa_fast, theta_fast, raw[..., 7])
        rho_slow, rho_fast = _correlation_disk_map(raw[..., 3], raw[..., 8], raw.dtype, raw.device)
        v0_slow = positive(raw[..., 4])
        v0_fast = positive(raw[..., 9])
        return torch.stack(
            [
                kappa_slow,
                theta_slow,
                sigma_slow,
                rho_slow,
                v0_slow,
                kappa_fast,
                theta_fast,
                sigma_fast,
                rho_fast,
                v0_fast,
            ],
            dim=-1,
        )


class PhysicsInformedInverseCalibrator(nn.Module):
    """Surface-to-parameter network used by the PINN training path."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int] = (512, 512, 256, 256, 128),
        activation: str = "gelu",
        dropout: float = 0.05,
        output_size: int = PARAMETER_COUNT,
    ) -> None:
        super().__init__()
        if input_size <= 0:
            raise ValueError("input_size must be strictly positive")
        if output_size != PARAMETER_COUNT:
            raise ValueError(f"output_size must equal {PARAMETER_COUNT}")
        if not hidden_sizes or any(size <= 0 for size in hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")

        activation_factory = _activation_factory(activation)
        blocks: list[nn.Module] = []
        previous_size = input_size
        for hidden_size in hidden_sizes:
            blocks.append(_FeedForwardBlock(previous_size, hidden_size, activation_factory, dropout))
            previous_size = hidden_size
        self.backbone = nn.Sequential(*blocks)
        self.raw_head = nn.Linear(previous_size, output_size)
        self.constraint_map = DoubleHestonConstraintMap()
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_size:
            raise ValueError(
                f"Expected features shaped (batch, {self.input_size}), got {tuple(features.shape)}"
            )
        if not torch.is_floating_point(features):
            raise TypeError("features must use a floating-point dtype")
        encoded = self.backbone(features)
        raw_parameters = self.raw_head(encoded)
        return self.constraint_map(raw_parameters)


class _FeedForwardBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation_factory: type[nn.Module],
        dropout: float,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.normalization = nn.LayerNorm(output_size)
        self.activation = activation_factory()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.use_residual = input_size == output_size

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.linear(inputs)
        outputs = self.normalization(outputs)
        outputs = self.activation(outputs)
        outputs = self.dropout(outputs)
        if self.use_residual:
            outputs = outputs + inputs
        return outputs


def _positive(values: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(values) + _POSITIVE_EPS


def _sigma_with_feller_margin(
    kappa: torch.Tensor,
    theta: torch.Tensor,
    raw_sigma: torch.Tensor,
) -> torch.Tensor:
    safe_ceiling = torch.sqrt(2.0 * kappa * theta) * _FELLER_SIGMA_SAFETY
    return safe_ceiling * torch.sigmoid(raw_sigma)


def _correlation_disk_map(
    raw_radius: torch.Tensor,
    raw_angle: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    radius = _CORRELATION_RADIUS_SAFETY * torch.sigmoid(raw_radius)
    angle = math.pi * torch.tanh(raw_angle)
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    rho_slow = radius * cosine
    rho_fast = radius * sine
    return (
        rho_slow.to(device=device, dtype=dtype),
        rho_fast.to(device=device, dtype=dtype),
    )


def _activation_factory(name: str) -> type[nn.Module]:
    factories: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
    }
    try:
        return factories[name.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported activation: {name}") from error


__all__ = [
    "DoubleHestonConstraintMap",
    "PhysicsInformedInverseCalibrator",
]
