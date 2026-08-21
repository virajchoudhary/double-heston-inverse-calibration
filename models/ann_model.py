"""Configurable non-physics MLP for ten-parameter inverse calibration."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from src.constants import PARAMETER_COUNT


class ANNInverseCalibrator(nn.Module):
    """Ordinary feed-forward ANN; intentionally contains no PDE residual."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int] = (512, 256, 128, 64),
        activation: str = "relu",
        dropout: float = 0.10,
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
        layers: list[nn.Module] = []
        previous_size = input_size
        for index, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(activation_factory())
            if dropout > 0.0 and index < len(hidden_sizes) - 1:
                layers.append(nn.Dropout(dropout))
            previous_size = hidden_size
        layers.append(nn.Linear(previous_size, output_size))
        self.network = nn.Sequential(*layers)
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, input_size)`` surfaces to ``(batch, 10)`` outputs."""
        if features.ndim != 2 or features.shape[1] != self.input_size:
            raise ValueError(
                f"Expected features shaped (batch, {self.input_size}), "
                f"got {tuple(features.shape)}"
            )
        if not torch.is_floating_point(features):
            raise TypeError("features must use a floating-point dtype")
        return self.network(features)


def _activation_factory(name: str) -> type[nn.Module]:
    factories: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
    }
    try:
        return factories[name.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported activation: {name}") from error
