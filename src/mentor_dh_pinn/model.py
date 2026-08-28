"""Conditional forward network for the mentor Double Heston baseline."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .config import DomainConfig

FEATURE_NAMES = ("spot", "variance_slow", "variance_fast", "tau", "strike", "rate", "carry")
FEATURE_COUNT = len(FEATURE_NAMES)


class InputNormalizer(nn.Module):
    """Affine map from raw contract features to the network's [-1, 1] box."""

    def __init__(
        self,
        feature_min: Sequence[float] | None = None,
        feature_max: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        minimum = tuple(DomainConfig().feature_min if feature_min is None else feature_min)
        maximum = tuple(DomainConfig().feature_max if feature_max is None else feature_max)
        if len(minimum) != FEATURE_COUNT or len(maximum) != FEATURE_COUNT:
            raise ValueError(f"feature bounds must each contain {FEATURE_COUNT} values")
        if any(not float(lo) < float(hi) for lo, hi in zip(minimum, maximum, strict=True)):
            raise ValueError("every feature bound must be strictly increasing")
        self.register_buffer("feature_min", torch.tensor(minimum, dtype=torch.float64))
        self.register_buffer("feature_max", torch.tensor(maximum, dtype=torch.float64))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != FEATURE_COUNT:
            raise ValueError(f"features must have shape (batch, {FEATURE_COUNT})")
        if not features.is_floating_point():
            raise TypeError("features must use a floating-point dtype")
        values = features.to(dtype=torch.float64)
        if not torch.isfinite(values).all():
            raise ValueError("features must be finite")
        denominator = self.feature_max - self.feature_min
        return 2.0 * (values - self.feature_min) / denominator - 1.0


class DoubleHestonForwardPINN(nn.Module):
    """7-input, five-hidden-layer, raw-call-price forward PINN.

    The returned scalar is intentionally an unconstrained raw price.  This is
    important for the PDE: derivatives are taken from the raw price, not from
    a normalized target or a post-hoc bounded representation.
    """

    def __init__(
        self,
        *,
        feature_min: Sequence[float] | None = None,
        feature_max: Sequence[float] | None = None,
        hidden_layers: int = 5,
        hidden_width: int = 128,
    ) -> None:
        super().__init__()
        if hidden_layers != 5 or hidden_width != 128:
            raise ValueError("V1 architecture is fixed at five hidden layers of width 128")
        self.normalizer = InputNormalizer(feature_min, feature_max)
        layers: list[nn.Module] = []
        previous = FEATURE_COUNT
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(previous, hidden_width), nn.Tanh()))
            previous = hidden_width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)
        self.double()

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def normalized_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.normalizer(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = self.normalized_features(features)
        # No output normalization, sigmoid, or no-arbitrage projection: the
        # PDE must see derivatives of this raw call-price field.
        return self.network(normalized)

    def predict_price(self, features: torch.Tensor) -> torch.Tensor:
        """Return one raw price per row as a flat tensor."""
        return self.forward(features).reshape(-1)


# A descriptive alias keeps imports readable in scripts and notebooks.
ForwardPINN = DoubleHestonForwardPINN

__all__ = [
    "DoubleHestonForwardPINN",
    "FEATURE_COUNT",
    "FEATURE_NAMES",
    "ForwardPINN",
    "InputNormalizer",
]
