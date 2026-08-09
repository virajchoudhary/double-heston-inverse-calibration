"""Bounded and standardized parameter transformations for ANN outputs."""

from __future__ import annotations

import torch
from torch import nn

from src.constants import PARAMETER_COUNT


class BoundedParameterTransform(nn.Module):
    """Map unconstrained outputs into confirmed element-wise finite bounds."""

    def __init__(self, lower: torch.Tensor, upper: torch.Tensor) -> None:
        super().__init__()
        lower_tensor, upper_tensor = _validated_bounds(lower, upper)
        self.register_buffer("lower", lower_tensor)
        self.register_buffer("upper", upper_tensor)

    def forward(self, unconstrained: torch.Tensor) -> torch.Tensor:
        _validate_last_dimension(unconstrained, "unconstrained")
        if not torch.isfinite(unconstrained).all():
            raise ValueError("unconstrained values must be finite")
        return self.lower + (self.upper - self.lower) * torch.sigmoid(unconstrained)

    def inverse(self, bounded: torch.Tensor) -> torch.Tensor:
        """Invert the transform only for values strictly inside the bounds."""
        _validate_last_dimension(bounded, "bounded")
        if not torch.isfinite(bounded).all():
            raise ValueError("bounded values must be finite")
        if torch.any(bounded <= self.lower) or torch.any(bounded >= self.upper):
            raise ValueError("inverse requires values strictly inside every bound")
        ratio = (bounded - self.lower) / (self.upper - self.lower)
        return torch.log(ratio) - torch.log1p(-ratio)


class TargetStandardizer:
    """Training-only target standardization when final financial bounds are absent."""

    def __init__(self) -> None:
        self.mean: torch.Tensor | None = None
        self.scale: torch.Tensor | None = None

    def fit(self, training_targets: torch.Tensor) -> "TargetStandardizer":
        _validate_target_matrix(training_targets, "training_targets")
        if training_targets.shape[0] < 2:
            raise ValueError("At least two training targets are required")
        mean = training_targets.mean(dim=0)
        scale = training_targets.std(dim=0, unbiased=False)
        if torch.any(scale <= 0.0):
            raise ValueError("Every target parameter must vary in the training data")
        self.mean = mean.detach().clone()
        self.scale = scale.detach().clone()
        return self

    def transform(self, targets: torch.Tensor) -> torch.Tensor:
        _validate_target_matrix(targets, "targets")
        mean, scale = self._fitted_values(targets)
        return (targets - mean) / scale

    def inverse_transform(self, standardized: torch.Tensor) -> torch.Tensor:
        _validate_target_matrix(standardized, "standardized")
        mean, scale = self._fitted_values(standardized)
        return standardized * scale + mean

    def state_dict(self) -> dict[str, torch.Tensor]:
        mean, scale = self._fitted_values(None)
        return {"mean": mean.detach().cpu(), "scale": scale.detach().cpu()}

    def _fitted_values(
        self, reference: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mean is None or self.scale is None:
            raise RuntimeError("TargetStandardizer has not been fitted")
        if reference is None:
            return self.mean, self.scale
        return (
            self.mean.to(device=reference.device, dtype=reference.dtype),
            self.scale.to(device=reference.device, dtype=reference.dtype),
        )


def _validated_bounds(
    lower: torch.Tensor, upper: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    lower_tensor = torch.as_tensor(lower, dtype=torch.float32)
    upper_tensor = torch.as_tensor(upper, dtype=torch.float32)
    if lower_tensor.shape != (PARAMETER_COUNT,) or upper_tensor.shape != (
        PARAMETER_COUNT,
    ):
        raise ValueError(f"Bounds must each have shape ({PARAMETER_COUNT},)")
    if not torch.isfinite(lower_tensor).all() or not torch.isfinite(upper_tensor).all():
        raise ValueError("Bounds must be finite; unconfirmed null bounds cannot be used")
    if torch.any(lower_tensor >= upper_tensor):
        raise ValueError("Every lower bound must be strictly below its upper bound")
    return lower_tensor, upper_tensor


def _validate_last_dimension(values: torch.Tensor, name: str) -> None:
    if values.ndim < 1 or values.shape[-1] != PARAMETER_COUNT:
        raise ValueError(f"{name} must have final dimension {PARAMETER_COUNT}")


def _validate_target_matrix(values: torch.Tensor, name: str) -> None:
    if values.ndim != 2 or values.shape[1] != PARAMETER_COUNT:
        raise ValueError(f"{name} must have shape (samples, {PARAMETER_COUNT})")
    if not torch.is_floating_point(values) or not torch.isfinite(values).all():
        raise ValueError(f"{name} must contain finite floating-point values")
