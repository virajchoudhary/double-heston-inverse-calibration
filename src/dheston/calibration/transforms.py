from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


PARAMETER_NAMES = [
    "v01",
    "kappa1",
    "theta1",
    "sigma1",
    "rho1",
    "v02",
    "kappa2",
    "theta2",
    "sigma2",
    "rho2",
]

PARAMETER_BOUNDS = {
    "v01": (0.01, 0.60),
    "kappa1": (0.30, 8.00),
    "theta1": (0.01, 0.60),
    "sigma1": (0.05, 1.50),
    "rho1": (-0.95, -0.05),
    "v02": (0.01, 0.60),
    "kappa2": (0.10, 6.00),
    "theta2": (0.01, 0.60),
    "sigma2": (0.05, 1.50),
    "rho2": (-0.95, -0.05),
}


def parameter_bounds_array() -> np.ndarray:
    return np.asarray([PARAMETER_BOUNDS[name] for name in PARAMETER_NAMES], dtype=np.float64)


def dict_to_array(parameters: dict[str, float]) -> np.ndarray:
    return np.asarray([parameters[name] for name in PARAMETER_NAMES], dtype=np.float64)


def array_to_dict(parameters: Iterable[float]) -> dict[str, float]:
    values = list(parameters)
    return {name: float(value) for name, value in zip(PARAMETER_NAMES, values, strict=True)}


def sample_parameter_vector(rng: np.random.Generator) -> np.ndarray:
    bounds = PARAMETER_BOUNDS
    v01 = rng.uniform(*bounds["v01"])
    kappa1 = rng.uniform(*bounds["kappa1"])
    theta1 = rng.uniform(*bounds["theta1"])
    sigma1 = rng.uniform(*bounds["sigma1"])
    rho1 = rng.uniform(*bounds["rho1"])
    v02 = rng.uniform(*bounds["v02"])
    kappa2 = rng.uniform(bounds["kappa2"][0], min(bounds["kappa2"][1], kappa1))
    theta2 = rng.uniform(*bounds["theta2"])
    sigma2 = rng.uniform(*bounds["sigma2"])
    rho2 = rng.uniform(*bounds["rho2"])
    return np.asarray([v01, kappa1, theta1, sigma1, rho1, v02, kappa2, theta2, sigma2, rho2], dtype=np.float64)


def scale_parameters_to_unit(parameters: np.ndarray) -> np.ndarray:
    bounds = parameter_bounds_array()
    lows = bounds[:, 0]
    highs = bounds[:, 1]
    return (parameters - lows) / (highs - lows)


def scale_parameters_to_unit_torch(parameters: torch.Tensor) -> torch.Tensor:
    bounds = torch.as_tensor(parameter_bounds_array(), dtype=parameters.dtype, device=parameters.device)
    lows = bounds[:, 0]
    highs = bounds[:, 1]
    return (parameters - lows) / (highs - lows)


def constrain_parameter_tensor(raw: torch.Tensor) -> torch.Tensor:
    def to_interval(values: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
        return lower + (upper - lower) * torch.sigmoid(values)

    bounds = PARAMETER_BOUNDS
    v01 = to_interval(raw[..., 0], *bounds["v01"])
    kappa1 = to_interval(raw[..., 1], *bounds["kappa1"])
    theta1 = to_interval(raw[..., 2], *bounds["theta1"])
    sigma1 = to_interval(raw[..., 3], *bounds["sigma1"])
    rho1 = to_interval(raw[..., 4], *bounds["rho1"])
    v02 = to_interval(raw[..., 5], *bounds["v02"])
    kappa2_low = bounds["kappa2"][0]
    kappa2 = kappa2_low + (kappa1 - kappa2_low) * torch.sigmoid(raw[..., 6])
    theta2 = to_interval(raw[..., 7], *bounds["theta2"])
    sigma2 = to_interval(raw[..., 8], *bounds["sigma2"])
    rho2 = to_interval(raw[..., 9], *bounds["rho2"])
    return torch.stack([v01, kappa1, theta1, sigma1, rho1, v02, kappa2, theta2, sigma2, rho2], dim=-1)

