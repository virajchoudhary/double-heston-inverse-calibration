from __future__ import annotations

import math

import numpy as np
import torch

from dheston.calibration.transforms import PARAMETER_NAMES


def mae(predicted: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.abs(predicted - actual)))


def rmse(predicted: np.ndarray, actual: np.ndarray) -> float:
    return float(math.sqrt(np.mean(np.square(predicted - actual))))


def parameter_summary(predicted: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    predicted_np = predicted.detach().cpu().numpy()
    actual_np = actual.detach().cpu().numpy()
    metrics: dict[str, float] = {}
    for index, name in enumerate(PARAMETER_NAMES):
        metrics[f"{name}_mae"] = mae(predicted_np[:, index], actual_np[:, index])
        metrics[f"{name}_rmse"] = rmse(predicted_np[:, index], actual_np[:, index])
    return metrics

