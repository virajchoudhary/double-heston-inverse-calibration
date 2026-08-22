"""Local-information (Jacobian) diagnostics for the G2 R2-vs-R3 study.

Conventions are the committed G2 ones:
- Jacobian of spot-normalized prices with respect to full-range-scaled
  parameters ``((theta - lower) / width)``;
- central differences with validity-aware step reduction, relative step 1e-4;
- SVD singular values, condition number, numerical rank, practical rank at
  1e-6 relative tolerance;
- per-parameter normalized sensitivity (column norms) and the weakest
  singular directions (right singular vectors).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..constants import PARAMETER_NAMES
from ..constraints import validate_parameters
from . import frozen
from .calibration import WIDTHS, clean_observables
from .geometry import build_geometry


def scaled_parameter_jacobian(
    truth_vector: np.ndarray,
    slots,
    *,
    spot: float = frozen.SYNTHETIC_SPOT,
    relative_step: float = frozen.JACOBIAN_RELATIVE_STEP,
) -> np.ndarray:
    vector = np.asarray(truth_vector, dtype=np.float64)
    columns: list[np.ndarray] = []
    for index, width in enumerate(WIDTHS):
        step = relative_step * width
        for _ in range(12):
            lower = vector.copy()
            upper = vector.copy()
            lower[index] -= step
            upper[index] += step
            if (
                validate_parameters(lower)["is_valid"]
                and validate_parameters(upper)["is_valid"]
            ):
                break
            step *= 0.5
        else:
            raise RuntimeError(
                f"Could not form a valid central difference for {PARAMETER_NAMES[index]}"
            )
        lower_prices = clean_observables(lower, slots, spot=spot)
        upper_prices = clean_observables(upper, slots, spot=spot)
        columns.append((upper_prices - lower_prices) * width / (2.0 * step))
    return np.column_stack(columns)


def jacobian_summary(jacobian: np.ndarray) -> dict[str, Any]:
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    largest = float(singular_values[0])
    smallest = float(singular_values[-1])
    numerical_tolerance = largest * singular_values.size * np.finfo(np.float64).eps
    return {
        "largest_singular_value": largest,
        "smallest_singular_value": smallest,
        "condition_number": largest / smallest,
        "numerical_rank": int(np.sum(singular_values > numerical_tolerance)),
        "practical_rank": int(
            np.sum(singular_values > frozen.PRACTICAL_RANK_RELATIVE_TOLERANCE * largest)
        ),
        **{
            f"singular_value_{index + 1:02d}": float(value)
            for index, value in enumerate(singular_values)
        },
    }


def weakest_directions(jacobian: np.ndarray, count: int = 2) -> dict[str, Any]:
    """Weakest right-singular directions with per-parameter loadings."""
    _, singular_values, vh = np.linalg.svd(jacobian, full_matrices=False)
    result: dict[str, Any] = {}
    for position in range(1, min(count, vh.shape[0]) + 1):
        direction = vh[-position]
        result[f"weakest_direction_{position}_singular_value"] = float(
            singular_values[-position]
        )
        result.update(
            {
                f"weakest_direction_{position}_loading_{name}": float(value)
                for name, value in zip(PARAMETER_NAMES, direction)
            }
        )
    return result


def parameter_sensitivities(jacobian: np.ndarray) -> dict[str, float]:
    norms = np.linalg.norm(jacobian, axis=0)
    return {
        name: float(value) for name, value in zip(PARAMETER_NAMES, norms)
    }


def full_jacobian_record(
    truth_vector: np.ndarray, slots, *, spot: float = frozen.SYNTHETIC_SPOT
) -> dict[str, Any]:
    jacobian = scaled_parameter_jacobian(truth_vector, slots, spot=spot)
    record: dict[str, Any] = jacobian_summary(jacobian)
    record.update(weakest_directions(jacobian))
    record.update(parameter_sensitivities(jacobian))
    return record
