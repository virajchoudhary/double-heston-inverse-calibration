"""Frozen-metric evaluation for the R2 primary comparison.

Implements the metric families frozen in
``configs/r2_primary_comparison_FINAL.yaml`` section METRICS: parameter
recovery, constraint validity, repricing (production pricer), identifiability
awareness, stability, and runtime.  All three methods are evaluated on the
same untouched test split with the same evaluators.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..constants import PARAMETER_INDICES, PARAMETER_NAMES
from ..constraints import validate_parameters
from ..double_heston import price_double_heston_surface
from ..utils import write_json
from .dataset import R2PrimaryDataset

REPRICING_TOLERANCES = (1.0e-4, 1.0e-3)
PARAMETER_TOLERANCES = (0.10, 0.25)
_LOG2 = float(np.log(2.0))


def reprice_normalized(
    dataset: R2PrimaryDataset,
    indices: list[int],
    predicted_parameters: np.ndarray,
    *,
    node_count: int = 64,
) -> np.ndarray:
    """Reprice predicted parameters through the production pricer.

    Returns spot-normalized prices aligned with the canonical slots
    (masked slots are set to NaN so they can never enter metrics).
    """
    if predicted_parameters.shape != (len(indices), 10):
        raise ValueError("predicted parameters must align with indices")
    output = np.full((len(indices), 20), np.nan, dtype=np.float64)
    for row, index in enumerate(indices):
        item = dataset.items[index]
        dollar = price_double_heston_surface(
            item.spot,
            item.strikes,
            item.maturities,
            item.rate,
            item.carry,
            item.option_types,
            predicted_parameters[row],
            node_count=node_count,
        )
        normalized = np.asarray(dollar, dtype=np.float64) / item.spot
        normalized = np.where(item.mask, normalized, np.nan)
        output[row] = normalized
    return output


def train_split_scaling(dataset: R2PrimaryDataset) -> dict[str, dict[str, float]]:
    """Train-split per-parameter ranges, means, and standard deviations."""
    train_targets = np.stack(
        [dataset.items[index].targets for index in dataset.indices_for_split("train")]
    )
    scaling: dict[str, dict[str, float]] = {}
    for column, name in enumerate(PARAMETER_NAMES):
        values = train_targets[:, column]
        scaling[name] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "range": float(values.max() - values.min()),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
        }
    return scaling


def parameter_recovery_metrics(
    truth: np.ndarray,
    predicted: np.ndarray,
    scaling: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Frozen parameter-recovery family (truth known: synthetic test)."""
    truth = np.asarray(truth, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if truth.shape != predicted.shape or truth.shape[1] != 10:
        raise ValueError("truth/predicted must be aligned (N, 10)")
    errors = predicted - truth
    metrics: dict[str, Any] = {
        "per_parameter": {},
        "aggregate": {},
        "factorwise": {},
    }
    range_scaled_columns: list[np.ndarray] = []
    standardized_squared: list[np.ndarray] = []
    for column, name in enumerate(PARAMETER_NAMES):
        absolute = np.abs(errors[:, column])
        entry = scaling[name]
        range_scaled = absolute / entry["range"]
        range_scaled_columns.append(range_scaled)
        standardized = errors[:, column] / entry["std"]
        standardized_squared.append(standardized**2)
        metrics["per_parameter"][name] = {
            "mae": float(absolute.mean()),
            "median_ae": float(np.median(absolute)),
            "rmse": float(np.sqrt((errors[:, column] ** 2).mean())),
            "mean_range_scaled_ae": float(range_scaled.mean()),
            "median_range_scaled_ae": float(np.median(range_scaled)),
            "bias": float(errors[:, column].mean()),
        }
    range_scaled_matrix = np.stack(range_scaled_columns, axis=1)
    metrics["aggregate"] = {
        "range_scaled_parameter_rmse": float(
            np.sqrt((range_scaled_matrix**2).mean())
        ),
        "standardized_parameter_rmse": float(
            np.sqrt(np.concatenate(standardized_squared).mean())
        ),
        "mean_range_scaled_ae": float(range_scaled_matrix.mean()),
    }
    v0_total_error = np.abs(
        (predicted[:, 4] + predicted[:, 9]) - (truth[:, 4] + truth[:, 9])
    )
    theta_total_error = np.abs(
        (predicted[:, 1] + predicted[:, 6]) - (truth[:, 1] + truth[:, 6])
    )
    half_life_slow = _LOG2 / predicted[:, 0] - _LOG2 / truth[:, 0]
    half_life_fast = _LOG2 / predicted[:, 5] - _LOG2 / truth[:, 5]
    metrics["factorwise"] = {
        "v0_total_mae": float(v0_total_error.mean()),
        "theta_total_mae": float(theta_total_error.mean()),
        "half_life_slow_mae_years": float(np.abs(half_life_slow).mean()),
        "half_life_fast_mae_years": float(np.abs(half_life_fast).mean()),
        "half_life_slow_bias_years": float(half_life_slow.mean()),
        "half_life_fast_bias_years": float(half_life_fast.mean()),
        "note": "never add kappa/sigma/rho across factors",
    }
    metrics["factorwise"]["factor_swap_confusion_rate"] = float(
        _factor_swap_confusion_rate(truth, predicted, scaling)
    )
    return metrics


def _factor_swap_confusion_rate(
    truth: np.ndarray, predicted: np.ndarray, scaling: dict[str, dict[str, float]]
) -> float:
    """Fraction of surfaces closer (standardized) to the factor-swapped truth."""
    swapped = truth.copy()
    swapped[:, [0, 1, 2, 3, 4]] = truth[:, [5, 6, 7, 8, 9]]
    swapped[:, [5, 6, 7, 8, 9]] = truth[:, [0, 1, 2, 3, 4]]
    std = np.array([scaling[name]["std"] for name in PARAMETER_NAMES])
    distance_truth = np.sqrt((((predicted - truth) / std) ** 2).mean(axis=1))
    distance_swapped = np.sqrt((((predicted - swapped) / std) ** 2).mean(axis=1))
    return float((distance_swapped < distance_truth).mean())


def constraint_validity_metrics(predicted: np.ndarray) -> dict[str, Any]:
    """Frozen structural-validity family on predicted physical parameters."""
    predicted = np.asarray(predicted, dtype=np.float64)
    diagnostics = [validate_parameters(row) for row in predicted]
    is_valid = np.array([item["is_valid"] for item in diagnostics])
    return {
        "constraint_validity_rate": float(is_valid.mean()),
        "positivity_violation_rate": float(
            1.0 - np.mean([item["positive_valid"] for item in diagnostics])
        ),
        "ordering_violation_rate": float(
            1.0 - np.mean([item["ordering_valid"] for item in diagnostics])
        ),
        "slow_feller_violation_rate": float(
            np.mean([item["slow_feller_gap"] <= 0.0 for item in diagnostics])
        ),
        "fast_feller_violation_rate": float(
            np.mean([item["fast_feller_gap"] <= 0.0 for item in diagnostics])
        ),
        "correlation_disk_violation_rate": float(
            np.mean([item["correlation_disk_value"] >= 1.0 for item in diagnostics])
        ),
    }


def repricing_metrics(
    observed_normalized: np.ndarray,
    repriced_normalized: np.ndarray,
) -> dict[str, Any]:
    """Frozen repricing family on spot-normalized prices (NaN slots excluded)."""
    observed = np.asarray(observed_normalized, dtype=np.float64)
    repriced = np.asarray(repriced_normalized, dtype=np.float64)
    if observed.shape != repriced.shape:
        raise ValueError("observed and repriced must align")
    valid = np.isfinite(observed) & np.isfinite(repriced)
    errors = np.where(valid, repriced - observed, np.nan)
    per_surface_rmse = np.sqrt(np.nanmean(errors**2, axis=1))
    per_surface_mae = np.nanmean(np.abs(errors), axis=1)
    per_surface_max = np.nanmax(np.abs(errors), axis=1)
    return {
        "normalized_price_rmse_mean": float(per_surface_rmse.mean()),
        "normalized_price_rmse_median": float(np.median(per_surface_rmse)),
        "normalized_price_rmse_p95": float(np.percentile(per_surface_rmse, 95)),
        "normalized_price_mae_mean": float(per_surface_mae.mean()),
        "normalized_price_max_abs_error_mean": float(per_surface_max.mean()),
        "normalized_price_max_abs_error_p95": float(
            np.percentile(per_surface_max, 95)
        ),
        "implied_volatility_error": "NOT_REPORTED_numerical_validity_not_established",
        "per_surface_rmse": per_surface_rmse,
    }


def identifiability_aware_metrics(
    observed_normalized: np.ndarray,
    repriced_normalized: np.ndarray,
    truth: np.ndarray,
    predicted: np.ndarray,
    scaling: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Tolerance/equivalence-conditioned recovery (never equate the two)."""
    per_surface_rmse = repricing_metrics(
        observed_normalized, repriced_normalized
    )["per_surface_rmse"]
    range_scaled = _range_scaled_error_matrix(truth, predicted, scaling)
    range_scaled_rmse = np.sqrt((range_scaled**2).mean(axis=1))
    result: dict[str, Any] = {
        "repricing_tolerance_success": {},
        "parameter_tolerance_success": {},
        "conditioned_recovery": {},
    }
    for tolerance in REPRICING_TOLERANCES:
        meets = per_surface_rmse <= tolerance
        result["repricing_tolerance_success"][f"rmse<={tolerance:g}"] = {
            "rate": float(meets.mean()),
        }
        for parameter_tolerance in PARAMETER_TOLERANCES:
            recovers = range_scaled_rmse <= parameter_tolerance
            result["conditioned_recovery"][
                f"repricing<={tolerance:g}_and_parameter<={parameter_tolerance:g}"
            ] = float((meets & recovers).mean())
            result["conditioned_recovery"][
                f"parameter_recovery_given_repricing<={tolerance:g}_param<={parameter_tolerance:g}"
            ] = float(recovers[meets].mean()) if meets.any() else None
    for tolerance in PARAMETER_TOLERANCES:
        result["parameter_tolerance_success"][f"range_scaled_rmse<={tolerance:g}"] = {
            "rate": float((range_scaled_rmse <= tolerance).mean())
        }
    result["warning"] = (
        "repricing success is NOT parameter recovery: near-equivalent repricing "
        "with materially different parameters is the documented practical "
        "non-identifiability finding"
    )
    return result


def _range_scaled_error_matrix(
    truth: np.ndarray, predicted: np.ndarray, scaling: dict[str, dict[str, float]]
) -> np.ndarray:
    ranges = np.array([scaling[name]["range"] for name in PARAMETER_NAMES])
    return np.abs(predicted - truth) / ranges


def stability_metrics(
    per_run_predictions: dict[str, np.ndarray],
    per_run_headline: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Cross-seed stability for a neural method (all seeds retained)."""
    seeds = sorted(per_run_predictions)
    if len(seeds) < 2:
        return {"note": "single run; cross-seed dispersion undefined"}
    stacked = np.stack([per_run_predictions[seed] for seed in seeds], axis=0)
    per_surface_std = stacked.std(axis=0, ddof=1)  # (N, 10)
    dispersion = {
        name: float(per_surface_std[:, column].mean())
        for column, name in enumerate(PARAMETER_NAMES)
    }
    headline_std: dict[str, float] = {}
    first = per_run_headline[seeds[0]]
    for key in first:
        values = [per_run_headline[seed][key] for seed in seeds if key in per_run_headline[seed]]
        if all(isinstance(value, (int, float)) for value in values):
            headline_std[key] = float(np.std(values, ddof=1))
    return {
        "seeds": [int(seed) for seed in seeds],
        "mean_per_surface_cross_seed_prediction_std": dispersion,
        "mean_overall_cross_seed_prediction_std": float(per_surface_std.mean()),
        "headline_metric_cross_seed_std": headline_std,
    }


def measure_inference_runtime(
    model: Any,
    dataset: R2PrimaryDataset,
    indices: list[int],
    *,
    standardizer: Any = None,
    repetitions: int = 3,
) -> dict[str, float]:
    """Amortized per-surface inference milliseconds over the full split."""
    import torch

    model.eval()
    features = torch.as_tensor(
        np.stack([dataset.items[index].features for index in indices])
    )
    timings: list[float] = []
    with torch.no_grad():
        for _ in range(repetitions):
            started = time.perf_counter()
            output = model(features)
            if standardizer is not None:
                output = standardizer.inverse_transform(output)
            _ = output.sum().item()
            timings.append(time.perf_counter() - started)
    best = min(timings)
    return {
        "per_surface_inference_ms_amortized": 1000.0 * best / len(indices),
        "full_split_inference_seconds": best,
        "repetitions": repetitions,
    }


def summarize_run(
    dataset: R2PrimaryDataset,
    indices: list[int],
    predicted_parameters: np.ndarray,
    scaling: dict[str, dict[str, float]],
    *,
    method_label: str,
    runtime: dict[str, Any] | None = None,
    per_surface_repricing_rmse: np.ndarray | None = None,
) -> dict[str, Any]:
    """One method's full frozen-metric summary on the given surfaces."""
    truth = np.stack([dataset.items[index].targets for index in indices])
    observed = np.stack(
        [
            np.where(
                dataset.items[index].mask,
                dataset.items[index].normalized_prices,
                np.nan,
            )
            for index in indices
        ]
    )
    repriced = reprice_normalized(dataset, indices, predicted_parameters)
    recovery = parameter_recovery_metrics(truth, predicted_parameters, scaling)
    validity = constraint_validity_metrics(predicted_parameters)
    repricing = repricing_metrics(observed, repriced)
    identifiability = identifiability_aware_metrics(
        observed, repriced, truth, predicted_parameters, scaling
    )
    per_surface_rmse = repricing.pop("per_surface_rmse")
    summary: dict[str, Any] = {
        "method": method_label,
        "surfaces": len(indices),
        "parameter_recovery": recovery,
        "constraint_validity": validity,
        "repricing": repricing,
        "identifiability_aware": identifiability,
    }
    if runtime is not None:
        summary["runtime"] = runtime
    return summary
