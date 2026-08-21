"""Parameter-recovery metrics and constraint-validity diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from .constants import PARAMETER_COUNT, PARAMETER_NAMES
from .constraints import validate_parameters
from .utils import write_json


def evaluate_parameter_recovery(
    true_parameters: np.ndarray,
    predicted_parameters: np.ndarray,
    surface_ids: list[str],
    output_directory: str | Path,
) -> dict[str, Any]:
    """Save per-parameter aggregates and every row-level prediction/error."""
    truth = np.asarray(true_parameters, dtype=np.float64)
    predictions = np.asarray(predicted_parameters, dtype=np.float64)
    if truth.shape != predictions.shape or truth.ndim != 2 or truth.shape[1] != PARAMETER_COUNT:
        raise ValueError(f"Expected matching matrices shaped (samples, {PARAMETER_COUNT})")
    if len(surface_ids) != len(truth):
        raise ValueError("surface_ids length does not match parameter rows")
    if not np.isfinite(truth).all():
        raise ValueError("True parameters must all be finite")

    parameter_metrics: dict[str, dict[str, float | None]] = {}
    row_records: list[dict[str, Any]] = []
    for index, surface_id in enumerate(surface_ids):
        record: dict[str, Any] = {"surface_id": surface_id}
        for column, name in enumerate(PARAMETER_NAMES):
            predicted = predictions[index, column]
            true = truth[index, column]
            record[f"true_{name}"] = true
            record[f"predicted_{name}"] = predicted
            record[f"error_{name}"] = predicted - true
            record[f"absolute_error_{name}"] = abs(predicted - true)
        diagnostics = _safe_diagnostics(predictions[index])
        record.update(
            {
                "parameter_vector_valid": diagnostics["is_valid"],
                "ordering_valid": diagnostics["ordering_valid"],
                "slow_feller_valid": diagnostics["slow_feller_gap"] > 0.0,
                "fast_feller_valid": diagnostics["fast_feller_gap"] > 0.0,
                "correlation_disk_valid": diagnostics["correlation_disk_value"] < 1.0,
                "violations": " | ".join(diagnostics["violations"]),
            }
        )
        row_records.append(record)

    for column, name in enumerate(PARAMETER_NAMES):
        true_values = truth[:, column]
        predicted_values = predictions[:, column]
        finite = np.isfinite(predicted_values)
        if not finite.any():
            parameter_metrics[name] = {
                key: None
                for key in (
                    "mae",
                    "rmse",
                    "mean_relative_error",
                    "r2",
                    "bias",
                    "median_absolute_error",
                    "maximum_absolute_error",
                )
            }
            continue
        error = predicted_values[finite] - true_values[finite]
        absolute = np.abs(error)
        denominator = np.maximum(np.abs(true_values[finite]), 1e-12)
        r2 = (
            float(r2_score(true_values[finite], predicted_values[finite]))
            if finite.sum() >= 2 and np.ptp(true_values[finite]) > 0.0
            else None
        )
        parameter_metrics[name] = {
            "mae": float(np.mean(absolute)),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "mean_relative_error": float(np.mean(absolute / denominator)),
            "r2": r2,
            "bias": float(np.mean(error)),
            "median_absolute_error": float(np.median(absolute)),
            "maximum_absolute_error": float(np.max(absolute)),
        }

    rows = pd.DataFrame(row_records)
    aggregate = {
        "sample_count": len(truth),
        "parameter_metrics": parameter_metrics,
        "full_vector_validity_rate": float(rows["parameter_vector_valid"].mean()),
        "factor_order_violation_rate": float((~rows["ordering_valid"]).mean()),
        "slow_feller_violation_rate": float((~rows["slow_feller_valid"]).mean()),
        "fast_feller_violation_rate": float((~rows["fast_feller_valid"]).mean()),
        "correlation_disk_violation_rate": float(
            (~rows["correlation_disk_valid"]).mean()
        ),
    }
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_path / "parameter_predictions_and_errors.csv", index=False)
    pd.DataFrame(parameter_metrics).T.to_csv(
        output_path / "parameter_metrics.csv", index_label="parameter"
    )
    write_json(output_path / "parameter_metrics.json", aggregate)
    return aggregate


def _safe_diagnostics(vector: np.ndarray) -> dict[str, Any]:
    if not np.isfinite(vector).all():
        return {
            "is_valid": False,
            "ordering_valid": False,
            "slow_feller_gap": float("-inf"),
            "fast_feller_gap": float("-inf"),
            "correlation_disk_value": float("inf"),
            "violations": ["predicted parameter vector contains non-finite values"],
        }
    return validate_parameters(vector)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions_json", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.predictions_json.read_text(encoding="utf-8"))
    evaluate_parameter_recovery(
        np.asarray(payload["true_parameters"]),
        np.asarray(payload["predicted_parameters"]),
        [str(value) for value in payload["surface_ids"]],
        args.output_directory,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
