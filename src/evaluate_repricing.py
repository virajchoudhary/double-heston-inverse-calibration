"""Reprice target grids using ANN-predicted Double Heston parameters."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import PARAMETER_NAMES
from .pricing_interface import price_double_heston_surface
from .utils import write_json


def evaluate_repricing(
    surfaces: pd.DataFrame,
    predicted_parameters: dict[str, np.ndarray],
    output_directory: str | Path,
) -> dict[str, Any]:
    """Reprice every surface, retaining both pricing successes and failure rows."""
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for surface_id, group in surfaces.groupby("surface_id", sort=True):
        try:
            parameters = predicted_parameters[str(surface_id)]
            repriced = price_double_heston_surface(
                float(group["spot"].iloc[0]),
                group["strike"].to_numpy(dtype=float),
                group["maturity_years"].to_numpy(dtype=float),
                float(group["risk_free_rate"].iloc[0]),
                float(group["dividend_yield"].iloc[0]),
                group["option_type"].astype(str).tolist(),
                parameters,
            )
            target_column = (
                "generated_price" if "generated_price" in group else "observed_price"
            )
            target = group[target_column].to_numpy(dtype=float)
            for row_index, (actual, predicted) in enumerate(
                zip(target, repriced, strict=True)
            ):
                records.append(
                    {
                        "surface_id": surface_id,
                        "row_index": row_index,
                        "target_price": actual,
                        "reconstructed_price": predicted,
                        "absolute_error": abs(predicted - actual),
                        "relative_error": abs(predicted - actual) / max(abs(actual), 1e-12),
                    }
                )
        except Exception as error:  # retain every failure; never impute
            failures.append(
                {
                    "surface_id": surface_id,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
    record_frame = pd.DataFrame(records)
    failure_frame = pd.DataFrame(failures)
    record_frame.to_csv(output_path / "repricing_rows.csv", index=False)
    failure_frame.to_csv(output_path / "repricing_failures.csv", index=False)
    if record_frame.empty:
        summary = {
            "status": "failed_no_successful_repricing_rows",
            "successful_rows": 0,
            "failure_rows": len(failure_frame),
        }
    else:
        errors = record_frame["reconstructed_price"] - record_frame["target_price"]
        summary = {
            "status": "completed_with_failures" if len(failure_frame) else "completed",
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "mae": float(np.mean(np.abs(errors))),
            "mean_relative_error": float(record_frame["relative_error"].mean()),
            "successful_rows": len(record_frame),
            "failure_rows": len(failure_frame),
        }
    write_json(output_path / "repricing_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("surfaces_csv", type=Path, nargs="?")
    parser.add_argument("predictions_csv", type=Path, nargs="?")
    parser.add_argument("output_directory", type=Path, nargs="?", default=Path("outputs/metrics/repricing"))
    args = parser.parse_args()
    if args.surfaces_csv is None or args.predictions_csv is None:
        validation_directory = Path("outputs/double_heston_validation")
        surfaces_path = validation_directory / "clean_surface.csv"
        starts_path = validation_directory / "clean_recovery_starts.csv"
        if not surfaces_path.exists() or not starts_path.exists():
            print(
                "Canonical pricing engine is available. Run "
                "python -m src.run_double_heston_validation before the default "
                "controlled repricing check, or provide explicit CSV paths."
            )
            return 0
        surfaces = pd.read_csv(surfaces_path)
        starts = pd.read_csv(starts_path)
        best = starts.loc[starts["loss"].idxmin()]
        surface_id = str(surfaces["surface_id"].iloc[0])
        predicted = {
            surface_id: np.asarray(
                [best[f"predicted_{name}"] for name in PARAMETER_NAMES],
                dtype=np.float64,
            )
        }
        summary = evaluate_repricing(
            surfaces,
            predicted,
            validation_directory / "repricing_self_check",
        )
        summary["evaluation_scope"] = (
            "controlled canonical calibration output; not an ANN research result"
        )
        write_json(
            validation_directory / "repricing_self_check/repricing_summary.json",
            summary,
        )
        print(summary)
        return 0
    surfaces = pd.read_csv(args.surfaces_csv)
    predictions = pd.read_csv(args.predictions_csv).set_index("surface_id")
    parameter_columns = [f"predicted_{name}" for name in PARAMETER_NAMES]
    predicted = {
        str(index): row[parameter_columns].to_numpy(dtype=float)
        for index, row in predictions.iterrows()
    }
    summary = evaluate_repricing(surfaces, predicted, args.output_directory)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
