"""Aggregate frozen neural and N=250 traditional robustness evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.r2_primary.dataset import R2PrimaryDataset
from src.r2_primary.evaluation import (
    _range_scaled_error_matrix,
    reprice_normalized,
    train_split_scaling,
)
from src.r2_primary.final_evaluation import _traditional_multi_start_dispersion
from src.r2_primary.calibration import select_representatives
from src.utils import write_json

from .execution import (
    CLEAN_DATASET_PATH,
    EVIDENCE_ROOT,
    assert_clean_dataset_identity,
    load_frozen_protocol,
)
from .neural_evaluation import _headline_row, _run_metrics, safe_level_label

TRADITIONAL_ROOT = EVIDENCE_ROOT / "traditional"
NEURAL_ROOT = EVIDENCE_ROOT / "neural"


def _observed(dataset: R2PrimaryDataset) -> np.ndarray:
    return np.stack(
        [
            np.where(item.mask, item.normalized_prices, np.nan)
            for item in dataset.items
        ]
    )


def evaluate_traditional_levels(
    levels: tuple[float, ...],
) -> dict[str, Any]:
    protocol = load_frozen_protocol()
    assert_clean_dataset_identity()
    clean_dataset = R2PrimaryDataset.from_jsonl(CLEAN_DATASET_PATH, splits={"test"})
    scaling = train_split_scaling(
        R2PrimaryDataset.from_jsonl(CLEAN_DATASET_PATH)
    )
    truth = np.stack([item.targets for item in clean_dataset.items])
    clean_observed = _observed(clean_dataset)
    labels = {
        float(level): label
        for level, label in zip(protocol["noise_levels"], protocol["noise_level_labels"])
    }
    generated: dict[str, str] = {}
    headline_rows: list[dict[str, Any]] = []
    stability: dict[str, Any] = {}
    for level in levels:
        if level not in labels:
            raise ValueError(f"non-frozen level: {level}")
        label = labels[level]
        source_path = (
            CLEAN_DATASET_PATH
            if level == 0.0
            else Path(CLEAN_DATASET_PATH).parents[1]
            / "r2_noise_robustness"
            / "levels"
            / label
            / "noisy_surfaces.jsonl"
        )
        run_dir = TRADITIONAL_ROOT / f"level_{safe_level_label(label)}"
        starts = pd.read_csv(run_dir / "traditional_calibration_starts.csv")
        representatives = select_representatives(starts).sort_values("surface_id")
        derived = R2PrimaryDataset.from_jsonl(source_path, splits={"test"})
        index_by_id = {item.surface_id: index for index, item in enumerate(derived.items)}
        missing = set(representatives["surface_id"]) - set(index_by_id)
        if missing or len(representatives) != 250:
            raise ValueError("traditional representative population mismatch")
        aligned_indices = [index_by_id[surface_id] for surface_id in representatives["surface_id"]]
        aligned_dataset = R2PrimaryDataset([derived.items[index] for index in aligned_indices])
        predicted_columns = [f"predicted_{name}" for name in scaling]
        predicted = representatives[predicted_columns].to_numpy(dtype=np.float64)
        repriced = reprice_normalized(aligned_dataset, list(range(250)), predicted)
        noisy_observed = _observed(aligned_dataset)
        aligned_truth = np.stack([item.targets for item in aligned_dataset.items])
        metrics = _run_metrics(
            truth=aligned_truth,
            predicted=predicted,
            repriced=repriced,
            noisy_observed=noisy_observed,
            clean_observed=clean_observed[aligned_indices],
            scaling=scaling,
        )
        headline = _headline_row(
            "traditional_calibration", "representative", label, metrics
        )
        headline_rows.append(headline)
        parameter_rmse = np.sqrt(
            (_range_scaled_error_matrix(aligned_truth, predicted, scaling) ** 2).mean(axis=1)
        )
        per_surface = pd.DataFrame(
            {
                "method": "traditional_calibration",
                "seed": "representative",
                "surface_id": representatives["surface_id"].to_numpy(),
                "parameter_range_scaled_rmse": parameter_rmse,
                "noisy_price_rmse": metrics["fit_to_noisy_observation"]["per_surface_rmse"],
                "clean_latent_price_rmse": metrics[
                    "clean_latent_repricing"
                ]["per_surface_rmse"],
            }
        )
        per_surface.to_csv(run_dir / "per_surface_metrics.csv", index=False)
        write_json(run_dir / "representative_metrics.json", metrics)
        stability[label] = _traditional_multi_start_dispersion(starts, scaling)
        generated[f"{run_dir.name}/per_surface_metrics.csv"] = str(
            (run_dir / "per_surface_metrics.csv").resolve()
        )
        generated[f"{run_dir.name}/representative_metrics.json"] = str(
            (run_dir / "representative_metrics.json").resolve()
        )
    frame = pd.DataFrame(headline_rows)
    frame.to_csv(TRADITIONAL_ROOT / "all_traditional_headline.csv", index=False)
    write_json(TRADITIONAL_ROOT / "multi_start_stability.json", stability)
    return {"headline_rows": headline_rows, "stability": stability}


def paired_degradation() -> dict[str, Any]:
    protocol = load_frozen_protocol()
    labels = [str(label) for label in protocol["noise_level_labels"]]
    neural = pd.read_csv(NEURAL_ROOT / "all_neural_seed_headline.csv")
    traditional = pd.read_csv(TRADITIONAL_ROOT / "all_traditional_headline.csv")
    all_rows = pd.concat([neural, traditional], ignore_index=True)
    metric_names = [
        "range_scaled_parameter_rmse",
        "standardized_parameter_rmse",
        "noisy_price_rmse_mean",
        "clean_latent_price_rmse_mean",
    ]
    output_rows: list[dict[str, Any]] = []
    for key, group in all_rows.groupby(["method", "seed"], sort=True):
        base = group[group["noise_level_label"] == labels[0]].iloc[0]
        for label in labels[1:]:
            current = group[group["noise_level_label"] == label].iloc[0]
            row: dict[str, Any] = {
                "method": key[0],
                "seed": key[1],
                "noise_level_label": label,
            }
            for metric in metric_names:
                baseline_value = float(base[metric])
                current_value = float(current[metric])
                row[f"{metric}_delta"] = current_value - baseline_value
                row[f"{metric}_ratio"] = (
                    current_value / baseline_value if baseline_value != 0.0 else None
                )
            output_rows.append(row)

    conditioned_rows: list[dict[str, Any]] = []
    repricing_tolerances = (1e-4, 1e-3)
    parameter_tolerances = (0.10, 0.25)
    for method_seed, _ in pd.concat(
        [
            neural[["method", "seed"]],
            traditional[["method", "seed"]],
        ],
        ignore_index=True,
    ).groupby(["method", "seed"]):
        method, seed = method_seed
        frames: dict[str, pd.DataFrame] = {}
        for label in labels:
            if method == "traditional_calibration":
                path = (
                    TRADITIONAL_ROOT
                    / f"level_{safe_level_label(label)}"
                    / "per_surface_metrics.csv"
                )
            else:
                path = (
                    NEURAL_ROOT
                    / f"level_{safe_level_label(label)}"
                    / "per_surface_metrics.csv"
                )
            frame = pd.read_csv(path)
            frames[label] = frame[
                (frame["method"] == method) & (frame["seed"].astype(str) == str(seed))
            ].set_index("surface_id")
        for label in labels[1:]:
            baseline = frames[labels[0]]
            current = frames[label]
            for repricing_kind in ("noisy_price_rmse", "clean_latent_price_rmse"):
                for repricing_tolerance in repricing_tolerances:
                    stays_acceptable = (
                        baseline[repricing_kind] <= repricing_tolerance
                    ) & (current[repricing_kind] <= repricing_tolerance)
                    for parameter_tolerance in parameter_tolerances:
                        crosses = (
                            baseline["parameter_range_scaled_rmse"]
                            <= parameter_tolerance
                        ) & (
                            current["parameter_range_scaled_rmse"]
                            > parameter_tolerance
                        )
                        conditioned_rows.append(
                            {
                                "method": method,
                                "seed": str(seed),
                                "noise_level_label": label,
                                "repricing_quantity": repricing_kind,
                                "repricing_tolerance": repricing_tolerance,
                                "parameter_tolerance": parameter_tolerance,
                                "fraction_repricing_acceptable_and_parameter_crosses": float(
                                    (stays_acceptable & crosses).mean()
                                ),
                            }
                        )
    degradation_frame = pd.DataFrame(output_rows)
    degradation_frame.to_csv(EVIDENCE_ROOT / "paired_degradation.csv", index=False)
    conditioned_frame = pd.DataFrame(conditioned_rows)
    conditioned_frame.to_csv(
        EVIDENCE_ROOT / "identifiability_conditioned_degradation.csv", index=False
    )
    return {
        "paired_degradation": degradation_frame.to_dict(orient="records"),
        "conditioned_degradation": conditioned_frame.to_dict(orient="records"),
    }
