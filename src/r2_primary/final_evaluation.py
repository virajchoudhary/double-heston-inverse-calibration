"""Final frozen evaluation driver for the R2 primary comparison.

Runs AFTER all neural seeds are trained and the frozen traditional calibration
has executed on the test split.  Loads every best-validation checkpoint,
evaluates all methods on the untouched 1,250-surface test split with the
frozen metric families, and writes the evidence bundle + unified comparison
table under ``evidence/r2_primary_comparison_20260823/``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import PARAMETER_NAMES
from ..utils import write_json
from .calibration import select_representatives
from .dataset import R2PrimaryDataset, assert_split_isolation
from .evaluation import (
    constraint_validity_metrics,
    measure_inference_runtime,
    parameter_recovery_metrics,
    repricing_metrics,
    reprice_normalized,
    stability_metrics,
    summarize_run,
    train_split_scaling,
)
from .training import load_run

EVIDENCE_ROOT = Path("evidence/r2_primary_comparison_20260823")
CHECKPOINT_ROOT = Path("checkpoints/r2_primary_comparison")
DATASET_PATH = Path("data/final_r2_clean_10000/surfaces.jsonl")
PROTOCOL_CONFIG_PATH = Path("configs/r2_primary_comparison_FINAL.yaml")
NEURAL_SEEDS = (11, 22, 33)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _predicted_matrix_from_traditional(representatives: pd.DataFrame) -> np.ndarray:
    columns = [f"predicted_{name}" for name in PARAMETER_NAMES]
    missing = [name for name in columns if name not in representatives]
    if missing:
        raise ValueError(f"traditional starts frame missing columns: {missing}")
    return representatives[columns].to_numpy(dtype=np.float64)


def _traditional_multi_start_dispersion(
    starts_frame: pd.DataFrame, scaling: dict[str, dict[str, float]]
) -> dict[str, object]:
    columns = [f"predicted_{name}" for name in PARAMETER_NAMES]
    ranges = np.array([scaling[name]["range"] for name in PARAMETER_NAMES])
    per_parameter_std: dict[str, float] = {}
    disagreement_counts: list[int] = []
    for _, group in starts_frame.groupby("surface_id", sort=True):
        if len(group) < 2:
            continue
        matrix = group[columns].to_numpy(dtype=np.float64)
        if not np.isfinite(matrix).all():
            continue
        scaled = matrix / ranges
        std = scaled.std(axis=0, ddof=1)
        for index, name in enumerate(PARAMETER_NAMES):
            per_parameter_std.setdefault(name, []).append(std[index])
        pairwise = np.abs(scaled[:, None, :] - scaled[None, :, :])
        max_pair_distance = float(pairwise.max())
        disagreement_counts.append(1 if max_pair_distance > 0.5 else 0)
    return {
        "definition": (
            "mean over surfaces of the range-scaled cross-start standard "
            "deviation per parameter; disagreement = any pair of starts "
            "differs by more than 0.5 range-scaled units on any parameter"
        ),
        "mean_per_parameter_range_scaled_std": {
            name: float(np.mean(values))
            for name, values in per_parameter_std.items()
        },
        "start_disagreement_rate": float(np.mean(disagreement_counts))
        if disagreement_counts
        else None,
        "start_count_per_surface": 3,
    }


def main() -> int:
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    dataset = R2PrimaryDataset.from_jsonl(DATASET_PATH)
    assert_split_isolation(dataset)
    test_indices = dataset.indices_for_split("test")
    test_ids = {dataset.items[index].surface_id: index for index in test_indices}
    scaling = train_split_scaling(dataset)
    truth = np.stack([dataset.items[index].targets for index in test_indices])

    # ------------------------------------------------------------------ data
    write_json(
        EVIDENCE_ROOT / "dataset_identity.json",
        {
            "path": str(DATASET_PATH),
            "sha256": _sha256(DATASET_PATH),
            "total_surfaces": len(dataset),
            "split_counts": dataset.split_counts(),
            "test_surface_count": len(test_indices),
            "cross_split_overlap": "NONE_ASSERTED",
        },
    )
    write_json(
        EVIDENCE_ROOT / "protocol.json",
        {
            "protocol_config": json.loads(
                PROTOCOL_CONFIG_PATH.read_text(encoding="utf-8")
            ),
            "protocol_config_sha256": _sha256(PROTOCOL_CONFIG_PATH),
        },
    )

    # ------------------------------------------------------- neural methods
    training_manifest: dict[str, object] = {"model1": {}, "model2": {}}
    per_seed_predictions: dict[str, dict[int, np.ndarray]] = {
        "model1": {},
        "model2": {},
    }
    per_seed_headline: dict[str, dict[int, dict[str, float]]] = {
        "model1": {},
        "model2": {},
    }
    for model_kind in ("model1", "model2"):
        for seed in NEURAL_SEEDS:
            run_directory = CHECKPOINT_ROOT / f"{model_kind}_seed{seed}"
            checkpoint_path = run_directory / "best_validation_checkpoint.pt"
            if not checkpoint_path.exists():
                raise SystemExit(f"missing research run: {checkpoint_path}")
            run = load_run(run_directory, model_kind)
            from .training import predict_parameters

            predicted = predict_parameters(
                run["model"],
                dataset,
                test_indices,
                standardizer=run["standardizer"] if model_kind == "model1" else None,
            )
            per_seed_predictions[model_kind][seed] = predicted
            runtime = measure_inference_runtime(
                run["model"],
                dataset,
                test_indices,
                standardizer=run["standardizer"] if model_kind == "model1" else None,
            )
            summary = summarize_run(
                dataset,
                test_indices,
                predicted,
                scaling,
                method_label=f"{model_kind}_seed{seed}",
                runtime=runtime,
            )
            per_seed_headline[model_kind][seed] = {
                "range_scaled_parameter_rmse": summary["parameter_recovery"][
                    "aggregate"
                ]["range_scaled_parameter_rmse"],
                "standardized_parameter_rmse": summary["parameter_recovery"][
                    "aggregate"
                ]["standardized_parameter_rmse"],
                "constraint_validity_rate": summary["constraint_validity"][
                    "constraint_validity_rate"
                ],
                "normalized_price_rmse_mean": summary["repricing"][
                    "normalized_price_rmse_mean"
                ],
                "per_surface_inference_ms": runtime[
                    "per_surface_inference_ms_amortized"
                ],
            }
            training_summary = json.loads(
                (run_directory / "training_summary.json").read_text(encoding="utf-8")
            )
            training_manifest[model_kind][str(seed)] = {
                "seed": seed,
                "best_epoch": training_summary["best_epoch"],
                "epochs_completed": training_summary["epochs_completed"],
                "training_runtime_seconds": training_summary["runtime_seconds"],
                "git_sha": training_summary["git_sha"],
                "checkpoint_sha256": _sha256(checkpoint_path),
                "run_kind": training_summary["run_kind"],
            }
            pd.DataFrame(predicted, columns=PARAMETER_NAMES).assign(
                surface_id=[dataset.items[index].surface_id for index in test_indices]
            ).to_csv(
                EVIDENCE_ROOT / f"{model_kind}_seed{seed}_test_predictions.csv",
                index=False,
            )

    seed_rows: list[dict[str, object]] = []
    for model_kind in ("model1", "model2"):
        for seed in NEURAL_SEEDS:
            seed_rows.append(
                {"method": model_kind, "seed": seed, **per_seed_headline[model_kind][seed]}
            )
    pd.DataFrame(seed_rows).to_csv(
        EVIDENCE_ROOT / "neural_seed_results.csv", index=False
    )
    pd.DataFrame(
        [row for row in seed_rows if row["method"] == "model1"]
    ).to_csv(EVIDENCE_ROOT / "model1_seed_results.csv", index=False)
    pd.DataFrame(
        [row for row in seed_rows if row["method"] == "model2"]
    ).to_csv(EVIDENCE_ROOT / "model2_seed_results.csv", index=False)

    neural_stability = {
        model_kind: stability_metrics(
            per_seed_predictions[model_kind], per_seed_headline[model_kind]
        )
        for model_kind in ("model1", "model2")
    }

    # ----------------------------------------------- traditional calibration
    starts_path = EVIDENCE_ROOT / "traditional_calibration_starts.csv"
    starts_frame = pd.read_csv(starts_path)
    representatives = select_representatives(starts_frame)
    representatives = representatives[
        representatives["surface_id"].isin(test_ids)
    ].sort_values("surface_id")
    aligned_indices = [test_ids[sid] for sid in representatives["surface_id"]]
    traditional_predicted = _predicted_matrix_from_traditional(representatives)
    traditional_runtime = {
        "per_surface_calibration_seconds_mean": float(
            representatives["wall_seconds_all_starts"].mean()
        ),
        "per_surface_calibration_seconds_p95": float(
            representatives["wall_seconds_all_starts"].quantile(0.95)
        ),
        "starts_per_surface": 3,
        "budget": "3 starts x max_nfev 300 (frozen; see protocol)",
    }
    traditional_summary = summarize_run(
        dataset,
        aligned_indices,
        traditional_predicted,
        scaling,
        method_label="traditional_calibration",
        runtime=traditional_runtime,
    )
    traditional_dispersion = _traditional_multi_start_dispersion(starts_frame, scaling)
    representatives.to_csv(
        EVIDENCE_ROOT / "traditional_calibration_results.csv", index=False
    )

    # ------------------------------------------------------- full summaries
    final_summaries: dict[str, object] = {}
    for model_kind in ("model1", "model2"):
        stacked_mean = np.mean(
            [per_seed_predictions[model_kind][seed] for seed in NEURAL_SEEDS], axis=0
        )
        final_summaries[model_kind] = summarize_run(
            dataset,
            test_indices,
            stacked_mean,
            scaling,
            method_label=f"{model_kind}_seed_mean_prediction",
        )
    final_summaries["traditional_calibration"] = traditional_summary
    write_json(
        EVIDENCE_ROOT / "parameter_metrics.json",
        {
            method: summary["parameter_recovery"]
            for method, summary in final_summaries.items()
        },
    )
    write_json(
        EVIDENCE_ROOT / "repricing_metrics.json",
        {method: summary["repricing"] for method, summary in final_summaries.items()},
    )
    write_json(
        EVIDENCE_ROOT / "validity_metrics.json",
        {
            method: summary["constraint_validity"]
            for method, summary in final_summaries.items()
        },
    )
    write_json(
        EVIDENCE_ROOT / "identifiability_metrics.json",
        {
            method: summary["identifiability_aware"]
            for method, summary in final_summaries.items()
        },
    )
    write_json(
        EVIDENCE_ROOT / "stability_metrics.json",
        {
            "neural_cross_seed": neural_stability,
            "traditional_multi_start": traditional_dispersion,
        },
    )
    write_json(
        EVIDENCE_ROOT / "runtime_metrics.json",
        {
            method: summary.get("runtime")
            for method, summary in final_summaries.items()
        },
    )
    write_json(
        EVIDENCE_ROOT / "training_run_manifest.json", training_manifest
    )

    # -------------------------------------------------- unified comparison
    def _headline(method: str, summary: dict) -> dict[str, object]:
        recovery = summary["parameter_recovery"]
        return {
            "method": method,
            "range_scaled_param_rmse": recovery["aggregate"][
                "range_scaled_parameter_rmse"
            ],
            "standardized_param_rmse": recovery["aggregate"][
                "standardized_parameter_rmse"
            ],
            "v0_total_mae": recovery["factorwise"]["v0_total_mae"],
            "theta_total_mae": recovery["factorwise"]["theta_total_mae"],
            "half_life_slow_mae_years": recovery["factorwise"][
                "half_life_slow_mae_years"
            ],
            "half_life_fast_mae_years": recovery["factorwise"][
                "half_life_fast_mae_years"
            ],
            "factor_swap_confusion_rate": recovery["factorwise"][
                "factor_swap_confusion_rate"
            ],
            "constraint_validity_rate": summary["constraint_validity"][
                "constraint_validity_rate"
            ],
            "repricing_normalized_rmse_mean": summary["repricing"][
                "normalized_price_rmse_mean"
            ],
            "repricing_normalized_rmse_p95": summary["repricing"][
                "normalized_price_rmse_p95"
            ],
            "repricing_success_rate_le_1e-4": summary["identifiability_aware"][
                "repricing_tolerance_success"
            ]["rmse<=0.0001"]["rate"],
            "repricing_success_rate_le_1e-3": summary["identifiability_aware"][
                "repricing_tolerance_success"
            ]["rmse<=0.001"]["rate"],
            "param_recovery_rate_le_0.25": summary["identifiability_aware"][
                "parameter_tolerance_success"
            ]["range_scaled_rmse<=0.25"]["rate"],
            "param_recovery_given_repricing_le_1e-4_and_le_0.25": summary[
                "identifiability_aware"
            ]["conditioned_recovery"][
                "parameter_recovery_given_repricing<=0.0001_param<=0.25"
            ],
        }

    comparison = pd.DataFrame(
        [
            _headline("model1_seed_mean", final_summaries["model1"]),
            _headline("model2_seed_mean", final_summaries["model2"]),
            _headline("traditional_calibration", traditional_summary),
        ]
    )
    comparison.to_csv(EVIDENCE_ROOT / "synthetic_test_comparison.csv", index=False)
    print(comparison.to_string(index=False))
    print(f"\nevidence written under {EVIDENCE_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
