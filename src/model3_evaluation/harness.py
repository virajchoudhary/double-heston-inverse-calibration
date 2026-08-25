"""Canonical sealed clean-test harness for future frozen Model3 checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.constants import PARAMETER_NAMES
from src.r2_primary.dataset import R2PrimaryDataset, assert_split_isolation
from src.r2_primary.evaluation import (
    measure_inference_runtime,
    stability_metrics,
    summarize_run,
    train_split_scaling,
)
from src.utils import write_json

from .adapter import Model3EvaluationAdapter
from .contracts import (
    DATASET_PATH,
    EXPECTED_DATASET_SHA256,
    REPO_ROOT,
    REQUIRED_SEEDS,
    build_seed_contract,
    deterministic_digest,
    read_json,
    sha256_file,
    verify_freeze_manifest,
)
from .locking import require_frozen_test_authorization


FROZEN_BASELINE_EVIDENCE_ROOT = (
    Path(__file__).resolve().parents[2] / "evidence/r2_primary_comparison_20260823"
)
BASELINE_MANIFEST_PATH = FROZEN_BASELINE_EVIDENCE_ROOT / "FINAL_EVALUATION_EVIDENCE_MANIFEST.json"
FROZEN_BASELINE_REVIEW_COMMIT = "a42566e4ba746aad4db56dfd2620b3e118a72f4b"
FROZEN_BASELINE_GIT_BLOB_SHAS = {
    "FINAL_EVALUATION_EVIDENCE_MANIFEST.json": "515ce5e892e13669e84bb893bfddacbf0e921b09",
    "neural_seed_results.csv": "bd3bb60c84f005755526119d4c0f2556971a6c85",
    "synthetic_test_comparison.csv": "910c7925ec48a5eedb479f94dc8ad6d743725c29",
    "traditional_calibration_results.csv": "06876469757f2eb8297bbbf9482e7468d05db41c",
}
EVALUATION_SCHEMA = "MODEL3_CLEAN_EVALUATION_RESULT_MANIFEST_V1"


class SealedEvaluationError(ValueError):
    """A future result bundle failed the sealed intake contract."""


def canonical_train_validation_signatures(dataset: R2PrimaryDataset) -> tuple[str, str]:
    """Digest exact non-test surface-id/parameter-hash ordering."""
    signatures: list[str] = []
    for split in ("train", "validation"):
        indices = dataset.indices_for_split(split)
        signatures.append(deterministic_digest({
            "surface_ids": [dataset.items[index].surface_id for index in indices],
            "parameter_vector_hashes": [
                dataset.items[index].parameter_vector_hash for index in indices
            ],
        }))
    return signatures[0], signatures[1]


def _current_git_sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    return completed.stdout.strip()


def _git_blob_sha(path: Path) -> str:
    completed = subprocess.run(
        ["git", "hash-object", "--", str(path)],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    return completed.stdout.strip()


def load_frozen_baseline_evidence(
    evidence_root: str | Path = FROZEN_BASELINE_EVIDENCE_ROOT,
) -> dict[str, Any]:
    """Validate and pin existing ANN/Model2/traditional result artifacts."""
    root = Path(evidence_root)
    manifest_path = root / "FINAL_EVALUATION_EVIDENCE_MANIFEST.json"
    manifest = read_json(manifest_path)
    required_header = {
        "manifest_kind": "FINAL_R2_PRIMARY_COMPARISON_EVIDENCE_MANIFEST",
        "first_synthetic_test_read": True,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
    }
    mismatches = [
        key for key, value in required_header.items() if manifest.get(key) != value
    ]
    if mismatches:
        raise SealedEvaluationError(f"frozen baseline manifest mismatch: {mismatches}")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise SealedEvaluationError("baseline evidence manifest has no file map")
    for relative, expected_blob in FROZEN_BASELINE_GIT_BLOB_SHAS.items():
        path = root / relative
        if not path.is_file() or _git_blob_sha(path) != expected_blob:
            raise SealedEvaluationError(f"frozen baseline artifact differs from reviewed Git blob: {relative}")
    serialization_mismatches = [
        relative
        for relative, expected_hash in files.items()
        if not (root / relative).is_file() or sha256_file(root / relative) != expected_hash
    ]
    neural = pd.read_csv(root / "neural_seed_results.csv")
    aggregate = pd.read_csv(root / "synthetic_test_comparison.csv")
    traditional = pd.read_csv(root / "traditional_calibration_results.csv")
    return {
        "evidence_root": root,
        "manifest": manifest,
        "manifest_sha256": sha256_file(manifest_path),
        "reviewed_git_commit": FROZEN_BASELINE_REVIEW_COMMIT,
        "reviewed_git_blob_shas": dict(FROZEN_BASELINE_GIT_BLOB_SHAS),
        "original_manifest_byte_hash_mismatch_count": len(serialization_mismatches),
        "provenance_note": (
            "The reviewed commit contains a known final-manifest serialization defect: "
            "its embedded byte hashes predate committed JSON/CSV bytes. Authorization "
            "uses exact reviewed Git blob identities; this does not alter result values."
        ),
        "neural_seed_results": neural,
        "synthetic_test_comparison": aggregate,
        "traditional_calibration_results": traditional,
    }


def _headline_from_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    recovery = summary["parameter_recovery"]
    validity = summary["constraint_validity"]
    repricing = summary["repricing"]
    identifiability = summary["identifiability_aware"]
    runtime = summary.get("runtime", {})
    return {
        "method": summary["method"],
        "range_scaled_param_rmse": recovery["aggregate"]["range_scaled_parameter_rmse"],
        "standardized_param_rmse": recovery["aggregate"]["standardized_parameter_rmse"],
        "v0_total_mae": recovery["factorwise"]["v0_total_mae"],
        "theta_total_mae": recovery["factorwise"]["theta_total_mae"],
        "factor_swap_confusion_rate": recovery["factorwise"]["factor_swap_confusion_rate"],
        "constraint_validity_rate": validity["constraint_validity_rate"],
        "repricing_normalized_rmse_mean": repricing["normalized_price_rmse_mean"],
        "repricing_normalized_rmse_p95": repricing["normalized_price_rmse_p95"],
        "per_surface_inference_ms": runtime.get("per_surface_inference_ms_amortized"),
    }


def _write_predictions(
    path: Path,
    dataset: R2PrimaryDataset,
    indices: list[int],
    predictions: np.ndarray,
) -> None:
    frame = pd.DataFrame(predictions, columns=list(PARAMETER_NAMES))
    frame.insert(0, "surface_id", [dataset.items[index].surface_id for index in indices])
    frame.to_csv(path, index=False, lineterminator="\n")


def _partial_failure(output_root: Path, error: BaseException) -> None:
    try:
        write_json(
            output_root / "evaluation_status.json",
            {
                "status": "PARTIAL_FAILED_CLOSED",
                "error_type": type(error).__name__,
                "research_metrics_complete": False,
            },
        )
    except OSError:
        pass


def run_clean_evaluation(
    *,
    freeze_manifest_path: str | Path,
    checkpoint_roots: Mapping[int, str | Path],
    output_root: str | Path,
    exact_command: str,
    authorize_frozen_test_evaluation: bool = False,
    dataset_path: str | Path = DATASET_PATH,
    device: str = "cpu",
) -> dict[str, Any]:
    """Evaluate only after valid three-seed freeze plus explicit authorization."""
    # Authorization is deliberately checked before opening the dataset path.
    verification = require_frozen_test_authorization(
        freeze_manifest_path,
        authorized=authorize_frozen_test_evaluation,
    )
    started_unix_seconds = time.time()
    destination = Path(output_root)
    if destination.exists():
        raise SealedEvaluationError(f"refusing to overwrite evaluation output: {destination}")
    destination.mkdir(parents=True)
    write_json(
        destination / "evaluation_status.json",
        {"status": "PARTIAL_IN_PROGRESS", "research_metrics_complete": False},
    )
    try:
        freeze_payload = read_json(freeze_manifest_path)
        if set(checkpoint_roots) != set(REQUIRED_SEEDS):
            raise SealedEvaluationError("exactly checkpoint roots for seeds 11/22/33 are required")
        dataset_sha = sha256_file(dataset_path)
        if dataset_sha != EXPECTED_DATASET_SHA256:
            raise SealedEvaluationError("frozen clean dataset changed")
        dataset = R2PrimaryDataset.from_jsonl(dataset_path)
        assert_split_isolation(dataset)
        if dataset.split_counts() != {"train": 7500, "validation": 1250, "test": 1250}:
            raise SealedEvaluationError("frozen clean dataset population mismatch")
        train_indices = dataset.indices_for_split("train")
        validation_indices = dataset.indices_for_split("validation")
        test_indices = dataset.indices_for_split("test")
        train_signature, validation_signature = canonical_train_validation_signatures(dataset)
        scaling = train_split_scaling(dataset)

        seed_contracts: dict[str, dict[str, Any]] = {}
        adapters: dict[int, Model3EvaluationAdapter] = {}
        predictions: dict[str, np.ndarray] = {}
        seed_rows: list[dict[str, Any]] = []
        seed_summaries: dict[str, dict[str, Any]] = {}
        for seed in REQUIRED_SEEDS:
            raw_contract = build_seed_contract(
                checkpoint_roots[seed],
                seed=seed,
                experiment_id=str(freeze_payload["experiment_id"]),
                expected_train_population_sha256=train_signature,
                expected_validation_population_sha256=validation_signature,
            )
            frozen_contract = freeze_payload["seed_contracts"][str(seed)]
            if raw_contract != frozen_contract:
                raise SealedEvaluationError(f"raw seed {seed} differs from its frozen contract")
            seed_contracts[str(seed)] = raw_contract
            adapter = Model3EvaluationAdapter(
                Path(checkpoint_roots[seed]) / "checkpoint.pt",
                expected_seed=seed,
                device=device,
            )
            adapters[seed] = adapter
            predicted = adapter.predict_parameters(
                dataset, test_indices, seed_identity=seed
            )
            predictions[str(seed)] = predicted
            _write_predictions(
                destination / f"model3_seed{seed}_test_predictions.csv",
                dataset,
                test_indices,
                predicted,
            )
            measured_runtime = measure_inference_runtime(
                adapter.system,
                dataset,
                test_indices,
                standardizer=adapter.standardizer,
                repetitions=3,
            )
            runtime = {
                **measured_runtime,
                "inference_device": str(device),
                "measurement_definition": "canonical best-of-three full-split wall clock",
            }
            summary = summarize_run(
                dataset,
                test_indices,
                predicted,
                scaling,
                method_label=f"model3_seed{seed}",
                runtime=runtime,
            )
            seed_summaries[str(seed)] = summary
            seed_rows.append(
                {
                    "seed": seed,
                    "best_epoch": raw_contract["best_epoch"],
                    **_headline_from_summary(summary),
                }
            )

        mean_prediction = np.mean(
            [predictions[str(seed)] for seed in REQUIRED_SEEDS], axis=0
        )
        mean_summary = summarize_run(
            dataset,
            test_indices,
            mean_prediction,
            scaling,
            method_label="model3_seed_mean_prediction",
        )
        headline_values = {
            key: [
                row[key]
                for row in seed_rows
                if isinstance(row.get(key), (int, float))
            ]
            for key in (
                "range_scaled_param_rmse", "standardized_param_rmse",
                "constraint_validity_rate", "repricing_normalized_rmse_mean",
                "repricing_normalized_rmse_p95", "per_surface_inference_ms",
            )
        }
        dispersion = {
            key: float(np.std(values, ddof=1)) if len(values) == len(REQUIRED_SEEDS) else None
            for key, values in headline_values.items()
        }
        cross_seed_prediction = stability_metrics(predictions, {
            seed: {
                "range_scaled_parameter_rmse": seed_summaries[seed]["parameter_recovery"]["aggregate"]["range_scaled_parameter_rmse"],
                "standardized_parameter_rmse": seed_summaries[seed]["parameter_recovery"]["aggregate"]["standardized_parameter_rmse"],
                "constraint_validity_rate": seed_summaries[seed]["constraint_validity"]["constraint_validity_rate"],
                "normalized_price_rmse_mean": seed_summaries[seed]["repricing"]["normalized_price_rmse_mean"],
            }
            for seed in predictions
        })

        pd.DataFrame(seed_rows).to_csv(
            destination / "model3_seed_results.csv", index=False, lineterminator="\n"
        )
        write_json(destination / "parameter_metrics.json", {
            **{f"model3_seed{seed}": seed_summaries[str(seed)]["parameter_recovery"] for seed in REQUIRED_SEEDS},
            "model3_seed_mean_prediction": mean_summary["parameter_recovery"],
        })
        write_json(destination / "repricing_metrics.json", {
            **{f"model3_seed{seed}": seed_summaries[str(seed)]["repricing"] for seed in REQUIRED_SEEDS},
            "model3_seed_mean_prediction": mean_summary["repricing"],
        })
        write_json(destination / "validity_metrics.json", {
            **{f"model3_seed{seed}": seed_summaries[str(seed)]["constraint_validity"] for seed in REQUIRED_SEEDS},
            "model3_seed_mean_prediction": mean_summary["constraint_validity"],
        })
        write_json(destination / "identifiability_metrics.json", {
            **{f"model3_seed{seed}": seed_summaries[str(seed)]["identifiability_aware"] for seed in REQUIRED_SEEDS},
            "model3_seed_mean_prediction": mean_summary["identifiability_aware"],
        })
        write_json(destination / "runtime_metrics.json", {
            f"model3_seed{seed}": seed_summaries[str(seed)].get("runtime") for seed in REQUIRED_SEEDS
        })
        write_json(destination / "three_seed_aggregation.json", {
            "schema": "MODEL3_THREE_SEED_AGGREGATION_V1",
            "seed_level_retained": True,
            "headline_dispersion": dispersion,
            "cross_seed_prediction_dispersion": cross_seed_prediction,
            "warning": "Seed-level rows are authoritative; means never hide an unstable seed.",
        })

        baselines = load_frozen_baseline_evidence()
        baseline_table = baselines["synthetic_test_comparison"].to_dict(orient="records")
        model3_headline = _headline_from_summary(mean_summary)
        aligned_keys = sorted(set(baseline_table[0]) & set(model3_headline))
        comparison_rows = [
            {key: row.get(key) for key in aligned_keys} for row in baseline_table
        ] + [{key: model3_headline.get(key) for key in aligned_keys}]
        pd.DataFrame(comparison_rows)[aligned_keys].to_csv(
            destination / "model3_vs_frozen_baselines.csv",
            index=False,
            lineterminator="\n",
        )
        write_json(destination / "frozen_baseline_intake.json", {
            "schema": "FROZEN_BASELINE_INTAKE_V1",
            "evidence_root": baselines["evidence_root"].as_posix(),
            "manifest_sha256": baselines["manifest_sha256"],
            "recomputed_baselines": False,
            "conceptual_warning": (
                "LOW PRICING ERROR IS NOT CORRECT PARAMETER RECOVERY."
            ),
            "allowed_outcomes": [
                "improves_recovery", "robustness_only_benefit",
                "stability_only_benefit", "no_measurable_benefit",
                "excessive_cost",
            ],
        })

        generated_files = [
            "evaluation_status.json", "freeze_manifest_copy.json",
            *[f"model3_seed{seed}_test_predictions.csv" for seed in REQUIRED_SEEDS],
            "model3_seed_results.csv", "parameter_metrics.json",
            "repricing_metrics.json", "validity_metrics.json",
            "identifiability_metrics.json", "runtime_metrics.json",
            "three_seed_aggregation.json", "model3_vs_frozen_baselines.csv",
            "frozen_baseline_intake.json",
        ]
        shutil.copyfile(freeze_manifest_path, destination / "freeze_manifest_copy.json")
        artifact_hashes = {
            name: sha256_file(destination / name) for name in generated_files
        }
        test_ids = [dataset.items[index].surface_id for index in test_indices]
        manifest = {
            "schema": EVALUATION_SCHEMA,
            "completion_state": "COMPLETE",
            "evaluation_source_git_sha": _current_git_sha(),
            "training_checkpoint_freeze_manifest_sha256": sha256_file(freeze_manifest_path),
            "checkpoint_contracts": seed_contracts,
            "final_r2_dataset_sha256": dataset_sha,
            "test_population_identity": {
                "count": len(test_indices),
                "surface_id_order_sha256": deterministic_digest(test_ids),
            },
            "exact_command": exact_command,
            "environment": {
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "pandas_version": pd.__version__,
            },
            "hardware": {
                "host": socket.gethostname(),
                "platform": platform.platform(),
                "requested_inference_device": device,
            },
            "artifact_hashes": artifact_hashes,
            "metric_hashes": {
                name: artifact_hashes[name]
                for name in (
                    "parameter_metrics.json", "repricing_metrics.json",
                    "validity_metrics.json", "identifiability_metrics.json",
                )
            },
            "prediction_hashes": {
                f"model3_seed{seed}": artifact_hashes[f"model3_seed{seed}_test_predictions.csv"]
                for seed in REQUIRED_SEEDS
            },
            "repricing_hash": artifact_hashes["repricing_metrics.json"],
            "runtime_hash": artifact_hashes["runtime_metrics.json"],
            "started_unix_seconds": started_unix_seconds,
            "completed_unix_seconds": time.time(),
        }
        write_json(destination / "final_evaluation_manifest.json", manifest)
        write_json(
            destination / "evaluation_status.json",
            {"status": "COMPLETE", "research_metrics_complete": True},
        )
        # Refresh the manifest's own status hash without changing any metric bytes.
        manifest["artifact_hashes"]["evaluation_status.json"] = sha256_file(
            destination / "evaluation_status.json"
        )
        write_json(destination / "final_evaluation_manifest.json", manifest)
        return manifest
    except BaseException as error:
        _partial_failure(destination, error)
        raise
