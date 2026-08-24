"""Frozen-checkpoint neural evaluation over the R2 noise cohorts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.constants import PARAMETER_NAMES
from src.r2_primary.dataset import R2PrimaryDataset
from src.r2_primary.evaluation import (
    _range_scaled_error_matrix,
    constraint_validity_metrics,
    identifiability_aware_metrics,
    parameter_recovery_metrics,
    reprice_normalized,
    repricing_metrics,
    stability_metrics,
    train_split_scaling,
)
from src.r2_primary.training import load_run, predict_parameters
from src.utils import write_json

from .execution import (
    CLEAN_DATASET_PATH,
    EVIDENCE_ROOT,
    REPO_ROOT,
    assert_clean_dataset_identity,
    iter_test_records,
    level_label,
    load_frozen_protocol,
    sha256_path,
)

DATA_COHORT_ROOT = REPO_ROOT / "data" / "r2_noise_robustness"

NEURAL_SEEDS = (11, 22, 33)
CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / "r2_primary_comparison"
CANONICAL_PRIMARY_EVIDENCE = (
    REPO_ROOT / "evidence" / "r2_primary_comparison_20260823"
)
CANONICAL_PRIMARY_MERGE_COMMIT = (
    "72ad8e1aa845ec4c6f0fc61fc526df75438639bb"
)


def safe_level_label(label: str) -> str:
    return label.replace(".", "_").replace("%", "pct")


def _observed_matrix(dataset: R2PrimaryDataset, indices: list[int]) -> np.ndarray:
    return np.stack(
        [
            np.where(
                dataset.items[index].mask,
                dataset.items[index].normalized_prices,
                np.nan,
            )
            for index in indices
        ]
    )


def static_arbitrage_diagnostics(
    clean_records: list[dict[str, Any]], noisy_records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Count retained raw-noise violations without repairing any observation."""
    if len(clean_records) != len(noisy_records):
        raise ValueError("clean/noisy record populations are misaligned")
    parity_counts: list[int] = []
    vertical_counts: list[int] = []
    for clean, noisy in zip(clean_records, noisy_records, strict=True):
        if clean["surface_id"] != noisy["surface_id"]:
            raise ValueError("surface ordering differs in arbitrage diagnostics")
        clean_prices = np.asarray(clean["prices"], dtype=np.float64)
        noisy_prices = np.asarray(noisy["prices"], dtype=np.float64)
        mask = np.asarray(clean["mask"], dtype=bool)
        keys = [tuple(key) for key in clean["slot_keys"]]
        by_identity: dict[tuple[int, float], dict[str, int]] = {}
        by_vertical: dict[tuple[str, int], list[tuple[float, int]]] = {}
        for index, (rank, moneyness, option_type) in enumerate(keys):
            if not mask[index]:
                continue
            by_identity.setdefault((rank, moneyness), {})[option_type] = index
            by_vertical.setdefault((option_type, rank), []).append(
                (float(moneyness), index)
            )

        parity_violations = 0
        for pair in by_identity.values():
            if set(pair) != {"call", "put"}:
                continue
            call_index, put_index = pair["call"], pair["put"]
            clean_parity = (
                clean_prices[call_index] - clean_prices[put_index]
            )
            noisy_parity = (
                noisy_prices[call_index] - noisy_prices[put_index]
            )
            tolerance = max(16.0 * np.finfo(float).eps, abs(clean_parity) * 1e-12)
            parity_violations += int(abs(noisy_parity - clean_parity) > tolerance)

        vertical_violations = 0
        for (vertical_option_type, _rank), members in by_vertical.items():
            ordered = sorted(members)
            for (_, left), (_, right) in zip(ordered[:-1], ordered[1:], strict=True):
                tolerance = 16.0 * np.finfo(float).eps
                if vertical_option_type == "call":
                    vertical_violations += int(
                        noisy_prices[right] > noisy_prices[left] + tolerance
                    )
                else:
                    vertical_violations += int(
                        noisy_prices[right] < noisy_prices[left] - tolerance
                    )
        parity_counts.append(parity_violations)
        vertical_counts.append(vertical_violations)

    parity_array = np.asarray(parity_counts)
    vertical_array = np.asarray(vertical_counts)
    return {
        "definition": {
            "parity": (
                "same-rank/moneyness call-minus-put parity differs from the "
                "clean parity beyond floating-point tolerance"
            ),
            "vertical_spread": (
                "adjacent central-strike call or put spread violates raw "
                "monotonicity beyond floating-point tolerance"
            ),
            "policy": "RETAINED_AND_FLAGGED_NEVER_REPAIRED",
        },
        "parity_violating_slot_rate": float((parity_array > 0).mean()),
        "parity_violation_slot_count_total": int(parity_array.sum()),
        "vertical_violating_surface_rate": float((vertical_array > 0).mean()),
        "vertical_violation_pair_count_total": int(vertical_array.sum()),
        "per_surface_parity_violations": parity_counts,
        "per_surface_vertical_violations": vertical_counts,
    }


def _run_metrics(
    *,
    truth: np.ndarray,
    predicted: np.ndarray,
    repriced: np.ndarray,
    noisy_observed: np.ndarray,
    clean_observed: np.ndarray,
    scaling: dict[str, dict[str, float]],
) -> dict[str, Any]:
    recovery = parameter_recovery_metrics(truth, predicted, scaling)
    validity = constraint_validity_metrics(predicted)
    noisy_repricing = repricing_metrics(noisy_observed, repriced)
    clean_repricing = repricing_metrics(clean_observed, repriced)
    noisy_identifiability = identifiability_aware_metrics(
        noisy_observed, repriced, truth, predicted, scaling
    )
    clean_identifiability = identifiability_aware_metrics(
        clean_observed, repriced, truth, predicted, scaling
    )
    return {
        "parameter_recovery": recovery,
        "constraint_validity": validity,
        "fit_to_noisy_observation": noisy_repricing,
        "clean_latent_repricing": clean_repricing,
        "identifiability_fit_to_noisy": noisy_identifiability,
        "identifiability_clean_latent": clean_identifiability,
        "population": f"N{len(truth)}",
    }


def _headline_row(method: str, seed: int | str, level: str, metrics: dict[str, Any]) -> dict[str, Any]:
    recovery = metrics["parameter_recovery"]
    noisy = metrics["fit_to_noisy_observation"]
    clean = metrics["clean_latent_repricing"]
    return {
        "method": method,
        "seed": seed,
        "noise_level_label": level,
        "range_scaled_parameter_rmse": recovery["aggregate"][
            "range_scaled_parameter_rmse"
        ],
        "standardized_parameter_rmse": recovery["aggregate"][
            "standardized_parameter_rmse"
        ],
        "constraint_validity_rate": metrics["constraint_validity"][
            "constraint_validity_rate"
        ],
        "noisy_price_rmse_mean": noisy["normalized_price_rmse_mean"],
        "noisy_price_rmse_p95": noisy["normalized_price_rmse_p95"],
        "clean_latent_price_rmse_mean": clean["normalized_price_rmse_mean"],
        "clean_latent_price_rmse_p95": clean["normalized_price_rmse_p95"],
        "noisy_repricing_success_le_1e-4": metrics[
            "identifiability_fit_to_noisy"
        ]["repricing_tolerance_success"]["rmse<=0.0001"]["rate"],
        "noisy_repricing_success_le_1e-3": metrics[
            "identifiability_fit_to_noisy"
        ]["repricing_tolerance_success"]["rmse<=0.001"]["rate"],
        "clean_repricing_success_le_1e-4": metrics[
            "identifiability_clean_latent"
        ]["repricing_tolerance_success"]["rmse<=0.0001"]["rate"],
        "parameter_recovery_le_0_10": metrics["identifiability_fit_to_noisy"][
            "parameter_tolerance_success"
        ]["range_scaled_rmse<=0.1"]["rate"],
        "parameter_recovery_le_0_25": metrics["identifiability_fit_to_noisy"][
            "parameter_tolerance_success"
        ]["range_scaled_rmse<=0.25"]["rate"],
    }


def _load_checkpoint_run(model_kind: str, seed: int) -> dict[str, Any]:
    directory = CHECKPOINT_ROOT / f"{model_kind}_seed{seed}"
    run = load_run(directory, model_kind)
    training = json.loads((directory / "training_summary.json").read_text())
    if training.get("run_kind") != "RESEARCH":
        raise ValueError(f"non-research checkpoint rejected: {directory}")
    if model_kind == "model2":
        required_sha = "2b5d41cd232241a1f23348411643a9ce58b53b7a"
        if training.get("git_sha") != required_sha or training.get("device_used") != "cuda":
            raise ValueError(f"Model2 P100 provenance mismatch: {directory}")
    return {**run, "training_summary": training}


def json_safe_metrics(value: Any) -> Any:
    """Convert NumPy metric containers to JSON-safe builtins recursively."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_safe_metrics(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe_metrics(item) for item in value]
    return value


def _seed_mean_metrics(
    *,
    dataset: R2PrimaryDataset,
    indices: list[int],
    truth: np.ndarray,
    predictions: dict[int, np.ndarray],
    noisy_observed: np.ndarray,
    clean_observed: np.ndarray,
    scaling: dict[str, dict[str, float]],
) -> dict[str, Any]:
    predicted = np.mean(list(predictions.values()), axis=0)
    repriced = reprice_normalized(
        dataset,
        indices,
        predicted,
    )
    return _run_metrics(
        truth=truth,
        predicted=predicted,
        repriced=repriced,
        noisy_observed=noisy_observed,
        clean_observed=clean_observed,
        scaling=scaling,
    )


def _compare_zero_percent_gate(neural_root: Path) -> dict[str, Any]:
    canonical = CANONICAL_PRIMARY_EVIDENCE
    zero_dir = neural_root / "level_0pct"
    prediction_checks: dict[str, bool] = {}
    for model_kind in ("model1", "model2"):
        for seed in NEURAL_SEEDS:
            name = f"{model_kind}_seed{seed}_test_predictions.csv"
            produced = (zero_dir / name).read_bytes()
            expected = (canonical / name).read_bytes()
            prediction_checks[name] = produced == expected

    canonical_rows = pd.read_csv(canonical / "neural_seed_results.csv")
    produced_rows = pd.read_csv(zero_dir / "seed_headline.csv")
    canonical_science = canonical_rows.drop(columns=["per_surface_inference_ms"])
    renamed = produced_rows.rename(
        columns={"noisy_price_rmse_mean": "normalized_price_rmse_mean"}
    )
    comparison_columns = list(canonical_science.columns)
    metric_bitwise = renamed[comparison_columns].equals(canonical_science)
    checks = {
        "prediction_csv_bitwise_equal": all(prediction_checks.values()),
        "neural_headline_metrics_exact": bool(metric_bitwise),
        "all_six_primary_seeds_present": len(produced_rows) == 6,
    }
    report = {
        "artifact_kind": "R2_NOISE_ROBUSTNESS_ZERO_PERCENT_NEURAL_GATE",
        "canonical_evidence_commit": CANONICAL_PRIMARY_MERGE_COMMIT,
        "comparison_note": (
            "prediction CSV values are compared bitwise; frozen headline "
            "metrics are compared exactly with runtime excluded"
        ),
        "checks": checks,
        "prediction_csv_bitwise_checks": prediction_checks,
        "runtime_column_excluded_from_metric_check": "per_surface_inference_ms",
        "status": (
            "PASSED"
            if all(checks.values())
            else "FAILED_STOP_BEFORE_POSITIVE_NOISE_INTERPRETATION"
        ),
    }
    write_json(neural_root / "ZERO_PERCENT_GATE.json", report)
    if report["status"] != "PASSED":
        raise RuntimeError("0% neural reproduction gate failed")
    return report


def evaluate_neural_levels(
    requested_levels: tuple[float, ...],
    *,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate all six frozen research checkpoints at requested levels."""
    protocol = load_frozen_protocol()
    assert_clean_dataset_identity()
    allowed = tuple(float(value) for value in protocol["noise_levels"])
    if any(level not in allowed for level in requested_levels):
        raise ValueError("requested level outside frozen protocol")
    root = Path(output_root or EVIDENCE_ROOT / "neural")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite neural evidence: {root}")
    root.mkdir(parents=True)

    clean_dataset = R2PrimaryDataset.from_jsonl(CLEAN_DATASET_PATH)
    test_indices = clean_dataset.indices_for_split("test")
    clean_ids = [clean_dataset.items[index].surface_id for index in test_indices]
    scaling = train_split_scaling(clean_dataset)
    truth = np.stack([clean_dataset.items[index].targets for index in test_indices])
    clean_observed = _observed_matrix(clean_dataset, test_indices)
    clean_record_by_id = {
        record["surface_id"]: record for _, record in iter_test_records(CLEAN_DATASET_PATH)
    }

    checkpoint_provenance: dict[str, Any] = {"model1": {}, "model2": {}}
    runs: dict[str, dict[int, dict[str, Any]]] = {"model1": {}, "model2": {}}
    for model_kind in ("model1", "model2"):
        for seed in NEURAL_SEEDS:
            runs[model_kind][seed] = _load_checkpoint_run(model_kind, seed)
            checkpoint_provenance[model_kind][str(seed)] = {
                "checkpoint_directory": (CHECKPOINT_ROOT / f"{model_kind}_seed{seed}").as_posix(),
                **runs[model_kind][seed]["training_summary"],
            }
    write_json(root / "checkpoint_provenance.json", checkpoint_provenance)

    level_rows: list[dict[str, Any]] = []
    stability: dict[str, Any] = {}
    generated_files: dict[str, str] = {}
    for level in requested_levels:
        label = level_label(level, protocol["noise_level_labels"])
        source_path = (
            CLEAN_DATASET_PATH
            if level == 0.0
            else DATA_COHORT_ROOT / "levels" / label / "noisy_surfaces.jsonl"
        )
        if level > 0.0:
            cohort_manifest = json.loads(
                (DATA_COHORT_ROOT / "MANIFEST.json").read_text(encoding="utf-8")
            )
            relative = f"levels/{label}/noisy_surfaces.jsonl"
            if cohort_manifest["files"][relative]["sha256"] != sha256_path(source_path):
                raise ValueError(f"cohort manifest mismatch: {source_path}")
        derived_dataset = R2PrimaryDataset.from_jsonl(source_path, splits={"test"})
        derived_indices = derived_dataset.indices_for_split("test")
        derived_ids = [
            derived_dataset.items[index].surface_id for index in derived_indices
        ]
        if derived_ids != clean_ids or len(derived_ids) != 1250:
            raise ValueError("derived population does not match canonical test order")
        noisy_observed = _observed_matrix(derived_dataset, derived_indices)
        clean_records = [clean_record_by_id[surface_id] for surface_id in clean_ids]
        noisy_records = [
            record
            for _, record in iter_test_records(source_path)
        ]
        diagnostics = static_arbitrage_diagnostics(clean_records, noisy_records)

        per_seed_predictions: dict[int, np.ndarray] = {}
        per_seed_repriced: dict[int, np.ndarray] = {}
        per_seed_headline: dict[tuple[str, int], dict[str, Any]] = {}
        per_surface_rows: list[dict[str, Any]] = []
        level_dir = root / f"level_{safe_level_label(label)}"
        level_dir.mkdir(exist_ok=True)
        for model_kind in ("model1", "model2"):
            for seed in NEURAL_SEEDS:
                run = runs[model_kind][seed]
                predicted = predict_parameters(
                    run["model"],
                    derived_dataset,
                    derived_indices,
                    standardizer=(
                        run["standardizer"] if model_kind == "model1" else None
                    ),
                )
                repriced = reprice_normalized(
                    derived_dataset, derived_indices, predicted
                )
                per_seed_repriced[seed] = repriced
                metrics = _run_metrics(
                    truth=truth,
                    predicted=predicted,
                    repriced=repriced,
                    noisy_observed=noisy_observed,
                    clean_observed=clean_observed,
                    scaling=scaling,
                )
                per_seed_predictions[seed] = predicted
                per_seed_headline[(model_kind, seed)] = _headline_row(
                    model_kind, seed, label, metrics
                )
                parameter_rmse = np.sqrt(
                    (
                        _range_scaled_error_matrix(truth, predicted, scaling) ** 2
                    ).mean(axis=1)
                )
                for position, surface_id in enumerate(clean_ids):
                    per_surface_rows.append(
                        {
                            "method": model_kind,
                            "seed": seed,
                            "surface_id": surface_id,
                            "parameter_range_scaled_rmse": float(parameter_rmse[position]),
                            "noisy_price_rmse": float(metrics[
                                "fit_to_noisy_observation"
                            ]["per_surface_rmse"][position]),
                            "clean_latent_price_rmse": float(metrics[
                                "clean_latent_repricing"
                            ]["per_surface_rmse"][position]),
                        }
                    )
                prediction_frame = pd.DataFrame(
                    predicted, columns=PARAMETER_NAMES
                ).assign(surface_id=clean_ids)
                prediction_name = f"{model_kind}_seed{seed}_test_predictions.csv"
                prediction_frame.to_csv(level_dir / prediction_name, index=False)
                generated_files[f"level_{safe_level_label(label)}/{prediction_name}"] = sha256_path(
                    level_dir / prediction_name
                )
        del per_seed_repriced  # per-seed repricing arrays are not persisted
        seed_mean = _seed_mean_metrics(
            dataset=derived_dataset,
            indices=derived_indices,
            truth=truth,
            predictions=per_seed_predictions,
            noisy_observed=noisy_observed,
            clean_observed=clean_observed,
            scaling=scaling,
        )
        write_json(level_dir / "seed_mean_metrics.json", seed_mean)
        generated_files[
            f"level_{safe_level_label(label)}/seed_mean_metrics.json"
        ] = sha256_path(level_dir / "seed_mean_metrics.json")
        seed_frame = pd.DataFrame(list(per_seed_headline.values()))
        seed_frame.to_csv(level_dir / "seed_headline.csv", index=False)
        pd.DataFrame(per_surface_rows).to_csv(
            level_dir / "per_surface_metrics.csv", index=False
        )
        write_json(level_dir / "arbitrage_diagnostics.json", diagnostics)
        stability[label] = {
            model_kind: stability_metrics(
                per_seed_predictions,
                {
                    seed: {
                        key: value
                        for key, value in per_seed_headline[
                            (model_kind, seed)
                        ].items()
                        if isinstance(value, (int, float))
                    }
                    for seed in NEURAL_SEEDS
                },
            )
            for model_kind in ("model1", "model2")
        }
        level_rows.extend(per_seed_headline.values())
        generated_files[f"level_{safe_level_label(label)}/seed_headline.csv"] = sha256_path(
            level_dir / "seed_headline.csv"
        )
        generated_files[
            f"level_{safe_level_label(label)}/per_surface_metrics.csv"
        ] = sha256_path(level_dir / "per_surface_metrics.csv")

    pd.DataFrame(level_rows).to_csv(root / "all_neural_seed_headline.csv", index=False)
    write_json(root / "stability_by_level.json", stability)
    if tuple(requested_levels) == (0.0,):
        _compare_zero_percent_gate(root)
    manifest = {
        "artifact_kind": "R2_NOISE_ROBUSTNESS_NEURAL_FULL_N1250",
        "levels": [
            level_label(level, protocol["noise_level_labels"])
            for level in requested_levels
        ],
        "models_and_seeds": {
            "model1": list(NEURAL_SEEDS),
            "model2_primary_P100": list(NEURAL_SEEDS),
            "excluded": ["model2_seed11_local_cpu_replication"],
        },
        "files": generated_files,
        "population": "FULL_TEST_NEURAL_N1250",
        "status": "COMPLETE",
    }
    write_json(root / "MANIFEST.json", manifest)
    return manifest


def recheck_zero_percent_gate(
    output_root: str | Path = EVIDENCE_ROOT / "zero_percent_neural",
) -> dict[str, Any]:
    """Recheck existing complete 0% evidence without rerunning models."""
    root = Path(output_root)
    required = [root / "level_0pct" / "seed_headline.csv", root / "checkpoint_provenance.json"]
    for model_kind in ("model1", "model2"):
        for seed in NEURAL_SEEDS:
            required.append(
                root
                / "level_0pct"
                / f"{model_kind}_seed{seed}_test_predictions.csv"
            )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"incomplete 0% neural evidence: {missing}")
    return _compare_zero_percent_gate(root)
