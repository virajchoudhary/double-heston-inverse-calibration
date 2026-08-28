"""Sealed Phase 2C validation-only confirmation of the Phase 2B winner."""

from __future__ import annotations

import csv
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from .config import BaselineConfig, load_baseline_config
from .lambda_sweep import LambdaSetting, build_run_config
from .model import DoubleHestonForwardPINN
from .synthetic_data import load_synthetic_dataset
from .trainer import seed_everything, train_baseline

DEFAULT_CONFIRMATION_CONFIG = Path(
    "configs/mentor_dh_pinn/full_horizon_confirmation_v1.yaml"
)
EXPECTED_CANDIDATE = ("C1", 0.1, 0.1, 1.0, 1.0)
EXPECTED_PHASE1_REFERENCE = (
    "phase1_equal_weight",
    1.0,
    1.0,
    1.0,
    1.0,
    909,
    0.0012545129490993494,
)
COMPARISON_FIELDS = (
    "run_id",
    "c1_best_validation_nrmse",
    "phase1_best_epoch",
    "phase1_best_validation_nrmse",
    "absolute_difference",
    "percentage_improvement",
    "comparison",
    "validation_rmse",
    "validation_mae",
    "pde_rms",
    "terminal_rmse",
    "boundary_low_rmse",
    "boundary_high_rmse",
)
_SETTING_FIELDS = {
    "run_id", "lambda_pde", "lambda_boundary", "lambda_terminal", "lambda_data"
}
_REFERENCE_FIELDS = _SETTING_FIELDS | {"best_epoch", "best_validation_nrmse"}


@dataclass(frozen=True)
class Phase1Reference:
    run_id: str
    lambda_pde: float
    lambda_boundary: float
    lambda_terminal: float
    lambda_data: float
    best_epoch: int
    best_validation_nrmse: float

    @property
    def weights(self) -> dict[str, float]:
        return {
            "pde": self.lambda_pde,
            "boundary": self.lambda_boundary,
            "terminal": self.lambda_terminal,
            "data": self.lambda_data,
        }


@dataclass(frozen=True)
class FullHorizonConfirmationSpec:
    schema_version: str
    base_config_path: str
    phase1_output_dir: str
    phase1_checkpoint_path: str
    output_root: str
    comparison_summary_filename: str
    seed: int
    max_epochs: int
    patience: int
    candidate: LambdaSetting
    phase1_reference: Phase1Reference


def _strict_keys(mapping: Mapping[str, Any], expected: set[str], section: str) -> None:
    actual = set(mapping)
    if actual != expected:
        raise ValueError(
            f"invalid {section} fields: unknown={sorted(actual - expected)}, "
            f"missing={sorted(expected - actual)}"
        )


def _parse_setting(raw: Any) -> LambdaSetting:
    if not isinstance(raw, Mapping):
        raise TypeError("Phase 2C candidate must be a mapping")
    _strict_keys(raw, _SETTING_FIELDS, "Phase 2C candidate")
    return LambdaSetting(
        run_id=str(raw["run_id"]),
        lambda_pde=float(raw["lambda_pde"]),
        lambda_boundary=float(raw["lambda_boundary"]),
        lambda_terminal=float(raw["lambda_terminal"]),
        lambda_data=float(raw["lambda_data"]),
    )


def _parse_reference(raw: Any) -> Phase1Reference:
    if not isinstance(raw, Mapping):
        raise TypeError("Phase 1 reference must be a mapping")
    _strict_keys(raw, _REFERENCE_FIELDS, "Phase 1 reference")
    return Phase1Reference(
        run_id=str(raw["run_id"]),
        lambda_pde=float(raw["lambda_pde"]),
        lambda_boundary=float(raw["lambda_boundary"]),
        lambda_terminal=float(raw["lambda_terminal"]),
        lambda_data=float(raw["lambda_data"]),
        best_epoch=int(raw["best_epoch"]),
        best_validation_nrmse=float(raw["best_validation_nrmse"]),
    )


def _candidate_tuple(setting: LambdaSetting) -> tuple[str, float, float, float, float]:
    return (
        setting.run_id,
        setting.lambda_pde,
        setting.lambda_boundary,
        setting.lambda_terminal,
        setting.lambda_data,
    )


def _reference_tuple(reference: Phase1Reference) -> tuple[Any, ...]:
    return (
        reference.run_id,
        reference.lambda_pde,
        reference.lambda_boundary,
        reference.lambda_terminal,
        reference.lambda_data,
        reference.best_epoch,
        reference.best_validation_nrmse,
    )


def load_full_horizon_confirmation_spec(
    path: str | Path = DEFAULT_CONFIRMATION_CONFIG,
) -> FullHorizonConfirmationSpec:
    """Load Phase 2C and reject any candidate, control, or reference drift."""
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise TypeError("full-horizon confirmation YAML must contain a mapping")
    top_fields = {
        "schema_version", "base_config_path", "phase1_output_dir",
        "phase1_checkpoint_path", "output_root", "comparison_summary_filename",
        "seed", "max_epochs", "patience", "candidate", "phase1_reference",
    }
    _strict_keys(raw, top_fields, "full-horizon confirmation")
    spec = FullHorizonConfirmationSpec(
        schema_version=str(raw["schema_version"]),
        base_config_path=str(raw["base_config_path"]),
        phase1_output_dir=str(raw["phase1_output_dir"]),
        phase1_checkpoint_path=str(raw["phase1_checkpoint_path"]),
        output_root=str(raw["output_root"]),
        comparison_summary_filename=str(raw["comparison_summary_filename"]),
        seed=int(raw["seed"]),
        max_epochs=int(raw["max_epochs"]),
        patience=int(raw["patience"]),
        candidate=_parse_setting(raw["candidate"]),
        phase1_reference=_parse_reference(raw["phase1_reference"]),
    )
    if spec.schema_version != "mentor_dh_pinn_full_horizon_confirmation_v1":
        raise ValueError("unexpected full-horizon confirmation schema")
    if _candidate_tuple(spec.candidate) != EXPECTED_CANDIDATE:
        raise ValueError("Phase 2C candidate must remain the frozen C1 setting")
    if _reference_tuple(spec.phase1_reference) != EXPECTED_PHASE1_REFERENCE:
        raise ValueError("Phase 1 full-horizon reference has drifted")
    if (spec.seed, spec.max_epochs, spec.patience) != (3407, 1000, 100):
        raise ValueError("Phase 2C controls are frozen at seed=3407, epochs=1000, patience=100")
    if spec.base_config_path != "configs/mentor_dh_pinn/baseline_v1.yaml":
        raise ValueError("Phase 2C must use the sealed Phase 1 baseline config")
    if spec.phase1_output_dir != "outputs/mentor_dh_pinn_baseline_v1":
        raise ValueError("Phase 2C must reuse the sealed Phase 1 cohort")
    if spec.phase1_checkpoint_path != "outputs/mentor_dh_pinn_baseline_v1/checkpoint.pt":
        raise ValueError("Phase 2C must cross-check the sealed Phase 1 checkpoint")
    if spec.output_root != "outputs/mentor_dh_pinn_full_horizon_confirmation":
        raise ValueError("unexpected Phase 2C output root")
    if spec.comparison_summary_filename != "full_horizon_comparison.csv":
        raise ValueError("unexpected Phase 2C comparison summary filename")
    return spec


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _load_phase1_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _cohort_identity(dataset: Any) -> dict[str, str]:
    manifest = dataset.manifest
    split_hashes = manifest.get("split_id_hashes", {})
    return {
        "cohort_dataset_sha256": str(manifest.get("dataset_sha256", "")),
        "train_split_id_hash": str(split_hashes.get("train", "")),
        "validation_split_id_hash": str(split_hashes.get("validation", "")),
    }


def _validate_phase1_anchor(
    checkpoint: Mapping[str, Any],
    dataset: Any,
    baseline: BaselineConfig,
    reference: Phase1Reference,
) -> dict[str, str]:
    """Bind C1 to the sealed Phase 1 protocol, metric, and cohort identities."""
    if checkpoint.get("config") != baseline.to_dict():
        raise ValueError("Phase 1 checkpoint config identity mismatch")
    if checkpoint.get("weights") != reference.weights:
        raise ValueError("Phase 1 checkpoint equal-weight identity mismatch")
    if checkpoint.get("best_epoch") != reference.best_epoch:
        raise ValueError("Phase 1 checkpoint best epoch mismatch")
    if checkpoint.get("best_validation_nrmse") != reference.best_validation_nrmse:
        raise ValueError("Phase 1 checkpoint validation nRMSE mismatch")
    identity = _cohort_identity(dataset)
    if not all(identity.values()):
        raise ValueError("Phase 1 cohort identity is incomplete")
    checkpoint_dataset = checkpoint.get("dataset_identity")
    if not isinstance(checkpoint_dataset, Mapping):
        raise ValueError("Phase 1 checkpoint dataset identity is missing")
    checkpoint_hashes = checkpoint_dataset.get("split_id_hashes", {})
    expected = {
        "cohort_dataset_sha256": str(checkpoint_dataset.get("dataset_sha256", "")),
        "train_split_id_hash": str(checkpoint_hashes.get("train", "")),
        "validation_split_id_hash": str(checkpoint_hashes.get("validation", "")),
    }
    if identity != expected:
        raise ValueError("Phase 1 cohort dataset or train/validation split identity mismatch")
    return identity


def _git_sha(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    if not value:
        raise RuntimeError("git SHA is unavailable")
    return value


def _selected_epoch_metrics(history_path: Path, best_epoch: int) -> dict[str, Any]:
    with history_path.open("r", newline="", encoding="utf-8") as handle:
        matches = [row for row in csv.DictReader(handle) if int(row["epoch"]) == best_epoch]
    if len(matches) != 1:
        raise ValueError("C1 best epoch is absent or duplicated in train history")
    row = matches[0]
    finite = str(row["finite_gradients"]).strip().lower() == "true"
    if not finite:
        raise ValueError("C1 selected epoch reports non-finite gradients")
    return {
        "validation_rmse": float(row["validation_price_rmse"]),
        "validation_mae": float(row["validation_price_mae"]),
        "pde_rms": float(row["pde_residual_rms"]),
        "terminal_rmse": float(row["terminal_rmse"]),
        "boundary_low_rmse": float(row["boundary_low_rmse"]),
        "boundary_high_rmse": float(row["boundary_high_rmse"]),
        "finite_gradients": True,
    }


def run_full_horizon_confirmation(
    config_path: str | Path = DEFAULT_CONFIRMATION_CONFIG,
    *,
    repo_root: str | Path | None = None,
    device: str = "cpu",
) -> Path:
    """Train C1 using validation only; this module never imports the test evaluator."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    spec = load_full_horizon_confirmation_spec(_resolve(root, config_path))
    baseline = load_baseline_config(root / spec.base_config_path)
    dataset = load_synthetic_dataset(root / spec.phase1_output_dir, config=baseline)
    phase1_checkpoint = _load_phase1_checkpoint(root / spec.phase1_checkpoint_path)
    cohort_identity = _validate_phase1_anchor(
        phase1_checkpoint, dataset, baseline, spec.phase1_reference
    )
    output_dir = root / spec.output_root / spec.candidate.run_id
    result_path = output_dir / "confirmation_result.json"
    if result_path.exists():
        raise FileExistsError(f"completed Phase 2C run already exists: {result_path}")
    config = build_run_config(baseline, spec, spec.candidate)
    seed_everything(spec.seed)
    model = DoubleHestonForwardPINN(
        feature_min=baseline.domain.feature_min,
        feature_max=baseline.domain.feature_max,
    )
    started = time.perf_counter()
    result = train_baseline(
        model,
        dataset,
        output_dir,
        config=config,
        cohort_config=baseline,
        repo_root=root,
        device=device,
    )
    selected_metrics = _selected_epoch_metrics(result.train_history_path, result.best_epoch)
    payload = {
        **spec.candidate.as_dict(),
        **result.as_dict(),
        **selected_metrics,
        **cohort_identity,
        "training_seconds": time.perf_counter() - started,
        "seed": spec.seed,
        "max_epochs": spec.max_epochs,
        "patience": spec.patience,
        "git_sha": _git_sha(root),
        "phase1_best_epoch": spec.phase1_reference.best_epoch,
        "phase1_best_validation_nrmse": spec.phase1_reference.best_validation_nrmse,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result_path


def _format_number(value: Any) -> str:
    return format(float(value), ".17g")


def summarize_full_horizon_confirmation(
    config_path: str | Path = DEFAULT_CONFIRMATION_CONFIG,
    *,
    repo_root: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Write a deterministic validation-only C1 versus Phase 1 comparison."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    spec = load_full_horizon_confirmation_spec(_resolve(root, config_path))
    confirmation_root = root / spec.output_root
    result_path = confirmation_root / spec.candidate.run_id / "confirmation_result.json"
    with result_path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    for name, value in spec.candidate.as_dict().items():
        if result.get(name) != value:
            raise ValueError("persisted C1 lambda setting does not match the sealed candidate")
    if result.get("phase1_best_epoch") != spec.phase1_reference.best_epoch:
        raise ValueError("persisted Phase 1 best epoch has drifted")
    if result.get("phase1_best_validation_nrmse") != spec.phase1_reference.best_validation_nrmse:
        raise ValueError("persisted Phase 1 validation nRMSE has drifted")
    c1_metric = float(result["best_validation_nrmse"])
    phase1_metric = spec.phase1_reference.best_validation_nrmse
    signed_difference = c1_metric - phase1_metric
    comparison = "improvement" if signed_difference < 0 else (
        "degradation" if signed_difference > 0 else "equal"
    )
    row = {
        "run_id": spec.candidate.run_id,
        "c1_best_validation_nrmse": _format_number(c1_metric),
        "phase1_best_epoch": str(spec.phase1_reference.best_epoch),
        "phase1_best_validation_nrmse": _format_number(phase1_metric),
        "absolute_difference": _format_number(abs(signed_difference)),
        "percentage_improvement": _format_number(
            100.0 * (phase1_metric - c1_metric) / phase1_metric
        ),
        "comparison": comparison,
        "validation_rmse": _format_number(result["validation_rmse"]),
        "validation_mae": _format_number(result["validation_mae"]),
        "pde_rms": _format_number(result["pde_rms"]),
        "terminal_rmse": _format_number(result["terminal_rmse"]),
        "boundary_low_rmse": _format_number(result["boundary_low_rmse"]),
        "boundary_high_rmse": _format_number(result["boundary_high_rmse"]),
    }
    destination = (
        Path(output_path)
        if output_path is not None
        else confirmation_root / spec.comparison_summary_filename
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
    return destination


__all__ = [
    "COMPARISON_FIELDS",
    "DEFAULT_CONFIRMATION_CONFIG",
    "EXPECTED_CANDIDATE",
    "EXPECTED_PHASE1_REFERENCE",
    "FullHorizonConfirmationSpec",
    "load_full_horizon_confirmation_spec",
    "run_full_horizon_confirmation",
    "summarize_full_horizon_confirmation",
]
