"""Phase 2B validation-only refinement around the sealed Phase 2A A9 winner."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .config import load_baseline_config
from .lambda_sweep import (
    SUMMARY_FIELDS,
    LambdaSetting,
    build_run_config,
)
from .model import DoubleHestonForwardPINN
from .synthetic_data import load_synthetic_dataset
from .trainer import seed_everything, train_baseline

DEFAULT_REFINEMENT_CONFIG = Path("configs/mentor_dh_pinn/lambda_refinement_v1.yaml")
EXPECTED_PHASE2A_WINNER = ("A9", 0.1, 0.1, 1.0, 1.0)
EXPECTED_REFINEMENT_LAMBDAS = (
    ("B0", 0.1, 0.1, 1.0, 1.0),
    ("B1", 0.05, 0.1, 1.0, 1.0),
    ("B2", 0.2, 0.1, 1.0, 1.0),
    ("B3", 0.1, 0.05, 1.0, 1.0),
    ("B4", 0.1, 0.2, 1.0, 1.0),
    ("B5", 0.05, 0.05, 1.0, 1.0),
    ("B6", 0.05, 0.2, 1.0, 1.0),
    ("B7", 0.2, 0.05, 1.0, 1.0),
    ("B8", 0.2, 0.2, 1.0, 1.0),
)
_SETTING_FIELDS = {
    "run_id", "lambda_pde", "lambda_boundary", "lambda_terminal", "lambda_data"
}


@dataclass(frozen=True)
class LambdaRefinementSpec:
    schema_version: str
    base_config_path: str
    phase1_output_dir: str
    phase2a_winner: LambdaSetting
    output_root: str
    seed: int
    max_epochs: int
    patience: int
    runs: tuple[LambdaSetting, ...]


def _strict_keys(mapping: Mapping[str, Any], expected: set[str], section: str) -> None:
    actual = set(mapping)
    if actual != expected:
        raise ValueError(
            f"invalid {section} fields: unknown={sorted(actual - expected)}, "
            f"missing={sorted(expected - actual)}"
        )


def _parse_setting(raw: Any, section: str) -> LambdaSetting:
    if not isinstance(raw, Mapping):
        raise TypeError(f"{section} must be a mapping")
    _strict_keys(raw, _SETTING_FIELDS, section)
    return LambdaSetting(
        run_id=str(raw["run_id"]),
        lambda_pde=float(raw["lambda_pde"]),
        lambda_boundary=float(raw["lambda_boundary"]),
        lambda_terminal=float(raw["lambda_terminal"]),
        lambda_data=float(raw["lambda_data"]),
    )


def _setting_tuple(setting: LambdaSetting) -> tuple[str, float, float, float, float]:
    return (
        setting.run_id,
        setting.lambda_pde,
        setting.lambda_boundary,
        setting.lambda_terminal,
        setting.lambda_data,
    )


def load_lambda_refinement_spec(
    path: str | Path = DEFAULT_REFINEMENT_CONFIG,
) -> LambdaRefinementSpec:
    """Load Phase 2B and fail closed on any winner, matrix, or control drift."""
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise TypeError("lambda refinement YAML must contain a mapping")
    top_fields = {
        "schema_version", "base_config_path", "phase1_output_dir", "phase2a_winner",
        "output_root", "seed", "max_epochs", "patience", "runs",
    }
    _strict_keys(raw, top_fields, "lambda refinement")
    if not isinstance(raw["runs"], list):
        raise TypeError("lambda refinement runs must be a list")
    winner = _parse_setting(raw["phase2a_winner"], "Phase 2A winner")
    runs = tuple(
        _parse_setting(item, f"lambda refinement run {index}")
        for index, item in enumerate(raw["runs"])
    )
    spec = LambdaRefinementSpec(
        schema_version=str(raw["schema_version"]),
        base_config_path=str(raw["base_config_path"]),
        phase1_output_dir=str(raw["phase1_output_dir"]),
        phase2a_winner=winner,
        output_root=str(raw["output_root"]),
        seed=int(raw["seed"]),
        max_epochs=int(raw["max_epochs"]),
        patience=int(raw["patience"]),
        runs=runs,
    )
    if spec.schema_version != "mentor_dh_pinn_lambda_refinement_v1":
        raise ValueError("unexpected lambda refinement schema")
    if _setting_tuple(spec.phase2a_winner) != EXPECTED_PHASE2A_WINNER:
        raise ValueError("Phase 2A winner must remain the sealed A9 setting")
    if tuple(_setting_tuple(run) for run in spec.runs) != EXPECTED_REFINEMENT_LAMBDAS:
        raise ValueError("lambda refinement run IDs or settings differ from the frozen B0-B8 matrix")
    if spec.seed != 3407 or spec.max_epochs != 300 or spec.patience != 50:
        raise ValueError("lambda refinement controls are frozen at seed=3407, epochs=300, patience=50")
    if spec.phase1_output_dir != "outputs/mentor_dh_pinn_baseline_v1":
        raise ValueError("Phase 2B must reuse the sealed Phase 1 cohort")
    if spec.output_root != "outputs/mentor_dh_pinn_lambda_refinement":
        raise ValueError("unexpected lambda refinement output root")
    if _setting_tuple(spec.runs[0])[1:] != EXPECTED_PHASE2A_WINNER[1:]:
        raise ValueError("B0 must reproduce the sealed A9 lambda setting")
    return spec


def _resolve_config(root: Path, config_path: str | Path) -> Path:
    resolved = Path(config_path)
    return resolved if resolved.is_absolute() else root / resolved


def run_lambda_refinement(
    config_path: str | Path = DEFAULT_REFINEMENT_CONFIG,
    *,
    repo_root: str | Path | None = None,
    device: str = "cpu",
    selected_run_ids: Iterable[str] | None = None,
) -> list[Path]:
    """Train selected B-runs using only the sealed Phase 1 train/validation cohort."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    spec = load_lambda_refinement_spec(_resolve_config(root, config_path))
    baseline = load_baseline_config(root / spec.base_config_path)
    dataset = load_synthetic_dataset(root / spec.phase1_output_dir, config=baseline)
    requested = set(selected_run_ids) if selected_run_ids is not None else None
    known = {run.run_id for run in spec.runs}
    if requested is not None and not requested <= known:
        raise ValueError(f"unknown refinement run IDs: {sorted(requested - known)}")
    settings = [run for run in spec.runs if requested is None or run.run_id in requested]
    written: list[Path] = []
    for setting in settings:
        output_dir = root / spec.output_root / setting.run_id
        if output_dir.exists():
            raise FileExistsError(f"refinement output already exists: {output_dir}")
        config = build_run_config(baseline, spec, setting)
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
        payload = {
            **setting.as_dict(),
            **result.as_dict(),
            "training_seconds": time.perf_counter() - started,
            "seed": spec.seed,
            "max_epochs": spec.max_epochs,
            "patience": spec.patience,
            "phase2a_winner_run_id": spec.phase2a_winner.run_id,
            "cohort_dataset_sha256": dataset.manifest["dataset_sha256"],
            "train_split_id_hash": dataset.manifest["split_id_hashes"]["train"],
            "validation_split_id_hash": dataset.manifest["split_id_hashes"]["validation"],
        }
        result_path = output_dir / "refinement_result.json"
        with result_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        written.append(result_path)
    return written


def _format_number(value: Any) -> str:
    return format(float(value), ".17g")


def summarize_lambda_refinement(
    config_path: str | Path = DEFAULT_REFINEMENT_CONFIG,
    *,
    repo_root: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Write the deterministic B0-B8 validation-only summary."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    spec = load_lambda_refinement_spec(_resolve_config(root, config_path))
    refinement_root = root / spec.output_root
    destination = (
        Path(output_path)
        if output_path is not None
        else refinement_root / "lambda_refinement_summary.csv"
    )
    rows: list[dict[str, str]] = []
    for setting in spec.runs:
        run_dir = refinement_root / setting.run_id
        with (run_dir / "refinement_result.json").open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        if any(result.get(name) != value for name, value in setting.as_dict().items()):
            raise ValueError(f"persisted settings do not match {setting.run_id}")
        with (run_dir / "train_history.csv").open("r", newline="", encoding="utf-8") as handle:
            history = list(csv.DictReader(handle))
        best_epoch = int(result["best_epoch"])
        matching = [row for row in history if int(row["epoch"]) == best_epoch]
        if len(matching) != 1:
            raise ValueError(f"best epoch is absent or duplicated for {setting.run_id}")
        best = matching[0]
        finite = str(best["finite_gradients"]).strip().lower() == "true"
        rows.append(
            {
                "run_id": setting.run_id,
                "lambda_pde": _format_number(setting.lambda_pde),
                "lambda_boundary": _format_number(setting.lambda_boundary),
                "lambda_terminal": _format_number(setting.lambda_terminal),
                "lambda_data": _format_number(setting.lambda_data),
                "best_epoch": str(best_epoch),
                "best_validation_nrmse": _format_number(result["best_validation_nrmse"]),
                "validation_rmse": _format_number(best["validation_price_rmse"]),
                "validation_mae": _format_number(best["validation_price_mae"]),
                "pde_rms": _format_number(best["pde_residual_rms"]),
                "terminal_rmse": _format_number(best["terminal_rmse"]),
                "boundary_low_rmse": _format_number(best["boundary_low_rmse"]),
                "boundary_high_rmse": _format_number(best["boundary_high_rmse"]),
                "training_seconds": _format_number(result["training_seconds"]),
                "finite_gradients": "true" if finite else "false",
            }
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return destination


__all__ = [
    "DEFAULT_REFINEMENT_CONFIG",
    "EXPECTED_PHASE2A_WINNER",
    "EXPECTED_REFINEMENT_LAMBDAS",
    "LambdaRefinementSpec",
    "load_lambda_refinement_spec",
    "run_lambda_refinement",
    "summarize_lambda_refinement",
]
