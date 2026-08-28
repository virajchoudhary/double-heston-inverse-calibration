"""Validation-only lambda sweep orchestration for the sealed Phase-1 baseline."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .config import BaselineConfig, LossConfig, TrainingConfig, load_baseline_config
from .model import DoubleHestonForwardPINN
from .synthetic_data import load_synthetic_dataset
from .trainer import seed_everything, train_baseline

DEFAULT_SWEEP_CONFIG = Path("configs/mentor_dh_pinn/lambda_sweep_v1.yaml")
SUMMARY_FIELDS = (
    "run_id",
    "lambda_pde",
    "lambda_boundary",
    "lambda_terminal",
    "lambda_data",
    "best_epoch",
    "best_validation_nrmse",
    "validation_rmse",
    "validation_mae",
    "pde_rms",
    "terminal_rmse",
    "boundary_low_rmse",
    "boundary_high_rmse",
    "training_seconds",
    "finite_gradients",
)
EXPECTED_LAMBDAS = (
    ("A1", 0.1, 1.0, 1.0, 1.0),
    ("A2", 0.01, 1.0, 1.0, 1.0),
    ("A3", 10.0, 1.0, 1.0, 1.0),
    ("A4", 1.0, 0.1, 1.0, 1.0),
    ("A5", 1.0, 10.0, 1.0, 1.0),
    ("A6", 1.0, 1.0, 0.1, 1.0),
    ("A7", 1.0, 1.0, 10.0, 1.0),
    ("A8", 0.1, 1.0, 10.0, 1.0),
    ("A9", 0.1, 0.1, 1.0, 1.0),
)


@dataclass(frozen=True)
class LambdaSetting:
    run_id: str
    lambda_pde: float
    lambda_boundary: float
    lambda_terminal: float
    lambda_data: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LambdaSweepSpec:
    schema_version: str
    base_config_path: str
    phase1_output_dir: str
    output_root: str
    seed: int
    max_epochs: int
    patience: int
    runs: tuple[LambdaSetting, ...]


@dataclass(frozen=True)
class LambdaRunConfig:
    """Duck-typed BaselineConfig view with sweep-only loss/training overrides."""

    baseline: BaselineConfig
    losses: LossConfig
    training: TrainingConfig

    def __getattr__(self, name: str) -> Any:
        return getattr(self.baseline, name)

    def validate(self) -> None:
        self.baseline.validate()
        self.training.validate()
        if any(value <= 0 for value in self.losses.weights.values()):
            raise ValueError("sweep lambdas must be strictly positive")
        if self.losses.pde_scale_floor <= 0 or self.losses.relative_error_epsilon <= 0:
            raise ValueError("loss numerical policies must remain positive")

    def to_dict(self) -> dict[str, Any]:
        result = self.baseline.to_dict()
        result["losses"] = asdict(self.losses)
        result["training"] = asdict(self.training)
        return json.loads(json.dumps(result))


def _strict_keys(mapping: Mapping[str, Any], expected: set[str], section: str) -> None:
    actual = set(mapping)
    if actual != expected:
        raise ValueError(
            f"invalid {section} fields: unknown={sorted(actual - expected)}, "
            f"missing={sorted(expected - actual)}"
        )


def load_lambda_sweep_spec(path: str | Path = DEFAULT_SWEEP_CONFIG) -> LambdaSweepSpec:
    """Load the predeclared nine-run matrix and reject any drift."""
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise TypeError("lambda sweep YAML must contain a mapping")
    top_fields = {
        "schema_version", "base_config_path", "phase1_output_dir", "output_root",
        "seed", "max_epochs", "patience", "runs",
    }
    _strict_keys(raw, top_fields, "lambda sweep")
    run_fields = {
        "run_id", "lambda_pde", "lambda_boundary", "lambda_terminal", "lambda_data"
    }
    if not isinstance(raw["runs"], list):
        raise TypeError("lambda sweep runs must be a list")
    runs: list[LambdaSetting] = []
    for index, item in enumerate(raw["runs"]):
        if not isinstance(item, Mapping):
            raise TypeError(f"lambda sweep run {index} must be a mapping")
        _strict_keys(item, run_fields, f"lambda sweep run {index}")
        runs.append(
            LambdaSetting(
                run_id=str(item["run_id"]),
                lambda_pde=float(item["lambda_pde"]),
                lambda_boundary=float(item["lambda_boundary"]),
                lambda_terminal=float(item["lambda_terminal"]),
                lambda_data=float(item["lambda_data"]),
            )
        )
    observed = tuple(
        (run.run_id, run.lambda_pde, run.lambda_boundary, run.lambda_terminal, run.lambda_data)
        for run in runs
    )
    if observed != EXPECTED_LAMBDAS:
        raise ValueError("lambda sweep run IDs or settings differ from the predeclared A1-A9 matrix")
    spec = LambdaSweepSpec(
        schema_version=str(raw["schema_version"]),
        base_config_path=str(raw["base_config_path"]),
        phase1_output_dir=str(raw["phase1_output_dir"]),
        output_root=str(raw["output_root"]),
        seed=int(raw["seed"]),
        max_epochs=int(raw["max_epochs"]),
        patience=int(raw["patience"]),
        runs=tuple(runs),
    )
    if spec.schema_version != "mentor_dh_pinn_lambda_sweep_v1":
        raise ValueError("unexpected lambda sweep schema")
    if spec.seed != 3407 or spec.max_epochs != 300 or spec.patience != 50:
        raise ValueError("lambda sweep screening controls are frozen at seed=3407, epochs=300, patience=50")
    if spec.output_root != "outputs/mentor_dh_pinn_lambda_sweep":
        raise ValueError("unexpected lambda sweep output root")
    return spec


def build_run_config(
    baseline: BaselineConfig,
    spec: LambdaSweepSpec,
    setting: LambdaSetting,
) -> LambdaRunConfig:
    if baseline.seed != spec.seed:
        raise ValueError("sweep seed must match the Phase-1 baseline seed")
    losses = replace(
        baseline.losses,
        pde_lambda=setting.lambda_pde,
        boundary_lambda=setting.lambda_boundary,
        terminal_lambda=setting.lambda_terminal,
        data_lambda=setting.lambda_data,
    )
    training = replace(
        baseline.training,
        max_epochs=spec.max_epochs,
        patience=spec.patience,
    )
    result = LambdaRunConfig(baseline=baseline, losses=losses, training=training)
    result.validate()
    return result


def run_lambda_sweep(
    config_path: str | Path = DEFAULT_SWEEP_CONFIG,
    *,
    repo_root: str | Path | None = None,
    device: str = "cpu",
    selected_run_ids: Iterable[str] | None = None,
) -> list[Path]:
    """Train selected sweep runs without importing or invoking the test evaluator."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    resolved_config = Path(config_path)
    if not resolved_config.is_absolute():
        resolved_config = root / resolved_config
    spec = load_lambda_sweep_spec(resolved_config)
    baseline = load_baseline_config(root / spec.base_config_path)
    dataset = load_synthetic_dataset(root / spec.phase1_output_dir, config=baseline)
    requested = set(selected_run_ids) if selected_run_ids is not None else None
    known = {run.run_id for run in spec.runs}
    if requested is not None and not requested <= known:
        raise ValueError(f"unknown sweep run IDs: {sorted(requested - known)}")
    settings = [run for run in spec.runs if requested is None or run.run_id in requested]
    written: list[Path] = []
    for setting in settings:
        output_dir = root / spec.output_root / setting.run_id
        if output_dir.exists():
            raise FileExistsError(f"sweep output already exists: {output_dir}")
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
        training_seconds = time.perf_counter() - started
        payload = {
            **setting.as_dict(),
            **result.as_dict(),
            "training_seconds": training_seconds,
            "seed": spec.seed,
            "max_epochs": spec.max_epochs,
            "patience": spec.patience,
            "cohort_dataset_sha256": dataset.manifest["dataset_sha256"],
            "train_split_id_hash": dataset.manifest["split_id_hashes"]["train"],
            "validation_split_id_hash": dataset.manifest["split_id_hashes"]["validation"],
        }
        result_path = output_dir / "sweep_result.json"
        with result_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        written.append(result_path)
    return written


def _format_number(value: Any) -> str:
    return format(float(value), ".17g")


def summarize_lambda_sweep(
    config_path: str | Path = DEFAULT_SWEEP_CONFIG,
    *,
    repo_root: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Create a deterministic A1-A9 validation-only summary table."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    resolved_config = Path(config_path)
    if not resolved_config.is_absolute():
        resolved_config = root / resolved_config
    spec = load_lambda_sweep_spec(resolved_config)
    sweep_root = root / spec.output_root
    destination = Path(output_path) if output_path is not None else sweep_root / "lambda_summary.csv"
    rows: list[dict[str, str]] = []
    for setting in spec.runs:
        run_dir = sweep_root / setting.run_id
        with (run_dir / "sweep_result.json").open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        expected = setting.as_dict()
        if any(result.get(name) != value for name, value in expected.items()):
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
    "DEFAULT_SWEEP_CONFIG",
    "EXPECTED_LAMBDAS",
    "SUMMARY_FIELDS",
    "LambdaRunConfig",
    "LambdaSetting",
    "LambdaSweepSpec",
    "build_run_config",
    "load_lambda_sweep_spec",
    "run_lambda_sweep",
    "summarize_lambda_sweep",
]
