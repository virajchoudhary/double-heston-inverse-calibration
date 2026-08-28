"""Phase 2D paired multi-seed validation-only confirmation."""

from __future__ import annotations

import csv
import json
import math
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import yaml

from .config import BaselineConfig, LossConfig, TrainingConfig, load_baseline_config
from .model import DoubleHestonForwardPINN
from .synthetic_data import load_synthetic_dataset
from .trainer import seed_everything, train_baseline

DEFAULT_MULTISEED_CONFIG = Path("configs/mentor_dh_pinn/multiseed_confirmation_v1.yaml")
EXPECTED_COHORT_DATASET_SHA256 = (
    "35a813036fe50eddcfe66db132b86d2dbfba01b4c99e341b8e5518c04d0f1ad4"
)
EXPECTED_SEEDS = (11, 22, 33)
EQUAL_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
OPTIMIZED_WEIGHTS = (0.1, 0.1, 1.0, 1.0)
EXPECTED_RUN_MATRIX = (
    ("EQ11", "equal", 11, *EQUAL_WEIGHTS),
    ("OPT11", "optimized", 11, *OPTIMIZED_WEIGHTS),
    ("EQ22", "equal", 22, *EQUAL_WEIGHTS),
    ("OPT22", "optimized", 22, *OPTIMIZED_WEIGHTS),
    ("EQ33", "equal", 33, *EQUAL_WEIGHTS),
    ("OPT33", "optimized", 33, *OPTIMIZED_WEIGHTS),
)
EXPECTED_PHASE1_REFERENCE = (909, 0.0012545129490993494)

RESULT_FIELDS = (
    "run_id",
    "variant",
    "seed",
    "lambda_pde",
    "lambda_boundary",
    "lambda_terminal",
    "lambda_data",
    "best_epoch",
    "epochs_completed",
    "best_validation_nrmse",
    "validation_rmse",
    "validation_mae",
    "pde_rms",
    "terminal_rmse",
    "boundary_low_rmse",
    "boundary_high_rmse",
    "finite_gradients",
    "training_seconds",
    "cohort_dataset_sha256",
    "train_split_id_hash",
    "validation_split_id_hash",
    "git_sha",
)
RUN_SUMMARY_FIELDS = RESULT_FIELDS
PAIR_SUMMARY_FIELDS = (
    "seed",
    "equal_validation_nrmse",
    "optimized_validation_nrmse",
    "absolute_difference",
    "percentage_improvement",
    "optimized_wins",
    "equal_pde_rms",
    "optimized_pde_rms",
    "pde_ratio_opt_vs_equal",
    "physics_gate_pass",
)
_RUN_FIELDS = {
    "run_id", "variant", "seed", "lambda_pde", "lambda_boundary",
    "lambda_terminal", "lambda_data",
}
_REFERENCE_FIELDS = {"best_epoch", "best_validation_nrmse"}
_NUMERIC_RESULT_FIELDS = {
    "best_validation_nrmse", "validation_rmse", "validation_mae", "pde_rms",
    "terminal_rmse", "boundary_low_rmse", "boundary_high_rmse", "training_seconds",
}


@dataclass(frozen=True)
class MultiseedRunSetting:
    run_id: str
    variant: str
    seed: int
    lambda_pde: float
    lambda_boundary: float
    lambda_terminal: float
    lambda_data: float

    @property
    def weights(self) -> dict[str, float]:
        return {
            "pde": self.lambda_pde,
            "boundary": self.lambda_boundary,
            "terminal": self.lambda_terminal,
            "data": self.lambda_data,
        }

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Phase1Reference:
    best_epoch: int
    best_validation_nrmse: float


@dataclass(frozen=True)
class MultiseedConfirmationSpec:
    schema_version: str
    base_config_path: str
    phase1_output_dir: str
    phase1_checkpoint_path: str
    expected_cohort_dataset_sha256: str
    output_root: str
    run_summary_filename: str
    pair_summary_filename: str
    aggregate_summary_filename: str
    max_epochs: int
    patience: int
    phase1_reference: Phase1Reference
    seeds: tuple[int, ...]
    runs: tuple[MultiseedRunSetting, ...]


@dataclass(frozen=True)
class MultiseedRunConfig:
    """Baseline view that changes only seed, loss weights, and frozen horizon controls."""

    baseline: BaselineConfig
    seed: int
    losses: LossConfig
    training: TrainingConfig

    def __getattr__(self, name: str) -> Any:
        return getattr(self.baseline, name)

    def validate(self) -> None:
        self.baseline.validate()
        self.training.validate()
        if self.seed not in EXPECTED_SEEDS:
            raise ValueError("Phase 2D seed is outside the frozen confirmation set")
        if self.training != self.baseline.training:
            raise ValueError("Phase 2D training controls must match the sealed full-horizon baseline")
        if any(value <= 0 for value in self.losses.weights.values()):
            raise ValueError("Phase 2D lambdas must be strictly positive")

    def to_dict(self) -> dict[str, Any]:
        result = self.baseline.to_dict()
        result["seed"] = self.seed
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


def _parse_run(raw: Any, index: int) -> MultiseedRunSetting:
    if not isinstance(raw, Mapping):
        raise TypeError(f"Phase 2D run {index} must be a mapping")
    _strict_keys(raw, _RUN_FIELDS, f"Phase 2D run {index}")
    return MultiseedRunSetting(
        run_id=str(raw["run_id"]),
        variant=str(raw["variant"]),
        seed=int(raw["seed"]),
        lambda_pde=float(raw["lambda_pde"]),
        lambda_boundary=float(raw["lambda_boundary"]),
        lambda_terminal=float(raw["lambda_terminal"]),
        lambda_data=float(raw["lambda_data"]),
    )


def _run_tuple(run: MultiseedRunSetting) -> tuple[Any, ...]:
    return (
        run.run_id,
        run.variant,
        run.seed,
        run.lambda_pde,
        run.lambda_boundary,
        run.lambda_terminal,
        run.lambda_data,
    )


def load_multiseed_confirmation_spec(
    path: str | Path = DEFAULT_MULTISEED_CONFIG,
) -> MultiseedConfirmationSpec:
    """Load Phase 2D and fail closed on any matrix, seed, or control drift."""
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise TypeError("Phase 2D YAML must contain a mapping")
    top_fields = {
        "schema_version", "base_config_path", "phase1_output_dir",
        "phase1_checkpoint_path", "expected_cohort_dataset_sha256", "output_root",
        "run_summary_filename", "pair_summary_filename", "aggregate_summary_filename",
        "max_epochs", "patience", "phase1_reference", "seeds", "runs",
    }
    _strict_keys(raw, top_fields, "Phase 2D")
    reference_raw = raw["phase1_reference"]
    if not isinstance(reference_raw, Mapping):
        raise TypeError("Phase 1 reference must be a mapping")
    _strict_keys(reference_raw, _REFERENCE_FIELDS, "Phase 1 reference")
    if not isinstance(raw["seeds"], list) or not isinstance(raw["runs"], list):
        raise TypeError("Phase 2D seeds and runs must be lists")
    spec = MultiseedConfirmationSpec(
        schema_version=str(raw["schema_version"]),
        base_config_path=str(raw["base_config_path"]),
        phase1_output_dir=str(raw["phase1_output_dir"]),
        phase1_checkpoint_path=str(raw["phase1_checkpoint_path"]),
        expected_cohort_dataset_sha256=str(raw["expected_cohort_dataset_sha256"]),
        output_root=str(raw["output_root"]),
        run_summary_filename=str(raw["run_summary_filename"]),
        pair_summary_filename=str(raw["pair_summary_filename"]),
        aggregate_summary_filename=str(raw["aggregate_summary_filename"]),
        max_epochs=int(raw["max_epochs"]),
        patience=int(raw["patience"]),
        phase1_reference=Phase1Reference(
            best_epoch=int(reference_raw["best_epoch"]),
            best_validation_nrmse=float(reference_raw["best_validation_nrmse"]),
        ),
        seeds=tuple(int(seed) for seed in raw["seeds"]),
        runs=tuple(_parse_run(item, index) for index, item in enumerate(raw["runs"])),
    )
    if spec.schema_version != "mentor_dh_pinn_multiseed_confirmation_v1":
        raise ValueError("unexpected Phase 2D schema")
    if spec.seeds != EXPECTED_SEEDS or 3407 in spec.seeds:
        raise ValueError("Phase 2D seeds are frozen at 11, 22, and 33; seed 3407 is forbidden")
    if tuple(_run_tuple(run) for run in spec.runs) != EXPECTED_RUN_MATRIX:
        raise ValueError("Phase 2D run IDs, variants, seeds, or lambdas differ from the frozen matrix")
    if (spec.max_epochs, spec.patience) != (1000, 100):
        raise ValueError("Phase 2D controls are frozen at max_epochs=1000 and patience=100")
    if spec.expected_cohort_dataset_sha256 != EXPECTED_COHORT_DATASET_SHA256:
        raise ValueError("Phase 2D expected cohort dataset SHA has drifted")
    if (
        spec.phase1_reference.best_epoch,
        spec.phase1_reference.best_validation_nrmse,
    ) != EXPECTED_PHASE1_REFERENCE:
        raise ValueError("Phase 1 full-horizon checkpoint anchor has drifted")
    expected_paths = {
        "base_config_path": "configs/mentor_dh_pinn/baseline_v1.yaml",
        "phase1_output_dir": "outputs/mentor_dh_pinn_baseline_v1",
        "phase1_checkpoint_path": "outputs/mentor_dh_pinn_baseline_v1/checkpoint.pt",
        "output_root": "outputs/mentor_dh_pinn_multiseed_confirmation",
        "run_summary_filename": "multiseed_run_summary.csv",
        "pair_summary_filename": "multiseed_pair_summary.csv",
        "aggregate_summary_filename": "multiseed_aggregate_summary.json",
    }
    for name, expected in expected_paths.items():
        if getattr(spec, name) != expected:
            raise ValueError(f"unexpected Phase 2D {name}")
    return spec


def build_multiseed_run_config(
    baseline: BaselineConfig,
    spec: MultiseedConfirmationSpec,
    setting: MultiseedRunSetting,
) -> MultiseedRunConfig:
    """Change only the fresh seed and predeclared lambda variant."""
    if baseline.training.max_epochs != spec.max_epochs or baseline.training.patience != spec.patience:
        raise ValueError("Phase 2D horizon controls must equal the sealed Phase 1 baseline")
    losses = replace(
        baseline.losses,
        pde_lambda=setting.lambda_pde,
        boundary_lambda=setting.lambda_boundary,
        terminal_lambda=setting.lambda_terminal,
        data_lambda=setting.lambda_data,
    )
    result = MultiseedRunConfig(
        baseline=baseline,
        seed=setting.seed,
        losses=losses,
        training=baseline.training,
    )
    result.validate()
    return result


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
    spec: MultiseedConfirmationSpec,
) -> dict[str, str]:
    if checkpoint.get("config") != baseline.to_dict():
        raise ValueError("Phase 1 checkpoint config identity mismatch")
    if checkpoint.get("weights") != {"pde": 1.0, "boundary": 1.0, "terminal": 1.0, "data": 1.0}:
        raise ValueError("Phase 1 checkpoint equal-weight identity mismatch")
    if checkpoint.get("best_epoch") != spec.phase1_reference.best_epoch:
        raise ValueError("Phase 1 checkpoint best epoch mismatch")
    if checkpoint.get("best_validation_nrmse") != spec.phase1_reference.best_validation_nrmse:
        raise ValueError("Phase 1 checkpoint validation nRMSE mismatch")
    identity = _cohort_identity(dataset)
    if identity["cohort_dataset_sha256"] != spec.expected_cohort_dataset_sha256:
        raise ValueError("Phase 1 cohort dataset SHA mismatch")
    if not identity["train_split_id_hash"] or not identity["validation_split_id_hash"]:
        raise ValueError("Phase 1 train/validation split identity is incomplete")
    checkpoint_dataset = checkpoint.get("dataset_identity")
    if not isinstance(checkpoint_dataset, Mapping):
        raise ValueError("Phase 1 checkpoint dataset identity is missing")
    checkpoint_hashes = checkpoint_dataset.get("split_id_hashes", {})
    checkpoint_identity = {
        "cohort_dataset_sha256": str(checkpoint_dataset.get("dataset_sha256", "")),
        "train_split_id_hash": str(checkpoint_hashes.get("train", "")),
        "validation_split_id_hash": str(checkpoint_hashes.get("validation", "")),
    }
    if identity != checkpoint_identity:
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
        raise ValueError("selected epoch is absent or duplicated in train history")
    row = matches[0]
    finite = str(row["finite_gradients"]).strip().lower() == "true"
    if not finite:
        raise ValueError("selected epoch reports non-finite gradients")
    return {
        "validation_rmse": float(row["validation_price_rmse"]),
        "validation_mae": float(row["validation_price_mae"]),
        "pde_rms": float(row["pde_residual_rms"]),
        "terminal_rmse": float(row["terminal_rmse"]),
        "boundary_low_rmse": float(row["boundary_low_rmse"]),
        "boundary_high_rmse": float(row["boundary_high_rmse"]),
        "finite_gradients": True,
    }


def run_multiseed_confirmation(
    config_path: str | Path = DEFAULT_MULTISEED_CONFIG,
    *,
    repo_root: str | Path | None = None,
    device: str = "cpu",
    selected_run_ids: Iterable[str] | None = None,
) -> list[Path]:
    """Train selected Phase 2D runs without importing or invoking the test evaluator."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    spec = load_multiseed_confirmation_spec(_resolve(root, config_path))
    requested = set(selected_run_ids) if selected_run_ids is not None else None
    known = {run.run_id for run in spec.runs}
    if requested is not None and not requested <= known:
        raise ValueError(f"unknown Phase 2D run IDs: {sorted(requested - known)}")
    settings = [run for run in spec.runs if requested is None or run.run_id in requested]
    for setting in settings:
        run_dir = root / spec.output_root / setting.run_id
        result_path = run_dir / "multiseed_result.json"
        if result_path.exists():
            raise FileExistsError(f"completed Phase 2D run already exists: {result_path}")
        if run_dir.exists():
            raise FileExistsError(
                f"partial Phase 2D run directory exists and requires explicit recovery: {run_dir}"
            )
    baseline = load_baseline_config(root / spec.base_config_path)
    dataset = load_synthetic_dataset(root / spec.phase1_output_dir, config=baseline)
    phase1_checkpoint = _load_phase1_checkpoint(root / spec.phase1_checkpoint_path)
    cohort_identity = _validate_phase1_anchor(phase1_checkpoint, dataset, baseline, spec)
    written: list[Path] = []
    for setting in settings:
        output_dir = root / spec.output_root / setting.run_id
        config = build_multiseed_run_config(baseline, spec, setting)
        seed_everything(setting.seed)
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
            **setting.as_dict(),
            **result.as_dict(),
            **selected_metrics,
            **cohort_identity,
            "training_seconds": time.perf_counter() - started,
            "git_sha": _git_sha(root),
        }
        result_path = output_dir / "multiseed_result.json"
        with result_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        written.append(result_path)
    return written


def _format_number(value: Any) -> str:
    return format(float(value), ".17g")


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _load_complete_result(
    run_dir: Path,
    setting: MultiseedRunSetting,
    expected_dataset_sha256: str,
) -> dict[str, Any]:
    for filename in ("checkpoint.pt", "train_history.csv", "validation_history.csv", "multiseed_result.json"):
        if not (run_dir / filename).is_file():
            raise ValueError(f"incomplete Phase 2D result for {setting.run_id}: missing {filename}")
    with (run_dir / "multiseed_result.json").open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Phase 2D result for {setting.run_id} must be a mapping")
    missing = set(RESULT_FIELDS) - set(payload)
    if missing:
        raise ValueError(f"incomplete Phase 2D result for {setting.run_id}: missing {sorted(missing)}")
    if any(payload.get(name) != value for name, value in setting.as_dict().items()):
        raise ValueError(f"persisted run identity differs from frozen setting {setting.run_id}")
    if payload["cohort_dataset_sha256"] != expected_dataset_sha256:
        raise ValueError(f"cohort dataset identity drift for {setting.run_id}")
    if not payload["train_split_id_hash"] or not payload["validation_split_id_hash"]:
        raise ValueError(f"split identity is incomplete for {setting.run_id}")
    if not payload["git_sha"] or payload["git_sha"] == "unknown":
        raise ValueError(f"git identity is incomplete for {setting.run_id}")
    if type(payload["finite_gradients"]) is not bool:
        raise ValueError(f"finite-gradient status is malformed for {setting.run_id}")
    if type(payload["best_epoch"]) is not int or type(payload["epochs_completed"]) is not int:
        raise ValueError(f"epoch fields are malformed for {setting.run_id}")
    if payload["best_epoch"] <= 0 or payload["epochs_completed"] < payload["best_epoch"]:
        raise ValueError(f"epoch fields are inconsistent for {setting.run_id}")
    for name in _NUMERIC_RESULT_FIELDS:
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"non-finite or malformed {name} for {setting.run_id}")
        if float(value) < 0:
            raise ValueError(f"negative {name} for {setting.run_id}")
    return dict(payload)


def _run_summary_row(payload: Mapping[str, Any]) -> dict[str, str]:
    row: dict[str, str] = {}
    for field in RUN_SUMMARY_FIELDS:
        value = payload[field]
        if field == "finite_gradients":
            row[field] = _format_bool(bool(value))
        elif field in {"seed", "best_epoch", "epochs_completed"}:
            row[field] = str(int(value))
        elif field.startswith("lambda_") or field in _NUMERIC_RESULT_FIELDS:
            row[field] = _format_number(value)
        else:
            row[field] = str(value)
    return row


def _pde_ratio(optimized: float, equal: float) -> float | str:
    if equal > 0:
        return optimized / equal
    return 0.0 if optimized == 0 else "inf"


def summarize_multiseed_confirmation(
    config_path: str | Path = DEFAULT_MULTISEED_CONFIG,
    *,
    repo_root: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    """Write deterministic run, paired, and aggregate Phase 2D summaries."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    spec = load_multiseed_confirmation_spec(_resolve(root, config_path))
    output_root = root / spec.output_root
    results = {
        setting.run_id: _load_complete_result(
            output_root / setting.run_id,
            setting,
            spec.expected_cohort_dataset_sha256,
        )
        for setting in spec.runs
    }
    split_pairs = {
        (
            results[run.run_id]["train_split_id_hash"],
            results[run.run_id]["validation_split_id_hash"],
        )
        for run in spec.runs
    }
    if len(split_pairs) != 1:
        raise ValueError("Phase 2D runs do not share one train/validation cohort identity")

    run_summary_path = output_root / spec.run_summary_filename
    with run_summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(_run_summary_row(results[run.run_id]) for run in spec.runs)

    pair_rows: list[dict[str, str]] = []
    paired_percentages: list[float] = []
    optimized_wins = 0
    physics_gate_passes = 0
    optimized_finite = True
    equal_metrics: list[float] = []
    optimized_metrics: list[float] = []
    for seed in spec.seeds:
        equal = results[f"EQ{seed}"]
        optimized = results[f"OPT{seed}"]
        equal_nrmse = float(equal["best_validation_nrmse"])
        optimized_nrmse = float(optimized["best_validation_nrmse"])
        if equal_nrmse <= 0:
            raise ValueError(f"equal-weight validation nRMSE must be positive for seed {seed}")
        percentage = 100.0 * (equal_nrmse - optimized_nrmse) / equal_nrmse
        wins = optimized_nrmse < equal_nrmse
        equal_pde = float(equal["pde_rms"])
        optimized_pde = float(optimized["pde_rms"])
        finite_pair = bool(equal["finite_gradients"] and optimized["finite_gradients"])
        physics_gate = finite_pair and optimized_pde <= 2.0 * equal_pde
        optimized_finite = optimized_finite and bool(optimized["finite_gradients"])
        optimized_wins += int(wins)
        physics_gate_passes += int(physics_gate)
        paired_percentages.append(percentage)
        equal_metrics.append(equal_nrmse)
        optimized_metrics.append(optimized_nrmse)
        ratio = _pde_ratio(optimized_pde, equal_pde)
        pair_rows.append(
            {
                "seed": str(seed),
                "equal_validation_nrmse": _format_number(equal_nrmse),
                "optimized_validation_nrmse": _format_number(optimized_nrmse),
                "absolute_difference": _format_number(abs(equal_nrmse - optimized_nrmse)),
                "percentage_improvement": _format_number(percentage),
                "optimized_wins": _format_bool(wins),
                "equal_pde_rms": _format_number(equal_pde),
                "optimized_pde_rms": _format_number(optimized_pde),
                "pde_ratio_opt_vs_equal": ratio if isinstance(ratio, str) else _format_number(ratio),
                "physics_gate_pass": _format_bool(physics_gate),
            }
        )

    pair_summary_path = output_root / spec.pair_summary_filename
    with pair_summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(pair_rows)

    equal_mean = statistics.mean(equal_metrics)
    optimized_mean = statistics.mean(optimized_metrics)
    aggregate = {
        "equal_mean_validation_nrmse": equal_mean,
        "equal_std_validation_nrmse": statistics.pstdev(equal_metrics),
        "optimized_mean_validation_nrmse": optimized_mean,
        "optimized_std_validation_nrmse": statistics.pstdev(optimized_metrics),
        "mean_paired_percentage_improvement": statistics.mean(paired_percentages),
        "median_paired_percentage_improvement": statistics.median(paired_percentages),
        "optimized_win_count": optimized_wins,
        "physics_gate_pass_count": physics_gate_passes,
        "seeds": list(spec.seeds),
        "optimized_confirmed": (
            optimized_mean < equal_mean
            and optimized_wins >= 2
            and optimized_finite
            and physics_gate_passes == len(spec.seeds)
        ),
        "std_definition": "population",
        "interpretation": "descriptive robustness confirmation only",
        "inferential_significance_test_performed": False,
    }
    aggregate_summary_path = output_root / spec.aggregate_summary_filename
    with aggregate_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return run_summary_path, pair_summary_path, aggregate_summary_path


__all__ = [
    "DEFAULT_MULTISEED_CONFIG",
    "EXPECTED_COHORT_DATASET_SHA256",
    "EXPECTED_RUN_MATRIX",
    "EXPECTED_SEEDS",
    "PAIR_SUMMARY_FIELDS",
    "RESULT_FIELDS",
    "RUN_SUMMARY_FIELDS",
    "build_multiseed_run_config",
    "load_multiseed_confirmation_spec",
    "run_multiseed_confirmation",
    "summarize_multiseed_confirmation",
]
