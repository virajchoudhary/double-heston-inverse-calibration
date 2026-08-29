"""Frozen one-shot Phase 3A final synthetic holdout evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from src.double_heston import price_double_heston_call

from .collocation import (
    BoundaryPoints,
    PDEPoints,
    TerminalPoints,
    sample_high_s_boundary_points,
    sample_low_s_boundary_points,
    sample_pde_points,
    sample_terminal_points,
)
from .config import BaselineConfig, load_baseline_config, variance_state_bounds
from .losses import high_s_boundary_loss, low_s_boundary_loss, pde_loss, terminal_loss
from .model import DoubleHestonForwardPINN
from .parameter_source import ParameterSource, select_first_eligible_train_record

DEFAULT_FINAL_EVAL_CONFIG = Path("configs/mentor_dh_pinn/final_synthetic_eval_v1.yaml")
FINAL_SCHEMA = "mentor_dh_pinn_final_synthetic_eval_v1"
FINAL_DATASET_SCHEMA = "mentor_dh_pinn_final_synthetic_dataset_v1"
FINAL_CLAIM_SCHEMA = "mentor_dh_pinn_final_evaluation_claim_v1"
FINAL_EVAL_SEED = 73129
FINAL_EVAL_COUNT = 4096
FORBIDDEN_DEVELOPMENT_SEEDS = (3407, 11, 22, 33)
EXPECTED_PHASE1_DATASET_SHA256 = (
    "35a813036fe50eddcfe66db132b86d2dbfba01b4c99e341b8e5518c04d0f1ad4"
)
EXPECTED_PHASE1_ANCHOR = (909, 0.0012545129490993494)
FEATURE_NAMES = ("spot", "variance_slow", "variance_fast", "tau", "strike", "rate", "carry")
EQUAL_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
OPTIMIZED_WEIGHTS = (0.1, 0.1, 1.0, 1.0)
EXPECTED_CHECKPOINT_MATRIX = (
    ("EQ3407", "primary", "equal", 3407, "outputs/mentor_dh_pinn_baseline_v1/checkpoint.pt", *EQUAL_WEIGHTS),
    ("EQ11", "primary", "equal", 11, "outputs/mentor_dh_pinn_multiseed_confirmation/EQ11/checkpoint.pt", *EQUAL_WEIGHTS),
    ("EQ22", "primary", "equal", 22, "outputs/mentor_dh_pinn_multiseed_confirmation/EQ22/checkpoint.pt", *EQUAL_WEIGHTS),
    ("EQ33", "primary", "equal", 33, "outputs/mentor_dh_pinn_multiseed_confirmation/EQ33/checkpoint.pt", *EQUAL_WEIGHTS),
    ("OPT11", "secondary_ablation", "optimized", 11, "outputs/mentor_dh_pinn_multiseed_confirmation/OPT11/checkpoint.pt", *OPTIMIZED_WEIGHTS),
    ("OPT22", "secondary_ablation", "optimized", 22, "outputs/mentor_dh_pinn_multiseed_confirmation/OPT22/checkpoint.pt", *OPTIMIZED_WEIGHTS),
    ("OPT33", "secondary_ablation", "optimized", 33, "outputs/mentor_dh_pinn_multiseed_confirmation/OPT33/checkpoint.pt", *OPTIMIZED_WEIGHTS),
)
PRIMARY_RUN_IDS = ("EQ3407", "EQ11", "EQ22", "EQ33")
SECONDARY_RUN_IDS = ("OPT11", "OPT22", "OPT33")
PHASE2D_RUN_MATRIX = (
    ("EQ11", "equal", 11, *EQUAL_WEIGHTS),
    ("OPT11", "optimized", 11, *OPTIMIZED_WEIGHTS),
    ("EQ22", "equal", 22, *EQUAL_WEIGHTS),
    ("OPT22", "optimized", 22, *OPTIMIZED_WEIGHTS),
    ("EQ33", "equal", 33, *EQUAL_WEIGHTS),
    ("OPT33", "optimized", 33, *OPTIMIZED_WEIGHTS),
)

RUN_SUMMARY_FIELDS = (
    "run_id", "role", "variant", "seed", "lambda_pde", "lambda_boundary",
    "lambda_terminal", "lambda_data", "checkpoint_sha256", "price_rmse", "price_mae",
    "price_nrmse", "pde_rms", "pde_max_abs", "terminal_rmse", "terminal_max_abs",
    "boundary_low_s_rmse", "boundary_high_s_rmse", "inference_seconds_total",
    "inference_seconds_per_contract", "all_finite",
)
SECONDARY_FIELDS = RUN_SUMMARY_FIELDS + (
    "secondary_ablation_only", "may_reopen_model_selection",
)
METRIC_NAMES = (
    "price_nrmse", "price_rmse", "price_mae", "pde_rms", "terminal_rmse",
    "boundary_low_s_rmse", "boundary_high_s_rmse",
)
REQUIRED_METRIC_FIELDS = set(RUN_SUMMARY_FIELDS) | {
    "schema_version", "final_dataset_sha256", "sample_id_hash", "checkpoint_git_sha",
    "checkpoint_path", "diagnostic_identity",
}


@dataclass(frozen=True)
class DiagnosticConfig:
    pde_count: int
    terminal_count: int
    boundary_count: int
    pde_seed_offset: int
    terminal_seed_offset: int
    low_boundary_seed_offset: int
    high_boundary_seed_offset: int


@dataclass(frozen=True)
class CheckpointSetting:
    run_id: str
    role: str
    variant: str
    seed: int
    path: str
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


@dataclass(frozen=True)
class FinalSyntheticEvalSpec:
    schema_version: str
    base_config_path: str
    phase1_output_dir: str
    phase1_checkpoint_path: str
    phase2d_config_path: str
    phase2d_output_dir: str
    output_root: str
    expected_phase1_dataset_sha256: str
    expected_phase1_best_epoch: int
    expected_phase1_best_validation_nrmse: float
    final_eval_seed: int
    final_eval_count: int
    dataset_filename: str
    dataset_manifest_filename: str
    claim_filename: str
    metrics_subdirectory: str
    run_summary_filename: str
    primary_aggregate_filename: str
    secondary_ablation_filename: str
    evaluation_manifest_filename: str
    diagnostics: DiagnosticConfig
    checkpoints: tuple[CheckpointSetting, ...]


@dataclass(frozen=True)
class FinalSyntheticDataset:
    """Immutable split-free final holdout; no train/validation/test API exists."""

    features: np.ndarray
    reference_prices: np.ndarray
    sample_ids: np.ndarray
    parameter_source: ParameterSource
    manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=np.float64)
        prices = np.asarray(self.reference_prices, dtype=np.float64)
        sample_ids = np.asarray(self.sample_ids, dtype=str)
        if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
            raise ValueError("final holdout features have an invalid shape")
        if prices.shape != (features.shape[0],) or sample_ids.shape != prices.shape:
            raise ValueError("final holdout arrays must align")
        if not np.isfinite(features).all() or not np.isfinite(prices).all():
            raise ValueError("final holdout arrays must be finite")
        if len(set(sample_ids.tolist())) != len(sample_ids):
            raise ValueError("final holdout sample IDs must be unique")
        for array in (features, prices, sample_ids):
            array.setflags(write=False)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "reference_prices", prices)
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))

    @property
    def size(self) -> int:
        return int(self.features.shape[0])


@dataclass(frozen=True)
class CheckpointIdentity:
    setting: CheckpointSetting
    path: Path
    sha256: str
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class CommonDiagnostics:
    pde: PDEPoints
    terminal: TerminalPoints
    low_boundary: BoundaryPoints
    high_boundary: BoundaryPoints
    identity: Mapping[str, Any]


def _strict_keys(mapping: Mapping[str, Any], expected: set[str], section: str) -> None:
    actual = set(mapping)
    if actual != expected:
        raise ValueError(
            f"invalid {section} fields: unknown={sorted(actual - expected)}, "
            f"missing={sorted(expected - actual)}"
        )


def _parse_checkpoint(raw: Any, index: int) -> CheckpointSetting:
    fields = {
        "run_id", "role", "variant", "seed", "path", "lambda_pde",
        "lambda_boundary", "lambda_terminal", "lambda_data",
    }
    if not isinstance(raw, Mapping):
        raise TypeError(f"checkpoint {index} must be a mapping")
    _strict_keys(raw, fields, f"checkpoint {index}")
    return CheckpointSetting(
        run_id=str(raw["run_id"]),
        role=str(raw["role"]),
        variant=str(raw["variant"]),
        seed=int(raw["seed"]),
        path=str(raw["path"]),
        lambda_pde=float(raw["lambda_pde"]),
        lambda_boundary=float(raw["lambda_boundary"]),
        lambda_terminal=float(raw["lambda_terminal"]),
        lambda_data=float(raw["lambda_data"]),
    )


def _checkpoint_tuple(setting: CheckpointSetting) -> tuple[Any, ...]:
    return (
        setting.run_id, setting.role, setting.variant, setting.seed, setting.path,
        setting.lambda_pde, setting.lambda_boundary, setting.lambda_terminal,
        setting.lambda_data,
    )


def load_final_synthetic_eval_spec(
    path: str | Path = DEFAULT_FINAL_EVAL_CONFIG,
) -> FinalSyntheticEvalSpec:
    """Load and strictly validate the frozen Phase 3A protocol."""
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise TypeError("Phase 3A YAML must contain a mapping")
    top_fields = {
        "schema_version", "base_config_path", "phase1_output_dir", "phase1_checkpoint_path",
        "phase2d_config_path", "phase2d_output_dir", "output_root",
        "expected_phase1_dataset_sha256", "expected_phase1_best_epoch",
        "expected_phase1_best_validation_nrmse", "final_eval_seed", "final_eval_count",
        "dataset_filename", "dataset_manifest_filename", "claim_filename",
        "metrics_subdirectory", "run_summary_filename", "primary_aggregate_filename",
        "secondary_ablation_filename", "evaluation_manifest_filename", "diagnostics",
        "checkpoints",
    }
    _strict_keys(raw, top_fields, "Phase 3A")
    diagnostic_fields = {
        "pde_count", "terminal_count", "boundary_count", "pde_seed_offset",
        "terminal_seed_offset", "low_boundary_seed_offset", "high_boundary_seed_offset",
    }
    diagnostics_raw = raw["diagnostics"]
    if not isinstance(diagnostics_raw, Mapping):
        raise TypeError("Phase 3A diagnostics must be a mapping")
    _strict_keys(diagnostics_raw, diagnostic_fields, "Phase 3A diagnostics")
    if not isinstance(raw["checkpoints"], list):
        raise TypeError("Phase 3A checkpoints must be a list")
    spec = FinalSyntheticEvalSpec(
        schema_version=str(raw["schema_version"]),
        base_config_path=str(raw["base_config_path"]),
        phase1_output_dir=str(raw["phase1_output_dir"]),
        phase1_checkpoint_path=str(raw["phase1_checkpoint_path"]),
        phase2d_config_path=str(raw["phase2d_config_path"]),
        phase2d_output_dir=str(raw["phase2d_output_dir"]),
        output_root=str(raw["output_root"]),
        expected_phase1_dataset_sha256=str(raw["expected_phase1_dataset_sha256"]),
        expected_phase1_best_epoch=int(raw["expected_phase1_best_epoch"]),
        expected_phase1_best_validation_nrmse=float(raw["expected_phase1_best_validation_nrmse"]),
        final_eval_seed=int(raw["final_eval_seed"]),
        final_eval_count=int(raw["final_eval_count"]),
        dataset_filename=str(raw["dataset_filename"]),
        dataset_manifest_filename=str(raw["dataset_manifest_filename"]),
        claim_filename=str(raw["claim_filename"]),
        metrics_subdirectory=str(raw["metrics_subdirectory"]),
        run_summary_filename=str(raw["run_summary_filename"]),
        primary_aggregate_filename=str(raw["primary_aggregate_filename"]),
        secondary_ablation_filename=str(raw["secondary_ablation_filename"]),
        evaluation_manifest_filename=str(raw["evaluation_manifest_filename"]),
        diagnostics=DiagnosticConfig(**{name: int(diagnostics_raw[name]) for name in diagnostic_fields}),
        checkpoints=tuple(_parse_checkpoint(item, index) for index, item in enumerate(raw["checkpoints"])),
    )
    if spec.schema_version != FINAL_SCHEMA:
        raise ValueError("unexpected Phase 3A schema")
    if spec.final_eval_seed != FINAL_EVAL_SEED or spec.final_eval_seed in FORBIDDEN_DEVELOPMENT_SEEDS:
        raise ValueError("final evaluation seed is frozen at 73129 and must differ from all development seeds")
    if spec.final_eval_count != FINAL_EVAL_COUNT:
        raise ValueError("final evaluation count is frozen at 4096")
    if spec.expected_phase1_dataset_sha256 != EXPECTED_PHASE1_DATASET_SHA256:
        raise ValueError("Phase 1 dataset SHA anchor has drifted")
    if (
        spec.expected_phase1_best_epoch,
        spec.expected_phase1_best_validation_nrmse,
    ) != EXPECTED_PHASE1_ANCHOR:
        raise ValueError("Phase 1 checkpoint anchor has drifted")
    if tuple(_checkpoint_tuple(item) for item in spec.checkpoints) != EXPECTED_CHECKPOINT_MATRIX:
        raise ValueError("Phase 3A primary/secondary checkpoint matrix has drifted")
    if tuple(item.run_id for item in spec.checkpoints if item.role == "primary") != PRIMARY_RUN_IDS:
        raise ValueError("Phase 3A primary checkpoint set has drifted")
    if tuple(item.run_id for item in spec.checkpoints if item.role == "secondary_ablation") != SECONDARY_RUN_IDS:
        raise ValueError("Phase 3A secondary checkpoint set has drifted")
    if (
        spec.diagnostics.pde_count,
        spec.diagnostics.terminal_count,
        spec.diagnostics.boundary_count,
    ) != (256, 128, 128):
        raise ValueError("Phase 3A diagnostic counts must match the sealed baseline evaluation")
    if len({
        spec.diagnostics.pde_seed_offset,
        spec.diagnostics.terminal_seed_offset,
        spec.diagnostics.low_boundary_seed_offset,
        spec.diagnostics.high_boundary_seed_offset,
    }) != 4:
        raise ValueError("Phase 3A diagnostic seed offsets must be distinct")
    expected_paths = {
        "base_config_path": "configs/mentor_dh_pinn/baseline_v1.yaml",
        "phase1_output_dir": "outputs/mentor_dh_pinn_baseline_v1",
        "phase1_checkpoint_path": "outputs/mentor_dh_pinn_baseline_v1/checkpoint.pt",
        "phase2d_config_path": "configs/mentor_dh_pinn/multiseed_confirmation_v1.yaml",
        "phase2d_output_dir": "outputs/mentor_dh_pinn_multiseed_confirmation",
        "output_root": "outputs/mentor_dh_pinn_final_synthetic_eval",
        "dataset_filename": "final_synthetic_dataset.npz",
        "dataset_manifest_filename": "final_synthetic_dataset_manifest.json",
        "claim_filename": "final_evaluation_claim.json",
        "metrics_subdirectory": "checkpoint_metrics",
        "run_summary_filename": "final_synthetic_run_summary.csv",
        "primary_aggregate_filename": "final_synthetic_primary_aggregate.json",
        "secondary_ablation_filename": "final_synthetic_secondary_ablation.csv",
        "evaluation_manifest_filename": "final_synthetic_eval_manifest.json",
    }
    for name, expected in expected_paths.items():
        if getattr(spec, name) != expected:
            raise ValueError(f"unexpected Phase 3A {name}")
    return spec


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stable_hash_strings(values: Sequence[str] | np.ndarray) -> str:
    return hashlib.sha256("\n".join(str(value) for value in values).encode("utf-8")).hexdigest()


def _git_sha(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
        capture_output=True, text=True,
    )
    value = completed.stdout.strip()
    if not value:
        raise RuntimeError("git SHA is unavailable")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"JSON artifact must contain a mapping: {path}")
    return value


def _load_checkpoint(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise TypeError(f"checkpoint must contain a mapping: {path}")
    return value


def _complete_identity(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} is missing or incomplete")
    return value


def _scientific_source_identity(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Exclude the host-specific locator while retaining every scientific identity."""
    required = {
        "dataset_sha256", "surface_id", "stored_split", "parameter_hash",
        "canonical_parameter_order", "parameter_vector", "selection_rule",
    }
    missing = required - set(provenance)
    if missing:
        raise ValueError(f"parameter/source provenance is incomplete: {sorted(missing)}")
    return {name: provenance[name] for name in sorted(required)}


def _expected_checkpoint_config(
    baseline: BaselineConfig,
    setting: CheckpointSetting,
) -> dict[str, Any]:
    expected = baseline.to_dict()
    expected["seed"] = setting.seed
    expected["losses"].update(
        pde_lambda=setting.lambda_pde,
        boundary_lambda=setting.lambda_boundary,
        terminal_lambda=setting.lambda_terminal,
        data_lambda=setting.lambda_data,
    )
    return expected


def _validate_phase2d_config(path: Path) -> None:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise TypeError("Phase 2D config must contain a mapping")
    if tuple(int(seed) for seed in raw.get("seeds", [])) != (11, 22, 33):
        raise ValueError("Phase 2D seed provenance drift")
    observed = tuple(
        (
            str(item.get("run_id")), str(item.get("variant")), int(item.get("seed")),
            float(item.get("lambda_pde")), float(item.get("lambda_boundary")),
            float(item.get("lambda_terminal")), float(item.get("lambda_data")),
        )
        for item in raw.get("runs", [])
    )
    if observed != PHASE2D_RUN_MATRIX:
        raise ValueError("Phase 2D run or lambda provenance drift")
    if int(raw.get("max_epochs", -1)) != 1000 or int(raw.get("patience", -1)) != 100:
        raise ValueError("Phase 2D training-control provenance drift")
    if str(raw.get("expected_cohort_dataset_sha256")) != EXPECTED_PHASE1_DATASET_SHA256:
        raise ValueError("Phase 2D cohort provenance drift")


def validate_frozen_checkpoint_provenance(
    spec: FinalSyntheticEvalSpec,
    baseline: BaselineConfig,
    parameter_source: ParameterSource,
    *,
    repo_root: str | Path,
) -> tuple[tuple[CheckpointIdentity, ...], Mapping[str, Any]]:
    """Validate Phase 1 and Phase 2D identities before final cohort generation."""
    root = Path(repo_root)
    phase1_path = root / spec.phase1_checkpoint_path
    phase1 = _load_checkpoint(phase1_path)
    if phase1.get("config") != baseline.to_dict():
        raise ValueError("Phase 1 exact baseline config identity mismatch")
    if phase1.get("seed") != 3407 or phase1.get("weights") != {
        "pde": 1.0, "boundary": 1.0, "terminal": 1.0, "data": 1.0,
    }:
        raise ValueError("Phase 1 seed or equal-weight identity mismatch")
    if phase1.get("best_epoch") != spec.expected_phase1_best_epoch:
        raise ValueError("Phase 1 best epoch mismatch")
    if phase1.get("best_validation_nrmse") != spec.expected_phase1_best_validation_nrmse:
        raise ValueError("Phase 1 best validation nRMSE mismatch")
    phase1_dataset = _complete_identity(phase1.get("dataset_identity"), "Phase 1 dataset identity")
    phase1_provenance = _complete_identity(phase1.get("provenance"), "Phase 1 source identity")
    if phase1_dataset.get("dataset_sha256") != spec.expected_phase1_dataset_sha256:
        raise ValueError("Phase 1 generated dataset SHA mismatch")
    split_hashes = _complete_identity(phase1_dataset.get("split_id_hashes"), "Phase 1 split identities")
    if not all(split_hashes.get(name) for name in ("train", "validation", "test")):
        raise ValueError("Phase 1 split identities are incomplete")
    if _scientific_source_identity(phase1_provenance) != _scientific_source_identity(
        parameter_source.provenance()
    ):
        raise ValueError("Phase 1 parameter/source provenance mismatch")
    if phase1_dataset.get("parameter_source") != phase1_provenance:
        raise ValueError("Phase 1 checkpoint dataset/source identities disagree")
    if not phase1.get("git_sha") or phase1.get("git_sha") == "unknown":
        raise ValueError("Phase 1 checkpoint git identity is incomplete")

    _validate_phase2d_config(root / spec.phase2d_config_path)
    phase2d_root = root / spec.phase2d_output_dir
    aggregate = _load_json(phase2d_root / "multiseed_aggregate_summary.json")
    if aggregate.get("optimized_confirmed") is not False:
        raise ValueError("Phase 2D aggregate must state optimized_confirmed == false")
    if aggregate.get("inferential_significance_test_performed") is not False:
        raise ValueError("Phase 2D aggregate inferential-test identity mismatch")
    if aggregate.get("seeds") != [11, 22, 33]:
        raise ValueError("Phase 2D aggregate seed identity mismatch")

    identities: list[CheckpointIdentity] = []
    for setting in spec.checkpoints:
        checkpoint_path = root / setting.path
        checkpoint = phase1 if setting.run_id == "EQ3407" else _load_checkpoint(checkpoint_path)
        expected_config = baseline.to_dict() if setting.run_id == "EQ3407" else _expected_checkpoint_config(baseline, setting)
        if checkpoint.get("config") != expected_config:
            raise ValueError(f"checkpoint config identity mismatch for {setting.run_id}")
        if checkpoint.get("seed") != setting.seed or checkpoint.get("weights") != setting.weights:
            raise ValueError(f"checkpoint seed/lambda identity mismatch for {setting.run_id}")
        if checkpoint.get("dataset_identity") != phase1_dataset:
            raise ValueError(f"sealed cohort identity mismatch for {setting.run_id}")
        if checkpoint.get("provenance") != phase1_provenance:
            raise ValueError(f"parameter/source provenance mismatch for {setting.run_id}")
        checkpoint_git = checkpoint.get("git_sha")
        if not checkpoint_git or checkpoint_git == "unknown":
            raise ValueError(f"checkpoint git identity is incomplete for {setting.run_id}")
        if "model_state_dict" not in checkpoint:
            raise ValueError(f"checkpoint model state is missing for {setting.run_id}")
        if setting.run_id != "EQ3407":
            result = _load_json(phase2d_root / setting.run_id / "multiseed_result.json")
            expected_result_identity = {
                "run_id": setting.run_id,
                "variant": setting.variant,
                "seed": setting.seed,
                "lambda_pde": setting.lambda_pde,
                "lambda_boundary": setting.lambda_boundary,
                "lambda_terminal": setting.lambda_terminal,
                "lambda_data": setting.lambda_data,
            }
            if any(result.get(name) != value for name, value in expected_result_identity.items()):
                raise ValueError(f"Phase 2D result identity mismatch for {setting.run_id}")
            if result.get("cohort_dataset_sha256") != spec.expected_phase1_dataset_sha256:
                raise ValueError(f"Phase 2D result cohort SHA mismatch for {setting.run_id}")
            if (
                result.get("train_split_id_hash") != split_hashes["train"]
                or result.get("validation_split_id_hash") != split_hashes["validation"]
            ):
                raise ValueError(f"Phase 2D result split identity mismatch for {setting.run_id}")
            if not result.get("git_sha") or result.get("git_sha") != checkpoint_git:
                raise ValueError(f"Phase 2D result/checkpoint git identity mismatch for {setting.run_id}")
        identities.append(
            CheckpointIdentity(
                setting=setting,
                path=checkpoint_path,
                sha256=_sha256_file(checkpoint_path),
                payload=checkpoint,
            )
        )
    return tuple(identities), phase1_dataset


def _price_call(
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    carry: float,
    parameters: np.ndarray,
    node_count: int,
) -> float:
    return float(
        price_double_heston_call(
            spot, strike, tau, rate, carry, parameters, node_count=node_count
        )
    )


def _domain_identity(baseline: BaselineConfig) -> dict[str, Any]:
    domain = baseline.domain
    return {
        "spot": [domain.spot_min, domain.spot_max],
        "moneyness": [domain.moneyness_min, domain.moneyness_max],
        "tau_years": [domain.tau_min, domain.tau_max],
        "rate": [domain.rate_min, domain.rate_max],
        "carry": [domain.carry_min, domain.carry_max],
        "variance_state_rule": domain.variance_state_rule,
        "variance_theta_min_multiplier": domain.variance_theta_min_multiplier,
        "variance_theta_max_multiplier": domain.variance_theta_max_multiplier,
        "variance_floor": domain.variance_floor,
        "variance_ceiling": domain.variance_ceiling,
    }


def generate_final_synthetic_dataset(
    spec: FinalSyntheticEvalSpec,
    baseline: BaselineConfig,
    parameter_source: ParameterSource,
    output_dir: str | Path,
    *,
    repo_root: str | Path,
) -> FinalSyntheticDataset:
    """Generate and persist the deterministic split-free final holdout."""
    if spec.final_eval_seed in FORBIDDEN_DEVELOPMENT_SEEDS:
        raise ValueError("final holdout seed collides with a development seed")
    count = spec.final_eval_count
    generator = np.random.default_rng(spec.final_eval_seed)
    domain = baseline.domain
    spot = generator.uniform(domain.spot_min, domain.spot_max, count)
    moneyness = generator.uniform(domain.moneyness_min, domain.moneyness_max, count)
    tau = generator.uniform(domain.tau_min, domain.tau_max, count)
    rate = generator.uniform(domain.rate_min, domain.rate_max, count)
    carry = generator.uniform(domain.carry_min, domain.carry_max, count)
    slow_bounds = variance_state_bounds(float(parameter_source.vector[1]), domain)
    fast_bounds = variance_state_bounds(float(parameter_source.vector[6]), domain)
    variance_slow = generator.uniform(*slow_bounds, count)
    variance_fast = generator.uniform(*fast_bounds, count)
    strike = spot * moneyness
    features = np.column_stack(
        (spot, variance_slow, variance_fast, tau, strike, rate, carry)
    ).astype(np.float64, copy=False)
    references = np.empty(count, dtype=np.float64)
    for index in range(count):
        parameters = parameter_source.parameters_for_state(
            float(variance_slow[index]), float(variance_fast[index])
        )
        references[index] = _price_call(
            float(spot[index]), float(strike[index]), float(tau[index]),
            float(rate[index]), float(carry[index]), parameters,
            baseline.pricing_node_count,
        )
    sample_ids = np.asarray(
        [f"mentor_dh_pinn_final_v1_{index:06d}" for index in range(count)], dtype=str
    )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    dataset_path = root / spec.dataset_filename
    np.savez_compressed(
        dataset_path,
        features=features,
        reference_prices=references,
        sample_ids=sample_ids,
    )
    domain_identity = _domain_identity(baseline)
    scientific_identity = {
        "feature_names": list(FEATURE_NAMES),
        "domain": domain_identity,
        "option_type": baseline.option_type,
        "pricing_node_count": baseline.pricing_node_count,
        "frozen_surfaces_sha256": baseline.data.frozen_surfaces_sha256,
        "parameter_hash": baseline.expected_parameter_hash,
        "surface_id": parameter_source.surface_id,
        "variance_state_rule": baseline.domain.variance_state_rule,
    }
    manifest = {
        "schema_version": FINAL_DATASET_SCHEMA,
        "seed": spec.final_eval_seed,
        "count": count,
        "feature_names": list(FEATURE_NAMES),
        "frozen_surfaces_sha256": parameter_source.dataset_sha256,
        "parameter_hash": parameter_source.parameter_hash,
        "source_surface_id": parameter_source.surface_id,
        "parameter_source": parameter_source.provenance(),
        "domain": domain_identity,
        "baseline_config_sha256": _json_sha256(baseline.to_dict()),
        "scientific_identity_sha256": _json_sha256(scientific_identity),
        "pricing_node_count": baseline.pricing_node_count,
        "option_type": baseline.option_type,
        "sample_ids": sample_ids.tolist(),
        "sample_id_hash": _stable_hash_strings(sample_ids),
        "dataset_filename": spec.dataset_filename,
        "dataset_npz_sha256": _sha256_file(dataset_path),
        "generation_code_git_sha": _git_sha(Path(repo_root)),
        "final_holdout_only": True,
        "used_for_training": False,
        "used_for_validation": False,
    }
    with (root / spec.dataset_manifest_filename).open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return FinalSyntheticDataset(
        features=features,
        reference_prices=references,
        sample_ids=sample_ids,
        parameter_source=parameter_source,
        manifest=manifest,
    )


def _tensor_hash(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        values = np.ascontiguousarray(tensor.detach().cpu().numpy())
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def build_common_diagnostics(
    spec: FinalSyntheticEvalSpec,
    baseline: BaselineConfig,
    parameter_source: ParameterSource,
    *,
    device: torch.device | str,
) -> CommonDiagnostics:
    """Build one common deterministic physics point set for every checkpoint."""
    seeds = {
        "pde": spec.final_eval_seed + spec.diagnostics.pde_seed_offset,
        "terminal": spec.final_eval_seed + spec.diagnostics.terminal_seed_offset,
        "low_boundary": spec.final_eval_seed + spec.diagnostics.low_boundary_seed_offset,
        "high_boundary": spec.final_eval_seed + spec.diagnostics.high_boundary_seed_offset,
    }
    pde = sample_pde_points(
        spec.diagnostics.pde_count,
        config=baseline,
        parameter_source=parameter_source,
        seed=seeds["pde"],
        device=device,
    )
    terminal = sample_terminal_points(
        spec.diagnostics.terminal_count,
        config=baseline,
        parameter_source=parameter_source,
        seed=seeds["terminal"],
        device=device,
    )
    low = sample_low_s_boundary_points(
        spec.diagnostics.boundary_count,
        config=baseline,
        parameter_source=parameter_source,
        seed=seeds["low_boundary"],
        device=device,
    )
    high = sample_high_s_boundary_points(
        spec.diagnostics.boundary_count,
        config=baseline,
        parameter_source=parameter_source,
        seed=seeds["high_boundary"],
        device=device,
    )
    identity = {
        "derived_only_from_final_eval_seed": True,
        "final_eval_seed": spec.final_eval_seed,
        "seeds": seeds,
        "counts": {
            "pde": spec.diagnostics.pde_count,
            "terminal": spec.diagnostics.terminal_count,
            "low_boundary": spec.diagnostics.boundary_count,
            "high_boundary": spec.diagnostics.boundary_count,
        },
        "pde_sha256": _tensor_hash(pde.features, pde.parameters),
        "terminal_sha256": _tensor_hash(terminal.features),
        "low_boundary_sha256": _tensor_hash(low.features, low.target),
        "high_boundary_sha256": _tensor_hash(high.features, high.target),
    }
    return CommonDiagnostics(pde, terminal, low, high, MappingProxyType(identity))


def _rmse(values: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(values.square())).item())


def _mae(values: torch.Tensor) -> float:
    return float(torch.mean(values.abs()).item())


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_inference_model(
    identity: CheckpointIdentity,
    baseline: BaselineConfig,
    device: torch.device,
) -> DoubleHestonForwardPINN:
    model = DoubleHestonForwardPINN(
        feature_min=baseline.domain.feature_min,
        feature_max=baseline.domain.feature_max,
    ).to(device=device, dtype=torch.float64)
    model.load_state_dict(identity.payload["model_state_dict"])
    model.eval()
    return model


def _evaluate_checkpoint(
    identity: CheckpointIdentity,
    dataset: FinalSyntheticDataset,
    diagnostics: CommonDiagnostics,
    baseline: BaselineConfig,
    *,
    device: torch.device | str,
) -> dict[str, Any]:
    resolved = torch.device(device)
    model = _load_inference_model(identity, baseline, resolved)
    features = torch.from_numpy(np.asarray(dataset.features)).to(resolved, dtype=torch.float64)
    references = torch.from_numpy(np.asarray(dataset.reference_prices)).to(resolved, dtype=torch.float64)
    _sync(resolved)
    started = time.perf_counter()
    with torch.no_grad():
        predictions = model(features).reshape(-1)
    _sync(resolved)
    inference_seconds = time.perf_counter() - started
    errors = predictions - references
    normalized_errors = errors / features[:, 0]
    _, residual = pde_loss(
        model, diagnostics.pde, scale_floor=baseline.losses.pde_scale_floor
    )
    _, terminal_predictions, terminal_targets = terminal_loss(model, diagnostics.terminal)
    _, low_predictions, low_targets = low_s_boundary_loss(model, diagnostics.low_boundary)
    _, high_predictions, high_targets = high_s_boundary_loss(model, diagnostics.high_boundary)
    terminal_errors = terminal_predictions - terminal_targets
    low_errors = low_predictions - low_targets
    high_errors = high_predictions - high_targets
    metrics: dict[str, Any] = {
        "schema_version": "mentor_dh_pinn_final_checkpoint_metrics_v1",
        "run_id": identity.setting.run_id,
        "role": identity.setting.role,
        "variant": identity.setting.variant,
        "seed": identity.setting.seed,
        "lambda_pde": identity.setting.lambda_pde,
        "lambda_boundary": identity.setting.lambda_boundary,
        "lambda_terminal": identity.setting.lambda_terminal,
        "lambda_data": identity.setting.lambda_data,
        "checkpoint_path": identity.setting.path,
        "checkpoint_sha256": identity.sha256,
        "checkpoint_git_sha": identity.payload["git_sha"],
        "final_dataset_sha256": dataset.manifest["dataset_npz_sha256"],
        "sample_id_hash": dataset.manifest["sample_id_hash"],
        "diagnostic_identity": dict(diagnostics.identity),
        "price_rmse": _rmse(errors),
        "price_mae": _mae(errors),
        "price_nrmse": _rmse(normalized_errors),
        "pde_rms": _rmse(residual.detach()),
        "pde_max_abs": float(residual.detach().abs().max().item()),
        "terminal_rmse": _rmse(terminal_errors.detach()),
        "terminal_max_abs": float(terminal_errors.detach().abs().max().item()),
        "boundary_low_s_rmse": _rmse(low_errors.detach()),
        "boundary_high_s_rmse": _rmse(high_errors.detach()),
        "inference_seconds_total": float(inference_seconds),
        "inference_seconds_per_contract": float(inference_seconds / dataset.size),
    }
    numeric = [value for key, value in metrics.items() if key in set(RUN_SUMMARY_FIELDS)]
    metrics["all_finite"] = all(
        not isinstance(value, float) or math.isfinite(value) for value in numeric
    ) and bool(
        torch.isfinite(predictions).all()
        and torch.isfinite(residual).all()
        and torch.isfinite(terminal_errors).all()
        and torch.isfinite(low_errors).all()
        and torch.isfinite(high_errors).all()
    )
    if not metrics["all_finite"]:
        raise FloatingPointError(f"non-finite final evaluation for {identity.setting.run_id}")
    return metrics


def _preflight_output_root(output_root: Path) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Phase 3A output/claim already exists and requires explicit recovery: {output_root}"
        )


def _write_atomic_claim(path: Path, claim: Mapping[str, Any]) -> None:
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(dict(claim), handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as error:
        raise FileExistsError(f"final evaluation was already claimed at {path}") from error


def run_final_synthetic_eval(
    config_path: str | Path = DEFAULT_FINAL_EVAL_CONFIG,
    *,
    repo_root: str | Path | None = None,
    device: str = "cpu",
) -> dict[str, Path]:
    """Generate the final holdout, claim once, and evaluate every frozen checkpoint."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    spec = load_final_synthetic_eval_spec(_resolve(root, config_path))
    output_root = root / spec.output_root
    _preflight_output_root(output_root)
    baseline = load_baseline_config(root / spec.base_config_path)
    source = select_first_eligible_train_record(
        root / baseline.data.frozen_surfaces_path,
        expected_sha256=baseline.data.frozen_surfaces_sha256,
        expected_parameter_hash=baseline.expected_parameter_hash,
    )
    checkpoints, phase1_dataset_identity = validate_frozen_checkpoint_provenance(
        spec, baseline, source, repo_root=root
    )
    output_root.mkdir(parents=True, exist_ok=True)
    dataset = generate_final_synthetic_dataset(
        spec, baseline, source, output_root, repo_root=root
    )
    original_hashes = phase1_dataset_identity["split_id_hashes"]
    if dataset.manifest["sample_id_hash"] in set(original_hashes.values()):
        raise ValueError("final sample ID hash collides with an original mentor split identity")
    diagnostics = build_common_diagnostics(spec, baseline, source, device=device)
    claim = {
        "schema_version": FINAL_CLAIM_SCHEMA,
        "final_cohort_sha256": dataset.manifest["dataset_npz_sha256"],
        "sample_id_hash": dataset.manifest["sample_id_hash"],
        "final_seed": spec.final_eval_seed,
        "current_git_sha": _git_sha(root),
        "checkpoint_identities": [
            {
                "run_id": item.setting.run_id,
                "role": item.setting.role,
                "variant": item.setting.variant,
                "seed": item.setting.seed,
                "checkpoint_sha256": item.sha256,
                "checkpoint_git_sha": item.payload["git_sha"],
                "weights": item.setting.weights,
            }
            for item in checkpoints
        ],
        "primary_run_ids": list(PRIMARY_RUN_IDS),
        "secondary_ablation_run_ids": list(SECONDARY_RUN_IDS),
        "final_loss_choice": "equal",
        "loss_choice_frozen_before_final_evaluation": True,
        "diagnostic_identity": dict(diagnostics.identity),
    }
    claim_path = output_root / spec.claim_filename
    _write_atomic_claim(claim_path, claim)
    metrics_root = output_root / spec.metrics_subdirectory
    metrics_root.mkdir(parents=True, exist_ok=False)
    for identity in checkpoints:
        metrics = _evaluate_checkpoint(
            identity, dataset, diagnostics, baseline, device=device
        )
        with (metrics_root / f"{identity.setting.run_id}.json").open(
            "x", encoding="utf-8"
        ) as handle:
            json.dump(metrics, handle, indent=2, sort_keys=True)
            handle.write("\n")
    outputs = summarize_final_synthetic_eval(config_path, repo_root=root)
    return {
        "claim": claim_path,
        "dataset_manifest": output_root / spec.dataset_manifest_filename,
        **outputs,
    }


def _format_number(value: Any) -> str:
    return format(float(value), ".17g")


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _validate_completed_metric(
    payload: Mapping[str, Any],
    setting: CheckpointSetting,
    dataset_manifest: Mapping[str, Any],
    claim: Mapping[str, Any],
) -> dict[str, Any]:
    missing = REQUIRED_METRIC_FIELDS - set(payload)
    if missing:
        raise ValueError(f"incomplete Phase 3A metric for {setting.run_id}: missing {sorted(missing)}")
    expected = {
        "run_id": setting.run_id, "role": setting.role, "variant": setting.variant,
        "seed": setting.seed, "lambda_pde": setting.lambda_pde,
        "lambda_boundary": setting.lambda_boundary, "lambda_terminal": setting.lambda_terminal,
        "lambda_data": setting.lambda_data,
    }
    if any(payload.get(name) != value for name, value in expected.items()):
        raise ValueError(f"Phase 3A metric identity mismatch for {setting.run_id}")
    if payload.get("final_dataset_sha256") != dataset_manifest.get("dataset_npz_sha256"):
        raise ValueError(f"Phase 3A dataset identity mismatch for {setting.run_id}")
    if payload.get("sample_id_hash") != dataset_manifest.get("sample_id_hash"):
        raise ValueError(f"Phase 3A sample identity mismatch for {setting.run_id}")
    if payload.get("diagnostic_identity") != claim.get("diagnostic_identity"):
        raise ValueError(f"Phase 3A common diagnostic identity mismatch for {setting.run_id}")
    claim_entries = claim.get("checkpoint_identities", [])
    matches = [item for item in claim_entries if item.get("run_id") == setting.run_id]
    if len(matches) != 1:
        raise ValueError(f"Phase 3A claim checkpoint identity missing for {setting.run_id}")
    claim_identity = matches[0]
    expected_claim_identity = {
        "run_id": setting.run_id,
        "role": setting.role,
        "variant": setting.variant,
        "seed": setting.seed,
        "checkpoint_sha256": payload.get("checkpoint_sha256"),
        "checkpoint_git_sha": payload.get("checkpoint_git_sha"),
        "weights": setting.weights,
    }
    if claim_identity != expected_claim_identity:
        raise ValueError(f"Phase 3A claim/metric checkpoint identity mismatch for {setting.run_id}")
    if type(payload.get("all_finite")) is not bool:
        raise ValueError(f"Phase 3A finite status is malformed for {setting.run_id}")
    for name in set(RUN_SUMMARY_FIELDS) - {
        "run_id", "role", "variant", "checkpoint_sha256", "all_finite",
    }:
        if name in {"seed"}:
            continue
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"non-finite or malformed {name} for {setting.run_id}")
    return dict(payload)


def _summary_row(payload: Mapping[str, Any]) -> dict[str, str]:
    row: dict[str, str] = {}
    for field in RUN_SUMMARY_FIELDS:
        value = payload[field]
        if field == "all_finite":
            row[field] = _format_bool(bool(value))
        elif field == "seed":
            row[field] = str(int(value))
        elif field.startswith("lambda_") or field in METRIC_NAMES or field in {
            "pde_max_abs", "terminal_max_abs", "inference_seconds_total",
            "inference_seconds_per_contract",
        }:
            row[field] = _format_number(value)
        else:
            row[field] = str(value)
    return row


def _aggregate_metric(payloads: Sequence[Mapping[str, Any]], name: str) -> dict[str, float]:
    values = [float(payload[name]) for payload in payloads]
    return {"mean": statistics.mean(values), "population_std": statistics.pstdev(values)}


def summarize_final_synthetic_eval(
    config_path: str | Path = DEFAULT_FINAL_EVAL_CONFIG,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Path]:
    """Pure deterministic summarization over one completed Phase 3A evaluation."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    spec = load_final_synthetic_eval_spec(_resolve(root, config_path))
    output_root = root / spec.output_root
    claim = _load_json(output_root / spec.claim_filename)
    dataset_manifest = _load_json(output_root / spec.dataset_manifest_filename)
    if claim.get("schema_version") != FINAL_CLAIM_SCHEMA:
        raise ValueError("Phase 3A claim schema mismatch")
    if dataset_manifest.get("schema_version") != FINAL_DATASET_SCHEMA:
        raise ValueError("Phase 3A dataset manifest schema mismatch")
    required_dataset_fields = {
        "seed", "count", "feature_names", "frozen_surfaces_sha256", "parameter_hash",
        "source_surface_id", "domain", "baseline_config_sha256", "pricing_node_count", "sample_ids",
        "sample_id_hash", "dataset_filename", "dataset_npz_sha256",
        "generation_code_git_sha", "final_holdout_only", "used_for_training",
        "used_for_validation",
    }
    missing_dataset_fields = required_dataset_fields - set(dataset_manifest)
    if missing_dataset_fields:
        raise ValueError(
            f"Phase 3A dataset manifest is incomplete: {sorted(missing_dataset_fields)}"
        )
    if dataset_manifest.get("seed") != spec.final_eval_seed:
        raise ValueError("Phase 3A dataset seed mismatch")
    if dataset_manifest.get("count") != spec.final_eval_count:
        raise ValueError("Phase 3A dataset count mismatch")
    if dataset_manifest.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("Phase 3A dataset feature identity mismatch")
    if len(dataset_manifest.get("sample_ids", [])) != spec.final_eval_count:
        raise ValueError("Phase 3A deterministic sample ID set is incomplete")
    if _stable_hash_strings(dataset_manifest["sample_ids"]) != dataset_manifest.get("sample_id_hash"):
        raise ValueError("Phase 3A sample ID hash mismatch")
    dataset_path = output_root / spec.dataset_filename
    if not dataset_path.is_file() or _sha256_file(dataset_path) != dataset_manifest.get(
        "dataset_npz_sha256"
    ):
        raise ValueError("Phase 3A persisted dataset SHA mismatch")
    with np.load(dataset_path, allow_pickle=False) as arrays:
        if set(arrays.files) != {"features", "reference_prices", "sample_ids"}:
            raise ValueError("Phase 3A persisted dataset arrays are malformed")
        features = np.asarray(arrays["features"], dtype=np.float64)
        references = np.asarray(arrays["reference_prices"], dtype=np.float64)
        sample_ids = np.asarray(arrays["sample_ids"], dtype=str)
    if (
        features.shape != (spec.final_eval_count, len(FEATURE_NAMES))
        or references.shape != (spec.final_eval_count,)
        or sample_ids.shape != (spec.final_eval_count,)
        or not np.isfinite(features).all()
        or not np.isfinite(references).all()
        or sample_ids.tolist() != dataset_manifest["sample_ids"]
    ):
        raise ValueError("Phase 3A persisted dataset content/manifest mismatch")
    if (
        dataset_manifest.get("final_holdout_only") is not True
        or dataset_manifest.get("used_for_training") is not False
        or dataset_manifest.get("used_for_validation") is not False
    ):
        raise ValueError("Phase 3A final-holdout role declaration mismatch")
    if claim.get("final_seed") != spec.final_eval_seed:
        raise ValueError("Phase 3A claim seed mismatch")
    if claim.get("final_cohort_sha256") != dataset_manifest.get("dataset_npz_sha256"):
        raise ValueError("Phase 3A claim/dataset SHA mismatch")
    if claim.get("sample_id_hash") != dataset_manifest.get("sample_id_hash"):
        raise ValueError("Phase 3A claim/sample identity mismatch")
    if claim.get("primary_run_ids") != list(PRIMARY_RUN_IDS):
        raise ValueError("Phase 3A claim primary set mismatch")
    if claim.get("secondary_ablation_run_ids") != list(SECONDARY_RUN_IDS):
        raise ValueError("Phase 3A claim secondary set mismatch")
    if not claim.get("current_git_sha"):
        raise ValueError("Phase 3A claim git identity is incomplete")
    if claim.get("current_git_sha") != dataset_manifest.get("generation_code_git_sha"):
        raise ValueError("Phase 3A claim/dataset generation git identity mismatch")
    claim_checkpoints = claim.get("checkpoint_identities")
    if not isinstance(claim_checkpoints, list) or [
        item.get("run_id") for item in claim_checkpoints
    ] != [item.run_id for item in spec.checkpoints]:
        raise ValueError("Phase 3A claim checkpoint matrix mismatch")
    if (
        claim.get("final_loss_choice") != "equal"
        or claim.get("loss_choice_frozen_before_final_evaluation") is not True
    ):
        raise ValueError("Phase 3A claim frozen loss-choice declaration mismatch")
    metrics_root = output_root / spec.metrics_subdirectory
    results: dict[str, dict[str, Any]] = {}
    for setting in spec.checkpoints:
        path = metrics_root / f"{setting.run_id}.json"
        if not path.is_file():
            raise ValueError(f"incomplete Phase 3A result: missing {path.name}")
        results[setting.run_id] = _validate_completed_metric(
            _load_json(path), setting, dataset_manifest, claim
        )

    run_summary_path = output_root / spec.run_summary_filename
    with run_summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(_summary_row(results[item.run_id]) for item in spec.checkpoints)

    primary_results = [results[run_id] for run_id in PRIMARY_RUN_IDS]
    primary_aggregate = {
        "schema_version": "mentor_dh_pinn_final_primary_aggregate_v1",
        "primary_run_ids": list(PRIMARY_RUN_IDS),
        "checkpoint_count": len(PRIMARY_RUN_IDS),
        "price_nrmse": _aggregate_metric(primary_results, "price_nrmse"),
        "price_rmse": _aggregate_metric(primary_results, "price_rmse"),
        "price_mae": _aggregate_metric(primary_results, "price_mae"),
        "pde_rms": _aggregate_metric(primary_results, "pde_rms"),
        "terminal_rmse": _aggregate_metric(primary_results, "terminal_rmse"),
        "boundary_low_s_rmse": _aggregate_metric(primary_results, "boundary_low_s_rmse"),
        "boundary_high_s_rmse": _aggregate_metric(primary_results, "boundary_high_s_rmse"),
        "all_finite": all(bool(item["all_finite"]) for item in primary_results),
        "individual_checkpoints": {
            run_id: results[run_id] for run_id in PRIMARY_RUN_IDS
        },
        "final_loss_choice": "equal",
        "loss_choice_frozen_before_final_evaluation": True,
        "model_selection_performed_on_final_holdout": False,
        "final_holdout_used_for_training": False,
        "final_holdout_used_for_validation": False,
        "inferential_significance_test_performed": False,
    }
    primary_path = output_root / spec.primary_aggregate_filename
    with primary_path.open("w", encoding="utf-8") as handle:
        json.dump(primary_aggregate, handle, indent=2, sort_keys=True)
        handle.write("\n")

    secondary_path = output_root / spec.secondary_ablation_filename
    with secondary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SECONDARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        for run_id in SECONDARY_RUN_IDS:
            row = _summary_row(results[run_id])
            row["secondary_ablation_only"] = "true"
            row["may_reopen_model_selection"] = "false"
            writer.writerow(row)

    manifest_path = output_root / spec.evaluation_manifest_filename
    artifact_paths = {
        "claim": output_root / spec.claim_filename,
        "dataset": output_root / spec.dataset_filename,
        "dataset_manifest": output_root / spec.dataset_manifest_filename,
        "run_summary": run_summary_path,
        "primary_aggregate": primary_path,
        "secondary_ablation": secondary_path,
        **{
            f"metrics_{run_id}": metrics_root / f"{run_id}.json"
            for run_id in (*PRIMARY_RUN_IDS, *SECONDARY_RUN_IDS)
        },
    }
    manifest = {
        "schema_version": "mentor_dh_pinn_final_synthetic_eval_manifest_v1",
        "final_cohort_sha256": dataset_manifest["dataset_npz_sha256"],
        "sample_id_hash": dataset_manifest["sample_id_hash"],
        "primary_run_ids": list(PRIMARY_RUN_IDS),
        "secondary_ablation_run_ids": list(SECONDARY_RUN_IDS),
        "artifact_sha256": {
            name: _sha256_file(path) for name, path in artifact_paths.items()
        },
        "final_loss_choice": "equal",
        "loss_choice_frozen_before_final_evaluation": True,
        "model_selection_performed_on_final_holdout": False,
        "final_holdout_used_for_training": False,
        "final_holdout_used_for_validation": False,
        "secondary_ablation_only": True,
        "may_reopen_model_selection": False,
        "inferential_significance_test_performed": False,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {
        "run_summary": run_summary_path,
        "primary_aggregate": primary_path,
        "secondary_ablation": secondary_path,
        "evaluation_manifest": manifest_path,
    }


__all__ = [
    "DEFAULT_FINAL_EVAL_CONFIG",
    "EXPECTED_CHECKPOINT_MATRIX",
    "FINAL_EVAL_COUNT",
    "FINAL_EVAL_SEED",
    "FinalSyntheticDataset",
    "PRIMARY_RUN_IDS",
    "SECONDARY_RUN_IDS",
    "build_common_diagnostics",
    "generate_final_synthetic_dataset",
    "load_final_synthetic_eval_spec",
    "run_final_synthetic_eval",
    "summarize_final_synthetic_eval",
    "validate_frozen_checkpoint_provenance",
]
