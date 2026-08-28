"""Strict configuration for the mentor-aligned forward PINN baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

CANONICAL_PARAMETER_ORDER = (
    "kappa_slow", "theta_slow", "sigma_slow", "rho_slow", "v0_slow",
    "kappa_fast", "theta_fast", "sigma_fast", "rho_fast", "v0_fast",
)
FEATURE_ORDER = ("spot", "v_slow", "v_fast", "tau", "strike", "r", "q")
NETWORK_INPUT_ORDER = ("S", "v_slow", "v_fast", "tau", "K", "r", "q")
FROZEN_SURFACES_SHA256 = "148b579a4f6ce572e34796e872479c4c016c89bbcd20438c2bb62d6b6960f1f6"


@dataclass(frozen=True)
class DomainConfig:
    spot_min: float = 0.70
    spot_max: float = 1.30
    moneyness_min: float = 0.70
    moneyness_max: float = 1.30
    tau_min: float = 7.0 / 365.0
    tau_max: float = 180.0 / 365.0
    rate_min: float = 0.01
    rate_max: float = 0.08
    carry_min: float = 0.0
    carry_max: float = 0.03
    variance_floor: float = 1.0e-4
    variance_ceiling: float = 0.30
    variance_theta_min_multiplier: float = 0.25
    variance_theta_max_multiplier: float = 2.0
    variance_state_rule: str = "max(floor, min_multiplier * theta) to min(ceiling, max_multiplier * theta)"
    boundary_spot_low: float = 1.0e-4
    boundary_spot_high: float = 2.00
    terminal_tau: float = 0.0

    @property
    def feature_min(self) -> tuple[float, ...]:
        return (self.boundary_spot_low, self.variance_floor, self.variance_floor,
                self.terminal_tau, self.spot_min * self.moneyness_min,
                self.rate_min, self.carry_min)

    @property
    def feature_max(self) -> tuple[float, ...]:
        return (self.boundary_spot_high, self.variance_ceiling, self.variance_ceiling,
                self.tau_max, self.spot_max * self.moneyness_max,
                self.rate_max, self.carry_max)

    def validate(self) -> None:
        for name, minimum, maximum in (
            ("spot", self.spot_min, self.spot_max),
            ("moneyness", self.moneyness_min, self.moneyness_max),
            ("tau", self.tau_min, self.tau_max),
            ("rate", self.rate_min, self.rate_max),
            ("carry", self.carry_min, self.carry_max),
        ):
            if not minimum < maximum:
                raise ValueError(f"{name} range must be strictly increasing")
        if not 0 < self.variance_floor < self.variance_ceiling:
            raise ValueError("variance floor/ceiling must be positive and increasing")
        if not 0 < self.variance_theta_min_multiplier < self.variance_theta_max_multiplier:
            raise ValueError("variance theta multipliers must be positive and increasing")
        if self.variance_state_rule != "max(floor, min_multiplier * theta) to min(ceiling, max_multiplier * theta)":
            raise ValueError("unexpected variance state rule")
        if self.boundary_spot_low <= 0 or self.boundary_spot_high <= 0:
            raise ValueError("boundary spots must be strictly positive")
        if self.boundary_spot_low >= self.spot_min or self.boundary_spot_high <= self.spot_max:
            raise ValueError("stock boundaries must lie outside the interior spot domain")
        if self.terminal_tau != 0.0:
            raise ValueError("terminal_tau must be exactly zero")


def variance_state_bounds(theta: float, domain: DomainConfig) -> tuple[float, float]:
    minimum = max(domain.variance_floor, domain.variance_theta_min_multiplier * float(theta))
    maximum = min(domain.variance_ceiling, domain.variance_theta_max_multiplier * float(theta))
    if minimum >= maximum:
        raise ValueError("variance state range has no positive width")
    return minimum, maximum


@dataclass(frozen=True)
class DataConfig:
    sampling_method: str = "seeded deterministic pseudo-random with independent split streams"
    frozen_surfaces_path: str = "data/final_r2_clean_10000/surfaces.jsonl"
    frozen_surfaces_sha256: str = FROZEN_SURFACES_SHA256
    train_count: int = 4096
    validation_count: int = 1024
    test_count: int = 1024
    split_id_prefix: str = "mentor_dh_pinn_v1"
    dataset_filename: str = "dataset.npz"
    manifest_filename: str = "dataset_manifest.json"
    parameter_provenance_filename: str = "parameter_provenance.json"

    @property
    def total_count(self) -> int:
        return self.train_count + self.validation_count + self.test_count

    def validate(self) -> None:
        if self.sampling_method != "seeded deterministic pseudo-random with independent split streams":
            raise ValueError("unexpected sampling method")
        if any(isinstance(value, bool) or value <= 0 for value in
               (self.train_count, self.validation_count, self.test_count)):
            raise ValueError("all split counts must be strictly positive integers")
        if self.frozen_surfaces_sha256.lower() != FROZEN_SURFACES_SHA256:
            raise ValueError("the frozen surfaces hash is part of the V1 contract")


@dataclass(frozen=True)
class NetworkConfig:
    input_size: int = 7
    input_order: tuple[str, ...] = NETWORK_INPUT_ORDER
    hidden_layers: int = 5
    hidden_width: int = 128
    activation: str = "tanh"
    dtype: str = "float64"
    output_kind: str = "raw_call_price"
    input_normalization: str = "affine map using frozen full loss-domain bounds"

    def validate(self) -> None:
        if self.input_size != 7 or self.hidden_layers != 5 or self.hidden_width != 128:
            raise ValueError("V1 network is fixed at 7 -> five 128-wide layers -> 1")
        if self.input_order != NETWORK_INPUT_ORDER:
            raise ValueError("unexpected V1 feature order")
        if self.activation.lower() != "tanh" or self.dtype != "float64" or self.output_kind != "raw_call_price":
            raise ValueError("V1 requires tanh, float64, raw call-price output")
        if self.input_normalization != "affine map using frozen full loss-domain bounds":
            raise ValueError("unexpected input normalization")


@dataclass(frozen=True)
class LossConfig:
    data_lambda: float = 1.0
    pde_lambda: float = 1.0
    boundary_lambda: float = 1.0
    terminal_lambda: float = 1.0
    pde_scale_floor: float = 1.0
    relative_error_epsilon: float = 1.0e-8
    data_formula: str = "mean(((predicted - reference) / S)^2)"
    pde_formula: str = "residual / max(abs(raw_price), pde_scale_floor)"

    @property
    def weights(self) -> dict[str, float]:
        return {"data": self.data_lambda, "pde": self.pde_lambda,
                "boundary": self.boundary_lambda, "terminal": self.terminal_lambda}

    def validate(self) -> None:
        if self.weights != {name: 1.0 for name in self.weights}:
            raise ValueError("all V1 loss lambdas must equal exactly 1.0")
        if self.pde_scale_floor <= 0 or self.relative_error_epsilon <= 0:
            raise ValueError("loss scale floor and relative-error epsilon must be positive")
        if self.data_formula != "mean(((predicted - reference) / S)^2)" or self.pde_formula != "residual / max(abs(raw_price), pde_scale_floor)":
            raise ValueError("unexpected loss formula")


@dataclass(frozen=True)
class TrainingConfig:
    device_policy: str = "operator-selected cpu or cuda; Kaggle P100 uses cuda"
    optimizer: str = "AdamW"
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-6
    batch_size: int = 256
    pde_batch_size: int = 256
    terminal_batch_size: int = 128
    boundary_batch_size: int = 128
    max_epochs: int = 1000
    patience: int = 100
    validation_frequency: int = 1
    checkpoint_metric: str = "validation_nrmse"
    checkpoint_rule: str = "minimum validation_nrmse only; test metrics forbidden"
    determinism_policy: str = "deterministic algorithms; CUBLAS_WORKSPACE_CONFIG=:4096:8 on CUDA"
    gradient_clip_norm: float | None = None
    collocation_seed_offset: int = 100_003
    terminal_seed_offset: int = 200_003
    low_boundary_seed_offset: int = 300_003
    high_boundary_seed_offset: int = 400_003
    epoch_seed_stride: int = 1009
    evaluation_seed_offset: int = 999_983

    def validate(self) -> None:
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer rates are invalid")
        if any(isinstance(value, bool) or value <= 0 for value in
               (self.batch_size, self.pde_batch_size, self.terminal_batch_size,
                self.boundary_batch_size, self.max_epochs, self.patience,
                self.validation_frequency)):
            raise ValueError("batch sizes and epoch controls must be positive")
        if self.optimizer.lower() != "adamw" or self.checkpoint_metric != "validation_nrmse":
            raise ValueError("V1 uses AdamW and validation normalized RMSE checkpointing")
        if self.device_policy != "operator-selected cpu or cuda; Kaggle P100 uses cuda":
            raise ValueError("unexpected device policy")
        if self.checkpoint_rule != "minimum validation_nrmse only; test metrics forbidden":
            raise ValueError("unexpected checkpoint rule")
        if self.determinism_policy != "deterministic algorithms; CUBLAS_WORKSPACE_CONFIG=:4096:8 on CUDA":
            raise ValueError("unexpected determinism policy")


@dataclass(frozen=True)
class EvaluationConfig:
    test_evaluation_mode: str = "explicit_operator_invocation_once"
    metrics_filename: str = "test_metrics.json"
    summary_filename: str = "MENTOR_SUMMARY.md"
    figures_subdirectory: str = "figures"
    figure_dpi: int = 300
    figure_color: str = "grayscale"
    slice_spot: float = 1.0
    slice_rate: float = 0.04
    slice_carry: float = 0.01
    slice_maturity_days: tuple[int, ...] = (30, 90, 180)
    marker_maturity_days: tuple[int, ...] = (7, 30, 60, 90, 120, 180)
    dense_grid_count: int = 61
    marker_strike_count: int = 13

    def validate(self) -> None:
        if self.test_evaluation_mode != "explicit_operator_invocation_once":
            raise ValueError("test evaluation must be explicit and one-shot")
        if self.figure_dpi != 300 or self.figure_color.lower() != "grayscale":
            raise ValueError("V1 figures must be grayscale at 300 dpi")
        if self.slice_spot <= 0 or self.dense_grid_count < 2 or self.marker_strike_count < 2:
            raise ValueError("invalid scientific figure slice")
        if not self.slice_maturity_days or not self.marker_maturity_days:
            raise ValueError("maturity slices must be non-empty")


_SCIENTIFIC_CONTRACT = {
    "frozen_surfaces_sha256": FROZEN_SURFACES_SHA256,
    "pde_operator": "src.model3_pde.operator.double_heston_pde_residual",
    "pde_cross_derivatives": "spot-v_slow and spot-v_fast only; no v_slow-v_fast term",
    "parameter_state_rule": "keep eight structural parameters fixed and substitute current v_slow/v_fast into v0_slow/v0_fast for pricing",
    "test_policy": "evaluate once through the explicit evaluator after training",
    "test_isolation_declaration": "test rows are forbidden in training, lambda selection, and checkpoint selection",
}


@dataclass(frozen=True)
class BaselineConfig:
    experiment_id: str = "mentor_dh_pinn_baseline_v1"
    seed: int = 3407
    option_type: str = "call"
    pricing_node_count: int = 64
    parameter_source_record: str = "first_eligible_stored_train_record"
    expected_parameter_hash: str = "b262584d6c2e76f7b46d635f580a5a00c7cb195e28a3536f18303e856704e8cd"
    canonical_parameter_order: tuple[str, ...] = CANONICAL_PARAMETER_ORDER
    feature_order: tuple[str, ...] = FEATURE_ORDER
    scientific_contract: dict[str, str] = field(default_factory=lambda: dict(_SCIENTIFIC_CONTRACT))
    data: DataConfig = field(default_factory=DataConfig)
    domain: DomainConfig = field(default_factory=DomainConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    losses: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self) -> None:
        if self.experiment_id != "mentor_dh_pinn_baseline_v1" or self.seed != 3407:
            raise ValueError("unexpected experiment identity or seed")
        if self.option_type != "call" or self.pricing_node_count != 64:
            raise ValueError("V1 supports CALL only with 64 pricing nodes")
        if self.parameter_source_record != "first_eligible_stored_train_record":
            raise ValueError("unexpected parameter selection rule")
        if self.canonical_parameter_order != CANONICAL_PARAMETER_ORDER or self.feature_order != FEATURE_ORDER:
            raise ValueError("unexpected canonical parameter or feature order")
        if self.scientific_contract != _SCIENTIFIC_CONTRACT:
            raise ValueError("unexpected scientific contract")
        self.data.validate(); self.domain.validate(); self.network.validate()
        self.losses.validate(); self.training.validate(); self.evaluation.validate()

    def to_dict(self) -> dict[str, Any]:
        # Normalize tuples to JSON arrays so YAML, manifests, and checkpoints
        # compare byte-for-byte after serialization round trips.
        return json.loads(json.dumps(asdict(self)))

    def with_overrides(self, **overrides: Any) -> "BaselineConfig":
        data, training = self.data, self.training
        for name in ("train_count", "validation_count", "test_count"):
            if name in overrides:
                data = replace(data, **{name: int(overrides.pop(name))})
        for name in ("max_epochs", "patience"):
            if name in overrides:
                training = replace(training, **{name: int(overrides.pop(name))})
        if overrides:
            raise TypeError(f"unknown baseline override(s): {', '.join(sorted(overrides))}")
        result = replace(self, data=data, training=training)
        result.validate()
        return result


def _strict_values(cls: type[Any], values: Any, section: str) -> dict[str, Any]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{section} must be a mapping")
    expected = {item.name for item in fields(cls) if item.init}
    actual = set(values)
    unknown, missing = actual - expected, expected - actual
    if unknown or missing:
        details = []
        if unknown: details.append(f"unknown={sorted(unknown)}")
        if missing: details.append(f"missing={sorted(missing)}")
        raise ValueError(f"invalid {section} fields: {', '.join(details)}")
    return dict(values)


def baseline_config_from_mapping(mapping: Mapping[str, Any]) -> BaselineConfig:
    """Build a config while rejecting every unknown or missing field."""
    top = _strict_values(BaselineConfig, mapping, "baseline")
    for name, cls in (("data", DataConfig), ("domain", DomainConfig),
                      ("network", NetworkConfig), ("losses", LossConfig),
                      ("training", TrainingConfig), ("evaluation", EvaluationConfig)):
        values = _strict_values(cls, top[name], name)
        if name == "network": values["input_order"] = tuple(values["input_order"])
        if name == "evaluation":
            values["slice_maturity_days"] = tuple(values["slice_maturity_days"])
            values["marker_maturity_days"] = tuple(values["marker_maturity_days"])
        top[name] = cls(**values)
    top["canonical_parameter_order"] = tuple(top["canonical_parameter_order"])
    top["feature_order"] = tuple(top["feature_order"])
    top["scientific_contract"] = dict(top["scientific_contract"])
    result = BaselineConfig(**top)
    result.validate()
    return result


def load_baseline_config(path: str | Path | None = None) -> BaselineConfig:
    config_path = Path(path) if path is not None else Path(__file__).resolve().parents[2] / "configs" / "mentor_dh_pinn" / "baseline_v1.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        mapping = yaml.safe_load(handle) or {}
    if not isinstance(mapping, Mapping):
        raise TypeError("baseline YAML must contain a mapping")
    return baseline_config_from_mapping(mapping)


load_config = load_baseline_config

__all__ = ["BaselineConfig", "CANONICAL_PARAMETER_ORDER", "FEATURE_ORDER", "NETWORK_INPUT_ORDER",
           "FROZEN_SURFACES_SHA256", "baseline_config_from_mapping",
           "load_baseline_config", "load_config", "variance_state_bounds"]
