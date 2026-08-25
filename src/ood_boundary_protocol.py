"""Deterministic pre-outcome OOD/boundary cohort contract and generator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import yaml
from scipy.stats import qmc

from .audit_reviewed_sampling import (
    _lhs,
    _row,
    load_reviewed_config,
    sample_challenge,
    sample_distribution,
)
from .constants import PARAMETER_NAMES
from .constraints import validate_parameters
from .r2_representation import (
    CANONICAL_SLOT_KEYS,
    NOMINAL_SLOT_COUNT,
    REPRESENTATION_NAME,
    REPRESENTATION_VERSION,
    R2Conditioning,
    R2Surface,
    build_synthetic_surface,
    payload_to_surface,
    surface_to_payload,
    validate_payload,
)
from .r2_synthetic_generation import parameter_vector_hash


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "ood_boundary_protocol.yaml"
REVIEWED_CONFIG_PATH = ROOT / "configs" / "parameter_sampling_REVIEWED.yaml"
BOUNDS_CONFIG_PATH = ROOT / "configs" / "parameter_bounds_PROVISIONAL.yaml"
PRIMARY_CONFIG_PATH = ROOT / "configs" / "r2_primary_comparison_FINAL.yaml"
DATASET_PATH = ROOT / "data" / "final_r2_clean_10000" / "surfaces.jsonl"
FREEZE_MARKER_PATH = ROOT / "evidence" / "OOD_BOUNDARY_PROTOCOL_FROZEN.txt"
COHORT_ORDER = (
    "boundary_challenge",
    "distribution_shift",
    "maturity_conditioning_shift",
)
BOUNDARY_REGIMES = (
    "near_feller",
    "weak_separation",
    "near_hard_bound",
    "near_correlation_disk",
)
SHIFT_REGIMES = (
    "slow_low_mean_reversion_high_variance",
    "fast_high_mean_reversion",
)
MISSING_PATTERNS = (
    "rank1_only",
    "rank2_only",
    "central_three_moneyness",
    "calls_only",
    "even_slot_checkerboard",
)
REQUIRED_ARTIFACTS = (
    "development_sanity_panel.csv",
    "boundary_candidates.csv",
    "distribution_shift_candidates.csv",
    "maturity_conditioning_shift_candidates.csv",
    "selected_parameters.csv",
    "clean_surfaces.jsonl",
    "incomplete_surfaces.jsonl",
    "all_research_surfaces.jsonl",
    "numerical_sanity.jsonl",
    "pricing_failures.jsonl",
    "integrity_report.json",
    "manifest.json",
)


class OODProtocolError(RuntimeError):
    """Raised when the frozen OOD/boundary contract cannot be honored."""


class RemoteCheckpointRequired(OODProtocolError):
    """Raised when generation is requested before push verification."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def deterministic_json_bytes(payload: Any) -> bytes:
    text = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    )
    return (text + "\n").encode("utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _jsonl_bytes(payloads: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(deterministic_json_bytes(payload) for payload in payloads)


def _write_bytes(path: str | Path, payload: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)


def _write_csv(frame: pd.DataFrame, path: str | Path) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False, lineterminator="\n", float_format="%.17g")
    return sha256_file(target)


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
        default=_json_default,
    ) + "\n"
    target.write_text(text, encoding="utf-8", newline="\n")
    return sha256_file(target)


def load_protocol_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Load and fully validate the frozen protocol identity and settings."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if config.get("protocol_name") != "DOUBLE_HESTON_OOD_BOUNDARY_ROBUSTNESS_V1":
        raise OODProtocolError("unexpected OOD protocol name")
    if config.get("status") != "FROZEN_BEFORE_MODEL3_RESEARCH_RESULTS":
        raise OODProtocolError("OOD protocol is not frozen")
    if config.get("base_commit_sha256") != (
        "72ad8e1aa845ec4c6f0fc61fc526df75438639bb"
    ):
        raise OODProtocolError("unexpected OOD protocol base commit")

    representation = config["representation"]
    if representation["name"] != REPRESENTATION_NAME or representation[
        "version"
    ] != REPRESENTATION_VERSION:
        raise OODProtocolError("representation identity drift")
    if int(representation["nominal_slot_count"]) != NOMINAL_SLOT_COUNT != 20:
        raise OODProtocolError("nominal slot-count drift")
    if int(representation["input_size"]) != 5 * NOMINAL_SLOT_COUNT:
        raise OODProtocolError("neural input-size drift")

    authority = config["authority"]
    if not DATASET_PATH.is_file():
        raise OODProtocolError("frozen R2 dataset is missing")
    dataset_hash = sha256_file(DATASET_PATH)
    if dataset_hash != authority["r2_dataset"]["sha256"]:
        raise OODProtocolError("frozen R2 dataset hash mismatch")
    if sha256_file(PRIMARY_CONFIG_PATH).lower() != authority[
        "primary_comparison_config"
    ]["sha256"].lower():
        raise OODProtocolError("primary comparison config hash mismatch")
    for label, item in authority["primary_baseline_artifacts"].items():
        artifact_path = ROOT / item["path"]
        if not artifact_path.is_file():
            raise OODProtocolError(f"missing primary baseline artifact: {label}")
        if sha256_file(artifact_path).lower() != item["sha256"].lower():
            raise OODProtocolError(f"primary baseline hash mismatch: {label}")

    leakage = config["leakage_controls"]
    required_false = {
        "model3_results_used_in_design",
        "real_market_inputs_allowed",
        "real_market_weight_updates_allowed",
        "frozen_primary_test_metrics_opened_during_design",
        "frozen_primary_test_metrics_may_tune_thresholds",
    }
    if any(leakage.get(field) is not False for field in required_false):
        raise OODProtocolError("leakage-control gate drift")
    if leakage.get("completed_primary_files_modifiable") is not False:
        raise OODProtocolError("completed-primary protection drift")

    cohorts = config["frozen_cohorts"]
    expected_counts = {
        "boundary_challenge": 120,
        "distribution_shift": 120,
        "maturity_conditioning_shift": 120,
        "incomplete_observation": 60,
    }
    observed_counts = {
        name: int(item.get("surface_count", item.get("derived_surface_count")))
        for name, item in cohorts.items()
    }
    if observed_counts != expected_counts:
        raise OODProtocolError("research-cohort quota drift")
    if list(cohorts["boundary_challenge"]["regimes"]) != list(BOUNDARY_REGIMES):
        raise OODProtocolError("boundary-regime drift")
    if any(int(item["count"]) != 30 for item in cohorts[
        "boundary_challenge"
    ]["regimes"].values()):
        raise OODProtocolError("boundary-regime quota drift")
    if list(cohorts["distribution_shift"]["regimes"]) != list(SHIFT_REGIMES):
        raise OODProtocolError("distribution-shift regime drift")
    if any(int(item["count"]) != 60 for item in cohorts[
        "distribution_shift"
    ]["regimes"].values()):
        raise OODProtocolError("distribution-shift quota drift")
    if list(cohorts["incomplete_observation"]["patterns_cycle"]) != list(
        MISSING_PATTERNS
    ):
        raise OODProtocolError("missingness-pattern drift")

    _validate_hard_bounds_against_reviewed(config)
    metrics = config["metrics"]
    if metrics.get("frozen_before_model3_results") is not True:
        raise OODProtocolError("metric freeze flag drift")
    degradation = metrics["degradation"]
    if float(degradation["material_relative_degradation_ratio"]) != 1.25:
        raise OODProtocolError("post-hoc materiality threshold drift")
    if float(degradation["ratio_denominator_floor"]) <= 0.0:
        raise OODProtocolError("degradation denominator floor must be positive")
    gates = config["execution_gates"]
    if gates.get(
        "protocol_commit_push_and_remote_verification_before_generation"
    ) != "REQUIRED":
        raise OODProtocolError("remote checkpoint gate drift")
    if "FORBIDDEN" not in str(gates.get("expensive_method_evaluations_this_milestone")):
        raise OODProtocolError("expensive-evaluation gate drift")
    if "FORBIDDEN" not in str(gates.get("neural_training_or_fine_tuning")):
        raise OODProtocolError("training gate drift")
    return config


def _validate_hard_bounds_against_reviewed(config: dict[str, Any]) -> None:
    reviewed = load_reviewed_config(REVIEWED_CONFIG_PATH)
    bounds_yaml = yaml.safe_load(BOUNDS_CONFIG_PATH.read_text(encoding="utf-8"))
    hard_bounds = bounds_yaml["hard_numerical_safety_bounds"]
    if set(hard_bounds) != set(PARAMETER_NAMES):
        raise OODProtocolError("hard-bound parameter-name drift")
    for name in PARAMETER_NAMES:
        declared = reviewed["hard_constraints"][name]
        configured = hard_bounds[name]
        if float(declared["lower"]) != float(configured["lower"]) or float(
            declared["upper"]
        ) != float(configured["upper"]):
            raise OODProtocolError(f"hard-bound drift for {name}")
    parameter_order = config["parameter_contract"]["order"]
    if parameter_order != PARAMETER_NAMES:
        raise OODProtocolError("canonical parameter-order drift")


def require_remote_checkpoint(confirmed: bool) -> None:
    """Fail closed unless the committed marker and operator confirmation exist."""
    if not FREEZE_MARKER_PATH.is_file():
        raise OODProtocolError("OOD protocol freeze marker is missing")
    marker = FREEZE_MARKER_PATH.read_text(encoding="utf-8")
    required = (
        "NO OOD RESEARCH SURFACE, SELECTED PANEL, COHORT MANIFEST, OR MODEL3 "
        "PREDICTION"
    )
    if required not in marker:
        raise OODProtocolError("OOD freeze-marker statement is invalid")
    if not confirmed:
        raise RemoteCheckpointRequired(
            "generation requires a pushed, remote-verified checkpoint and "
            "--remote-checkpoint-confirmed"
        )


def _load_frozen_parameter_hashes() -> set[str]:
    hashes: set[str] = set()
    with DATASET_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            stored = record["metadata"]["parameters_canonical_order"]
            vector = [float(stored[name]) for name in PARAMETER_NAMES]
            hashes.add(parameter_vector_hash(dict(zip(PARAMETER_NAMES, vector))))
    return hashes


def _add_eligibility(
    frame: pd.DataFrame, frozen_hashes: set[str]
) -> pd.DataFrame:
    result = frame.copy()
    vectors = result[PARAMETER_NAMES].astype(float)
    vector_hashes: list[str] = []
    finite_rows: list[bool] = []
    for row in vectors.itertuples(index=False, name=None):
        vector = np.asarray([float(value) for value in row], dtype=np.float64)
        is_finite = bool(np.isfinite(vector).all())
        finite_rows.append(is_finite)
        vector_hashes.append(
            parameter_vector_hash(dict(zip(PARAMETER_NAMES, vector)))
            if is_finite
            else ""
        )
    result["parameter_vector_hash"] = vector_hashes
    structural = result["accepted"].fillna(False).astype(bool)
    structural &= pd.Series(finite_rows, index=result.index)
    if "hard_bounds_valid" in result:
        structural &= result["hard_bounds_valid"].fillna(False).astype(bool)
    if "canonical_valid" in result:
        structural &= result["canonical_valid"].fillna(False).astype(bool)
    result["protocol_eligible"] = [
        bool(valid and vector_hash not in frozen_hashes)
        for valid, vector_hash in zip(structural, result["parameter_vector_hash"])
    ]
    result.loc[~result["protocol_eligible"], "protocol_exclusion_reason"] = (
        result.loc[~result["protocol_eligible"], "primary_rejection_reason"].fillna(
            ""
        ).where(
            ~structural.loc[~result["protocol_eligible"]],
            "overlap_with_frozen_r2_parameter_vectors",
        )
    )
    return result


def build_parameter_pools(
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Build every fixed candidate pool and select the frozen quotas."""
    config = config or load_protocol_config()
    reviewed = load_reviewed_config(REVIEWED_CONFIG_PATH)
    frozen_hashes = _load_frozen_parameter_hashes()
    pools: dict[str, pd.DataFrame] = {}
    selected_parts: list[pd.DataFrame] = []

    boundary_spec = config["frozen_cohorts"]["boundary_challenge"]
    boundary = sample_challenge(
        count=int(boundary_spec["candidate_pool_count"]),
        seed=int(boundary_spec["parameter_seed"]),
        config=reviewed,
    )
    boundary = _add_eligibility(boundary, frozen_hashes)
    pools["boundary_challenge"] = boundary
    boundary_quota = int(boundary_spec["required_quota"])
    chosen_boundary: list[pd.DataFrame] = []
    for regime in BOUNDARY_REGIMES:
        part = boundary.loc[
            (boundary["regime"] == regime) & boundary["protocol_eligible"]
        ].sort_values("candidate_id", kind="mergesort")
        wanted = int(boundary_spec["regimes"][regime]["count"])
        if len(part) < wanted:
            raise OODProtocolError(
                f"fixed boundary pool has {len(part)} eligible {regime} rows; "
                f"{wanted} required and refill is forbidden"
            )
        chosen = part.iloc[:wanted].copy()
        chosen["cohort"] = "boundary_challenge"
        chosen_boundary.append(chosen)
    selected_parts.append(pd.concat(chosen_boundary, ignore_index=True))

    shift_spec = config["frozen_cohorts"]["distribution_shift"]
    shift_rows: list[dict[str, Any]] = []
    shift_count = int(shift_spec["candidate_pool_count"])
    shift_seed = int(shift_spec["parameter_seed"])
    for candidate_id, latent in enumerate(_lhs(shift_count, shift_seed)):
        regime = SHIFT_REGIMES[candidate_id % len(SHIFT_REGIMES)]
        overrides = dict(shift_spec["regimes"][regime]["overrides"])
        try:
            row = _row(
                candidate_id,
                latent,
                "wide_valid_train",
                reviewed,
                regime,
                overrides,
            )
        except ValueError as error:
            row = {
                **{name: float("nan") for name in PARAMETER_NAMES},
                "candidate_id": candidate_id,
                "distribution": "wide_valid_train",
                "regime": regime,
                "attempt": 0,
                "accepted": False,
                "primary_rejection_reason": str(error),
                "rejection_reasons": str(error),
            }
        shift_rows.append(row)
    shift = _add_eligibility(pd.DataFrame(shift_rows), frozen_hashes)
    pools["distribution_shift"] = shift
    chosen_shift: list[pd.DataFrame] = []
    for regime in SHIFT_REGIMES:
        part = shift.loc[
            (shift["regime"] == regime) & shift["protocol_eligible"]
        ].sort_values("candidate_id", kind="mergesort")
        wanted = int(shift_spec["regimes"][regime]["count"])
        if len(part) < wanted:
            raise OODProtocolError(
                f"fixed distribution-shift pool has {len(part)} eligible "
                f"{regime} rows; {wanted} required and refill is forbidden"
            )
        chosen = part.iloc[:wanted].copy()
        chosen["cohort"] = "distribution_shift"
        chosen_shift.append(chosen)
    selected_parts.append(pd.concat(chosen_shift, ignore_index=True))

    maturity_spec = config["frozen_cohorts"]["maturity_conditioning_shift"]
    maturity = sample_distribution(
        "wide_valid_train",
        count=int(maturity_spec["candidate_pool_count"]),
        seed=int(maturity_spec["parameter_seed"]),
        config=reviewed,
    )
    maturity = _add_eligibility(maturity, frozen_hashes)
    pools["maturity_conditioning_shift"] = maturity
    maturity_selected = maturity.loc[maturity["protocol_eligible"]].sort_values(
        "candidate_id", kind="mergesort"
    )
    wanted = int(maturity_spec["required_quota"])
    if len(maturity_selected) < wanted:
        raise OODProtocolError(
            f"fixed maturity parameter pool has {len(maturity_selected)} "
            f"eligible rows; {wanted} required and refill is forbidden"
        )
    chosen_maturity = maturity_selected.iloc[:wanted].copy()
    chosen_maturity["cohort"] = "maturity_conditioning_shift"
    chosen_maturity["regime"] = "accepted_wide_valid_margin_screen"
    selected_parts.append(chosen_maturity)

    selected = pd.concat(selected_parts, ignore_index=True)
    if len(selected) != 360:
        raise OODProtocolError("selected parameter-panel count drift")
    if selected["parameter_vector_hash"].duplicated().any():
        raise OODProtocolError("duplicate selected parameter vector")
    return pools, selected


def build_development_panel(selected: pd.DataFrame) -> pd.DataFrame:
    """Build the explicitly non-research 12-row sampler sanity panel."""
    parts: list[pd.DataFrame] = []
    for cohort in COHORT_ORDER:
        part = selected.loc[selected["cohort"] == cohort].iloc[:4].copy()
        parts.append(part)
    development = pd.concat(parts, ignore_index=True)
    if len(development) != 12:
        raise OODProtocolError("development-sanity panel count drift")
    development.insert(0, "development_label", "DEVELOPMENT_SANITY_NOT_RESEARCH_RESULT")
    for _, row in development.iterrows():
        vector = np.asarray([float(row[name]) for name in PARAMETER_NAMES])
        if not validate_parameters(vector)["is_valid"]:
            raise OODProtocolError("development-sanity vector is invalid")
    return development


def _standard_conditioning(
    cohort: str, generation_index: int, config: dict[str, Any]
) -> tuple[R2Conditioning, dict[str, Any]]:
    spec = config["frozen_cohorts"][cohort]["conditioning"]
    rank1_values = [7, 14, 21, 30, 45, 60, 75, 90]
    gap_values = [7, 14, 21, 30, 45, 60, 90]
    rate_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    offset_values = [-0.02, -0.01, 0.00, 0.01, 0.02, 0.03]
    lattice_size = 2016
    return _mixed_radix_conditioning(
        cohort,
        generation_index,
        spec,
        rank1_values,
        gap_values,
        rate_values,
        offset_values,
        lattice_size,
    )


def _shifted_conditioning(
    generation_index: int, config: dict[str, Any]
) -> tuple[R2Conditioning, dict[str, Any]]:
    spec = config["frozen_cohorts"]["maturity_conditioning_shift"]["conditioning"]
    return _mixed_radix_conditioning(
        "maturity_conditioning_shift",
        generation_index,
        spec,
        [int(value) for value in spec["rank1_dte_days"]],
        [int(value) for value in spec["rank2_gap_dte_days"]],
        [float(value) for value in spec["rates"]],
        [float(value) for value in spec["carry_offsets"]],
        int(spec["combination_count"]),
    )


def _mixed_radix_conditioning(
    cohort: str,
    generation_index: int,
    spec: Mapping[str, Any],
    rank1_values: list[int],
    gap_values: list[int],
    rate_values: list[float],
    offset_values: list[float],
    lattice_size: int,
) -> tuple[R2Conditioning, dict[str, Any]]:
    stride = int(spec["stride"])
    if math.gcd(stride, lattice_size) != 1:
        raise OODProtocolError(f"{cohort} conditioning stride is not coprime")
    seed = int(spec["seed"])
    index = (int(generation_index) * stride) % lattice_size
    dte1 = rank1_values[index % len(rank1_values)]
    gap = gap_values[(index // len(rank1_values)) % len(gap_values)]
    rate_index = index // (len(rank1_values) * len(gap_values))
    rate = rate_values[rate_index % len(rate_values)]
    offset_index = rate_index // len(rate_values)
    carry_offset = offset_values[offset_index % len(offset_values)]
    carry = rate + carry_offset
    dte2 = dte1 + gap
    conditioning = R2Conditioning(
        date_id=f"SYNTHETIC_OOD_{cohort.upper()}_{generation_index:06d}",
        spot=float(spec.get("spot", 100.0)),
        expiry_dates=("SYNTHETIC_RANK_1", "SYNTHETIC_RANK_2"),
        dte=(dte1, dte2),
        rates=(rate, rate),
        carries=(carry, carry),
    )
    provenance = {
        "seed": seed,
        "stride": stride,
        "lattice_size": lattice_size,
        "lattice_index": index,
        "generation_index": generation_index,
        "rank1_dte_days": dte1,
        "rank2_gap_dte_days": gap,
        "rank2_dte_days": dte2,
        "rate": rate,
        "carry_offset": carry_offset,
        "carry": carry,
        "real_market_inputs_used": False,
    }
    return conditioning, provenance


def _conditioning_for(
    cohort: str, generation_index: int, config: dict[str, Any]
) -> tuple[R2Conditioning, dict[str, Any]]:
    if cohort == "maturity_conditioning_shift":
        return _shifted_conditioning(generation_index, config)
    return _standard_conditioning(cohort, generation_index, config)


def _surface_provenance(
    row: Mapping[str, Any],
    conditioning_provenance: Mapping[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    cohort = str(row["cohort"])
    parameter_seed = int(config["frozen_cohorts"][cohort]["parameter_seed"])
    return {
        "dataset_status": "FROZEN_OOD_BOUNDARY_EVALUATION_ONLY_V1",
        "cohort": cohort,
        "split": "evaluation_only_not_train_validation_test",
        "regime": str(row["regime"]),
        "candidate_id": int(row["candidate_id"]),
        "parameter_vector_hash": str(row["parameter_vector_hash"]),
        "parameter_sampler_seed": parameter_seed,
        "conditioning": dict(conditioning_provenance),
        "protocol_name": config["protocol_name"],
        "protocol_status": config["status"],
        "evaluation_only": True,
        "train_validation_eligible": False,
        "real_market_inputs_used": False,
        "noise_level": 0.0,
    }


def _arbitrage_and_shape_checks(payload: Mapping[str, Any]) -> dict[str, Any]:
    prices = np.asarray(payload["prices"], dtype=np.float64)
    mask = np.asarray(payload["mask"], dtype=bool)
    spot = float(payload["spot"])
    lower = np.zeros(NOMINAL_SLOT_COUNT, dtype=np.float64)
    upper = np.zeros(NOMINAL_SLOT_COUNT, dtype=np.float64)
    calls = np.zeros(NOMINAL_SLOT_COUNT, dtype=bool)
    for index, key in enumerate(CANONICAL_SLOT_KEYS):
        strike = spot * math.exp(key.target_log_moneyness)
        maturity = float(payload["maturities"][index])
        rate = float(payload["rates"][index])
        carry = float(payload["carries"][index])
        discount_spot = spot * math.exp(-carry * maturity)
        discount_strike = strike * math.exp(-rate * maturity)
        intrinsic = (
            max(discount_spot - discount_strike, 0.0)
            if key.option_type == "call"
            else max(discount_strike - discount_spot, 0.0)
        )
        lower[index] = intrinsic
        upper[index] = (
            discount_spot if key.option_type == "call" else discount_strike
        )
        calls[index] = key.option_type == "call"
    valid = mask
    errors = prices[valid]
    low = lower[valid]
    high = upper[valid]
    call_prices = prices[valid & calls]
    put_prices = prices[valid & ~calls]
    # The clean generator supplies complete surfaces. Keep the shape guard so
    # this numerical validator cannot silently reinterpret a partial input.
    call_monotonic = bool(
        len(call_prices) == 10
        and np.all(np.diff(call_prices.reshape(2, 5), axis=1) <= 1e-9)
    )
    put_monotonic = bool(
        len(put_prices) == 10
        and np.all(np.diff(put_prices.reshape(2, 5), axis=1) >= -1e-9)
    )
    call_convex = bool(
        len(call_prices) == 10
        and np.all(np.diff(call_prices.reshape(2, 5), n=2, axis=1) >= -1e-8)
    )
    put_convex = bool(
        len(put_prices) == 10
        and np.all(np.diff(put_prices.reshape(2, 5), n=2, axis=1) >= -1e-8)
    )
    return {
        "finite_prices": bool(np.isfinite(prices).all()),
        "all_clean_masks_true": bool(mask.all()),
        "all_clean_prices_positive": bool(np.all(prices > 0.0)),
        "no_arbitrage_valid": bool(
            np.all(errors >= low - 1e-8) and np.all(errors <= high + 1e-8)
        ),
        "strike_monotonicity_valid": bool(call_monotonic and put_monotonic),
        "strike_convexity_valid": bool(call_convex and put_convex),
    }


def generate_clean_surfaces(
    selected: pd.DataFrame, config: dict[str, Any]
) -> tuple[list[R2Surface], list[dict[str, Any]], list[dict[str, Any]]]:
    """Price all selected truths once through the unchanged production pricer."""
    surfaces: list[R2Surface] = []
    sanity_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    def retained_failure(
        row: Mapping[str, Any],
        surface_id: str,
        conditioning_provenance: Mapping[str, Any],
        error_type: str,
        error_message: str,
    ) -> dict[str, Any]:
        return {
            "surface_id": surface_id,
            "candidate_id": int(row["candidate_id"]),
            "cohort": str(row["cohort"]),
            "regime": str(row["regime"]),
            "parameter_vector_hash": str(row["parameter_vector_hash"]),
            "parameters": {
                name: float(row[name]) for name in PARAMETER_NAMES
            },
            "conditioning": dict(conditioning_provenance),
            "error_type": error_type,
            "error_message": error_message,
        }

    for generation_index, (_, row) in enumerate(selected.iterrows()):
        cohort = str(row["cohort"])
        surface_id = f"OODV1_{cohort.upper()}_{generation_index:06d}"
        conditioning, conditioning_provenance = _conditioning_for(
            cohort, generation_index, config
        )
        vector = np.asarray([float(row[name]) for name in PARAMETER_NAMES])
        metadata = _surface_provenance(row, conditioning_provenance, config)
        try:
            surface = build_synthetic_surface(
                vector,
                conditioning,
                surface_id=surface_id,
                metadata=metadata,
                node_count=int(config["pricing"]["node_count"]),
            )
            payload = surface_to_payload(surface)
            sanity = {"surface_id": surface_id, **_arbitrage_and_shape_checks(payload)}
            sanity["serialization_round_trip"] = surface_to_payload(
                payload_to_surface(payload)
            ) == payload
            passed = all(value is True for key, value in sanity.items() if key != "surface_id")
            sanity["passed"] = passed
            if not passed:
                failures.append(
                    retained_failure(
                        row,
                        surface_id,
                        conditioning_provenance,
                        "NUMERICAL_SANITY_FAILURE",
                        json.dumps(sanity, sort_keys=True),
                    )
                )
            else:
                surfaces.append(surface)
            sanity_rows.append(sanity)
        except Exception as error:
            failures.append(
                retained_failure(
                    row,
                    surface_id,
                    conditioning_provenance,
                    type(error).__name__,
                    str(error),
                )
            )
    if len(surfaces) + len(failures) != 360 or len(sanity_rows) + len(
        failures
    ) != 360:
        raise OODProtocolError("clean surface count drift")
    return surfaces, sanity_rows, failures


def _mask_pattern(pattern: str) -> np.ndarray:
    mask = np.ones(NOMINAL_SLOT_COUNT, dtype=bool)
    if pattern == "rank1_only":
        mask[[key.expiry_rank == 2 for key in CANONICAL_SLOT_KEYS]] = False
    elif pattern == "rank2_only":
        mask[[key.expiry_rank == 1 for key in CANONICAL_SLOT_KEYS]] = False
    elif pattern == "central_three_moneyness":
        mask[[abs(key.target_log_moneyness) > 0.05 for key in CANONICAL_SLOT_KEYS]] = False
    elif pattern == "calls_only":
        mask[[key.option_type != "call" for key in CANONICAL_SLOT_KEYS]] = False
    elif pattern == "even_slot_checkerboard":
        mask[np.arange(NOMINAL_SLOT_COUNT) % 2 == 1] = False
    else:
        raise OODProtocolError(f"unknown missingness pattern: {pattern}")
    if int(mask.sum()) < 10:
        raise OODProtocolError(f"pattern {pattern} has fewer than ten usable slots")
    return mask


def make_incomplete_surface(
    parent: R2Surface,
    *,
    pattern: str,
    surface_id: str,
    sequence_index: int,
) -> R2Surface:
    """Derive one input challenge using exact frozen R2 mask semantics."""
    prices = parent.prices_array().copy()
    mask = _mask_pattern(pattern)
    prices[~mask] = 0.0
    metadata = dict(parent.metadata)
    user_metadata = dict(metadata.get("user_metadata", {}))
    user_metadata.update(
        {
            "dataset_status": "FROZEN_INCOMPLETE_OBSERVATION_EVALUATION_ONLY_V1",
            "cohort": "incomplete_observation",
            "split": "evaluation_only_not_train_validation_test",
            "parent_surface_id": parent.surface_id,
            "missingness_pattern": pattern,
            "sequence_index": int(sequence_index),
            "usable_slot_count": int(mask.sum()),
            "imputation": "NONE_MASKED_EXACT_ZERO",
            "evaluation_only": True,
            "train_validation_eligible": False,
        }
    )
    metadata["user_metadata"] = user_metadata
    return R2Surface(
        prices=tuple(float(value) for value in prices),
        mask=tuple(bool(value) for value in mask),
        maturities=parent.maturities,
        rates=parent.rates,
        carries=parent.carries,
        spot=parent.spot,
        surface_id=surface_id,
        source=parent.source,
        slot_keys=parent.slot_keys,
        metadata=metadata,
    )


def derive_incomplete_surfaces(
    clean_surfaces: list[R2Surface], config: dict[str, Any]
) -> list[R2Surface]:
    by_cohort = {
        cohort: [item for item in clean_surfaces if str(
            item.metadata["user_metadata"]["cohort"]
        ) == cohort]
        for cohort in COHORT_ORDER
    }
    parents: list[R2Surface] = []
    for position in range(60):
        cohort = COHORT_ORDER[position % len(COHORT_ORDER)]
        local_index = position // len(COHORT_ORDER)
        parents.append(by_cohort[cohort][local_index])
    derivatives: list[R2Surface] = []
    for index, parent in enumerate(parents):
        pattern = MISSING_PATTERNS[index % len(MISSING_PATTERNS)]
        derivative = make_incomplete_surface(
            parent,
            pattern=pattern,
            surface_id=f"{parent.surface_id}_INCOMPLETE_{index:06d}",
            sequence_index=index,
        )
        payload = surface_to_payload(derivative)
        validate_payload(payload)
        if int(sum(derivative.mask)) < 10 or any(
            price == 0.0 and valid
            for price, valid in zip(derivative.prices, derivative.mask)
        ):
            raise OODProtocolError("incomplete-surface mask invariant failure")
        derivatives.append(derivative)
    expected = int(config["frozen_cohorts"]["incomplete_observation"][
        "derived_surface_count"
    ])
    if len(derivatives) != expected:
        raise OODProtocolError("incomplete-surface count drift")
    return derivatives


def _environment() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "platform": platform.platform(),
    }


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return "UNAVAILABLE"


def _payload_records(surfaces: list[R2Surface]) -> list[dict[str, Any]]:
    return [surface_to_payload(surface) for surface in surfaces]


def run_generation(
    output_directory: str | Path,
    *,
    remote_checkpoint_confirmed: bool,
) -> tuple[Path, dict[str, Any]]:
    """Generate fixed cohorts only after the remote-checkpoint gate passes."""
    config = load_protocol_config()
    require_remote_checkpoint(remote_checkpoint_confirmed)
    output = Path(output_directory)
    if output.exists():
        raise OODProtocolError(f"refusing to overwrite OOD output: {output}")

    pools, selected = build_parameter_pools(config)
    development = build_development_panel(selected)
    output.mkdir(parents=True)
    # Retain panels before pricing so an unexpected exception cannot discard
    # the fixed candidate provenance.
    _write_csv(development, output / "development_sanity_panel.csv")
    for cohort in COHORT_ORDER:
        filename = f"{cohort}_candidates.csv"
        _write_csv(pools[cohort], output / filename)
    _write_csv(selected, output / "selected_parameters.csv")
    _write_bytes(output / "pricing_failures.jsonl", b"")
    clean_surfaces, sanity_rows, failures = generate_clean_surfaces(selected, config)
    failure_bytes = _jsonl_bytes(failures)
    _write_bytes(output / "pricing_failures.jsonl", failure_bytes)
    if failures:
        _write_json(
            output / "failed_generation_manifest.json",
            {
                "status": "FAILED_CLOSED_PRICING_FAILURES_RETAINED",
                "failure_count": len(failures),
                "pricing_failures_jsonl_sha256": sha256_bytes(failure_bytes),
            },
        )
        raise OODProtocolError(
            f"clean pricing produced {len(failures)} retained failures; "
            "no replacement or manifest approval is permitted"
        )
    incomplete_surfaces = derive_incomplete_surfaces(clean_surfaces, config)
    clean_payloads = _payload_records(clean_surfaces)
    incomplete_payloads = _payload_records(incomplete_surfaces)
    all_payloads = clean_payloads + incomplete_payloads

    artifact_hashes: dict[str, str] = {}
    artifact_hashes["development_sanity_panel_csv_sha256"] = sha256_file(
        output / "development_sanity_panel.csv"
    )
    for cohort in COHORT_ORDER:
        filename = f"{cohort}_candidates.csv"
        artifact_hashes[f"{filename.replace('.', '_')}_sha256"] = sha256_file(
            output / filename
        )
    artifact_hashes["selected_parameters_csv_sha256"] = sha256_file(
        output / "selected_parameters.csv"
    )
    clean_bytes = _jsonl_bytes(clean_payloads)
    incomplete_bytes = _jsonl_bytes(incomplete_payloads)
    all_bytes = _jsonl_bytes(all_payloads)
    _write_bytes(output / "clean_surfaces.jsonl", clean_bytes)
    _write_bytes(output / "incomplete_surfaces.jsonl", incomplete_bytes)
    _write_bytes(output / "all_research_surfaces.jsonl", all_bytes)
    artifact_hashes["clean_surfaces_jsonl_sha256"] = sha256_bytes(clean_bytes)
    artifact_hashes["incomplete_surfaces_jsonl_sha256"] = sha256_bytes(incomplete_bytes)
    artifact_hashes["all_research_surfaces_jsonl_sha256"] = sha256_bytes(all_bytes)
    sanity_bytes = _jsonl_bytes(sanity_rows)
    _write_bytes(output / "numerical_sanity.jsonl", sanity_bytes)
    artifact_hashes["numerical_sanity_jsonl_sha256"] = sha256_bytes(sanity_bytes)
    artifact_hashes["pricing_failures_jsonl_sha256"] = sha256_bytes(failure_bytes)

    cohort_counts = {
        cohort: int((selected["cohort"] == cohort).sum())
        for cohort in COHORT_ORDER
    }
    pattern_counts = {
        pattern: sum(
            item.metadata["user_metadata"]["missingness_pattern"] == pattern
            for item in incomplete_surfaces
        )
        for pattern in MISSING_PATTERNS
    }
    integrity = {
        "schema_version": "1.0",
        "representation_validated": True,
        "clean_surface_count": len(clean_payloads),
        "incomplete_surface_count": len(incomplete_payloads),
        "all_research_surface_count": len(all_payloads),
        "unique_surface_ids": len(
            {payload["surface_id"] for payload in all_payloads}
        ) == len(all_payloads),
        "unique_selected_parameter_vectors": not selected[
            "parameter_vector_hash"
        ].duplicated(),
        "no_frozen_r2_parameter_overlap": not bool(
            set(selected["parameter_vector_hash"]) & _load_frozen_parameter_hashes()
        ),
        "all_numerical_sanity_passed": all(row["passed"] for row in sanity_rows),
        "pricing_failure_count": len(failures),
        "missing_pattern_counts": pattern_counts,
    }
    integrity_bytes = deterministic_json_bytes(integrity)
    _write_bytes(output / "integrity_report.json", integrity_bytes)
    artifact_hashes["integrity_report_json_sha256"] = sha256_bytes(integrity_bytes)
    deterministic_content_sha256 = artifact_hashes[
        "all_research_surfaces_jsonl_sha256"
    ]
    manifest = {
        "schema_version": "1.0",
        "protocol_name": config["protocol_name"],
        "protocol_status": config["status"],
        "protocol_config_sha256": sha256_file(CONFIG_PATH),
        "reviewed_sampling_config_sha256": sha256_file(REVIEWED_CONFIG_PATH),
        "production_pricer_source_sha256": sha256_file(
            ROOT / "src" / "double_heston.py"
        ),
        "generator_source_sha256": sha256_file(Path(__file__)),
        "r2_dataset_sha256": sha256_file(DATASET_PATH),
        "representation": {
            "name": REPRESENTATION_NAME,
            "version": REPRESENTATION_VERSION,
            "slot_keys": [
                [key.expiry_rank, key.target_log_moneyness, key.option_type]
                for key in CANONICAL_SLOT_KEYS
            ],
        },
        "counts": {
            **cohort_counts,
            "incomplete_observation": len(incomplete_payloads),
            "serialized_research_total": len(all_payloads),
            "clean_pricing_calls": len(clean_payloads),
            "pricing_failure_count": len(failures),
            "development_sanity_rows": len(development),
        },
        "evaluation_only": True,
        "training_or_fine_tuning_started": False,
        "expensive_method_evaluation_performed": False,
        "real_market_inputs_used": False,
        "artifact_hashes": artifact_hashes,
        "deterministic_content_sha256": deterministic_content_sha256,
        "numerical_sanity": {
            "surface_count": len(sanity_rows),
            "failed_surface_count": sum(not row["passed"] for row in sanity_rows),
            "all_passed": all(row["passed"] for row in sanity_rows),
        },
        "environment": _environment(),
        "command": (
            "python -m src.ood_boundary_protocol generate --output "
            f"{output} --remote-checkpoint-confirmed"
        ),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "timestamp_is_rng_input": False,
        "git_commit_sha": _git_head(),
        "remote_checkpoint_confirmed": True,
        "replay_status": "PENDING",
    }
    _write_json(output / "manifest.json", manifest)
    validate_generated_output(output)
    return output, manifest


def validate_generated_output(output_directory: str | Path) -> dict[str, Any]:
    """Validate artifacts, hashes, identities, masks, and exact counts."""
    output = Path(output_directory)
    missing = [name for name in REQUIRED_ARTIFACTS if not (output / name).is_file()]
    if missing:
        raise OODProtocolError(f"generated output is missing artifacts: {missing}")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    artifact_files = {
        "development_sanity_panel_csv_sha256": "development_sanity_panel.csv",
        "boundary_candidates_csv_sha256": "boundary_candidates.csv",
        "distribution_shift_candidates_csv_sha256": "distribution_shift_candidates.csv",
        "maturity_conditioning_shift_candidates_csv_sha256": (
            "maturity_conditioning_shift_candidates.csv"
        ),
        "selected_parameters_csv_sha256": "selected_parameters.csv",
        "clean_surfaces_jsonl_sha256": "clean_surfaces.jsonl",
        "incomplete_surfaces_jsonl_sha256": "incomplete_surfaces.jsonl",
        "all_research_surfaces_jsonl_sha256": "all_research_surfaces.jsonl",
        "numerical_sanity_jsonl_sha256": "numerical_sanity.jsonl",
        "pricing_failures_jsonl_sha256": "pricing_failures.jsonl",
        "integrity_report_json_sha256": "integrity_report.json",
    }
    for hash_field, filename in artifact_files.items():
        path = output / filename
        if sha256_file(path) != manifest["artifact_hashes"].get(hash_field):
            raise OODProtocolError(f"artifact hash mismatch: {filename}")

    def read_payloads(filename: str) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        with (output / filename).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    payload = json.loads(line)
                    validate_payload(payload)
                    payloads.append(payload)
        return payloads

    clean = read_payloads("clean_surfaces.jsonl")
    incomplete = read_payloads("incomplete_surfaces.jsonl")
    all_payloads = read_payloads("all_research_surfaces.jsonl")
    if len(clean) != 360 or len(incomplete) != 60 or len(all_payloads) != 420:
        raise OODProtocolError("serialized research-record counts are wrong")
    if all_payloads != clean + incomplete:
        raise OODProtocolError("combined research JSONL ordering/content mismatch")
    ids = [payload["surface_id"] for payload in all_payloads]
    clean_ids = {payload["surface_id"] for payload in clean}
    if len(ids) != len(set(ids)):
        raise OODProtocolError("duplicate research surface ID")
    for payload in clean:
        if not all(payload["mask"]) or any(price <= 0 for price in payload["prices"]):
            raise OODProtocolError("clean surface contains an invalid complete mask")
        metadata = payload["metadata"]["user_metadata"]
        if metadata["evaluation_only"] is not True or metadata[
            "train_validation_eligible"
        ] is not False:
            raise OODProtocolError("clean research record is not evaluation-only")
    for payload in incomplete:
        mask = payload["mask"]
        if len(mask) != 20 or sum(mask) < 10 or any(
            price != 0.0 for price, valid in zip(payload["prices"], mask) if not valid
        ):
            raise OODProtocolError("invalid incomplete-surface mask semantics")
        metadata = payload["metadata"]["user_metadata"]
        if metadata["parent_surface_id"] not in clean_ids:
            raise OODProtocolError("incomplete record has no retained clean parent")
    integrity = json.loads((output / "integrity_report.json").read_text())
    if not all(integrity[key] for key in (
        "representation_validated",
        "unique_surface_ids",
        "unique_selected_parameter_vectors",
        "no_frozen_r2_parameter_overlap",
        "all_numerical_sanity_passed",
    )):
        raise OODProtocolError("integrity-report gate failed")
    return {"manifest": manifest, "clean": len(clean), "incomplete": len(incomplete)}


def verify_replay(
    primary_manifest: Mapping[str, Any], replay_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare every scientific artifact hash; exclude provenance timestamps."""
    ignored = {
        "manifest_json_sha256",
    }
    fields = [
        key
        for key in primary_manifest["artifact_hashes"]
        if key not in ignored
    ]
    comparisons = {
        field: primary_manifest["artifact_hashes"][field]
        == replay_manifest["artifact_hashes"].get(field)
        for field in fields
    }
    identical = all(comparisons.values()) and primary_manifest["counts"] == (
        replay_manifest["counts"]
    )
    return {
        "identical": identical,
        "hash_comparisons": comparisons,
        "counts_identical": primary_manifest["counts"] == replay_manifest["counts"],
        "timestamps_excluded": True,
    }


def run_replay(
    primary_directory: str | Path,
    replay_directory: str | Path,
    *,
    remote_checkpoint_confirmed: bool,
) -> tuple[Path, Path, dict[str, Any]]:
    """Independently regenerate and byte-compare the frozen research artifacts."""
    require_remote_checkpoint(remote_checkpoint_confirmed)
    primary = Path(primary_directory)
    validate_generated_output(primary)
    replay_path = Path(replay_directory)
    if replay_path.exists():
        raise OODProtocolError(f"refusing to overwrite replay output: {replay_path}")
    _, replay_manifest = run_generation(
        replay_path, remote_checkpoint_confirmed=True
    )
    primary_manifest = json.loads(
        (primary / "manifest.json").read_text(encoding="utf-8")
    )
    report = verify_replay(primary_manifest, replay_manifest)
    report["primary_output"] = str(primary)
    report["replay_output"] = str(replay_path)
    report["verified_at_utc"] = datetime.now(UTC).isoformat()
    _write_json(replay_path / "replay_report.json", report)
    _write_json(primary / "replay_report.json", report)
    primary_manifest["replay_status"] = (
        "VERIFIED_IDENTICAL" if report["identical"] else "FAILED"
    )
    primary_manifest["replay_report"] = report
    replay_manifest["replay_status"] = "INTERNAL_REPLAY_PENDING_COMPARISON"
    replay_manifest["replay_report"] = report
    _write_json(primary / "manifest.json", primary_manifest)
    _write_json(replay_path / "manifest.json", replay_manifest)
    validate_generated_output(primary)
    validate_generated_output(replay_path)
    if not report["identical"]:
        raise OODProtocolError("deterministic OOD replay mismatch")
    return primary, replay_path, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate-contract")
    generate = subparsers.add_parser("generate")
    generate.add_argument("--output", type=Path)
    generate.add_argument("--remote-checkpoint-confirmed", action="store_true")
    replay = subparsers.add_parser("replay")
    replay.add_argument("--output", type=Path)
    replay.add_argument("--replay-output", type=Path)
    replay.add_argument("--remote-checkpoint-confirmed", action="store_true")
    arguments = parser.parse_args(argv)

    if arguments.command == "validate-contract":
        config = load_protocol_config()
        print(
            json.dumps(
                {
                    "status": config["status"],
                    "cohorts": config["frozen_cohorts"],
                },
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "generate":
        output = arguments.output or Path(
            "evidence/ood_boundary_protocol_v1"
        )
        _, manifest = run_generation(
            output,
            remote_checkpoint_confirmed=arguments.remote_checkpoint_confirmed,
        )
        print(
            json.dumps(
                {"status": manifest["protocol_status"], "counts": manifest["counts"]},
                sort_keys=True,
            )
        )
        return 0
    output = arguments.output or Path("evidence/ood_boundary_protocol_v1")
    replay_output = arguments.replay_output or Path(
        "evidence/ood_boundary_protocol_v1_replay"
    )
    _, _, report = run_replay(
        output,
        replay_output,
        remote_checkpoint_confirmed=arguments.remote_checkpoint_confirmed,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
