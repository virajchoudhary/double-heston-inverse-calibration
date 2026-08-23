"""Deterministic final R2 synthetic-generation contract implementation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml
import scipy

from .audit_reviewed_sampling import load_reviewed_config, sample_distribution
from .constants import PARAMETER_NAMES
from .r2_representation import (
    CANONICAL_SLOT_KEYS,
    REPRESENTATION_NAME,
    REPRESENTATION_VERSION,
    R2Conditioning,
    R2Surface,
    surface_to_payload,
    validate_payload,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "r2_synthetic_generation_FINAL.yaml"
REVIEWED_CONFIG_PATH = ROOT / "configs" / "parameter_sampling_REVIEWED.yaml"
PRICER_PATH = ROOT / "src" / "double_heston.py"
SAMPLER_SOURCE_PATH = ROOT / "src" / "audit_reviewed_sampling.py"
R2_SYNTHETIC_INTERFACE_PATH = ROOT / "src" / "r2_representation" / "synthetic.py"
CONTRACT_FREEZE_MARKER = (
    ROOT / "evidence" / "R2_CONTRACT_FROZEN_BEFORE_PILOT.txt"
)
PILOT_OUTPUT = ROOT / "evidence" / "final_r2_synthetic_pilot_20260822"
READINESS_OUTPUT = (
    ROOT / "evidence" / "final_r2_candidate_pool_readiness_20260822"
)
DISTRIBUTIONS = ("interior_train", "wide_valid_train")
SPLIT_ORDER = ("train", "validation", "test")


class GenerationContractError(RuntimeError):
    """Raised when the frozen generation contract cannot be honored."""


class CandidatePoolInsufficientError(GenerationContractError):
    """Raised when a fixed pool cannot satisfy its frozen quota."""


def load_generation_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Load and structurally validate the frozen generation config."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if config.get("contract_name") != "R2_SYNTHETIC_GENERATION_FINAL":
        raise GenerationContractError("unexpected generation contract name")
    if config.get("status") != "FROZEN_BEFORE_PILOT":
        raise GenerationContractError("generation contract is not frozen")
    representation = config["representation"]
    if representation["name"] != REPRESENTATION_NAME:
        raise GenerationContractError("active generation representation is not R2")
    if representation["version"] != REPRESENTATION_VERSION:
        raise GenerationContractError("R2 representation version mismatch")
    if representation["nominal_slot_count"] != len(CANONICAL_SLOT_KEYS):
        raise GenerationContractError("nominal slot count is not canonical R2")
    if representation["slot_order"] != "option_type_major_then_expiry_rank_then_log_moneyness":
        raise GenerationContractError("slot ordering contract drift")
    if list(config["parameter_contract"]["order"]) != PARAMETER_NAMES:
        raise GenerationContractError("canonical parameter order drift")
    expected_seeds = {
        "pilot_interior_train": 20260822,
        "pilot_wide_valid_train": 20260823,
        "final_interior_train": 20260807,
        "final_wide_valid_train": 20260808,
    }
    if config["sampling"]["seeds"] != expected_seeds:
        raise GenerationContractError("sampling seed contract drift")
    expected_pools = {
        "pilot": {
            "interior_train": {"candidate_count": 400, "required_quota": 200},
            "wide_valid_train": {"candidate_count": 200, "required_quota": 40},
        },
        "final": {
            "interior_train": {"candidate_count": 15000, "required_quota": 8334},
            "wide_valid_train": {"candidate_count": 5000, "required_quota": 1666},
        },
    }
    if config["sampling"]["pools"] != expected_pools:
        raise GenerationContractError("candidate pool contract drift")
    expected_quotas = {
        "final_clean_core": {
            "total_surfaces": 10000,
            "distributions": {"interior_train": 8334, "wide_valid_train": 1666},
            "splits": {
                "train": {"total": 7500, "interior_train": 6250, "wide_valid_train": 1250},
                "validation": {"total": 1250, "interior_train": 1042, "wide_valid_train": 208},
                "test": {"total": 1250, "interior_train": 1042, "wide_valid_train": 208},
            },
        },
        "development_pilot_not_final_research_dataset": {
            "total_surfaces": 240,
            "distributions": {"interior_train": 200, "wide_valid_train": 40},
            "splits": {
                "train": {"total": 180, "interior_train": 150, "wide_valid_train": 30},
                "validation": {"total": 30, "interior_train": 25, "wide_valid_train": 5},
                "test": {"total": 30, "interior_train": 25, "wide_valid_train": 5},
            },
        },
    }
    if config["quotas"] != expected_quotas:
        raise GenerationContractError("generation quota contract drift")
    if config["conditioning"]["real_market_inputs_used"] is not False:
        raise GenerationContractError(
            "synthetic conditioning must exclude real market inputs"
        )
    lattice = config["conditioning"]["lattice"]
    expected_conditioning_support = {
        "rank1_dte_days": [7, 14, 21, 30, 45, 60, 75, 90],
        "rank2_gap_dte_days": [7, 14, 21, 30, 45, 60, 90],
        "rates": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        "carry_offsets": [-0.02, -0.01, 0.00, 0.01, 0.02, 0.03],
    }
    for field, expected_value in expected_conditioning_support.items():
        if lattice[field] != expected_value:
            raise GenerationContractError(f"conditioning {field} contract drift")
    if config["pricing"]["production_source"] != "src/double_heston.py":
        raise GenerationContractError("production pricing source drift")
    if config["pricing"]["entrypoint"] != "price_double_heston_surface":
        raise GenerationContractError("production pricing entrypoint drift")
    if int(config["pricing"]["node_count"]) != 64:
        raise GenerationContractError("production node-count contract drift")
    if config["noise"]["clean_core_level"] != 0.0 or config["noise"]["generation_in_this_milestone"] is not False:
        raise GenerationContractError("clean-core noise contract drift")
    expected_combinations = (
        len(lattice["rank1_dte_days"])
        * len(lattice["rank2_gap_dte_days"])
        * len(lattice["rates"])
        * len(lattice["carry_offsets"])
    )
    if int(lattice["combination_count"]) != expected_combinations:
        raise GenerationContractError("conditioning combination count mismatch")
    for stride in config["conditioning"]["strides"].values():
        if math.gcd(int(stride), expected_combinations) != 1:
            raise GenerationContractError(
                "conditioning stride must be coprime to lattice size"
            )
    gates = config["execution_gates"]
    if gates["final_10k_generation_command"] != "NOT_AUTHORIZED_IN_THIS_MILESTONE":
        raise GenerationContractError(
            "final 10k gate must remain closed in this milestone"
        )
    if gates["training_commands_in_this_milestone"] != "NONE":
        raise GenerationContractError("training gate must remain closed in this milestone")
    if gates["readiness_requires_verified_pilot"] is not True:
        raise GenerationContractError("readiness pilot-ordering gate is required")
    return config


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def deterministic_json_bytes(payload: Any) -> bytes:
    text = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    )
    return (text + "\n").encode("utf-8")


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
        default=_json_default,
    ) + "\n"
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def parameter_vector_hash(row: Mapping[str, Any]) -> str:
    vector = [float(row[name]) for name in PARAMETER_NAMES]
    return sha256_bytes(deterministic_json_bytes(vector))


def generate_candidate_pools(
    cohort: str,
    config: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    """Generate one entire fixed reviewed candidate pool per distribution."""
    config = config or load_generation_config()
    reviewed = load_reviewed_config(REVIEWED_CONFIG_PATH)
    pools: dict[str, pd.DataFrame] = {}
    for distribution in DISTRIBUTIONS:
        pool_spec = config["sampling"]["pools"][cohort][distribution]
        seed_key = f"{cohort}_{distribution}"
        seed = int(config["sampling"]["seeds"][seed_key])
        frame = sample_distribution(
            distribution,
            count=int(pool_spec["candidate_count"]),
            seed=seed,
            config=reviewed,
        ).copy()
        frame["candidate_key"] = [
            f"{distribution}_{int(candidate_id):06d}"
            for candidate_id in frame["candidate_id"]
        ]
        # The reviewed sampler's modulo split is a historical diagnostic and is
        # superseded here; retain it under an explicit non-authoritative name.
        frame.rename(
            columns={"split": "reviewed_sampler_diagnostic_split"},
            inplace=True,
        )
        pools[distribution] = frame
    return pools


def select_accepted_candidates(
    pools: Mapping[str, pd.DataFrame],
    cohort: str,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select exact quotas by candidate ID; never refill or reseed."""
    selected_parts: list[pd.DataFrame] = []
    retained_parts: list[pd.DataFrame] = []
    for distribution in DISTRIBUTIONS:
        frame = pools[distribution].sort_values(
            "candidate_id", kind="mergesort"
        ).copy()
        required = int(
            config["sampling"]["pools"][cohort][distribution]["required_quota"]
        )
        accepted_count = int(frame["accepted"].sum())
        if accepted_count < required:
            raise CandidatePoolInsufficientError(
                f"{cohort}/{distribution}: fixed pool has {accepted_count} "
                f"accepted candidates but requires {required}; refill/reseed "
                "is forbidden"
            )
        chosen = frame.loc[frame["accepted"]].iloc[:required].copy()
        quota = config["quotas"][
            "development_pilot_not_final_research_dataset"
            if cohort == "pilot"
            else "final_clean_core"
        ]["splits"]
        counts = [int(quota[split][distribution]) for split in SPLIT_ORDER]
        boundaries = np.cumsum(counts)
        starts = np.concatenate(([0], boundaries[:-1]))
        assignments: list[str] = []
        for split, start, stop in zip(SPLIT_ORDER, starts, boundaries, strict=True):
            assignments.extend([split] * int(stop - start))
        if len(assignments) != len(chosen):
            raise GenerationContractError(
                "frozen split quotas do not match the selected distribution quota"
            )
        chosen["split"] = assignments
        frame["selected_for_clean_core"] = False
        frame.loc[chosen.index, "selected_for_clean_core"] = True
        selected_parts.append(chosen)
        retained_parts.append(frame)
    selected = pd.concat(selected_parts, ignore_index=True)
    retained = pd.concat(retained_parts, ignore_index=True)
    vector_ids = selected[PARAMETER_NAMES].map(lambda value: repr(float(value)))
    vector_ids = vector_ids.apply("|".join, axis=1)
    if bool(vector_ids.duplicated().any()):
        raise GenerationContractError("duplicate parameter vector selected")
    expected_total = sum(
        int(config["sampling"]["pools"][cohort][distribution]["required_quota"])
        for distribution in DISTRIBUTIONS
    )
    if len(selected) != expected_total:
        raise GenerationContractError("selection count does not equal frozen quota")
    if set(selected["split"]) - set(SPLIT_ORDER):
        raise GenerationContractError("invalid split assignment")
    return selected, retained


def build_conditioning(
    generation_index: int,
    cohort: str,
    config: dict[str, Any],
) -> tuple[R2Conditioning, dict[str, Any]]:
    """Build deterministic synthetic-only two-rank conditioning."""
    conditioning_config = config["conditioning"]
    lattice = conditioning_config["lattice"]
    rank1_values = [int(value) for value in lattice["rank1_dte_days"]]
    gap_values = [int(value) for value in lattice["rank2_gap_dte_days"]]
    rate_values = [float(value) for value in lattice["rates"]]
    carry_offset_values = [float(value) for value in lattice["carry_offsets"]]
    combination_count = int(lattice["combination_count"])
    stride = int(conditioning_config["strides"][cohort])
    seed = int(conditioning_config["seeds"][cohort])
    lattice_index = (int(generation_index) * stride) % combination_count
    dte1 = rank1_values[lattice_index % len(rank1_values)]
    gap = gap_values[(lattice_index // len(rank1_values)) % len(gap_values)]
    rate_index = (
        lattice_index // (len(rank1_values) * len(gap_values))
    ) % len(rate_values)
    carry_offset_index = (
        lattice_index
        // (len(rank1_values) * len(gap_values) * len(rate_values))
    ) % len(carry_offset_values)
    rate = rate_values[rate_index]
    carry_offset = carry_offset_values[carry_offset_index]
    carry = rate + carry_offset
    dte2 = dte1 + gap
    conditioning = R2Conditioning(
        date_id=f"SYNTHETIC_R2_{cohort.upper()}_{generation_index:06d}",
        spot=float(conditioning_config["spot"]),
        expiry_dates=("SYNTHETIC_RANK_1", "SYNTHETIC_RANK_2"),
        dte=(dte1, dte2),
        rates=(rate, rate),
        carries=(carry, carry),
    )
    provenance = {
        "seed": seed,
        "stride": stride,
        "lattice_index": lattice_index,
        "generation_index": generation_index,
        "rank1_dte_days": dte1,
        "rank2_gap_dte_days": gap,
        "rank2_dte_days": dte2,
        "rate": rate,
        "carry_offset": carry_offset,
        "carry": carry,
        "classification": conditioning_config["classification"],
        "real_market_inputs_used": False,
    }
    return conditioning, provenance


def validate_conditioning_support(
    surfaces: list[R2Surface],
    conditioning_rows: list[dict[str, Any]],
    combination_count: int,
    conditioning_seed: int,
) -> None:
    """Fail closed on any rank, support, finiteness, or provenance violation."""
    if len(surfaces) != len(conditioning_rows):
        raise GenerationContractError("conditioning provenance count mismatch")
    allowed_dte1 = {7, 14, 21, 30, 45, 60, 75, 90}
    allowed_gaps = {7, 14, 21, 30, 45, 60, 90}
    allowed_rates = {0.01, 0.02, 0.03, 0.04, 0.05, 0.06}
    allowed_offsets = {-0.02, -0.01, 0.0, 0.01, 0.02, 0.03}
    for surface, row in zip(surfaces, conditioning_rows, strict=True):
        dte1 = int(surface.metadata["dte"][0])
        dte2 = int(surface.metadata["dte"][1])
        gap = dte2 - dte1
        rate = float(surface.rates[0])
        carry = float(surface.carries[0])
        rank2_rate = float(surface.rates[5])
        rank2_carry = float(surface.carries[5])
        offset = float(row["carry_offset"])
        seed = int(row["seed"])
        if seed != conditioning_seed:
            raise GenerationContractError("conditioning seed provenance mismatch")
        stride = int(row["stride"])
        generation_index = int(row["generation_index"])
        expected_lattice_index = (generation_index * stride) % combination_count
        if dte1 not in allowed_dte1 or gap not in allowed_gaps:
            raise GenerationContractError(f"conditioning outside DTE support: {row}")
        if rate not in allowed_rates or offset not in allowed_offsets:
            raise GenerationContractError(f"conditioning outside financial support: {row}")
        if not all(math.isfinite(value) for value in (rate, carry)):
            raise GenerationContractError("non-finite synthetic conditioning")
        if dte2 <= dte1:
            raise GenerationContractError("rank-2 maturity must exceed rank-1")
        if carry != rate + offset:
            raise GenerationContractError("synthetic carry identity mismatch")
        if rank2_rate != rate or rank2_carry != carry:
            raise GenerationContractError(
                "synthetic conditioning must use the configured value per rank"
            )
        if row["lattice_index"] != expected_lattice_index:
            raise GenerationContractError("conditioning lattice mapping mismatch")
        if surface.spot != 100.0 or not all(surface.mask):
            raise GenerationContractError("synthetic spot/mask contract violation")
        if row["rank1_dte_days"] != dte1 or row["rank2_dte_days"] != dte2:
            raise GenerationContractError("conditioning provenance mismatch")


def _surface_metadata(
    row: Mapping[str, Any],
    cohort: str,
    generation_index: int,
    conditioning_provenance: Mapping[str, Any],
    hashes: Mapping[str, str],
) -> dict[str, Any]:
    status = (
        "DEVELOPMENT_PILOT_NOT_FINAL_RESEARCH_DATASET"
        if cohort == "pilot"
        else "FINAL_PARAMETER_PANEL_ONLY_SURFACES_NOT_GENERATED_NOT_YET_TRAINING_DATA"
    )
    return {
        "dataset_status": status,
        "distribution": str(row["distribution"]),
        "split": str(row["split"]),
        "candidate_id": int(row["candidate_id"]),
        "candidate_key": str(row["candidate_key"]),
        "parameter_vector_hash": parameter_vector_hash(row),
        "parameter_sampler_seed": int(
            hashes["parameter_sampler_seeds"][str(row["distribution"])]
        ),
        "conditioning_seed": int(conditioning_provenance["seed"]),
        "conditioning_stride": int(conditioning_provenance["stride"]),
        "conditioning_lattice_index": int(conditioning_provenance["lattice_index"]),
        "generation_index": generation_index,
        "noise_level": 0.0,
        "generation_config_sha256": hashes["config"],
        "production_pricer_sha256": hashes["pricer"],
        "generator_source_sha256": hashes["generator_source"],
        "reviewed_sampler_source_sha256": hashes["reviewed_sampler_source"],
        "r2_synthetic_interface_sha256": hashes["r2_synthetic_interface"],
        "generator_version": hashes["generator_version"],
        "reviewed_sampling_policy": "UNCHANGED_REVIEWED_DESIGN",
        "real_market_inputs_used": False,
    }


def generate_selected_surfaces(
    selected: pd.DataFrame,
    cohort: str,
    config: dict[str, Any],
    hashes: Mapping[str, Any],
) -> tuple[list[R2Surface], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Price every selected surface; preserve and retain any failure."""
    from .r2_representation import build_synthetic_surface

    surfaces: list[R2Surface] = []
    conditioning_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    sanity_rows: list[dict[str, Any]] = []
    combination_count = int(config["conditioning"]["lattice"]["combination_count"])
    for generation_index, (_, row) in enumerate(selected.iterrows()):
        surface_id = f"R2_{cohort.upper()}_{row['candidate_key']}"
        conditioning_record: dict[str, Any] = {
            "surface_id": surface_id,
            "candidate_key": row["candidate_key"],
        }
        conditioning: R2Conditioning | None = None
        conditioning_provenance: dict[str, Any] | None = None
        try:
            conditioning, next_conditioning_provenance = build_conditioning(
                generation_index, cohort, config
            )
            conditioning_provenance = next_conditioning_provenance
            metadata = _surface_metadata(
                row,
                cohort,
                generation_index,
                conditioning_provenance,
                hashes,
            )
            surface = build_synthetic_surface(
                row[PARAMETER_NAMES].to_numpy(dtype=np.float64),
                conditioning,
                surface_id=surface_id,
                metadata=metadata,
                node_count=int(config["pricing"]["node_count"]),
            )
            surfaces.append(surface)
            conditioning_rows.append(
                {
                    **conditioning_record,
                    **conditioning_provenance,
                    "parameters_canonical_order": dict(
                        surface.metadata["parameters_canonical_order"]
                    ),
                }
            )
            sanity_rows.append(_numerical_sanity(surface))
        except Exception as error:
            failures.append(
                {
                    **conditioning_record,
                    "dataset_status": "RETAINED_GENERATION_FAILURE",
                    "candidate_id": int(row["candidate_id"]),
                    "distribution": str(row["distribution"]),
                    "split": str(row["split"]),
                    "parameters": {
                        name: float(row[name]) for name in PARAMETER_NAMES
                    },
                    "parameter_vector_hash": parameter_vector_hash(row),
                    **(
                        {"conditioning": conditioning.__dict__}
                        if conditioning is not None
                        else {}
                    ),
                    **(conditioning_provenance or {}),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
    validate_conditioning_support(
        surfaces,
        conditioning_rows,
        combination_count,
        int(config["conditioning"]["seeds"][cohort]),
    )
    return surfaces, conditioning_rows, failures, sanity_rows


def _numerical_sanity(surface: R2Surface) -> dict[str, Any]:
    prices = surface.prices_array()
    calls = prices[:10].reshape(2, 5)
    puts = prices[10:].reshape(2, 5)
    call_monotone = bool(np.all(np.diff(calls, axis=1) <= 1e-9))
    put_monotone = bool(np.all(np.diff(puts, axis=1) >= -1e-9))
    parity_errors: list[float] = []
    for rank in (1, 2):
        maturity = float(surface.maturities[(rank - 1) * 5])
        strike_ratio = np.exp(
            np.asarray([-0.10, -0.05, 0.0, 0.05, 0.10], dtype=np.float64)
        )
        expected = np.exp(-float(surface.carries[(rank - 1) * 5]) * maturity) - (
            strike_ratio
            * np.exp(-float(surface.rates[(rank - 1) * 5]) * maturity)
        )
        observed = calls[rank - 1] - puts[rank - 1]
        parity_errors.extend(np.abs(observed - expected).tolist())
    parity_ok = bool(max(parity_errors) <= 5e-13)
    finite_positive = bool(np.isfinite(prices).all() and np.all(prices > 0.0))
    passed = finite_positive and call_monotone and put_monotone and parity_ok
    return {
        "surface_id": surface.surface_id,
        "finite_positive": finite_positive,
        "call_strike_monotonicity": call_monotone,
        "put_strike_monotonicity": put_monotone,
        "put_call_parity_max_abs_error": max(parity_errors),
        "put_call_parity": parity_ok,
        "passed": passed,
        "normalized_price_min": float(prices.min()),
        "normalized_price_max": float(prices.max()),
    }


def _dataframe_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode("utf-8")


def _write_csv(frame: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _dataframe_bytes(frame)
    path.write_bytes(payload)
    return sha256_bytes(payload)


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(deterministic_json_bytes(record) for record in records)
    path.write_bytes(payload)
    return sha256_bytes(payload)


def _surface_payloads(surfaces: list[R2Surface]) -> list[dict[str, Any]]:
    payloads = [surface_to_payload(surface) for surface in surfaces]
    for payload in payloads:
        from .r2_representation import validate_payload

        validate_payload(payload)
        restored = json.loads(json.dumps(payload, allow_nan=False))
        if restored != payload:
            raise GenerationContractError(f"serialization round-trip failed: {payload['surface_id']}")
    return payloads


def _git_head(root: Path = ROOT) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "GIT_SHA_UNAVAILABLE"


def _milestone_git_head(root: Path = ROOT) -> str:
    """Resolve the canonical milestone branch even when this checkout is detached."""
    try:
        result = subprocess.run(
            [
                "git",
                "rev-parse",
                "refs/heads/feature/final-r2-synthetic-contract-pilot",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return _git_head(root)


def environment_metadata() -> dict[str, Any]:
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
    }


def rejection_summary(retained: pd.DataFrame) -> dict[str, Any]:
    rejected = retained.loc[~retained["accepted"]]
    reasons = (
        rejected.assign(reason=rejected["rejection_reasons"].str.split(";"))
        .explode("reason")
        .groupby("reason")
        .size()
        .sort_index()
        .to_dict()
    )
    return {
        "candidate_count": int(len(retained)),
        "accepted_count": int(retained["accepted"].sum()),
        "rejected_count": int(len(rejected)),
        "rejection_reason_counts": {str(key): int(value) for key, value in reasons.items()},
    }


def split_summary(selected: pd.DataFrame, cohort: str) -> dict[str, Any]:
    quota = config_quotas(cohort)["splits"]
    actual: dict[str, Any] = {}
    for distribution in DISTRIBUTIONS:
        actual[distribution] = {
            split: int(
                ((selected["distribution"] == distribution) & (selected["split"] == split)).sum()
            )
            for split in SPLIT_ORDER
        }
    totals = {
        split: sum(actual[distribution][split] for distribution in DISTRIBUTIONS)
        for split in SPLIT_ORDER
    }
    matches_quota = all(
        actual[distribution][split] == int(quota[split][distribution])
        for distribution in DISTRIBUTIONS
        for split in SPLIT_ORDER
    )
    return {"by_distribution": actual, "totals": totals, "matches_frozen_quota": matches_quota}


def config_quotas(cohort: str) -> dict[str, Any]:
    config = load_generation_config()
    quota_key = (
        "development_pilot_not_final_research_dataset"
        if cohort == "pilot"
        else "final_clean_core"
    )
    return config["quotas"][quota_key]


def _retain_insufficient_pool_evidence(
    output: Path,
    pools: Mapping[str, pd.DataFrame],
    cohort: str,
) -> None:
    """Retain the complete fixed pool when quota selection fails closed."""
    retained = pd.concat(pools.values(), ignore_index=True)
    rejections = retained.loc[~retained["accepted"]].to_dict(orient="records")
    sufficiency = {
        distribution: {
            "candidate_count": int(len(pools[distribution])),
            "accepted_count": int(pools[distribution]["accepted"].sum()),
            "required_quota": int(
                load_generation_config()["sampling"]["pools"][cohort][
                    distribution
                ]["required_quota"]
            ),
        }
        for distribution in DISTRIBUTIONS
    }
    candidates_sha256 = _write_csv(retained, output / "candidates.csv")
    rejections_sha256 = _write_jsonl(rejections, output / "rejections.jsonl")
    write_json(
        output / "manifest.json",
        {
            "status": "FAILED_INSUFFICIENT_FIXED_POOL_RETAINED_CANDIDATES",
            "cohort": cohort,
            "sufficiency": sufficiency,
            "candidates_csv_sha256": candidates_sha256,
            "rejections_jsonl_sha256": rejections_sha256,
            "refill_or_reseed_used": False,
        },
    )


def _build_generation_cohort(
    cohort: str,
    output_directory: str | Path,
    command: str,
) -> tuple[Path, dict[str, Any]]:
    """Generate one cohort after the caller has enforced all ordering gates."""
    if cohort != "pilot":
        raise GenerationContractError(
            "this milestone permits surface generation only for the contracted "
            "development pilot; final 10k pricing is separately gated"
        )
    config = load_generation_config()
    output = Path(output_directory)
    if output.exists():
        raise GenerationContractError(f"refusing to overwrite controlled output: {output}")
    output.mkdir(parents=True)
    pools = generate_candidate_pools(cohort, config)
    seeds = {
        distribution: int(config["sampling"]["seeds"][f"{cohort}_{distribution}"])
        for distribution in DISTRIBUTIONS
    }
    hashes: dict[str, Any] = {
        "config": sha256_file(CONFIG_PATH),
        "pricer": sha256_file(PRICER_PATH),
        "generator_source": sha256_file(Path(__file__)),
        "reviewed_sampler_source": sha256_file(SAMPLER_SOURCE_PATH),
        "r2_synthetic_interface": sha256_file(R2_SYNTHETIC_INTERFACE_PATH),
        "generator_version": config["generator_version"],
        "parameter_sampler_seeds": seeds,
    }
    try:
        selected, retained = select_accepted_candidates(pools, cohort, config)
    except CandidatePoolInsufficientError:
        _retain_insufficient_pool_evidence(output, pools, cohort)
        raise
    surfaces, conditioning_rows, failures, sanity_rows = generate_selected_surfaces(
        selected, cohort, config, hashes
    )
    if failures:
        failures_hash = _write_jsonl(failures, output / "failures.jsonl")
        manifest = {
            "status": "FAILED_CLOSED_RETAINED_FAILURES",
            "failure_count": len(failures),
            "failures_sha256": failures_hash,
        }
        write_json(output / "manifest.json", manifest)
        raise GenerationContractError(
            f"generation failed closed with {len(failures)} retained pricing/numerical failures"
        )

    payloads = _surface_payloads(surfaces)
    surfaces_payload = b"".join(deterministic_json_bytes(item) for item in payloads)
    (output / "surfaces.jsonl").write_bytes(surfaces_payload)
    selected_parameters = selected[
        ["candidate_key", "distribution", "split", *PARAMETER_NAMES]
    ].copy()
    selected_parameters["parameter_vector_hash"] = [
        parameter_vector_hash(row) for _, row in selected.iterrows()
    ]
    rejections = retained.loc[~retained["accepted"]].to_dict(orient="records")
    split_map = {
        surface.surface_id: str(metadata_split)
        for surface, metadata_split in zip(
            surfaces,
            selected["split"],
            strict=True,
        )
    }
    acceptance_by_distribution = {
        distribution: {
            "pool_count": int(len(pools[distribution])),
            "accepted_count": int(pools[distribution]["accepted"].sum()),
            "required_quota": int(
                config["sampling"]["pools"][cohort][distribution]["required_quota"]
            ),
            "selected_count": int((selected["distribution"] == distribution).sum()),
        }
        for distribution in DISTRIBUTIONS
    }
    file_hashes = {
        "surfaces_jsonl_sha256": sha256_bytes(surfaces_payload),
        "selected_parameters_csv_sha256": _write_csv(
            selected_parameters, output / "selected_parameters.csv"
        ),
        "candidates_csv_sha256": _write_csv(retained, output / "candidates.csv"),
        "rejections_jsonl_sha256": _write_jsonl(rejections, output / "rejections.jsonl"),
        "conditioning_jsonl_sha256": _write_jsonl(
            conditioning_rows, output / "conditioning.jsonl"
        ),
        "numerical_sanity_jsonl_sha256": _write_jsonl(
            sanity_rows, output / "numerical_sanity.jsonl"
        ),
        "split_assignment_sha256": sha256_bytes(
            deterministic_json_bytes(split_map)
        ),
    }
    integrity = {
        "representation_validated": True,
        "all_masks_true": all(all(surface.mask) for surface in surfaces),
        "all_slot_counts_20": all(surface.slot_count == 20 for surface in surfaces),
        "unique_surface_ids": len({surface.surface_id for surface in surfaces}) == len(surfaces),
        "unique_parameter_vectors": len(
            {parameter_vector_hash(row) for _, row in selected.iterrows()}
        ) == len(selected),
        "serialization_round_trip": True,
        "manifest_complete": True,
    }
    file_hashes["integrity_report_json_sha256"] = sha256_bytes(
        deterministic_json_bytes(integrity)
    )
    deterministic_content_sha256 = file_hashes["surfaces_jsonl_sha256"]
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "contract_name": config["contract_name"],
        "generator_version": config["generator_version"],
        "cohort": cohort,
        "dataset_status": (
            "DEVELOPMENT_PILOT_NOT_FINAL_RESEARCH_DATASET"
            if cohort == "pilot"
            else "FINAL_PARAMETER_PANEL_ONLY_SURFACES_NOT_GENERATED_NOT_YET_TRAINING_DATA"
        ),
        "representation_name": REPRESENTATION_NAME,
        "representation_version": REPRESENTATION_VERSION,
        "slot_keys": [
            [key.expiry_rank, key.target_log_moneyness, key.option_type]
            for key in CANONICAL_SLOT_KEYS
        ],
        "surface_count": len(surfaces),
        "quotas": config_quotas(cohort),
        "sampling": {
            "policy": config["sampling"]["fixed_pool_policy"],
            "seeds": seeds,
            "acceptance_by_distribution": acceptance_by_distribution,
            "insufficient_pool_behavior": config["sampling"]["insufficient_pool_behavior"],
        },
        "selection_order": config["sampling"]["selection_order"],
        "split_algorithm": config["split_assignment"]["algorithm"],
        "split_summary": split_summary(selected, cohort),
        "conditioning_policy": {
            "classification": config["conditioning"]["classification"],
            "seed": int(config["conditioning"]["seeds"][cohort]),
            "stride": int(config["conditioning"]["strides"][cohort]),
            "lattice": config["conditioning"]["lattice"],
            "real_market_inputs_used": False,
        },
        "pricing": {
            "source": config["pricing"]["production_source"],
            "entrypoint": config["pricing"]["entrypoint"],
            "node_count": int(config["pricing"]["node_count"]),
            "noise_level": 0.0,
            "dummy_mapping_used": False,
        },
        "exclusions": config["exclusions"],
        "failure_retention": config["failure_retention"],
        "provenance_hashes": {
            "generation_config_sha256": hashes["config"],
            "production_pricer_sha256": hashes["pricer"],
            "generator_source_sha256": hashes["generator_source"],
            "reviewed_sampler_source_sha256": hashes["reviewed_sampler_source"],
            "r2_synthetic_interface_sha256": hashes["r2_synthetic_interface"],
            **file_hashes,
            "deterministic_content_sha256": deterministic_content_sha256,
        },
        "numerical_sanity": {
            "surface_count": len(sanity_rows),
            "failed_surface_count": sum(not row["passed"] for row in sanity_rows),
            "all_passed": all(row["passed"] for row in sanity_rows),
        },
        "environment": environment_metadata(),
        "command": command,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "timestamp_is_rng_input": False,
        "git_commit_sha": _milestone_git_head(),
        "final_10k_generated": cohort == "final" and False,
        "training_started": False,
        "replay_status": "PENDING",
    }
    write_json(output / "integrity_report.json", integrity)
    write_json(output / "manifest.json", manifest)
    return output, manifest


def verify_replay(primary_manifest: Mapping[str, Any], replay_manifest: Mapping[str, Any]) -> dict[str, Any]:
    hash_fields = [
        "deterministic_content_sha256",
        "surfaces_jsonl_sha256",
        "selected_parameters_csv_sha256",
        "candidates_csv_sha256",
        "rejections_jsonl_sha256",
        "conditioning_jsonl_sha256",
        "numerical_sanity_jsonl_sha256",
        "integrity_report_json_sha256",
        "split_assignment_sha256",
        "generation_config_sha256",
        "production_pricer_sha256",
        "generator_source_sha256",
        "reviewed_sampler_source_sha256",
        "r2_synthetic_interface_sha256",
    ]
    primary_hashes = primary_manifest["provenance_hashes"]
    replay_hashes = replay_manifest["provenance_hashes"]
    comparisons = {
        field: primary_hashes[field] == replay_hashes[field]
        for field in hash_fields
    }
    identical = all(comparisons.values()) and (
        primary_manifest["split_summary"] == replay_manifest["split_summary"]
    ) and (
        primary_manifest["sampling"] == replay_manifest["sampling"]
    )
    return {
        "identical": identical,
        "hash_comparisons": comparisons,
        "split_summary_identical": primary_manifest["split_summary"] == replay_manifest["split_summary"],
        "sampling_identical": primary_manifest["sampling"] == replay_manifest["sampling"],
        "timestamps_excluded": True,
    }


def run_generation_cohort(
    cohort: str,
    output_directory: str | Path,
    command: str,
) -> tuple[Path, dict[str, Any]]:
    """Generate a controlled cohort; pilot calls always enforce freeze order."""
    if cohort == "pilot":
        _require_contract_freeze()
    return _build_generation_cohort(cohort, output_directory, command)


def _require_contract_freeze() -> None:
    if not CONTRACT_FREEZE_MARKER.is_file():
        raise GenerationContractError(
            "CONTRACT_FROZEN_BEFORE_PILOT marker is missing; pilot execution is forbidden"
        )
    marker = CONTRACT_FREEZE_MARKER.read_text(encoding="utf-8")
    required_statement = "NO PILOT OUTPUT EXISTED WHEN THIS CONTRACT SHA WAS COMMITTED"
    if required_statement not in marker:
        raise GenerationContractError("contract-freeze marker statement is invalid")


def _verified_pilot_evidence() -> dict[str, Any]:
    manifest_path = PILOT_OUTPUT / "manifest.json"
    replay_manifest_path = Path(str(PILOT_OUTPUT) + "_replay") / "manifest.json"
    if not manifest_path.is_file() or not replay_manifest_path.is_file():
        raise GenerationContractError(
            "final readiness requires an authoritative verified pilot; no complete "
            "pilot/replay manifests were found"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    replay_manifest = json.loads(replay_manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("numerical_sanity", {}).get("all_passed") is True:
        raise GenerationContractError("authoritative pilot has a numerical-sanity failure")
    integrity_path = PILOT_OUTPUT / "integrity_report.json"
    if not integrity_path.is_file():
        raise GenerationContractError("authoritative pilot integrity report is missing")
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    required_integrity = {
        "representation_validated",
        "all_masks_true",
        "all_slot_counts_20",
        "unique_surface_ids",
        "unique_parameter_vectors",
        "serialization_round_trip",
    }
    if any(integrity.get(field) is not True for field in required_integrity):
        raise GenerationContractError("authoritative pilot failed an integrity gate")

    artifact_files = {
        "surfaces_jsonl_sha256": "surfaces.jsonl",
        "selected_parameters_csv_sha256": "selected_parameters.csv",
        "candidates_csv_sha256": "candidates.csv",
        "rejections_jsonl_sha256": "rejections.jsonl",
        "conditioning_jsonl_sha256": "conditioning.jsonl",
        "numerical_sanity_jsonl_sha256": "numerical_sanity.jsonl",
        "integrity_report_json_sha256": "integrity_report.json",
    }
    for root, item_manifest in (
        (PILOT_OUTPUT, manifest),
        (Path(str(PILOT_OUTPUT) + "_replay"), replay_manifest),
    ):
        for hash_field, filename in artifact_files.items():
            artifact_path = root / filename
            if not artifact_path.is_file():
                raise GenerationContractError(f"pilot artifact is missing: {filename}")
            if sha256_file(artifact_path) != item_manifest["provenance_hashes"][hash_field]:
                raise GenerationContractError(f"pilot artifact hash mismatch: {filename}")

    surfaces_path = PILOT_OUTPUT / "surfaces.jsonl"
    payloads = [
        json.loads(line)
        for line in surfaces_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(payloads) != 240:
        raise GenerationContractError("authoritative pilot does not contain exactly 240 payloads")
    seen_ids: set[str] = set()
    seen_vectors: set[tuple[float, ...]] = set()
    observed_split_counts: dict[tuple[str, str], int] = {}
    for payload in payloads:
        validate_payload(payload)
        if not all(payload["mask"]):
            raise GenerationContractError("authoritative pilot contains a non-complete surface")
        user_metadata = payload["metadata"].get("user_metadata", {})
        distribution = str(user_metadata.get("distribution"))
        split = str(user_metadata.get("split"))
        if distribution not in DISTRIBUTIONS or split not in SPLIT_ORDER:
            raise GenerationContractError("authoritative pilot has invalid cohort/split provenance")
        key = (distribution, split)
        observed_split_counts[key] = observed_split_counts.get(key, 0) + 1
        surface_id = str(payload["surface_id"])
        vector = tuple(
            float(payload["metadata"]["user_metadata"]["parameters_canonical_order"][name])
            for name in PARAMETER_NAMES
        )
        if surface_id in seen_ids or vector in seen_vectors:
            raise GenerationContractError("authoritative pilot contains duplicate identity")
        seen_ids.add(surface_id)
        seen_vectors.add(vector)
    expected_split_counts = {
        (distribution, split): int(quota[split][distribution])
        for distribution in DISTRIBUTIONS
        for split in SPLIT_ORDER
        for quota in [config_quotas("pilot")["splits"]]
    }
    if observed_split_counts != expected_split_counts:
        raise GenerationContractError("authoritative pilot split counts do not match frozen quotas")

    if (
        manifest.get("cohort") != "pilot"
        or int(manifest.get("surface_count", -1)) != 240
        or manifest.get("replay_status") != "VERIFIED_IDENTICAL"
        or manifest.get("replay_report", {}).get("identical") is not True
    ):
        raise GenerationContractError("authoritative pilot did not pass deterministic verification")
    replay_report = manifest["replay_report"]
    if not all(replay_report.get("hash_comparisons", {}).values()):
        raise GenerationContractError("authoritative pilot replay has a scientific hash mismatch")
    if (
        replay_manifest.get("cohort") != "pilot"
        or int(replay_manifest.get("surface_count", -1)) != 240
        or replay_manifest.get("replay_status") != "PENDING"
    ):
        raise GenerationContractError("pilot replay evidence is incomplete")
    return {"primary_manifest": manifest, "replay_manifest": replay_manifest}


def run_pilot(
    output_directory: str | Path = PILOT_OUTPUT,
    replay_directory: str | Path | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Run exactly one contracted pilot and an independent deterministic replay."""
    _require_contract_freeze()
    command = f"python -m src.r2_synthetic_generation pilot --output {output_directory}"
    primary_path, primary_manifest = run_generation_cohort(
        "pilot", output_directory, command
    )
    if replay_directory is None:
        replay_path = Path(str(output_directory).rstrip("/\\") + "_replay")
    else:
        replay_path = Path(replay_directory)
    replay_command = f"python -m src.r2_synthetic_generation pilot --output {replay_path} [INTERNAL_REPLAY_AFTER_PRIMARY_GATE]"
    replay_path_actual, replay_manifest = _build_generation_cohort(
        "pilot", replay_path, replay_command
    )
    replay_report = verify_replay(primary_manifest, replay_manifest)
    replay_report["primary_output"] = str(primary_path)
    replay_report["replay_output"] = str(replay_path_actual)
    replay_report["verified_at_utc"] = datetime.now(UTC).isoformat()
    write_json(replay_path_actual / "replay_report.json", replay_report)
    write_json(primary_path / "replay_report.json", replay_report)
    primary_manifest["replay_status"] = (
        "VERIFIED_IDENTICAL" if replay_report["identical"] else "FAILED"
    )
    primary_manifest["replay_report"] = replay_report
    write_json(primary_path / "manifest.json", primary_manifest)
    if not replay_report["identical"]:
        raise GenerationContractError("deterministic replay mismatch")
    return primary_path, replay_path_actual, replay_report


def run_final_readiness(
    output_directory: str | Path = READINESS_OUTPUT,
) -> tuple[Path, dict[str, Any]]:
    """Evaluate fixed final candidate pools without generating/pricing surfaces."""
    config = load_generation_config()
    _verified_pilot_evidence()
    pools = generate_candidate_pools("final", config)
    sufficiency = {
        distribution: {
            "pool_count": int(len(pools[distribution])),
            "accepted_count": int(pools[distribution]["accepted"].sum()),
            "required_quota": int(
                config["sampling"]["pools"]["final"][distribution]["required_quota"]
            ),
            "sufficient": bool(
                pools[distribution]["accepted"].sum()
                >= config["sampling"]["pools"]["final"][distribution]["required_quota"]
            ),
        }
        for distribution in DISTRIBUTIONS
    }
    all_sufficient = all(item["sufficient"] for item in sufficiency.values())

    def _write_failure_retention(target: Path, rejection_frame: pd.DataFrame) -> None:
        target.mkdir(parents=True)
        rejections = rejection_frame.loc[~rejection_frame["accepted"]].to_dict(
            orient="records"
        )
        write_json(
            target / "readiness_manifest.json",
            {
                "schema_version": "1.0",
                "status": "FAILED_INSUFFICIENT_FIXED_POOL_RETAINED_REJECTIONS",
                "sufficiency": sufficiency,
                "all_sufficient": False,
                "selected_panel_count": 0,
                "surfaces_generated": False,
                "pricing_performed": False,
                "final_10k_generated": False,
                "training_started": False,
                "command": (
                    "python -m src.r2_synthetic_generation readiness "
                    f"--output {target}"
                ),
                "all_candidates_csv_sha256": _write_csv(
                    rejection_frame, target / "all_final_candidates.csv"
                ),
                "rejections_jsonl_sha256": _write_jsonl(
                    rejections, target / "rejections.jsonl"
                ),
                "generated_at_utc": datetime.now(UTC).isoformat(),
            },
        )

    if not all_sufficient:
        combined = pd.concat(pools.values(), ignore_index=True)
        failure_output = Path(str(output) + "_failed_insufficient")
        if failure_output.exists():
            raise GenerationContractError(
                "refusing to overwrite failed readiness retention output: "
                f"{failure_output}"
            )
        _write_failure_retention(failure_output, combined)
        raise CandidatePoolInsufficientError(
            "final fixed candidate pool failed its frozen quota; candidates and "
            "rejections were retained"
        )

    output = Path(output_directory)
    if output.exists():
        raise GenerationContractError(f"refusing to overwrite readiness output: {output}")
    output.mkdir(parents=True)
    selected, retained = select_accepted_candidates(pools, "final", config)
    seeds = {
        distribution: int(config["sampling"]["seeds"][f"final_{distribution}"])
        for distribution in DISTRIBUTIONS
    }
    panels = selected[
        ["candidate_key", "distribution", "split", *PARAMETER_NAMES]
    ].copy()
    panels["parameter_vector_hash"] = [
        parameter_vector_hash(row) for _, row in selected.iterrows()
    ]
    rejections = retained.loc[~retained["accepted"]].to_dict(orient="records")
    manifest = {
        "schema_version": "1.0",
        "status": "FINAL_CANDIDATE_POOL_READINESS_VERIFIED_NO_PRICING",
        "panel_label": "FINAL_PARAMETER_PANEL_ONLY",
        "surface_label": "SURFACES_NOT_GENERATED",
        "training_label": "NOT_YET_TRAINING_DATA",
        "sufficiency": sufficiency,
        "all_sufficient": all_sufficient,
        "selected_panel_count": int(len(panels)),
        "split_summary": split_summary(selected, "final"),
        "rejection_summary": rejection_summary(retained),
        "seeds": seeds,
        "generation_config_sha256": sha256_file(CONFIG_PATH),
        "reviewed_sampling_config_sha256": sha256_file(REVIEWED_CONFIG_PATH),
        "reviewed_sampler_source_sha256": sha256_file(SAMPLER_SOURCE_PATH),
        "generator_source_sha256": sha256_file(Path(__file__)),
        "r2_synthetic_interface_sha256": sha256_file(R2_SYNTHETIC_INTERFACE_PATH),
        "selected_panel_csv_sha256": _write_csv(panels, output / "final_parameter_panel.csv"),
        "all_candidates_csv_sha256": _write_csv(retained, output / "all_final_candidates.csv"),
        "rejections_jsonl_sha256": _write_jsonl(rejections, output / "rejections.jsonl"),
        "surfaces_generated": False,
        "pricing_performed": False,
        "final_10k_generated": False,
        "training_started": False,
        "command": f"python -m src.r2_synthetic_generation readiness --output {output}",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "git_commit_sha": _milestone_git_head(),
        "environment": environment_metadata(),
    }
    if not manifest["all_sufficient"]:
        write_json(output / "readiness_manifest.json", manifest)
        raise CandidatePoolInsufficientError("final fixed candidate pool failed its frozen quota")
    write_json(output / "readiness_manifest.json", manifest)
    return output, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-contract")
    pilot = subparsers.add_parser("pilot")
    pilot.add_argument("--output", type=Path, default=PILOT_OUTPUT)
    pilot.add_argument("--replay-output", type=Path, default=None)
    readiness = subparsers.add_parser("readiness")
    readiness.add_argument("--output", type=Path, default=READINESS_OUTPUT)
    arguments = parser.parse_args(argv)
    if arguments.command == "validate-contract":
        config = load_generation_config()
        print(json.dumps({"status": config["status"], "representation": config["representation"]["name"]}, sort_keys=True))
        return 0
    if arguments.command == "readiness":
        _, manifest = run_final_readiness(arguments.output)
        print(json.dumps({"status": manifest["status"], "sufficiency": manifest["sufficiency"]}, sort_keys=True))
        return 0
    # The public CLI and public API both enforce the freeze marker. Only the
    # internal replay, after the primary gate has passed, calls the builder.
    _, _, report = run_pilot(arguments.output, arguments.replay_output)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
