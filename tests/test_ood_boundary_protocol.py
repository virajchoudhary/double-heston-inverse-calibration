"""Focused pre-execution checks for the frozen OOD/boundary protocol."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.constants import PARAMETER_NAMES
from src.constraints import validate_parameters
from src.ood_boundary_protocol import (
    BOUNDARY_REGIMES,
    CONFIG_PATH,
    FREEZE_MARKER_PATH,
    MISSING_PATTERNS,
    ROOT,
    SHIFT_REGIMES,
    build_development_panel,
    build_parameter_pools,
    load_protocol_config,
    make_incomplete_surface,
    require_remote_checkpoint,
)
from src.r2_representation.contract import (
    CANONICAL_SLOT_KEYS,
    NOMINAL_SLOT_COUNT,
)
from src.r2_representation.surface import R2Surface


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "data" / "final_r2_clean_10000" / "surfaces.jsonl"
PROTOCOL_DOC = REPO_ROOT / "docs" / "OOD_BOUNDARY_PROTOCOL.md"
AUDIT_DOC = REPO_ROOT / "docs" / "OOD_BOUNDARY_PREEXECUTION_AUDIT.md"


def _config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_protocol_config_is_frozen_and_additive() -> None:
    config = _config()
    assert config["status"] == "FROZEN_BEFORE_MODEL3_RESEARCH_RESULTS"
    assert config["base_commit_sha256"] == (
        "72ad8e1aa845ec4c6f0fc61fc526df75438639bb"
    )
    assert config["leakage_controls"]["model3_results_used_in_design"] is False
    assert config["leakage_controls"]["real_market_weight_updates_allowed"] is False
    assert config["leakage_controls"]["completed_primary_files_modifiable"] is False
    assert config["execution_gates"]["neural_training_or_fine_tuning"] == "FORBIDDEN"
    assert config["execution_gates"][
        "expensive_method_evaluations_this_milestone"
    ] == "FORBIDDEN"
    assert "pricing_failures.jsonl" in config["manifests"]["required_artifacts"]
    assert "boundary_challenge_candidates.csv" in config["manifests"][
        "required_artifacts"
    ]


def test_frozen_dataset_and_primary_baseline_hashes_are_exact() -> None:
    # Full validation also proves these identities; this direct check makes a
    # silent dataset/baseline substitution visible in the focused test name.
    config = _config()
    digest = hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest()
    assert digest == config["authority"]["r2_dataset"]["sha256"]
    primary = REPO_ROOT / config["authority"]["primary_comparison_config"]["path"]
    assert hashlib.sha256(primary.read_bytes()).hexdigest().lower() == config[
        "authority"
    ]["primary_comparison_config"]["sha256"].lower()


def test_load_protocol_config_enforces_full_contract() -> None:
    config = load_protocol_config()
    assert config["representation"]["nominal_slot_count"] == NOMINAL_SLOT_COUNT == 20
    assert config["parameter_contract"]["order"] == PARAMETER_NAMES
    assert sum(
        item["count"] for item in config["frozen_cohorts"][
            "boundary_challenge"
        ]["regimes"].values()
    ) == 120
    assert config["metrics"]["uncertainty"]["bootstrap_seed"] == 20260829


def test_remote_checkpoint_gate_fails_closed() -> None:
    assert FREEZE_MARKER_PATH.is_file()
    text = FREEZE_MARKER_PATH.read_text(encoding="utf-8")
    assert "NO OOD RESEARCH SURFACE" in text
    with pytest.raises(Exception, match="remote-verified checkpoint"):
        require_remote_checkpoint(False)
    require_remote_checkpoint(True)


def test_fixed_parameter_pools_select_exact_distinct_quotas() -> None:
    _, selected = build_parameter_pools()
    assert len(selected) == 360
    assert selected["parameter_vector_hash"].is_unique
    assert selected.groupby("cohort").size().to_dict() == {
        "boundary_challenge": 120,
        "distribution_shift": 120,
        "maturity_conditioning_shift": 120,
    }
    boundary = selected[selected["cohort"] == "boundary_challenge"]
    assert boundary["regime"].value_counts().to_dict() == {
        regime: 30 for regime in BOUNDARY_REGIMES
    }
    shift = selected[selected["cohort"] == "distribution_shift"]
    assert shift["regime"].value_counts().to_dict() == {
        regime: 60 for regime in SHIFT_REGIMES
    }


def test_distribution_shift_supports_are_admissible_and_disjoint() -> None:
    _, selected = build_parameter_pools()
    shift = selected[selected["cohort"] == "distribution_shift"]
    slow = shift[shift["regime"] == SHIFT_REGIMES[0]]
    fast = shift[shift["regime"] == SHIFT_REGIMES[1]]
    for frame in (slow, fast):
        vectors = frame[PARAMETER_NAMES].to_numpy(dtype=float)
        assert all(validate_parameters(vector)["is_valid"] for vector in vectors)
        assert bool((frame["minimum_hard_bound_distance"] >= 0.05).all())
        assert bool((frame["slow_feller_margin"] > 0.05).all())
        assert bool((frame["fast_feller_margin"] > 0.05).all())
        assert bool((frame["correlation_margin"] > 0.05).all())
        assert bool((frame["ordering_margin"] > 0.10).all())
    assert float(slow["kappa_slow"].min()) >= 0.20
    assert float(slow["kappa_slow"].max()) <= 0.29
    assert float(slow["theta_slow"].min()) >= 0.205
    assert float(slow["theta_slow"].max()) <= 0.2325
    assert float(slow["v0_slow"].min()) >= 0.255
    assert float(slow["v0_slow"].max()) <= 0.2775
    assert float(fast["kappa_fast"].min()) > 10.0
    assert float(fast["kappa_fast"].max()) <= 11.35


def test_development_panel_is_small_nonresearch_sampler_only() -> None:
    _, selected = build_parameter_pools()
    development = build_development_panel(selected)
    assert len(development) == 12
    assert development["development_label"].eq(
        "DEVELOPMENT_SANITY_NOT_RESEARCH_RESULT"
    ).all()
    assert set(development["cohort"]) == {
        "boundary_challenge",
        "distribution_shift",
        "maturity_conditioning_shift",
    }


def _parent_surface() -> R2Surface:
    count = NOMINAL_SLOT_COUNT
    maturities = np.asarray(
        [7 / 365 if key.expiry_rank == 1 else 42 / 365 for key in CANONICAL_SLOT_KEYS]
    )
    return R2Surface(
        prices=tuple(1.0 + index / 100 for index in range(count)),
        mask=tuple([True] * count),
        maturities=tuple(float(value) for value in maturities),
        rates=tuple([0.02] * count),
        carries=tuple([0.01] * count),
        spot=100.0,
        surface_id="PARENT",
        source="synthetic_canonical_double_heston_production_pricer",
        slot_keys=CANONICAL_SLOT_KEYS,
    )


@pytest.mark.parametrize("pattern", MISSING_PATTERNS)
def test_incomplete_surface_preserves_frozen_mask_semantics(pattern: str) -> None:
    parent = _parent_surface()
    derived = make_incomplete_surface(
        parent,
        pattern=pattern,
        surface_id="CHILD",
        sequence_index=0,
    )
    assert sum(derived.mask) >= 10
    for index, (price, valid) in enumerate(zip(derived.prices, derived.mask)):
        if valid:
            assert price == parent.prices[index] > 0.0
        else:
            assert price == 0.0
    assert derived.maturities == parent.maturities
    assert derived.rates == parent.rates
    assert derived.carries == parent.carries
    metadata = derived.metadata["user_metadata"]
    assert metadata["imputation"] == "NONE_MASKED_EXACT_ZERO"
    assert metadata["evaluation_only"] is True


def test_documents_freeze_all_requested_challenge_families() -> None:
    protocol = PROTOCOL_DOC.read_text(encoding="utf-8")
    audit = AUDIT_DOC.read_text(encoding="utf-8")
    for phrase in (
        "FROZEN_BEFORE_MODEL3_RESEARCH_RESULTS",
        "boundary_challenge",
        "distribution_shift",
        "maturity_conditioning_shift",
        "incomplete_observation",
        "OOD_SENSITIVE_NEGATIVE",
        "INCONCLUSIVE",
    ):
        assert phrase in protocol
    for phrase in (
        "Parameter bounds and interior sampler",
        "Prior boundary/OOD ideas",
        "Masks and incomplete surfaces",
        "Evaluation metrics",
        "Primary interfaces",
    ):
        assert phrase in audit
    assert "--remote-checkpoint-confirmed" in protocol
