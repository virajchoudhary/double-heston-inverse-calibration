"""Standalone pre-freeze checks for the R2 generation contract."""

from __future__ import annotations

import math
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "r2_synthetic_generation_FINAL.yaml"
LEGACY_CONFIG_PATH = ROOT / "configs" / "ann_dataset_FIRST_RESEARCH.yaml"
CONTRACT_PATH = ROOT / "docs" / "R2_SYNTHETIC_GENERATION_CONTRACT.md"
FREEZE_MARKER_PATH = ROOT / "evidence" / "R2_CONTRACT_FROZEN_BEFORE_PILOT.txt"


def _config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_contract_identity_and_canonical_r2() -> None:
    config = _config()
    representation = config["representation"]
    assert config["schema_version"] == "1.0"
    assert config["contract_name"] == "R2_SYNTHETIC_GENERATION_FINAL"
    assert config["status"] == "FROZEN_BEFORE_PILOT"
    assert config["tracking_issue"] == 27
    assert config["final_10k_generated"] is False
    assert representation["name"] == "FROZEN_R2_RANKED_TWO_EXPIRY_CENTRAL_FIVE"
    assert representation["version"] == "1.0"
    assert representation["nominal_slot_count"] == 20
    assert representation["expiry_ranks"] == [1, 2]
    assert representation["target_log_moneyness"] == [
        -0.10, -0.05, 0.00, +0.05, +0.10,
    ]
    assert representation["option_types"] == ["call", "put"]
    assert representation["slot_order"] == (
        "option_type_major_then_expiry_rank_then_log_moneyness"
    )
    assert representation["implementation_package"] == "src/r2_representation"
    assert representation["synthetic_mask_policy"] == (
        "all_true_complete_by_construction"
    )
    assert representation["price_normalization"] == "spot_normalized_with_spot_100"


def test_parameter_order_and_reviewed_sampling_policy() -> None:
    config = _config()
    parameter_contract = config["parameter_contract"]
    assert parameter_contract["order"] == [
        "kappa_slow",
        "theta_slow",
        "sigma_slow",
        "rho_slow",
        "v0_slow",
        "kappa_fast",
        "theta_fast",
        "sigma_fast",
        "rho_fast",
        "v0_fast",
    ]
    assert parameter_contract["reviewed_sampling_config"] == (
        "configs/parameter_sampling_REVIEWED.yaml"
    )
    assert parameter_contract["transforms"] == (
        "reuse_reviewed_latent_lhs_conditional_transforms_unchanged"
    )
    assert parameter_contract["hard_constraints"] == (
        "reuse_reviewed_hard_constraints_unchanged"
    )
    assert parameter_contract["acceptance_margin_policy"] == (
        "reuse_reviewed_distribution_acceptance_gate_unchanged"
    )
    sampling = config["sampling"]
    assert sampling["method"] == (
        "scipy_stats_qmc_latin_hypercube_latent_coordinates"
    )
    assert sampling["candidate_id_rule"] == (
        "zero_based_row_order_within_distribution"
    )
    assert sampling["selection_order"] == (
        "distribution_then_candidate_id_ascending"
    )
    assert sampling["fixed_pool_policy"] == (
        "generate_entire_pool_retain_all_candidates_and_rejections_"
        "no_refill_no_reseed"
    )
    assert sampling["insufficient_pool_behavior"] == "HARD_FAIL"


def test_final_and_pilot_quotas_pools_and_seeds() -> None:
    config = _config()
    assert config["quotas"]["final_clean_core"] == {
        "total_surfaces": 10_000,
        "distributions": {"interior_train": 8_334, "wide_valid_train": 1_666},
        "splits": {
            "train": {
                "total": 7_500,
                "interior_train": 6_250,
                "wide_valid_train": 1_250,
            },
            "validation": {
                "total": 1_250,
                "interior_train": 1_042,
                "wide_valid_train": 208,
            },
            "test": {
                "total": 1_250,
                "interior_train": 1_042,
                "wide_valid_train": 208,
            },
        },
    }
    assert config["quotas"]["development_pilot_not_final_research_dataset"] == {
        "total_surfaces": 240,
        "distributions": {"interior_train": 200, "wide_valid_train": 40},
        "splits": {
            "train": {
                "total": 180,
                "interior_train": 150,
                "wide_valid_train": 30,
            },
            "validation": {"total": 30, "interior_train": 25, "wide_valid_train": 5},
            "test": {"total": 30, "interior_train": 25, "wide_valid_train": 5},
        },
    }
    assert config["sampling"]["pools"] == {
        "pilot": {
            "interior_train": {"candidate_count": 400, "required_quota": 200},
            "wide_valid_train": {"candidate_count": 200, "required_quota": 40},
        },
        "final": {
            "interior_train": {"candidate_count": 15_000, "required_quota": 8_334},
            "wide_valid_train": {"candidate_count": 5_000, "required_quota": 1_666},
        },
    }
    assert config["sampling"]["seeds"] == {
        "pilot_interior_train": 20260822,
        "pilot_wide_valid_train": 20260823,
        "final_interior_train": 20260807,
        "final_wide_valid_train": 20260808,
    }


def test_atomic_deterministic_split_policy() -> None:
    split = _config()["split_assignment"]
    assert split["algorithm"] == (
        "accepted_candidates_sorted_by_candidate_id_exact_nonoverlapping_"
        "slices_per_distribution"
    )
    assert split["random_permutation_used"] is False
    assert split["atomic_unit"] == "whole_parameter_vector_and_whole_surface"
    assert split["quote_row_splitting_allowed"] is False
    assert split["cross_split_parameter_overlap_allowed"] is False
    assert split["duplicate_parameter_vectors_allowed"] is False


def test_synthetic_only_conditioning_lattice() -> None:
    conditioning = _config()["conditioning"]
    lattice = conditioning["lattice"]
    rank1 = lattice["rank1_dte_days"]
    gaps = lattice["rank2_gap_dte_days"]
    rates = lattice["rates"]
    offsets = lattice["carry_offsets"]
    assert conditioning["classification"] == (
        "SYNTHETIC_ONLY_ENGINEERING_RESEARCH_DESIGN_NOT_REAL_MARKET_DISTRIBUTION_CLAIM"
    )
    assert conditioning["real_market_inputs_used"] is False
    assert conditioning["spot"] == 100.0
    assert conditioning["deterministic_mode"] == (
        "predeclared_lattice_mixed_radix_stride"
    )
    assert conditioning["seeds"] == {"pilot": 20260822, "final": 20260807}
    assert conditioning["strides"] == {"pilot": 997, "final": 1103}
    assert rank1 == [7, 14, 21, 30, 45, 60, 75, 90]
    assert gaps == [7, 14, 21, 30, 45, 60, 90]
    assert rates == [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    assert offsets == [-0.02, -0.01, 0.00, 0.01, 0.02, 0.03]
    combination_count = len(rank1) * len(gaps) * len(rates) * len(offsets)
    assert lattice["combination_count"] == combination_count == 2016
    assert all(gap > 0 for gap in gaps)
    assert max(rank1) + max(gaps) == 180
    assert all(
        math.gcd(stride, combination_count) == 1
        for stride in conditioning["strides"].values()
    )
    assert conditioning["invariants"]["rank2_dte_strictly_greater_than_rank1"] is True
    assert conditioning["invariants"][
        "observed_nse_option_spot_future_rate_or_carry_used"
    ] is False


def test_pricing_noise_exclusions_and_failure_gates() -> None:
    config = _config()
    pricing = config["pricing"]
    assert pricing["production_source"] == "src/double_heston.py"
    assert pricing["entrypoint"] == "price_double_heston_surface"
    assert pricing["node_count"] == 64
    assert pricing["target_strike_rule"] == "spot_exp_target_log_moneyness"
    assert pricing["rank_pricing"] == "constant_conditioning_per_rank_piece"
    assert pricing["clipping_or_imputation_allowed"] is False
    assert pricing["replacement_on_failure_allowed"] is False
    assert config["noise"] == {
        "clean_core_level": 0.0,
        "future_levels_are_separate_derivatives": [0.005, 0.01, 0.02],
        "generation_in_this_milestone": False,
    }
    exclusions = config["exclusions"]
    assert exclusions["clean_core_excludes"] == [
        "boundary_challenge",
        "ood_test",
        "noisy_copies",
        "real_market_data",
    ]
    assert exclusions["global_sampler_status_preserved"] == (
        "NEEDS_SAMPLER_CORRECTION"
    )
    assert exclusions["challenge_stress_ready"] is False
    failures = config["failure_retention"]
    assert failures["pricing_failure_policy"] == (
        "retain_candidate_surface_id_parameters_conditioning_and_error_"
        "then_fail_closed"
    )
    assert failures["rejection_policy"] == (
        "retain_every_candidate_and_all_rejection_reasons"
    )
    assert failures["silent_replacement_or_refill_forbidden"] is True


def test_execution_gates_remain_closed() -> None:
    config = _config()
    gates = config["execution_gates"]
    assert gates["contract_freeze_before_pilot_output"] == "REQUIRED"
    assert gates["pilot_command_requires_contract_checkpoint"] is True
    assert gates["readiness_requires_verified_pilot"] is True
    assert gates["final_readiness_command"] == (
        "python -m src.r2_synthetic_generation readiness "
        "--output evidence/final_r2_candidate_pool_readiness_20260822"
    )
    assert gates["final_10k_generation_command"] == (
        "NOT_AUTHORIZED_IN_THIS_MILESTONE"
    )
    assert gates["final_10k_requires_separate_explicit_command_and_authorization"] is True
    assert gates["training_commands_in_this_milestone"] == "NONE"
    assert config["final_10k_generated"] is False


def test_freeze_marker_asserts_no_prior_authoritative_pilot() -> None:
    assert FREEZE_MARKER_PATH.is_file()
    text = FREEZE_MARKER_PATH.read_text(encoding="utf-8")
    assert "CONTRACT_FROZEN_BEFORE_PILOT" in text
    assert (
        "NO AUTHORITATIVE PILOT OUTPUT EXISTED WHEN THIS CONTRACT SHA WAS "
        "COMMITTED" in text
    )


def test_legacy_108_plan_is_historical_not_active() -> None:
    legacy = yaml.safe_load(LEGACY_CONFIG_PATH.read_text(encoding="utf-8"))
    superseded = legacy["superseded_by"]
    assert legacy["status"] == (
        "HISTORICAL_DEPRECATED_SUPERSEDED_BY_R2_FINAL_CONTRACT"
    )
    assert superseded["active_config"] == (
        "configs/r2_synthetic_generation_FINAL.yaml"
    )
    assert superseded["representation"] == (
        "FROZEN_R2_RANKED_TWO_EXPIRY_CENTRAL_FIVE"
    )
    assert superseded["reason"] == (
        "legacy_fixed_calendar_108_grid_is_not_the_active_final_representation"
    )
    assert legacy["ann_representation"]["input_dimension"] == 108
    assert legacy["ann_representation"]["log_moneyness"] == [
        -0.30, -0.20, -0.10, -0.05, 0, 0.05, 0.10, 0.20, 0.30,
    ]
    assert legacy["ann_representation"]["maturity_days"] == [
        7, 14, 30, 60, 90, 180,
    ]


def test_contract_document_states_material_boundaries() -> None:
    text = CONTRACT_PATH.read_text(encoding="utf-8")
    required_statements = (
        "FROZEN_BEFORE_PILOT",
        "FINAL_10K_NOT_GENERATED",
        "ANN_RESEARCH_TRAINING = NOT_STARTED",
        "No range or\nthreshold is tuned by this milestone.",
        "There is no refill loop, second seed, range change,\nthreshold change, "
        "or replacement of difficult selected rows.",
        "No observed NSE option price",
        "DTE2\nis always strictly greater than DTE1",
        "Prices come only from unchanged production source",
        "A pricing or numerical failure preserves the candidate identity",
        "The contract checkpoint must be committed before any pilot output exists.",
    )
    for statement in required_statements:
        assert statement in text
