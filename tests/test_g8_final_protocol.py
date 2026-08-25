"""Focused contracts for the frozen G8 real-market protocol."""

from __future__ import annotations

from pathlib import Path

import yaml

from scripts.validate_g8_protocol import (
    check_candidate,
    is_development_observation,
    sha256_file,
    validate_config,
    verify_checkpoint_registry,
)


def test_config_is_frozen_and_execution_disabled() -> None:
    config = validate_config(Path("configs/g8_final_real_market.yaml"))
    assert config["status"] == "FROZEN_PENDING_UNTOUCHED_DATA_ACQUISITION"
    assert config["execution"]["authorized_now"] is False
    assert config["execution"]["data_acquisition_authorized_now"] is False
    assert config["data_blocker"]["status"] == "BLOCKED_PENDING_PROTOCOL_COMPLIANT_ACQUISITION"


def test_development_registry_contains_pilot_without_relabeling_it() -> None:
    config = validate_config(Path("configs/g8_final_real_market.yaml"))
    pilot = config["development_exclusions"]["ntpc_single_stock_pilot"]
    assert pilot["valuation_date"] == "2026-07-15"
    assert pilot["classification"] == "DEVELOPMENT_PILOT_NEVER_FINAL_EVALUATION"
    assert pilot["realized_close_window"]["last_date_inclusive"] == "2026-07-28"


def test_known_development_identities_fail_closed() -> None:
    excluded, reason = is_development_observation("NTPC", "2026-07-15")
    assert excluded is True
    assert reason == "BEFORE_G8_DATE_FLOOR"
    assert is_development_observation("cipla", "2026-07-01")[0] is True
    assert is_development_observation("POWERGRID", "2026-07-29")[0] is True
    assert is_development_observation("NIFTY", "2027-03-01") == (
        True,
        "NIFTY_REFERENCE_ONLY_PROHIBITED",
    )


def test_future_identity_passes_registry_but_reads_nothing() -> None:
    result = check_candidate("infy", "2026-10-01")
    assert result == {
        "symbol": "INFY",
        "valuation_date": "2026-10-01",
        "development_excluded": False,
        "reason": "",
        "contract_key_overlap_checked": False,
        "market_data_read": False,
        "model_executed": False,
    }


def test_pre_floor_identity_is_unconditionally_excluded() -> None:
    assert is_development_observation("UNKNOWN_SYMBOL", "2026-08-01") == (
        True,
        "BEFORE_G8_DATE_FLOOR",
    )


def test_invalid_candidate_date_is_rejected() -> None:
    try:
        check_candidate("INFY", "2026-02-30")
    except ValueError as exc:
        assert "invalid ISO date" in str(exc)
    else:
        raise AssertionError("invalid date accepted")


def test_protocol_document_and_machine_twin_share_core_boundaries() -> None:
    config_path = Path("configs/g8_final_real_market.yaml")
    protocol = Path("docs/G8_FINAL_REAL_MARKET_PROTOCOL.md").read_text(encoding="utf-8")
    audit = Path("docs/G8_PREEXECUTION_AUDIT.md").read_text(encoding="utf-8")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert "G8_PROTOCOL_FROZEN" in protocol
    assert "DEVELOPMENT PILOT" in audit
    assert "NO_CLEAR_PRICING_FAMILY_WINNER" in protocol
    assert config["pricing_model_family_comparison"]["winner_rule"]["default_label"] in protocol
    assert config["inverse_method_comparison"]["interpretation"][
        "real_parameter_recovery_metric"
    ] == "NOT_APPLICABLE_NO_REAL_TRUTH"
    assert "NOT_APPLICABLE_NO_REAL_TRUTH" in protocol


def test_document_does_not_misstate_double_heston_transform_module() -> None:
    protocol = Path("docs/G8_FINAL_REAL_MARKET_PROTOCOL.md").read_text(encoding="utf-8")
    assert "src.double_heston.unconstrained_to_parameters" not in protocol
    assert "src.calibrate_double_heston.unconstrained_to_parameters" in protocol


def test_validation_tool_reports_stable_config_identity() -> None:
    path = Path("configs/g8_final_real_market.yaml")
    first = sha256_file(path)
    second = sha256_file(path)
    assert first == second
    assert len(first) == 64


def test_surface_support_and_comparison_contracts_are_locked() -> None:
    config = validate_config(Path("configs/g8_final_real_market.yaml"))
    construction = config["surface_construction"]
    assert construction["minimum_usable_slots_total"] == 12
    assert construction["minimum_pricing_family_calibration_slots"] == 6
    assert construction["minimum_pricing_family_holdout_slots"] == 3
    assert construction["masked_slots"] == "explicit_false_with_zero_placeholder_no_imputation"
    family = config["pricing_model_family_comparison"]
    assert family["winner_rule"]["default_label"] == "NO_CLEAR_PRICING_FAMILY_WINNER"
    heston = family["models"]["STANDARD_HESTON"]
    assert heston["parameters"] == ["kappa", "theta", "sigma", "rho", "v0"]
    assert heston["pricing_interface"] == (
        "src.double_heston.heston_log_characteristic_exponent_with_pilot_put_call_parity_wrapper"
    )
    traditional = config["inverse_method_comparison"]["methods"]["TRADITIONAL"]
    assert traditional["real_g8_effective_start_count"] == 2
    assert traditional["forbidden_start_strategy"] == "disclosed_target_perturbation"


def test_failure_runtime_and_contract_overlap_requirements_are_frozen() -> None:
    config = validate_config(Path("configs/g8_final_real_market.yaml"))
    assert config["failure_reporting"]["atomic_evidence_bundle_required"] is True
    assert config["failure_reporting"]["no_silent_rerun_with_changed_settings"] is True
    assert config["metrics"]["runtime"]["wall_seconds_per_surface_by_model"] is True
    assert config["metrics"]["runtime"]["neural_batched_eval_mode_no_grad_cpu_ms_per_surface"] is True
    assert config["data_blocker"]["acquisition_requirements"][
        "compare_candidate_contract_keys_against_development_registry_fail_closed"
    ] is True


def test_heston_schedule_and_checkpoint_registry_are_pinned() -> None:
    config = validate_config(Path("configs/g8_final_real_market.yaml"))
    heston = config["pricing_model_family_comparison"]["models"]["STANDARD_HESTON"]
    schedule = heston["deterministic_start_generator"]
    assert schedule["seed"] == 20260912
    assert schedule["vector_count"] == 8
    assert heston["optimizer_coordinate_map"]["rho"] == "0.95 * tanh(raw_coordinate[4])"
    double_heston = config["pricing_model_family_comparison"]["models"]["DOUBLE_HESTON"]
    dh_schedule = double_heston["deterministic_start_generator"]
    assert dh_schedule["seed"] == 20260922
    assert dh_schedule["vector_count"] == 12
    restore = config["inverse_method_comparison"]["shared_rules"]["checkpoint_restore_contract"]
    assert len(restore["required_best_validation_checkpoints"]) == 6
    assert all(item["sha256"] for item in restore["required_best_validation_checkpoints"])
    assert restore["required_before_data_acquisition"] is True
    assert restore["executable_preacquisition_command"] == (
        "python scripts/validate_g8_protocol.py check-checkpoints"
    )


def test_winner_aggregation_and_mask_shift_are_frozen() -> None:
    config = validate_config(Path("configs/g8_final_real_market.yaml"))
    winner = config["pricing_model_family_comparison"]["winner_rule"]
    assert winner["aggregation_unit"] == "ONE_UNWEIGHTED_OBSERVATION_PER_ELIGIBLE_SURFACE"
    assert winner["failure_denominator"] == "all_eligible_surfaces_assigned_to_the_family_test"
    disclosure = config["metrics"]["mask_support_disclosure"]
    assert "partial mask" in disclosure["distribution_shift_warning"]
    assert disclosure["report_by_expiry_rank_and_option_type"] is True


def test_staged_checkpoint_gate_passes_without_model_execution() -> None:
    report = verify_checkpoint_registry(
        validate_config(Path("configs/g8_final_real_market.yaml"))
    )
    assert report["checkpoint_count"] == 6
    assert len(report["results"]) == 6
    assert all(item["status"] == "PASS" for item in report["results"])
    assert report["all_checks_passed"] is True
    assert report["pricing_executed"] is False
    assert report["calibration_executed"] is False
    assert report["evaluation_executed"] is False
