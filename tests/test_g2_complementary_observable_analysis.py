from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.run_g2_complementary_observable_analysis as complementary
from src.calibrate_double_heston import load_hard_safety_bounds


VALID_PARAMETERS = np.asarray(
    [1.2, 0.04, 0.25, -0.35, 0.03, 3.0, 0.02, 0.25, -0.25, 0.02],
    dtype=float,
)


def test_predeclared_design_matrix_is_small_and_preserves_ten_targets() -> None:
    matrix = complementary.experiment_matrix().set_index("design_id")
    assert list(matrix.index) == ["A", "B", "C", "D"]
    assert matrix["complementary_observation_count"].tolist() == [0, 1, 2, 3]
    assert matrix["canonical_target_count"].eq(10).all()
    assert not matrix["broad_feature_search"].any()
    assert matrix.loc["B", "complementary_observables"] == "oracle_total_variance"
    assert matrix.loc["D", "complementary_observables"].split(";") == [
        "log_rv_21",
        "log_rv_126",
        "rv_block_persistence",
    ]


def test_windows_and_noise_are_frozen_before_recovery() -> None:
    assert complementary.SHORT_RV_WINDOW == 21
    assert complementary.LONG_RV_WINDOW == 126
    assert complementary.HISTORY_TRADING_DAYS == 252
    assert complementary.PERSISTENCE_BLOCK_DAYS == 5
    assert complementary.PERSISTENCE_LAG_BLOCKS == 4
    assert complementary.NOISE_LEVELS == (0.0, 0.005, 0.01)
    assert complementary.NOISY_WARM_START_COUNT == 5
    assert complementary.LOG_RV_NOISE_SD == {0.0: 0.0, 0.005: 0.05, 0.01: 0.10}
    assert complementary.PERSISTENCE_NOISE_SD == {0.0: 0.0, 0.005: 0.05, 0.01: 0.10}


def test_decision_rule_is_part_of_predeclared_contract() -> None:
    assert complementary.predeclared_contract()["decision_rule"] == {
        "cluster_ratio_max": 0.50,
        "material_solution_ratio_max": 0.50,
        "parameter_error_ratio_max": 0.75,
        "minimum_distinct_case_count": 3,
        "required_case_ids": complementary.EXPECTED_CASE_IDS,
        "required_market_securities": complementary.MARKET_SECURITIES,
        "required_market_valuation_dates": complementary.MARKET_VALUATION_DATES,
    }
    assert complementary.predeclared_contract()["reporting_contract"] == {
        "global_clean_case_ids": complementary.EXPECTED_CASE_IDS,
        "matched_clean_noisy_case_ids": complementary.NOISE_COMPARISON_CASE_IDS,
        "clean_noisy_comparison_requires_identical_case_population": True,
    }


def test_oracle_total_variance_constrains_sum_not_allocation() -> None:
    base = complementary.model_features(VALID_PARAMETERS)
    shifted = VALID_PARAMETERS.copy()
    shifted[4] += 0.005
    shifted[9] -= 0.005
    changed = complementary.model_features(shifted)
    assert np.isclose(
        base["oracle_total_variance"], changed["oracle_total_variance"], rtol=0.0, atol=1.0e-15
    )
    assert base["log_rv_21"] != changed["log_rv_21"]


def test_causal_history_is_deterministic_and_ends_at_option_state() -> None:
    first = complementary.simulate_causal_return_history(VALID_PARAMETERS, 0)
    second = complementary.simulate_causal_return_history(VALID_PARAMETERS, 0)
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    assert len(first) == 252
    assert first["trading_day_offset"].min() == -251
    assert first["trading_day_offset"].max() == 0
    assert first.iloc[-1]["v_slow_end"] == VALID_PARAMETERS[4]
    assert first.iloc[-1]["v_fast_end"] == VALID_PARAMETERS[9]
    observables = complementary.path_observables(first)
    assert observables["rv_21"] > 0.0
    assert observables["rv_126"] > 0.0
    assert -1.0 <= observables["rv_block_persistence"] <= 1.0


def test_expected_horizons_have_distinct_slow_fast_weights() -> None:
    short_slow = complementary._expected_trailing_variance(1.2, 0.04, 0.03, 21)
    short_fast = complementary._expected_trailing_variance(3.0, 0.02, 0.03, 21)
    long_slow = complementary._expected_trailing_variance(1.2, 0.04, 0.03, 126)
    long_fast = complementary._expected_trailing_variance(3.0, 0.02, 0.03, 126)
    assert short_slow != long_slow
    assert short_fast != long_fast
    assert abs(short_fast - long_fast) > abs(short_slow - long_slow)


def test_persistence_is_transparent_finite_and_sensitive_to_kappa() -> None:
    base = complementary.expected_persistence(VALID_PARAMETERS)
    changed = VALID_PARAMETERS.copy()
    changed[5] = 5.0
    changed[7] = 0.30  # retain a positive fast-factor Feller gap
    assert np.isfinite(base)
    assert -1.0 < base < 1.0
    assert complementary.expected_persistence(changed) != base


def test_market_feasibility_fails_closed_without_bulk_acquisition() -> None:
    frame = complementary.market_feasibility()
    assert len(frame) == 12
    assert set(frame["security"]) == {"NTPC", "CIPLA", "INFY", "HDFCBANK"}
    assert frame["required_trading_day_lookback"].eq(252).all()
    assert frame["new_data_acquired"].eq(False).all()  # noqa: E712
    assert frame["observable_contract"].eq("UNRESOLVED").all()
    assert frame["corporate_action_handling"].eq("UNIMPLEMENTED_IN_STAGE_A").all()


def test_quick_replay_is_byte_deterministic(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_report = tmp_path / "first.md"
    second_report = tmp_path / "second.md"
    first = complementary.run_analysis(
        output_root=first_root,
        report_path=first_report,
        node_count=8,
        sample_limit=1,
        skip_recovery=True,
    )
    second = complementary.run_analysis(
        output_root=second_root,
        report_path=second_report,
        node_count=8,
        sample_limit=1,
        skip_recovery=True,
    )
    assert first["artifact_hashes"] == second["artifact_hashes"]
    assert first_report.read_bytes() == second_report.read_bytes()
    for relative in first["artifact_hashes"]:
        assert (first_root / relative).read_bytes() == (second_root / relative).read_bytes()


def test_prior_checkpoint_manifests_are_in_protected_snapshot() -> None:
    snapshot = complementary._protected_snapshot(
        complementary.DEFAULT_OUTPUT_ROOT, complementary.DEFAULT_REPORT_PATH
    )
    assert "docs/evidence/G2_CHECKPOINT_MANIFEST.json" in snapshot
    assert "docs/evidence/G2_GLOBAL_AMBIGUITY_MANIFEST.json" in snapshot
    for relative in (
        "docs/G2_IDENTIFIABILITY_ANALYSIS.md",
        "docs/G2_GLOBAL_AMBIGUITY_ANALYSIS.md",
    ):
        digest = hashlib.sha256((complementary.REPOSITORY_ROOT / relative).read_bytes()).hexdigest().upper()
        assert snapshot[relative] == digest


def test_render_only_fails_closed_without_preserved_artifacts(tmp_path: Path) -> None:
    try:
        complementary.render_existing_outputs(
            output_root=tmp_path / "missing", report_path=tmp_path / "report.md"
        )
    except FileNotFoundError as error:
        assert "Missing canonical replay artifacts" in str(error)
    else:
        raise AssertionError("render-only replay must require canonical CSV/JSON artifacts")


def _decision_inputs(*, market_ready: bool, truth_pass: bool, improve: bool):
    recovery = pd.DataFrame(
        [
            {
                "design_id": design,
                "noise_level": 0.0,
                "best_solution_parameter_rmse_median": error,
            }
            for design, error in (
                ("A", 0.20),
                ("C", 0.10 if improve else 0.19),
                ("D", 0.10 if improve else 0.19),
            )
        ]
    )
    ambiguity = pd.DataFrame(
        [
            {
                "design_id": design,
                "noise_level": 0.0,
                "scaled_parameter_cluster_count": clusters,
                "materially_displaced_solution_count": material,
                "near_equivalent_case_count": cases,
            }
            for design, clusters, material, cases in (
                ("A", 40, 40, 4),
                ("C", 10 if improve else 35, 10 if improve else 35, 4),
                ("D", 10 if improve else 35, 10 if improve else 35, 4),
            )
        ]
    )
    market = pd.DataFrame(
        [
            {
                "security": security,
                "valuation_date": valuation_date,
                "observable_contract": "RESOLVED" if market_ready else "UNRESOLVED",
            }
            for security in complementary.MARKET_SECURITIES
            for valuation_date in complementary.MARKET_VALUATION_DATES
        ]
    )
    truth = pd.DataFrame(
        [
            {
                "design_id": design,
                "case_id": case_id,
                "truth_passes_complementary_screen": truth_pass,
            }
            for design in ("C", "D")
            for case_id in complementary.EXPECTED_CASE_IDS
        ]
    )
    return recovery, ambiguity, market, truth


def test_decision_vocabulary_and_truth_validity_gate_are_non_vacuous() -> None:
    promising = complementary.classify_decision(
        *_decision_inputs(market_ready=True, truth_pass=True, improve=True)
    )
    information = complementary.classify_decision(
        *_decision_inputs(market_ready=False, truth_pass=True, improve=True)
    )
    insufficient = complementary.classify_decision(
        *_decision_inputs(market_ready=True, truth_pass=True, improve=False)
    )
    invalid = complementary.classify_decision(
        *_decision_inputs(market_ready=True, truth_pass=False, improve=True)
    )
    assert promising["complementary_observable"] == "PROMISING"
    assert information["complementary_observable"] == "INFORMATION_ONLY"
    assert insufficient["complementary_observable"] == "INSUFFICIENT"
    assert invalid["complementary_observable"] == "INSUFFICIENT"
    assert invalid["experiment_validity"] == (
        "NOT_PASSED_TRUTH_OUTSIDE_COMPLEMENTARY_SCREEN"
    )


def test_decision_fails_closed_for_duplicate_partial_and_empty_truth_panels() -> None:
    recovery, ambiguity, market, truth = _decision_inputs(
        market_ready=True, truth_pass=True, improve=True
    )
    variants = [
        pd.concat([truth.iloc[:-1], truth.iloc[[0]]], ignore_index=True),
        truth.iloc[:-1].copy(),
        truth.iloc[:0].copy(),
    ]
    for invalid_truth in variants:
        result = complementary.classify_decision(
            recovery, ambiguity, market, invalid_truth
        )
        assert result["complementary_observable"] == "INSUFFICIENT"
        assert result["experiment_validity"] == "NOT_PASSED_NONCANONICAL_TRUTH_PANEL"


def test_market_panel_must_be_exact_distinct_twelve_rows() -> None:
    recovery, ambiguity, market, truth = _decision_inputs(
        market_ready=True, truth_pass=True, improve=True
    )
    variants = [
        pd.concat([market.iloc[:-1], market.iloc[[0]]], ignore_index=True),
        market.iloc[:-1].copy(),
        market.iloc[:0].copy(),
    ]
    for invalid_market in variants:
        result = complementary.classify_decision(
            recovery, ambiguity, invalid_market, truth
        )
        assert result["complementary_observable"] == "INFORMATION_ONLY"
        assert not result["market_panel_valid"]


def test_noncanonical_modes_cannot_overwrite_canonical_outputs(tmp_path: Path) -> None:
    for kwargs in (
        {"node_count": 32, "sample_limit": None, "skip_recovery": False},
        {"node_count": 64, "sample_limit": 1, "skip_recovery": False},
        {"node_count": 64, "sample_limit": None, "skip_recovery": True},
    ):
        try:
            complementary._validate_run_mode(
                complementary.DEFAULT_OUTPUT_ROOT,
                complementary.DEFAULT_REPORT_PATH,
                **kwargs,
            )
        except ValueError as error:
            assert "Noncanonical" in str(error)
        else:
            raise AssertionError("noncanonical modes must not target canonical evidence")
        complementary._validate_run_mode(
            tmp_path / "out", tmp_path / "report.md", **kwargs
        )


def test_replay_validation_rejects_noncanonical_matrix(tmp_path: Path) -> None:
    contract_path = tmp_path / "predeclared_contract.json"
    payload = complementary.predeclared_contract()
    contract_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    frames = {"experiment_matrix.csv": pd.DataFrame({"design_id": ["A", "B", "C"]})}
    contract = {
        "node_count": complementary.FULL_PRICER_NODE_COUNT,
        "predeclared_contract_sha256": complementary._sha256(contract_path),
    }
    try:
        complementary._validate_replay_evidence(frames, contract, {}, contract_path)
    except ValueError as error:
        assert "A/B/C/D" in str(error)
    else:
        raise AssertionError("render-only replay must reject noncanonical evidence")


def _matched_noise_solution_fixture() -> pd.DataFrame:
    rows = []
    for design_id in ("A", "B", "C", "D"):
        for noise_level in complementary.NOISE_LEVELS:
            case_ids = (
                complementary.EXPECTED_CASE_IDS
                if noise_level == 0.0
                else complementary.NOISE_COMPARISON_CASE_IDS
            )
            for case_id in case_ids:
                case_number = int(case_id.split("_")[1])
                rows.append(
                    {
                        "design_id": design_id,
                        "noise_level": noise_level,
                        "case_id": case_id,
                        "joint_objective_rmse": 0.01,
                        "start_index": 0,
                        "constraint_valid": True,
                        "finite_solution": True,
                        "parameter_rmse_full_range": case_number / 10.0,
                        "maximum_absolute_parameter_error_full_range": case_number / 5.0,
                        "bound_hit": noise_level > 0.0,
                    }
                )
    return pd.DataFrame(rows)


def test_clean_noisy_summary_uses_exact_shared_case_population() -> None:
    recovery = pd.DataFrame(
        [
            {"design_id": design_id, "noise_level": noise_level}
            for design_id in ("A", "B", "C", "D")
            for noise_level in complementary.NOISE_LEVELS
        ]
    )
    matched = complementary.add_matched_noise_comparison(
        recovery, _matched_noise_solution_fixture()
    )
    assert matched["noise_comparison_case_ids"].eq("case_1;case_3").all()
    assert matched["noise_comparison_case_count"].eq(2).all()
    assert np.allclose(matched["noise_comparison_parameter_rmse_median"], 0.2)


def test_replay_decision_comparison_normalizes_json_container_types(tmp_path: Path) -> None:
    contract_path = tmp_path / "predeclared_contract.json"
    contract_path.write_text(
        json.dumps(complementary.predeclared_contract(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    recovery, ambiguity, market, truth = _decision_inputs(
        market_ready=True, truth_pass=True, improve=True
    )
    solutions = _matched_noise_solution_fixture()
    recovery = complementary.add_matched_noise_comparison(recovery, solutions)
    frames = {
        "experiment_matrix.csv": complementary.experiment_matrix(),
        "cases.csv": pd.DataFrame({"case_id": complementary.EXPECTED_CASE_IDS}),
        "jacobian_summary.csv": pd.DataFrame(
            [
                {"case_id": case_id, "design_id": design_id}
                for case_id in complementary.EXPECTED_CASE_IDS
                for design_id in ("A", "B", "C", "D")
            ]
        ),
        "truth_fit_diagnostics.csv": truth,
        "recovery_solutions.csv": solutions,
        "recovery_summary.csv": recovery,
        "ambiguity_summary.csv": ambiguity,
        "market_feasibility.csv": market,
    }
    contract = {
        "node_count": complementary.FULL_PRICER_NODE_COUNT,
        "predeclared_contract_sha256": complementary._sha256(contract_path),
    }
    decision = json.loads(
        json.dumps(complementary.classify_decision(recovery, ambiguity, market, truth))
    )
    complementary._validate_replay_evidence(
        frames,
        contract,
        decision,
        contract_path,
    )


def test_custom_repository_output_paths_are_rejected(tmp_path: Path) -> None:
    complementary._validate_output_paths(
        complementary.DEFAULT_OUTPUT_ROOT, complementary.DEFAULT_REPORT_PATH
    )
    complementary._validate_output_paths(tmp_path / "out", tmp_path / "report.md")
    try:
        complementary._validate_output_paths(
            complementary.REPOSITORY_ROOT / "docs" / "evidence" / "overwrite",
            complementary.DEFAULT_REPORT_PATH,
        )
    except ValueError as error:
        assert "outside the repository" in str(error) or "protected" in str(error)
    else:
        raise AssertionError("custom repository output paths must fail closed")
