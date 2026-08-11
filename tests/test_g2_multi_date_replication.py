from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.run_g2_multi_date_identifiability as original
import scripts.run_g2_multi_date_replication as replication
from src.calibrate_double_heston import load_hard_safety_bounds


ORIGINAL_MULTI_DATE_HASHES = {
    "docs/G2_MULTI_DATE_IDENTIFIABILITY.md": "927E7013C5E5333788CA0900388BC495069F4E71F154A4ECE0224F59E95D20C1",
    "scripts/run_g2_multi_date_identifiability.py": "7742E03DE61D719D640AD06E3B4A2E31BEDB11EAF739C229F038B4C9DAF48E42",
    "tests/test_g2_multi_date_identifiability.py": "F5A8797496852FD9CD8C6CC827346A1D45153BCB8D0FB93D46920A5D6BDEC5EC",
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _one_sample() -> pd.DataFrame:
    bounds = load_hard_safety_bounds(original.baseline.BOUNDS_PATH)
    return original.baseline.select_representative_parameters(
        bounds, per_distribution=4
    ).head(1)


def test_replication_changes_only_predeclared_cir_path_seed() -> None:
    contract = replication.replication_contract().iloc[0]
    assert replication.ORIGINAL_CIR_PATH_SEED == original.ANALYSIS_SEED == 20260811
    assert replication.REPLICATION_CIR_PATH_SEED == 27182818
    assert replication.REPLICATION_CIR_PATH_SEED != original.ANALYSIS_SEED
    assert replication.RECOVERY_NOISE_START_SEED == original.ANALYSIS_SEED
    assert contract["only_changed_scientific_field"] == "cir_path_seed"
    assert contract["additional_observables"] == 0


def test_replication_contract_freezes_the_completed_experiment() -> None:
    contract = replication.replication_contract().iloc[0]
    assert contract["valuation_dates"] == "2026-07-01|2026-07-15|2026-07-22"
    assert contract["date_gaps_days"] == "14|7"
    assert contract["maturity_profiles_days"] == (
        "2026-07-01:27|55;2026-07-15:13|41;2026-07-22:6|34"
    )
    assert contract["moneyness_nodes"] == "-0.10|-0.05|0.00|0.05|0.10"
    assert contract["option_types"] == "call|put"
    assert contract["canonical_target_count"] == 10
    assert contract["designs"] == "A|B|C|D"
    assert contract["optimizer"] == "L-BFGS-B"
    assert contract["optimizer_maxiter"] == 80
    assert contract["jacobian_target_sample_count"] == 8
    assert contract["recovery_target_sample_count"] == 2
    assert contract["starts_per_target"] == 3
    assert contract["noise_levels"] == "0.000|0.005|0.010"


def test_replication_paths_replay_exactly_but_differ_from_original_seed() -> None:
    samples = _one_sample()
    first = replication.simulate_replication_state_paths(samples)
    second = replication.simulate_replication_state_paths(samples)
    original_states = original.simulate_state_paths(samples)
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    np.testing.assert_array_equal(
        first[["v_slow_t0", "v_fast_t0"]],
        original_states[["v_slow_t0", "v_fast_t0"]],
    )
    assert not np.array_equal(
        first[list(original.NUISANCE_STATE_NAMES)].to_numpy(float),
        original_states[list(original.NUISANCE_STATE_NAMES)].to_numpy(float),
    )
    assert first.iloc[0]["base_cir_path_seed"] == 27182818
    assert first.iloc[0]["transition_seed_t0_to_t1"] == 27182818


def test_weakest_direction_comparison_is_sign_invariant() -> None:
    parameters = list(original.CANONICAL_TARGET_NAMES)
    rows_left: list[dict[str, object]] = []
    rows_right: list[dict[str, object]] = []
    for design_index, design_id in enumerate(("A", "B", "C", "D")):
        vector = np.arange(1.0, 11.0) + design_index
        vector /= np.linalg.norm(vector)
        for parameter, loading in zip(parameters, vector, strict=True):
            rows_left.append(
                {
                    "design_id": design_id,
                    "sample_id": "sample",
                    "parameter": parameter,
                    "weakest_direction_loading": loading,
                    "absolute_weakest_direction_loading": abs(loading),
                }
            )
            rows_right.append(
                {
                    "design_id": design_id,
                    "sample_id": "sample",
                    "parameter": parameter,
                    "weakest_direction_loading": -loading,
                    "absolute_weakest_direction_loading": abs(loading),
                }
            )
    stability = replication.weakest_direction_stability(
        pd.DataFrame(rows_left), pd.DataFrame(rows_right)
    )
    np.testing.assert_allclose(stability["median_absolute_cosine"], 1.0)
    assert stability["top3_overlap_count"].eq(3).all()


def test_hypothesis_rules_confirm_the_canonical_qualitative_pattern() -> None:
    rows = []
    values = {
        "A": (0.00, 4.5e7),
        "B": (1.00, 7.2e3),
        "C": (0.75, 5.1e4),
        "D": (1.00, 6.3e3),
    }
    for design_id, (rank, condition) in values.items():
        row = {
            "design_id": design_id,
            "replication_practical_full_rank_frequency": rank,
            "replication_median_condition_number": condition,
        }
        for label in ("clean", "noise_0_5pct", "noise_1pct"):
            row[f"replication_{label}_pass_frequency"] = 0.0
        rows.append(row)
    diagnostic = {
        "verdict": "MULTI_DATE_DIAGNOSTIC = INSUFFICIENT",
        "design_pass": {design_id: False for design_id in values},
    }
    hypotheses = replication.classify_hypotheses(pd.DataFrame(rows), diagnostic)
    assert hypotheses.set_index("hypothesis")["status"].to_dict() == {
        "H1": "REPLICATED",
        "H2": "REPLICATED",
        "H3": "REPLICATED",
        "H4": "REPLICATED",
    }
    decision = replication.decide_replication(hypotheses, diagnostic)
    assert decision["replication_verdict"] == "REPLICATION = CONFIRMED"
    assert not decision["g2_changed"]


def test_original_multi_date_evidence_remains_byte_identical() -> None:
    for relative, expected in ORIGINAL_MULTI_DATE_HASHES.items():
        assert _digest(replication.REPOSITORY_ROOT / relative) == expected


def test_replication_output_contract_is_separate_and_complete() -> None:
    assert replication.DEFAULT_OUTPUT_ROOT != replication.ORIGINAL_OUTPUT_ROOT
    assert replication.DEFAULT_REPORT_PATH != original.DEFAULT_REPORT_PATH
    assert len(replication.EXPECTED_OUTPUT_FILES) == 17
    assert len(set(replication.EXPECTED_OUTPUT_FILES)) == 17
    assert "decision.json" in replication.EXPECTED_OUTPUT_FILES
    assert "figures/mentor_replication_summary.png" in replication.EXPECTED_OUTPUT_FILES
