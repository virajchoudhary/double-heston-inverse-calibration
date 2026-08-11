from __future__ import annotations

import numpy as np
import pandas as pd

import scripts.run_g2_global_ambiguity_analysis as ambiguity
from src.calibrate_double_heston import load_hard_safety_bounds


def test_case_selection_is_deterministic_balanced_and_rotates_profiles() -> None:
    bounds = load_hard_safety_bounds(ambiguity.baseline.BOUNDS_PATH)
    first = ambiguity.select_cases(bounds)
    second = ambiguity.select_cases(bounds)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 4
    assert first.groupby("distribution").size().to_dict() == {
        "interior_train": 2,
        "wide_valid_train": 2,
    }
    assert first["maturity_profile"].tolist() == [
        "2026-07-01", "2026-07-15", "2026-07-22", "2026-07-01"
    ]


def test_starts_are_deterministic_with_required_neutral_then_broad_pattern() -> None:
    first = ambiguity.deterministic_starts(31415926, 4)
    second = ambiguity.deterministic_starts(31415926, 4)
    assert first[0][0] == "neutral_transform_midpoint"
    np.testing.assert_array_equal(first[0][1], np.zeros(10))
    for (first_name, first_values), (second_name, second_values) in zip(first, second):
        assert first_name == second_name
        np.testing.assert_array_equal(first_values, second_values)
    assert first[1][0] == "deterministic_broad_1"
    assert not np.array_equal(first[1][1], np.zeros(10))


def test_exact_canonical_start_schedule_is_frozen() -> None:
    schedule = ambiguity.start_schedule()
    assert len(schedule) == 120
    assert schedule.groupby(["case_id", "noise_level"]).size().to_dict() == {
        ("case_1", 0.0): 20,
        ("case_1", 0.005): 10,
        ("case_1", 0.01): 10,
        ("case_2", 0.0): 20,
        ("case_3", 0.0): 20,
        ("case_3", 0.005): 10,
        ("case_3", 0.01): 10,
        ("case_4", 0.0): 20,
    }
    assert ambiguity.start_schedule_sha256() == (
        "7831622B03BFEE7AE3E4A5BFA5F458A7153F16198C20D8293139389B004F400E"
    )


def test_canonical_optimizer_runtime_is_frozen() -> None:
    runtime = ambiguity.runtime_provenance()
    assert runtime == {
        "python": "3.13.4 (tags/v3.13.4:8a526ec, Jun  3 2025, 17:46:04) [MSC v.1943 64 bit (AMD64)]",
        "platform": "Windows-11-10.0.26200-SP0",
        "numpy": "2.2.6",
        "scipy": "1.16.2",
        "pandas": "2.3.2",
        "matplotlib": "3.10.6",
    }


def test_noise_comparison_uses_only_cases_with_noisy_runs() -> None:
    solutions = pd.DataFrame(
        [
            {"case_id": "case_1", "noise_level": 0.0},
            {"case_id": "case_1", "noise_level": 0.005},
            {"case_id": "case_2", "noise_level": 0.0},
        ]
    )
    matched = ambiguity.matched_noise_comparison(solutions)
    assert matched.case_id.unique().tolist() == ["case_1"]


def test_render_only_fails_closed_without_preserved_artifacts(tmp_path) -> None:
    try:
        ambiguity.render_existing_outputs(
            output_root=tmp_path / "missing",
            report_path=tmp_path / "report.md",
        )
    except FileNotFoundError as error:
        assert "Missing canonical replay artifacts" in str(error)
    else:
        raise AssertionError("render-only replay must require preserved artifacts")


def test_complete_linkage_does_not_chain_two_separated_basins() -> None:
    points = np.asarray([[0.0, 0.0], [0.09, 0.0], [0.18, 0.0]], dtype=float)
    labels = ambiguity.complete_linkage_clusters(points, cutoff=0.10)
    assert labels.tolist() == [1, 1, 2]


def test_basin_and_global_verdict_use_predeclared_thresholds() -> None:
    near = pd.DataFrame(
        {
            "cluster_id": [1, 1, 2, 2, 2],
            **{f"scaled_{name}": np.linspace(0.1, 0.5, 5) for name in ambiguity.PARAMETER_NAMES},
        }
    )
    basin = ambiguity.classify_basin(near)
    assert basin["basin_classification"] == "multiple_basin"
    cases = pd.DataFrame({"ambiguous_case": [True, True, True, False]})
    assert ambiguity.classify_global_ambiguity(cases)["global_ambiguity_verdict"] == "ESTABLISHED"
    cases.loc[:, "ambiguous_case"] = [True, True, False, False]
    assert ambiguity.classify_global_ambiguity(cases)["global_ambiguity_verdict"] == "PARTIALLY_ESTABLISHED"
    cases.loc[:, "ambiguous_case"] = False
    assert ambiguity.classify_global_ambiguity(cases)["global_ambiguity_verdict"] == "NOT_ESTABLISHED"


def test_alignment_uses_exact_requested_status_vocabulary() -> None:
    assert ambiguity.classify_alignment([0.6, 0.5])["alignment"] == "CONSISTENT"
    assert ambiguity.classify_alignment([0.3, 0.25])["alignment"] == "PARTIALLY_CONSISTENT"
    assert ambiguity.classify_alignment([0.24])["alignment"] == "INCONSISTENT"
    assert ambiguity.classify_alignment([])["alignment"] == "UNAVAILABLE"


def test_constraint_margins_include_each_required_numeric_distance() -> None:
    bounds = load_hard_safety_bounds(ambiguity.baseline.BOUNDS_PATH)
    values = np.asarray(
        [1.2, 0.04, 0.25, -0.35, 0.03, 3.0, 0.02, 0.25, -0.25, 0.02]
    )
    margins = ambiguity.constraint_margins(values, bounds)
    assert set(margins) == {
        "slow_feller_gap", "fast_feller_gap", "ordering_margin",
        "correlation_disk_margin", "minimum_scaled_hard_bound_distance",
    }
    assert margins["slow_feller_gap"] > 0.0
    assert margins["fast_feller_gap"] > 0.0
    assert margins["ordering_margin"] > 0.0


def test_report_preserves_g2_boundary_and_artifact_hash_section() -> None:
    case_inputs = pd.DataFrame(
        [{"case_id": "case_1", "sample_id": "sample_1", "maturity_profile": "2026-07-01",
          **{name: 0.1 for name in ambiguity.PARAMETER_NAMES}}]
    )
    cases = pd.DataFrame(
        [{"case_id": "case_1", "maturity_profile": "2026-07-01", "near_equivalent_count": 5,
          "cluster_count": 2, "basin_classification": "multiple_basin",
          "boundary_associated_count": 1, "ambiguous_case": True}]
    )
    decision = ambiguity.classify_global_ambiguity(cases)
    clusters = pd.DataFrame([{"case_id": "case_1", "cluster_id": 1, "cluster_size": 3,
                              "center_distance_from_truth": .1, "within_cluster_diameter": .2,
                              "nearest_between_cluster_separation": .3}])
    noise = pd.DataFrame([{"case_id": "case_1", "noise_level": 0.0,
                           "usable_solution_count": 20, "near_equivalent_fit_count": 5,
                           "price_rmse_median": 1e-8, "parameter_rmse_median": .1,
                           "material_solution_count": 2, "bound_hit_count": 0,
                           "near_equivalent_cluster_count": 2,
                           "basin_classification": "multiple_basin"}])
    contract = {"protected_snapshot": {"file_count": 1, "aggregate_sha256": "DEF"}}
    solutions = pd.DataFrame(
        [{"case_id": "case_1", "noise_level": 0.0, "optimizer_success": True,
          "price_rmse_normalized": 1e-8, "parameter_rmse_full_range": .1}]
    )
    near = pd.DataFrame(
        [{"case_id": "case_1", "cluster_id": 1, "optimizer_success": True,
          "material_displacement": True, "price_rmse_normalized": 1e-8,
          "parameter_rmse_full_range": .1}]
    )
    report = ambiguity.render_report(contract, case_inputs, solutions, near, cases, clusters, noise, pd.DataFrame(columns=["case_id", "parameter_a", "parameter_b", "absolute_spearman", "spearman_correlation"]), pd.DataFrame(), decision, {"median_absolute_cosine": 0.3, "alignment": "PARTIALLY_CONSISTENT"}, {"contract.json": "abc"})
    assert "G2 remains **NOT_PASSED**" in report
    assert "final representation remains unfrozen" in report
    assert "No final dataset was generated" in report
    assert "ANN or PINN training was performed" in report
    assert "`contract.json` | `abc`" in report
    assert "Exact cases and true ten-vectors" in report
    assert "Six figures" in report
    assert "GLOBAL_AMBIGUITY = PARTIALLY_ESTABLISHED" in report
    assert "Joint historical inference" in report


def test_protected_snapshot_covers_stage_a_and_prior_g2_sources() -> None:
    snapshot = ambiguity._protected_snapshot(
        ambiguity.DEFAULT_OUTPUT_ROOT, ambiguity.DEFAULT_REPORT_PATH
    )
    assert "docs/evidence/G2_CHECKPOINT_MANIFEST.json" in snapshot
    assert "scripts/run_g2_identifiability_analysis.py" in snapshot
    assert "tests/test_g2_identifiability_analysis.py" in snapshot


def test_cluster_and_noise_summaries_retain_required_counts_and_ranges() -> None:
    bounds = load_hard_safety_bounds(ambiguity.baseline.BOUNDS_PATH)
    cases = pd.DataFrame(
        [{"case_id": "case_1", **{name: 0.1 for name in ambiguity.PARAMETER_NAMES}}]
    )
    rows = []
    for index, coordinate in enumerate((0.2, 0.3)):
        rows.append({"case_id": "case_1", "cluster_id": 1, "noise_level": 0.0,
                     "constraint_valid": True, "finite_solution": True, "bound_hit": False,
                     "price_rmse_normalized": 1e-8, "parameter_rmse_full_range": .1,
                     "material_displacement": True,
                     **{f"scaled_{name}": coordinate for name in ambiguity.PARAMETER_NAMES}})
    clustered = pd.DataFrame(rows)
    summary = ambiguity.cluster_summary(clustered, cases, bounds)
    assert summary.loc[0, "cluster_size"] == 2
    assert {"within_cluster_dispersion", "within_cluster_diameter", "price_rmse_median", "parameter_rmse_median"}.issubset(summary.columns)
    noise = ambiguity.noise_summary(clustered)
    assert noise.loc[0, "usable_solution_count"] == 2
    assert noise.loc[0, "near_equivalent_fit_count"] == 2
    assert noise.loc[0, "near_equivalent_cluster_count"] == 2
    assert noise.loc[0, "basin_classification"] == "multiple_basin"
