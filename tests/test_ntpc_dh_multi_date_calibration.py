"""Focused contracts for the real NTPC three-date Double Heston calibration."""

from __future__ import annotations

import numpy as np
import pytest

from scripts import run_ntpc_dh_multi_date_calibration as multi
from src.calibrate_double_heston import load_hard_safety_bounds
from src.constraints import validate_parameters


def test_three_date_selection_is_deterministic_and_has_no_role_leakage() -> None:
    first = multi.build_three_date_panel()
    second = multi.build_three_date_panel()
    assert first.equals(second)
    assert tuple(sorted(first["valuation_date"].unique())) == multi.VALUATION_DATES
    assert set(first["sample_role"]) == {"CALIBRATION", "HOLDOUT"}
    assert not first.duplicated(["valuation_date", "expiry_date", "option_type", "strike"]).any()
    assert set(first.loc[first["sample_role"] == "CALIBRATION", "target_log_moneyness"]) <= {-0.05, 0.0, 0.05}
    assert set(first.loc[first["sample_role"] == "HOLDOUT", "target_log_moneyness"]) <= {-0.10, 0.10}
    np.testing.assert_array_equal(first["T"].to_numpy(), first["DTE"].to_numpy(float) / 365.0)
    counts = first.groupby(["valuation_date", "sample_role"]).size().to_dict()
    assert counts == {
        ("2026-07-01", "CALIBRATION"): 8, ("2026-07-01", "HOLDOUT"): 3,
        ("2026-07-15", "CALIBRATION"): 12, ("2026-07-15", "HOLDOUT"): 7,
        ("2026-07-22", "CALIBRATION"): 12, ("2026-07-22", "HOLDOUT"): 6,
    }
    assert all(first["rate_observation_date"] <= first["valuation_date"])


def test_rate_contract_is_hash_sealed_and_has_no_future_leakage() -> None:
    contract = multi.validated_rate_contract()
    assert tuple(contract) == multi.VALUATION_DATES
    assert contract["2026-07-01"]["source_identifier"] == "RBI Press Release 2026-2027/584"
    assert contract["2026-07-15"]["source_identifier"] == "RBI Press Release 2026-2027/672"
    assert contract["2026-07-22"]["source_identifier"] == "RBI Press Release 2026-2027/672"
    assert contract["2026-07-22"]["observed"] == "2026-07-15"
    for valuation_date, source in contract.items():
        assert source["observed"] <= valuation_date
        assert multi.sha256(multi.REPOSITORY_ROOT / source["preserved_path"]) == source["sha256"]


def test_rate_contract_fails_closed_on_tampered_provenance(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    source = tmp_path / multi.RATE_SOURCES["2026-07-01"]["html"]
    source.write_text("not RBI evidence", encoding="utf-8")
    monkeypatch.setattr(multi, "RATE_PROVENANCE_ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="hash mismatch"):
        multi.validated_rate_contract()


def test_support_inventory_records_missing_targets_without_fabrication() -> None:
    panel = multi.build_three_date_panel()
    inventory = multi.support_inventory(panel)
    assert len(inventory) == 3 * 2 * 2 * 5
    assert inventory["selected"].sum() == len(panel)
    missing = inventory.loc[~inventory["selected"]]
    assert not missing.empty
    assert ((missing["valuation_date"] == "2026-07-01") & (missing["expiry_position"] == "middle")).any()


def test_shared_structure_and_date_specific_states_make_valid_canonical_vectors() -> None:
    bounds = load_hard_safety_bounds(multi.BOUNDS_PATH)
    mapped = multi.map_joint_coordinate(np.zeros(14), bounds)
    assert tuple(mapped["states"]) == multi.VALUATION_DATES
    for valuation_date in multi.VALUATION_DATES:
        vector = multi.canonical_vector(mapped, valuation_date)
        assert vector.shape == (10,)
        assert validate_parameters(vector)["is_valid"] is True
    vectors = [multi.canonical_vector(mapped, value) for value in multi.VALUATION_DATES]
    for left, right in zip(vectors[:-1], vectors[1:], strict=True):
        np.testing.assert_array_equal(left[[0, 1, 2, 3, 5, 6, 7, 8]], right[[0, 1, 2, 3, 5, 6, 7, 8]])


def test_date_balanced_residual_gives_each_date_equal_weight() -> None:
    errors = {"2026-07-01": np.ones(2), "2026-07-15": np.ones(8) * 2, "2026-07-22": np.ones(18) * 3}
    residual = multi.date_balanced_residual(errors)
    chunks = np.split(residual, [2, 10])
    assert [float(np.sum(chunk**2)) for chunk in chunks] == pytest.approx([1.0, 4.0, 9.0])
    assert multi.reported_date_balanced_objective(errors) == pytest.approx(np.sqrt((1 + 4 + 9) / 3))


def test_joint_start_population_is_deterministic_and_paired() -> None:
    first, hashes_first = multi.joint_start_population()
    second, hashes_second = multi.joint_start_population()
    assert len(first) == len(second) == 12
    assert hashes_first.equals(hashes_second)
    assert hashes_first["start_id"].tolist() == list(range(12))
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)


def test_single_date_shared_baseline_is_recomputed_from_reviewed_vectors() -> None:
    bounds = load_hard_safety_bounds(multi.BOUNDS_PATH)
    baseline = multi.single_date_shared_baseline(bounds)
    assert baseline["near_equivalent_count"] == 12
    assert baseline["cluster_count"] >= 1
    assert baseline["median_pairwise_distance"] > 0
    assert baseline["boundary_hit_rate"] > 0


def test_predeclared_multi_date_classification() -> None:
    baseline = {"median": 0.40, "maximum": 0.80, "clusters": 10, "displaced": 10, "holdout": 1.0}
    strong = {"median": 0.30, "maximum": 0.60, "clusters": 6, "displaced": 5, "holdout": 1.05}
    partial = {"median": 0.35, "maximum": 0.79, "clusters": 10, "displaced": 10, "holdout": 1.04}
    insufficient = {"median": 0.39, "maximum": 0.79, "clusters": 11, "displaced": 10, "holdout": 1.01}
    assert multi.classify_multi_date(baseline, strong) == "STRONG_MULTI_DATE_STABILITY_IMPROVEMENT"
    assert multi.classify_multi_date(baseline, partial) == "PARTIAL_MULTI_DATE_STABILITY_IMPROVEMENT"
    assert multi.classify_multi_date(baseline, insufficient) == "MULTI_DATE_INSUFFICIENT"


def test_persisted_replay_hashes_verify() -> None:
    assert multi.verify_manifest_artifacts() == []
    manifest = __import__("json").loads(multi.MANIFEST_PATH.read_text(encoding="utf-8"))
    rate_sources = [path for path in manifest["source_hashes"] if "/rate_provenance/" in path]
    assert len(rate_sources) == 2


def test_persisted_timescale_comparison_is_explicit() -> None:
    summary = __import__("json").loads((multi.OUTPUT_ROOT / "summary.json").read_text(encoding="utf-8"))
    comparison = summary["timescale_comparison"]
    assert comparison["conclusion"] == "MIXED_TIMESCALE_STABILITY_INSUFFICIENT"
    assert comparison["single_date_best"]["slow_half_life_days"] == pytest.approx(96.360889621621)
    assert comparison["multi_date_best"]["slow_half_life_days"] == pytest.approx(
        np.log(2) / comparison["multi_date_best"]["kappa_slow"] * 365
    )
    assert comparison["multi_date_near_equivalent"]["fast"]["half_life_maximum_days"] > 700
