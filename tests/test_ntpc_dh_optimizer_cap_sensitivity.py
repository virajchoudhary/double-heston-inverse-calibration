"""Focused contracts for the paired NTPC optimizer-cap sensitivity experiment."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from scripts import run_ntpc_dh_optimizer_cap_sensitivity as experiment
from scripts import run_ntpc_dh_stability_reparameterization as geometry
from scripts import run_ntpc_single_stock_pilot as pilot
from src.calibrate_double_heston import load_hard_safety_bounds


@pytest.fixture(scope="module")
def bounds() -> dict[str, tuple[float, float]]:
    return load_hard_safety_bounds(pilot.BOUNDS_PATH)


def test_frozen_contract_and_exact_row_hashes() -> None:
    contract = experiment.verify_frozen_contract()
    manifest = json.loads(pilot.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert contract["valuation_date"] == "2026-07-15"
    assert contract["calibration_rows"] == 12
    assert contract["holdout_rows"] == 7
    assert contract["calibration_row_sha256"] == experiment.CALIBRATION_ROW_SHA256
    assert contract["holdout_row_sha256"] == experiment.HOLDOUT_ROW_SHA256
    assert contract["carry_rule"] == "q=r-log(F/S)/T"
    assert contract["discount_rule"] == "D(T)=1/(1+y*T)"
    assert contract["risk_free_simple_yield"] == 0.053324
    assert contract["primary_price_field"] == "ClsPric"
    assert contract["activity_screen"] == manifest["selection"]["activity_rule"]


def test_frozen_rows_reconstruct_pre_serialization_pricing_inputs() -> None:
    loaded = experiment.load_frozen_selected_options()
    serialized = pd.read_csv(pilot.OUTPUT_ROOT / "selected_options.csv")
    np.testing.assert_array_equal(loaded["T"].to_numpy(), loaded["DTE"].to_numpy(float) / 365.0)
    assert not np.array_equal(loaded["T"].to_numpy(), serialized["T"].to_numpy())
    assert np.isfinite(
        loaded[["discount_factor", "continuous_rate", "futures_implied_carry"]].to_numpy(float)
    ).all()


def test_identical_paired_start_population(bounds: dict[str, tuple[float, float]]) -> None:
    canonical_z, structured_z, frame = experiment.frozen_starts(bounds)
    reviewed_canonical, reviewed_structured, reviewed = geometry.paired_start_population(bounds)
    assert frame["start_id"].tolist() == list(range(12))
    assert frame["canonical_start_sha256"].tolist() == reviewed["canonical_start_sha256"].tolist()
    for actual, expected in zip(canonical_z, reviewed_canonical, strict=True):
        np.testing.assert_allclose(
            experiment.canonical_from_coordinate(actual, bounds), expected, rtol=0.0, atol=2e-12
        )
    for actual, expected in zip(structured_z, reviewed_structured, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_optimizer_configuration_differs_only_by_budget() -> None:
    low = experiment.optimizer_contract(160)
    high = experiment.optimizer_contract(320)
    assert low | {"max_nfev": 320} == high
    assert low == {
        "method": "trf",
        "max_nfev": 160,
        "ftol": 1e-9,
        "xtol": 1e-9,
        "gtol": 1e-9,
        "diff_step": 2e-5,
    }
    assert experiment.NODE_COUNT == pilot.NODE_COUNT == 64
    assert experiment.OBJECTIVE == "unweighted observed-price residual vector"
    assert experiment.REGULARIZATION == "NONE"


@pytest.mark.parametrize(
    ("nfev", "budget", "expected"), [(159, 160, False), (160, 160, True), (319, 320, False), (320, 320, True)]
)
def test_exact_cap_detection(nfev: int, budget: int, expected: bool) -> None:
    assert experiment.reached_cap(nfev, budget) is expected


def test_predeclared_dispersion_and_final_classification() -> None:
    baseline = {"materially_displaced_start_count": 12, "cluster_count": 10,
                "median_pairwise_range_scaled_distance": 0.40,
                "maximum_pairwise_range_scaled_distance": 0.80}
    strong = {"materially_displaced_start_count": 6, "cluster_count": 6,
              "median_pairwise_range_scaled_distance": 0.30,
              "maximum_pairwise_range_scaled_distance": 0.60}
    partial = {"materially_displaced_start_count": 8, "cluster_count": 10,
               "median_pairwise_range_scaled_distance": 0.35,
               "maximum_pairwise_range_scaled_distance": 0.78}
    persistent = dict(baseline)
    assert experiment.classify_dispersion(baseline, strong) == "STRONG_DISPERSION_COLLAPSE"
    assert experiment.classify_dispersion(baseline, partial) == "PARTIAL_DISPERSION_IMPROVEMENT"
    assert experiment.classify_dispersion(baseline, persistent) == "DISPERSION_PERSISTS"
    assert experiment.classify_cap(0.25) == "CAP_MATERIALLY_REDUCED"
    assert experiment.classify_cap(0.50) == "CAP_PARTIALLY_REDUCED"
    assert experiment.classify_cap(0.75) == "CAP_NOT_RESOLVED"
    assert experiment.classify_final([0.25, 0.25], ["STRONG_DISPERSION_COLLAPSE"] * 2, True) == "NUMERICAL_CAP_LIMITATION_SUPPORTED"
    assert experiment.classify_final([0.50, 0.50], ["DISPERSION_PERSISTS"] * 2, True) == "PERSISTENT_PARAMETER_AMBIGUITY"
    assert experiment.classify_final([0.75, 0.75], ["DISPERSION_PERSISTS"] * 2, True) == "OPTIMIZER_CAP_UNRESOLVED"
    assert experiment.classify_final([0.25, 0.25], ["STRONG_DISPERSION_COLLAPSE"] * 2, False) == "INVALID"


def test_320_gate_requires_reviewed_canonical_160_cell() -> None:
    passing = {
        "canonical": {"passed": True, "absolute_differences": {}},
        "transformed": {"passed": False, "absolute_differences": {"holdout_price_rmse": 1e-4}},
    } | {"passed": True}
    experiment.require_reproduced_160_baseline(passing)
    failing = {
        "canonical": {"passed": False, "absolute_differences": {"holdout_price_rmse": 1e-4}},
        "transformed": {"passed": True, "absolute_differences": {}},
        "passed": False,
    }
    with pytest.raises(RuntimeError, match="before any 320 fit"):
        experiment.require_reproduced_160_baseline(failing)


def test_deterministic_stability_metrics(bounds: dict[str, tuple[float, float]]) -> None:
    starts = pd.read_csv(pilot.OUTPUT_ROOT / "double_heston_multistart.csv")
    first, _, _ = experiment.stability_metrics(starts, bounds, 160)
    second, _, _ = experiment.stability_metrics(starts, bounds, 160)
    assert first == second
    assert first["cap_count"] == 10
    assert first["cap_rate"] == pytest.approx(10 / 12)
    assert first["materially_displaced_start_count"] == 11
    assert first["cluster_count"] == 7


def test_no_data_leakage_and_no_hidden_regularization() -> None:
    source = experiment.__file__.read_text(encoding="utf-8") if hasattr(experiment.__file__, "read_text") else open(experiment.__file__, encoding="utf-8").read()
    assert "sample_role" in source
    assert experiment.REGULARIZATION == "NONE"
    assert experiment.OBJECTIVE == "unweighted observed-price residual vector"
    assert "penalty_residual" not in source
    assert "prior_residual" not in source
    assert "download" not in source.lower()
    assert "requests." not in source.lower()


def test_reviewed_protected_evidence_hashes_are_stable() -> None:
    before = experiment.protected_evidence_hashes()
    after = experiment.protected_evidence_hashes()
    assert before == after
    assert all(len(value) == 64 for value in before.values())


def test_preserved_render_only_evidence_matches_manifest() -> None:
    manifest = json.loads(experiment.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["classification"] in {
        "NUMERICAL_CAP_LIMITATION_SUPPORTED", "PARTIAL_NUMERICAL_EFFECT",
        "PERSISTENT_PARAMETER_AMBIGUITY", "OPTIMIZER_CAP_UNRESOLVED",
    }
    assert manifest["changed_variable_only"] == "max_nfev"
    assert manifest["baseline_reproduction"]["canonical"]["passed"] is True
    assert hashlib.sha256(experiment.REPORT_PATH.read_bytes()).hexdigest().upper() == manifest["report_sha256"]
    for relative, expected in manifest["generated_artifact_hashes"].items():
        path = experiment.REPOSITORY_ROOT / relative
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest().upper() == expected
