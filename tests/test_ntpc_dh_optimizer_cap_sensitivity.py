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


def _dispersion_case(
    *, median_reduction: float, maximum_reduction: float, clusters: int
) -> tuple[dict[str, float], dict[str, float]]:
    baseline = {"materially_displaced_start_count": 12, "cluster_count": 10,
                "median_pairwise_range_scaled_distance": 0.40,
                "maximum_pairwise_range_scaled_distance": 0.80}
    result = {
        "materially_displaced_start_count": 12,
        "cluster_count": clusters,
        "median_pairwise_range_scaled_distance": 0.40 * (1.0 - median_reduction),
        "maximum_pairwise_range_scaled_distance": 0.80 * (1.0 - maximum_reduction),
    }
    return baseline, result


@pytest.mark.parametrize(
    ("median_reduction", "maximum_reduction", "clusters", "expected"),
    [
        (0.25, 0.25, 9, "STRONG_DISPERSION_COLLAPSE"),
        (0.25 - 1e-8, 0.25, 9, "PARTIAL_DISPERSION_IMPROVEMENT"),
        (0.25, 0.25 - 1e-8, 9, "PARTIAL_DISPERSION_IMPROVEMENT"),
        (0.25, 0.25, 10, "PARTIAL_DISPERSION_IMPROVEMENT"),
        (0.10, 0.10, 10, "PARTIAL_DISPERSION_IMPROVEMENT"),
        (0.10 - 1e-8, 0.10, 10, "DISPERSION_PERSISTS"),
        (0.10, 0.10 - 1e-8, 10, "DISPERSION_PERSISTS"),
        (0.20, 0.05, 9, "DISPERSION_PERSISTS"),
        (0.05, 0.20, 9, "DISPERSION_PERSISTS"),
        (0.25, 0.25, 11, "DISPERSION_PERSISTS"),
    ],
)
def test_frozen_dispersion_threshold_boundaries(
    median_reduction: float, maximum_reduction: float, clusters: int, expected: str
) -> None:
    baseline, result = _dispersion_case(
        median_reduction=median_reduction,
        maximum_reduction=maximum_reduction,
        clusters=clusters,
    )
    assert experiment.classify_dispersion(baseline, result) == expected


def test_displaced_count_is_reported_but_not_a_dispersion_gate() -> None:
    baseline, result = _dispersion_case(median_reduction=0.25, maximum_reduction=0.25, clusters=9)
    result["materially_displaced_start_count"] = baseline["materially_displaced_start_count"]
    assert experiment.classify_dispersion(baseline, result) == "STRONG_DISPERSION_COLLAPSE"


def test_predeclared_final_classification() -> None:
    assert experiment.classify_cap(0.25) == "CAP_MATERIALLY_REDUCED"
    assert experiment.classify_cap(0.50) == "CAP_PARTIALLY_REDUCED"
    assert experiment.classify_cap(0.75) == "CAP_NOT_RESOLVED"
    assert experiment.classify_final([0.25, 0.25], ["STRONG_DISPERSION_COLLAPSE"] * 2, True) == "NUMERICAL_CAP_LIMITATION_SUPPORTED"
    assert experiment.classify_final([0.50, 0.50], ["DISPERSION_PERSISTS"] * 2, True) == "PERSISTENT_PARAMETER_AMBIGUITY"
    assert experiment.classify_final([0.75, 0.75], ["DISPERSION_PERSISTS"] * 2, True) == "OPTIMIZER_CAP_UNRESOLVED"
    assert experiment.classify_final([0.25, 0.25], ["STRONG_DISPERSION_COLLAPSE"] * 2, False) == "INVALID"


def _minimal_cell(chart: str, budget: int, start_ids: list[int] | None = None) -> pd.DataFrame:
    ids = list(range(12)) if start_ids is None else start_ids
    return pd.DataFrame({
        "chart": chart,
        "max_nfev": budget,
        "start_id": ids,
        "valid": [True] * len(ids),
    })


def test_historical_transformed_reference_is_explicitly_non_identical() -> None:
    cells = {
        "canonical_160": {"pricing": {}, "stability": {}},
        "transformed_160": {"pricing": {}, "stability": {}},
    }
    for chart in experiment.CHARTS:
        combined = experiment.REVIEWED_160[chart]
        cells[f"{chart}_160"]["pricing"] = {
            "best_calibration_price_rmse": combined["calibration_price_rmse"],
            "calibration_iv_rmse": combined["calibration_iv_rmse"],
            "best_holdout_price_rmse": combined["holdout_price_rmse"],
            "holdout_iv_rmse": combined["holdout_iv_rmse"],
        }
        cells[f"{chart}_160"]["stability"] = {
            key: value for key, value in combined.items() if key not in {
                "calibration_price_rmse", "calibration_iv_rmse",
                "holdout_price_rmse", "holdout_iv_rmse",
            }
        }
    cells["transformed_160"]["pricing"]["best_holdout_price_rmse"] += 1e-4
    comparison = experiment.historical_160_comparison(cells)
    assert comparison["canonical"]["exact_reproduction_required"] is True
    assert comparison["canonical"]["passed"] is True
    assert comparison["transformed"]["provenance"] == "HISTORICAL_SERIALIZED_DERIVED_INPUTS"
    assert comparison["transformed"]["exact_reproduction_required"] is False
    assert comparison["transformed"]["passed"] is False
    assert comparison["passed"] is True

    comparison["canonical"]["passed"] = False
    comparison["passed"] = False
    with pytest.raises(RuntimeError, match="canonical historical"):
        experiment.require_valid_160_controls(comparison, {"passed": True})


def test_experiment_validation_fails_closed_on_required_contract_breaks() -> None:
    frames = {
        f"{chart}_{budget}": _minimal_cell(chart, budget)
        for chart in experiment.CHARTS for budget in experiment.BUDGETS
    }
    protected = {"source": "A" * 64}
    valid = experiment.validate_experiment_cells(frames, protected, protected)
    assert valid["passed"] is True

    canonical_bad = {key: value.copy() for key, value in frames.items()}
    canonical_bad["canonical_160"] = _minimal_cell("canonical", 160, list(range(11)))
    assert experiment.validate_experiment_cells(canonical_bad, protected, protected)["passed"] is False

    transformed_bad = {key: value.copy() for key, value in frames.items()}
    transformed_bad["transformed_160"] = _minimal_cell("transformed", 160, list(range(11)))
    transformed_validation = experiment.validate_experiment_cells(transformed_bad, protected, protected)
    assert transformed_validation["passed"] is False
    with pytest.raises(RuntimeError, match="corrected 160 control"):
        experiment.require_valid_160_controls(
            {"passed": True}, experiment.validate_160_control_frames(transformed_bad, protected, protected)
        )

    starts_bad = {key: value.copy() for key, value in frames.items()}
    starts_bad["canonical_320"] = _minimal_cell("canonical", 320, list(range(1, 13)))
    assert experiment.validate_experiment_cells(starts_bad, protected, protected)["passed"] is False

    source_bad = experiment.validate_experiment_cells(frames, protected, {"source": "B" * 64})
    assert source_bad["passed"] is False


def test_non_budget_optimizer_settings_must_match() -> None:
    low = experiment.optimizer_contract(160)
    high = experiment.optimizer_contract(320)
    assert experiment.optimizer_contracts_differ_only_by_budget(low, high) is True
    high["ftol"] = 1e-8
    assert experiment.optimizer_contracts_differ_only_by_budget(low, high) is False


def test_persisted_start_artifact_validation_rejects_changed_320(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(experiment, "OUTPUT_ROOT", tmp_path)
    for chart in experiment.CHARTS:
        for budget in experiment.BUDGETS:
            (tmp_path / f"{chart}_{budget}_starts.csv").write_text(
                f"chart,max_nfev,start_id,valid\n{chart},{budget},0,True\n",
                encoding="utf-8",
            )
    expected = {
        f"{chart}_{budget}": experiment.sha256(tmp_path / f"{chart}_{budget}_starts.csv")
        for chart in experiment.CHARTS for budget in experiment.BUDGETS
    }
    missing_seal = experiment.validate_persisted_start_artifacts({})
    assert missing_seal["sealed_reference_used"] is False
    assert missing_seal["passed"] is False
    assert experiment.validate_persisted_start_artifacts(expected)["passed"] is True

    (tmp_path / "transformed_320_starts.csv").write_text(
        "chart,max_nfev,start_id,valid\ntransformed,320,0,False\n",
        encoding="utf-8",
    )
    validation = experiment.validate_persisted_start_artifacts(expected)
    assert validation["passed"] is False
    assert validation["checks"]["transformed_320"]["passed"] is False


def test_render_only_fails_before_publish_when_start_seal_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(experiment, "verify_frozen_contract", lambda: {})
    monkeypatch.setattr(experiment, "protected_evidence_hashes", lambda: {"source": "A" * 64})
    monkeypatch.setattr(experiment, "_require_prior_protected_seal", lambda current: None)
    monkeypatch.setattr(experiment, "_expected_start_hashes_from_manifest", lambda: {})
    monkeypatch.setattr(
        experiment,
        "validate_persisted_start_artifacts",
        lambda expected: {"passed": False, "checks": {}, "sealed_reference_used": False},
    )
    with pytest.raises(RuntimeError, match="persisted optimizer start artifact"):
        experiment.render_existing_outputs()


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
    assert manifest["historical_160_comparison"]["canonical"]["passed"] is True
    assert manifest["historical_160_comparison"]["transformed"]["exact_reproduction_required"] is False
    assert manifest["corrected_160_controls"]["charts"]["transformed"]["passed"] is True
    assert manifest["experiment_validation"]["passed"] is True
    assert manifest["valid"] is True
    assert set(manifest["corrected_control_artifacts"]) == set(experiment.CHARTS)
    assert hashlib.sha256(experiment.REPORT_PATH.read_bytes()).hexdigest().upper() == manifest["report_sha256"]
    for relative, expected in manifest["generated_artifact_hashes"].items():
        path = experiment.REPOSITORY_ROOT / relative
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest().upper() == expected
    assert experiment.verify_manifest_artifacts() == []
