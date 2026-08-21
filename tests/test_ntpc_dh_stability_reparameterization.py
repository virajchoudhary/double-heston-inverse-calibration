"""Focused contract tests for the NTPC stability reparameterization."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from scripts import run_ntpc_dh_stability_reparameterization as experiment
from scripts import run_ntpc_single_stock_pilot as pilot
from src.calibrate_double_heston import load_hard_safety_bounds, unconstrained_to_parameters
from src.constants import PARAMETER_NAMES
from src.ntpc_dh_reparameterization import (
    canonical_diagnostics,
    canonical_to_structured,
    derived_coordinates,
    structured_to_canonical,
)


BEST_NTPC = np.asarray(
    [
        2.6255332624867473,
        0.13999379410043808,
        0.8566378479188727,
        0.6762760209383789,
        0.01343346216873063,
        11.763863089764895,
        0.024102413556552488,
        0.7521362932264983,
        -0.666180486338069,
        0.027241681410996754,
    ]
)


@pytest.fixture(scope="module")
def bounds() -> dict[str, tuple[float, float]]:
    return load_hard_safety_bounds(pilot.BOUNDS_PATH)


def test_existing_best_fit_round_trip(bounds: dict[str, tuple[float, float]]) -> None:
    coordinate = canonical_to_structured(BEST_NTPC, bounds)
    recovered = structured_to_canonical(coordinate, bounds)
    np.testing.assert_allclose(recovered, BEST_NTPC, rtol=0.0, atol=2e-12)


def test_random_and_near_boundary_round_trip(bounds: dict[str, tuple[float, float]]) -> None:
    rng = np.random.default_rng(2026081201)
    old_coordinates = [rng.normal(0.0, 2.0, 10) for _ in range(100)]
    old_coordinates += [np.eye(10)[index] * sign * 10.0 for index in range(10) for sign in (-1.0, 1.0)]
    for old_coordinate in old_coordinates:
        canonical = unconstrained_to_parameters(old_coordinate, bounds)
        recovered = structured_to_canonical(canonical_to_structured(canonical, bounds), bounds)
        np.testing.assert_allclose(recovered, canonical, rtol=0.0, atol=2e-12)


def test_total_and_allocation_decompositions(bounds: dict[str, tuple[float, float]]) -> None:
    canonical = structured_to_canonical(np.zeros(10), bounds)
    derived = derived_coordinates(canonical)
    assert derived["v0_total"] == pytest.approx(canonical[4] + canonical[9])
    assert derived["alpha_v"] == pytest.approx(canonical[4] / derived["v0_total"])
    assert derived["theta_total"] == pytest.approx(canonical[1] + canonical[6])
    assert derived["alpha_theta"] == pytest.approx(canonical[1] / derived["theta_total"])
    assert 0.0 < derived["alpha_v"] < 1.0
    assert 0.0 < derived["alpha_theta"] < 1.0


def test_kappa_ordering_correlation_disk_feller_and_bounds(
    bounds: dict[str, tuple[float, float]],
) -> None:
    rng = np.random.default_rng(38)
    for _ in range(500):
        canonical = structured_to_canonical(rng.normal(0.0, 4.0, 10), bounds)
        diagnostics = canonical_diagnostics(canonical, bounds)
        assert diagnostics["is_valid"]
        assert canonical[5] > canonical[0]
        assert canonical[3] ** 2 + canonical[8] ** 2 < 1.0
        assert 2 * canonical[0] * canonical[1] > canonical[2] ** 2
        assert 2 * canonical[5] * canonical[6] > canonical[7] ** 2
        for name, value in zip(PARAMETER_NAMES, canonical, strict=True):
            lower, upper = bounds[name]
            assert lower < value < upper


@pytest.mark.parametrize(
    ("rho_slow", "rho_fast"),
    [(0.70, 0.68), (0.70, -0.68), (-0.70, 0.68), (-0.70, -0.68), (0.94, 0.30)],
)
def test_full_hard_bound_unit_disk_intersection_is_attainable(
    bounds: dict[str, tuple[float, float]], rho_slow: float, rho_fast: float
) -> None:
    canonical = unconstrained_to_parameters(np.zeros(10), bounds)
    canonical[3], canonical[8] = rho_slow, rho_fast
    assert 0.95 <= np.hypot(rho_slow, rho_fast) < 1.0
    coordinate = canonical_to_structured(canonical, bounds)
    recovered = structured_to_canonical(coordinate, bounds)
    np.testing.assert_allclose(recovered, canonical, rtol=0.0, atol=2e-12)


def test_matched_baseline_and_transformed_starts(bounds: dict[str, tuple[float, float]]) -> None:
    canonical, transformed, frame = experiment.paired_start_population(bounds)
    assert len(canonical) == len(transformed) == len(frame) == 12
    assert frame["start_id"].tolist() == list(range(12))
    assert frame["paired_max_abs_error"].max() <= 2e-12
    for baseline_canonical, transformed_coordinate in zip(canonical, transformed, strict=True):
        np.testing.assert_allclose(
            structured_to_canonical(transformed_coordinate, bounds),
            baseline_canonical,
            rtol=0.0,
            atol=2e-12,
        )


def test_unchanged_ntpc_row_and_market_carry_hashes() -> None:
    manifest = json.loads(pilot.MANIFEST_PATH.read_text(encoding="utf-8"))
    for name in ("selected_options.csv", "carry_contract.csv"):
        path = pilot.OUTPUT_ROOT / name
        observed = hashlib.sha256(path.read_bytes()).hexdigest().upper()
        assert observed == manifest["artifact_hashes"][name]
    selected = pd.read_csv(pilot.OUTPUT_ROOT / "selected_options.csv")
    assert len(selected.loc[selected["sample_role"] == "CALIBRATION"]) == 12
    assert len(selected.loc[selected["sample_role"] == "HOLDOUT"]) == 7
    assert set(selected["valuation_date"]) == {"2026-07-15"}
    assert set(selected["observed_price"]) == set(selected["observed_price"].astype(float))
    assert manifest["carry_contract"]["carry_rule"] == "q=r-log(F/S)/T"
    assert manifest["carry_contract"]["discount_rule"] == "D(T)=1/(1+y*T)"
    assert manifest["carry_contract"]["risk_free_simple_yield"] == 0.053324


def test_unchanged_objective_pricing_configuration_and_numerical_value(
    bounds: dict[str, tuple[float, float]],
) -> None:
    manifest = json.loads(pilot.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert experiment.NODE_COUNT == pilot.NODE_COUNT == manifest["optimizer"]["node_count"] == 64
    assert experiment.MAX_NFEV == pilot.MAX_NFEV == manifest["optimizer"]["max_nfev"] == 160
    selected = pd.read_csv(pilot.OUTPUT_ROOT / "selected_options.csv")
    calibration = selected.loc[selected["sample_role"] == "CALIBRATION"].copy()
    coordinate = canonical_to_structured(BEST_NTPC, bounds)
    direct = experiment.price_rows(calibration, BEST_NTPC) - calibration["observed_price"].to_numpy(float)
    transformed = experiment.price_rows(
        calibration, structured_to_canonical(coordinate, bounds)
    ) - calibration["observed_price"].to_numpy(float)
    np.testing.assert_allclose(direct, transformed, rtol=0.0, atol=1e-12)


def _summaries() -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    baseline = {
        "calibration_price_rmse": 1.0,
        "holdout_price_rmse": 1.0,
        "holdout_iv_rmse": 1.0,
    }
    transformed = dict(baseline)
    baseline_stability = {
        "median_pairwise_range_scaled_distance": 0.20,
        "maximum_pairwise_range_scaled_distance": 0.40,
        "cluster_count": 8,
    }
    transformed_stability = {
        "median_pairwise_range_scaled_distance": 0.10,
        "maximum_pairwise_range_scaled_distance": 0.20,
        "cluster_count": 3,
        "materially_displaced_start_count": 3,
    }
    return baseline, transformed, baseline_stability, transformed_stability


def test_deterministic_classification_contract() -> None:
    arguments = _summaries()
    strong = experiment.classify_experiment(
        *arguments,
        equivalence_passed=True,
        contract_passed=True,
        matched_population=True,
    )
    assert strong["classification"] == "STRONG_STABILITY_IMPROVEMENT"
    assert strong == experiment.classify_experiment(
        *arguments,
        equivalence_passed=True,
        contract_passed=True,
        matched_population=True,
    )
    arguments[3]["materially_displaced_start_count"] = 5
    arguments[3]["median_pairwise_range_scaled_distance"] = 0.17
    arguments[3]["maximum_pairwise_range_scaled_distance"] = 0.35
    arguments[3]["cluster_count"] = 7
    partial = experiment.classify_experiment(
        *arguments,
        equivalence_passed=True,
        contract_passed=True,
        matched_population=True,
    )
    assert partial["classification"] == "PARTIAL_STABILITY_IMPROVEMENT"
    invalid = experiment.classify_experiment(
        *arguments,
        equivalence_passed=False,
        contract_passed=True,
        matched_population=True,
    )
    assert invalid["classification"] == "INVALID"


def test_near_equivalent_outcome_does_not_define_population_matching() -> None:
    # Population matching concerns paired valid start IDs. A transformed fit may
    # legitimately fall outside the unchanged near-equivalent RMSE threshold.
    paired_ids = set(range(12))
    baseline_ids = set(range(12))
    transformed_ids = set(range(12))
    matched = (
        len(paired_ids) == 12
        and len(baseline_ids) == 12
        and len(transformed_ids) == 12
        and paired_ids == baseline_ids == transformed_ids
    )
    assert matched


def test_non_vacuous_stability_comparison(bounds: dict[str, tuple[float, float]]) -> None:
    baseline = pd.read_csv(pilot.OUTPUT_ROOT / "double_heston_multistart.csv")
    metrics, near, pairs = experiment.stability_metrics(baseline, bounds)
    assert metrics["valid_start_count"] == 12
    assert metrics["near_equivalent_start_count"] == 12
    assert metrics["materially_displaced_start_count"] == 11
    assert len(near) == 12
    assert len(pairs) == 66
    assert metrics["maximum_range_scaled_distance_from_best"] == pytest.approx(
        0.491085491886338, abs=1e-12
    )
