"""Tests for the independent adaptive-quadrature benchmark seam."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.constants import PARAMETER_NAMES
from src.constraints import validate_parameters
from src.double_heston import price_double_heston_option
import src.run_independent_pricing_benchmark as benchmark
import src.audit_parameter_bounds as audit
from src.double_heston_reference import (
    REFERENCE_EPSABS,
    REFERENCE_EPSREL,
    REFERENCE_LIMIT,
    reference_double_heston_characteristic_function,
    reference_double_heston_option,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "double_heston_benchmark_cases.json"


@pytest.fixture(scope="module")
def fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_reference_cf_at_zero_is_one(fixture: dict) -> None:
    case = fixture["cases"][0]
    actual = reference_double_heston_characteristic_function(
        0.0, case["spot"], case["maturity"], case["rate"], case["dividend_yield"], case["parameters"]
    )
    assert actual == pytest.approx(1.0 + 0.0j, abs=1e-13)


def test_controlled_reference_price_has_full_reliable_diagnostics(fixture: dict) -> None:
    case = fixture["cases"][0]
    price, diagnostic = reference_double_heston_option(
        case["spot"], case["strike"], case["maturity"], case["rate"], case["dividend_yield"],
        case["option_type"], case["parameters"],
    )
    assert np.isfinite(price)
    assert diagnostic["reliable"] is True
    assert diagnostic["p1"]["absolute_error_estimate"] <= diagnostic["p1"]["tolerance"]
    assert diagnostic["p2"]["absolute_error_estimate"] <= diagnostic["p2"]["tolerance"]
    assert "evaluation_count" in diagnostic["p1"] and "subdivisions" in diagnostic["p2"]
    assert diagnostic["epsabs"] == REFERENCE_EPSABS
    assert diagnostic["epsrel"] == REFERENCE_EPSREL
    assert diagnostic["limit"] == REFERENCE_LIMIT


def test_reference_no_arbitrage_and_parity(fixture: dict) -> None:
    call_case = next(case for case in fixture["cases"] if case["case_id"] == "B_2_call")
    call, _ = reference_double_heston_option(
        call_case["spot"], call_case["strike"], call_case["maturity"], call_case["rate"], call_case["dividend_yield"],
        "call", call_case["parameters"],
    )
    put, _ = reference_double_heston_option(
        call_case["spot"], call_case["strike"], call_case["maturity"], call_case["rate"], call_case["dividend_yield"],
        "put", call_case["parameters"],
    )
    discounted_spot = call_case["spot"] * np.exp(-call_case["dividend_yield"] * call_case["maturity"])
    discounted_strike = call_case["strike"] * np.exp(-call_case["rate"] * call_case["maturity"])
    assert max(discounted_spot - discounted_strike, 0.0) <= call <= discounted_spot
    assert max(discounted_strike - discounted_spot, 0.0) <= put <= discounted_strike
    assert call - put == pytest.approx(discounted_spot - discounted_strike, abs=1e-10)


@pytest.mark.parametrize("case_id", ["A_1_call", "D_2_put"])
def test_production_reference_closeness_uses_frozen_combined_tolerance(fixture: dict, case_id: str) -> None:
    case = next(item for item in fixture["cases"] if item["case_id"] == case_id)
    reference, diagnostic = reference_double_heston_option(
        case["spot"], case["strike"], case["maturity"], case["rate"], case["dividend_yield"],
        case["option_type"], case["parameters"],
    )
    assert diagnostic["reliable"]
    tolerance = fixture["immutable_tolerances"]["absolute"] + fixture["immutable_tolerances"]["relative"] * abs(reference)
    for nodes in (64, 96):
        production = price_double_heston_option(
            case["spot"], case["strike"], case["maturity"], case["rate"], case["dividend_yield"],
            case["option_type"], case["parameters"], node_count=nodes,
        )
        assert abs(production - reference) <= tolerance


def test_fixture_is_valid_and_has_declared_coverage(fixture: dict) -> None:
    assert fixture["canonical_parameter_order"] == PARAMETER_NAMES
    assert fixture["immutable_tolerances"] == {
        "absolute": 2e-5, "relative": 2e-6, "no_arbitrage": 1e-8,
        "parity": 1e-10, "reference_epsabs": REFERENCE_EPSABS,
        "reference_epsrel": REFERENCE_EPSREL, "reference_limit": REFERENCE_LIMIT,
    }
    assert len(fixture["cases"]) == 36
    required_fields = {
        "case_id", "spot", "strike", "maturity", "rate", "dividend_yield",
        "option_type", "parameters", "case_category", "generation_metadata",
    }
    assert all(required_fields <= set(case) for case in fixture["cases"])
    assert all(set(case["generation_metadata"]) >= {"pair_id", "creation_method", "design_dimensions"} for case in fixture["cases"])
    assert all(len(case["parameters"]) == 10 and np.isfinite(case["parameters"]).all() for case in fixture["cases"])
    assert {case["option_type"] for case in fixture["cases"]} == {"call", "put"}
    assert {"short", "medium", "long"} == {
        "short" if case["maturity"] <= .25 else "medium" if case["maturity"] <= 1 else "long"
        for case in fixture["cases"]
    }
    assert all(validate_parameters(case["parameters"])["is_valid"] for case in fixture["cases"])
    categories = {case["case_category"] for case in fixture["cases"]}
    assert any("low_total_variance" in category for category in categories)
    assert any("high_total_variance" in category for category in categories)
    assert any("weak_negative_skew" in category for category in categories)
    assert any("strong_negative_skew" in category for category in categories)
    assert "moderately_close_feller_boundary" in categories
    assert "safely_interior_low_total_variance" in categories
    assert len({case["spot"] for case in fixture["cases"]}) > 1
    assert len({case["rate"] for case in fixture["cases"]}) > 1
    assert len({case["dividend_yield"] for case in fixture["cases"]}) > 1
    for option_type in ("call", "put"):
        labels = {benchmark._moneyness_bucket(case)[0] for case in fixture["cases"] if case["option_type"] == option_type}
        assert labels == {"ITM", "ATM", "OTM"}
    pairs: dict[str, list[dict]] = {}
    for case in fixture["cases"]:
        pairs.setdefault(case["generation_metadata"]["pair_id"], []).append(case)
    assert len(pairs) == 18
    for cases in pairs.values():
        assert len(cases) == 2
        assert {case["option_type"] for case in cases} == {"call", "put"}
        assert cases[0]["parameters"] == cases[1]["parameters"]
        for name in ("spot", "strike", "maturity", "rate", "dividend_yield"):
            assert cases[0][name] == cases[1][name]


def test_fixture_cases_have_no_silent_reference_failure(fixture: dict) -> None:
    for case in fixture["cases"]:
        _, diagnostic = reference_double_heston_option(
            case["spot"], case["strike"], case["maturity"], case["rate"], case["dividend_yield"],
            case["option_type"], case["parameters"],
        )
        assert diagnostic["reliable"] and diagnostic["failure"] is None


def _without_runtime(value: object) -> object:
    if isinstance(value, dict):
        return {key: _without_runtime(item) for key, item in value.items() if "runtime" not in key and "seconds" not in key}
    if isinstance(value, list):
        return [_without_runtime(item) for item in value]
    return value


def test_benchmark_output_is_deterministic_except_wall_clock_runtime(tmp_path: Path) -> None:
    first = benchmark.run_benchmark(tmp_path / "first")
    second = benchmark.run_benchmark(tmp_path / "second")
    assert _without_runtime(first) == _without_runtime(second)
    for filename in ("benchmark_case_results.csv", "benchmark_failures.csv", "benchmark_by_maturity.csv", "benchmark_by_moneyness.csv", "quadrature_comparison.csv"):
        left = pd.read_csv(tmp_path / "first" / filename)
        right = pd.read_csv(tmp_path / "second" / filename)
        kept = [column for column in left.columns if "runtime" not in column and "seconds" not in column]
        pd.testing.assert_frame_equal(left[kept], right[kept], check_exact=False, rtol=1e-12, atol=1e-14)
    assert not (Path("outputs") / "double_heston_benchmark" / "benchmark_case_results.csv").samefile(tmp_path / "first" / "benchmark_case_results.csv")


def test_benchmark_no_arbitrage_and_parity_control_pass_semantics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    passing = benchmark.run_benchmark(tmp_path / "passing")
    assert passing["benchmark_pass"] is True
    assert passing["no_arbitrage_failures_total"] == 0
    assert all(count == 0 for count in passing["parity"]["failures"].values())

    monkeypatch.setattr(benchmark, "price_double_heston_option", lambda *args, **kwargs: -1.0)
    failing = benchmark.run_benchmark(tmp_path / "failing")
    failures = pd.read_csv(tmp_path / "failing" / "benchmark_failures.csv")
    assert failing["benchmark_pass"] is False
    assert failing["no_arbitrage_failures_total"] > 0
    assert sum(failing["parity"]["failures"].values()) > 0
    assert {"no_arbitrage", "put_call_parity"} <= set(failures["failure_type"])


def test_audit_boundary_flags_exclude_rejected_negative_margin_vectors() -> None:
    hard = audit._load_bounds()["hard_numerical_safety_bounds"]
    invalid = np.asarray([1.0, .02, .40, -.2, .02, 3.0, .02, .20, -.2, .01])
    row = audit._margin_row(1, invalid, hard)
    assert row["accepted"] is False
    assert row["constraint_violating"] is True
    assert row["slow_feller_margin"] < 0.0
    assert not any(row[name] for name in (
        "accepted_hard_bound_near", "accepted_feller_near",
        "accepted_correlation_disk_near", "accepted_weak_slow_fast_separation",
        "accepted_any_boundary_near", "boundary_near", "feller_near",
        "correlation_disk_near", "weak_slow_fast_separation",
    ))


def test_audit_accepted_near_and_interior_population_flags() -> None:
    hard = audit._load_bounds()["hard_numerical_safety_bounds"]
    near = np.asarray([1.5, .04, .335, -.3, .035, 5.0, .02, .40, -.3, .015])
    near_row = audit._margin_row(2, near, hard)
    assert near_row["accepted"] is True
    assert near_row["accepted_feller_near"] is True
    assert near_row["accepted_any_boundary_near"] is True

    interior = np.asarray([1.0, .06, .20, -.3, .04, 4.0, .04, .25, -.3, .03])
    interior_row = audit._margin_row(3, interior, hard)
    assert interior_row["accepted"] is True
    assert interior_row["constraint_violating"] is False
    assert not any(interior_row[name] for name in (
        "accepted_hard_bound_near", "accepted_feller_near",
        "accepted_correlation_disk_near", "accepted_weak_slow_fast_separation",
        "accepted_any_boundary_near",
    ))


def test_audit_output_preserves_pair_details_and_accepted_denominator() -> None:
    root = Path(__file__).resolve().parents[1]
    summary = json.loads((root / "outputs" / "parameter_bounds_audit" / "bounds_audit_summary.json").read_text(encoding="utf-8"))
    proximity = pd.read_csv(root / "outputs" / "parameter_bounds_audit" / "boundary_proximity.csv")
    priced = pd.read_csv(root / "outputs" / "parameter_bounds_audit" / "priced_surface_summary.csv")
    assert len(priced) == 250
    assert summary["boundary_proximity"]["population"] == "accepted_valid_candidates"
    assert summary["boundary_proximity"]["denominator"] == summary["accepted_count"]
    assert set(proximity["population"]) == {"accepted_valid_candidates"}
    assert set(proximity["denominator"]) == {summary["accepted_count"]}
    pairs = [
        (int(row.candidate_id), int(item["nearest_candidate_id"]))
        for row in priced.itertuples()
        for item in json.loads(row.similar_surface_pairs_json)
    ]
    assert len(pairs) == sum(priced["similar_surface_pair_count"])
    assert len(set(pairs)) == summary["priced_surface"]["near_parameter_pairs_similar_surface_count"]


def test_reviewed_sampling_config_is_lf_and_matches_recorded_checksum() -> None:
    root = Path(__file__).resolve().parents[1]
    config = root / "configs" / "parameter_sampling_REVIEWED.yaml"
    checksums = json.loads((root / "outputs" / "engine_freeze" / "source_checksums.json").read_text(encoding="utf-8"))
    payload = config.read_bytes()
    assert b"\r" not in payload
    assert hashlib.sha256(payload).hexdigest() == checksums["configs/parameter_sampling_REVIEWED.yaml"]


def test_reference_and_production_are_statically_independent() -> None:
    root = Path(__file__).resolve().parents[1]
    reference_source = (root / "src" / "double_heston_reference.py").read_text(encoding="utf-8")
    production_source = (root / "src" / "double_heston.py").read_text(encoding="utf-8")
    assert "from .double_heston import" not in reference_source
    assert "from .pricing_interface import" not in reference_source
    assert "double_heston_reference" not in production_source


def test_reference_is_numerically_deterministic_without_runtime_fields(fixture: dict) -> None:
    case = fixture["cases"][10]
    first = reference_double_heston_option(
        case["spot"], case["strike"], case["maturity"], case["rate"], case["dividend_yield"],
        case["option_type"], case["parameters"],
    )
    second = reference_double_heston_option(
        case["spot"], case["strike"], case["maturity"], case["rate"], case["dividend_yield"],
        case["option_type"], case["parameters"],
    )
    assert first == second
