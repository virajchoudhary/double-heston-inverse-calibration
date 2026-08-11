from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.run_g2_identifiability_analysis as baseline
import scripts.run_g2_information_remediation as remediation
from src.calibrate_double_heston import load_hard_safety_bounds


VALID_PARAMETERS = np.asarray(
    [1.2, 0.04, 0.25, -0.35, 0.03, 3.0, 0.02, 0.25, -0.25, 0.02],
    dtype=np.float64,
)


def test_experiment_matrix_is_frozen_to_two_and_three_central5_expiries() -> None:
    matrix = remediation.experiment_matrix().set_index("representation_id")
    assert list(matrix.index) == ["2exp_central5", "3exp_central5"]
    assert matrix.loc["2exp_central5", "candidate_input_dimension"] == 26
    assert matrix.loc["3exp_central5", "candidate_input_dimension"] == 39
    assert matrix.loc["2exp_central5", "normalized_price_count"] == 20
    assert matrix.loc["3exp_central5", "normalized_price_count"] == 30
    assert remediation.MATERIAL_IMPROVEMENT_FACTOR == 10.0


def test_generalized_observables_accept_three_aligned_carry_terms() -> None:
    two = baseline.normalized_observables(
        VALID_PARAMETERS,
        remediation.REPRESENTATION,
        (27, 55),
        node_count=8,
        rates=remediation.CONTROLLED_RATES[:2],
        dividend_yields=remediation.CONTROLLED_DIVIDEND_YIELDS[:2],
    )
    three = baseline.normalized_observables(
        VALID_PARAMETERS,
        remediation.REPRESENTATION,
        (27, 55, 90),
        node_count=8,
        rates=remediation.CONTROLLED_RATES,
        dividend_yields=remediation.CONTROLLED_DIVIDEND_YIELDS,
    )
    assert two.shape == (20,)
    assert three.shape == (30,)
    np.testing.assert_allclose(
        two,
        three[remediation.COMMON_COORDINATES_IN_THREE_EXPIRY_ORDER],
        rtol=0.0,
        atol=2.0e-14,
    )
    with pytest.raises(ValueError, match="same positive number"):
        baseline.normalized_observables(
            VALID_PARAMETERS,
            remediation.REPRESENTATION,
            (27, 55, 90),
            node_count=8,
            rates=remediation.CONTROLLED_RATES[:2],
            dividend_yields=remediation.CONTROLLED_DIVIDEND_YIELDS,
        )


def test_three_expiry_scaled_jacobian_has_declared_shape() -> None:
    bounds = load_hard_safety_bounds(baseline.BOUNDS_PATH)
    jacobian = baseline.scaled_parameter_jacobian(
        VALID_PARAMETERS,
        remediation.REPRESENTATION,
        (27, 55, 90),
        bounds,
        node_count=8,
        rates=remediation.CONTROLLED_RATES,
        dividend_yields=remediation.CONTROLLED_DIVIDEND_YIELDS,
    )
    assert jacobian.shape == (30, 10)
    assert np.isfinite(jacobian).all()


def test_noise_coupling_replays_baseline_and_appends_only_far_shocks() -> None:
    seed = 20260810
    noise_level = 0.005
    expected_baseline = np.random.default_rng(seed).normal(
        0.0, noise_level, size=20
    )
    two_expiry, three_expiry = remediation.coupled_recovery_noise(
        seed, noise_level
    )
    np.testing.assert_array_equal(two_expiry, expected_baseline)
    np.testing.assert_array_equal(
        three_expiry[remediation.COMMON_COORDINATES_IN_THREE_EXPIRY_ORDER],
        two_expiry,
    )
    assert three_expiry.shape == (30,)


def _far_support_frames(active_pct: float = 20.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel_rows: list[dict[str, object]] = []
    support_rows: list[dict[str, object]] = []
    nodes = baseline.MONEYNESS_CENTRAL_5
    for underlying in remediation.PRIMARY_UNDERLYINGS:
        for valuation_date, dte in (
            ("2026-07-01", 90),
            ("2026-07-15", 76),
            ("2026-07-22", 69),
        ):
            actual_expiry = date.fromisoformat(valuation_date) + timedelta(days=dte)
            for index in range(10):
                panel_rows.append(
                    {
                        "underlying": underlying,
                        "valuation_date": valuation_date,
                        "actual_expiry": actual_expiry,
                        "DTE": dte,
                        "expiry_slot": "far",
                        "close_positive": True,
                        "settlement_positive": index != 0,
                        "last_positive": index < 5,
                    }
                )
            active_count = int(round(active_pct * 10 / 100.0))
            for index, (option_type, node) in enumerate(
                (value for value in ((kind, node) for kind in ("CE", "PE") for node in nodes))
            ):
                support_rows.append(
                    {
                        "underlying": underlying,
                        "valuation_date": valuation_date,
                        "expiry_slot": "far",
                        "option_type": option_type,
                        "log_moneyness_node": node,
                        "inside_observed_bounds": True,
                        "active_inside_observed_bounds": index < active_count,
                        "bracket_width": 0.02,
                        "extrapolation_required": False,
                    }
                )
    return pd.DataFrame(panel_rows), pd.DataFrame(support_rows)


def test_far_expiry_separates_structure_price_usability_and_activity() -> None:
    panel, support = _far_support_frames(active_pct=20.0)
    result = remediation.build_far_expiry_support(panel, support)
    assert len(result) == 12
    assert result["structurally_observed"].all()
    assert result["usable_under_declared_close_policy"].all()
    assert not result["actively_traded_under_75pct_rule"].any()
    assert not result["admitted_under_unchanged_market_quality_rule"].any()
    assert result["settlement_positive_pct"].eq(90.0).all()


def test_material_improvement_trigger_is_predeclared_and_rank_safe() -> None:
    jacobian = pd.DataFrame(
        {
            "representation_id": ["2exp_central5"] * 2 + ["3exp_central5"] * 2,
            "smallest_singular_value": [1.0e-9, 2.0e-9, 2.0e-8, 4.0e-8],
            "condition_number": [2.0e9, 1.0e9, 1.0e8, 5.0e7],
            "practical_rank_1e_minus_6": [8, 9, 9, 10],
        }
    )
    result = remediation.material_improvement(jacobian)
    assert result["smallest_singular_value_gain"] == pytest.approx(20.0)
    assert result["condition_number_reduction"] == pytest.approx(20.0)
    assert result["recovery_triggered"] is True


def test_failed_optimizer_start_is_retained_as_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    bounds = load_hard_safety_bounds(baseline.BOUNDS_PATH)
    spec = remediation.EXPERIMENT_SPECS[0]
    observed = baseline.normalized_observables(
        VALID_PARAMETERS,
        remediation.REPRESENTATION,
        (27, 55),
        node_count=8,
        rates=spec.rates,
        dividend_yields=spec.dividend_yields,
    )

    def fail_optimizer(*args: object, **kwargs: object) -> None:
        raise FloatingPointError("synthetic degenerate denominator")

    monkeypatch.setattr(remediation, "least_squares", fail_optimizer)
    rows = remediation._recovery_rows(
        VALID_PARAMETERS,
        observed,
        spec,
        (27, 55),
        bounds,
        sample_id="test",
        profile_id="2026-07-01",
        noise_level=0.0,
        node_count=8,
        max_nfev=2,
        start_count=1,
        seed=1,
    )
    assert len(rows) == 1
    assert rows[0]["optimizer_success"] is False
    assert rows[0]["parameter_recovery_success"] is False
    assert rows[0]["error"].startswith("FloatingPointError:")
    assert np.isnan(rows[0]["recovered_kappa_slow"])


def _decision_frames(
    *, recovery_success: bool, far_admitted: bool
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    jacobian = pd.DataFrame(
        {
            "representation_id": ["3exp_central5"] * 3,
            "practical_rank_1e_minus_6": [10, 10, 10],
            "condition_number": [1.0e5, 2.0e5, 3.0e5],
        }
    )
    recovery = pd.DataFrame(
        {
            "representation_id": ["3exp_central5"] * 3,
            "noise_level": [0.0, 0.005, 0.01],
            "parameter_recovery_success_count": [
                10 if recovery_success else 0,
                10 if recovery_success else 0,
                10 if recovery_success else 0,
            ],
            "start_count": [10, 10, 10],
        }
    )
    far = pd.DataFrame(
        {"admitted_under_unchanged_market_quality_rule": [far_admitted]}
    )
    return jacobian, recovery, far


def test_exact_a_b_c_decision_rule() -> None:
    improvement = {"recovery_triggered": True}
    jacobian, recovery, far = _decision_frames(
        recovery_success=True, far_admitted=True
    )
    passed = remediation.classify_g2(
        jacobian,
        recovery,
        improvement,
        far,
        discount_source_status="VALIDATED",
    )
    assert passed["classification"] == "G2 = PASSED"
    assert passed["final_input_dimension"] == 39

    unresolved = remediation.classify_g2(
        jacobian,
        recovery,
        improvement,
        far,
        discount_source_status="UNRESOLVED",
    )
    assert (
        unresolved["classification"]
        == "G2 = NOT_PASSED — INFORMATION REMEDY IDENTIFIED"
    )
    assert unresolved["final_input_dimension"] is None

    jacobian, recovery, far = _decision_frames(
        recovery_success=False, far_admitted=False
    )
    structural = remediation.classify_g2(
        jacobian,
        recovery,
        improvement,
        far,
        discount_source_status="UNRESOLVED",
    )
    assert (
        structural["classification"]
        == "G2 = NOT_PASSED — STRUCTURAL IDENTIFIABILITY PROBLEM REMAINS"
    )


def test_discount_section_is_preserved_and_fails_closed(tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    report.write_text(
        "# Draft\n\n## Discount-source provenance\n\n"
        "`DISCOUNT_SOURCE = UNRESOLVED`.\n\n## Later\n\ntext\n",
        encoding="utf-8",
    )
    section, status = remediation._discount_source_section(report)
    assert status == "UNRESOLVED"
    assert section.startswith("## Discount-source provenance")
    assert "## Later" not in section
