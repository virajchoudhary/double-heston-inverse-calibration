from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_g2_identifiability_analysis import (
    BOUNDS_PATH,
    CALL_OPTION,
    CANDIDATE_INPUT_DIMENSION,
    CARRY_CONTRACT_ID,
    CARRY_INPUT_ORDER,
    MATURITY_PROFILES,
    PARAMETER_NAMES,
    REPRESENTATIONS,
    build_grid,
    decide_gate,
    inspect_carry_contract_sources,
    normalized_observables,
    scaled_parameter_jacobian,
    select_representative_parameters,
)
from src.calibrate_double_heston import load_hard_safety_bounds
from src.double_heston import price_double_heston_surface


VALID_PARAMETERS = np.asarray(
    [1.2, 0.04, 0.25, -0.35, 0.03, 3.0, 0.02, 0.25, -0.25, 0.02],
    dtype=np.float64,
)


def test_carry_contract_is_maturity_aligned_discount_forward_and_dimension_26() -> None:
    assert CARRY_CONTRACT_ID == "discount_forward_per_maturity_v1"
    assert CARRY_INPUT_ORDER == (
        "discount_factor_near",
        "forward_over_spot_near",
        "discount_factor_middle",
        "forward_over_spot_middle",
    )
    assert CANDIDATE_INPUT_DIMENSION == 26


def test_carry_source_inspection_fails_closed_without_verified_discount_source() -> None:
    evidence = inspect_carry_contract_sources()
    assert evidence["futures_combined_carry_available"] is True
    assert evidence["verified_external_rate_or_discount_source"] is False
    assert evidence["market_discount_forward_ready"] is False
    assert evidence["reviewed_synthetic_contract_implemented"] is False
    assert evidence["generic_synthetic_has_hidden_r_q_confound"] is True


def test_proposed_grid_has_canonical_option_expiry_moneyness_order() -> None:
    strikes, maturities, option_types = build_grid(
        REPRESENTATIONS[0], MATURITY_PROFILES[0][1]
    )
    assert len(strikes) == 20
    assert option_types.tolist() == [CALL_OPTION] * 10 + ["put"] * 10
    assert maturities[:5].tolist() == [27.0 / 365.0] * 5
    assert maturities[5:10].tolist() == [55.0 / 365.0] * 5
    assert np.all(np.diff(strikes[:5]) > 0.0)


def test_discount_forward_conversion_matches_flat_scalar_pricing() -> None:
    representation = REPRESENTATIONS[0]
    maturity_days = MATURITY_PROFILES[0][1]
    observed = normalized_observables(
        VALID_PARAMETERS,
        representation,
        maturity_days,
        node_count=8,
        rates=(0.06, 0.06),
        dividend_yields=(0.02, 0.02),
    )
    strikes, maturities, option_types = build_grid(representation, maturity_days)
    expected = price_double_heston_surface(
        100.0,
        strikes,
        maturities,
        0.06,
        0.02,
        option_types,
        VALID_PARAMETERS,
        node_count=8,
    ) / 100.0
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=2.0e-14)


def test_calls_and_puts_duplicate_parameter_information_conditional_on_carry() -> None:
    bounds = load_hard_safety_bounds(BOUNDS_PATH)
    maturity_days = MATURITY_PROFILES[0][1]
    both = scaled_parameter_jacobian(
        VALID_PARAMETERS,
        REPRESENTATIONS[0],
        maturity_days,
        bounds,
        node_count=8,
    )
    calls = scaled_parameter_jacobian(
        VALID_PARAMETERS,
        REPRESENTATIONS[3],
        maturity_days,
        bounds,
        node_count=8,
    )
    assert both.shape == (20, 10)
    assert calls.shape == (10, 10)
    np.testing.assert_allclose(both[:10], calls, rtol=0.0, atol=5.0e-10)
    np.testing.assert_allclose(both[10:], calls, rtol=0.0, atol=5.0e-10)


def test_representative_parameter_selection_is_deterministic_and_balanced() -> None:
    bounds = load_hard_safety_bounds(BOUNDS_PATH)
    first = select_representative_parameters(bounds, per_distribution=2)
    second = select_representative_parameters(bounds, per_distribution=2)
    pd.testing.assert_frame_equal(first, second)
    assert first.groupby("distribution").size().to_dict() == {
        "interior_train": 2,
        "wide_valid_train": 2,
    }
    assert first["sample_id"].is_unique


def test_gate_remains_not_passed_when_market_carry_is_unavailable() -> None:
    carry = {"market_discount_forward_ready": False}
    jacobian = pd.DataFrame(
        {
            "representation_id": ["central5_calls_puts"],
            "practical_rank_1e_minus_6": [10],
            "condition_number": [1.0e4],
        }
    )
    recovery = pd.DataFrame(
        {
            "noise_level": [0.0, 0.005, 0.01],
            "parameter_recovery_success_count": [10, 10, 10],
            "start_count": [10, 10, 10],
        }
    )
    decision = decide_gate(carry, jacobian, recovery)
    assert decision["g2_verdict"] == "NOT_PASSED"
    assert decision["market_carry_pass"] is False


def test_canonical_parameter_order_is_unchanged() -> None:
    assert PARAMETER_NAMES == [
        "kappa_slow",
        "theta_slow",
        "sigma_slow",
        "rho_slow",
        "v0_slow",
        "kappa_fast",
        "theta_fast",
        "sigma_fast",
        "rho_fast",
        "v0_fast",
    ]
