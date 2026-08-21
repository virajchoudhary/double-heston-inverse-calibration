"""Mathematical and numerical tests for the canonical reimplementation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.double_heston import (
    double_heston_characteristic_function,
    price_double_heston_call,
    price_double_heston_option,
    price_double_heston_put,
    price_double_heston_surface,
    propagate_variance_state,
    validate_double_heston_inputs,
)
from src.pricing_interface import (
    REAL_PRICING_ENGINE_AVAILABLE,
    price_double_heston_surface as adapter_surface_price,
)


VALID_PARAMETERS = np.array(
    [0.8, 0.04, 0.20, -0.45, 0.05, 3.0, 0.02, 0.25, -0.25, 0.025],
    dtype=np.float64,
)
SPOT = 100.0
RATE = 0.05
DIVIDEND = 0.01


def test_characteristic_function_at_zero_is_one() -> None:
    value = double_heston_characteristic_function(
        0.0, SPOT, 1.0, RATE, DIVIDEND, VALID_PARAMETERS
    )
    assert value == pytest.approx(1.0 + 0.0j, abs=1e-13)


def test_canonical_reimplementation_fixture_is_reproducible() -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "double_heston_clean_fixture.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert fixture["fixture_type"] == "CANONICAL_REIMPLEMENTATION_FIXTURE"
    actual = price_double_heston_surface(
        fixture["spot"],
        fixture["strikes"],
        fixture["maturities"],
        fixture["rate"],
        fixture["dividend_yield"],
        fixture["option_types"],
        fixture["parameters"],
        node_count=fixture["pricing_node_count"],
    )
    np.testing.assert_allclose(actual, fixture["expected_prices"], rtol=0.0, atol=1e-12)


def test_ann_pricing_adapter_routes_to_canonical_engine() -> None:
    assert REAL_PRICING_ENGINE_AVAILABLE is True
    arguments = (
        SPOT,
        [90.0, 100.0, 110.0],
        [0.25, 0.5, 1.0],
        RATE,
        DIVIDEND,
        ["call", "put", "call"],
        VALID_PARAMETERS,
    )
    np.testing.assert_array_equal(
        adapter_surface_price(*arguments), price_double_heston_surface(*arguments)
    )


def test_prices_are_finite_and_non_negative() -> None:
    prices = price_double_heston_surface(
        SPOT,
        [75.0, 90.0, 100.0, 110.0, 125.0, 100.0],
        [0.25, 0.5, 1.0, 1.0, 2.0, 0.75],
        RATE,
        DIVIDEND,
        ["call", "call", "call", "put", "put", "put"],
        VALID_PARAMETERS,
    )
    assert np.isfinite(prices).all()
    assert np.all(prices >= 0.0)


@pytest.mark.parametrize("maturity", [0.1, 0.5, 1.0, 2.0])
def test_call_prices_satisfy_no_arbitrage_bounds(maturity: float) -> None:
    strikes = np.array([70.0, 90.0, 100.0, 110.0, 130.0])
    prices = price_double_heston_surface(
        SPOT,
        strikes,
        np.full(strikes.shape, maturity),
        RATE,
        DIVIDEND,
        np.full(strikes.shape, "call"),
        VALID_PARAMETERS,
    )
    discounted_spot = SPOT * np.exp(-DIVIDEND * maturity)
    discounted_strikes = strikes * np.exp(-RATE * maturity)
    lower = np.maximum(discounted_spot - discounted_strikes, 0.0)
    assert np.all(prices >= lower - 1e-9)
    assert np.all(prices <= discounted_spot + 1e-9)


@pytest.mark.parametrize("maturity", [0.1, 0.5, 1.0, 2.0])
def test_put_prices_satisfy_no_arbitrage_bounds(maturity: float) -> None:
    strikes = np.array([70.0, 90.0, 100.0, 110.0, 130.0])
    prices = price_double_heston_surface(
        SPOT,
        strikes,
        np.full(strikes.shape, maturity),
        RATE,
        DIVIDEND,
        np.full(strikes.shape, "put"),
        VALID_PARAMETERS,
    )
    discounted_spot = SPOT * np.exp(-DIVIDEND * maturity)
    discounted_strikes = strikes * np.exp(-RATE * maturity)
    lower = np.maximum(discounted_strikes - discounted_spot, 0.0)
    assert np.all(prices >= lower - 1e-9)
    assert np.all(prices <= discounted_strikes + 1e-9)


def test_put_call_parity_holds() -> None:
    strike = 105.0
    maturity = 0.8
    call = price_double_heston_call(
        SPOT, strike, maturity, RATE, DIVIDEND, VALID_PARAMETERS
    )
    put = price_double_heston_put(
        SPOT, strike, maturity, RATE, DIVIDEND, VALID_PARAMETERS
    )
    parity_error = (
        call
        - put
        - SPOT * np.exp(-DIVIDEND * maturity)
        + strike * np.exp(-RATE * maturity)
    )
    assert abs(parity_error) < 1e-12


def test_call_prices_decrease_with_strike() -> None:
    strikes = np.linspace(70.0, 130.0, 13)
    prices = price_double_heston_surface(
        SPOT,
        strikes,
        np.full(strikes.shape, 1.0),
        RATE,
        DIVIDEND,
        np.full(strikes.shape, "call"),
        VALID_PARAMETERS,
    )
    assert np.all(np.diff(prices) < 0.0)


def test_output_shape_matches_quote_arrays() -> None:
    prices = price_double_heston_surface(
        SPOT,
        [90.0, 100.0, 110.0],
        [0.25, 0.5, 1.0],
        RATE,
        DIVIDEND,
        ["call", "put", "call"],
        VALID_PARAMETERS,
    )
    assert prices.shape == (3,)
    assert prices.dtype == np.float64


def test_repeated_calls_are_deterministic() -> None:
    arguments = (
        SPOT,
        [85.0, 100.0, 115.0],
        [0.25, 0.75, 1.5],
        RATE,
        DIVIDEND,
        ["call", "put", "call"],
        VALID_PARAMETERS,
    )
    first = price_double_heston_surface(*arguments)
    second = price_double_heston_surface(*arguments)
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(
    "parameters",
    [
        VALID_PARAMETERS[:-1],
        np.where(np.arange(10) == 1, -0.04, VALID_PARAMETERS),
        np.where(np.arange(10) == 2, 1.0, VALID_PARAMETERS),
        np.where(np.arange(10) == 3, 0.99, VALID_PARAMETERS),
        np.where(np.arange(10) == 8, -0.95, VALID_PARAMETERS),
        np.where(np.arange(10) == 0, 4.0, VALID_PARAMETERS),
    ],
)
def test_invalid_parameter_vectors_are_rejected(parameters: np.ndarray) -> None:
    with pytest.raises(ValueError):
        price_double_heston_call(
            SPOT, 100.0, 1.0, RATE, DIVIDEND, parameters
        )


@pytest.mark.parametrize(
    ("spot", "strike", "maturity"),
    [
        (0.0, 100.0, 1.0),
        (-1.0, 100.0, 1.0),
        (SPOT, 0.0, 1.0),
        (SPOT, -1.0, 1.0),
        (SPOT, 100.0, 0.0),
        (SPOT, 100.0, -1.0),
        (np.inf, 100.0, 1.0),
        (SPOT, np.nan, 1.0),
    ],
)
def test_invalid_spot_strike_and_maturity_are_rejected(
    spot: float, strike: float, maturity: float
) -> None:
    with pytest.raises(ValueError):
        price_double_heston_call(
            spot, strike, maturity, RATE, DIVIDEND, VALID_PARAMETERS
        )


def test_factor_swapping_is_price_invariant_without_label_ordering() -> None:
    swapped = np.concatenate([VALID_PARAMETERS[5:], VALID_PARAMETERS[:5]])
    original_price = price_double_heston_call(
        SPOT,
        105.0,
        0.9,
        RATE,
        DIVIDEND,
        VALID_PARAMETERS,
        enforce_ordering=False,
    )
    swapped_price = price_double_heston_call(
        SPOT,
        105.0,
        0.9,
        RATE,
        DIVIDEND,
        swapped,
        enforce_ordering=False,
    )
    assert swapped_price == pytest.approx(original_price, abs=2e-12)


def test_declared_ordering_validator_rejects_swapped_labels() -> None:
    swapped = np.concatenate([VALID_PARAMETERS[5:], VALID_PARAMETERS[:5]])
    with pytest.raises(ValueError, match="kappa_slow"):
        validate_double_heston_inputs(
            SPOT, [100.0], [1.0], RATE, DIVIDEND, ["call"], swapped
        )


def test_64_and_96_node_results_are_close() -> None:
    arguments = (SPOT, 115.0, 1.7, RATE, DIVIDEND, VALID_PARAMETERS)
    price_64 = price_double_heston_call(*arguments, node_count=64)
    price_96 = price_double_heston_call(*arguments, node_count=96)
    assert price_96 == pytest.approx(price_64, rel=2e-8, abs=2e-8)


def test_near_one_factor_limit_is_consistent() -> None:
    tiny_fast = VALID_PARAMETERS.copy()
    tiny_fast[6:10] = [1e-8, 1e-4, 0.0, 1e-8]
    tinier_fast = VALID_PARAMETERS.copy()
    tinier_fast[6:10] = [1e-10, 1e-5, 0.0, 1e-10]
    price_tiny = price_double_heston_call(
        SPOT, 100.0, 1.0, RATE, DIVIDEND, tiny_fast
    )
    price_tinier = price_double_heston_call(
        SPOT, 100.0, 1.0, RATE, DIVIDEND, tinier_fast
    )
    assert price_tiny == pytest.approx(price_tinier, abs=2e-5)


def test_vectorized_surface_agrees_with_scalar_calls() -> None:
    strikes = np.array([85.0, 100.0, 115.0, 125.0])
    maturities = np.array([0.2, 0.5, 1.0, 1.8])
    option_types = np.array(["call", "put", "call", "put"])
    surface = price_double_heston_surface(
        SPOT,
        strikes,
        maturities,
        RATE,
        DIVIDEND,
        option_types,
        VALID_PARAMETERS,
    )
    scalar = np.array(
        [
            price_double_heston_option(
                SPOT,
                float(strike),
                float(maturity),
                RATE,
                DIVIDEND,
                str(option_type),
                VALID_PARAMETERS,
            )
            for strike, maturity, option_type in zip(
                strikes, maturities, option_types, strict=True
            )
        ]
    )
    np.testing.assert_array_equal(surface, scalar)


def test_state_propagation_matches_documented_formula() -> None:
    delta_days = np.array([0.0, 7.0, 30.0, 365.0])
    expected = 0.04 + (0.07 - 0.04) * np.exp(-0.8 * delta_days / 365.0)
    actual = propagate_variance_state(0.8, 0.04, 0.07, delta_days)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)
