"""Pricing adapter used by the ANN dataset and evaluation infrastructure."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .constants import CALL_OPTION, PARAMETER_NAMES, PUT_OPTION
from .constraints import dictionary_to_vector, validate_parameters
from .double_heston import price_double_heston_surface as _canonical_surface_price

REAL_PRICING_ENGINE_AVAILABLE = True
NOT_RESEARCH_DATA = "NOT_RESEARCH_DATA"


class MissingPricingEngineError(RuntimeError):
    """Retained for backward compatibility with pre-engine callers."""


def price_double_heston_surface(
    spot: float,
    strikes: Sequence[float],
    maturities: Sequence[float],
    risk_free_rate: float,
    dividend_yield: float,
    option_types: Sequence[str],
    parameters: Sequence[float] | Mapping[str, float],
) -> np.ndarray:
    """Price a surface with the independent canonical implementation.

    Research-mode callers always reach the real engine. The development-only
    dummy mapping remains a separate, explicitly named smoke-test function and
    cannot be selected implicitly through this adapter.
    """
    result = _canonical_surface_price(
        spot,
        strikes,
        maturities,
        risk_free_rate,
        dividend_yield,
        option_types,
        parameters,
    )
    _validate_surface_prices(
        result,
        float(spot),
        np.asarray(strikes, dtype=np.float64),
        np.asarray(maturities, dtype=np.float64),
        np.asarray(option_types, dtype=str),
        risk_free_rate,
        dividend_yield,
    )
    return result


def dummy_surface_generator_for_smoke_test(
    spot: float,
    strikes: Sequence[float],
    maturities: Sequence[float],
    risk_free_rate: float,
    dividend_yield: float,
    option_types: Sequence[str],
    parameters: Sequence[float] | Mapping[str, float],
) -> np.ndarray:
    """Return deterministic development-only prices labelled NOT_RESEARCH_DATA.

    This is deliberately not a Heston implementation. It is a smooth bounded
    mapping used only to test data flow, tensor shapes, training, and checkpoints.
    """
    if not np.isfinite(spot) or spot <= 0.0:
        raise ValueError("spot must be finite and strictly positive")
    strike_array = np.asarray(strikes, dtype=np.float64)
    maturity_array = np.asarray(maturities, dtype=np.float64)
    option_array = np.asarray(option_types, dtype=str)
    if strike_array.ndim != 1:
        raise ValueError("strikes must be one-dimensional")
    expected_shape = strike_array.shape
    if maturity_array.shape != expected_shape or option_array.shape != expected_shape:
        raise ValueError("strikes, maturities, and option_types must have equal shapes")
    if (
        not np.isfinite(strike_array).all()
        or not np.isfinite(maturity_array).all()
        or np.any(strike_array <= 0.0)
        or np.any(maturity_array <= 0.0)
    ):
        raise ValueError("strikes and maturities must be positive and finite")
    if not set(option_array).issubset({CALL_OPTION, PUT_OPTION}):
        raise ValueError("option_types must contain only 'call' or 'put'")

    vector = (
        dictionary_to_vector(parameters)
        if isinstance(parameters, Mapping)
        else np.asarray(parameters, dtype=np.float64)
    )
    diagnostics = validate_parameters(vector)
    if not diagnostics["is_valid"]:
        raise ValueError(f"Invalid smoke-test parameters: {diagnostics['violations']}")

    named = dict(zip(PARAMETER_NAMES, vector, strict=True))
    log_moneyness = np.log(strike_array / spot)
    total_variance = (
        named["theta_slow"]
        + named["theta_fast"]
        + 0.5 * (named["v0_slow"] + named["v0_fast"])
    )
    skew = 0.20 * (named["rho_slow"] + named["rho_fast"])
    curvature = 0.05 * (named["sigma_slow"] + named["sigma_fast"])
    time_value = spot * (
        0.01
        + 0.08
        * np.sqrt(maturity_array)
        * np.exp(-2.0 * np.abs(log_moneyness))
        * (1.0 + 0.25 * np.tanh(total_variance + skew * log_moneyness))
        + curvature * maturity_array / 100.0
    )
    discounted_spot = spot * np.exp(-dividend_yield * maturity_array)
    discounted_strike = strike_array * np.exp(-risk_free_rate * maturity_array)
    call_intrinsic = np.maximum(discounted_spot - discounted_strike, 0.0)
    put_intrinsic = np.maximum(discounted_strike - discounted_spot, 0.0)
    call_prices = np.minimum(call_intrinsic + time_value, discounted_spot)
    put_prices = np.minimum(put_intrinsic + time_value, discounted_strike)
    result = np.where(option_array == CALL_OPTION, call_prices, put_prices)
    _validate_surface_prices(
        result,
        spot,
        strike_array,
        maturity_array,
        option_array,
        risk_free_rate,
        dividend_yield,
    )
    return result.astype(np.float64)


def _validate_surface_prices(
    prices: np.ndarray,
    spot: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
    option_types: np.ndarray,
    risk_free_rate: float,
    dividend_yield: float,
) -> None:
    if prices.shape != strikes.shape or not np.isfinite(prices).all():
        raise ValueError("Pricing output has the wrong shape or non-finite values")
    if np.any(prices < 0.0):
        raise ValueError("Pricing output contains negative prices")
    discounted_spot = spot * np.exp(-dividend_yield * maturities)
    discounted_strike = strikes * np.exp(-risk_free_rate * maturities)
    call_upper = discounted_spot
    put_upper = discounted_strike
    upper = np.where(option_types == CALL_OPTION, call_upper, put_upper)
    if np.any(prices > upper + 1e-10):
        raise ValueError("Pricing output violates basic option upper bounds")
    call_lower = np.maximum(discounted_spot - discounted_strike, 0.0)
    put_lower = np.maximum(discounted_strike - discounted_spot, 0.0)
    lower = np.where(option_types == CALL_OPTION, call_lower, put_lower)
    if np.any(prices < lower - 1e-10):
        raise ValueError("Pricing output violates basic option lower bounds")
