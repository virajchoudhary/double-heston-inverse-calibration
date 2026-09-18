"""Canonical two-factor Heston pricing for European options.

This module is an independent implementation of the documented model contract.
It is not a reproduction of the unavailable teammate source code.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import TypeAlias

import numpy as np

from .constants import CALL_OPTION, PARAMETER_INDICES, PUT_OPTION
from .constraints import dictionary_to_vector, validate_parameters

ParameterInput: TypeAlias = Sequence[float] | Mapping[str, float]
ComplexResult: TypeAlias = complex | np.ndarray

DEFAULT_GAUSS_LAGUERRE_NODES = 64
_MIN_GAUSS_LAGUERRE_NODES = 8
_MAX_GAUSS_LAGUERRE_NODES = 128


def validate_double_heston_inputs(
    spot: float,
    strikes: Sequence[float],
    maturities: Sequence[float],
    risk_free_rate: float,
    dividend_yield: float,
    option_types: Sequence[str],
    parameters: ParameterInput,
    *,
    enforce_ordering: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate and normalize a quote-aligned pricing request.

    ``strikes``, ``maturities`` (in years), and ``option_types`` must be
    one-dimensional arrays of equal length. The returned tuple contains those
    three normalized arrays followed by the canonical ten-parameter vector.
    ``enforce_ordering=False`` exists only for factor-symmetry diagnostics; all
    normal pricing entry points enforce the declared slow/fast ordering.
    """
    if not np.isfinite(spot) or spot <= 0.0:
        raise ValueError("spot must be finite and strictly positive")
    if not np.isfinite(risk_free_rate) or not np.isfinite(dividend_yield):
        raise ValueError("risk_free_rate and dividend_yield must be finite")

    strike_array = np.asarray(strikes, dtype=np.float64)
    maturity_array = np.asarray(maturities, dtype=np.float64)
    option_array = np.asarray(option_types, dtype=str)
    if strike_array.ndim != 1 or maturity_array.ndim != 1 or option_array.ndim != 1:
        raise ValueError("strikes, maturities, and option_types must be one-dimensional")
    if not (
        strike_array.shape == maturity_array.shape == option_array.shape
    ):
        raise ValueError("strikes, maturities, and option_types must have equal shapes")
    if strike_array.size == 0:
        raise ValueError("at least one option quote is required")
    if not np.isfinite(strike_array).all() or np.any(strike_array <= 0.0):
        raise ValueError("strikes must be finite and strictly positive")
    if not np.isfinite(maturity_array).all() or np.any(maturity_array <= 0.0):
        raise ValueError("maturities must be finite and strictly positive")
    if not set(option_array.tolist()).issubset({CALL_OPTION, PUT_OPTION}):
        raise ValueError("option_types must contain only 'call' or 'put'")

    vector = (
        dictionary_to_vector(parameters)
        if isinstance(parameters, Mapping)
        else np.asarray(parameters, dtype=np.float64)
    )
    diagnostics = validate_parameters(vector)
    violations = list(diagnostics["violations"])
    if not enforce_ordering:
        violations = [
            item
            for item in violations
            if item != "kappa_slow must be strictly less than kappa_fast"
        ]
    if violations:
        raise ValueError(f"Invalid Double Heston parameters: {violations}")
    return strike_array, maturity_array, option_array, vector.copy()


def heston_log_characteristic_exponent(
    u: complex | np.ndarray,
    maturity: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    v0: float,
) -> ComplexResult:
    """Return one variance factor's affine log-characteristic exponent.

    The implementation uses the stable Little-Heston-Trap representation with
    ``g=(b-d)/(b+d)`` and ``exp(-d*T)``. The square-root branch is normalized so
    that ``Re(d) >= 0``. The deterministic log-spot and carry terms are not part
    of this factor exponent.
    """
    if not np.isfinite(maturity) or maturity < 0.0:
        raise ValueError("maturity must be finite and non-negative")
    values = np.asarray([kappa, theta, sigma, rho, v0], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("factor parameters must be finite")
    if kappa <= 0.0 or theta <= 0.0 or sigma <= 0.0 or v0 <= 0.0:
        raise ValueError("kappa, theta, sigma, and v0 must be strictly positive")
    if not -1.0 < rho < 1.0:
        raise ValueError("rho must lie strictly inside (-1, 1)")
    if 2.0 * kappa * theta - sigma**2 <= 0.0:
        raise ValueError("the factor Feller gap must be strictly positive")

    u_array = np.asarray(u, dtype=np.complex128)
    if not np.isfinite(u_array.real).all() or not np.isfinite(u_array.imag).all():
        raise ValueError("u must be finite")
    if maturity == 0.0:
        zero = np.zeros_like(u_array, dtype=np.complex128)
        return complex(zero) if zero.ndim == 0 else zero

    iu = 1j * u_array
    b = kappa - rho * sigma * iu
    discriminant = b * b + sigma * sigma * (u_array * u_array + iu)
    d = np.sqrt(discriminant)
    d = np.where(np.real(d) < 0.0, -d, d)
    denominator = b + d
    if np.any(np.abs(denominator) < np.finfo(np.float64).eps):
        raise FloatingPointError("degenerate Little-Heston-Trap denominator")
    g = (b - d) / denominator
    exp_minus_dt = np.exp(-d * maturity)
    numerator = 1.0 - g * exp_minus_dt
    denominator_log = 1.0 - g
    if np.any(np.abs(numerator) == 0.0) or np.any(np.abs(denominator_log) == 0.0):
        raise FloatingPointError("zero logarithm argument in Heston exponent")
    log_ratio = np.log1p(-g * exp_minus_dt) - np.log1p(-g)

    c_term = (kappa * theta / sigma**2) * (
        (b - d) * maturity - 2.0 * log_ratio
    )
    d_term = ((b - d) / sigma**2) * (
        (-np.expm1(-d * maturity)) / numerator
    )
    result = c_term + d_term * v0
    if not np.isfinite(result.real).all() or not np.isfinite(result.imag).all():
        raise FloatingPointError("non-finite Heston characteristic exponent")
    return complex(result) if result.ndim == 0 else result


def double_heston_characteristic_function(
    u: complex | np.ndarray,
    spot: float,
    maturity: float,
    risk_free_rate: float,
    dividend_yield: float,
    parameters: ParameterInput,
    *,
    enforce_ordering: bool = True,
) -> ComplexResult:
    """Return the characteristic function of ``log(S_T)``.

    The two independent variance factors combine additively in log-characteristic
    space, equivalently multiplicatively in characteristic-function space.
    """
    _, _, _, vector = validate_double_heston_inputs(
        spot,
        [spot],
        [maturity],
        risk_free_rate,
        dividend_yield,
        [CALL_OPTION],
        parameters,
        enforce_ordering=enforce_ordering,
    )
    u_array = np.asarray(u, dtype=np.complex128)
    slow = heston_log_characteristic_exponent(
        u_array,
        maturity,
        vector[PARAMETER_INDICES["kappa_slow"]],
        vector[PARAMETER_INDICES["theta_slow"]],
        vector[PARAMETER_INDICES["sigma_slow"]],
        vector[PARAMETER_INDICES["rho_slow"]],
        vector[PARAMETER_INDICES["v0_slow"]],
    )
    fast = heston_log_characteristic_exponent(
        u_array,
        maturity,
        vector[PARAMETER_INDICES["kappa_fast"]],
        vector[PARAMETER_INDICES["theta_fast"]],
        vector[PARAMETER_INDICES["sigma_fast"]],
        vector[PARAMETER_INDICES["rho_fast"]],
        vector[PARAMETER_INDICES["v0_fast"]],
    )
    exponent = (
        1j * u_array * (np.log(spot) + (risk_free_rate - dividend_yield) * maturity)
        + slow
        + fast
    )
    result = np.exp(exponent)
    if not np.isfinite(result.real).all() or not np.isfinite(result.imag).all():
        raise FloatingPointError("non-finite Double Heston characteristic function")
    return complex(result) if result.ndim == 0 else result


@lru_cache(maxsize=8)
def _gauss_laguerre_rule(node_count: int) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(node_count, bool) or not isinstance(node_count, (int, np.integer)):
        raise TypeError("node_count must be an integer")
    if not _MIN_GAUSS_LAGUERRE_NODES <= int(node_count) <= _MAX_GAUSS_LAGUERRE_NODES:
        raise ValueError(
            f"node_count must be between {_MIN_GAUSS_LAGUERRE_NODES} and "
            f"{_MAX_GAUSS_LAGUERRE_NODES}"
        )
    nodes, weights = np.polynomial.laguerre.laggauss(int(node_count))
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


def price_double_heston_call(
    spot: float,
    strike: float,
    maturity: float,
    risk_free_rate: float,
    dividend_yield: float,
    parameters: ParameterInput,
    *,
    node_count: int = DEFAULT_GAUSS_LAGUERRE_NODES,
    enforce_ordering: bool = True,
) -> float:
    """Price one European call using Gauss-Laguerre Fourier integration."""
    _, _, _, vector = validate_double_heston_inputs(
        spot,
        [strike],
        [maturity],
        risk_free_rate,
        dividend_yield,
        [CALL_OPTION],
        parameters,
        enforce_ordering=enforce_ordering,
    )
    nodes, weights = _gauss_laguerre_rule(node_count)
    phi_u = double_heston_characteristic_function(
        nodes,
        spot,
        maturity,
        risk_free_rate,
        dividend_yield,
        vector,
        enforce_ordering=enforce_ordering,
    )
    phi_shifted = double_heston_characteristic_function(
        nodes - 1j,
        spot,
        maturity,
        risk_free_rate,
        dividend_yield,
        vector,
        enforce_ordering=enforce_ordering,
    )
    phi_minus_i = double_heston_characteristic_function(
        -1j,
        spot,
        maturity,
        risk_free_rate,
        dividend_yield,
        vector,
        enforce_ordering=enforce_ordering,
    )
    if abs(phi_minus_i) < np.finfo(np.float64).tiny:
        raise FloatingPointError("characteristic-function normalization is zero")

    oscillation = np.exp(-1j * nodes * np.log(strike))
    inverse_iu = 1.0 / (1j * nodes)
    laguerre_compensation = np.exp(nodes)
    p1_integrand = np.real(
        oscillation * phi_shifted * inverse_iu / phi_minus_i
    )
    p2_integrand = np.real(oscillation * phi_u * inverse_iu)
    p1 = 0.5 + np.sum(weights * laguerre_compensation * p1_integrand) / np.pi
    p2 = 0.5 + np.sum(weights * laguerre_compensation * p2_integrand) / np.pi
    price = (
        spot * np.exp(-dividend_yield * maturity) * p1
        - strike * np.exp(-risk_free_rate * maturity) * p2
    )
    if not np.isfinite(price):
        raise FloatingPointError("non-finite Double Heston call price")
    return float(price)


def price_double_heston_put(
    spot: float,
    strike: float,
    maturity: float,
    risk_free_rate: float,
    dividend_yield: float,
    parameters: ParameterInput,
    *,
    node_count: int = DEFAULT_GAUSS_LAGUERRE_NODES,
    enforce_ordering: bool = True,
) -> float:
    """Price one European put from the call via put-call parity."""
    call_price = price_double_heston_call(
        spot,
        strike,
        maturity,
        risk_free_rate,
        dividend_yield,
        parameters,
        node_count=node_count,
        enforce_ordering=enforce_ordering,
    )
    price = (
        call_price
        - spot * np.exp(-dividend_yield * maturity)
        + strike * np.exp(-risk_free_rate * maturity)
    )
    if not np.isfinite(price):
        raise FloatingPointError("non-finite Double Heston put price")
    return float(price)


def price_double_heston_option(
    spot: float,
    strike: float,
    maturity: float,
    risk_free_rate: float,
    dividend_yield: float,
    option_type: str,
    parameters: ParameterInput,
    *,
    node_count: int = DEFAULT_GAUSS_LAGUERRE_NODES,
    enforce_ordering: bool = True,
) -> float:
    """Price one call or put while preserving the canonical parameter order."""
    if option_type == CALL_OPTION:
        return price_double_heston_call(
            spot,
            strike,
            maturity,
            risk_free_rate,
            dividend_yield,
            parameters,
            node_count=node_count,
            enforce_ordering=enforce_ordering,
        )
    if option_type == PUT_OPTION:
        return price_double_heston_put(
            spot,
            strike,
            maturity,
            risk_free_rate,
            dividend_yield,
            parameters,
            node_count=node_count,
            enforce_ordering=enforce_ordering,
        )
    raise ValueError("option_type must be 'call' or 'put'")


def price_double_heston_surface(
    spot: float,
    strikes: Sequence[float],
    maturities: Sequence[float],
    risk_free_rate: float,
    dividend_yield: float,
    option_types: Sequence[str],
    parameters: ParameterInput,
    *,
    node_count: int = DEFAULT_GAUSS_LAGUERRE_NODES,
    enforce_ordering: bool = True,
) -> np.ndarray:
    """Price a one-dimensional, quote-aligned European option surface."""
    strike_array, maturity_array, option_array, vector = validate_double_heston_inputs(
        spot,
        strikes,
        maturities,
        risk_free_rate,
        dividend_yield,
        option_types,
        parameters,
        enforce_ordering=enforce_ordering,
    )
    result = np.empty_like(strike_array)
    for index, (strike, maturity, option_type) in enumerate(
        zip(strike_array, maturity_array, option_array, strict=True)
    ):
        result[index] = price_double_heston_option(
            spot,
            float(strike),
            float(maturity),
            risk_free_rate,
            dividend_yield,
            str(option_type),
            vector,
            node_count=node_count,
            enforce_ordering=enforce_ordering,
        )
    if not np.isfinite(result).all():
        raise FloatingPointError("surface contains non-finite Double Heston prices")
    return result


def propagate_variance_state(
    kappa: float,
    theta: float,
    v0: float,
    delta_days: float | np.ndarray,
) -> float | np.ndarray:
    """Return ``theta + (v0-theta)*exp(-kappa*delta_days/365)``."""
    values = np.asarray([kappa, theta, v0], dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("kappa, theta, and v0 must be finite and strictly positive")
    days = np.asarray(delta_days, dtype=np.float64)
    if not np.isfinite(days).all() or np.any(days < 0.0):
        raise ValueError("delta_days must be finite and non-negative")
    result = theta + (v0 - theta) * np.exp(-kappa * days / 365.0)
    return float(result) if result.ndim == 0 else result
