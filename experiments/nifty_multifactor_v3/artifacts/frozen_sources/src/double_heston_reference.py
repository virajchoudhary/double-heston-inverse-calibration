"""Independent adaptive-quadrature reference for the Double Heston contract.

This module deliberately does not import the production pricer or pricing adapter.
It independently implements the affine transform and uses SciPy adaptive quadrature
as a numerical benchmark for the production Gauss--Laguerre implementation.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

import numpy as np
from scipy.integrate import IntegrationWarning, quad

from .constants import CALL_OPTION, PARAMETER_INDICES, PUT_OPTION
from .constraints import dictionary_to_vector, validate_parameters

ParameterInput: TypeAlias = Sequence[float] | Mapping[str, float]
ComplexResult: TypeAlias = complex | np.ndarray

REFERENCE_EPSABS = 1e-10
REFERENCE_EPSREL = 1e-10
REFERENCE_LIMIT = 500


def _vector(parameters: ParameterInput) -> np.ndarray:
    vector = dictionary_to_vector(parameters) if isinstance(parameters, Mapping) else np.asarray(parameters, dtype=np.float64)
    diagnostics = validate_parameters(vector)
    if not diagnostics["is_valid"]:
        raise ValueError(f"Invalid Double Heston parameters: {diagnostics['violations']}")
    return vector


def _validate_quote(spot: float, strike: float, maturity: float, rate: float, dividend_yield: float, parameters: ParameterInput) -> np.ndarray:
    values = np.asarray([spot, strike, maturity, rate, dividend_yield], dtype=np.float64)
    if not np.isfinite(values).all() or spot <= 0.0 or strike <= 0.0 or maturity <= 0.0:
        raise ValueError("spot, strike and maturity must be finite and strictly positive; rates must be finite")
    return _vector(parameters)


def _factor_exponent(u: complex | np.ndarray, maturity: float, kappa: float, theta: float, sigma: float, rho: float, v0: float) -> ComplexResult:
    """Independently evaluate one Little-Heston-Trap affine factor exponent."""
    argument = np.asarray(u, dtype=np.complex128)
    iu = 1j * argument
    b = kappa - rho * sigma * iu
    d = np.sqrt(b * b + sigma * sigma * (argument * argument + iu))
    d = np.where(np.real(d) < 0.0, -d, d)
    denominator = b + d
    if np.any(np.abs(denominator) < np.finfo(np.float64).eps):
        raise FloatingPointError("degenerate reference transform denominator")
    g = (b - d) / denominator
    decay = np.exp(-d * maturity)
    numerator = 1.0 - g * decay
    if np.any(np.abs(numerator) < np.finfo(np.float64).tiny) or np.any(np.abs(1.0 - g) < np.finfo(np.float64).tiny):
        raise FloatingPointError("degenerate reference logarithm argument")
    log_ratio = np.log(1.0 - g * decay) - np.log(1.0 - g)
    c_term = (kappa * theta / (sigma * sigma)) * ((b - d) * maturity - 2.0 * log_ratio)
    d_term = ((b - d) / (sigma * sigma)) * ((1.0 - decay) / numerator)
    result = c_term + d_term * v0
    if not np.isfinite(result.real).all() or not np.isfinite(result.imag).all():
        raise FloatingPointError("non-finite reference factor exponent")
    return complex(result) if result.ndim == 0 else result


def reference_double_heston_characteristic_function(
    u: complex | np.ndarray,
    spot: float,
    maturity: float,
    risk_free_rate: float,
    dividend_yield: float,
    parameters: ParameterInput,
) -> ComplexResult:
    """Return the independently coded characteristic function of ``log(S_T)``."""
    vector = _validate_quote(spot, spot, maturity, risk_free_rate, dividend_yield, parameters)
    argument = np.asarray(u, dtype=np.complex128)
    slow = _factor_exponent(argument, maturity, vector[0], vector[1], vector[2], vector[3], vector[4])
    fast = _factor_exponent(argument, maturity, vector[5], vector[6], vector[7], vector[8], vector[9])
    result = np.exp(1j * argument * (np.log(spot) + (risk_free_rate - dividend_yield) * maturity) + slow + fast)
    if not np.isfinite(result.real).all() or not np.isfinite(result.imag).all():
        raise FloatingPointError("non-finite reference characteristic function")
    return complex(result) if result.ndim == 0 else result


def _quad_diagnostic(integrand: Any, *, epsabs: float, epsrel: float, limit: int) -> tuple[float, dict[str, Any]]:
    caught: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always", IntegrationWarning)
            output = quad(integrand, 0.0, np.inf, epsabs=epsabs, epsrel=epsrel, limit=limit, full_output=1)
        caught = [str(item.message) for item in records]
        value, error, information = output[:3]
        message = str(output[3]) if len(output) > 3 else ""
        diagnostic = {
            "integral_value": float(value), "absolute_error_estimate": float(error),
            "evaluation_count": int(information.get("neval", 0)), "subdivisions": int(information.get("last", 0)),
            "message": message, "warnings": caught, "exception": None,
            "tolerance": float(max(5e-9, 20.0 * epsabs, 20.0 * epsrel * abs(value))),
        }
    except Exception as error:  # retain failure evidence rather than substituting a price
        diagnostic = {"integral_value": float("nan"), "absolute_error_estimate": float("inf"), "evaluation_count": 0,
                      "subdivisions": 0, "message": "", "warnings": caught, "exception": f"{type(error).__name__}: {error}",
                      "tolerance": float(max(5e-9, 20.0 * epsabs))}
        return float("nan"), diagnostic
    diagnostic["reliable"] = bool(np.isfinite(diagnostic["integral_value"]) and np.isfinite(diagnostic["absolute_error_estimate"])
        and not diagnostic["warnings"] and not diagnostic["message"] and diagnostic["exception"] is None
        and diagnostic["absolute_error_estimate"] <= diagnostic["tolerance"])
    return float(value), diagnostic


def reference_double_heston_call(
    spot: float, strike: float, maturity: float, risk_free_rate: float, dividend_yield: float,
    parameters: ParameterInput, *, epsabs: float = REFERENCE_EPSABS, epsrel: float = REFERENCE_EPSREL,
    limit: int = REFERENCE_LIMIT,
) -> tuple[float, dict[str, Any]]:
    """Price one call and retain adaptive-integration evidence for both integrals."""
    vector = _validate_quote(spot, strike, maturity, risk_free_rate, dividend_yield, parameters)
    if epsabs <= 0.0 or epsrel <= 0.0 or limit < 1:
        raise ValueError("epsabs, epsrel and limit must be positive")
    normalization = reference_double_heston_characteristic_function(-1j, spot, maturity, risk_free_rate, dividend_yield, vector)
    if abs(normalization) < np.finfo(np.float64).tiny:
        raise FloatingPointError("reference characteristic-function normalization is zero")
    log_strike = np.log(strike)
    def p1_integrand(u: float) -> float:
        value = np.exp(-1j * u * log_strike) * reference_double_heston_characteristic_function(u - 1j, spot, maturity, risk_free_rate, dividend_yield, vector) / (1j * u * normalization)
        return float(np.real(value))
    def p2_integrand(u: float) -> float:
        value = np.exp(-1j * u * log_strike) * reference_double_heston_characteristic_function(u, spot, maturity, risk_free_rate, dividend_yield, vector) / (1j * u)
        return float(np.real(value))
    p1_integral, p1 = _quad_diagnostic(p1_integrand, epsabs=epsabs, epsrel=epsrel, limit=limit)
    p2_integral, p2 = _quad_diagnostic(p2_integrand, epsabs=epsabs, epsrel=epsrel, limit=limit)
    p1_probability = 0.5 + p1_integral / np.pi
    p2_probability = 0.5 + p2_integral / np.pi
    price = spot * np.exp(-dividend_yield * maturity) * p1_probability - strike * np.exp(-risk_free_rate * maturity) * p2_probability
    diagnostics: dict[str, Any] = {"method": "independent_scipy_quad_fourier", "option_type": CALL_OPTION,
        "p1": p1, "p2": p2, "p1_probability": float(p1_probability), "p2_probability": float(p2_probability),
        "epsabs": float(epsabs), "epsrel": float(epsrel), "limit": int(limit), "price": float(price),
        "price_finite": bool(np.isfinite(price)), "warnings": p1["warnings"] + p2["warnings"],
        "failure": p1["exception"] or p2["exception"],
    }
    diagnostics["reliable"] = bool(p1.get("reliable") and p2.get("reliable") and diagnostics["price_finite"] and not diagnostics["failure"])
    return float(price), diagnostics


def reference_double_heston_option(
    spot: float, strike: float, maturity: float, risk_free_rate: float, dividend_yield: float, option_type: str,
    parameters: ParameterInput, *, epsabs: float = REFERENCE_EPSABS, epsrel: float = REFERENCE_EPSREL, limit: int = REFERENCE_LIMIT,
) -> tuple[float, dict[str, Any]]:
    """Return a reference call or parity-derived put with structured diagnostics."""
    call, diagnostics = reference_double_heston_call(spot, strike, maturity, risk_free_rate, dividend_yield, parameters, epsabs=epsabs, epsrel=epsrel, limit=limit)
    if option_type == CALL_OPTION:
        return call, diagnostics
    if option_type != PUT_OPTION:
        raise ValueError("option_type must be 'call' or 'put'")
    put = call - spot * np.exp(-dividend_yield * maturity) + strike * np.exp(-risk_free_rate * maturity)
    diagnostics = {**diagnostics, "option_type": PUT_OPTION, "price": float(put), "parity_call_price": float(call), "price_finite": bool(np.isfinite(put))}
    diagnostics["reliable"] = bool(diagnostics["reliable"] and diagnostics["price_finite"])
    return float(put), diagnostics


def reference_double_heston_surface(
    spot: float, strikes: Sequence[float], maturities: Sequence[float], risk_free_rate: float, dividend_yield: float,
    option_types: Sequence[str], parameters: ParameterInput, *, epsabs: float = REFERENCE_EPSABS,
    epsrel: float = REFERENCE_EPSREL, limit: int = REFERENCE_LIMIT,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Price aligned quotes while retaining one diagnostic object per quote."""
    if not (len(strikes) == len(maturities) == len(option_types)) or len(strikes) == 0:
        raise ValueError("strikes, maturities and option_types must have equal non-zero length")
    prices: list[float] = []
    diagnostics: list[dict[str, Any]] = []
    for strike, maturity, option_type in zip(strikes, maturities, option_types, strict=True):
        price, evidence = reference_double_heston_option(spot, float(strike), float(maturity), risk_free_rate, dividend_yield, str(option_type), parameters, epsabs=epsabs, epsrel=epsrel, limit=limit)
        prices.append(price)
        diagnostics.append(evidence)
    return np.asarray(prices, dtype=np.float64), diagnostics
