"""Differentiable torch implementation of the canonical Double Heston pricer."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import numpy as np
import torch

from .constants import CALL_OPTION, PARAMETER_COUNT, PUT_OPTION
from .double_heston import DEFAULT_GAUSS_LAGUERRE_NODES


_MIN_GAUSS_LAGUERRE_NODES = 8
_MAX_GAUSS_LAGUERRE_NODES = 128
_FELLER_SIGMA_SAFETY = 0.995


def price_double_heston_surface_batch(
    parameters: torch.Tensor,
    spots: torch.Tensor,
    strikes: torch.Tensor,
    maturities: torch.Tensor,
    risk_free_rates: torch.Tensor,
    dividend_yields: torch.Tensor,
    option_types: Sequence[Sequence[str]],
    *,
    node_count: int = DEFAULT_GAUSS_LAGUERRE_NODES,
) -> torch.Tensor:
    """Price a batch of quote-aligned surfaces with gradients in parameter space."""
    if parameters.ndim != 2 or parameters.shape[1] != PARAMETER_COUNT:
        raise ValueError(f"parameters must have shape (batch, {PARAMETER_COUNT})")
    batch_size = parameters.shape[0]
    _validate_batch_vector(spots, batch_size, "spots")
    _validate_batch_vector(risk_free_rates, batch_size, "risk_free_rates")
    _validate_batch_vector(dividend_yields, batch_size, "dividend_yields")
    if strikes.ndim != 2 or maturities.ndim != 2 or strikes.shape != maturities.shape:
        raise ValueError("strikes and maturities must have identical shape (batch, quotes)")
    if strikes.shape[0] != batch_size:
        raise ValueError("strikes batch dimension must match parameters")
    if len(option_types) != batch_size:
        raise ValueError("option_types must contain one sequence per batch item")
    if not torch.isfinite(parameters).all():
        raise ValueError("parameters must be finite")

    results: list[torch.Tensor] = []
    for index in range(batch_size):
        results.append(
            price_double_heston_surface_tensor(
                spots[index],
                strikes[index],
                maturities[index],
                risk_free_rates[index],
                dividend_yields[index],
                option_types[index],
                parameters[index],
                node_count=node_count,
            )
        )
    return torch.stack(results, dim=0)


def price_double_heston_surface_tensor(
    spot: torch.Tensor,
    strikes: torch.Tensor,
    maturities: torch.Tensor,
    risk_free_rate: torch.Tensor,
    dividend_yield: torch.Tensor,
    option_types: Sequence[str],
    parameters: torch.Tensor,
    *,
    node_count: int = DEFAULT_GAUSS_LAGUERRE_NODES,
) -> torch.Tensor:
    """Price one quote-aligned surface using torch complex arithmetic."""
    _validate_node_count(node_count)
    if strikes.ndim != 1 or maturities.ndim != 1 or strikes.shape != maturities.shape:
        raise ValueError("strikes and maturities must be one-dimensional and aligned")
    if len(option_types) != len(strikes):
        raise ValueError("option_types length must match strikes and maturities")
    if not torch.isfinite(strikes).all() or not torch.isfinite(maturities).all():
        raise ValueError("strikes and maturities must be finite")
    if torch.any(strikes <= 0.0) or torch.any(maturities <= 0.0):
        raise ValueError("strikes and maturities must be strictly positive")
    if parameters.ndim != 1 or parameters.shape[0] != PARAMETER_COUNT:
        raise ValueError(f"parameters must have shape ({PARAMETER_COUNT},)")
    if not torch.isfinite(parameters).all():
        raise ValueError("parameters must be finite")

    prices = [
        price_double_heston_option_tensor(
            spot,
            strikes[index],
            maturities[index],
            risk_free_rate,
            dividend_yield,
            option_types[index],
            parameters,
            node_count=node_count,
        )
        for index in range(len(option_types))
    ]
    return torch.stack(prices)


def price_double_heston_option_tensor(
    spot: torch.Tensor,
    strike: torch.Tensor,
    maturity: torch.Tensor,
    risk_free_rate: torch.Tensor,
    dividend_yield: torch.Tensor,
    option_type: str,
    parameters: torch.Tensor,
    *,
    node_count: int = DEFAULT_GAUSS_LAGUERRE_NODES,
) -> torch.Tensor:
    if option_type == CALL_OPTION:
        return price_double_heston_call_tensor(
            spot,
            strike,
            maturity,
            risk_free_rate,
            dividend_yield,
            parameters,
            node_count=node_count,
        )
    if option_type == PUT_OPTION:
        return price_double_heston_put_tensor(
            spot,
            strike,
            maturity,
            risk_free_rate,
            dividend_yield,
            parameters,
            node_count=node_count,
        )
    raise ValueError("option_type must be 'call' or 'put'")


def price_double_heston_call_tensor(
    spot: torch.Tensor,
    strike: torch.Tensor,
    maturity: torch.Tensor,
    risk_free_rate: torch.Tensor,
    dividend_yield: torch.Tensor,
    parameters: torch.Tensor,
    *,
    node_count: int = DEFAULT_GAUSS_LAGUERRE_NODES,
) -> torch.Tensor:
    """Price one call with torch so repricing loss is differentiable."""
    dtype = _real_dtype(parameters)
    device = parameters.device
    nodes, weights = _gauss_laguerre_rule(node_count, device, dtype)
    phi_u = double_heston_characteristic_function_tensor(
        nodes,
        spot,
        maturity,
        risk_free_rate,
        dividend_yield,
        parameters,
    )
    phi_shifted = double_heston_characteristic_function_tensor(
        nodes - 1j,
        spot,
        maturity,
        risk_free_rate,
        dividend_yield,
        parameters,
    )
    phi_minus_i = double_heston_characteristic_function_tensor(
        torch.tensor(-1j, device=device, dtype=_complex_dtype(dtype)),
        spot,
        maturity,
        risk_free_rate,
        dividend_yield,
        parameters,
    )
    if torch.abs(phi_minus_i).detach().item() < torch.finfo(dtype).tiny:
        raise FloatingPointError("characteristic-function normalization is zero")

    log_strike = torch.log(_scalar_tensor(strike, dtype=dtype, device=device))
    oscillation = torch.exp(-1j * nodes * log_strike)
    inverse_iu = 1.0 / (1j * nodes)
    laguerre_compensation = torch.exp(nodes)
    p1_integrand = torch.real(oscillation * phi_shifted * inverse_iu / phi_minus_i)
    p2_integrand = torch.real(oscillation * phi_u * inverse_iu)
    p1 = 0.5 + torch.sum(weights * laguerre_compensation * p1_integrand) / torch.pi
    p2 = 0.5 + torch.sum(weights * laguerre_compensation * p2_integrand) / torch.pi
    spot_value = _scalar_tensor(spot, dtype=dtype, device=device)
    rate = _scalar_tensor(risk_free_rate, dtype=dtype, device=device)
    dividend = _scalar_tensor(dividend_yield, dtype=dtype, device=device)
    maturity_value = _scalar_tensor(maturity, dtype=dtype, device=device)
    discount_spot = spot_value * torch.exp(-dividend * maturity_value)
    discount_strike = _scalar_tensor(strike, dtype=dtype, device=device) * torch.exp(
        -rate * maturity_value
    )
    price = discount_spot * p1 - discount_strike * p2
    if not torch.isfinite(price):
        raise FloatingPointError("non-finite Double Heston call price")
    return price


def price_double_heston_put_tensor(
    spot: torch.Tensor,
    strike: torch.Tensor,
    maturity: torch.Tensor,
    risk_free_rate: torch.Tensor,
    dividend_yield: torch.Tensor,
    parameters: torch.Tensor,
    *,
    node_count: int = DEFAULT_GAUSS_LAGUERRE_NODES,
) -> torch.Tensor:
    """Price one put from the call via put-call parity."""
    call_price = price_double_heston_call_tensor(
        spot,
        strike,
        maturity,
        risk_free_rate,
        dividend_yield,
        parameters,
        node_count=node_count,
    )
    dtype = _real_dtype(parameters)
    device = parameters.device
    spot_value = _scalar_tensor(spot, dtype=dtype, device=device)
    rate = _scalar_tensor(risk_free_rate, dtype=dtype, device=device)
    dividend = _scalar_tensor(dividend_yield, dtype=dtype, device=device)
    maturity_value = _scalar_tensor(maturity, dtype=dtype, device=device)
    strike_value = _scalar_tensor(strike, dtype=dtype, device=device)
    price = call_price - spot_value * torch.exp(-dividend * maturity_value) + strike_value * torch.exp(
        -rate * maturity_value
    )
    if not torch.isfinite(price):
        raise FloatingPointError("non-finite Double Heston put price")
    return price


def double_heston_characteristic_function_tensor(
    u: torch.Tensor,
    spot: torch.Tensor,
    maturity: torch.Tensor,
    risk_free_rate: torch.Tensor,
    dividend_yield: torch.Tensor,
    parameters: torch.Tensor,
) -> torch.Tensor:
    """Return the characteristic function of log spot under the canonical model."""
    dtype = _real_dtype(parameters)
    device = parameters.device
    u_tensor = torch.as_tensor(u, device=device, dtype=_complex_dtype(dtype))
    spot_value = _scalar_tensor(spot, dtype=dtype, device=device)
    maturity_value = _scalar_tensor(maturity, dtype=dtype, device=device)
    rate = _scalar_tensor(risk_free_rate, dtype=dtype, device=device)
    dividend = _scalar_tensor(dividend_yield, dtype=dtype, device=device)
    if maturity_value.detach().item() < 0.0:
        raise ValueError("maturity must be non-negative")

    slow = heston_log_characteristic_exponent_tensor(
        u_tensor,
        maturity_value,
        parameters[0],
        parameters[1],
        parameters[2],
        parameters[3],
        parameters[4],
    )
    fast = heston_log_characteristic_exponent_tensor(
        u_tensor,
        maturity_value,
        parameters[5],
        parameters[6],
        parameters[7],
        parameters[8],
        parameters[9],
    )
    exponent = 1j * u_tensor * (torch.log(spot_value) + (rate - dividend) * maturity_value) + slow + fast
    result = torch.exp(exponent)
    if not torch.isfinite(torch.real(result)).all() or not torch.isfinite(torch.imag(result)).all():
        raise FloatingPointError("non-finite Double Heston characteristic function")
    return result


def heston_log_characteristic_exponent_tensor(
    u: torch.Tensor,
    maturity: torch.Tensor,
    kappa: torch.Tensor,
    theta: torch.Tensor,
    sigma: torch.Tensor,
    rho: torch.Tensor,
    v0: torch.Tensor,
) -> torch.Tensor:
    """Return one factor's stable affine log-characteristic exponent."""
    dtype = _real_dtype(kappa)
    device = kappa.device
    maturity_value = _scalar_tensor(maturity, dtype=dtype, device=device)
    if maturity_value.detach().item() == 0.0:
        return torch.zeros_like(u, dtype=_complex_dtype(dtype))

    for name, value in {
        "kappa": kappa,
        "theta": theta,
        "sigma": sigma,
        "rho": rho,
        "v0": v0,
    }.items():
        if not torch.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if torch.any(torch.stack([kappa, theta, sigma, v0]) <= 0.0):
        raise ValueError("kappa, theta, sigma, and v0 must be strictly positive")
    if torch.abs(rho).detach().item() >= 1.0:
        raise ValueError("rho must lie strictly inside (-1, 1)")
    if (2.0 * kappa * theta - sigma.square()).detach().item() <= 0.0:
        raise ValueError("the factor Feller gap must be strictly positive")

    u_tensor = torch.as_tensor(u, device=device, dtype=_complex_dtype(dtype))
    iu = 1j * u_tensor
    b = kappa - rho * sigma * iu
    discriminant = b * b + sigma.square() * (u_tensor * u_tensor + iu)
    d = torch.sqrt(discriminant)
    d = torch.where(torch.real(d) < 0.0, -d, d)
    denominator = b + d
    if torch.any(torch.abs(denominator) < torch.finfo(dtype).eps).detach().item():
        raise FloatingPointError("degenerate Little-Heston-Trap denominator")
    g = (b - d) / denominator
    exp_minus_dt = torch.exp(-d * maturity_value)
    numerator = 1.0 - g * exp_minus_dt
    denominator_log = 1.0 - g
    if torch.any(torch.abs(numerator) == 0.0).detach().item() or torch.any(
        torch.abs(denominator_log) == 0.0
    ).detach().item():
        raise FloatingPointError("zero logarithm argument in Heston exponent")
    log_ratio = torch.log1p(-g * exp_minus_dt) - torch.log1p(-g)
    c_term = (kappa * theta / sigma.square()) * ((b - d) * maturity_value - 2.0 * log_ratio)
    d_term = ((b - d) / sigma.square()) * ((-torch.expm1(-d * maturity_value)) / numerator)
    result = c_term + d_term * v0
    if not torch.isfinite(torch.real(result)).all() or not torch.isfinite(torch.imag(result)).all():
        raise FloatingPointError("non-finite Heston characteristic exponent")
    return result


@lru_cache(maxsize=8)
def _cached_gauss_laguerre(node_count: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.laguerre.laggauss(node_count)
    return nodes.astype(np.float64), weights.astype(np.float64)


def _gauss_laguerre_rule(
    node_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_node_count(node_count)
    nodes, weights = _cached_gauss_laguerre(node_count)
    return (
        torch.as_tensor(nodes, device=device, dtype=dtype),
        torch.as_tensor(weights, device=device, dtype=dtype),
    )


def _validate_node_count(node_count: int) -> None:
    if isinstance(node_count, bool) or not isinstance(node_count, (int, np.integer)):
        raise TypeError("node_count must be an integer")
    if not _MIN_GAUSS_LAGUERRE_NODES <= int(node_count) <= _MAX_GAUSS_LAGUERRE_NODES:
        raise ValueError(
            f"node_count must be between {_MIN_GAUSS_LAGUERRE_NODES} and {_MAX_GAUSS_LAGUERRE_NODES}"
        )


def _validate_batch_vector(values: torch.Tensor, batch_size: int, name: str) -> None:
    if values.ndim != 1 or values.shape[0] != batch_size:
        raise ValueError(f"{name} must have shape ({batch_size},)")
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} must be finite")


def _real_dtype(reference: torch.Tensor) -> torch.dtype:
    if reference.dtype in {torch.float32, torch.complex64}:
        return torch.float32
    return torch.float64


def _complex_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.float32:
        return torch.complex64
    if dtype == torch.float64:
        return torch.complex128
    raise TypeError(f"Unsupported real dtype: {dtype}")


def _scalar_tensor(
    value: torch.Tensor | float,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if tensor.ndim != 0:
        raise ValueError("expected a scalar tensor")
    if not torch.isfinite(tensor):
        raise ValueError("scalar inputs must be finite")
    return tensor


__all__ = [
    "price_double_heston_surface_batch",
    "price_double_heston_surface_tensor",
]
