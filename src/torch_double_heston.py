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


def price_double_heston_surface_batch_vectorized(
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
    """Batch-vectorized evaluation of the identical differentiable pricer.

    Computes exactly the formulation of
    :func:`price_double_heston_surface_batch` (same Little Heston Trap
    exponent, same Gauss-Laguerre rule and node count, same guards and
    put-call parity) with every quote of every surface in one vectorized
    pass, removing the per-quote Python loop that makes CPU training
    infeasible (see docs/R2_PRIMARY_COMPARISON_PRE_TRAINING_AUDIT.md, D).

    Numerical equivalence with the loop implementation is pinned by test.
    Inputs MUST be float64 (float32 overflow is invalid for this pricer:
    the training path upcasts before calling; gradients flow through the
    cast).

    Shapes: ``parameters`` (batch, 10); ``spots``/``rates``/``carries``
    (batch,); ``strikes``/``maturities`` (batch, quotes);
    ``option_types`` batch sequences of quote labels.  Returns
    (batch, quotes) prices differentiable w.r.t. ``parameters``.
    """
    if parameters.ndim != 2 or parameters.shape[1] != PARAMETER_COUNT:
        raise ValueError(f"parameters must have shape (batch, {PARAMETER_COUNT})")
    if parameters.dtype != torch.float64:
        raise TypeError(
            "the vectorized differentiable pricer requires float64 inputs "
            "(float32 is numerically invalid for this formulation)"
        )
    batch_size, quote_count = strikes.shape
    _validate_batch_vector(spots, batch_size, "spots")
    _validate_batch_vector(risk_free_rates, batch_size, "risk_free_rates")
    _validate_batch_vector(dividend_yields, batch_size, "dividend_yields")
    if strikes.ndim != 2 or maturities.ndim != 2 or strikes.shape != maturities.shape:
        raise ValueError("strikes and maturities must have identical shape (batch, quotes)")
    if strikes.shape[0] != batch_size:
        raise ValueError("strikes batch dimension must match parameters")
    if len(option_types) != batch_size:
        raise ValueError("option_types must contain one sequence per batch item")
    if not torch.isfinite(parameters).all() or not torch.isfinite(strikes).all():
        raise ValueError("parameters and strikes must be finite")
    for row in option_types:
        if len(row) != quote_count:
            raise ValueError("every option_types row must match the quote count")
        for label in row:
            if label not in (CALL_OPTION, PUT_OPTION):
                raise ValueError("option_type must be 'call' or 'put'")

    device = parameters.device
    nodes, weights = _gauss_laguerre_rule(node_count, device, torch.float64)
    total = batch_size * quote_count
    flat_parameters = (
        parameters.unsqueeze(1)
        .expand(batch_size, quote_count, PARAMETER_COUNT)
        .reshape(total, PARAMETER_COUNT)
    )
    flat_spot = (
        spots.unsqueeze(1).expand(batch_size, quote_count).reshape(total)
    )
    flat_strike = strikes.reshape(total)
    flat_maturity = maturities.reshape(total)
    flat_rate = (
        risk_free_rates.unsqueeze(1).expand(batch_size, quote_count).reshape(total)
    )
    flat_carry = (
        dividend_yields.unsqueeze(1).expand(batch_size, quote_count).reshape(total)
    )
    is_put = torch.zeros(total, dtype=torch.bool, device=device)
    for row_index, row in enumerate(option_types):
        for quote_index, label in enumerate(row):
            is_put[row_index * quote_count + quote_index] = label == PUT_OPTION

    price = _price_quotes_vectorized(
        flat_parameters,
        flat_spot,
        flat_strike,
        flat_maturity,
        flat_rate,
        flat_carry,
        nodes,
        weights,
    )
    call_price = price
    parity = flat_spot * torch.exp(-flat_carry * flat_maturity) - flat_strike * torch.exp(
        -flat_rate * flat_maturity
    )
    price = torch.where(is_put, call_price - parity, call_price)
    if not torch.isfinite(price).all():
        raise FloatingPointError("non-finite Double Heston price (vectorized)")
    return price.reshape(batch_size, quote_count)


def _price_quotes_vectorized(
    parameters: torch.Tensor,
    spot: torch.Tensor,
    strike: torch.Tensor,
    maturity: torch.Tensor,
    risk_free_rate: torch.Tensor,
    dividend_yield: torch.Tensor,
    nodes: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Vectorized call prices for N quotes; (N,) inputs and (N,) output."""
    quote_count = spot.shape[0]
    if torch.any(maturity < 0.0):
        raise ValueError("maturity must be non-negative")
    if not torch.isfinite(spot).all() or not torch.isfinite(strike).all():
        raise ValueError("spot and strike must be finite")

    column = parameters.reshape(quote_count, PARAMETER_COUNT)
    u = nodes.reshape(1, -1)  # (1, K)
    spot_c = spot.reshape(quote_count, 1)
    maturity_c = maturity.reshape(quote_count, 1)
    rate_c = risk_free_rate.reshape(quote_count, 1)
    carry_c = dividend_yield.reshape(quote_count, 1)

    slow = _heston_log_characteristic_exponent_vectorized(
        u,
        maturity_c,
        column[:, 0].reshape(quote_count, 1),
        column[:, 1].reshape(quote_count, 1),
        column[:, 2].reshape(quote_count, 1),
        column[:, 3].reshape(quote_count, 1),
        column[:, 4].reshape(quote_count, 1),
    )
    fast = _heston_log_characteristic_exponent_vectorized(
        u,
        maturity_c,
        column[:, 5].reshape(quote_count, 1),
        column[:, 6].reshape(quote_count, 1),
        column[:, 7].reshape(quote_count, 1),
        column[:, 8].reshape(quote_count, 1),
        column[:, 9].reshape(quote_count, 1),
    )
    exponent = (
        1j
        * u
        * (torch.log(spot_c) + (rate_c - carry_c) * maturity_c)
        + slow
        + fast
    )
    phi_u = torch.exp(exponent)
    phi_shifted = torch.exp(
        _shifted_exponent_vectorized(
            u - 1j, spot_c, maturity_c, rate_c, carry_c, column, quote_count
        )
    )
    phi_minus_i = torch.exp(
        _shifted_exponent_vectorized(
            torch.zeros_like(rate_c) - 1j,
            spot_c,
            maturity_c,
            rate_c,
            carry_c,
            column,
            quote_count,
        ).reshape(quote_count)
    )
    if torch.any(torch.abs(phi_minus_i) < torch.finfo(torch.float64).tiny):
        raise FloatingPointError("characteristic-function normalization is zero")

    log_strike = torch.log(strike.reshape(quote_count, 1))
    oscillation = torch.exp(-1j * u * log_strike)
    inverse_iu = 1.0 / (1j * u)
    laguerre_compensation = torch.exp(u)
    p1_integrand = torch.real(
        oscillation * phi_shifted * inverse_iu / phi_minus_i.reshape(quote_count, 1)
    )
    p2_integrand = torch.real(oscillation * phi_u * inverse_iu)
    p1 = 0.5 + torch.sum(
        weights * laguerre_compensation * p1_integrand, dim=1
    ) / torch.pi
    p2 = 0.5 + torch.sum(
        weights * laguerre_compensation * p2_integrand, dim=1
    ) / torch.pi
    discount_spot = spot * torch.exp(-dividend_yield * maturity)
    discount_strike = strike * torch.exp(-risk_free_rate * maturity)
    price = discount_spot * p1 - discount_strike * p2
    if not torch.isfinite(price).all():
        raise FloatingPointError("non-finite Double Heston call price (vectorized)")
    return price


def _shifted_exponent_vectorized(
    u_shifted: torch.Tensor,
    spot_c: torch.Tensor,
    maturity_c: torch.Tensor,
    rate_c: torch.Tensor,
    carry_c: torch.Tensor,
    column: torch.Tensor,
    quote_count: int,
) -> torch.Tensor:
    """Exponent at a shifted frequency grid (u-1j or the scalar -1j)."""
    slow = _heston_log_characteristic_exponent_vectorized(
        u_shifted,
        maturity_c,
        column[:, 0].reshape(quote_count, 1),
        column[:, 1].reshape(quote_count, 1),
        column[:, 2].reshape(quote_count, 1),
        column[:, 3].reshape(quote_count, 1),
        column[:, 4].reshape(quote_count, 1),
    )
    fast = _heston_log_characteristic_exponent_vectorized(
        u_shifted,
        maturity_c,
        column[:, 5].reshape(quote_count, 1),
        column[:, 6].reshape(quote_count, 1),
        column[:, 7].reshape(quote_count, 1),
        column[:, 8].reshape(quote_count, 1),
        column[:, 9].reshape(quote_count, 1),
    )
    return (
        1j * u_shifted * (torch.log(spot_c) + (rate_c - carry_c) * maturity_c)
        + slow
        + fast
    )


def _heston_log_characteristic_exponent_vectorized(
    u: torch.Tensor,
    maturity: torch.Tensor,
    kappa: torch.Tensor,
    theta: torch.Tensor,
    sigma: torch.Tensor,
    rho: torch.Tensor,
    v0: torch.Tensor,
) -> torch.Tensor:
    """Vectorized stable affine log-characteristic exponent (N quotes x K nodes)."""
    scalars = [kappa, theta, sigma, rho, v0]
    if any(not torch.isfinite(value).all() for value in scalars):
        raise ValueError("kappa, theta, sigma, rho, and v0 must be finite")
    if torch.any(kappa <= 0.0) or torch.any(theta <= 0.0) or torch.any(sigma <= 0.0) or torch.any(v0 <= 0.0):
        raise ValueError("kappa, theta, sigma, and v0 must be strictly positive")
    if torch.any(torch.abs(rho) >= 1.0):
        raise ValueError("rho must lie strictly inside (-1, 1)")
    if torch.any(2.0 * kappa * theta - sigma.square() <= 0.0):
        raise ValueError("the factor Feller gap must be strictly positive")

    iu = 1j * u
    b = kappa - rho * sigma * iu
    discriminant = b * b + sigma.square() * (u * u + iu)
    d = torch.sqrt(discriminant)
    d = torch.where(torch.real(d) < 0.0, -d, d)
    denominator = b + d
    if torch.any(torch.abs(denominator) < torch.finfo(torch.float64).eps):
        raise FloatingPointError("degenerate Little-Heston-Trap denominator")
    g = (b - d) / denominator
    exp_minus_dt = torch.exp(-d * maturity)
    numerator = 1.0 - g * exp_minus_dt
    denominator_log = 1.0 - g
    if torch.any(torch.abs(numerator) == 0.0) or torch.any(torch.abs(denominator_log) == 0.0):
        raise FloatingPointError("zero logarithm argument in Heston exponent")
    log_ratio = torch.log1p(-g * exp_minus_dt) - torch.log1p(-g)
    c_term = (kappa * theta / sigma.square()) * (
        (b - d) * maturity - 2.0 * log_ratio
    )
    d_term = ((b - d) / sigma.square()) * (
        (-torch.expm1(-d * maturity)) / numerator
    )
    result = c_term + d_term * v0
    if not torch.isfinite(torch.real(result)).all() or not torch.isfinite(
        torch.imag(result)
    ).all():
        raise FloatingPointError("non-finite Heston characteristic exponent")
    return result


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
    "price_double_heston_surface_batch_vectorized",
    "price_double_heston_surface_tensor",
]
