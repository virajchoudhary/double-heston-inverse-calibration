from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class FourierConfig:
    alpha: float = 1.5
    integration_steps: int = 256
    u_max: float = 120.0
    integration_eps: float = 1e-4
    truncation_scaler: float = 12.0
    min_truncation_width: float = 0.5


def _as_numpy_vector(values: np.ndarray | float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        array = array.reshape(1)
    return array


def _as_torch_vector(values: torch.Tensor | float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tensor = values if isinstance(values, torch.Tensor) else torch.tensor(values, device=device, dtype=dtype)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1)
    return tensor.to(device=device, dtype=dtype)


def _broadcast_parameter_grid_numpy(parameters: np.ndarray, count: int) -> np.ndarray:
    params = np.asarray(parameters, dtype=np.float64)
    if params.ndim == 1:
        return np.broadcast_to(params, (1, count, params.shape[0]))
    if params.ndim == 2:
        if params.shape[0] != count:
            raise ValueError(f"Expected {count} parameter rows, received {params.shape[0]}.")
        return params.reshape(1, count, params.shape[1])
    if params.ndim == 3:
        return params
    raise ValueError("Unsupported parameter shape for pricing.")


def _broadcast_parameter_grid_torch(parameters: torch.Tensor, count: int) -> torch.Tensor:
    if parameters.ndim == 1:
        return parameters.reshape(1, 1, -1).expand(1, count, -1)
    if parameters.ndim == 2:
        if parameters.shape[0] != count:
            raise ValueError(f"Expected {count} parameter rows, received {parameters.shape[0]}.")
        return parameters.reshape(1, count, parameters.shape[1])
    if parameters.ndim == 3:
        return parameters
    raise ValueError("Unsupported parameter shape for pricing.")


def _heston_factor_terms_numpy(
    u: np.ndarray,
    tau: np.ndarray,
    v0: np.ndarray,
    kappa: np.ndarray,
    theta: np.ndarray,
    sigma: np.ndarray,
    rho: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sigma = np.maximum(sigma, 1e-6)
    iu = 1j * u
    d = np.sqrt((kappa - rho * sigma * iu) ** 2 + sigma**2 * (u**2 + iu))
    g_num = kappa - rho * sigma * iu - d
    g_den = kappa - rho * sigma * iu + d
    g = g_num / (g_den + 1e-12)
    exp_term = np.exp(-d * tau)
    log_term = np.log((1.0 - g * exp_term) / (1.0 - g + 1e-12))
    c = (kappa * theta / (sigma**2)) * (g_num * tau - 2.0 * log_term)
    d_term = (g_num / (sigma**2)) * ((1.0 - exp_term) / (1.0 - g * exp_term + 1e-12))
    return c, d_term * v0


def _heston_factor_terms_torch(
    u: torch.Tensor,
    tau: torch.Tensor,
    v0: torch.Tensor,
    kappa: torch.Tensor,
    theta: torch.Tensor,
    sigma: torch.Tensor,
    rho: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    sigma = torch.clamp(sigma, min=1e-6)
    iu = 1j * u
    d = torch.sqrt((kappa - rho * sigma * iu) ** 2 + sigma.square() * (u.square() + iu))
    g_num = kappa - rho * sigma * iu - d
    g_den = kappa - rho * sigma * iu + d
    g = g_num / (g_den + 1e-12)
    exp_term = torch.exp(-d * tau)
    log_term = torch.log((1.0 - g * exp_term) / (1.0 - g + 1e-12))
    c = (kappa * theta / sigma.square()) * (g_num * tau - 2.0 * log_term)
    d_term = (g_num / sigma.square()) * ((1.0 - exp_term) / (1.0 - g * exp_term + 1e-12))
    return c, d_term * v0


def standard_heston_characteristic_function_numpy(
    u: np.ndarray,
    spot: np.ndarray,
    tau: np.ndarray,
    rates: np.ndarray,
    dividends: np.ndarray,
    parameters: np.ndarray,
) -> np.ndarray:
    spot = np.asarray(spot, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    rates = np.asarray(rates, dtype=np.float64)
    dividends = np.asarray(dividends, dtype=np.float64)
    v0, kappa, theta, sigma, rho = [np.asarray(parameters[..., index], dtype=np.float64) for index in range(5)]
    c, d_term = _heston_factor_terms_numpy(u, tau, v0, kappa, theta, sigma, rho)
    drift = 1j * u * (np.log(spot) + (rates - dividends) * tau)
    return np.exp(drift + c + d_term)


def double_heston_characteristic_function_numpy(
    u: np.ndarray,
    spot: np.ndarray,
    tau: np.ndarray,
    rates: np.ndarray,
    dividends: np.ndarray,
    parameters: np.ndarray,
) -> np.ndarray:
    params = np.asarray(parameters, dtype=np.float64)
    c1, d1 = _heston_factor_terms_numpy(u, tau, params[..., 0], params[..., 1], params[..., 2], params[..., 3], params[..., 4])
    c2, d2 = _heston_factor_terms_numpy(u, tau, params[..., 5], params[..., 6], params[..., 7], params[..., 8], params[..., 9])
    drift = 1j * u * (np.log(spot) + (rates - dividends) * tau)
    return np.exp(drift + c1 + d1 + c2 + d2)


def standard_heston_characteristic_function_torch(
    u: torch.Tensor,
    spot: torch.Tensor,
    tau: torch.Tensor,
    rates: torch.Tensor,
    dividends: torch.Tensor,
    parameters: torch.Tensor,
) -> torch.Tensor:
    c, d_term = _heston_factor_terms_torch(u, tau, parameters[..., 0], parameters[..., 1], parameters[..., 2], parameters[..., 3], parameters[..., 4])
    drift = 1j * u * (torch.log(spot) + (rates - dividends) * tau)
    return torch.exp(drift + c + d_term)


def double_heston_characteristic_function_torch(
    u: torch.Tensor,
    spot: torch.Tensor,
    tau: torch.Tensor,
    rates: torch.Tensor,
    dividends: torch.Tensor,
    parameters: torch.Tensor,
) -> torch.Tensor:
    c1, d1 = _heston_factor_terms_torch(u, tau, parameters[..., 0], parameters[..., 1], parameters[..., 2], parameters[..., 3], parameters[..., 4])
    c2, d2 = _heston_factor_terms_torch(u, tau, parameters[..., 5], parameters[..., 6], parameters[..., 7], parameters[..., 8], parameters[..., 9])
    drift = 1j * u * (torch.log(spot) + (rates - dividends) * tau)
    return torch.exp(drift + c1 + d1 + c2 + d2)


def _cos_reduced_cf_numpy(
    u: np.ndarray,
    spot: np.ndarray,
    strikes: np.ndarray,
    tau: np.ndarray,
    rates: np.ndarray,
    dividends: np.ndarray,
    characteristic_function,
    parameters: np.ndarray,
) -> np.ndarray:
    count = strikes.shape[0]
    strike_grid = strikes.reshape(1, count)
    spot_grid = spot.reshape(1, count)
    tau_grid = tau.reshape(1, count)
    rate_grid = rates.reshape(1, count)
    dividend_grid = dividends.reshape(1, count)
    parameter_grid = _broadcast_parameter_grid_numpy(parameters, count)
    log_strike = np.log(strike_grid)
    return characteristic_function(u, spot_grid, tau_grid, rate_grid, dividend_grid, parameter_grid) * np.exp(-1j * u * log_strike)


def _cos_reduced_cf_torch(
    u: torch.Tensor,
    spot: torch.Tensor,
    strikes: torch.Tensor,
    tau: torch.Tensor,
    rates: torch.Tensor,
    dividends: torch.Tensor,
    characteristic_function,
    parameters: torch.Tensor,
) -> torch.Tensor:
    count = strikes.shape[0]
    strike_grid = strikes.reshape(1, count)
    spot_grid = spot.reshape(1, count)
    tau_grid = tau.reshape(1, count)
    rate_grid = rates.reshape(1, count)
    dividend_grid = dividends.reshape(1, count)
    parameter_grid = _broadcast_parameter_grid_torch(parameters, count)
    log_strike = torch.log(strike_grid)
    return characteristic_function(u, spot_grid, tau_grid, rate_grid, dividend_grid, parameter_grid) * torch.exp(-1j * u * log_strike)


def _heston_factor_cumulants_numpy(
    tau: np.ndarray,
    v0: np.ndarray,
    kappa: np.ndarray,
    theta: np.ndarray,
    sigma: np.ndarray,
    rho: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute analytic cumulants c1, c2, c4 for a single Heston variance factor (numpy).

    These are the variance-related contributions to the cumulants of ln(S_T).
    The drift part (r - q)*tau is added separately.

    Returns the factor's contribution to c1, c2, and c4.
    """
    kappa = np.maximum(kappa, 1e-8)
    sigma = np.maximum(sigma, 1e-6)

    exp_kt = np.exp(-kappa * tau)

    # Factor contribution to c1 (first cumulant / mean of log-return)
    c1_factor = (theta - v0) * (1.0 - exp_kt) / (2.0 * kappa) - 0.5 * theta * tau

    # Factor contribution to c2 (second cumulant / variance of log-return)
    # Gatheral (2006) / Fang-Oosterlee (2008) exact expression
    kappa2 = kappa * kappa
    kappa3 = kappa2 * kappa
    sigma2 = sigma * sigma

    term1 = sigma * tau * kappa * exp_kt * (v0 - theta) * (8.0 * kappa * rho - 4.0 * sigma)
    term2 = kappa * rho * sigma * (1.0 - exp_kt) * (16.0 * theta - 8.0 * v0)
    term3 = 2.0 * theta * kappa * tau * (-4.0 * kappa * rho * sigma + sigma2 + 4.0 * kappa2)
    term4 = sigma2 * ((theta - 2.0 * v0) * exp_kt * exp_kt + theta * (6.0 * exp_kt - 7.0) + 2.0 * v0)
    term5 = 8.0 * kappa2 * (v0 - theta) * (1.0 - exp_kt)
    c2_factor = (term1 + term2 + term3 + term4 + term5) / (8.0 * kappa3)

    # Factor contribution to c4 (fourth cumulant) - simplified approximation
    # The exact c4 is extremely long; we use the leading-order approximation
    # which is sufficient for setting the truncation range
    c4_factor = sigma2 / (4.0 * kappa3) * (
        v0 * (1.0 - exp_kt) * (sigma2 * (1.0 - exp_kt) + 4.0 * kappa * (1.0 + exp_kt))
        + 2.0 * theta * kappa * tau * (sigma2 + 4.0 * kappa)
    )

    return c1_factor, np.maximum(c2_factor, 1e-12), np.maximum(np.abs(c4_factor), 1e-20)


def _cos_truncation_range_numpy(
    spot: np.ndarray,
    strikes: np.ndarray,
    tau: np.ndarray,
    rates: np.ndarray,
    dividends: np.ndarray,
    characteristic_function,
    parameters: np.ndarray,
    config: FourierConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """True COS truncation range using analytic cumulants (Fang & Oosterlee 2008)."""
    count = strikes.shape[0]
    params = np.asarray(parameters, dtype=np.float64)
    parameter_grid = _broadcast_parameter_grid_numpy(params, count)
    tau_grid = tau.reshape(1, count)
    log_moneyness = np.log(spot.reshape(1, count) / strikes.reshape(1, count))
    num_params = parameter_grid.shape[-1]

    # Factor 1 cumulants
    c1_f1, c2_f1, c4_f1 = _heston_factor_cumulants_numpy(
        tau_grid,
        parameter_grid[..., 0], parameter_grid[..., 1], parameter_grid[..., 2],
        parameter_grid[..., 3], parameter_grid[..., 4],
    )

    if num_params >= 10:
        # Factor 2 cumulants (Double Heston)
        c1_f2, c2_f2, c4_f2 = _heston_factor_cumulants_numpy(
            tau_grid,
            parameter_grid[..., 5], parameter_grid[..., 6], parameter_grid[..., 7],
            parameter_grid[..., 8], parameter_grid[..., 9],
        )
    else:
        # Standard Heston — single factor only
        c1_f2 = np.zeros_like(c1_f1)
        c2_f2 = np.zeros_like(c2_f1)
        c4_f2 = np.zeros_like(c4_f1)

    # Total cumulants of ln(S_T/K)
    drift = (rates.reshape(1, count) - dividends.reshape(1, count)) * tau_grid
    c1 = log_moneyness + drift + c1_f1 + c1_f2
    c2 = c2_f1 + c2_f2
    c4 = c4_f1 + c4_f2

    # Fang-Oosterlee truncation: [c1 - L*sqrt(c2 + sqrt(c4)), c1 + L*sqrt(c2 + sqrt(c4))]
    L = float(config.truncation_scaler)
    width = L * np.sqrt(np.maximum(c2 + np.sqrt(c4), 1e-8))
    width = np.maximum(width, float(config.min_truncation_width))
    a = np.minimum(c1 - width, -1e-4)
    b = np.maximum(c1 + width, 1e-4)
    return a.reshape(1, count), b.reshape(1, count)


def _heston_factor_cumulants_torch(
    tau: torch.Tensor,
    v0: torch.Tensor,
    kappa: torch.Tensor,
    theta: torch.Tensor,
    sigma: torch.Tensor,
    rho: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute analytic cumulants c1, c2, c4 for a single Heston variance factor (torch).

    These are the variance-related contributions to the cumulants of ln(S_T).
    The drift part (r - q)*tau is added separately.

    Returns the factor's contribution to c1, c2, and c4.
    """
    kappa = torch.clamp(kappa, min=1e-8)
    sigma = torch.clamp(sigma, min=1e-6)

    exp_kt = torch.exp(-kappa * tau)

    # Factor contribution to c1 (first cumulant / mean of log-return)
    c1_factor = (theta - v0) * (1.0 - exp_kt) / (2.0 * kappa) - 0.5 * theta * tau

    # Factor contribution to c2 (second cumulant / variance of log-return)
    kappa2 = kappa * kappa
    kappa3 = kappa2 * kappa
    sigma2 = sigma * sigma

    term1 = sigma * tau * kappa * exp_kt * (v0 - theta) * (8.0 * kappa * rho - 4.0 * sigma)
    term2 = kappa * rho * sigma * (1.0 - exp_kt) * (16.0 * theta - 8.0 * v0)
    term3 = 2.0 * theta * kappa * tau * (-4.0 * kappa * rho * sigma + sigma2 + 4.0 * kappa2)
    term4 = sigma2 * ((theta - 2.0 * v0) * exp_kt * exp_kt + theta * (6.0 * exp_kt - 7.0) + 2.0 * v0)
    term5 = 8.0 * kappa2 * (v0 - theta) * (1.0 - exp_kt)
    c2_factor = (term1 + term2 + term3 + term4 + term5) / (8.0 * kappa3)

    # Factor contribution to c4 (fourth cumulant) - leading-order approximation
    c4_factor = sigma2 / (4.0 * kappa3) * (
        v0 * (1.0 - exp_kt) * (sigma2 * (1.0 - exp_kt) + 4.0 * kappa * (1.0 + exp_kt))
        + 2.0 * theta * kappa * tau * (sigma2 + 4.0 * kappa)
    )

    return c1_factor, torch.clamp(c2_factor, min=1e-12), torch.clamp(torch.abs(c4_factor), min=1e-20)


def _cos_truncation_range_torch(
    spot: torch.Tensor,
    strikes: torch.Tensor,
    tau: torch.Tensor,
    rates: torch.Tensor,
    dividends: torch.Tensor,
    characteristic_function,
    parameters: torch.Tensor,
    config: FourierConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """True COS truncation range using analytic cumulants (Fang & Oosterlee 2008)."""
    count = strikes.shape[0]
    dtype = spot.dtype
    device = spot.device
    parameter_grid = _broadcast_parameter_grid_torch(parameters, count)
    tau_grid = tau.reshape(1, count)
    log_moneyness = torch.log(spot.reshape(1, count) / strikes.reshape(1, count))
    num_params = parameter_grid.shape[-1]

    # Factor 1 cumulants
    c1_f1, c2_f1, c4_f1 = _heston_factor_cumulants_torch(
        tau_grid,
        parameter_grid[..., 0], parameter_grid[..., 1], parameter_grid[..., 2],
        parameter_grid[..., 3], parameter_grid[..., 4],
    )

    if num_params >= 10:
        # Factor 2 cumulants (Double Heston)
        c1_f2, c2_f2, c4_f2 = _heston_factor_cumulants_torch(
            tau_grid,
            parameter_grid[..., 5], parameter_grid[..., 6], parameter_grid[..., 7],
            parameter_grid[..., 8], parameter_grid[..., 9],
        )
    else:
        # Standard Heston — single factor only
        c1_f2 = torch.zeros_like(c1_f1)
        c2_f2 = torch.zeros_like(c2_f1)
        c4_f2 = torch.zeros_like(c4_f1)

    # Total cumulants of ln(S_T/K)
    drift = (rates.reshape(1, count) - dividends.reshape(1, count)) * tau_grid
    c1 = log_moneyness + drift + c1_f1 + c1_f2
    c2 = c2_f1 + c2_f2
    c4 = c4_f1 + c4_f2

    # Fang-Oosterlee truncation: [c1 - L*sqrt(c2 + sqrt(c4)), c1 + L*sqrt(c2 + sqrt(c4))]
    L = float(config.truncation_scaler)
    width = L * torch.sqrt(torch.clamp(c2 + torch.sqrt(c4), min=1e-8))
    width = torch.clamp(width, min=float(config.min_truncation_width))
    a = torch.minimum(c1 - width, torch.full_like(c1, -1e-4))
    b = torch.maximum(c1 + width, torch.full_like(c1, 1e-4))
    return a.reshape(1, count), b.reshape(1, count)


def _put_cos_coefficients_numpy(a: np.ndarray, b: np.ndarray, num_terms: int) -> tuple[np.ndarray, np.ndarray]:
    k = np.arange(num_terms, dtype=np.float64).reshape(-1, 1)
    width = b - a
    u = k * np.pi / width
    phase = -u * a
    exp_a = np.exp(a)
    chi = (np.cos(phase) - exp_a + u * np.sin(phase)) / (1.0 + u**2)
    psi = np.empty_like(chi)
    psi[0:1, :] = -a
    if num_terms > 1:
        psi[1:, :] = np.sin(phase[1:, :]) / u[1:, :]
    put_u = 2.0 / width * (-chi + psi)
    return u, put_u


def _put_cos_coefficients_torch(a: torch.Tensor, b: torch.Tensor, num_terms: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    k = torch.arange(num_terms, device=device, dtype=dtype).reshape(-1, 1)
    width = b - a
    u = k * np.pi / width
    phase = -u * a
    exp_a = torch.exp(a)
    chi = (torch.cos(phase) - exp_a + u * torch.sin(phase)) / (1.0 + u.square())
    psi = torch.empty_like(chi)
    psi[0:1, :] = -a
    if num_terms > 1:
        psi[1:, :] = torch.sin(phase[1:, :]) / u[1:, :]
    put_u = 2.0 / width * (-chi + psi)
    return u, put_u


def _cos_price_numpy(
    spot: np.ndarray | float,
    strikes: np.ndarray | float,
    tau: np.ndarray | float,
    rates: np.ndarray | float,
    dividends: np.ndarray | float,
    is_call: np.ndarray | float,
    characteristic_function,
    parameters: np.ndarray,
    config: FourierConfig,
) -> np.ndarray:
    spot = _as_numpy_vector(spot)
    strikes = _as_numpy_vector(strikes)
    tau = _as_numpy_vector(tau)
    rates = _as_numpy_vector(rates)
    dividends = _as_numpy_vector(dividends)
    is_call = _as_numpy_vector(is_call)
    count = strikes.shape[0]

    if spot.shape[0] == 1 and count > 1:
        spot = np.broadcast_to(spot, (count,))
    if tau.shape[0] == 1 and count > 1:
        tau = np.broadcast_to(tau, (count,))
    if rates.shape[0] == 1 and count > 1:
        rates = np.broadcast_to(rates, (count,))
    if dividends.shape[0] == 1 and count > 1:
        dividends = np.broadcast_to(dividends, (count,))
    if is_call.shape[0] == 1 and count > 1:
        is_call = np.broadcast_to(is_call, (count,))

    a, b = _cos_truncation_range_numpy(spot, strikes, tau, rates, dividends, characteristic_function, parameters, config)
    num_terms = int(config.integration_steps)
    u, put_u = _put_cos_coefficients_numpy(a, b, num_terms)
    weights = np.ones((num_terms, 1), dtype=np.float64)
    weights[0, 0] = 0.5
    cf_values = _cos_reduced_cf_numpy(u, spot, strikes, tau, rates, dividends, characteristic_function, parameters)
    discounted_put = np.exp(-rates.reshape(1, count) * tau.reshape(1, count))
    put_prices = strikes.reshape(1, count) * discounted_put * np.sum(weights * np.real(cf_values * np.exp(-1j * u * a) * put_u), axis=0, keepdims=True)
    put_prices = put_prices.reshape(count)
    forward = spot * np.exp(-dividends * tau) - strikes * np.exp(-rates * tau)
    call_prices = put_prices + forward
    prices = np.where(is_call > 0.5, call_prices, put_prices)
    return np.maximum(prices, 1e-8)


def _cos_price_torch(
    spot: torch.Tensor | float,
    strikes: torch.Tensor | float,
    tau: torch.Tensor | float,
    rates: torch.Tensor | float,
    dividends: torch.Tensor | float,
    is_call: torch.Tensor | float,
    characteristic_function,
    parameters: torch.Tensor,
    config: FourierConfig,
) -> torch.Tensor:
    device = parameters.device
    dtype = parameters.dtype
    spot = _as_torch_vector(spot, device, dtype)
    strikes = _as_torch_vector(strikes, device, dtype)
    tau = _as_torch_vector(tau, device, dtype)
    rates = _as_torch_vector(rates, device, dtype)
    dividends = _as_torch_vector(dividends, device, dtype)
    is_call = _as_torch_vector(is_call, device, dtype)
    count = strikes.shape[0]

    if spot.shape[0] == 1 and count > 1:
        spot = spot.expand(count)
    if tau.shape[0] == 1 and count > 1:
        tau = tau.expand(count)
    if rates.shape[0] == 1 and count > 1:
        rates = rates.expand(count)
    if dividends.shape[0] == 1 and count > 1:
        dividends = dividends.expand(count)
    if is_call.shape[0] == 1 and count > 1:
        is_call = is_call.expand(count)

    a, b = _cos_truncation_range_torch(spot, strikes, tau, rates, dividends, characteristic_function, parameters, config)
    num_terms = int(config.integration_steps)
    u, put_u = _put_cos_coefficients_torch(a, b, num_terms, dtype=dtype, device=device)
    weights = torch.ones((num_terms, 1), device=device, dtype=dtype)
    weights[0, 0] = 0.5
    cf_values = _cos_reduced_cf_torch(u.to(torch.complex128), spot, strikes, tau, rates, dividends, characteristic_function, parameters)
    exp_factor = torch.exp(-1j * u.to(torch.complex128) * a.to(torch.complex128))
    discounted_put = torch.exp(-rates.reshape(1, count) * tau.reshape(1, count))
    put_prices = strikes.reshape(1, count) * discounted_put * torch.sum(weights * torch.real(cf_values * exp_factor * put_u.to(torch.complex128)), dim=0, keepdim=True)
    put_prices = put_prices.reshape(count)
    forward = spot * torch.exp(-dividends * tau) - strikes * torch.exp(-rates * tau)
    call_prices = put_prices + forward
    prices = torch.where(is_call > 0.5, call_prices, put_prices)
    return torch.clamp(prices, min=1e-8)


def price_standard_heston_numpy(
    spot: np.ndarray | float,
    strikes: np.ndarray | float,
    tau: np.ndarray | float,
    rates: np.ndarray | float,
    dividends: np.ndarray | float,
    is_call: np.ndarray | float,
    parameters: np.ndarray,
    config: FourierConfig,
) -> np.ndarray:
    return _cos_price_numpy(spot, strikes, tau, rates, dividends, is_call, standard_heston_characteristic_function_numpy, parameters, config)


def price_double_heston_numpy(
    spot: np.ndarray | float,
    strikes: np.ndarray | float,
    tau: np.ndarray | float,
    rates: np.ndarray | float,
    dividends: np.ndarray | float,
    is_call: np.ndarray | float,
    parameters: np.ndarray,
    config: FourierConfig,
) -> np.ndarray:
    return _cos_price_numpy(spot, strikes, tau, rates, dividends, is_call, double_heston_characteristic_function_numpy, parameters, config)


def price_standard_heston_torch(
    spot: torch.Tensor | float,
    strikes: torch.Tensor | float,
    tau: torch.Tensor | float,
    rates: torch.Tensor | float,
    dividends: torch.Tensor | float,
    is_call: torch.Tensor | float,
    parameters: torch.Tensor,
    config: FourierConfig,
) -> torch.Tensor:
    return _cos_price_torch(spot, strikes, tau, rates, dividends, is_call, standard_heston_characteristic_function_torch, parameters, config)


def price_double_heston_torch(
    spot: torch.Tensor | float,
    strikes: torch.Tensor | float,
    tau: torch.Tensor | float,
    rates: torch.Tensor | float,
    dividends: torch.Tensor | float,
    is_call: torch.Tensor | float,
    parameters: torch.Tensor,
    config: FourierConfig,
) -> torch.Tensor:
    return _cos_price_torch(spot, strikes, tau, rates, dividends, is_call, double_heston_characteristic_function_torch, parameters, config)
