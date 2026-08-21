from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from dheston.calibration.transforms import PARAMETER_BOUNDS, sample_parameter_vector
from dheston.pricing.heston import FourierConfig, price_double_heston_numpy


@dataclass
class CalibrationResult:
    best_parameters: np.ndarray
    objective_value: float
    success: bool
    starts: int


def _surface_objective(
    parameters: np.ndarray,
    spot: np.ndarray,
    strikes: np.ndarray,
    tau: np.ndarray,
    rates: np.ndarray,
    dividends: np.ndarray,
    is_call: np.ndarray,
    market_prices: np.ndarray,
    pricing_config: FourierConfig,
) -> float:
    if parameters[6] > parameters[1]:
        return 1e6 + float((parameters[6] - parameters[1]) ** 2)
    model_prices = price_double_heston_numpy(spot, strikes, tau, rates, dividends, is_call, parameters, pricing_config)
    scaled_errors = (model_prices - market_prices) / np.maximum(spot, 1.0)
    return float(np.mean(np.square(scaled_errors)))


def calibrate_surface_multistart(
    spot: np.ndarray,
    strikes: np.ndarray,
    tau: np.ndarray,
    rates: np.ndarray,
    dividends: np.ndarray,
    is_call: np.ndarray,
    market_prices: np.ndarray,
    pricing_config: FourierConfig,
    starts: int = 5,
    seed: int = 42,
) -> CalibrationResult:
    rng = np.random.default_rng(seed)
    bounds = [PARAMETER_BOUNDS[name] for name in PARAMETER_BOUNDS]
    best_value = float("inf")
    best_vector = None
    best_success = False

    for _ in range(starts):
        x0 = sample_parameter_vector(rng)
        result = minimize(
            _surface_objective,
            x0=x0,
            bounds=bounds,
            method="L-BFGS-B",
            args=(spot, strikes, tau, rates, dividends, is_call, market_prices, pricing_config),
            options={"maxiter": 200},
        )
        if result.fun < best_value:
            best_value = float(result.fun)
            best_vector = result.x.copy()
            best_success = bool(result.success)

    if best_vector is None:
        raise RuntimeError("Conventional calibration did not evaluate any starting point.")

    return CalibrationResult(best_parameters=best_vector, objective_value=best_value, success=best_success, starts=starts)

