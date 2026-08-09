"""Deterministic constrained calibration for controlled synthetic experiments."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import least_squares
from scipy.special import expit

from .constants import PARAMETER_NAMES
from .constraints import validate_parameters
from .double_heston import price_double_heston_surface


def _logit(probability: float | np.ndarray) -> float | np.ndarray:
    value = np.clip(probability, 1e-8, 1.0 - 1e-8)
    return np.log(value / (1.0 - value))


def load_hard_safety_bounds(bounds_path: str | Path) -> dict[str, tuple[float, float]]:
    """Load the explicit hard numerical-safety envelope from the YAML config."""
    payload = yaml.safe_load(Path(bounds_path).read_text(encoding="utf-8"))
    if payload.get("status") != "PROVISIONAL_CANONICAL_REIMPLEMENTATION":
        raise ValueError("Expected provisional canonical-reimplementation bounds")
    raw = payload.get("hard_numerical_safety_bounds", {})
    result: dict[str, tuple[float, float]] = {}
    for name in PARAMETER_NAMES:
        entry = raw.get(name, {})
        lower, upper = float(entry.get("lower", np.nan)), float(
            entry.get("upper", np.nan)
        )
        if not np.isfinite([lower, upper]).all() or lower >= upper:
            raise ValueError(f"Invalid hard numerical-safety bounds for {name}")
        result[name] = (lower, upper)
    return result


def unconstrained_to_parameters(
    unconstrained: Sequence[float],
    hard_bounds: dict[str, tuple[float, float]],
) -> np.ndarray:
    """Map ten real values to a vector satisfying every declared constraint."""
    x = np.asarray(unconstrained, dtype=np.float64)
    if x.shape != (10,) or not np.isfinite(x).all():
        raise ValueError("unconstrained calibration vector must have shape (10,) and be finite")
    unit = expit(np.clip(x, -35.0, 35.0))

    def bounded(name: str, fraction: float) -> float:
        lower, upper = hard_bounds[name]
        return float(lower + fraction * (upper - lower))

    kappa_slow = bounded("kappa_slow", unit[0])
    theta_slow = bounded("theta_slow", unit[1])
    v0_slow = bounded("v0_slow", unit[2])
    fast_lower = max(hard_bounds["kappa_fast"][0], kappa_slow + 1e-5)
    fast_upper = hard_bounds["kappa_fast"][1]
    if fast_lower >= fast_upper:
        raise ValueError("hard bounds cannot satisfy strict slow/fast ordering")
    kappa_fast = float(fast_lower + unit[3] * (fast_upper - fast_lower))
    theta_fast = bounded("theta_fast", unit[4])
    v0_fast = bounded("v0_fast", unit[5])

    def feller_safe_sigma(
        name: str, kappa: float, theta: float, fraction: float
    ) -> float:
        lower, configured_upper = hard_bounds[name]
        feller_upper = np.sqrt(2.0 * kappa * theta) * (1.0 - 1e-7)
        upper = min(configured_upper, float(feller_upper))
        if lower >= upper:
            raise ValueError(f"hard bounds cannot satisfy the Feller rule for {name}")
        return float(lower + fraction * (upper - lower))

    sigma_slow = feller_safe_sigma("sigma_slow", kappa_slow, theta_slow, unit[6])
    sigma_fast = feller_safe_sigma("sigma_fast", kappa_fast, theta_fast, unit[7])

    correlation_limit = min(
        hard_bounds["rho_slow"][1],
        -hard_bounds["rho_slow"][0],
        hard_bounds["rho_fast"][1],
        -hard_bounds["rho_fast"][0],
        0.999999,
    )
    radius = correlation_limit * unit[8]
    angle = -np.pi + 2.0 * np.pi * unit[9]
    rho_slow = float(radius * np.cos(angle))
    rho_fast = float(radius * np.sin(angle))

    vector = np.asarray(
        [
            kappa_slow,
            theta_slow,
            sigma_slow,
            rho_slow,
            v0_slow,
            kappa_fast,
            theta_fast,
            sigma_fast,
            rho_fast,
            v0_fast,
        ],
        dtype=np.float64,
    )
    diagnostics = validate_parameters(vector)
    if not diagnostics["is_valid"]:
        raise RuntimeError(f"constraint transform failed: {diagnostics['violations']}")
    return vector


def parameters_to_unconstrained(
    parameters: Sequence[float],
    hard_bounds: dict[str, tuple[float, float]],
) -> np.ndarray:
    """Construct an inverse-transform coordinate for an interior valid vector."""
    vector = np.asarray(parameters, dtype=np.float64)
    diagnostics = validate_parameters(vector)
    if vector.shape != (10,) or not diagnostics["is_valid"]:
        raise ValueError("parameters must be a valid canonical ten-vector")

    def fraction(name: str, value: float) -> float:
        lower, upper = hard_bounds[name]
        return float((value - lower) / (upper - lower))

    x = np.empty(10, dtype=np.float64)
    x[0] = _logit(fraction("kappa_slow", vector[0]))
    x[1] = _logit(fraction("theta_slow", vector[1]))
    x[2] = _logit(fraction("v0_slow", vector[4]))
    fast_lower = max(hard_bounds["kappa_fast"][0], vector[0] + 1e-5)
    x[3] = _logit(
        (vector[5] - fast_lower) / (hard_bounds["kappa_fast"][1] - fast_lower)
    )
    x[4] = _logit(fraction("theta_fast", vector[6]))
    x[5] = _logit(fraction("v0_fast", vector[9]))
    slow_sigma_upper = min(
        hard_bounds["sigma_slow"][1],
        np.sqrt(2.0 * vector[0] * vector[1]) * (1.0 - 1e-7),
    )
    fast_sigma_upper = min(
        hard_bounds["sigma_fast"][1],
        np.sqrt(2.0 * vector[5] * vector[6]) * (1.0 - 1e-7),
    )
    x[6] = _logit(
        (vector[2] - hard_bounds["sigma_slow"][0])
        / (slow_sigma_upper - hard_bounds["sigma_slow"][0])
    )
    x[7] = _logit(
        (vector[7] - hard_bounds["sigma_fast"][0])
        / (fast_sigma_upper - hard_bounds["sigma_fast"][0])
    )
    correlation_limit = min(
        hard_bounds["rho_slow"][1],
        -hard_bounds["rho_slow"][0],
        hard_bounds["rho_fast"][1],
        -hard_bounds["rho_fast"][0],
        0.999999,
    )
    radius = float(np.hypot(vector[3], vector[8]))
    angle = float(np.arctan2(vector[8], vector[3]))
    x[8] = _logit(radius / correlation_limit)
    x[9] = _logit((angle + np.pi) / (2.0 * np.pi))
    return x


def deterministic_initial_starts(
    known_parameters: Sequence[float],
    hard_bounds: dict[str, tuple[float, float]],
    *,
    seed: int = 42,
) -> list[tuple[str, np.ndarray]]:
    """Return three named deterministic starts, including one disclosed informed start."""
    rng = np.random.default_rng(seed)
    target_coordinate = parameters_to_unconstrained(known_parameters, hard_bounds)
    return [
        ("neutral_transform_midpoint", np.zeros(10, dtype=np.float64)),
        ("deterministic_broad_start", rng.normal(0.0, 1.25, size=10)),
        (
            "disclosed_target_perturbation",
            target_coordinate + rng.normal(0.0, 0.35, size=10),
        ),
    ]


def boundary_diagnostics(
    parameters: Sequence[float],
    hard_bounds: dict[str, tuple[float, float]],
    *,
    threshold_fraction: float = 0.02,
) -> list[str]:
    """Identify hard-bound, ordering, Feller, and correlation-disk proximity."""
    vector = np.asarray(parameters, dtype=np.float64)
    reasons: list[str] = []
    for name, value in zip(PARAMETER_NAMES, vector, strict=True):
        lower, upper = hard_bounds[name]
        fraction = (value - lower) / (upper - lower)
        if fraction <= threshold_fraction:
            reasons.append(f"{name}:near_lower_hard_bound")
        if fraction >= 1.0 - threshold_fraction:
            reasons.append(f"{name}:near_upper_hard_bound")
    if vector[5] - vector[0] <= threshold_fraction * hard_bounds["kappa_fast"][1]:
        reasons.append("kappa_ordering_gap:near_boundary")
    for label, offset in (("slow", 0), ("fast", 5)):
        feller_ratio = vector[offset + 2] ** 2 / (
            2.0 * vector[offset] * vector[offset + 1]
        )
        if feller_ratio >= 1.0 - threshold_fraction:
            reasons.append(f"{label}_feller:near_boundary")
    if vector[3] ** 2 + vector[8] ** 2 >= (1.0 - threshold_fraction) ** 2:
        reasons.append("correlation_disk:near_boundary")
    return reasons


def calibrate_double_heston(
    spot: float,
    strikes: Sequence[float],
    maturities: Sequence[float],
    risk_free_rate: float,
    dividend_yield: float,
    option_types: Sequence[str],
    observed_prices: Sequence[float],
    known_parameters: Sequence[float],
    bounds_path: str | Path,
    *,
    output_csv: str | Path | None = None,
    node_count: int = 64,
    max_nfev: int = 300,
    seed: int = 42,
) -> pd.DataFrame:
    """Run deterministic starts and record recovery and repricing diagnostics.

    Optimizer ``success`` means only that SciPy met a local stopping criterion.
    It is never interpreted as proof of unique parameter recovery.
    """
    prices = np.asarray(observed_prices, dtype=np.float64)
    strikes_array = np.asarray(strikes, dtype=np.float64)
    maturities_array = np.asarray(maturities, dtype=np.float64)
    option_array = np.asarray(option_types, dtype=str)
    known = np.asarray(known_parameters, dtype=np.float64)
    if prices.shape != strikes_array.shape or not np.isfinite(prices).all():
        raise ValueError("observed_prices must be finite and match the quote shape")
    if np.any(prices < 0.0):
        raise ValueError("observed_prices must be non-negative")
    hard_bounds = load_hard_safety_bounds(bounds_path)
    starts = deterministic_initial_starts(known, hard_bounds, seed=seed)
    residual_scale = np.maximum(prices, 1.0)
    rows: list[dict[str, Any]] = []

    def residuals(x: np.ndarray) -> np.ndarray:
        candidate = unconstrained_to_parameters(x, hard_bounds)
        predicted = price_double_heston_surface(
            spot,
            strikes_array,
            maturities_array,
            risk_free_rate,
            dividend_yield,
            option_array,
            candidate,
            node_count=node_count,
        )
        return (predicted - prices) / residual_scale

    for start_index, (strategy, initial_x) in enumerate(starts):
        started = time.perf_counter()
        row: dict[str, Any] = {
            "start_index": start_index,
            "start_strategy": strategy,
            "optimizer_success_is_unique_recovery_proof": False,
        }
        initial_parameters = unconstrained_to_parameters(initial_x, hard_bounds)
        row.update(
            {f"initial_{name}": value for name, value in zip(PARAMETER_NAMES, initial_parameters, strict=True)}
        )
        try:
            result = least_squares(
                residuals,
                initial_x,
                method="trf",
                max_nfev=max_nfev,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
                diff_step=2e-5,
            )
            predicted_parameters = unconstrained_to_parameters(result.x, hard_bounds)
            predicted_prices = price_double_heston_surface(
                spot,
                strikes_array,
                maturities_array,
                risk_free_rate,
                dividend_yield,
                option_array,
                predicted_parameters,
                node_count=node_count,
            )
            price_errors = predicted_prices - prices
            parameter_errors = predicted_parameters - known
            relative_parameter_errors = np.abs(parameter_errors) / np.maximum(
                np.abs(known), 1e-4
            )
            boundary_reasons = boundary_diagnostics(predicted_parameters, hard_bounds)
            row.update(
                {
                    "success": bool(result.success),
                    "optimizer_status": int(result.status),
                    "optimizer_message": str(result.message),
                    "nfev": int(result.nfev),
                    "loss": float(np.mean(residuals(result.x) ** 2)),
                    "price_rmse": float(np.sqrt(np.mean(price_errors**2))),
                    "price_mae": float(np.mean(np.abs(price_errors))),
                    "max_abs_price_error": float(np.max(np.abs(price_errors))),
                    "parameter_rmse": float(np.sqrt(np.mean(parameter_errors**2))),
                    "parameter_mae": float(np.mean(np.abs(parameter_errors))),
                    "max_relative_parameter_error": float(
                        np.max(relative_parameter_errors)
                    ),
                    "boundary_near": bool(boundary_reasons),
                    "boundary_reasons": ";".join(boundary_reasons),
                }
            )
            row.update(
                {
                    f"predicted_{name}": value
                    for name, value in zip(
                        PARAMETER_NAMES, predicted_parameters, strict=True
                    )
                }
            )
        except Exception as error:  # Preserve every failed start in the audit table.
            row.update(
                {
                    "success": False,
                    "optimizer_status": -1,
                    "optimizer_message": f"{type(error).__name__}: {error}",
                    "nfev": 0,
                    "loss": np.nan,
                    "price_rmse": np.nan,
                    "price_mae": np.nan,
                    "max_abs_price_error": np.nan,
                    "parameter_rmse": np.nan,
                    "parameter_mae": np.nan,
                    "max_relative_parameter_error": np.nan,
                    "boundary_near": False,
                    "boundary_reasons": "",
                }
            )
            row.update({f"predicted_{name}": np.nan for name in PARAMETER_NAMES})
        row["runtime_seconds"] = float(time.perf_counter() - started)
        rows.append(row)

    frame = pd.DataFrame(rows)
    if output_csv is not None:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
    return frame
