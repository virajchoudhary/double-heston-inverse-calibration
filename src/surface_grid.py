"""Fixed option-surface grid construction and flattening utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .constants import (
    CALL_OPTION,
    LOG_MONEYNESS_GRID,
    MATURITY_DAYS_GRID,
    OPTION_TYPES,
    PUT_OPTION,
)


def construct_strikes(spot: float, log_moneyness: Sequence[float]) -> np.ndarray:
    """Construct strikes as ``spot * exp(log_moneyness)``."""
    if not np.isfinite(spot) or spot <= 0.0:
        raise ValueError("spot must be finite and strictly positive")
    values = np.asarray(log_moneyness, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("log_moneyness must be a finite one-dimensional sequence")
    return spot * np.exp(values)


def maturity_days_to_years(maturity_days: Sequence[int | float]) -> np.ndarray:
    """Convert positive calendar days to year fractions using days / 365."""
    days = np.asarray(maturity_days, dtype=np.float64)
    if days.ndim != 1 or not np.isfinite(days).all() or np.any(days <= 0.0):
        raise ValueError("maturity_days must be a positive finite one-dimensional sequence")
    return days / 365.0


def build_surface_grid(
    spot: float,
    log_moneyness: Sequence[float] = LOG_MONEYNESS_GRID,
    maturity_days: Sequence[int] = MATURITY_DAYS_GRID,
    option_types: Sequence[str] = OPTION_TYPES,
) -> pd.DataFrame:
    """Build deterministic option-major, maturity-major, moneyness-major rows."""
    invalid_types = [item for item in option_types if item not in OPTION_TYPES]
    if invalid_types or len(set(option_types)) != len(option_types):
        raise ValueError(f"Invalid or duplicate option types: {invalid_types}")
    moneyness_values = np.asarray(log_moneyness, dtype=np.float64)
    strike_values = construct_strikes(spot, moneyness_values)
    maturity_values = np.asarray(maturity_days, dtype=np.int64)
    year_values = maturity_days_to_years(maturity_values)
    rows: list[dict[str, float | int | str]] = []
    for option_type in option_types:
        for days, years in zip(maturity_values, year_values, strict=True):
            for moneyness, strike in zip(moneyness_values, strike_values, strict=True):
                rows.append(
                    {
                        "log_moneyness": float(moneyness),
                        "strike": float(strike),
                        "maturity_days": int(days),
                        "maturity_years": float(years),
                        "option_type": option_type,
                    }
                )
    return pd.DataFrame(rows)


def flatten_surface_prices(
    call_prices: Sequence[float],
    put_prices: Sequence[float],
    mask: Sequence[bool] | None = None,
) -> np.ndarray:
    """Flatten calls first and puts second, matching ``OPTION_TYPES`` order."""
    calls = _finite_vector(call_prices, "call_prices")
    puts = _finite_vector(put_prices, "put_prices")
    if calls.shape != puts.shape:
        raise ValueError(f"Call and put shapes differ: {calls.shape} != {puts.shape}")
    flattened = np.concatenate([calls, puts])
    if mask is None:
        return flattened
    mask_array = np.asarray(mask, dtype=bool)
    if mask_array.shape != flattened.shape:
        raise ValueError(f"mask shape {mask_array.shape} != prices shape {flattened.shape}")
    return np.where(mask_array, flattened, 0.0)


def expected_input_size(
    include_calls: bool = True,
    include_puts: bool = True,
    log_moneyness: Sequence[float] = LOG_MONEYNESS_GRID,
    maturity_days: Sequence[int] = MATURITY_DAYS_GRID,
) -> int:
    """Return the fixed flattened feature length."""
    option_count = int(include_calls) + int(include_puts)
    if option_count == 0:
        raise ValueError("At least one of calls or puts must be included")
    return option_count * len(log_moneyness) * len(maturity_days)


def normalize_prices_by_spot(
    prices: Sequence[float], spot: float, mask: Sequence[bool] | None = None
) -> np.ndarray:
    """Normalize prices by spot while retaining masked positions as zero."""
    if not np.isfinite(spot) or spot <= 0.0:
        raise ValueError("spot must be finite and strictly positive")
    values = _finite_vector(prices, "prices")
    normalized = values / spot
    if mask is None:
        return normalized
    mask_array = np.asarray(mask, dtype=bool)
    if mask_array.shape != normalized.shape:
        raise ValueError(f"mask shape {mask_array.shape} != prices shape {normalized.shape}")
    return np.where(mask_array, normalized, 0.0)


def _finite_vector(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite one-dimensional sequence")
    return array


__all__ = [
    "CALL_OPTION",
    "PUT_OPTION",
    "build_surface_grid",
    "construct_strikes",
    "expected_input_size",
    "flatten_surface_prices",
    "maturity_days_to_years",
    "normalize_prices_by_spot",
]
