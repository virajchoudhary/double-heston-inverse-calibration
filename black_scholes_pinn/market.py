"""Analytical reference utilities used only for data generation and evaluation."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr


REQUIRED_COLUMNS = ("spot", "strike", "tau", "rate", "dividend", "call_price")


def black_scholes_call(
    spot: np.ndarray | float,
    strike: np.ndarray | float,
    tau: np.ndarray | float,
    rate: float,
    dividend: float,
    sigma: float,
) -> np.ndarray:
    """Analytical call prices for reference data and out-of-sample scoring."""
    spot_array, strike_array, tau_array = np.broadcast_arrays(
        np.asarray(spot, dtype=float),
        np.asarray(strike, dtype=float),
        np.asarray(tau, dtype=float),
    )
    safe_tau = np.maximum(tau_array, 1.0e-14)
    vol_time = sigma * np.sqrt(safe_tau)
    d1 = (
        np.log(spot_array / strike_array)
        + (rate - dividend + 0.5 * sigma * sigma) * safe_tau
    ) / vol_time
    d2 = d1 - vol_time
    value = (
        spot_array * np.exp(-dividend * safe_tau) * ndtr(d1)
        - strike_array * np.exp(-rate * safe_tau) * ndtr(d2)
    )
    intrinsic = np.maximum(spot_array - strike_array, 0.0)
    return np.where(tau_array <= 0.0, intrinsic, value)


def generate_synthetic_market(
    *,
    true_sigma: float,
    rate: float,
    dividend: float,
    seed: int,
    noise_fraction: float = 0.0,
) -> pd.DataFrame:
    """Create a reproducible quote surface; true_sigma is not used by training."""
    if true_sigma <= 0.0 or noise_fraction < 0.0:
        raise ValueError("true_sigma must be positive and noise_fraction non-negative")
    spot = 100.0
    strikes = np.linspace(62.5, 145.0, 18)
    maturities = np.array([0.08, 0.15, 0.25, 0.40, 0.60, 0.85, 1.10, 1.45, 1.75, 2.00])
    strike_grid, tau_grid = np.meshgrid(strikes, maturities, indexing="xy")
    prices = black_scholes_call(
        spot, strike_grid.ravel(), tau_grid.ravel(), rate, dividend, true_sigma
    )
    if noise_fraction:
        rng = np.random.default_rng(seed)
        prices = np.maximum(
            prices + rng.normal(0.0, noise_fraction * np.maximum(prices, 0.25)),
            0.0,
        )
    return pd.DataFrame(
        {
            "spot": spot,
            "strike": strike_grid.ravel(),
            "tau": tau_grid.ravel(),
            "rate": rate,
            "dividend": dividend,
            "call_price": prices,
        }
    )


def load_market_csv(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"market CSV is missing required columns: {missing}")
    data = data.loc[:, REQUIRED_COLUMNS].copy()
    if data.isna().any().any():
        raise ValueError("market CSV contains missing values")
    if (data[["spot", "strike", "tau"]] <= 0.0).any().any():
        raise ValueError("spot, strike, and tau must be positive")
    if (data["call_price"] < 0.0).any():
        raise ValueError("call_price cannot be negative")
    if data["rate"].nunique() != 1 or data["dividend"].nunique() != 1:
        raise ValueError("this constant-coefficient PINN requires one rate and one dividend yield")
    return data


def market_to_normalized(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.log(data["spot"].to_numpy() / data["strike"].to_numpy())
    tau = data["tau"].to_numpy(dtype=float)
    normalized_price = data["call_price"].to_numpy(dtype=float) / data["strike"].to_numpy()
    return x, tau, normalized_price


def dense_evaluation_grid(
    *,
    rate: float,
    dividend: float,
    true_sigma: float,
) -> pd.DataFrame:
    spot = 100.0
    # Offset both axes from the training quote grid so every scored contract is
    # genuinely out of sample rather than a denser mix containing train points.
    strikes = np.linspace(63.1, 144.4, 67)
    maturities = np.linspace(0.065, 1.985, 35)
    strike_grid, tau_grid = np.meshgrid(strikes, maturities, indexing="xy")
    prices = black_scholes_call(
        spot, strike_grid.ravel(), tau_grid.ravel(), rate, dividend, true_sigma
    )
    return pd.DataFrame(
        {
            "spot": spot,
            "strike": strike_grid.ravel(),
            "tau": tau_grid.ravel(),
            "reference_price": prices,
        }
    )
