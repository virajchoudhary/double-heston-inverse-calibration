"""Deterministic NTPC pricing-input reconstruction after CSV loading."""

from __future__ import annotations

import numpy as np
import pandas as pd


def canonicalize_pricing_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    """Restore derived pricing inputs from stable primitive columns."""
    result = frame.copy()
    result["T"] = result["DTE"].to_numpy(dtype=np.float64) / np.float64(365.0)
    simple_yield = result["risk_free_simple_yield"].to_numpy(dtype=np.float64)
    maturity = result["T"].to_numpy(dtype=np.float64)
    discount = 1.0 / (1.0 + simple_yield * maturity)
    rate = -np.log(discount) / maturity
    spot = result["spot"].to_numpy(dtype=np.float64)
    forward = result["matched_futures_price"].to_numpy(dtype=np.float64)
    result["discount_factor"] = discount
    result["continuous_rate"] = rate
    result["futures_implied_carry"] = rate - np.log(forward / spot) / maturity
    return result
