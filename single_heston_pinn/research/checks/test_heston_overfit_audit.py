#!/usr/bin/env python3
"""Minimal checks for audit metrics and baselines."""

import numpy as np
import pandas as pd

from audit_single_heston_overfitting import baseline_iv, score


def main():
    calibration = pd.DataFrame(
        {
            "expiry_date": ["E1"] * 5,
            "market_iv": [0.24, 0.22, 0.20, 0.21, 0.23],
            "log_forward_moneyness": [-0.2, -0.1, 0.0, 0.1, 0.2],
            "weight": 1.0,
        }
    )
    holdout = pd.DataFrame(
        {"expiry_date": ["E1", "E1"], "log_forward_moneyness": [-0.05, 0.15]}
    )
    assert np.allclose(baseline_iv(calibration, holdout, 0), 0.22)
    frame = pd.DataFrame(
        {
            "symbol": ["A"] * 4,
            "trade_date": ["D"] * 4,
            "expiry_date": ["E"] * 4,
            "market_iv": [0.2, 0.3, 0.4, 0.5],
            "prediction": [0.2, 0.3, 0.4, 0.5],
        }
    )
    result = score(frame, "prediction")
    assert result["iv_rmse"] == 0
    assert result["pooled_iv_r2"] == 1
    assert result["within_surface_centered_r2"] == 1
    print("overfitting-audit metric checks passed")


if __name__ == "__main__":
    main()
