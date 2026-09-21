#!/usr/bin/env python3
"""Small numerical checks for the causal next-session forecast helpers."""

import math

import numpy as np
import pandas as pd

from forecast_single_heston_next_day import choose_candidates, propagate_variance


def main() -> None:
    v0, kappa, theta = 0.09, 2.0, 0.04
    assert math.isclose(propagate_variance(v0, kappa, theta, 0), v0, abs_tol=1e-14)
    assert theta < propagate_variance(v0, kappa, theta, 1) < v0
    assert math.isclose(propagate_variance(theta, kappa, theta, 4), theta, abs_tol=1e-14)

    rows = []
    actual = np.array([0.20, 0.25, 0.30])
    for symbol in ["AAA", "BBB"]:
        for candidate, shift in [("main", 0.03), ("half_a", 0.01), ("half_b", 0.02)]:
            for observed, forecast in zip(actual, actual + shift):
                rows.append({"symbol": symbol, "candidate": candidate, "market_iv": observed, "forecast_iv": forecast})
    selected, metrics = choose_candidates(pd.DataFrame(rows))
    assert selected.selected_candidate.eq("half_a").all()
    assert selected.selection_data.eq("validation_only").all()
    assert metrics.groupby("symbol").candidate.nunique().eq(3).all()
    print("next-session helper checks passed")


if __name__ == "__main__":
    main()
