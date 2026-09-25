#!/usr/bin/env python3
"""Minimal guard tests for validation-only Heston recalibration."""

import numpy as np
import pandas as pd

from improve_heston_generalization import apply_calibrator, fit_calibrator, select_models


def sample(dates: int) -> pd.DataFrame:
    rows = []
    for day in pd.date_range("2026-01-01", periods=dates):
        for candidate, shift in [("main", 0.0), ("half_a", 0.60), ("half_b", -0.60)]:
            for h in np.linspace(0.15, 0.45, 8):
                rows.append(
                    {
                        "symbol": "TEST",
                        "target_date": day,
                        "target_split": "validation",
                        "candidate": candidate,
                        "row_key": f"{day}-{candidate}-{h}",
                        "forecast_iv": h + shift,
                        "market_iv": 0.05 + 1.35 * h,
                        "log_forward_moneyness": h - 0.30,
                        "maturity": 0.10,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    selection, _ = select_models(sample(12))
    assert selection.iloc[0].candidate == "main"
    assert selection.iloc[0].correction_accepted
    assert selection.iloc[0].selected_specification != "raw"
    short_selection, _ = select_models(sample(6))
    assert not short_selection.iloc[0].correction_accepted
    assert short_selection.iloc[0].selected_specification == "raw"
    frame = sample(12).query("candidate == 'main'")
    coefficients = fit_calibrator(frame, "affine")
    assert np.sqrt(np.mean((apply_calibrator(frame, "affine", coefficients) - frame.market_iv) ** 2)) < 1e-10
    print("Heston generalization guard checks passed")


if __name__ == "__main__":
    main()
