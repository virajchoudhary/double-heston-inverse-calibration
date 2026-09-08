#!/usr/bin/env python3
"""Small deterministic numerical checks for the single-Heston implementation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from single_heston import (
    black_scholes_price,
    black_scholes_vega,
    encode_params,
    heston_call_prices,
    heston_prices,
    implied_volatility,
    least_squares,
    residuals,
    decode_params,
)


def quote_frame(params):
    rows = []
    for expiry_number, maturity in enumerate((0.08, 0.20, 0.45, 0.80)):
        for strike in np.linspace(70, 130, 13):
            option = "CE" if strike >= 100 * np.exp((0.06 - 0.02) * maturity) else "PE"
            rows.append(
                {
                    "expiry_date": f"E{expiry_number}", "spot": 100.0, "strike": strike,
                    "maturity": maturity, "rate": 0.06, "dividend": 0.02,
                    "option_type": option,
                }
            )
    quotes = pd.DataFrame(rows)
    quotes["market_price_adjusted"] = heston_prices(quotes, params)
    quotes["market_iv"] = implied_volatility(
        quotes.market_price_adjusted, quotes.spot, quotes.strike, quotes.maturity,
        quotes.rate, quotes.dividend, quotes.option_type.eq("CE"),
    )
    quotes["vega"] = black_scholes_vega(
        quotes.spot, quotes.strike, quotes.maturity, quotes.rate, quotes.dividend, quotes.market_iv
    )
    quotes["weight"] = 1.0
    return quotes


def main():
    strikes = np.array([80.0, 100.0, 120.0])
    heston = heston_call_prices(100, strikes, 1.0, 0.05, 0.02, (1.5, 0.04, 0.001, 0, 0.04))
    bs = black_scholes_price(100, strikes, 1.0, 0.05, 0.02, 0.20, np.ones(3, dtype=bool))
    assert np.max(np.abs(heston - bs)) < 0.003

    known = np.array([1.8, 0.055, 0.28, -0.65, 0.07])
    quotes = quote_frame(known)
    starts = ((1.5, 0.06, 0.30, -0.60, 0.06), (3.0, 0.04, 0.25, -0.30, 0.04), (0.7, 0.12, 0.35, -0.75, 0.12))

    def recover(frame):
        results = []
        for start in starts:
            fit = least_squares(residuals, encode_params(start), args=(frame,), bounds=(-8, 8), max_nfev=300)
            fitted = decode_params(fit.x)
            rmse = np.sqrt(np.mean((heston_prices(frame, fitted) - frame.market_price_adjusted) ** 2))
            results.append((rmse, fitted))
        return results

    results = recover(quotes)
    best_rmse, recovered = min(results, key=lambda item: item[0])
    assert best_rmse < 2e-4
    assert np.max(np.abs((recovered - known) / known)) < 0.08
    assert 2 * recovered[0] * recovered[1] > recovered[2] ** 2

    noisy = quotes.copy()
    rng = np.random.default_rng(20260806)
    noisy.market_price_adjusted = np.maximum(
        0.01, noisy.market_price_adjusted * (1 + rng.normal(0, 0.01, len(noisy)))
    )
    noisy.market_iv = implied_volatility(
        noisy.market_price_adjusted, noisy.spot, noisy.strike, noisy.maturity,
        noisy.rate, noisy.dividend, noisy.option_type.eq("CE"),
    )
    noisy.vega = black_scholes_vega(
        noisy.spot, noisy.strike, noisy.maturity, noisy.rate, noisy.dividend, noisy.market_iv
    )
    assert noisy.market_iv.notna().all()
    noisy_results = recover(noisy)
    noisy_rmse, noisy_recovered = min(noisy_results, key=lambda item: item[0])
    assert noisy_rmse < 0.08
    assert 2 * noisy_recovered[0] * noisy_recovered[1] > noisy_recovered[2] ** 2
    report = {
        "synthetic_data_is_separate_from_nse_results": True,
        "known_parameters": known.tolist(),
        "starting_values_tested": [list(start) for start in starts],
        "exact_recovered_parameters": recovered.tolist(),
        "exact_price_rmse": float(best_rmse),
        "exact_max_relative_parameter_error": float(np.max(np.abs((recovered - known) / known))),
        "noisy_price_noise_standard_deviation_fraction": 0.01,
        "noisy_recovered_parameters": noisy_recovered.tolist(),
        "noisy_price_rmse": float(noisy_rmse),
        "exact_start_rmse_range": [float(min(x[0] for x in results)), float(max(x[0] for x in results))],
        "noisy_start_rmse_range": [float(min(x[0] for x in noisy_results)), float(max(x[0] for x in noisy_results))],
        "all_checks_passed": True,
    }
    destination = Path(__file__).resolve().parent / "outputs" / "single_heston" / "single_heston_controlled_tests.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
