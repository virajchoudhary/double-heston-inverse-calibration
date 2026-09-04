#!/usr/bin/env python3
"""Domain construction and leakage-safe quote cache for the single-Heston PINN.

Two jobs:

1. Derive the physical PINN collocation domain that the study asked for --
   spot from 0 to 1.5x the ten-year maximum traded price, maturity in years out
   to the longest expiry the NSE stock-option cycle allows, and instantaneous
   variance taken from inverse-Black-Scholes implied volatility.
2. Re-use ``single_heston.prepare_date`` verbatim so the PINN is calibrated and
   scored on exactly the quote panel, folds and chronological splits the
   conventional benchmark already used.  Nothing here re-prices, re-weights or
   re-splits a quote.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "outputs" / "019fc8a0" / "model_input_option_prices.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "pinn_single_heston"

# NSE equity-derivative contract cycle: three serial monthly expiries are the
# longest stock-option tenor the exchange lists, so the PINN maturity axis stops
# at three months and is sliced one month at a time.
MONTH_SLICES_DAYS = (30, 60, 90)
MAX_CONTRACT_DAYS = 92
MIN_CONTRACT_DAYS = 1


def _load_single_heston():
    spec = importlib.util.spec_from_file_location("single_heston", ROOT / "single_heston.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SH = _load_single_heston()


def ten_year_price_domain(input_csv: Path) -> pd.DataFrame:
    """Per-symbol spot ceiling over the full ten-year archive.

    Legacy 2016-2023 bhavcopies never published the underlying value, so the
    ten-year price maximum cannot come from the spot column alone.  Listed
    strikes bracket the spot on every session the exchange quoted the name, so
    the ceiling is the larger of the observed spot maximum and the listed strike
    maximum.  That is an upper envelope, never an invented price.
    """
    usecols = ["symbol", "trade_date", "strike_price", "adjusted_strike_price",
               "underlying_value", "adjusted_underlying_value"]
    frame = pd.read_csv(input_csv, usecols=usecols, parse_dates=["trade_date"])
    rows = []
    for symbol, group in frame.groupby("symbol", sort=True):
        spot = pd.to_numeric(group.adjusted_underlying_value, errors="coerce")
        raw_spot = pd.to_numeric(group.underlying_value, errors="coerce")
        strike = pd.to_numeric(group.adjusted_strike_price, errors="coerce")
        raw_strike = pd.to_numeric(group.strike_price, errors="coerce")
        spot_max = float(np.nanmax(spot)) if spot.notna().any() else np.nan
        strike_max = float(np.nanmax(np.r_[strike.to_numpy(float), raw_strike.to_numpy(float)]))
        ceiling = float(np.nanmax([spot_max, strike_max]))
        rows.append(
            {
                "symbol": symbol,
                "first_trade_date": group.trade_date.min().date().isoformat(),
                "last_trade_date": group.trade_date.max().date().isoformat(),
                "observed_spot_max": spot_max,
                "observed_spot_min": float(np.nanmin(spot)) if spot.notna().any() else np.nan,
                "listed_strike_max": strike_max,
                "listed_strike_min": float(np.nanmin(np.r_[strike.to_numpy(float), raw_strike.to_numpy(float)])),
                "spot_rows_with_underlying": int(spot.notna().sum()),
                "spot_rows_total": int(len(group)),
                "ten_year_price_max": ceiling,
                "pinn_spot_low": 0.0,
                "pinn_spot_high": 1.5 * ceiling,
            }
        )
    return pd.DataFrame(rows)


def build_quote_panel(input_csv: Path) -> tuple[pd.DataFrame, dict, Counter]:
    """Every leakage-safe quote the conventional benchmark would have used."""
    data = SH.load_ready_data(input_csv)
    split_map = SH.make_split_map(data)
    audit = Counter()
    frames = []
    for (symbol, day), group in data.groupby(["symbol", "trade_date"], sort=True):
        quotes = SH.prepare_date(group, audit)
        if quotes is None:
            continue
        quotes = quotes.copy()
        quotes["split"] = split_map[(symbol, pd.Timestamp(day))]
        frames.append(quotes)
    panel = pd.concat(frames, ignore_index=True)
    panel["trade_date"] = pd.to_datetime(panel.trade_date)
    panel["expiry_date"] = pd.to_datetime(panel.expiry_date)
    panel["days_to_expiry"] = panel.days_to_expiry.astype(int)
    panel["is_call"] = panel.option_type.eq("CE")
    # Heston is homogeneous in (S, K), so the calibration state collapses onto
    # log forward moneyness; both are carried so the physical domain stays visible.
    panel["log_moneyness_spot"] = np.log(panel.spot / panel.strike)
    panel["variance_from_inverse_bsm"] = panel.market_iv ** 2
    return panel, split_map, audit


def variance_domain(panel: pd.DataFrame, low_quantile=0.001, high_quantile=0.999) -> dict:
    """The v axis, taken from inverse-Black-Scholes implied volatility."""
    variance = panel.variance_from_inverse_bsm.to_numpy(float)
    variance = variance[np.isfinite(variance)]
    quantiles = np.quantile(variance, [0.0, low_quantile, 0.01, 0.25, 0.5, 0.75, 0.99, high_quantile, 1.0])
    # Pad outward so calibration never runs into the edge of the trained box.
    low = float(max(0.002, 0.5 * quantiles[1]))
    high = float(min(2.5, 1.6 * quantiles[7]))
    return {
        "source": "v = (inverse Black-Scholes implied volatility)^2 on paired NSE quotes",
        "observed_min": float(quantiles[0]),
        "observed_q001": float(quantiles[1]),
        "observed_q01": float(quantiles[2]),
        "observed_q25": float(quantiles[3]),
        "observed_median": float(quantiles[4]),
        "observed_q75": float(quantiles[5]),
        "observed_q99": float(quantiles[6]),
        "observed_q999": float(quantiles[7]),
        "observed_max": float(quantiles[8]),
        "pinn_variance_low": low,
        "pinn_variance_high": high,
        "pinn_vol_low": float(np.sqrt(low)),
        "pinn_vol_high": float(np.sqrt(high)),
    }


def maturity_domain(panel: pd.DataFrame) -> dict:
    days = panel.days_to_expiry.to_numpy(int)
    return {
        "definition": "T = (expiry_date - trade_date) / 365, expressed in years",
        "unit": "years (1.0 = one year)",
        "observed_days_min": int(days.min()),
        "observed_days_max": int(days.max()),
        "observed_years_min": float(days.min() / 365.0),
        "observed_years_max": float(days.max() / 365.0),
        "contract_cycle": "NSE lists three serial monthly stock-option expiries",
        "month_slices_days": list(MONTH_SLICES_DAYS),
        "month_slices_years": [d / 365.0 for d in MONTH_SLICES_DAYS],
        "pinn_maturity_low_days": MIN_CONTRACT_DAYS,
        "pinn_maturity_high_days": MAX_CONTRACT_DAYS,
        "pinn_maturity_low_years": MIN_CONTRACT_DAYS / 365.0,
        "pinn_maturity_high_years": MAX_CONTRACT_DAYS / 365.0,
    }


def carry_domain(panel: pd.DataFrame) -> dict:
    rate = panel.rate.to_numpy(float)
    dividend = panel.dividend.to_numpy(float)
    return {
        "source": "robust put-call parity regression per (symbol, date, expiry)",
        "rate_min": float(np.nanmin(rate)), "rate_max": float(np.nanmax(rate)),
        "rate_median": float(np.nanmedian(rate)),
        "dividend_min": float(np.nanmin(dividend)), "dividend_max": float(np.nanmax(dividend)),
        "dividend_median": float(np.nanmedian(dividend)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    price_domain = ten_year_price_domain(args.input)
    price_domain.to_csv(args.output / "pinn_spot_domain.csv", index=False)

    panel, split_map, audit = build_quote_panel(args.input)
    panel.to_parquet(args.output / "pinn_quote_panel.parquet", index=False)

    spec = {
        "input_file": str(args.input),
        "input_sha256": SH.sha256(args.input),
        "quote_rows": int(len(panel)),
        "symbols": int(panel.symbol.nunique()),
        "trade_dates": int(panel.trade_date.nunique()),
        "surfaces": int(panel.groupby(["symbol", "trade_date"]).ngroups),
        "split_rows": {k: int(v) for k, v in panel.split.value_counts().items()},
        "fold_rows": {k: int(v) for k, v in panel.fold.value_counts().items()},
        "spot_domain": {
            "definition": "S from 0 to 1.5 x ten-year maximum traded price, per symbol",
            "per_symbol": price_domain.set_index("symbol")[
                ["ten_year_price_max", "pinn_spot_low", "pinn_spot_high"]
            ].to_dict("index"),
        },
        "maturity_domain": maturity_domain(panel),
        "variance_domain": variance_domain(panel),
        "carry_domain": carry_domain(panel),
        "prepare_audit": {k: int(v) for k, v in sorted(audit.items())},
    }
    (args.output / "pinn_domain_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in spec.items() if k != "spot_domain"}, indent=2))
    print("\nspot domain (S in [0, 1.5 x ten-year max]):")
    print(price_domain[["symbol", "observed_spot_max", "listed_strike_max",
                        "ten_year_price_max", "pinn_spot_high"]].to_string(index=False))


if __name__ == "__main__":
    main()
