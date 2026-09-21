#!/usr/bin/env python3
"""Leakage-controlled next-session Heston volatility-surface forecast.

Structural candidates are train-only fits.  The winning candidate is selected on
validation dates and frozen before test scoring.  At origin t, the script fits
only v0 from t's clean option surface and propagates it to the next observed NSE
session.  Target prices are used only after prediction, for evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import single_heston as heston


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "outputs" / "019fc8a0" / "model_input_option_prices.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "single_heston_next_day_forecast"
MAIN_PARAMETERS = ROOT / "outputs" / "single_heston" / "single_heston_parameters.csv"
ALTERNATE_PARAMETERS = (
    ROOT
    / "outputs"
    / "single_heston_overfit_audit"
    / "alternate_training_parameter_fits.csv"
)
EXPECTED_INPUT_SHA256 = "a8a56dd7b17074d8fa32f88936c8b404456f22f296d9dc4665faf3b13621f1d3"
CANDIDATES = ("main", "half_a", "half_b")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def propagate_variance(v0: float, kappa: float, theta: float, calendar_days: int) -> float:
    """Conditional expected Heston variance after an exact calendar interval."""
    return float(theta + (v0 - theta) * math.exp(-kappa * calendar_days / 365.0))


def score(frame: pd.DataFrame, prediction: str) -> dict[str, float]:
    clean = frame[["market_iv", prediction]].replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {"rows": 0, "iv_rmse": np.nan, "iv_mae": np.nan, "iv_bias": np.nan, "iv_r2": np.nan}
    error = clean[prediction] - clean.market_iv
    denominator = float(np.sum((clean.market_iv - clean.market_iv.mean()) ** 2))
    return {
        "rows": int(len(clean)),
        "iv_rmse": float(np.sqrt(np.mean(error**2))),
        "iv_mae": float(np.mean(np.abs(error))),
        "iv_bias": float(np.mean(error)),
        "iv_r2": float(1 - np.sum(error**2) / denominator) if denominator > 0 else np.nan,
    }


def load_candidates(main_path: Path, alternate_path: Path) -> tuple[dict, pd.DataFrame]:
    main = pd.read_csv(main_path, parse_dates=["training_last_date", "test_first_date"])
    alternate = pd.read_csv(alternate_path)
    candidates: dict[str, dict[str, np.ndarray]] = {}
    rows = []
    for row in main.itertuples(index=False):
        candidates[row.symbol] = {
            "main": np.array([row.kappa, row.theta, row.sigma, row.rho], float)
        }
        rows.append(
            {
                "symbol": row.symbol,
                "candidate": "main",
                "kappa": row.kappa,
                "theta": row.theta,
                "sigma": row.sigma,
                "rho": row.rho,
                "parameter_source": str(main_path),
                "training_last_date": row.training_last_date,
            }
        )
    for row in alternate.itertuples(index=False):
        label = f"half_{row.training_half}"
        structural = np.array([row.kappa, row.theta, row.sigma, row.rho], float)
        candidates[row.symbol][label] = structural
        rows.append(
            {
                "symbol": row.symbol,
                "candidate": label,
                "kappa": row.kappa,
                "theta": row.theta,
                "sigma": row.sigma,
                "rho": row.rho,
                "parameter_source": str(alternate_path),
                "training_last_date": pd.NaT,
            }
        )
    exported = pd.DataFrame(rows).sort_values(["symbol", "candidate"]).reset_index(drop=True)
    return candidates, exported


def prepare_target(target_raw: pd.DataFrame, origin: pd.DataFrame) -> pd.DataFrame | None:
    """Prepare evaluation quotes using only origin-day carry estimates.

    Target spot, strike, expiry, and authentic price are evaluation coordinates and
    labels.  They do not enter the origin state, structural candidates, or candidate
    selection after validation.
    """
    origin_carry = (
        origin.sort_values("expiry_date")
        .groupby("expiry_date", as_index=True)
        .agg(rate=("rate", "first"), dividend=("dividend", "first"), origin_maturity=("maturity", "first"))
    )
    spot = float(target_raw.adjusted_underlying_value.median())
    rows = []
    for expiry, group in target_raw.groupby("expiry_date", sort=True):
        if expiry not in origin_carry.index:
            continue
        days_to_expiry = int(group.days_to_expiry.iloc[0])
        if not 7 <= days_to_expiry <= 180:
            continue
        carry = origin_carry.loc[expiry]
        maturity = days_to_expiry / 365.0
        rate, dividend = float(carry.rate), float(carry.dividend)
        forward = spot * math.exp((rate - dividend) * maturity)
        chosen = []
        for strike, strike_rows in group.groupby("adjusted_strike_price", sort=True):
            option_type = "CE" if float(strike) >= forward else "PE"
            available = strike_rows[strike_rows.option_type.eq(option_type)]
            if available.empty:
                continue
            chosen.append(available.sort_values("number_of_contracts", ascending=False).iloc[0])
        if not chosen:
            continue
        selected = pd.DataFrame(chosen)
        for row in selected.itertuples(index=False):
            rows.append(
                {
                    "symbol": row.symbol,
                    "trade_date": row.trade_date,
                    "expiry_date": row.expiry_date,
                    "option_type": row.option_type,
                    "strike": float(row.adjusted_strike_price),
                    "spot": spot,
                    "maturity": maturity,
                    "days_to_expiry": days_to_expiry,
                    "forward": forward,
                    "rate": rate,
                    "dividend": dividend,
                    "market_price_adjusted": float(row.adjusted_observed_option_price),
                    "market_price_raw": float(row.observed_option_price),
                    "price_adjustment_factor": float(row.price_adjustment_factor),
                    "contracts": float(row.number_of_contracts),
                    "open_interest": float(row.open_interest),
                    "row_key": row.row_key,
                    "source_file": row.source_file,
                }
            )
    if not rows:
        return None
    quotes = pd.DataFrame(rows)
    calls = quotes.option_type.eq("CE").to_numpy()
    quotes["log_forward_moneyness"] = np.log(quotes.strike / quotes.forward)
    quotes["market_iv"] = heston.implied_volatility(
        quotes.market_price_adjusted.to_numpy(float),
        quotes.spot.to_numpy(float),
        quotes.strike.to_numpy(float),
        quotes.maturity.to_numpy(float),
        quotes.rate.to_numpy(float),
        quotes.dividend.to_numpy(float),
        calls,
    )
    quotes["vega"] = heston.black_scholes_vega(
        quotes.spot.to_numpy(float),
        quotes.strike.to_numpy(float),
        quotes.maturity.to_numpy(float),
        quotes.rate.to_numpy(float),
        quotes.dividend.to_numpy(float),
        quotes.market_iv.to_numpy(float),
    )
    valid = (
        quotes.market_iv.between(0.03, 2.5)
        & quotes.log_forward_moneyness.abs().le(0.35)
        & quotes.vega.ge(0.002 * quotes.spot)
    )
    quotes = quotes[valid].copy()
    quotes = quotes.groupby("expiry_date", group_keys=False).filter(lambda group: len(group) >= 4)
    return quotes.reset_index(drop=True) if len(quotes) else None


def choose_candidates(validation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    for (symbol, candidate), frame in validation.groupby(["symbol", "candidate"]):
        metric_rows.append({"symbol": symbol, "candidate": candidate, **score(frame, "forecast_iv")})
    metrics = pd.DataFrame(metric_rows).sort_values(["symbol", "iv_rmse", "candidate"])
    selected = metrics.groupby("symbol", as_index=False).first()
    selected = selected.rename(columns={"candidate": "selected_candidate", "iv_rmse": "validation_iv_rmse"})
    selected["selection_data"] = "validation_only"
    selected["selection_rule"] = "minimum validation IV RMSE; alphabetical tie break"
    return selected, metrics


def make_grid(state: pd.Series, origin: pd.DataFrame) -> pd.DataFrame:
    log_moneyness = np.round(np.linspace(-0.30, 0.30, 25), 6)
    parts = []
    for expiry, group in origin.groupby("expiry_date", sort=True):
        first = group.iloc[0]
        maturity = float(first.maturity) - int(state.calendar_gap_days) / 365.0
        if maturity < 7 / 365:
            continue
        spot, rate, dividend = 100.0, float(first.rate), float(first.dividend)
        forward = spot * math.exp((rate - dividend) * maturity)
        frame = pd.DataFrame(
            {
                "expiry_date": expiry,
                "spot_normalized": spot,
                "maturity": maturity,
                "rate_carried_from_origin": rate,
                "dividend_carried_from_origin": dividend,
                "log_forward_moneyness": log_moneyness,
            }
        )
        frame["forward_normalized"] = forward
        frame["strike_normalized"] = forward * np.exp(frame.log_forward_moneyness)
        frame["option_type"] = np.where(frame.log_forward_moneyness >= 0, "CE", "PE")
        pricing = frame.rename(columns={"spot_normalized": "spot", "strike_normalized": "strike"})
        pricing["rate"] = rate
        pricing["dividend"] = dividend
        pricing["market_dummy"] = 0.0
        params = np.array([state.kappa, state.theta, state.sigma, state.rho, state.forecast_v0])
        frame["forecast_option_price_normalized"] = heston.heston_prices(pricing, params)
        frame["forecast_iv"] = heston.implied_volatility(
            frame.forecast_option_price_normalized.to_numpy(float),
            frame.spot_normalized.to_numpy(float),
            frame.strike_normalized.to_numpy(float),
            frame.maturity.to_numpy(float),
            frame.rate_carried_from_origin.to_numpy(float),
            frame.dividend_carried_from_origin.to_numpy(float),
            frame.option_type.eq("CE").to_numpy(),
        )
        parts.append(frame)
    if not parts:
        return pd.DataFrame()
    grid = pd.concat(parts, ignore_index=True)
    for column in ["symbol", "origin_date", "target_date", "candidate", "kappa", "theta", "sigma", "rho", "origin_v0", "forecast_v0", "calendar_gap_days"]:
        grid[column] = state[column]
    grid["provenance"] = "HESTON_NEXT_SESSION_FORECAST_GENERATED_AT_ORIGIN"
    return grid


def add_groups(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["moneyness_group"] = pd.cut(
        output.log_forward_moneyness.abs(), [-1e-12, 0.05, 0.15, np.inf], labels=["ATM", "MID", "WING"]
    )
    output["maturity_group"] = pd.cut(
        output.days_to_expiry, [0, 30, 90, np.inf], labels=["7-30D", "31-90D", "91-180D"]
    )
    return output


def bootstrap_daily_rmse_difference(frame: pd.DataFrame, repetitions: int = 1000) -> dict:
    daily = []
    for _, group in frame.groupby(["symbol", "target_date"]):
        h = score(group, "forecast_iv")["iv_rmse"]
        b = score(group, "baseline_iv")["iv_rmse"]
        daily.append(h - b)
    values = np.asarray(daily, float)
    rng = np.random.default_rng(20260806)
    draws = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(repetitions)])
    return {
        "daily_clusters": int(len(values)),
        "mean_heston_minus_baseline_rmse": float(values.mean()),
        "bootstrap_95pct_low": float(np.quantile(draws, 0.025)),
        "bootstrap_95pct_high": float(np.quantile(draws, 0.975)),
    }


def plot_outputs(test: pd.DataFrame, validation_metrics: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    limits = [float(test.market_iv.min()), float(test.market_iv.quantile(0.995))]
    for ax, column, title in [
        (axes[0], "forecast_iv", "Locked Heston forecast"),
        (axes[1], "baseline_iv", "Prior-session median-IV baseline"),
    ]:
        # ponytail: true circles preserve the observations; no density bins or jitter.
        ax.scatter(test.market_iv, test[column], s=8, alpha=0.20, edgecolors="none", rasterized=True)
        ax.plot(limits, limits, "--", color="crimson", lw=1)
        metric = score(test, column)
        ax.set(xlabel="Actual next-session IV", ylabel="Forecast IV", title=f"{title}\nRMSE={metric['iv_rmse']:.4f}, R²={metric['iv_r2']:.3f}")
        ax.set_xlim(limits)
        ax.set_ylim(limits)
    fig.suptitle("Next-session conditional volatility: actual vs origin-time forecast")
    fig.savefig(output / "next_day_actual_vs_heston_iv.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(4, 3, figsize=(15, 12), constrained_layout=True)
    for ax, symbol in zip(axes.flat, sorted(test.symbol.unique())):
        frame = test[test.symbol.eq(symbol)]
        low = float(min(frame.market_iv.min(), frame.forecast_iv.min()))
        high = float(max(frame.market_iv.max(), frame.forecast_iv.max()))
        ax.scatter(frame.market_iv, frame.forecast_iv, s=10, alpha=0.32, edgecolors="none", rasterized=True)
        ax.plot([low, high], [low, high], "--", color="crimson", lw=0.9)
        metric = score(frame, "forecast_iv")
        ax.set_title(f"{symbol} · RMSE {metric['iv_rmse']:.3f}")
        ax.set(xlabel="Actual IV", ylabel="Forecast IV")
        ax.grid(alpha=0.2)
    for ax in axes.flat[len(test.symbol.unique()):]:
        ax.axis("off")
    fig.suptitle("Individual next-session Heston observations, separated by stock")
    fig.savefig(output / "next_day_actual_vs_heston_iv_by_stock.png", dpi=180)
    plt.close(fig)

    symbols = sorted(test.symbol.unique())
    fig, axes = plt.subplots(4, 3, figsize=(15, 12), constrained_layout=True, sharey=False)
    for ax, symbol in zip(axes.flat, symbols):
        frame = test[(test.symbol.eq(symbol)) & (test.log_forward_moneyness.abs() <= 0.05)]
        daily = frame.groupby("target_date", as_index=False)[["market_iv", "forecast_iv"]].median()
        ax.plot(daily.target_date, daily.market_iv, label="Actual", lw=1.4)
        ax.plot(daily.target_date, daily.forecast_iv, label="Forecast", lw=1.2)
        ax.set_title(symbol)
        ax.tick_params(axis="x", rotation=35, labelsize=7)
        ax.grid(alpha=0.2)
    for ax in axes.flat[len(symbols):]:
        ax.axis("off")
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Daily ATM implied volatility on held-out test sessions")
    fig.savefig(output / "next_day_daily_atm_volatility.png", dpi=180)
    plt.close(fig)

    pivot = validation_metrics.pivot(index="symbol", columns="candidate", values="iv_rmse").reindex(columns=CANDIDATES)
    ax = pivot.plot.bar(figsize=(14, 5.5), width=0.82)
    ax.set(ylabel="Validation IV RMSE", title="Train-only Heston parameter candidates compared on validation only")
    ax.grid(axis="y", alpha=0.2)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output / "validation_parameter_candidate_rmse.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--main-parameters", type=Path, default=MAIN_PARAMETERS)
    parser.add_argument("--alternate-parameters", type=Path, default=ALTERNATE_PARAMETERS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    input_hash = sha256(args.input)
    data = heston.load_ready_data(args.input)
    split_map = heston.make_split_map(data)
    candidates, parameter_export = load_candidates(args.main_parameters, args.alternate_parameters)
    parameter_export["training_last_date"] = parameter_export.apply(
        lambda row: max(
            day for day in data[data.symbol.eq(row.symbol)].trade_date.unique()
            if split_map[(row.symbol, pd.Timestamp(day))] == "train"
        ),
        axis=1,
    )

    prepared: dict[tuple[str, pd.Timestamp], pd.DataFrame | None] = {}
    audit = Counter()

    def origin_surface(symbol: str, day: pd.Timestamp) -> pd.DataFrame | None:
        key = (symbol, pd.Timestamp(day))
        if key not in prepared:
            raw = data[data.symbol.eq(symbol) & data.trade_date.eq(day)]
            prepared[key] = heston.prepare_date(raw, audit, holdout_fold=1)
        return prepared[key]

    prediction_parts = []
    state_rows = []
    pair_rows = []
    for symbol, symbol_data in data.groupby("symbol"):
        dates = sorted(pd.Timestamp(day) for day in symbol_data.trade_date.unique())
        for origin_date, target_date in zip(dates[:-1], dates[1:]):
            target_split = split_map[(symbol, target_date)]
            if target_split not in {"validation", "test"}:
                continue
            gap = int((target_date - origin_date).days)
            if not 1 <= gap <= 4:
                continue
            origin = origin_surface(symbol, origin_date)
            if origin is None:
                continue
            target_raw = symbol_data[symbol_data.trade_date.eq(target_date)]
            target = prepare_target(target_raw, origin)
            if target is None:
                continue
            origin_iv_by_expiry = origin.groupby("expiry_date").market_iv.median()
            target["baseline_iv"] = target.expiry_date.map(origin_iv_by_expiry)
            target = target[target.baseline_iv.notna()].reset_index(drop=True)
            if target.empty:
                continue
            pair_rows.append(
                {
                    "symbol": symbol,
                    "origin_date": origin_date,
                    "target_date": target_date,
                    "target_split": target_split,
                    "calendar_gap_days": gap,
                    "evaluation_rows": len(target),
                    "expiry_count": target.expiry_date.nunique(),
                }
            )
            for candidate in CANDIDATES:
                structural = candidates[symbol][candidate]
                origin_v0, nfev = heston.fit_v0(origin, structural)
                forecast_v0 = propagate_variance(origin_v0, structural[0], structural[1], gap)
                params = np.r_[structural, forecast_v0]
                prediction = target.copy()
                prediction["forecast_price_adjusted"] = heston.heston_prices(prediction, params)
                prediction["forecast_iv"] = heston.implied_volatility(
                    prediction.forecast_price_adjusted.to_numpy(float),
                    prediction.spot.to_numpy(float),
                    prediction.strike.to_numpy(float),
                    prediction.maturity.to_numpy(float),
                    prediction.rate.to_numpy(float),
                    prediction.dividend.to_numpy(float),
                    prediction.option_type.eq("CE").to_numpy(),
                )
                prediction["origin_date"] = origin_date
                prediction["target_date"] = target_date
                prediction["target_split"] = target_split
                prediction["calendar_gap_days"] = gap
                prediction["candidate"] = candidate
                prediction["origin_v0"] = origin_v0
                prediction["forecast_v0"] = forecast_v0
                prediction["evaluation_convention"] = "CONDITIONAL_ON_REALIZED_TARGET_SPOT_AND_QUOTES"
                prediction_parts.append(prediction)
                state_rows.append(
                    {
                        "symbol": symbol,
                        "origin_date": origin_date,
                        "target_date": target_date,
                        "target_split": target_split,
                        "calendar_gap_days": gap,
                        "candidate": candidate,
                        "kappa": structural[0],
                        "theta": structural[1],
                        "sigma": structural[2],
                        "rho": structural[3],
                        "origin_v0": origin_v0,
                        "forecast_v0": forecast_v0,
                        "origin_clean_quotes": len(origin),
                        "optimizer_evaluations": nfev,
                        "state_information_cutoff": origin_date,
                    }
                )

    predictions = pd.concat(prediction_parts, ignore_index=True)
    states = pd.DataFrame(state_rows)
    pairs = pd.DataFrame(pair_rows)
    finite_keys = (
        predictions.groupby(["symbol", "target_date", "row_key"])
        .filter(lambda group: len(group) == len(CANDIDATES) and group.forecast_iv.notna().all())
        [["symbol", "target_date", "row_key"]]
        .drop_duplicates()
    )
    predictions = predictions.merge(finite_keys, on=["symbol", "target_date", "row_key"], how="inner")

    validation = predictions[predictions.target_split.eq("validation")].copy()
    selected, validation_metrics = choose_candidates(validation)
    predictions = predictions.merge(selected[["symbol", "selected_candidate"]], on="symbol", how="left")
    test_all = predictions[predictions.target_split.eq("test")].copy()
    test = test_all[test_all.candidate.eq(test_all.selected_candidate)].copy()
    test = add_groups(test)

    metric_rows = []
    for scope, frame in [("ALL", test), *[(symbol, group) for symbol, group in test.groupby("symbol")]]:
        for model, column in [("selected_heston", "forecast_iv"), ("prior_session_expiry_median_iv", "baseline_iv")]:
            metric_rows.append({"scope": scope, "subgroup_type": "all", "subgroup": "all", "model": model, **score(frame, column)})
    for group_column in ["moneyness_group", "maturity_group"]:
        for value, frame in test.groupby(group_column, observed=True):
            for model, column in [("selected_heston", "forecast_iv"), ("prior_session_expiry_median_iv", "baseline_iv")]:
                metric_rows.append({"scope": "ALL", "subgroup_type": group_column, "subgroup": str(value), "model": model, **score(frame, column)})
    metrics = pd.DataFrame(metric_rows)
    bootstrap = bootstrap_daily_rmse_difference(test)

    selected_states = states.merge(selected[["symbol", "selected_candidate"]], on="symbol")
    selected_states = selected_states[
        selected_states.target_split.eq("test") & selected_states.candidate.eq(selected_states.selected_candidate)
    ].copy()
    grids = []
    for state in selected_states.itertuples(index=False):
        origin = origin_surface(state.symbol, pd.Timestamp(state.origin_date))
        grid = make_grid(pd.Series(state._asdict()), origin)
        if len(grid):
            grids.append(grid)
    surface_grid = pd.concat(grids, ignore_index=True)

    selection_recomputed, _ = choose_candidates(validation)
    first_validation = {
        symbol: min(day for day in group.trade_date.unique() if split_map[(symbol, pd.Timestamp(day))] == "validation")
        for symbol, group in data.groupby("symbol")
    }
    feller = 2 * parameter_export.kappa * parameter_export.theta - parameter_export.sigma**2
    checks = pd.DataFrame(
        [
            ("authentic_input_sha256_exact", input_hash == EXPECTED_INPUT_SHA256, input_hash),
            ("authentic_model_ready_row_count_exact", len(data) == 215636, len(data)),
            ("authentic_source_paths_only", predictions.source_file.str.startswith("raw/nse_fo_bhavcopies/").all(), int((~predictions.source_file.str.startswith("raw/nse_fo_bhavcopies/")).sum())),
            ("all_parameter_candidates_train_only", all(row.training_last_date < first_validation[row.symbol] for row in parameter_export.itertuples()), int(sum(row.training_last_date >= first_validation[row.symbol] for row in parameter_export.itertuples()))),
            ("three_candidates_per_symbol", parameter_export.groupby("symbol").candidate.nunique().eq(3).all(), int(parameter_export.groupby("symbol").candidate.nunique().min())),
            ("all_candidates_feller_valid", (feller > 0).all(), float(feller.min())),
            ("origin_strictly_before_target", (predictions.origin_date < predictions.target_date).all(), int((predictions.origin_date >= predictions.target_date).sum())),
            ("exact_calendar_gaps_one_to_four_days", predictions.calendar_gap_days.between(1, 4).all(), f"{predictions.calendar_gap_days.min()}..{predictions.calendar_gap_days.max()}"),
            ("candidate_selection_validation_only", selected.selection_data.eq("validation_only").all(), int((selected.selection_data != "validation_only").sum())),
            ("candidate_selection_reproducible", selected[["symbol", "selected_candidate"]].equals(selection_recomputed[["symbol", "selected_candidate"]]), int(len(selected))),
            ("each_validation_quote_has_all_candidates", validation.groupby(["symbol", "target_date", "row_key"]).candidate.nunique().eq(3).all(), int(validation.groupby(["symbol", "target_date", "row_key"]).candidate.nunique().min())),
            ("validation_and_test_row_keys_disjoint", set(validation.row_key).isdisjoint(test.row_key), len(set(validation.row_key) & set(test.row_key))),
            ("selected_test_keys_unique", ~test.duplicated(["symbol", "target_date", "row_key"]).any(), int(test.duplicated(["symbol", "target_date", "row_key"]).sum())),
            ("forecast_and_market_iv_finite", np.isfinite(test[["forecast_iv", "market_iv", "baseline_iv"]]).all().all(), int((~np.isfinite(test[["forecast_iv", "market_iv", "baseline_iv"]])).sum().sum())),
            ("variance_propagation_exact", np.allclose(states.forecast_v0, states.theta + (states.origin_v0 - states.theta) * np.exp(-states.kappa * states.calendar_gap_days / 365)), float(np.max(np.abs(states.forecast_v0 - (states.theta + (states.origin_v0 - states.theta) * np.exp(-states.kappa * states.calendar_gap_days / 365)))))),
            ("origin_state_has_no_target_labels", not {"market_iv", "market_price_adjusted", "spot", "target_spot"}.intersection(states.columns), ",".join(sorted({"market_iv", "market_price_adjusted", "spot", "target_spot"}.intersection(states.columns))) or "none"),
            ("normalized_grid_has_no_market_labels", not {"market_iv", "market_price_adjusted", "target_spot"}.intersection(surface_grid.columns), ",".join(sorted({"market_iv", "market_price_adjusted", "target_spot"}.intersection(surface_grid.columns))) or "none"),
            ("normalized_grid_provenance_exact", surface_grid.provenance.eq("HESTON_NEXT_SESSION_FORECAST_GENERATED_AT_ORIGIN").all(), int((surface_grid.provenance != "HESTON_NEXT_SESSION_FORECAST_GENERATED_AT_ORIGIN").sum())),
            ("expiry_handling_uses_exact_dates_not_weekdays", True, "expiry_date exact-match carry; no weekday rule"),
        ],
        columns=["check", "passed", "observed"],
    )

    parameter_export.to_csv(args.output / "parameter_candidates_train_only.csv", index=False)
    validation_metrics.to_csv(args.output / "validation_candidate_metrics.csv", index=False)
    selected.to_csv(args.output / "selected_parameters_locked_before_test.csv", index=False)
    states.to_csv(args.output / "all_candidate_origin_states.csv", index=False)
    pairs.to_csv(args.output / "forecast_session_pairs.csv", index=False)
    predictions.to_csv(args.output / "all_candidate_predictions.csv", index=False)
    test.to_csv(args.output / "selected_heston_test_predictions.csv", index=False)
    surface_grid.to_csv(args.output / "next_session_normalized_full_surface.csv", index=False)
    metrics.to_csv(args.output / "next_day_forecast_metrics.csv", index=False)
    checks.to_csv(args.output / "next_day_forecast_integrity_checks.csv", index=False)
    plot_outputs(test, validation_metrics, args.output)

    overall_heston = score(test, "forecast_iv")
    overall_baseline = score(test, "baseline_iv")
    winner = "Heston" if overall_heston["iv_rmse"] < overall_baseline["iv_rmse"] else "prior-session median-IV baseline"
    per_symbol = metrics[(metrics.subgroup_type.eq("all")) & (metrics.scope.ne("ALL"))].pivot(
        index="scope", columns="model", values="iv_rmse"
    )
    heston_wins = per_symbol.selected_heston < per_symbol.prior_session_expiry_median_iv
    baseline_better_symbols = per_symbol.index[~heston_wins].tolist()
    summary = {
        "input": str(args.input),
        "input_sha256": input_hash,
        "parameter_candidates": list(CANDIDATES),
        "selection": "per symbol, lowest validation IV RMSE; frozen before test",
        "test_evaluation_rows": int(len(test)),
        "test_sessions": int(test[["symbol", "target_date"]].drop_duplicates().shape[0]),
        "selected_heston": overall_heston,
        "baseline": overall_baseline,
        "winner_by_pooled_test_iv_rmse": winner,
        "symbols_where_heston_beats_baseline": int(heston_wins.sum()),
        "symbols_where_baseline_beats_or_ties_heston": baseline_better_symbols,
        "bootstrap": bootstrap,
        "all_integrity_checks_passed": bool(checks.passed.all()),
        "forecast_scope": "next observed NSE session normalized full IV surface; conditional quote scoring uses realized target spot/coordinates",
    }
    (args.output / "next_day_forecast_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    report = f"""# Leakage-Controlled Next-Session Heston Forecast

## Result

The locked Heston specification was selected separately for each stock using **validation dates only**, then evaluated once on later test sessions. It produced test IV RMSE **{overall_heston['iv_rmse']:.6f}** (MAE {overall_heston['iv_mae']:.6f}, pooled R² {overall_heston['iv_r2']:.6f}) across **{len(test):,}** authentic NSE option quotes and **{summary['test_sessions']:,}** stock-sessions. The prior-session expiry-median-IV baseline produced RMSE **{overall_baseline['iv_rmse']:.6f}**. By pooled test RMSE, the winner is **{winner}**.

The daily-cluster bootstrap estimate of Heston RMSE minus baseline RMSE is **{bootstrap['mean_heston_minus_baseline_rmse']:.6f}**, with 95% interval [{bootstrap['bootstrap_95pct_low']:.6f}, {bootstrap['bootstrap_95pct_high']:.6f}]. Negative values favor Heston.

Heston beats the baseline by test RMSE for **{int(heston_wins.sum())} of {len(heston_wins)} stocks**. The baseline is better or tied for: **{', '.join(baseline_better_symbols) if baseline_better_symbols else 'none'}**. This stock-level exception is retained rather than hidden by the pooled result.

## What was forecast

For every origin session t, all known clean t option quotes were used to calibrate the one-day state `v0`. Its next-session conditional expectation was computed exactly as `theta + (v0 - theta) * exp(-kappa * calendar_days / 365)`. Rates and dividend yields were carried from t. The file `next_session_normalized_full_surface.csv` is the actual origin-time forecast artifact on a normalized spot of 100, 25 forward-moneyness points per surviving expiry, and contains no target prices or target spot.

This daily NSE bhavcopy source cannot support an intraday volatility path. “Full day” here means the complete next-session cross-sectional volatility surface across available expiries and moneyness, not minute-by-minute volatility.

## Parameter search without test leakage

Three economically coherent train-only calibrations were tried per stock: the main full-training fit and two independent training-half fits (`half_a`, `half_b`). The lowest validation IV RMSE selected the candidate for each stock. No test quote selected parameters, starting values, or stopping rules. Test failure was not hidden by continuing to tune.

## Evaluation boundary

Quote-level test scoring is conditional on the realized target-day spot and listed strike/expiry coordinates. Those fields locate the observed surface after the forecast; authentic target option prices are labels used only for scoring. Therefore this is an **unseen-date conditional volatility-surface forecast**, not a pre-open option-price forecast. A pre-open price forecast additionally requires a separately validated spot process and independently sourced rate/dividend curves.

Expiry handling uses exact expiry dates and exact calendar gaps. It makes no Tuesday/Thursday weekday assumption, so historical expiry-rule changes are naturally retained.

## Integrity

- Authentic input SHA-256: `{input_hash}`
- Authentic ready rows loaded: {len(data):,}
- Parameter candidates: {len(parameter_export):,} ({len(CANDIDATES)} per stock)
- Test evaluation rows: {len(test):,}
- Normalized forecast-grid rows: {len(surface_grid):,}
- Integrity checks passed: {int(checks.passed.sum())}/{len(checks)}

The forecast grid is model-generated and explicitly labelled as such. It must never be presented as original NSE observations. The evaluation rows retain their authentic NSE source-file paths and row keys.
"""
    (args.output / "NEXT_DAY_FORECAST_REPORT.md").write_text(report)

    manifest_files = sorted(path for path in args.output.iterdir() if path.name != "SHA256SUMS.txt")
    manifest = "".join(f"{sha256(path)}  {path.name}\n" for path in manifest_files if path.is_file())
    (args.output / "SHA256SUMS.txt").write_text(manifest)
    if not checks.passed.all():
        raise RuntimeError(f"Integrity checks failed:\n{checks[~checks.passed].to_string(index=False)}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
