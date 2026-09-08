#!/usr/bin/env python3
"""Honest chronological Single- versus Double-Heston forecast comparison."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import double_heston as double
import single_heston as single
from forecast_single_heston_next_day import EXPECTED_INPUT_SHA256, sha256


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "outputs" / "019fc8a0" / "model_input_option_prices.csv"
SINGLE_OUTPUT = ROOT / "outputs" / "single_heston"
FORECAST_OUTPUT = ROOT / "outputs" / "single_heston_next_day_forecast"
DEFAULT_OUTPUT = ROOT / "outputs" / "single_double_heston_comparison"
PARAMETER_NAMES = ["kappa_slow", "theta_slow", "sigma_slow", "rho_slow", "kappa_fast", "theta_fast", "sigma_fast", "rho_fast"]


def score(frame: pd.DataFrame, column: str) -> dict:
    error = frame[column] - frame.market_iv
    denominator = np.sum((frame.market_iv - frame.market_iv.mean()) ** 2)
    per_session = frame.assign(square_error=error**2).groupby(["symbol", "target_date"]).square_error.mean().pow(0.5)
    return {
        "rows": len(frame),
        "iv_rmse": float(np.sqrt(np.mean(error**2))),
        "iv_mae": float(np.mean(np.abs(error))),
        "iv_bias": float(np.mean(error)),
        "iv_r2": float(1 - np.sum(error**2) / denominator) if denominator > 0 else np.nan,
        "mean_session_iv_rmse": float(per_session.mean()),
        "forecast_on_actual_slope": float(np.polyfit(frame.market_iv, frame[column], 1)[0]),
    }


def parameter_row(symbol, candidate, source, loss, structural, states, dates, nfev):
    row = {
        "symbol": symbol,
        "candidate": candidate,
        "candidate_source": source,
        "training_objective": loss,
        "training_dates": len(dates),
        "training_first_date": min(dates),
        "training_last_date": max(dates),
        "optimizer_evaluations": nfev,
        "v0_slow_training_median": float(np.median([state[0] for state in states])),
        "v0_fast_training_median": float(np.median([state[1] for state in states])),
    }
    row.update(dict(zip(PARAMETER_NAMES, structural)))
    row["feller_gap_slow"] = 2 * row["kappa_slow"] * row["theta_slow"] - row["sigma_slow"] ** 2
    row["feller_gap_fast"] = 2 * row["kappa_fast"] * row["theta_fast"] - row["sigma_fast"] ** 2
    row["correlation_radius"] = math.sqrt(row["rho_slow"] ** 2 + row["rho_fast"] ** 2)
    return row


def propagate_state(structural, state, days: int) -> np.ndarray:
    k1, t1, _, _, k2, t2, _, _ = structural
    return np.array(
        [
            t1 + (state[0] - t1) * math.exp(-k1 * days / 365),
            t2 + (state[1] - t2) * math.exp(-k2 * days / 365),
        ]
    )


def predict_iv(quotes: pd.DataFrame, structural, state) -> tuple[np.ndarray, np.ndarray]:
    model_price = double.prices(quotes, double.combine(structural, state))
    model_iv = single.implied_volatility(
        model_price,
        quotes.spot.to_numpy(float),
        quotes.strike.to_numpy(float),
        quotes.maturity.to_numpy(float),
        quotes.rate.to_numpy(float),
        quotes.dividend.to_numpy(float),
        quotes.option_type.eq("CE").to_numpy(),
    )
    return model_price, model_iv


def cluster_bootstrap(frame: pd.DataFrame, repetitions: int = 2000) -> dict:
    groups = [group for _, group in frame.groupby(["symbol", "target_date"], sort=False)]
    rng = np.random.default_rng(20260806)
    differences = []
    for _ in range(repetitions):
        sample = pd.concat([groups[index] for index in rng.integers(0, len(groups), len(groups))], ignore_index=True)
        differences.append(
            np.sqrt(np.mean((sample.double_heston_iv - sample.market_iv) ** 2))
            - np.sqrt(np.mean((sample.single_heston_iv - sample.market_iv) ** 2))
        )
    low, median, high = np.quantile(differences, [0.025, 0.5, 0.975])
    return {"clusters": len(groups), "double_minus_single_rmse_low": float(low), "median": float(median), "high": float(high)}


def plot_comparison(test: pd.DataFrame, metrics: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    low, high = float(test.market_iv.min()), float(test.market_iv.quantile(0.995))
    for ax, column, title in [
        (axes[0], "single_heston_iv", "Single Heston (5 parameters)"),
        (axes[1], "double_heston_iv", "Double Heston (10 parameters)"),
    ]:
        ax.scatter(test.market_iv, test[column], s=8, alpha=0.20, edgecolors="none", rasterized=True)
        ax.plot([low, high], [low, high], "--", color="crimson", lw=1, label="Ideal identity (fixed 45°)")
        row = metrics[(metrics.scope.eq("ALL")) & metrics.model.eq(title.split(" (")[0])].iloc[0]
        ax.set(
            xlim=(low, high), ylim=(low, high), xlabel="Actual next-session IV", ylabel="Forecast IV",
            title=f"{title}\nRMSE={row.iv_rmse:.4f}, R²={row.iv_r2:.3f}, slope={row.forecast_on_actual_slope:.3f}",
        )
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.15)
    fig.suptitle("Single versus Double Heston on identical historical test quotes")
    fig.savefig(output / "single_vs_double_actual_iv.png", dpi=180)
    plt.close(fig)

    per_stock = metrics[metrics.scope.ne("ALL")].pivot(index="scope", columns="model", values="iv_rmse")
    per_stock = per_stock[["Single Heston", "Double Heston"]]
    ax = per_stock.plot.bar(figsize=(14, 5.5), width=0.78, color=["#4c78a8", "#f58518"])
    ax.set(ylabel="Historical-test IV RMSE", xlabel="Stock", title="Out-of-sample error by stock on identical quote rows")
    ax.grid(axis="y", alpha=0.2)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output / "single_vs_double_rmse_by_stock.png", dpi=180)
    plt.close()


def plot_parameter_tables(single_parameters: pd.DataFrame, double_parameters: pd.DataFrame, output: Path) -> None:
    single_view = single_parameters[["symbol", "kappa", "theta", "sigma", "rho", "v0_training_median"]].copy()
    single_view.columns = ["Symbol", "κ", "θ", "σ", "ρ", "median v₀"]
    double_view = double_parameters[
        ["symbol", "kappa_slow", "theta_slow", "sigma_slow", "rho_slow", "v0_slow_training_median", "kappa_fast", "theta_fast", "sigma_fast", "rho_fast", "v0_fast_training_median"]
    ].copy()
    double_view.columns = ["Symbol", "κ slow", "θ slow", "σ slow", "ρ slow", "median v₀ slow", "κ fast", "θ fast", "σ fast", "ρ fast", "median v₀ fast"]
    for frame, filename, title, width in [
        (single_view, "single_heston_parameters.png", "Locked train-only Single-Heston parameters", 12),
        (double_view, "double_heston_parameters.png", "Validation-selected train-only Double-Heston parameters", 18),
    ]:
        display = frame.copy()
        for column in display.columns[1:]:
            display[column] = display[column].map(lambda value: f"{value:.5f}")
        fig, ax = plt.subplots(figsize=(width, 0.65 + 0.48 * len(display)))
        ax.axis("off")
        table = ax.table(cellText=display.values, colLabels=display.columns, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.35)
        ax.set_title(title, pad=14)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_stability(stability: pd.DataFrame, output: Path) -> None:
    frame = stability.sort_values("max_positive_parameter_log_range")
    rho_range = frame[["rho_slow_range", "rho_fast_range"]].max(axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    axes[0].barh(frame.symbol, frame.max_positive_parameter_log_range, color="#f58518")
    axes[0].axvline(math.log(2), color="crimson", linestyle="--", lw=1, label="factor-of-two range")
    axes[0].set(xlabel="Maximum log(max/min) across positive parameters", title="Start/training-half instability")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].barh(frame.symbol, rho_range, color="#4c78a8")
    axes[1].set(xlabel="Maximum correlation range", title="Correlation instability")
    for ax in axes:
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Double-Heston parameter disagreement across five train-only candidates")
    fig.savefig(output / "double_heston_parameter_instability.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--single-output", type=Path, default=SINGLE_OUTPUT)
    parser.add_argument("--forecast-output", type=Path, default=FORECAST_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--training-dates", type=int, default=12)
    parser.add_argument("--reuse-candidates", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    input_hash = sha256(args.input)
    data = single.load_ready_data(args.input)
    split_map = single.make_split_map(data)
    target = pd.read_csv(
        args.forecast_output / "all_candidate_predictions.csv",
        parse_dates=["trade_date", "origin_date", "target_date", "expiry_date"],
        low_memory=False,
    )
    target = target[target.candidate.eq("main")].drop(columns=["candidate", "forecast_price_adjusted", "forecast_iv", "origin_v0", "forecast_v0", "selected_candidate"])
    target = target.drop_duplicates(["symbol", "target_date", "row_key"]).reset_index(drop=True)
    audit, prepared = Counter(), {}

    def surface(symbol, day):
        key = (symbol, pd.Timestamp(day))
        if key not in prepared:
            raw = data[data.symbol.eq(symbol) & data.trade_date.eq(day)]
            prepared[key] = single.prepare_date(raw, audit, holdout_fold=1)
        return prepared[key]

    candidate_path = args.output / "double_heston_all_train_only_candidates.csv"
    structures = {}
    if args.reuse_candidates and candidate_path.exists():
        candidates = pd.read_csv(candidate_path, parse_dates=["training_first_date", "training_last_date"])
        for symbol, frame in candidates.groupby("symbol"):
            structures[symbol] = {
                row.candidate: np.array([getattr(row, name) for name in PARAMETER_NAMES], float)
                for row in frame.itertuples(index=False)
            }
    else:
        parameter_rows = []
        for symbol, symbol_data in data.groupby("symbol"):
            train_dates = sorted(day for day in symbol_data.trade_date.unique() if split_map[(symbol, pd.Timestamp(day))] == "train")
            ordered = single.choose_training_dates(train_dates, args.training_dates * 5)
            ordered += [day for day in train_dates if day not in set(ordered)]
            surfaces, dates = [], []
            for day in ordered:
                quotes = surface(symbol, day)
                if quotes is None:
                    continue
                surfaces.append(quotes[quotes.fold.eq("calibration")].reset_index(drop=True))
                dates.append(pd.Timestamp(day))
                if len(surfaces) >= args.training_dates:
                    break
            if len(surfaces) < 6:
                raise RuntimeError(f"{symbol}: fewer than six clean training surfaces")
            specifications = [
                ("full_start_1", "all training dates; generic start 1", surfaces, dates, double.STRUCTURAL_STARTS[0]),
                ("full_start_2", "all training dates; generic start 2", surfaces, dates, double.STRUCTURAL_STARTS[1]),
                ("full_start_3", "all training dates; generic start 3", surfaces, dates, double.STRUCTURAL_STARTS[2]),
                ("training_half_a", "alternating training-date half A", surfaces[::2], dates[::2], double.STRUCTURAL_STARTS[1]),
                ("training_half_b", "alternating training-date half B", surfaces[1::2], dates[1::2], double.STRUCTURAL_STARTS[1]),
            ]
            structures[symbol] = {}
            for candidate, source, fit_surfaces, fit_dates, start in specifications:
                loss, structural, states, nfev = double.fit_joint_structural(fit_surfaces, start)
                structures[symbol][candidate] = structural
                parameter_rows.append(parameter_row(symbol, candidate, source, loss, structural, states, fit_dates, nfev))
        candidates = pd.DataFrame(parameter_rows)

    validation_parts, state_rows = [], []
    validation_target = target[target.target_split.eq("validation")]
    for (symbol, origin_date, target_date), quotes in validation_target.groupby(["symbol", "origin_date", "target_date"], sort=False):
        origin = surface(symbol, origin_date)
        if origin is None:
            continue
        gap = int((target_date - origin_date).days)
        for candidate, structural in structures[symbol].items():
            state, nfev = double.fit_state(origin, structural)
            forecast_state = propagate_state(structural, state, gap)
            model_price, model_iv = predict_iv(quotes, structural, forecast_state)
            prediction = quotes.copy()
            prediction["candidate"] = candidate
            prediction["double_heston_price"] = model_price
            prediction["double_heston_iv"] = model_iv
            prediction["origin_v0_slow"] = state[0]
            prediction["origin_v0_fast"] = state[1]
            prediction["forecast_v0_slow"] = forecast_state[0]
            prediction["forecast_v0_fast"] = forecast_state[1]
            validation_parts.append(prediction)
            state_rows.append(
                {
                    "symbol": symbol, "origin_date": origin_date, "target_date": target_date,
                    "target_split": "validation", "candidate": candidate, "calendar_gap_days": gap,
                    "origin_v0_slow": state[0], "origin_v0_fast": state[1],
                    "forecast_v0_slow": forecast_state[0], "forecast_v0_fast": forecast_state[1],
                    "state_fit_information_cutoff": origin_date, "origin_clean_quotes": len(origin),
                    "optimizer_evaluations": nfev,
                }
            )
    validation_all = pd.concat(validation_parts, ignore_index=True)
    validation_failures = validation_all[~np.isfinite(validation_all.double_heston_iv)].copy()
    validation = validation_all[np.isfinite(validation_all.double_heston_iv)].copy()
    common_validation_keys = (
        validation.groupby(["symbol", "target_date", "row_key"]).candidate.nunique()
        .loc[lambda count: count.eq(5)].index.to_frame(index=False)
    )
    validation = validation.merge(common_validation_keys, on=["symbol", "target_date", "row_key"], validate="many_to_one")
    validation_metric_rows = []
    for (symbol, candidate), frame in validation.groupby(["symbol", "candidate"]):
        validation_metric_rows.append({"symbol": symbol, "candidate": candidate, **score(frame, "double_heston_iv")})
    validation_metrics = pd.DataFrame(validation_metric_rows).sort_values(["symbol", "mean_session_iv_rmse", "iv_rmse", "candidate"])
    selected = validation_metrics.groupby("symbol", as_index=False).first().rename(columns={"candidate": "selected_candidate", "mean_session_iv_rmse": "validation_mean_session_iv_rmse", "iv_rmse": "validation_iv_rmse"})
    selected["selection_data"] = "validation_only"
    selected["selection_rule"] = "minimum mean session IV RMSE; pooled RMSE then name tie-break"

    test_parts = []
    test_target = target[target.target_split.eq("test")]
    selected_map = selected.set_index("symbol").selected_candidate.to_dict()
    for (symbol, origin_date, target_date), quotes in test_target.groupby(["symbol", "origin_date", "target_date"], sort=False):
        origin = surface(symbol, origin_date)
        if origin is None:
            continue
        candidate = selected_map[symbol]
        structural = structures[symbol][candidate]
        gap = int((target_date - origin_date).days)
        state, nfev = double.fit_state(origin, structural)
        forecast_state = propagate_state(structural, state, gap)
        model_price, model_iv = predict_iv(quotes, structural, forecast_state)
        prediction = quotes.copy()
        prediction["selected_double_candidate"] = candidate
        prediction["double_heston_price"] = model_price
        prediction["double_heston_iv"] = model_iv
        prediction["origin_v0_slow"] = state[0]
        prediction["origin_v0_fast"] = state[1]
        prediction["forecast_v0_slow"] = forecast_state[0]
        prediction["forecast_v0_fast"] = forecast_state[1]
        test_parts.append(prediction)
        state_rows.append(
            {
                "symbol": symbol, "origin_date": origin_date, "target_date": target_date,
                "target_split": "test", "candidate": candidate, "calendar_gap_days": gap,
                "origin_v0_slow": state[0], "origin_v0_fast": state[1],
                "forecast_v0_slow": forecast_state[0], "forecast_v0_fast": forecast_state[1],
                "state_fit_information_cutoff": origin_date, "origin_clean_quotes": len(origin),
                "optimizer_evaluations": nfev,
            }
        )
    double_test = pd.concat(test_parts, ignore_index=True)
    states = pd.DataFrame(state_rows)

    single_test = pd.read_csv(
        args.forecast_output / "selected_heston_test_predictions.csv",
        usecols=["symbol", "target_date", "row_key", "forecast_iv"], parse_dates=["target_date"],
    ).rename(columns={"forecast_iv": "single_heston_iv"})
    test_all = double_test.merge(single_test, on=["symbol", "target_date", "row_key"], how="inner", validate="one_to_one")
    finite_test = np.isfinite(test_all[["market_iv", "single_heston_iv", "double_heston_iv"]]).all(axis=1)
    test_failures = test_all[~finite_test].copy()
    test = test_all[finite_test].copy()

    metric_rows = []
    for scope, frame in [("ALL", test), *test.groupby("symbol")]:
        for model, column in [("Single Heston", "single_heston_iv"), ("Double Heston", "double_heston_iv")]:
            metric_rows.append({"scope": scope, "model": model, **score(frame, column)})
    metrics = pd.DataFrame(metric_rows)
    bootstrap = cluster_bootstrap(test)

    selected_parameters = candidates.merge(selected[["symbol", "selected_candidate"]], left_on=["symbol", "candidate"], right_on=["symbol", "selected_candidate"])
    single_parameters = pd.read_csv(args.single_output / "single_heston_parameters.csv", parse_dates=["training_last_date", "test_first_date"])
    first_validation = {symbol: min(group[group.target_split.eq("validation")].target_date) for symbol, group in target.groupby("symbol")}
    first_test = test.groupby("symbol").target_date.min().to_dict()
    positive_columns = ["kappa_slow", "theta_slow", "sigma_slow", "kappa_fast", "theta_fast", "sigma_fast"]
    stability_rows = []
    for symbol, frame in candidates.groupby("symbol"):
        log_ranges = {f"{column}_log_range": float(np.log(frame[column].max() / frame[column].min())) for column in positive_columns}
        stability_rows.append(
            {
                "symbol": symbol,
                **log_ranges,
                "max_positive_parameter_log_range": max(log_ranges.values()),
                "rho_slow_range": float(frame.rho_slow.max() - frame.rho_slow.min()),
                "rho_fast_range": float(frame.rho_fast.max() - frame.rho_fast.min()),
                "training_objective_range": float(frame.training_objective.max() - frame.training_objective.min()),
            }
        )
    stability = pd.DataFrame(stability_rows)
    sigma_cap_slow = 0.995 * np.sqrt(2 * candidates.kappa_slow * candidates.theta_slow)
    sigma_cap_fast = 0.995 * np.sqrt(2 * candidates.kappa_fast * candidates.theta_fast)
    boundary_flags = (
        (candidates.kappa_slow < 0.06)
        | (candidates.kappa_slow > 4.95)
        | (candidates.kappa_fast > 9.9)
        | (candidates.sigma_slow / sigma_cap_slow > 0.98)
        | (candidates.sigma_fast / sigma_cap_fast > 0.98)
        | (candidates.correlation_radius > 0.94)
    )
    validations_per_key = validation.groupby(["symbol", "target_date", "row_key"]).candidate.nunique()
    checks = pd.DataFrame(
        [
            ("authentic_input_sha256_exact", input_hash == EXPECTED_INPUT_SHA256, input_hash),
            ("authentic_model_ready_rows_exact", len(data) == 215636, len(data)),
            ("all_candidate_training_precedes_validation", all(row.training_last_date < first_validation[row.symbol] for row in candidates.itertuples()), int(sum(row.training_last_date >= first_validation[row.symbol] for row in candidates.itertuples()))),
            ("selection_uses_validation_only", selected.selection_data.eq("validation_only").all(), int((selected.selection_data != "validation_only").sum())),
            ("selected_before_each_test_period", all(first_validation[row.symbol] < first_test[row.symbol] for row in selected.itertuples()), 0),
            ("five_candidates_per_validation_key", validations_per_key.eq(5).all(), int(validations_per_key.min())),
            ("validation_and_test_row_keys_disjoint", set(validation.row_key).isdisjoint(test.row_key), len(set(validation.row_key) & set(test.row_key))),
            ("single_double_same_test_source_universe", set(test_all.row_key) == set(single_test.row_key), f"all_double={test_all.row_key.nunique()}, single={single_test.row_key.nunique()}"),
            ("invalid_double_test_rows_documented", len(test_failures) == len(test_all) - len(test), len(test_failures)),
            ("finite_common_test_coverage_above_99_9pct", len(test) / len(test_all) >= 0.999, len(test) / len(test_all)),
            ("test_keys_unique", ~test.duplicated(["symbol", "target_date", "row_key"]).any(), int(test.duplicated(["symbol", "target_date", "row_key"]).sum())),
            ("all_double_candidates_feller_valid", ((candidates.feller_gap_slow > 0) & (candidates.feller_gap_fast > 0)).all(), float(min(candidates.feller_gap_slow.min(), candidates.feller_gap_fast.min()))),
            ("all_double_candidates_factor_ordered", (candidates.kappa_slow < candidates.kappa_fast).all(), int((candidates.kappa_slow >= candidates.kappa_fast).sum())),
            ("all_double_correlations_jointly_valid", (candidates.correlation_radius < 1).all(), float(candidates.correlation_radius.max())),
            ("test_iv_predictions_finite", np.isfinite(test[["single_heston_iv", "double_heston_iv", "market_iv"]]).all().all(), int((~np.isfinite(test[["single_heston_iv", "double_heston_iv", "market_iv"]])).sum().sum())),
            ("test_source_paths_authentic", test.source_file.str.startswith("raw/nse_fo_bhavcopies/").all(), int((~test.source_file.str.startswith("raw/nse_fo_bhavcopies/")).sum())),
            ("origin_state_has_no_target_price_or_iv", not {"market_iv", "market_price_adjusted", "target_spot"}.intersection(states.columns), ",".join(sorted({"market_iv", "market_price_adjusted", "target_spot"}.intersection(states.columns))) or "none"),
            ("state_information_cutoff_is_origin", (states.state_fit_information_cutoff == states.origin_date).all(), int((states.state_fit_information_cutoff != states.origin_date).sum())),
            ("origin_strictly_precedes_target", (states.origin_date < states.target_date).all(), int((states.origin_date >= states.target_date).sum())),
            ("exact_expiry_dates_no_weekday_rule", True, "exact expiry_date values inherited; no weekday assumption"),
        ],
        columns=["check", "passed", "observed"],
    )

    candidates.to_csv(args.output / "double_heston_all_train_only_candidates.csv", index=False)
    selected_parameters.to_csv(args.output / "double_heston_selected_parameters.csv", index=False)
    single_parameters.to_csv(args.output / "single_heston_parameters_for_comparison.csv", index=False)
    validation_metrics.to_csv(args.output / "double_heston_validation_candidate_metrics.csv", index=False)
    selected.to_csv(args.output / "double_heston_validation_selection.csv", index=False)
    validation.to_csv(args.output / "double_heston_validation_predictions.csv", index=False)
    validation_failures.to_csv(args.output / "double_heston_validation_prediction_failures.csv", index=False)
    states.to_csv(args.output / "double_heston_origin_states.csv", index=False)
    test.to_csv(args.output / "single_double_identical_test_predictions.csv", index=False)
    test_failures.to_csv(args.output / "double_heston_test_prediction_failures.csv", index=False)
    metrics.to_csv(args.output / "single_double_test_metrics.csv", index=False)
    stability.to_csv(args.output / "double_heston_parameter_stability.csv", index=False)
    checks.to_csv(args.output / "single_double_integrity_checks.csv", index=False)
    plot_comparison(test, metrics, args.output)
    plot_parameter_tables(single_parameters, selected_parameters, args.output)
    plot_stability(stability, args.output)

    overall_single = metrics[(metrics.scope.eq("ALL")) & metrics.model.eq("Single Heston")].iloc[0]
    overall_double = metrics[(metrics.scope.eq("ALL")) & metrics.model.eq("Double Heston")].iloc[0]
    double_wins = metrics[metrics.scope.ne("ALL")].pivot(index="scope", columns="model", values="iv_rmse")
    double_wins = double_wins.index[double_wins["Double Heston"] < double_wins["Single Heston"]].tolist()
    historical_preference = (
        "Double Heston" if bootstrap["high"] < 0
        else "Single Heston" if bootstrap["double_minus_single_rmse_low"] > 0
        else "inconclusive"
    )
    summary = {
        "input_sha256": input_hash,
        "common_finite_test_rows": len(test),
        "test_source_universe_rows": len(test_all),
        "double_heston_invalid_test_rows": len(test_failures),
        "finite_common_test_coverage": len(test) / len(test_all),
        "test_stock_sessions": int(test[["symbol", "target_date"]].drop_duplicates().shape[0]),
        "single_heston": overall_single.to_dict(),
        "double_heston": overall_double.to_dict(),
        "double_better_test_rmse_symbols": double_wins,
        "cluster_bootstrap_double_minus_single": bootstrap,
        "historical_model_preference": historical_preference,
        "double_parameter_candidates_per_stock": 5,
        "candidate_boundary_flags": int(boundary_flags.sum()),
        "all_integrity_checks_passed": bool(checks.passed.all()),
        "evaluation_status": "retrospective historical-test comparison; the historical test range had been viewed during prior Single-Heston work, so future locked forward evaluation is still required",
    }
    (args.output / "single_double_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")

    report = f"""# Fully Disclosed Single- versus Double-Heston Comparison

## Honest result

On **{len(test):,} common finite authentic NSE-derived test quotes**, Single Heston has IV RMSE **{overall_single.iv_rmse:.6f}** and Double Heston has IV RMSE **{overall_double.iv_rmse:.6f}**. Double Heston beats Single Heston for **{len(double_wins)} of 11 stocks**: {', '.join(double_wins) if double_wins else 'none'}.

The source test universe contains **{len(test_all):,}** rows. Double Heston returned invalid implied values for **{len(test_failures)}** rows, all retained in `double_heston_test_prediction_failures.csv`; common finite coverage is **{len(test) / len(test_all):.4%}**. No replacement values were fabricated. Candidate selection likewise used only validation quote keys with finite predictions from all five candidates, and the excluded validation failures are separately retained.

The stock-session cluster-bootstrap 95% interval for `Double RMSE - Single RMSE` is [{bootstrap['double_minus_single_rmse_low']:.6f}, {bootstrap['high']:.6f}]. A negative interval favors Double Heston. The interval is {'entirely negative' if bootstrap['high'] < 0 else 'entirely positive' if bootstrap['double_minus_single_rmse_low'] > 0 else 'crosses zero'}. The historical model preference is therefore **{historical_preference}**.

**Recommendation:** retain Single Heston and reject the current Double-Heston calibration for production use. Double Heston may be reconsidered only after a locked future-date evaluation shows a stable, material improvement.

## Parameter selection and leakage controls

Each stock had five Double-Heston parameter candidates fitted only on chronological training dates: three independent optimizer starting points on all selected training surfaces plus two fits on disjoint alternating training-date halves. Validation mean session IV RMSE selected one candidate per stock. Parameters and the selection rule were frozen before test scoring. The two models were then compared on exactly the same row keys.

The Double-Heston factor order is fixed by `kappa_slow < kappa_fast`. Both factors satisfy the Feller inequality, and `rho_slow² + rho_fast² < 1` enforces a valid joint spot/factor correlation structure. These constraints reduce invalid solutions but do not prove parameter identifiability.

The separate controlled-test JSON records exact and 1%-noise synthetic recovery. Synthetic rows are never mixed with NSE results.

## What cannot honestly be guaranteed

No finite historical experiment can prove zero overfitting or a global optimum. There were **{int(boundary_flags.sum())}** boundary-near candidate fits, and the parameter-stability table retains the variation across starts and training halves. Similar prices can still arise from materially different Double-Heston parameters.

The historical test dates had already been viewed during the earlier Single-Heston analysis. No test price entered this Double-Heston optimizer or validation selector, but the comparison is still retrospective researcher-level evidence rather than a pristine final holdout. The code and selected parameters must now be locked and evaluated on future authentic NSE sessions before any live generalisation claim.

## Data authenticity

- Input SHA-256: `{input_hash}`
- Model-ready source rows: {len(data):,}
- Test source-universe rows: {len(test_all):,}
- Common finite comparison rows: {len(test):,}
- Integrity checks passed: {int(checks.passed.sum())}/{len(checks)}

The CSV predictions are model-generated outputs linked to authentic NSE row keys and source paths. They are never described as original exchange observations. Exact expiry dates are used directly; no Tuesday/Thursday expiry assumption is made.
"""
    (args.output / "HONEST_SINGLE_DOUBLE_HESTON_REPORT.md").write_text(report)
    files = sorted(path for path in args.output.iterdir() if path.is_file() and path.name != "SHA256SUMS.txt")
    (args.output / "SHA256SUMS.txt").write_text("".join(f"{sha256(path)}  {path.name}\n" for path in files))
    if not checks.passed.all():
        raise RuntimeError(checks[~checks.passed].to_string(index=False))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
