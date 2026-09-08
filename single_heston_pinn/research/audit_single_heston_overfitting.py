#!/usr/bin/env python3
"""Independent leakage, overfitting, baseline, and stability audit for single Heston."""

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

import single_heston as heston


ROOT = Path(__file__).resolve().parent
MODEL_OUTPUT = ROOT / "outputs" / "single_heston"
DEFAULT_OUTPUT = ROOT / "outputs" / "single_heston_overfit_audit"
MODEL_COLUMNS = {
    "heston_same_day_v0": "heston_same_day_iv",
    "flat_anchor_iv": "flat_anchor_iv",
    "quadratic_anchor_smile": "quadratic_anchor_iv",
    "heston_training_median_v0": "heston_training_median_iv",
    "heston_prior_available_v0": "heston_prior_state_iv",
    "heston_shuffled_v0": "heston_shuffled_state_iv",
    "heston_train_half_a": "heston_half_a_iv",
    "heston_train_half_b": "heston_half_b_iv",
}


def iv_from_params(quotes, params):
    prices = heston.heston_prices(quotes, params)
    iv = heston.implied_volatility(
        prices,
        quotes.spot.to_numpy(float),
        quotes.strike.to_numpy(float),
        quotes.maturity.to_numpy(float),
        quotes.rate.to_numpy(float),
        quotes.dividend.to_numpy(float),
        quotes.option_type.eq("CE").to_numpy(),
    )
    return prices, iv


def baseline_iv(calibration, holdout, degree):
    output = pd.Series(index=holdout.index, dtype=float)
    for expiry, target in holdout.groupby("expiry_date"):
        anchor = calibration[calibration.expiry_date.eq(expiry)]
        if degree == 0 or len(anchor) < 4:
            output.loc[target.index] = float(anchor.market_iv.median())
            continue
        coefficients = np.polyfit(
            anchor.log_forward_moneyness,
            anchor.market_iv,
            deg=2,
            w=np.sqrt(anchor.weight),
        )
        output.loc[target.index] = np.clip(
            np.polyval(coefficients, target.log_forward_moneyness), 0.03, 2.5
        )
    return output.to_numpy(float)


def score(frame, prediction_column):
    clean = frame[["market_iv", prediction_column]].dropna()
    actual = clean.market_iv.to_numpy(float)
    predicted = clean[prediction_column].to_numpy(float)
    error = predicted - actual
    denominator = np.sum((actual - actual.mean()) ** 2)
    grouped = frame.dropna(subset=[prediction_column]).copy()
    keys = ["symbol", "trade_date", "expiry_date"]
    grouped["actual_centered"] = grouped.market_iv - grouped.groupby(keys).market_iv.transform("mean")
    grouped["predicted_centered"] = grouped[prediction_column] - grouped.groupby(keys)[prediction_column].transform("mean")
    centered_denominator = np.sum(grouped.actual_centered**2)
    grouped["square_error"] = (grouped[prediction_column] - grouped.market_iv) ** 2
    surface_rmse = np.sqrt(grouped.groupby(keys).square_error.mean())
    return {
        "rows": len(clean),
        "iv_rmse": float(np.sqrt(np.mean(error**2))),
        "iv_mae": float(np.mean(np.abs(error))),
        "iv_bias": float(np.mean(error)),
        "pooled_iv_r2": float(1 - np.sum(error**2) / denominator),
        "within_surface_centered_r2": float(
            1
            - np.sum((grouped.predicted_centered - grouped.actual_centered) ** 2)
            / centered_denominator
        )
        if centered_denominator > 0
        else np.nan,
        "mean_surface_iv_rmse": float(surface_rmse.mean()),
        "median_surface_iv_rmse": float(surface_rmse.median()),
    }


def fit_training_halves(data, training, output):
    rows = []
    structures = {}
    for symbol, dates in training.groupby("symbol"):
        dates = sorted(pd.to_datetime(dates.trade_date).unique())
        halves = {"a": dates[::2], "b": dates[1::2]}
        structures[symbol] = {}
        for half, half_dates in halves.items():
            surfaces = []
            for day in half_dates:
                raw = data[data.symbol.eq(symbol) & data.trade_date.eq(day)]
                quotes = heston.prepare_date(raw, Counter(), holdout_fold=1)
                if quotes is not None:
                    surfaces.append(quotes[quotes.fold.eq("calibration")].reset_index(drop=True))
            if len(surfaces) < 3:
                raise RuntimeError(f"{symbol} training half {half} has fewer than three valid dates")
            best, starts = heston.fit_joint_structural(surfaces)
            objective, structural, variances, nfev = best
            structures[symbol][half] = structural
            rows.append(
                {
                    "symbol": symbol,
                    "training_half": half,
                    "dates": len(surfaces),
                    "objective": objective,
                    "starting_objective_range": float(np.ptp([item[0] for item in starts])),
                    "optimizer_evaluations": nfev,
                    "kappa": structural[0],
                    "theta": structural[1],
                    "sigma": structural[2],
                    "rho": structural[3],
                    "median_v0": float(np.median(variances)),
                    "feller_gap": 2 * structural[0] * structural[1] - structural[2] ** 2,
                    "kappa_near_upper_bound": structural[0] > 9.9,
                    "theta_near_upper_bound": structural[1] > 0.99,
                    "sigma_near_feller_cap": structural[2]
                    / (0.995 * math.sqrt(2 * structural[0] * structural[1]))
                    > 0.99,
                }
            )
    stability = pd.DataFrame(rows)
    stability.to_csv(output / "alternate_training_parameter_fits.csv", index=False)
    wide = stability.pivot(index="symbol", columns="training_half", values=["kappa", "theta", "sigma", "rho"])
    comparison = []
    for symbol in wide.index:
        comparison.append(
            {
                "symbol": symbol,
                "kappa_log_ratio_abs": abs(math.log(wide.loc[symbol, ("kappa", "a")] / wide.loc[symbol, ("kappa", "b")])),
                "theta_log_ratio_abs": abs(math.log(wide.loc[symbol, ("theta", "a")] / wide.loc[symbol, ("theta", "b")])),
                "sigma_log_ratio_abs": abs(math.log(wide.loc[symbol, ("sigma", "a")] / wide.loc[symbol, ("sigma", "b")])),
                "rho_absolute_difference": abs(wide.loc[symbol, ("rho", "a")] - wide.loc[symbol, ("rho", "b")]),
            }
        )
    pd.DataFrame(comparison).to_csv(output / "parameter_sample_stability.csv", index=False)
    return structures, stability, pd.DataFrame(comparison)


def cluster_bootstrap_difference(frame, first, second, repetitions=1000):
    grouped = [group for _, group in frame.groupby(["symbol", "trade_date"], sort=False)]
    rng = np.random.default_rng(20260806)
    differences = []
    for _ in range(repetitions):
        sampled = [grouped[index] for index in rng.integers(0, len(grouped), len(grouped))]
        sample = pd.concat(sampled, ignore_index=True)
        first_rmse = np.sqrt(np.mean((sample[first] - sample.market_iv) ** 2))
        second_rmse = np.sqrt(np.mean((sample[second] - sample.market_iv) ** 2))
        differences.append(first_rmse - second_rmse)
    return np.quantile(differences, [0.025, 0.5, 0.975])


def plot_comparison(predictions, metrics, output):
    panels = [
        ("heston_same_day_iv", "Heston: same-day anchor v0"),
        ("flat_anchor_iv", "Flat IV from same-day anchors"),
        ("heston_prior_state_iv", "Heston: prior available v0"),
        ("heston_shuffled_state_iv", "Heston: shuffled v0 control"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    limit = min(0.75, float(np.nanpercentile(predictions.market_iv, 99.8)))
    for ax, (column, title) in zip(axes.flat, panels):
        ax.hexbin(predictions.market_iv, predictions[column], gridsize=42, mincnt=1, bins="log", cmap="viridis")
        ax.plot([0, limit], [0, limit], "r--", linewidth=1)
        row = metrics[metrics.prediction_column.eq(column)].iloc[0]
        ax.set(
            xlim=(0, limit), ylim=(0, limit),
            xlabel="Market IV (held-out)", ylabel="Predicted IV",
            title=f"{title}\nRMSE {row.iv_rmse:.4f}; pooled R² {row.pooled_iv_r2:.3f}",
        )
    fig.suptitle("Single-Heston overfitting controls: three-fold strike cross-fit")
    fig.savefig(output / "overfitting_control_comparison.png", dpi=180)
    plt.close(fig)

    ordered = metrics.sort_values("iv_rmse")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    axes[0].barh(ordered.model, ordered.iv_rmse, color="#3b4cc0")
    axes[0].set(xlabel="Three-fold held-out IV RMSE", title="Prediction error")
    positions = np.arange(len(ordered))
    width = 0.36
    axes[1].barh(positions - width / 2, ordered.pooled_iv_r2, height=width, label="pooled R²", color="#3b4cc0")
    axes[1].barh(positions + width / 2, ordered.within_surface_centered_r2, height=width, label="within-surface R²", color="#b40426")
    axes[1].set(yticks=positions, yticklabels=ordered.model, xlabel="R²", title="Pooled metric versus smile-shape metric")
    axes[1].axvline(0, color="grey", linewidth=1)
    axes[1].legend(frameon=False)
    fig.savefig(output / "metric_inflation_and_baselines.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=heston.DEFAULT_INPUT)
    parser.add_argument("--model-output", type=Path, default=MODEL_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reuse-alternate-fits", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    data = heston.load_ready_data(args.input)
    split_map = heston.make_split_map(data)
    parameters = pd.read_csv(args.model_output / "single_heston_parameters.csv", parse_dates=["training_last_date", "test_first_date"]).set_index("symbol")
    training = pd.read_csv(args.model_output / "single_heston_training_fits.csv", parse_dates=["trade_date"])
    if args.reuse_alternate_fits and (args.output / "alternate_training_parameter_fits.csv").exists():
        alternate_table = pd.read_csv(args.output / "alternate_training_parameter_fits.csv")
        stability = pd.read_csv(args.output / "parameter_sample_stability.csv")
        alternate = {
            symbol: {
                half: group.iloc[0][["kappa", "theta", "sigma", "rho"]].to_numpy(float)
                for half, group in rows.groupby("training_half")
            }
            for symbol, rows in alternate_table.groupby("symbol")
        }
    else:
        alternate, alternate_table, stability = fit_training_halves(data, training, args.output)

    audit = Counter()
    items = []
    calibration_records = []
    anchor_holdout_pair_overlap = 0
    for symbol, symbol_data in data.groupby("symbol"):
        structural = parameters.loc[symbol, ["kappa", "theta", "sigma", "rho"]].to_numpy(float)
        for day in sorted(symbol_data.trade_date.unique()):
            if split_map[(symbol, pd.Timestamp(day))] != "test":
                continue
            day_raw = symbol_data[symbol_data.trade_date.eq(day)]
            for fold in (0, 1, 2):
                quotes = heston.prepare_date(day_raw, audit, holdout_fold=fold)
                if quotes is None:
                    continue
                calibration = quotes[quotes.fold.eq("calibration")].reset_index(drop=True)
                holdout = quotes[quotes.fold.eq("holdout")].reset_index(drop=True)
                v0, _ = heston.fit_v0(calibration, structural)
                v0_a, _ = heston.fit_v0(calibration, alternate[symbol]["a"])
                v0_b, _ = heston.fit_v0(calibration, alternate[symbol]["b"])
                _, calibration_iv = iv_from_params(calibration, np.r_[structural, v0])
                calibration_records.extend(
                    {
                        "symbol": symbol,
                        "trade_date": pd.Timestamp(day),
                        "crossfit_fold": fold,
                        "market_iv": actual,
                        "predicted_iv": predicted,
                    }
                    for actual, predicted in zip(calibration.market_iv, calibration_iv)
                )
                item_anchor_keys = set(calibration.anchor_ce_row_key.dropna().astype(str))
                item_anchor_keys.update(calibration.anchor_pe_row_key.dropna().astype(str))
                item_holdout_pair_keys = set()
                for row in holdout.itertuples(index=False):
                    paired = day_raw[
                        day_raw.expiry_date.eq(row.expiry_date)
                        & day_raw.adjusted_strike_price.eq(row.strike)
                    ]
                    item_holdout_pair_keys.update(paired.row_key.astype(str))
                anchor_holdout_pair_overlap += len(item_anchor_keys & item_holdout_pair_keys)
                items.append(
                    {
                        "symbol": symbol,
                        "trade_date": pd.Timestamp(day),
                        "fold": fold,
                        "calibration": calibration,
                        "holdout": holdout,
                        "structural": structural,
                        "v0": v0,
                        "v0_a": v0_a,
                        "v0_b": v0_b,
                    }
                )

    state_table = pd.DataFrame(
        [{"symbol": item["symbol"], "trade_date": item["trade_date"], "fold": item["fold"], "v0": item["v0"]} for item in items]
    ).sort_values(["symbol", "fold", "trade_date"])
    state_table["prior_v0"] = state_table.groupby(["symbol", "fold"]).v0.shift(1)
    state_table["shuffled_v0"] = state_table.groupby(["symbol", "fold"]).v0.transform(
        lambda values: np.roll(values.to_numpy(), max(1, len(values) // 2))
    )
    state_lookup = state_table.set_index(["symbol", "trade_date", "fold"])[["prior_v0", "shuffled_v0"]].to_dict("index")

    frames = []
    for item in items:
        holdout = item["holdout"].copy()
        key = (item["symbol"], item["trade_date"], item["fold"])
        states = state_lookup[key]
        holdout["crossfit_fold"] = item["fold"]
        holdout["heston_same_day_iv"] = iv_from_params(holdout, np.r_[item["structural"], item["v0"]])[1]
        holdout["flat_anchor_iv"] = baseline_iv(item["calibration"], holdout, degree=0)
        holdout["quadratic_anchor_iv"] = baseline_iv(item["calibration"], holdout, degree=2)
        holdout["heston_training_median_iv"] = iv_from_params(
            holdout, np.r_[item["structural"], parameters.loc[item["symbol"], "v0_training_median"]]
        )[1]
        holdout["heston_prior_state_iv"] = (
            iv_from_params(holdout, np.r_[item["structural"], states["prior_v0"]])[1]
            if np.isfinite(states["prior_v0"])
            else np.nan
        )
        holdout["heston_shuffled_state_iv"] = iv_from_params(
            holdout, np.r_[item["structural"], states["shuffled_v0"]]
        )[1]
        holdout["heston_half_a_iv"] = iv_from_params(
            holdout, np.r_[alternate[item["symbol"]]["a"], item["v0_a"]]
        )[1]
        holdout["heston_half_b_iv"] = iv_from_params(
            holdout, np.r_[alternate[item["symbol"]]["b"], item["v0_b"]]
        )[1]
        frames.append(holdout)
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_csv(args.output / "three_fold_crossfit_predictions.csv", index=False)
    state_table.to_csv(args.output / "crossfit_daily_states.csv", index=False)

    metric_rows = []
    for model, column in MODEL_COLUMNS.items():
        metric_rows.append({"model": model, "prediction_column": column, **score(predictions, column)})
    metrics = pd.DataFrame(metric_rows)
    flat_rmse = float(metrics.loc[metrics.model.eq("flat_anchor_iv"), "iv_rmse"].iloc[0])
    metrics["rmse_improvement_vs_flat"] = flat_rmse - metrics.iv_rmse
    metrics.to_csv(args.output / "overfitting_model_comparison.csv", index=False)

    calibration_predictions = pd.DataFrame(calibration_records)
    fold_rows = []
    for fold in (0, 1, 2):
        held = predictions[predictions.crossfit_fold.eq(fold)]
        cal = calibration_predictions[calibration_predictions.crossfit_fold.eq(fold)]
        fold_rows.append(
            {
                "crossfit_fold": fold,
                "heldout_rows": len(held),
                "heldout_iv_rmse": float(np.sqrt(np.mean((held.heston_same_day_iv - held.market_iv) ** 2))),
                "calibration_rows": len(cal),
                "calibration_iv_rmse": float(np.sqrt(np.mean((cal.predicted_iv - cal.market_iv) ** 2))),
            }
        )
    pd.DataFrame(fold_rows).to_csv(args.output / "crossfit_fold_metrics.csv", index=False)

    subgroup_rows = []
    categories = {
        "moneyness": pd.cut(predictions.log_forward_moneyness.abs(), [-1e-9, 0.05, 0.15, 0.35], labels=["ATM", "moderate", "wing"]),
        "maturity": pd.cut(predictions.days_to_expiry, [6, 30, 60, 180], labels=["7-30d", "31-60d", "61-180d"]),
    }
    for dimension, category in categories.items():
        for label, frame in predictions.groupby(category, observed=True):
            for model, column in MODEL_COLUMNS.items():
                subgroup_rows.append({"dimension": dimension, "group": str(label), "model": model, **score(frame, column)})
    for symbol, frame in predictions.groupby("symbol"):
        for model, column in MODEL_COLUMNS.items():
            subgroup_rows.append({"dimension": "symbol", "group": symbol, "model": model, **score(frame, column)})
    pd.DataFrame(subgroup_rows).to_csv(args.output / "subgroup_metrics.csv", index=False)

    main_column = MODEL_COLUMNS["heston_same_day_v0"]
    flat_column = MODEL_COLUMNS["flat_anchor_iv"]
    confidence = cluster_bootstrap_difference(predictions, main_column, flat_column)
    source = data[["row_key", "symbol", "trade_date", "expiry_date", "observed_option_price", "source_file"]]
    joined = predictions.merge(source, on="row_key", suffixes=("_audit", "_source"), how="left", validate="one_to_one")
    raw_scale_iv = heston.implied_volatility(
        predictions.market_price_raw.to_numpy(float),
        (predictions.spot / predictions.price_adjustment_factor).to_numpy(float),
        (predictions.strike / predictions.price_adjustment_factor).to_numpy(float),
        predictions.maturity.to_numpy(float),
        predictions.rate.to_numpy(float),
        predictions.dividend.to_numpy(float),
        predictions.option_type.eq("CE").to_numpy(),
    )
    adjustment_iv_difference = float(np.nanmax(np.abs(raw_scale_iv - predictions.market_iv)))
    test_ready_rows = sum(
        len(day_group)
        for symbol, group in data.groupby("symbol")
        for day, day_group in group.groupby("trade_date")
        if split_map[(symbol, pd.Timestamp(day))] == "test"
    )
    leakage_checks = [
        ("crossfit_holdout_row_keys_unique", predictions.row_key.is_unique, predictions.row_key.duplicated().sum()),
        ("anchor_keys_disjoint_from_both_sides_of_holdout_strikes", anchor_holdout_pair_overlap == 0, anchor_holdout_pair_overlap),
        ("structural_training_dates_strictly_before_test", bool((parameters.training_last_date < parameters.test_first_date).all()), int((parameters.training_last_date >= parameters.test_first_date).sum())),
        ("source_rows_join_one_to_one", joined.symbol_source.notna().all(), joined.symbol_source.isna().sum()),
        ("source_symbol_exact", joined.symbol_audit.eq(joined.symbol_source).all(), (~joined.symbol_audit.eq(joined.symbol_source)).sum()),
        ("source_trade_date_exact", joined.trade_date_audit.eq(joined.trade_date_source).all(), (~joined.trade_date_audit.eq(joined.trade_date_source)).sum()),
        ("source_expiry_exact", joined.expiry_date_audit.eq(joined.expiry_date_source).all(), (~joined.expiry_date_audit.eq(joined.expiry_date_source)).sum()),
        ("source_raw_price_exact", np.allclose(joined.market_price_raw, joined.observed_option_price, rtol=0, atol=1e-12), int((np.abs(joined.market_price_raw - joined.observed_option_price) > 1e-12).sum())),
        ("source_file_exact", joined.source_file_audit.eq(joined.source_file_source).all(), (~joined.source_file_audit.eq(joined.source_file_source)).sum()),
        ("corporate_action_scaling_does_not_change_iv", adjustment_iv_difference < 1e-10, adjustment_iv_difference),
        ("every_fold_has_heldout_rows", predictions.groupby("crossfit_fold").size().reindex([0, 1, 2], fill_value=0).gt(0).all(), predictions.groupby("crossfit_fold").size().to_dict()),
    ]
    leakage = pd.DataFrame(leakage_checks, columns=["check", "passed", "observed"])
    leakage.to_csv(args.output / "independent_leakage_checks.csv", index=False)
    if not leakage.passed.all():
        raise RuntimeError("independent leakage audit failed\n" + leakage.to_string(index=False))

    plot_comparison(predictions, metrics, args.output)
    heston_metrics = metrics[metrics.model.eq("heston_same_day_v0")].iloc[0]
    flat_metrics = metrics[metrics.model.eq("flat_anchor_iv")].iloc[0]
    quadratic_metrics = metrics[metrics.model.eq("quadratic_anchor_smile")].iloc[0]
    training_median_metrics = metrics[metrics.model.eq("heston_training_median_v0")].iloc[0]
    prior_metrics = metrics[metrics.model.eq("heston_prior_available_v0")].iloc[0]
    shuffled_metrics = metrics[metrics.model.eq("heston_shuffled_v0")].iloc[0]
    half_a_metrics = metrics[metrics.model.eq("heston_train_half_a")].iloc[0]
    half_b_metrics = metrics[metrics.model.eq("heston_train_half_b")].iloc[0]
    boundary_hits = int(
        alternate_table.kappa_near_upper_bound.sum()
        + alternate_table.theta_near_upper_bound.sum()
        + alternate_table.sigma_near_feller_cap.sum()
    )
    summary = {
        "verdict": "not row memorization; strong pooled fit is partly same-day level calibration and must not be called unseen-date forecasting",
        "crossfit_unique_heldout_quotes": len(predictions),
        "test_model_ready_rows_before_surface_filters": int(test_ready_rows),
        "crossfit_coverage_of_test_ready_rows": len(predictions) / test_ready_rows,
        "heston_same_day_anchor_v0": heston_metrics.to_dict(),
        "flat_anchor_baseline": flat_metrics.to_dict(),
        "quadratic_anchor_baseline": quadratic_metrics.to_dict(),
        "training_median_state_control": training_median_metrics.to_dict(),
        "prior_available_state": prior_metrics.to_dict(),
        "shuffled_state_control": shuffled_metrics.to_dict(),
        "alternate_train_half_a": half_a_metrics.to_dict(),
        "alternate_train_half_b": half_b_metrics.to_dict(),
        "cluster_bootstrap_heston_minus_flat_rmse_95pct": confidence.tolist(),
        "parameter_boundary_flags_across_22_half_sample_fits": boundary_hits,
        "maximum_parameter_half_sample_instability": stability.drop(columns="symbol").max().to_dict(),
        "leakage_checks_passed": int(leakage.passed.sum()),
        "leakage_checks_total": len(leakage),
        "noise_audit": dict(audit),
    }
    (args.output / "overfitting_audit_summary.json").write_text(
        json.dumps(summary, indent=2, default=lambda value: value.item() if hasattr(value, "item") else str(value)),
        encoding="utf-8",
    )

    report = f"""# Independent single-Heston overfitting and leakage audit

## Verdict

The fitted model is **not perfectly matching the quotes and there is no evidence of row-level label memorization**. Three-fold cross-fitting assigns every strike, including both its call and put, wholly to calibration or evaluation. All {len(leakage)} independent leakage checks passed.

However, the original result is an **inverse calibration on observed dates**, not a fully unseen-date forecast. Structural parameters are learned from early dates, but each later date uses same-day anchor option prices to fit one variance state (`v0`) and infer carry. The publication must call the test “held-out-strike evaluation within later dates.”

## Three-fold cross-fit result

| Test | IV RMSE | Pooled R² | Within-surface centered R² |
|---|---:|---:|---:|
| Heston, same-day anchor `v0` | {heston_metrics.iv_rmse:.6f} | {heston_metrics.pooled_iv_r2:.6f} | {heston_metrics.within_surface_centered_r2:.6f} |
| Flat IV from same-day anchors | {flat_metrics.iv_rmse:.6f} | {flat_metrics.pooled_iv_r2:.6f} | {flat_metrics.within_surface_centered_r2:.6f} |
| Quadratic smile from same-day anchors | {quadratic_metrics.iv_rmse:.6f} | {quadratic_metrics.pooled_iv_r2:.6f} | {quadratic_metrics.within_surface_centered_r2:.6f} |
| Heston, fixed training-median `v0` | {training_median_metrics.iv_rmse:.6f} | {training_median_metrics.pooled_iv_r2:.6f} | {training_median_metrics.within_surface_centered_r2:.6f} |
| Heston, prior available `v0` | {prior_metrics.iv_rmse:.6f} | {prior_metrics.pooled_iv_r2:.6f} | {prior_metrics.within_surface_centered_r2:.6f} |
| Heston, shuffled `v0` control | {shuffled_metrics.iv_rmse:.6f} | {shuffled_metrics.pooled_iv_r2:.6f} | {shuffled_metrics.within_surface_centered_r2:.6f} |
| Heston, train-half A parameters | {half_a_metrics.iv_rmse:.6f} | {half_a_metrics.pooled_iv_r2:.6f} | {half_a_metrics.within_surface_centered_r2:.6f} |
| Heston, train-half B parameters | {half_b_metrics.iv_rmse:.6f} | {half_b_metrics.pooled_iv_r2:.6f} | {half_b_metrics.within_surface_centered_r2:.6f} |

The pooled R² is much larger than the within-surface centered R² because pooled variation includes easy differences in volatility level between symbols, dates and expiries. Only the centered measure tests how well the smile shape is captured after those levels are removed.

The cluster-bootstrap 95% interval for `Heston RMSE - flat-anchor RMSE` is [{confidence[0]:.6f}, {confidence[2]:.6f}]. A negative interval means Heston improves on the simple same-date flat baseline; an interval crossing zero means the improvement is not established.

The quadratic anchor-smile control is substantially more accurate than Heston on held-out strikes. It also uses more date/expiry-specific flexibility, so it is not a like-for-like structural model, but this result forbids any claim that Heston is the best empirical interpolator.

## Coverage and selection

The cross-fit contains {len(predictions):,} unique held-out quotes from {test_ready_rows:,} model-ready rows in the chronological test periods ({len(predictions) / test_ready_rows:.2%}). Results therefore apply to liquid, paired, parity-consistent quotes that pass the documented maturity, IV, moneyness and vega filters—not to every raw NSE option row.

## Parameter honesty

Two disjoint halves of the selected training dates were fitted independently for every stock. Across the 22 fits there were {boundary_hits} parameter-boundary flags. The largest half-sample changes were: `|log kappa ratio|` {stability.kappa_log_ratio_abs.max():.3f}, `|log theta ratio|` {stability.theta_log_ratio_abs.max():.3f}, `|log sigma ratio|` {stability.sigma_log_ratio_abs.max():.3f}, and `|rho difference|` {stability.rho_absolute_difference.max():.3f}.

Consequently, pricing performance may be stable while individual Heston parameters remain weakly identified. Do not describe the parameters as uniquely recovered economic constants.

Wing performance is materially worse than near-the-money performance, and CESC has negative pooled R². These subgroup failures must accompany the aggregate result. A fully no-same-day-option-input forecast was not claimed because the dataset lacks independently aligned point-in-time interest-rate and dividend inputs; same-day anchor quotes currently provide effective carry.

## Publication-safe claim

“On a strictly separated subset of later-date NSE power-stock option strikes, a one-factor Heston calibration with train-only structural parameters and same-day anchor-based variance state achieved the reported cross-fit error. Both sides of every held-out strike were excluded from carry and state estimation. Performance is conditional on liquidity and parity filters; it is not a no-same-day-data forecast, and structural parameters exhibit boundary and sample-instability warnings.”
"""
    (args.output / "PUBLICATION_HONEST_OVERFITTING_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, default=lambda value: value.item() if hasattr(value, "item") else str(value)))


if __name__ == "__main__":
    main()
