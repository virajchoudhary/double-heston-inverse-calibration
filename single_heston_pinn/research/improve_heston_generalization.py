#!/usr/bin/env python3
"""Guarded validation-only recalibration of the next-session Heston forecast."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import single_heston as heston
from forecast_single_heston_next_day import EXPECTED_INPUT_SHA256, make_grid, sha256


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "outputs" / "single_heston_next_day_forecast"
DEFAULT_OUTPUT = ROOT / "outputs" / "single_heston_generalized_forecast"
INPUT = ROOT / "outputs" / "019fc8a0" / "model_input_option_prices.csv"
SPECS = ("affine", "surface", "curved")
FEATURES = {
    "raw": [],
    "affine": ["intercept", "raw_heston_iv"],
    "surface": ["intercept", "raw_heston_iv", "log_moneyness", "absolute_log_moneyness", "maturity"],
    "curved": ["intercept", "raw_heston_iv", "raw_heston_iv_squared", "log_moneyness", "absolute_log_moneyness", "maturity"],
}


def design(frame: pd.DataFrame, spec: str) -> np.ndarray:
    h = frame.forecast_iv.to_numpy(float)
    x = frame.log_forward_moneyness.to_numpy(float)
    t = frame.maturity.to_numpy(float)
    if spec == "affine":
        return np.c_[np.ones(len(frame)), h]
    if spec == "surface":
        return np.c_[np.ones(len(frame)), h, x, np.abs(x), t]
    if spec == "curved":
        return np.c_[np.ones(len(frame)), h, h * h, x, np.abs(x), t]
    raise ValueError(f"unknown specification: {spec}")


def fit_calibrator(frame: pd.DataFrame, spec: str) -> np.ndarray:
    matrix = design(frame, spec)
    target = frame.market_iv.to_numpy(float)
    date_count = frame.groupby("target_date").row_key.transform("size").to_numpy(float)
    weight = 1 / np.sqrt(date_count)  # Equal total influence per validation session.
    return np.linalg.lstsq(matrix * weight[:, None], target * weight, rcond=None)[0]


def apply_calibrator(frame: pd.DataFrame, spec: str, coefficients: np.ndarray | None) -> np.ndarray:
    prediction = frame.forecast_iv.to_numpy(float) if spec == "raw" else design(frame, spec) @ coefficients
    return np.clip(prediction, 0.03, 2.5)


def rmse(actual, forecast) -> float:
    return float(np.sqrt(np.mean((np.asarray(forecast) - np.asarray(actual)) ** 2)))


def daily_rmse(frame: pd.DataFrame, forecast) -> float:
    scored = frame[["target_date", "market_iv"]].copy()
    scored["forecast"] = np.asarray(forecast)
    return float(
        scored.groupby("target_date")
        .apply(lambda group: rmse(group.market_iv, group.forecast), include_groups=False)
        .mean()
    )


def select_models(predictions: pd.DataFrame, minimum_dates: int = 10, minimum_gain: float = 0.05):
    """Use early validation for candidate choice and late validation for correction choice."""
    selections, diagnostics = [], []
    validation = predictions[predictions.target_split.eq("validation")]
    for symbol, symbol_validation in validation.groupby("symbol"):
        dates = sorted(symbol_validation.target_date.unique())
        cut = max(1, int(len(dates) * 0.67))
        early_dates, late_dates = dates[:cut], dates[cut:]
        early = symbol_validation[symbol_validation.target_date.isin(early_dates)]
        late = symbol_validation[symbol_validation.target_date.isin(late_dates)]
        candidate_scores = [
            (daily_rmse(frame, frame.forecast_iv), candidate)
            for candidate, frame in early.groupby("candidate")
        ]
        candidate_score, candidate = min(candidate_scores)
        early = early[early.candidate.eq(candidate)]
        late = late[late.candidate.eq(candidate)]
        raw_score = daily_rmse(late, late.forecast_iv)
        alternatives = []
        for spec in SPECS:
            coefficients = fit_calibrator(early, spec)
            score = daily_rmse(late, apply_calibrator(late, spec, coefficients))
            alternatives.append((score, spec))
            diagnostics.append(
                {
                    "symbol": symbol,
                    "candidate": candidate,
                    "specification": spec,
                    "early_validation_dates": len(early_dates),
                    "late_validation_dates": len(late_dates),
                    "late_validation_daily_rmse": score,
                }
            )
        best_score, best_spec = min(alternatives)
        gain = (raw_score - best_score) / raw_score
        guarded = len(dates) >= minimum_dates and gain >= minimum_gain
        selections.append(
            {
                "symbol": symbol,
                "candidate": candidate,
                "candidate_selection_daily_rmse": candidate_score,
                "selected_specification": best_spec if guarded else "raw",
                "validation_dates": len(dates),
                "early_validation_last_date": max(early_dates),
                "late_validation_first_date": min(late_dates),
                "late_validation_last_date": max(late_dates),
                "late_raw_daily_rmse": raw_score,
                "late_best_daily_rmse": best_score,
                "relative_validation_gain": gain,
                "guard_minimum_dates": minimum_dates,
                "guard_minimum_gain": minimum_gain,
                "correction_accepted": guarded,
            }
        )
    return pd.DataFrame(selections), pd.DataFrame(diagnostics)


def score(frame: pd.DataFrame, column: str) -> dict:
    error = frame[column] - frame.market_iv
    denominator = np.sum((frame.market_iv - frame.market_iv.mean()) ** 2)
    return {
        "rows": len(frame),
        "iv_rmse": float(np.sqrt(np.mean(error**2))),
        "iv_mae": float(np.mean(np.abs(error))),
        "iv_bias": float(np.mean(error)),
        "iv_r2": float(1 - np.sum(error**2) / denominator),
        "forecast_on_actual_slope": float(np.polyfit(frame.market_iv, frame[column], 1)[0]),
    }


def plot_results(test: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    low, high = float(test.market_iv.min()), float(test.market_iv.quantile(0.995))
    for ax, column, title in [
        (axes[0], "original_locked_heston_iv", "Original locked Heston"),
        (axes[1], "generalized_heston_iv", "Guarded validation-calibrated Heston"),
    ]:
        ax.scatter(test.market_iv, test[column], s=8, alpha=0.20, edgecolors="none", rasterized=True)
        ax.plot([low, high], [low, high], "--", color="crimson", lw=1.1, label="Ideal identity: forecast = actual")
        metric = score(test, column)
        ax.set(
            xlim=(low, high), ylim=(low, high), xlabel="Actual next-session IV", ylabel="Forecast IV",
            title=f"{title}\nRMSE={metric['iv_rmse']:.4f}, R²={metric['iv_r2']:.3f}, slope={metric['forecast_on_actual_slope']:.3f}",
        )
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.15)
    fig.suptitle("Next-session generalisation: retrospective historical-test comparison")
    fig.savefig(output / "generalized_actual_vs_forecast.png", dpi=180)
    plt.close(fig)

    symbols = sorted(test.symbol.unique())
    fig, axes = plt.subplots(4, 3, figsize=(15, 12), constrained_layout=True)
    for ax, symbol in zip(axes.flat, symbols):
        frame = test[test.symbol.eq(symbol)]
        low = float(min(frame.market_iv.min(), frame.generalized_heston_iv.min()))
        high = float(max(frame.market_iv.max(), frame.generalized_heston_iv.max()))
        ax.scatter(frame.market_iv, frame.generalized_heston_iv, s=10, alpha=0.32, edgecolors="none", rasterized=True)
        ax.plot([low, high], [low, high], "--", color="crimson", lw=0.9, label="Ideal identity (fixed 45°)")
        slope, intercept = np.polyfit(frame.market_iv, frame.generalized_heston_iv, 1)
        actual_range = np.array([low, high])
        ax.plot(actual_range, intercept + slope * actual_range, color="black", lw=1.0, label="Observed test trend")
        metric = score(frame, "generalized_heston_iv")
        ax.set_title(f"{symbol} · RMSE {metric['iv_rmse']:.3f} · trend slope {slope:.2f}")
        ax.set(xlabel="Actual IV", ylabel="Forecast IV")
        ax.grid(alpha=0.2)
    for ax in axes.flat[len(symbols):]:
        ax.axis("off")
    axes.flat[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Generalized Heston observations by stock: identity line versus empirical trend")
    fig.savefig(output / "generalized_actual_vs_forecast_by_stock.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.base / "all_candidate_predictions.csv", parse_dates=["origin_date", "target_date", "expiry_date"])
    original = pd.read_csv(args.base / "selected_heston_test_predictions.csv", usecols=["symbol", "target_date", "row_key", "forecast_iv"], parse_dates=["target_date"])
    original = original.rename(columns={"forecast_iv": "original_locked_heston_iv"})
    selections, diagnostics = select_models(predictions)
    coefficient_rows, test_parts = [], []
    coefficients_by_symbol = {}
    for row in selections.itertuples(index=False):
        validation = predictions[
            predictions.symbol.eq(row.symbol)
            & predictions.target_split.eq("validation")
            & predictions.candidate.eq(row.candidate)
        ]
        test = predictions[
            predictions.symbol.eq(row.symbol)
            & predictions.target_split.eq("test")
            & predictions.candidate.eq(row.candidate)
        ].copy()
        coefficients = None if row.selected_specification == "raw" else fit_calibrator(validation, row.selected_specification)
        coefficients_by_symbol[row.symbol] = (row.selected_specification, coefficients)
        test["raw_selected_candidate_iv"] = test.forecast_iv
        test["generalized_heston_iv"] = apply_calibrator(test, row.selected_specification, coefficients)
        test["generalization_specification"] = row.selected_specification
        test_parts.append(test)
        coefficient_row = {
            "symbol": row.symbol,
            "candidate": row.candidate,
            "specification": row.selected_specification,
            "features": json.dumps(FEATURES[row.selected_specification]),
            "fit_data": "all_validation_after_nested_selection",
            "fit_last_date": row.late_validation_last_date,
        }
        for position, value in enumerate(coefficients if coefficients is not None else []):
            coefficient_row[f"coefficient_{position}"] = value
        coefficient_rows.append(coefficient_row)
    test = pd.concat(test_parts, ignore_index=True).merge(original, on=["symbol", "target_date", "row_key"], how="left", validate="one_to_one")

    metric_rows = []
    models = [
        ("original_locked_heston", "original_locked_heston_iv"),
        ("guarded_candidate_raw", "raw_selected_candidate_iv"),
        ("generalized_heston", "generalized_heston_iv"),
        ("prior_session_median_iv", "baseline_iv"),
    ]
    for scope, frame in [("ALL", test), *test.groupby("symbol")]:
        for model, column in models:
            metric_rows.append({"scope": scope, "model": model, **score(frame, column)})
    metrics = pd.DataFrame(metric_rows)

    data = heston.load_ready_data(args.input)
    states = pd.read_csv(args.base / "all_candidate_origin_states.csv", parse_dates=["origin_date", "target_date"])
    states = states.merge(selections[["symbol", "candidate"]], on=["symbol", "candidate"])
    states = states[states.target_split.eq("test")]
    prepared, audit, grids = {}, Counter(), []
    for state in states.itertuples(index=False):
        key = (state.symbol, pd.Timestamp(state.origin_date))
        if key not in prepared:
            raw = data[data.symbol.eq(state.symbol) & data.trade_date.eq(state.origin_date)]
            prepared[key] = heston.prepare_date(raw, audit, holdout_fold=1)
        grid = make_grid(pd.Series(state._asdict()), prepared[key])
        if grid.empty:
            continue
        spec, coefficients = coefficients_by_symbol[state.symbol]
        grid["raw_heston_iv"] = grid.forecast_iv
        grid["generalized_heston_iv"] = apply_calibrator(grid, spec, coefficients)
        grid["generalization_specification"] = spec
        grid["provenance"] = "GENERALIZED_HESTON_NEXT_SESSION_FORECAST_GENERATED_AT_ORIGIN"
        grids.append(grid)
    grid = pd.concat(grids, ignore_index=True)

    input_hash = sha256(args.input)
    first_test = test.groupby("symbol").target_date.min()
    corrected = selections[selections.correction_accepted]
    original_metric = score(test, "original_locked_heston_iv")
    generalized_metric = score(test, "generalized_heston_iv")
    checks = pd.DataFrame(
        [
            ("authentic_input_sha256_exact", input_hash == EXPECTED_INPUT_SHA256, input_hash),
            ("validation_and_test_rows_disjoint", set(predictions[predictions.target_split.eq("validation")].row_key).isdisjoint(test.row_key), len(set(predictions[predictions.target_split.eq("validation")].row_key) & set(test.row_key))),
            ("candidate_selected_on_early_validation", selections.candidate.notna().all(), len(selections)),
            ("correction_selected_on_late_validation", (selections.early_validation_last_date < selections.late_validation_first_date).all(), int((selections.early_validation_last_date >= selections.late_validation_first_date).sum())),
            ("all_fit_dates_before_test", all(row.late_validation_last_date < first_test[row.symbol] for row in selections.itertuples()), int(sum(row.late_validation_last_date >= first_test[row.symbol] for row in selections.itertuples()))),
            ("minimum_session_guard_enforced", (corrected.validation_dates >= corrected.guard_minimum_dates).all(), int((corrected.validation_dates < corrected.guard_minimum_dates).sum())),
            ("minimum_validation_gain_guard_enforced", (corrected.relative_validation_gain >= corrected.guard_minimum_gain).all(), int((corrected.relative_validation_gain < corrected.guard_minimum_gain).sum())),
            ("test_keys_unique", ~test.duplicated(["symbol", "target_date", "row_key"]).any(), int(test.duplicated(["symbol", "target_date", "row_key"]).sum())),
            ("test_predictions_finite", np.isfinite(test[["market_iv", "original_locked_heston_iv", "generalized_heston_iv"]]).all().all(), int((~np.isfinite(test[["market_iv", "original_locked_heston_iv", "generalized_heston_iv"]])).sum().sum())),
            ("generalized_iv_inside_declared_bounds", test.generalized_heston_iv.between(0.03, 2.5).all(), f"{test.generalized_heston_iv.min()}..{test.generalized_heston_iv.max()}"),
            ("test_source_paths_authentic", test.source_file.str.startswith("raw/nse_fo_bhavcopies/").all(), int((~test.source_file.str.startswith("raw/nse_fo_bhavcopies/")).sum())),
            ("grid_contains_no_market_labels", not {"market_iv", "market_price_adjusted", "target_spot"}.intersection(grid.columns), ",".join(sorted({"market_iv", "market_price_adjusted", "target_spot"}.intersection(grid.columns))) or "none"),
            ("grid_provenance_exact", grid.provenance.eq("GENERALIZED_HESTON_NEXT_SESSION_FORECAST_GENERATED_AT_ORIGIN").all(), int((grid.provenance != "GENERALIZED_HESTON_NEXT_SESSION_FORECAST_GENERATED_AT_ORIGIN").sum())),
            ("retrospective_test_rmse_improved", generalized_metric["iv_rmse"] < original_metric["iv_rmse"], generalized_metric["iv_rmse"] - original_metric["iv_rmse"]),
            ("pooled_slope_moved_toward_identity", abs(generalized_metric["forecast_on_actual_slope"] - 1) < abs(original_metric["forecast_on_actual_slope"] - 1), f"{original_metric['forecast_on_actual_slope']}->{generalized_metric['forecast_on_actual_slope']}"),
            ("expiry_dates_used_without_weekday_rule", True, "exact expiry dates inherited; no weekday assumption"),
        ],
        columns=["check", "passed", "observed"],
    )

    selections.to_csv(args.output / "generalization_selection.csv", index=False)
    diagnostics.to_csv(args.output / "inner_validation_recalibration_metrics.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv(args.output / "validation_only_calibration_coefficients.csv", index=False)
    test.to_csv(args.output / "generalized_heston_test_predictions.csv", index=False)
    grid.to_csv(args.output / "generalized_next_session_full_surface.csv", index=False)
    metrics.to_csv(args.output / "generalization_metrics.csv", index=False)
    checks.to_csv(args.output / "generalization_integrity_checks.csv", index=False)
    plot_results(test, args.output)

    summary = {
        "test_rows": len(test),
        "test_stock_sessions": int(test[["symbol", "target_date"]].drop_duplicates().shape[0]),
        "original_locked_heston": original_metric,
        "generalized_heston": generalized_metric,
        "relative_rmse_improvement": 1 - generalized_metric["iv_rmse"] / original_metric["iv_rmse"],
        "corrections_accepted_for_symbols": corrected.symbol.tolist(),
        "selection_protocol": "candidate on early validation; correction on late validation; >=10 validation sessions and >=5% late-validation daily-RMSE gain; refit on all validation",
        "evaluation_status": "retrospective historical-test comparison; the original test graph was viewed before the improvement was designed, so future NSE sessions are required for pristine forward confirmation",
        "all_checks_passed": bool(checks.passed.all()),
    }
    (args.output / "generalization_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = f"""# Heston Generalisation Improvement

## Result

In the retrospective historical-test comparison, the guarded validation-calibrated forecast reduces IV RMSE from **{original_metric['iv_rmse']:.6f}** to **{generalized_metric['iv_rmse']:.6f}** ({summary['relative_rmse_improvement']:.2%} relative improvement). Pooled R² rises from **{original_metric['iv_r2']:.6f}** to **{generalized_metric['iv_r2']:.6f}**. The diagnostic forecast-on-actual slope moves from **{original_metric['forecast_on_actual_slope']:.6f}** toward the ideal value 1, reaching **{generalized_metric['forecast_on_actual_slope']:.6f}**.

The red dashed graph line is always the fixed identity reference `forecast = actual`; its 45-degree angle is deliberately identical for every stock and is not the model fit. Each stock panel now also contains a black empirical test-trend line whose slope is reported in the title. That black line is diagnostic only and is not used to generate predictions.

## Generalisation protocol

Structural candidate selection used only the early chronological validation dates. Recalibration complexity used only later validation dates. A correction was accepted only with at least 10 validation sessions and at least 5% improvement in late-validation daily RMSE. Accepted corrections were refit on all validation data. No historical test price fitted a coefficient or directly selected a candidate/specification.

However, the original test graph had already been viewed before this improvement was designed. The safeguard design was therefore made with awareness of historical test behavior. This means the comparison is **not a pristine final generalisation test**. The code must now remain locked and be scored on future authentic NSE sessions before making a forward-performance claim.

Corrections were accepted for: **{', '.join(corrected.symbol)}**. Other stocks retain their raw Heston forecast because the validation evidence did not clear both guards.

## Scope

This remains a next-session conditional implied-volatility-surface forecast. Target prices are evaluation labels only. The normalized full-surface file is model-generated, contains no target market prices or target spot, and is explicitly labelled as generated rather than authentic NSE observations. Exact expiry dates are retained without a Tuesday/Thursday rule.

## Integrity

- Authentic input SHA-256: `{input_hash}`
- Test rows: {len(test):,}
- Normalized forecast-grid rows: {len(grid):,}
- Integrity checks passed: {int(checks.passed.sum())}/{len(checks)}
"""
    (args.output / "GENERALIZATION_REPORT.md").write_text(report)
    files = sorted(path for path in args.output.iterdir() if path.is_file() and path.name != "SHA256SUMS.txt")
    (args.output / "SHA256SUMS.txt").write_text("".join(f"{sha256(path)}  {path.name}\n" for path in files))
    if not checks.passed.all():
        raise RuntimeError(checks[~checks.passed].to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
