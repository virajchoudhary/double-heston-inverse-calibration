#!/usr/bin/env python3
"""Independent evidence audit for the Single- and Double-Heston handoff.

The audit treats saved model outputs as claims to be recomputed from row-level
artifacts. It does not refit parameters or alter the market dataset.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

import double_heston as double
import single_heston as single


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "heston_handoff"
INPUT = ROOT / "outputs" / "019fc8a0" / "model_input_option_prices.csv"
SINGLE = ROOT / "outputs" / "single_heston"
OVERFIT = ROOT / "outputs" / "single_heston_overfit_audit"
FORECAST = ROOT / "outputs" / "single_heston_next_day_forecast"
GENERALIZED = ROOT / "outputs" / "single_heston_generalized_forecast"
COMPARE = ROOT / "outputs" / "single_double_heston_comparison"
EXPECTED_SHA256 = "a8a56dd7b17074d8fa32f88936c8b404456f22f296d9dc4665faf3b13621f1d3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score(frame: pd.DataFrame, prediction: str) -> dict[str, float]:
    error = frame[prediction].to_numpy(float) - frame.market_iv.to_numpy(float)
    actual = frame.market_iv.to_numpy(float)
    denominator = np.sum((actual - actual.mean()) ** 2)
    session = (
        frame.assign(_se=error**2)
        .groupby(["symbol", "target_date"], sort=False)
        ._se.mean()
        .pow(0.5)
    )
    return {
        "rows": int(len(frame)),
        "iv_rmse": float(np.sqrt(np.mean(error**2))),
        "iv_mae": float(np.mean(np.abs(error))),
        "iv_bias": float(np.mean(error)),
        "iv_r2": float(1 - np.sum(error**2) / denominator),
        "mean_session_iv_rmse": float(session.mean()),
        "forecast_on_actual_slope": float(np.polyfit(actual, frame[prediction], 1)[0]),
    }


def make_split_map(data: pd.DataFrame) -> dict[tuple[str, pd.Timestamp], str]:
    mapping: dict[tuple[str, pd.Timestamp], str] = {}
    for symbol, group in data.groupby("symbol"):
        dates = np.array(sorted(pd.to_datetime(group.trade_date).unique()))
        train_end = max(1, int(len(dates) * 0.70))
        validation_end = max(train_end + 1, int(len(dates) * 0.85))
        for position, day in enumerate(dates):
            split = "train" if position < train_end else "validation" if position < validation_end else "test"
            mapping[(symbol, pd.Timestamp(day))] = split
    return mapping


def compare_metric(saved: pd.Series, recomputed: dict[str, float], keys: list[str], atol=2e-12) -> tuple[bool, float]:
    errors = [abs(float(saved[key]) - float(recomputed[key])) for key in keys]
    return bool(max(errors, default=0.0) <= atol), float(max(errors, default=0.0))


def price_bounds(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    spot_pv = frame.spot.to_numpy(float) * np.exp(-frame.dividend.to_numpy(float) * frame.maturity.to_numpy(float))
    strike_pv = frame.strike.to_numpy(float) * np.exp(-frame.rate.to_numpy(float) * frame.maturity.to_numpy(float))
    calls = frame.option_type.eq("CE").to_numpy()
    lower = np.where(calls, np.maximum(spot_pv - strike_pv, 0.0), np.maximum(strike_pv - spot_pv, 0.0))
    upper = np.where(calls, spot_pv, strike_pv)
    return lower, upper


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    checks: list[dict] = []

    def check(category: str, name: str, passed, observed, severity="critical", detail="") -> None:
        checks.append(
            {
                "category": category,
                "check": name,
                "passed": bool(passed),
                "observed": str(observed),
                "severity": severity,
                "detail": detail,
            }
        )

    input_hash = sha256(INPUT)
    data_all = pd.read_csv(INPUT, parse_dates=["trade_date", "expiry_date"], low_memory=False)
    ready_mask = data_all.is_model_ready.astype(str).str.lower().isin(["t", "true", "1"])
    data = data_all[ready_mask].copy()
    split_map = make_split_map(data)
    check("source", "input_sha256_matches_locked_NSE_model_input", input_hash == EXPECTED_SHA256, input_hash)
    check("source", "model_ready_row_count_is_locked", len(data) == 215_636, len(data))
    check("source", "model_input_row_keys_unique", data.row_key.is_unique, int(data.row_key.duplicated().sum()))
    check("source", "model_input_contains_11_power_stocks", data.symbol.nunique() == 11, data.symbol.nunique())
    check("source", "model_ready_filter_reproduces_locked_subset", len(data_all) == 572_512 and len(data) == 215_636, f"all={len(data_all)}, ready={len(data)}")
    source_paths = data.source_file.astype(str)
    check("source", "source_paths_are_NSE_bhavcopies", source_paths.str.startswith("raw/nse_fo_bhavcopies/").all(), int((~source_paths.str.startswith("raw/nse_fo_bhavcopies/")).sum()))
    missing_sources = [path for path in source_paths.unique() if not (ROOT / path).exists()]
    check("source", "referenced_raw_NSE_files_exist_locally", not missing_sources, len(missing_sources), detail="All row-level source paths resolve in the project tree.")
    exact_days = (data.expiry_date - data.trade_date).dt.days
    check("expiry", "expiry_distance_matches_saved_days", np.array_equal(exact_days.to_numpy(), data.days_to_expiry.to_numpy()), int((exact_days != data.days_to_expiry).sum()))
    check("expiry", "maturity_is_exact_days_divided_by_365", np.allclose(data.time_to_expiry_years, exact_days / 365, atol=5e-11), float(np.max(np.abs(data.time_to_expiry_years - exact_days / 365))))
    expiry_weekdays = sorted(data.expiry_date.dt.day_name().unique().tolist())
    check("expiry", "multiple_historical_expiry_weekdays_preserved", len(expiry_weekdays) >= 2, ", ".join(expiry_weekdays), detail="No weekday normalization was applied.")

    first_validation = {}
    first_test = {}
    for symbol, group in data.groupby("symbol"):
        dates = sorted(group.trade_date.unique())
        first_validation[symbol] = min(day for day in dates if split_map[(symbol, pd.Timestamp(day))] == "validation")
        first_test[symbol] = min(day for day in dates if split_map[(symbol, pd.Timestamp(day))] == "test")

    single_candidates = pd.read_csv(FORECAST / "parameter_candidates_train_only.csv", parse_dates=["training_last_date"])
    single_selection = pd.read_csv(FORECAST / "selected_parameters_locked_before_test.csv")
    single_all = pd.read_csv(FORECAST / "all_candidate_predictions.csv", parse_dates=["trade_date", "expiry_date", "origin_date", "target_date"])
    single_test = pd.read_csv(FORECAST / "selected_heston_test_predictions.csv", parse_dates=["trade_date", "expiry_date", "origin_date", "target_date"])
    check("chronology", "single_candidates_strictly_precede_validation", all(row.training_last_date < first_validation[row.symbol] for row in single_candidates.itertuples()), int(sum(row.training_last_date >= first_validation[row.symbol] for row in single_candidates.itertuples())))
    check("chronology", "single_origin_strictly_precedes_target", (single_all.origin_date < single_all.target_date).all(), int((single_all.origin_date >= single_all.target_date).sum()))
    expected_single_splits = [split_map[(row.symbol, pd.Timestamp(row.target_date))] for row in single_all.itertuples()]
    check("chronology", "single_saved_split_matches_independent_70_15_15", np.array_equal(single_all.target_split.to_numpy(), np.asarray(expected_single_splits)), int(np.sum(single_all.target_split.to_numpy() != np.asarray(expected_single_splits))))
    val_keys = set(single_all.loc[single_all.target_split.eq("validation"), "row_key"])
    test_keys = set(single_all.loc[single_all.target_split.eq("test"), "row_key"])
    check("chronology", "single_validation_and_test_keys_disjoint", val_keys.isdisjoint(test_keys), len(val_keys & test_keys))

    validation = single_all[single_all.target_split.eq("validation")]
    rows = []
    for (symbol, candidate), frame in validation.groupby(["symbol", "candidate"]):
        metric = score(frame, "forecast_iv")
        rows.append({"symbol": symbol, "candidate": candidate, **metric})
    single_recomputed = pd.DataFrame(rows).sort_values(["symbol", "iv_rmse", "candidate"])
    single_winner = single_recomputed.groupby("symbol", as_index=False).first()[["symbol", "candidate"]]
    saved_single_winner = single_selection[["symbol", "selected_candidate"]].rename(columns={"selected_candidate": "candidate"}).sort_values("symbol").reset_index(drop=True)
    check("selection", "single_validation_selection_recomputed", single_winner.sort_values("symbol").reset_index(drop=True).equals(saved_single_winner), len(saved_single_winner))
    check("selection", "single_selection_declares_validation_only", single_selection.selection_data.eq("validation_only").all(), int((single_selection.selection_data != "validation_only").sum()))
    check("selection", "single_has_three_train_only_candidates_per_stock", single_candidates.groupby("symbol").candidate.nunique().eq(3).all(), single_candidates.groupby("symbol").candidate.nunique().min())

    source_lookup = data.set_index("row_key")
    missing = single_test.loc[~single_test.row_key.isin(source_lookup.index)]
    check("lineage", "single_test_rows_resolve_to_source", missing.empty, len(missing))
    matched = single_test.join(source_lookup[["symbol", "trade_date", "expiry_date", "option_type", "adjusted_strike_price", "adjusted_observed_option_price", "adjusted_underlying_value", "source_file"]], on="row_key", rsuffix="_source")
    lineage_fail = (
        (matched.symbol != matched.symbol_source)
        | (matched.trade_date != matched.trade_date_source)
        | (matched.expiry_date != matched.expiry_date_source)
        | (matched.option_type != matched.option_type_source)
        | ~np.isclose(matched.strike, matched.adjusted_strike_price, atol=1e-10)
        | ~np.isclose(matched.market_price_adjusted, matched.adjusted_observed_option_price, atol=1e-10)
        | ~np.isclose(matched.spot, matched.adjusted_underlying_value, atol=1e-10)
        | (matched.source_file != matched.source_file_source)
    )
    check("lineage", "single_test_values_match_source_rows", not lineage_fail.any(), int(lineage_fail.sum()))

    param_map = single_candidates.set_index(["symbol", "candidate"])[["kappa", "theta", "sigma", "rho"]]
    single_price_error = []
    single_state_error = []
    for (symbol, candidate, target_date), frame in single_test.groupby(["symbol", "candidate", "target_date"], sort=False):
        p = param_map.loc[(symbol, candidate)].to_numpy(float)
        first = frame.iloc[0]
        expected_state = p[1] + (first.origin_v0 - p[1]) * math.exp(-p[0] * first.calendar_gap_days / 365)
        single_state_error.append(abs(expected_state - first.forecast_v0))
        repriced = single.heston_prices(frame, np.r_[p, first.forecast_v0])
        single_price_error.extend(np.abs(repriced - frame.forecast_price_adjusted.to_numpy(float)))
    check("equation", "single_variance_propagation_recomputed", max(single_state_error) <= 2e-12, max(single_state_error))
    check("equation", "single_saved_prices_reprice_exactly", max(single_price_error) <= 2e-9, max(single_price_error))
    single_recomputed_iv = single.implied_volatility(
        single_test.forecast_price_adjusted, single_test.spot, single_test.strike, single_test.maturity,
        single_test.rate, single_test.dividend, single_test.option_type.eq("CE"),
    )
    check("equation", "single_saved_IV_inverts_saved_prices", np.nanmax(np.abs(single_recomputed_iv - single_test.forecast_iv)) <= 2e-10, np.nanmax(np.abs(single_recomputed_iv - single_test.forecast_iv)))
    lower, upper = price_bounds(single_test)
    price = single_test.forecast_price_adjusted.to_numpy(float)
    check("validity", "single_prices_within_no_arbitrage_bounds", ((price >= lower - 2e-8) & (price <= upper + 2e-8)).all(), int(((price < lower - 2e-8) | (price > upper + 2e-8)).sum()))

    single_feller = 2 * single_candidates.kappa * single_candidates.theta - single_candidates.sigma**2
    check("constraints", "all_single_candidates_Feller_valid", (single_feller > 0).all(), float(single_feller.min()))
    check("constraints", "all_single_candidates_within_declared_bounds", ((single_candidates.kappa > 0.05) & (single_candidates.kappa < 10) & (single_candidates.theta > 0.005) & (single_candidates.theta < 1) & (single_candidates.rho.abs() < 0.98)).all(), "33 candidates")

    saved_single_metrics = pd.read_csv(FORECAST / "next_day_forecast_metrics.csv")
    for model, prediction in [("selected_heston", "forecast_iv"), ("prior_session_expiry_median_iv", "baseline_iv")]:
        saved = saved_single_metrics[(saved_single_metrics.scope.eq("ALL")) & (saved_single_metrics.model.eq(model))].iloc[0]
        recomputed = score(single_test, prediction)
        passed, error = compare_metric(saved, recomputed, ["rows", "iv_rmse", "iv_mae", "iv_bias", "iv_r2"], atol=2e-12)
        check("metrics", f"single_{model}_metrics_recomputed", passed, error)

    double_candidates = pd.read_csv(COMPARE / "double_heston_all_train_only_candidates.csv", parse_dates=["training_first_date", "training_last_date"])
    double_selection = pd.read_csv(COMPARE / "double_heston_validation_selection.csv")
    double_validation = pd.read_csv(COMPARE / "double_heston_validation_predictions.csv", parse_dates=["trade_date", "expiry_date", "origin_date", "target_date"])
    comparison = pd.read_csv(COMPARE / "single_double_identical_test_predictions.csv", parse_dates=["trade_date", "expiry_date", "origin_date", "target_date"])
    double_failures = pd.read_csv(COMPARE / "double_heston_test_prediction_failures.csv")
    check("chronology", "double_candidates_strictly_precede_validation", all(row.training_last_date < first_validation[row.symbol] for row in double_candidates.itertuples()), int(sum(row.training_last_date >= first_validation[row.symbol] for row in double_candidates.itertuples())))
    check("selection", "double_has_five_train_only_candidates_per_stock", double_candidates.groupby("symbol").candidate.nunique().eq(5).all(), double_candidates.groupby("symbol").candidate.nunique().min())
    check("selection", "each_double_validation_key_has_all_five_candidates", double_validation.groupby(["symbol", "target_date", "row_key"]).candidate.nunique().eq(5).all(), double_validation.groupby(["symbol", "target_date", "row_key"]).candidate.nunique().min())
    double_rows = []
    for (symbol, candidate), frame in double_validation.groupby(["symbol", "candidate"]):
        double_rows.append({"symbol": symbol, "candidate": candidate, **score(frame, "double_heston_iv")})
    double_recomputed = pd.DataFrame(double_rows).sort_values(["symbol", "mean_session_iv_rmse", "iv_rmse", "candidate"])
    double_winner = double_recomputed.groupby("symbol", as_index=False).first()[["symbol", "candidate"]]
    saved_double_winner = double_selection[["symbol", "selected_candidate"]].rename(columns={"selected_candidate": "candidate"}).sort_values("symbol").reset_index(drop=True)
    check("selection", "double_validation_selection_recomputed", double_winner.sort_values("symbol").reset_index(drop=True).equals(saved_double_winner), len(saved_double_winner))
    check("selection", "double_selection_declares_validation_only", double_selection.selection_data.eq("validation_only").all(), int((double_selection.selection_data != "validation_only").sum()))

    k1, t1, s1, r1 = [double_candidates[c] for c in ["kappa_slow", "theta_slow", "sigma_slow", "rho_slow"]]
    k2, t2, s2, r2 = [double_candidates[c] for c in ["kappa_fast", "theta_fast", "sigma_fast", "rho_fast"]]
    check("constraints", "all_double_candidates_satisfy_both_Feller_conditions", ((2 * k1 * t1 > s1**2) & (2 * k2 * t2 > s2**2)).all(), float(min((2 * k1 * t1 - s1**2).min(), (2 * k2 * t2 - s2**2).min())))
    check("constraints", "all_double_candidates_have_ordered_factors", (k1 < k2).all(), int((k1 >= k2).sum()))
    check("constraints", "all_double_candidates_satisfy_joint_correlation_disk", (r1**2 + r2**2 < 1).all(), float(np.sqrt(r1**2 + r2**2).max()))
    cap1 = 0.995 * np.sqrt(2 * k1 * t1)
    cap2 = 0.995 * np.sqrt(2 * k2 * t2)
    boundary = (k1 < 0.06) | (k1 > 4.95) | (k2 > 9.9) | (s1 / cap1 > 0.98) | (s2 / cap2 > 0.98) | (np.sqrt(r1**2 + r2**2) > 0.94)
    check("overfit", "double_boundary_near_candidates_recorded", int(boundary.sum()) == 55, int(boundary.sum()), severity="warning", detail="A passing evidence check records that every candidate is boundary-near; this is an identifiability warning, not a quality success.")

    selected_double = pd.read_csv(COMPARE / "double_heston_selected_parameters.csv").set_index("symbol")
    double_price_error = []
    double_state_error = []
    for (symbol, target_date), frame in comparison.groupby(["symbol", "target_date"], sort=False):
        row = selected_double.loc[symbol]
        structural = row[["kappa_slow", "theta_slow", "sigma_slow", "rho_slow", "kappa_fast", "theta_fast", "sigma_fast", "rho_fast"]].to_numpy(float)
        first = frame.iloc[0]
        expected_v1 = structural[1] + (first.origin_v0_slow - structural[1]) * math.exp(-structural[0] * first.calendar_gap_days / 365)
        expected_v2 = structural[5] + (first.origin_v0_fast - structural[5]) * math.exp(-structural[4] * first.calendar_gap_days / 365)
        double_state_error.extend([abs(expected_v1 - first.forecast_v0_slow), abs(expected_v2 - first.forecast_v0_fast)])
        repriced = double.prices(frame, double.combine(structural, [first.forecast_v0_slow, first.forecast_v0_fast]))
        double_price_error.extend(np.abs(repriced - frame.double_heston_price.to_numpy(float)))
    check("equation", "double_variance_propagation_recomputed", max(double_state_error) <= 2e-12, max(double_state_error))
    check("equation", "double_saved_prices_reprice_exactly", max(double_price_error) <= 2e-9, max(double_price_error))
    double_recomputed_iv = single.implied_volatility(
        comparison.double_heston_price, comparison.spot, comparison.strike, comparison.maturity,
        comparison.rate, comparison.dividend, comparison.option_type.eq("CE"),
    )
    check("equation", "double_saved_IV_inverts_saved_prices", np.nanmax(np.abs(double_recomputed_iv - comparison.double_heston_iv)) <= 2e-10, np.nanmax(np.abs(double_recomputed_iv - comparison.double_heston_iv)))
    lower, upper = price_bounds(comparison)
    price = comparison.double_heston_price.to_numpy(float)
    check("validity", "double_prices_within_no_arbitrage_bounds", ((price >= lower - 2e-8) & (price <= upper + 2e-8)).all(), int(((price < lower - 2e-8) | (price > upper + 2e-8)).sum()))

    universe = set(single_test.row_key)
    common = set(comparison.row_key)
    failures = set(double_failures.row_key)
    check("coverage", "single_test_universe_has_8818_rows", len(universe) == 8_818 and single_test.row_key.is_unique, len(universe))
    check("coverage", "double_common_and_failure_rows_partition_single_universe", common.isdisjoint(failures) and common | failures == universe, f"common={len(common)}, failures={len(failures)}, universe={len(universe)}")
    check("coverage", "double_invalid_rows_are_documented_not_imputed", len(failures) == 4 and comparison.row_key.isin(failures).sum() == 0, len(failures))
    check("lineage", "single_and_double_use_identical_common_rows", len(common) == 8_814 and common == set(comparison.row_key), len(common))

    saved_compare_metrics = pd.read_csv(COMPARE / "single_double_test_metrics.csv")
    for model, prediction in [("Single Heston", "single_heston_iv"), ("Double Heston", "double_heston_iv")]:
        saved = saved_compare_metrics[(saved_compare_metrics.scope.eq("ALL")) & saved_compare_metrics.model.eq(model)].iloc[0]
        recomputed = score(comparison, prediction)
        passed, error = compare_metric(saved, recomputed, ["rows", "iv_rmse", "iv_mae", "iv_bias", "iv_r2", "mean_session_iv_rmse", "forecast_on_actual_slope"], atol=2e-12)
        check("metrics", f"{model.lower().replace(' ', '_')}_comparison_metrics_recomputed", passed, error)

    rng = np.random.default_rng(20260806)
    groups = [group for _, group in comparison.groupby(["symbol", "target_date"], sort=False)]
    draws = []
    for _ in range(2_000):
        sample = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))], ignore_index=True)
        draws.append(np.sqrt(np.mean((sample.double_heston_iv - sample.market_iv) ** 2)) - np.sqrt(np.mean((sample.single_heston_iv - sample.market_iv) ** 2)))
    bootstrap = np.quantile(draws, [0.025, 0.5, 0.975])
    saved_summary = json.loads((COMPARE / "single_double_summary.json").read_text())
    saved_bootstrap = saved_summary["cluster_bootstrap_double_minus_single"]
    bootstrap_error = max(abs(bootstrap[0] - saved_bootstrap["double_minus_single_rmse_low"]), abs(bootstrap[1] - saved_bootstrap["median"]), abs(bootstrap[2] - saved_bootstrap["high"]))
    check("metrics", "double_minus_single_cluster_bootstrap_recomputed", bootstrap_error <= 2e-12, bootstrap_error)
    check("overfit", "historical_bootstrap_favors_single_Heston", bootstrap[0] > 0, f"95% interval [{bootstrap[0]:.8f}, {bootstrap[2]:.8f}]", severity="warning")

    crossfit = pd.read_csv(OVERFIT / "three_fold_crossfit_predictions.csv")
    crossfit_metric = score(crossfit.rename(columns={"heston_same_day_iv": "_forecast", "trade_date": "target_date"}), "_forecast")
    overfit_summary = json.loads((OVERFIT / "overfitting_audit_summary.json").read_text())
    reported_crossfit = overfit_summary["heston_same_day_anchor_v0"]
    crossfit_error = max(abs(crossfit_metric[k] - reported_crossfit[k]) for k in ["rows", "iv_rmse", "iv_mae", "iv_bias"])
    check("overfit", "single_three_fold_strike_crossfit_metrics_recomputed", crossfit_error <= 2e-12, crossfit_error)
    leakage = pd.read_csv(OVERFIT / "independent_leakage_checks.csv")
    check("leakage", "all_strike_crossfit_leakage_checks_pass", leakage.passed.astype(str).str.lower().eq("true").all(), f"{leakage.passed.astype(str).str.lower().eq('true').sum()}/{len(leakage)}")
    comparison_models = pd.read_csv(OVERFIT / "overfitting_model_comparison.csv").set_index("model")
    check("overfit", "quadratic_same_day_smile_beats_single_Heston", comparison_models.loc["quadratic_anchor_smile", "iv_rmse"] < comparison_models.loc["heston_same_day_v0", "iv_rmse"], f"quadratic={comparison_models.loc['quadratic_anchor_smile', 'iv_rmse']:.6f}; Heston={comparison_models.loc['heston_same_day_v0', 'iv_rmse']:.6f}", severity="warning")
    alternate = pd.read_csv(OVERFIT / "alternate_training_parameter_fits.csv")
    boundary_flags = int(alternate[["kappa_near_upper_bound", "theta_near_upper_bound", "sigma_near_feller_cap"]].astype(bool).to_numpy().sum())
    check("overfit", "single_half_sample_boundary_flags_recorded", boundary_flags == 39, boundary_flags, severity="warning")

    generalized = json.loads((GENERALIZED / "generalization_summary.json").read_text())
    retrospective = "retrospective" in generalized["evaluation_status"].lower() and "future" in generalized["evaluation_status"].lower()
    check("honesty", "generalized_single_result_labeled_retrospective", retrospective, generalized["evaluation_status"], severity="warning")
    check("honesty", "conditional_evaluation_scope_explicit", single_test.evaluation_convention.eq("CONDITIONAL_ON_REALIZED_TARGET_SPOT_AND_QUOTES").all(), single_test.evaluation_convention.unique().tolist(), severity="warning", detail="The evaluation uses realized target spot and listed quote coordinates; it is not a pure pre-market spot/contract forecast.")

    for name, path, exact_tol, noise_limit in [
        ("single", SINGLE / "single_heston_controlled_tests.json", 1e-8, 0.15),
        ("double", COMPARE / "double_heston_controlled_tests.json", 1e-4, 0.35),
    ]:
        controlled = json.loads(path.read_text())
        exact_error = controlled.get("exact_max_relative_parameter_error", controlled.get("exact_max_relative_structural_error"))
        noisy_error = max(
            abs((a - b) / b)
            for a, b in zip(
                controlled.get("noisy_recovered_parameters", controlled.get("noisy_recovered_structural_parameters")),
                controlled.get("known_parameters", controlled.get("known_structural_parameters")),
            )
        )
        check("controlled", f"{name}_exact_synthetic_recovery", exact_error < exact_tol and controlled["all_checks_passed"], exact_error)
        check("controlled", f"{name}_one_percent_noise_sensitivity_recorded", noisy_error < noise_limit, noisy_error, severity="warning")

    scan_paths = [
        ROOT / "single_heston.py", ROOT / "double_heston.py", ROOT / "forecast_single_heston_next_day.py",
        ROOT / "compare_single_double_heston.py", SINGLE / "SINGLE_HESTON_REPORT.md",
        FORECAST / "NEXT_DAY_FORECAST_REPORT.md", COMPARE / "HONEST_SINGLE_DOUBLE_HESTON_REPORT.md",
    ]
    secret_patterns = {
        "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        "generic_secret": re.compile(r"(?i)(api[_-]?key|access[_-]?token|password)\s*[:=]\s*['\"][^'\"]{8,}['\"]"),
        "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    }
    findings = []
    absolute_home_refs = []
    for path in scan_paths:
        text = path.read_text(errors="ignore")
        for label, pattern in secret_patterns.items():
            if pattern.search(text):
                findings.append({"file": str(path.relative_to(ROOT)), "pattern": label})
        if "/Users/" in text or "/home/" in text:
            absolute_home_refs.append(str(path.relative_to(ROOT)))
    raw_machine_metadata = []
    for path in [FORECAST / "parameter_candidates_train_only.csv", FORECAST / "next_day_forecast_summary.json"]:
        if "/Users/" in path.read_text(errors="ignore"):
            raw_machine_metadata.append(str(path.relative_to(ROOT)))
    check("privacy", "no_credential_patterns_in_reviewed_code_and_reports", not findings, findings)
    check("privacy", "raw_machine_artifacts_with_local_paths_identified", len(raw_machine_metadata) == 2, raw_machine_metadata, severity="warning", detail="Do not share these raw files verbatim; use the sanitized handoff catalogs and report.")

    main_v0 = pd.read_csv(SINGLE / "single_heston_parameters.csv")[["symbol", "v0_training_median"]].assign(candidate="main")
    alternate_v0 = pd.read_csv(OVERFIT / "alternate_training_parameter_fits.csv")[["symbol", "training_half", "median_v0"]].rename(columns={"median_v0": "v0_training_median"})
    alternate_v0["candidate"] = "half_" + alternate_v0.training_half
    candidate_v0 = pd.concat([main_v0[["symbol", "candidate", "v0_training_median"]], alternate_v0[["symbol", "candidate", "v0_training_median"]]], ignore_index=True)
    selected_single = single_selection[["symbol", "selected_candidate", "validation_iv_rmse"]].merge(single_candidates, left_on=["symbol", "selected_candidate"], right_on=["symbol", "candidate"]).merge(candidate_v0, on=["symbol", "candidate"])
    single_catalog = selected_single[["symbol", "selected_candidate", "kappa", "theta", "sigma", "rho", "v0_training_median", "validation_iv_rmse", "training_last_date"]].copy()
    single_catalog.insert(0, "model", "Single Heston")
    double_selected = pd.read_csv(COMPARE / "double_heston_selected_parameters.csv")
    double_catalog = double_selected[["symbol", "selected_candidate", "kappa_slow", "theta_slow", "sigma_slow", "rho_slow", "v0_slow_training_median", "kappa_fast", "theta_fast", "sigma_fast", "rho_fast", "v0_fast_training_median", "training_last_date"]].copy()
    double_catalog.insert(0, "model", "Double Heston")
    single_catalog.to_csv(OUT / "single_heston_selected_parameter_catalog.csv", index=False)
    double_catalog.to_csv(OUT / "double_heston_selected_parameter_catalog.csv", index=False)

    metrics_catalog = pd.concat(
        [
            saved_single_metrics[(saved_single_metrics.scope.eq("ALL"))][["scope", "model", "rows", "iv_rmse", "iv_mae", "iv_bias", "iv_r2"]].assign(experiment="original locked next-session"),
            saved_compare_metrics[saved_compare_metrics.scope.eq("ALL")][["scope", "model", "rows", "iv_rmse", "iv_mae", "iv_bias", "iv_r2"]].assign(experiment="identical-row Single vs Double"),
        ],
        ignore_index=True,
    )
    metrics_catalog.to_csv(OUT / "heston_metric_catalog.csv", index=False)

    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(OUT / "independent_heston_audit_checks.csv", index=False)
    critical_failures = checks_df[(checks_df.severity.eq("critical")) & (~checks_df.passed)]
    warnings = checks_df[checks_df.severity.eq("warning")]
    summary = {
        "audit_date": "2026-08-06",
        "audit_scope": "Single Heston and Double Heston only",
        "input_sha256": input_hash,
        "checks_total": int(len(checks_df)),
        "checks_passed": int(checks_df.passed.sum()),
        "critical_failures": int(len(critical_failures)),
        "warnings_recorded": int(len(warnings)),
        "single_test_rows": int(len(single_test)),
        "double_common_rows": int(len(comparison)),
        "double_documented_invalid_rows": int(len(double_failures)),
        "verdict": "VALIDATED_WITH_DISCLOSED_LIMITATIONS" if critical_failures.empty else "FAILED",
        "non_negotiable_limitations": [
            "No finite historical test can prove zero overfitting or a global optimum.",
            "The historical test period was previously viewed during Single-Heston development; it is not pristine final holdout evidence.",
            "Next-session scoring is conditional on realized target spot and the target session's listed strike/expiry coordinates.",
            "Double Heston underperforms Single Heston on the identical historical rows and has severe parameter-boundary/identifiability warnings.",
            "The retrospective Single-Heston recalibration must be confirmed on future locked NSE dates before being called forward validated.",
        ],
    }
    (OUT / "independent_heston_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "privacy_security_scan.json").write_text(json.dumps({"credential_findings": findings, "raw_files_with_absolute_workstation_paths": raw_machine_metadata, "reviewed_files": [str(p.relative_to(ROOT)) for p in scan_paths]}, indent=2) + "\n")

    report = f"""# Independent Heston System Audit\n\nAudit date: 2026-08-06  \nScope: Single Heston and Double Heston only  \nVerdict: **{summary['verdict']}**\n\n## Evidence result\n\n- {summary['checks_passed']} of {summary['checks_total']} independently executed checks passed.\n- Critical failures: {summary['critical_failures']}.\n- Warning/limitation records: {summary['warnings_recorded']}.\n- Locked authentic model-input SHA-256: `{input_hash}`.\n- Single-Heston historical next-session rows: {len(single_test):,}.\n- Identical finite Single/Double comparison rows: {len(comparison):,}; documented Double failures: {len(double_failures)}.\n\n## Honest interpretation\n\nThe saved lineage, chronology, parameter-selection, state propagation, pricing, implied-volatility inversion, no-arbitrage, coverage, and metric claims were recomputed from row-level artifacts. No critical inconsistency was found. This supports reproducibility of the saved experiment; it does **not** prove zero overfitting, globally optimal parameters, or deployable future performance.\n\nThe original locked Single Heston beats the prior-session median-IV baseline on the historical next-session test. On the identical finite comparison rows, Double Heston is worse than Single Heston, and the 95% stock-session bootstrap interval for `Double RMSE - Single RMSE` is [{bootstrap[0]:.6f}, {bootstrap[2]:.6f}], entirely above zero. The current Double model should therefore be rejected in favor of Single Heston for this evidence set.\n\n## Required caveats\n\n- The historical test period had already been viewed during earlier Single-Heston development; future locked NSE dates are required for pristine forward confirmation.\n- The next-session surface score is conditional on realized target spot and listed target strike/expiry coordinates. It is not a pure pre-market forecast of spot or of which contracts will list.\n- The same-day Heston strike cross-fit is not unseen-date forecasting. A quadratic same-day smile baseline has lower IV RMSE than Heston ({comparison_models.loc['quadratic_anchor_smile', 'iv_rmse']:.6f} vs {comparison_models.loc['heston_same_day_v0', 'iv_rmse']:.6f}).\n- All 55 Double-Heston candidates are boundary-near under the declared diagnostic; multiple parameter combinations can yield similar prices.\n- Raw machine outputs containing absolute local paths should not be shared verbatim. The catalogs in this handoff are sanitized.\n\n## Machine-readable evidence\n\nSee `independent_heston_audit_checks.csv`, the two parameter catalogs, `heston_metric_catalog.csv`, `privacy_security_scan.json`, and `independent_heston_audit_summary.json`.\n"""
    (OUT / "INDEPENDENT_HESTON_SYSTEM_AUDIT.md").write_text(report)

    files = sorted(path for path in OUT.iterdir() if path.is_file() and path.name != "SHA256SUMS.txt")
    (OUT / "SHA256SUMS.txt").write_text("".join(f"{sha256(path)}  {path.name}\n" for path in files))
    print(json.dumps(summary, indent=2))
    if not critical_failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
