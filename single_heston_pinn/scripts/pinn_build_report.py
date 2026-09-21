#!/usr/bin/env python3
"""Assemble the PINN study: strict checks, figures, surfaces and the write-up."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import mlx.core as mx

import pinn_heston_core as C
import pinn_calibrate as K
import pinn_report as P
import pinn_run_training as R
import pinn_train as T

ROOT = Path(__file__).resolve().parent
DEFAULT_DIR = ROOT / "outputs" / "pinn_single_heston"
PUBLISHED = ROOT / "outputs" / "single_heston" / "single_heston_run_summary.json"


def build_checks(directory, box, collocation, spec, training_report, score, pde,
                 predictions, parameters, recovery, arb, quotes_above_box=0):
    ceiling_ok = bool((collocation["spot"] >= 0).all()
                      and (collocation["spot"] <= collocation["spot_ceiling"] + 1e-9).all())
    slices_ok = all(np.isclose(collocation["maturity_days"], d, atol=1e-6).sum() > 100
                    for d in C.MONTH_SLICE_DAYS)
    variance = spec["variance_domain"]
    rows = [
        ("collocation_budget_inside_14000_to_20000",
         14000 <= collocation["x"].size <= 20000, int(collocation["x"].size)),
        ("every_collocation_spot_inside_zero_to_1p5x_ten_year_maximum",
         ceiling_ok, float(np.max(collocation["spot"] / collocation["spot_ceiling"]))),
        ("maturity_axis_inside_nse_three_month_stock_option_cycle",
         bool(collocation["maturity_days"].max() <= 92.0 + 1e-9),
         float(collocation["maturity_days"].max())),
        ("one_two_and_three_month_slices_present", slices_ok,
         [int(np.isclose(collocation["maturity_days"], d, atol=1e-6).sum()) for d in C.MONTH_SLICE_DAYS]),
        ("variance_axis_covers_observed_inverse_bsm_range_to_the_99p99th_percentile",
         bool(box.variance_low <= variance["observed_min"]
              and box.variance_high >= variance["observed_q999"]),
         {"box": [box.variance_low, box.variance_high],
          "observed_min_q999_max": [variance["observed_min"], variance["observed_q999"],
                                    variance["observed_max"]],
          "quotes_above_box": int(quotes_above_box)}),
        ("terminal_condition_exact_by_construction", True,
         "c = Black76(x, w), w -> 0 as T -> 0"),
        ("pinn_prices_inside_static_no_arbitrage_bounds",
         bool(arb["violations"] == 0), int(arb["violations"])),
        ("no_calendar_spread_arbitrage_on_collocation_set",
         pde["calendar_violation_fraction"] == 0.0, pde["calendar_violation_fraction"]),
        ("no_butterfly_arbitrage_on_collocation_set",
         pde["butterfly_violation_fraction"] == 0.0, pde["butterfly_violation_fraction"]),
        ("pde_residual_below_acceptance_gate",
         pde["residual_rms_price_relevant_core"] <= R.GATES["pde_core_residual_rms"],
         pde["residual_rms_price_relevant_core"]),
        ("pinn_matches_exact_heston_in_traded_region",
         score["traded_region"]["iv_rmse"] <= R.GATES["traded_iv_rmse"],
         score["traded_region"]["iv_rmse"]),
        ("all_training_acceptance_gates_passed",
         bool(training_report["rounds"][-1]["all_passed"]),
         [k for k, g in training_report["rounds"][-1]["gates"].items() if not g["passed"]]),
    ]
    if len(predictions):
        test = predictions[(predictions.split == "test") & (predictions.fold == "holdout")]
        calib = predictions[predictions.fold == "calibration"]
        overlap = len(set(test.row_key) & set(calib.row_key))
        future = int((pd.to_datetime(parameters.training_last_date)
                      >= pd.to_datetime(parameters.test_first_date)).sum())
        rows += [
            ("test_row_keys_absent_from_calibration_fold", overlap == 0, overlap),
            ("structural_parameters_fitted_on_train_dates_only", future == 0, future),
            ("every_test_prediction_traces_to_an_official_nse_file",
             bool(test.source_file.str.startswith("raw/nse_fo_bhavcopies/").all()),
             int((~test.source_file.str.startswith("raw/nse_fo_bhavcopies/")).sum())),
            ("all_calibrated_correlations_inside_the_unit_interval",
             bool(parameters.rho.abs().lt(1).all()), float(parameters.rho.abs().max())),
            ("all_calibrated_variance_parameters_positive",
             bool((parameters[["theta", "v0_training_median"]] > 0).all().all()),
             float(parameters[["theta", "v0_training_median"]].min().min())),
        ]
    if recovery is not None and len(recovery):
        rows.append(("synthetic_surfaces_repriced_within_0p25pct_of_spot",
                     bool(recovery.price_rmse_pct_of_spot.max() < 0.25),
                     float(recovery.price_rmse_pct_of_spot.max())))
    return pd.DataFrame(rows, columns=["check", "passed", "observed"])


def arbitrage_audit(engine, box, n=40000, seed=808):
    """No predicted price may leave the static no-arbitrage band."""
    rng = np.random.default_rng(seed)
    pts = C.sample_anchor(n, box, seed=seed)
    spot = np.full(n, 500.0)
    strike = spot * np.exp(-pts["x"])
    rate = np.zeros(n); dividend = np.zeros(n)
    price = engine.price_rowwise(spot, strike, pts["maturity"], rate, dividend,
                                 np.ones(n, dtype=bool), pts["kappa"], pts["theta"],
                                 pts["sigma"], pts["rho"], pts["variance"])
    lower = np.maximum(spot - strike, 0.0)
    # The bound is scale-free, so the tolerance must be too: MLX prices on the
    # GPU in float32, and a deep in-the-money call whose strike is 16,000 rupees
    # carries about 1e-3 rupees of representation error on a price of the same
    # order. An absolute 1e-8 rupee tolerance measures float32 rounding, not
    # arbitrage.
    scale = np.maximum(spot, strike)
    tol = 1e-6 * scale
    bad = (price < lower - tol) | (price > spot + tol) | ~np.isfinite(price)
    return {"points": int(n), "violations": int(bad.sum()),
            "tolerance_rule": "1e-6 x max(spot, strike), the float32 representation floor",
            "worst_lower_breach_relative": float(np.max((lower - price) / scale)),
            "worst_upper_breach_relative": float(np.max((price - spot) / scale)),
            "violations_at_absolute_1e-8": int(((price < lower - 1e-8)
                                                | (price > spot + 1e-8)).sum())}


def engine_attribution(pinn, fourier, directory):
    """Split the PINN's market error into network error and calibration error.

    The PINN-calibrated parameters are re-priced with the exact engine.  If the
    two agree, the residual market error belongs to the Heston model and the
    optimiser, not to the network approximation.
    """
    par = pd.read_csv(directory / "calibration_parameters_pinn.csv").set_index("symbol")
    states = pd.read_csv(directory / "calibration_daily_state_pinn.csv")
    pred = pd.read_parquet(directory / "calibration_predictions_pinn.parquet")
    test = pred[(pred.split == "test") & (pred.fold == "holdout")]
    rows = []
    for symbol, g in test.groupby("symbol"):
        p = par.loc[symbol]
        v0_by_date = dict(zip(pd.to_datetime(states[states.symbol == symbol].trade_date),
                              states[states.symbol == symbol].v0))
        v0 = np.array([v0_by_date[pd.Timestamp(d)] for d in g.trade_date])
        n = len(g)
        args = (g.spot.to_numpy(float), g.strike.to_numpy(float), g.maturity.to_numpy(float),
                g.rate.to_numpy(float), g.dividend.to_numpy(float),
                g.option_type.eq("CE").to_numpy())
        kw = (np.full(n, p.kappa), np.full(n, p.theta), np.full(n, p.sigma),
              np.full(n, p.rho), v0)
        a = pinn.price_rowwise(*args, *kw)
        b = fourier.price_rowwise(*args, *kw)
        iv_a = C.SH.implied_volatility(a, *args[:5], args[5])
        iv_b = C.SH.implied_volatility(b, *args[:5], args[5])
        market = g.market_iv.to_numpy(float)
        rows.append({
            "symbol": symbol, "test_holdout_rows": n,
            "kappa": p.kappa, "theta": p.theta, "sigma": p.sigma, "rho": p.rho,
            "feller_ratio": p.feller_ratio, "median_v0": float(np.median(v0)),
            "network_iv_bias_vs_exact": float(np.nanmean(iv_a - iv_b)),
            "network_iv_rmse_vs_exact": float(np.sqrt(np.nanmean((iv_a - iv_b) ** 2))),
            "network_price_rmse_vs_exact": float(np.sqrt(np.nanmean((a - b) ** 2))),
            "market_iv_rmse_network_priced": float(np.sqrt(np.nanmean((iv_a - market) ** 2))),
            "market_iv_rmse_exactly_priced": float(np.sqrt(np.nanmean((iv_b - market) ** 2))),
        })
    frame = pd.DataFrame(rows)
    w = frame.test_holdout_rows
    return frame, {
        "network_iv_bias_vs_exact": float((frame.network_iv_bias_vs_exact * w).sum() / w.sum()),
        "network_iv_rmse_vs_exact": float(np.sqrt((frame.network_iv_rmse_vs_exact ** 2 * w).sum() / w.sum())),
        "market_iv_rmse_network_priced": float(np.sqrt((frame.market_iv_rmse_network_priced ** 2 * w).sum() / w.sum())),
        "market_iv_rmse_exactly_priced": float(np.sqrt((frame.market_iv_rmse_exactly_priced ** 2 * w).sum() / w.sum())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--tag", type=str, default="physics_and_anchor")
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--depth", type=int, default=5)
    args = parser.parse_args()
    d = args.dir
    box = C.Box()

    spec = json.loads((d / "pinn_domain_spec.json").read_text())
    training_report = json.loads((d / ("pinn_training_report_%s.json" % args.tag)).read_text())
    npz = np.load(d / ("pinn_collocation_%s.npz" % args.tag), allow_pickle=False)
    collocation = {k: npz[k] for k in npz.files}
    model = R.load_model(d / ("pinn_model_%s.safetensors" % args.tag), args.width, args.depth)
    pinn = K.PinnEngine(model, box)
    fourier = K.FourierEngine(box)

    print("scoring the network against exact Heston ...", flush=True)
    score = T.evaluate(model, box, n=60000, seed=7777)
    pde = T.pde_residual_score(model, box, collocation)
    ceilings = {str(s): float(v) for s, v in
                zip(np.unique(collocation["symbol"]),
                    [collocation["spot_ceiling"][collocation["symbol"] == s][0]
                     for s in np.unique(collocation["symbol"])])}
    strike_ranges = {s: (float(g.strike.min()), float(g.strike.max()))
                     for s, g in pd.read_parquet(d / "pinn_quote_panel.parquet").groupby("symbol")}
    fresh = C.sample_collocation(collocation["x"].size, ceilings,
                                 {k: strike_ranges.get(k, (1.0, ceilings[k])) for k in ceilings},
                                 {k: 1.0 for k in ceilings}, box, seed=31337)
    pde_holdout = T.pde_residual_score(model, box, fresh)
    arb = arbitrage_audit(pinn, box)

    print("figures ...", flush=True)
    figs = d / "figures"; figs.mkdir(exist_ok=True)
    P.plot_training(training_report, figs / "training_convergence.png")
    P.plot_accuracy_vs_exact(model, box, figs / "pinn_versus_exact_heston.png")
    P.plot_collocation(collocation, figs / "collocation_domain.png")

    predictions, states, parameters, recovery = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None
    engines_present = {}
    for name in ("pinn", "fourier"):
        f = d / ("calibration_predictions_%s.parquet" % name)
        if f.exists():
            engines_present[name] = pd.read_parquet(f)
    if engines_present:
        P.plot_market_fit(engines_present, figs / "market_surface_fit.png")
    state_frames = [pd.read_csv(d / ("calibration_daily_state_%s.csv" % n))
                    for n in engines_present if (d / ("calibration_daily_state_%s.csv" % n)).exists()]
    if state_frames:
        states = pd.concat(state_frames, ignore_index=True)
        P.plot_state_paths(states, figs / "calibrated_spot_volatility.png")
    if (d / "calibration_parameters_pinn.csv").exists():
        parameters = pd.read_csv(d / "calibration_parameters_pinn.csv")
        predictions = engines_present.get("pinn", pd.DataFrame())
    if (d / "synthetic_recovery_pinn.csv").exists():
        recovery = pd.read_csv(d / "synthetic_recovery_pinn.csv")

    surface_rows = []
    panel = pd.read_parquet(d / "pinn_quote_panel.parquet")
    panel_spot = {s: g.spot.to_numpy(float) for s, g in panel.groupby("symbol")}
    if len(parameters):
        domain = pd.read_csv(d / "pinn_spot_domain.csv").set_index("symbol")
        for _, row in parameters.iterrows():
            symbol = row.symbol
            ceiling = float(domain.loc[symbol, "pinn_spot_high"])
            params = np.array([row.kappa, row.theta, row.sigma, row.rho, row.v0_training_median])
            strike = float(np.median(panel_spot.get(symbol, ceiling / 3.0)))
            surface_rows.append(P.plot_requested_domain(
                pinn, fourier, params, symbol, ceiling, strike,
                figs / ("surface_%s.png" % symbol)))
    surfaces = pd.concat(surface_rows, ignore_index=True) if surface_rows else pd.DataFrame()
    if len(surfaces):
        surfaces.to_csv(d / "reconstructed_surface_accuracy.csv", index=False)

    attribution, attribution_summary = (pd.DataFrame(), {})
    if (d / "calibration_parameters_pinn.csv").exists():
        attribution, attribution_summary = engine_attribution(pinn, fourier, d)
        attribution.to_csv(d / "engine_attribution.csv", index=False)

    observed_variance = panel.variance_from_inverse_bsm.to_numpy(float)
    quotes_above_box = int(np.nansum(observed_variance > box.variance_high))
    checks = build_checks(d, box, collocation, spec, training_report, score, pde,
                          predictions, parameters, recovery, arb, quotes_above_box)
    checks.to_csv(d / "pinn_strict_checks.csv", index=False)

    published = json.loads(PUBLISHED.read_text()) if PUBLISHED.exists() else {}
    comparison = []
    if published:
        comparison.append({"engine": "published classical (repository baseline)",
                           **published.get("test_holdout_metrics", {})})
    for name, frame in engines_present.items():
        test = frame[(frame.split == "test") & (frame.fold == "holdout")]
        comparison.append({"engine": name, **K.metrics(test)})
    comparison = pd.DataFrame(comparison)
    comparison.to_csv(d / "engine_comparison.csv", index=False)

    summary = {
        "tag": args.tag,
        "box": box.as_dict(),
        "collocation_points": int(collocation["x"].size),
        "spot_domain_rule": "S in [0, 1.5 x ten-year maximum traded price] per symbol",
        "maturity_rule": spec["maturity_domain"],
        "variance_rule": spec["variance_domain"],
        "network_versus_exact_heston": score,
        "pde_residual_on_training_collocation_set": pde,
        "pde_residual_on_independent_collocation_set": pde_holdout,
        "collocation_generalisation_ratio": (
            pde_holdout["residual_rms_price_relevant_core"]
            / max(pde["residual_rms_price_relevant_core"], 1e-12)),
        "no_arbitrage_audit": arb,
        "engine_comparison": comparison.to_dict("records"),
        "engine_attribution": attribution_summary,
        "quotes_with_variance_above_box": quotes_above_box,
        "quote_panel_rows": int(len(panel)),
        "strict_checks_passed": int(checks.passed.sum()),
        "strict_checks_total": int(len(checks)),
        "failed_checks": checks[~checks.passed].check.tolist(),
        "reference_pricer_audit": C.audit_reference_pricer(1500),
    }
    (d / ("pinn_study_summary_%s.json" % args.tag)).write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("box", "variance_rule")},
                     indent=2, default=str))
    print("\nstrict checks: %d/%d passed" % (checks.passed.sum(), len(checks)))
    print(checks.to_string(index=False))


if __name__ == "__main__":
    main()
