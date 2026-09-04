#!/usr/bin/env python3
"""Run the frozen calibration protocol with the PINN and with exact Heston."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

import pinn_heston_core as C
import pinn_calibrate as K
import pinn_run_training as R

ROOT = Path(__file__).resolve().parent
DEFAULT_DIR = ROOT / "outputs" / "pinn_single_heston"
MAX_TRAINING_DATES = 12


def choose_training_dates(dates, maximum=12):
    """Identical to single_heston.choose_training_dates."""
    if len(dates) <= maximum:
        return list(dates)
    positions = np.linspace(0, len(dates) - 1, maximum).round().astype(int)
    return [dates[position] for position in np.unique(positions)]


def run_protocol(engine, transform, panel, max_training_dates=MAX_TRAINING_DATES, verbose=True):
    parameters, states, predictions, training = [], [], [], []
    fit_seconds_total = 0.0
    for symbol, symbol_panel in panel.groupby("symbol", sort=True):
        by_date = {d: g.reset_index(drop=True) for d, g in symbol_panel.groupby("trade_date")}
        train_dates = sorted(d for d, g in by_date.items() if g.split.iloc[0] == "train")
        if len(train_dates) < 3:
            continue
        candidates = choose_training_dates(train_dates, max_training_dates * 5)
        candidates += [d for d in train_dates if d not in set(candidates)]
        surfaces, surface_dates = [], []
        for day in candidates:
            calib = by_date[day][by_date[day].fold.eq("calibration")].reset_index(drop=True)
            if len(calib) < 6:
                continue
            surfaces.append(calib)
            surface_dates.append(day)
            if len(surfaces) >= max_training_dates:
                break
        if len(surfaces) < 3:
            continue
        t0 = time.perf_counter()
        fit = K.fit_joint_structural(engine, transform, surfaces)
        joint_seconds = time.perf_counter() - t0
        fit_seconds_total += joint_seconds
        structural = fit["structural"]
        for day, calib, v0 in zip(surface_dates, surfaces, fit["variances"]):
            hold = by_date[day][by_date[day].fold.eq("holdout")].reset_index(drop=True)
            params = np.r_[structural, v0]
            training.append({
                "symbol": symbol, "trade_date": day, "engine": engine.name,
                "kappa": params[0], "theta": params[1], "sigma": params[2],
                "rho": params[3], "v0": params[4],
                "calibration_iv_rmse": K.metrics(K.predict(engine, calib, params)).get("iv_rmse"),
                "holdout_iv_rmse": K.metrics(K.predict(engine, hold, params)).get("iv_rmse") if len(hold) else np.nan,
            })
        all_dates = sorted(by_date)
        test_dates = [d for d in all_dates if by_date[d].split.iloc[0] == "test"]
        parameters.append({
            "symbol": symbol, "engine": engine.name,
            "kappa": structural[0], "theta": structural[1],
            "sigma": structural[2], "rho": structural[3],
            "feller_ratio": structural[2] / math.sqrt(2 * structural[0] * structural[1]),
            "feller_gap": 2 * structural[0] * structural[1] - structural[2] ** 2,
            "v0_training_median": float(np.median(fit["variances"])),
            "objective": fit["objective"], "optimizer_evaluations": fit["evaluations"],
            "joint_fit_seconds": joint_seconds,
            "training_surface_fits": len(surfaces),
            "training_last_date": max(train_dates),
            "test_first_date": test_dates[0] if test_dates else pd.NaT,
        })
        for day in all_dates:
            quotes = by_date[day]
            split = quotes.split.iloc[0]
            if split == "train":
                continue
            calib = quotes[quotes.fold.eq("calibration")].reset_index(drop=True)
            if len(calib) < 6:
                continue
            t0 = time.perf_counter()
            v0, nfev = K.fit_v0(engine, transform, calib, structural)
            state_seconds = time.perf_counter() - t0
            fit_seconds_total += state_seconds
            params = np.r_[structural, v0]
            frame = K.predict(engine, quotes, params)
            frame["engine"] = engine.name
            predictions.append(frame)
            states.append({
                "symbol": symbol, "trade_date": day, "split": split, "engine": engine.name,
                "v0": v0, "spot_vol": math.sqrt(v0), "state_fit_seconds": state_seconds,
                "optimizer_evaluations": nfev,
                "calibration_rows": int(quotes.fold.eq("calibration").sum()),
                "holdout_rows": int(quotes.fold.eq("holdout").sum()),
                "calibration_iv_rmse": K.metrics(frame[frame.fold.eq("calibration")]).get("iv_rmse"),
                "holdout_iv_rmse": K.metrics(frame[frame.fold.eq("holdout")]).get("iv_rmse"),
            })
        if verbose:
            print("  %-11s %-8s kappa=%7.3f theta=%.4f sigma=%.3f rho=%+.3f  joint %.1fs" % (
                symbol, engine.name, structural[0], structural[1], structural[2],
                structural[3], joint_seconds), flush=True)
    return (pd.DataFrame(parameters), pd.DataFrame(states),
            pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(),
            pd.DataFrame(training), fit_seconds_total)


def synthetic_recovery(engine, transform, box, n_cases=8, seed=31):
    """Can the engine recover parameters it generated the surface from?"""
    rng = np.random.default_rng(seed)
    truth = C.draw_parameters(rng, n_cases, box)
    v0_true = np.exp(rng.uniform(math.log(0.02), math.log(0.35), n_cases))
    rows = []
    for i in range(n_cases):
        params = np.array([truth["kappa"][i], truth["theta"][i], truth["sigma"][i],
                           truth["rho"][i], v0_true[i]])
        spot = 500.0
        frames = []
        for days in (14, 30, 60):
            maturity = days / 365.0
            strikes = spot * np.exp(np.linspace(-0.30, 0.30, 15))
            rate, dividend = 0.07, 0.02
            x = np.log(spot / strikes) + (rate - dividend) * maturity
            c = C.normalised_forward_call_batch(x, np.full(15, maturity), *params[:4], params[4])
            call = strikes * math.exp(-rate * maturity) * c
            forward = spot * math.exp((rate - dividend) * maturity)
            is_call = strikes >= forward
            price = np.where(is_call, call,
                             call - spot * math.exp(-dividend * maturity)
                             + strikes * math.exp(-rate * maturity))
            iv = SH_iv(price, spot, strikes, maturity, rate, dividend, is_call)
            vega = C.SH.black_scholes_vega(spot, strikes, maturity, rate, dividend, iv)
            frames.append(pd.DataFrame({
                "spot": spot, "strike": strikes, "maturity": maturity,
                "rate": rate, "dividend": dividend,
                "option_type": np.where(is_call, "CE", "PE"),
                "market_price_adjusted": price, "market_iv": iv, "vega": vega,
                "weight": 1.0, "price_adjustment_factor": 1.0,
                "market_price_raw": price}))
        surface = pd.concat(frames, ignore_index=True)
        surface = surface[np.isfinite(surface.market_iv) & (surface.vega > 0.002 * spot)].reset_index(drop=True)
        t0 = time.perf_counter()
        fit = K.fit_joint_structural(engine, transform, [surface])
        seconds = time.perf_counter() - t0
        recovered = np.r_[fit["structural"], fit["variances"][0]]
        pred = K.predict(engine, surface, recovered)
        rows.append({
            "case": i, "engine": engine.name, "seconds": seconds,
            **{f"true_{n}": float(v) for n, v in zip(("kappa", "theta", "sigma", "rho", "v0"), params)},
            **{f"fit_{n}": float(v) for n, v in zip(("kappa", "theta", "sigma", "rho", "v0"), recovered)},
            "price_rmse_pct_of_spot": float(100 * np.sqrt(np.mean(
                (pred.model_price_adjusted - pred.market_price_adjusted) ** 2)) / spot),
            "iv_rmse": K.metrics(pred).get("iv_rmse"),
        })
    return pd.DataFrame(rows)


def SH_iv(price, spot, strike, maturity, rate, dividend, is_call):
    return C.SH.implied_volatility(price, spot, strike, maturity, rate, dividend, is_call)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--tag", type=str, default="physics_and_anchor")
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--engines", type=str, default="pinn,fourier")
    args = parser.parse_args()

    box = C.Box()
    transform = C.ParameterTransform(box)
    panel = pd.read_parquet(args.dir / "pinn_quote_panel.parquet")
    engines = []
    if "pinn" in args.engines:
        model = R.load_model(args.dir / ("pinn_model_%s.safetensors" % args.tag), args.width, args.depth)
        engines.append(K.PinnEngine(model, box))
    if "fourier" in args.engines:
        engines.append(K.FourierEngine(box))

    summary = {"tag": args.tag, "box": box.as_dict(), "engines": {}}
    for engine in engines:
        print("\n=== calibration protocol: %s ===" % engine.name, flush=True)
        t0 = time.perf_counter()
        params, states, preds, training, fit_seconds = run_protocol(engine, transform, panel)
        wall = time.perf_counter() - t0
        params.to_csv(args.dir / ("calibration_parameters_%s.csv" % engine.name), index=False)
        states.to_csv(args.dir / ("calibration_daily_state_%s.csv" % engine.name), index=False)
        training.to_csv(args.dir / ("calibration_training_fits_%s.csv" % engine.name), index=False)
        preds.to_parquet(args.dir / ("calibration_predictions_%s.parquet" % engine.name), index=False)
        test_hold = preds[(preds.split == "test") & (preds.fold == "holdout")]
        val_hold = preds[(preds.split == "validation") & (preds.fold == "holdout")]
        summary["engines"][engine.name] = {
            "test_holdout": K.metrics(test_hold),
            "validation_holdout": K.metrics(val_hold),
            "test_calibration_fold": K.metrics(preds[(preds.split == "test") & (preds.fold == "calibration")]),
            "wall_seconds": wall,
            "optimiser_seconds": fit_seconds,
            "joint_fit_seconds_median": float(params.joint_fit_seconds.median()),
            "state_fit_seconds_median": float(states.state_fit_seconds.median()),
            "state_fits": int(len(states)),
        }
        print(json.dumps(summary["engines"][engine.name], indent=2), flush=True)
        rec = synthetic_recovery(engine, transform, box)
        rec.to_csv(args.dir / ("synthetic_recovery_%s.csv" % engine.name), index=False)
        summary["engines"][engine.name]["synthetic_recovery"] = {
            "cases": int(len(rec)),
            "median_price_rmse_pct_of_spot": float(rec.price_rmse_pct_of_spot.median()),
            "max_price_rmse_pct_of_spot": float(rec.price_rmse_pct_of_spot.max()),
            "median_iv_rmse": float(rec.iv_rmse.median()),
            "median_seconds": float(rec.seconds.median()),
        }
        print("synthetic recovery:", json.dumps(
            summary["engines"][engine.name]["synthetic_recovery"], indent=2), flush=True)

    (args.dir / ("calibration_summary_%s.json" % args.tag)).write_text(json.dumps(summary, indent=2, default=str))
    print("\nwritten to", args.dir)


if __name__ == "__main__":
    main()
