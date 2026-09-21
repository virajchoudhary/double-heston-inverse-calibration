#!/usr/bin/env python3
"""Train the single-Heston PINN until it reconstructs option prices accurately.

The outer loop is the "repeat until it works" requirement made explicit: each
round trains, scores the network against the exact Heston model and against the
PDE itself, and stops only when every acceptance gate passes.  A failed round
warm-starts the next one with more steps, a lower floor on the learning rate and
more weight on whichever term failed.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

import pinn_heston_core as C
import pinn_train as T

ROOT = Path(__file__).resolve().parent
DEFAULT_DIR = ROOT / "outputs" / "pinn_single_heston"

GATES = {
    "traded_iv_rmse": 0.0025,          # 25 bps of implied volatility
    "traded_iv_p99": 0.0120,
    "traded_price_rmse_per_strike": 3.0e-4,
    "full_box_iv_rmse": 0.0120,
    "pde_core_residual_rms": 0.060,
    "arbitrage_violation_fraction": 0.0,
}


def check_gates(score, pde):
    checks = {
        "traded_iv_rmse": (score["traded_region"]["iv_rmse"], GATES["traded_iv_rmse"]),
        "traded_iv_p99": (score["traded_region"]["iv_p99"], GATES["traded_iv_p99"]),
        "traded_price_rmse_per_strike": (score["traded_region"]["price_rmse_per_strike"],
                                         GATES["traded_price_rmse_per_strike"]),
        "full_box_iv_rmse": (score["full_box"]["iv_rmse"], GATES["full_box_iv_rmse"]),
        "pde_core_residual_rms": (pde["residual_rms_price_relevant_core"],
                                  GATES["pde_core_residual_rms"]),
        "calendar_violation_fraction": (pde["calendar_violation_fraction"],
                                        GATES["arbitrage_violation_fraction"]),
        "butterfly_violation_fraction": (pde["butterfly_violation_fraction"],
                                         GATES["arbitrage_violation_fraction"]),
    }
    return {k: {"value": float(v), "limit": float(lim), "passed": bool(v <= lim)}
            for k, (v, lim) in checks.items()}


def load_domain(directory: Path):
    spec = json.loads((directory / "pinn_domain_spec.json").read_text())
    import pandas as pd
    domain = pd.read_csv(directory / "pinn_spot_domain.csv")
    panel = pd.read_parquet(directory / "pinn_quote_panel.parquet")
    active = sorted(panel.symbol.unique())
    domain = domain[domain.symbol.isin(active)]
    ceiling = dict(zip(domain.symbol, domain.pinn_spot_high))
    strikes = {}
    for symbol, group in panel.groupby("symbol"):
        strikes[symbol] = (float(group.strike.min()), float(group.strike.max()))
    weight = panel.symbol.value_counts().to_dict()
    return spec, ceiling, strikes, weight, panel


def save_model(model, path: Path):
    mx.save_safetensors(str(path), dict(tree_flatten(model.parameters())))


def load_model(path: Path, width, depth):
    model = C.ImpliedVarianceNet(width=width, depth=depth)
    model.update(tree_unflatten(list(mx.load(str(path)).items())))
    mx.eval(model.parameters())
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--collocation-points", type=int, default=18000)
    parser.add_argument("--anchor-points", type=int, default=18000)
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--steps", type=int, default=16000)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--tag", type=str, default="physics_and_anchor")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not (args.collocation_points <= 20000 and args.collocation_points >= 14000):
        raise SystemExit("collocation budget must stay inside the 14,000-20,000 range")

    box = C.Box()
    spec, ceiling, strikes, weight, panel = load_domain(args.dir)
    print("symbols: %s" % ", ".join(sorted(ceiling)))
    print("spot ceilings (1.5 x ten-year max): %s" %
          {k: round(v, 1) for k, v in sorted(ceiling.items())})

    collocation = C.sample_collocation(args.collocation_points, ceiling, strikes, weight,
                                       box, seed=101 + args.seed)
    anchor = C.sample_anchor(args.anchor_points, box, seed=202 + args.seed)
    anchor.update(C.anchor_targets(anchor, g_limit=C.G_LIMIT))
    print("collocation %d points, spot in [%.2f, %.1f]; anchor %d points, %d usable" % (
        collocation["x"].size, collocation["spot"].min(), collocation["spot"].max(),
        anchor["x"].size, int(anchor["usable"].sum())))

    cfg = T.TrainConfig(collocation_points=args.collocation_points,
                        anchor_points=args.anchor_points, width=args.width, depth=args.depth,
                        steps=args.steps, weight_anchor=args.anchor_weight,
                        seed=args.seed, label=args.tag)
    model, rounds, history = None, [], []
    started = time.time()
    for r in range(1, args.max_rounds + 1):
        print("\n=== round %d: %d steps, lr %.1e -> %.1e, anchor weight %.2f ===" % (
            r, cfg.steps, cfg.learning_rate, cfg.final_learning_rate, cfg.weight_anchor), flush=True)
        model, hist = T.train(cfg, collocation, anchor, box, model=model)
        history.extend([{**h, "round": r} for h in hist])
        score = T.evaluate(model, box, n=40000, seed=999)
        pde = T.pde_residual_score(model, box, collocation)
        gates = check_gates(score, pde)
        passed = all(g["passed"] for g in gates.values())
        rounds.append({"round": r, "config": asdict(cfg), "score": score,
                       "pde": pde, "gates": gates, "all_passed": passed})
        print(json.dumps({"round": r, "traded": score["traded_region"], "pde": pde,
                          "failed_gates": [k for k, g in gates.items() if not g["passed"]]},
                         indent=2), flush=True)
        save_model(model, args.dir / ("pinn_model_%s.safetensors" % args.tag))
        (args.dir / ("pinn_training_report_%s.json" % args.tag)).write_text(json.dumps(
            {"box": box.as_dict(), "gates": GATES, "rounds": rounds,
             "history": history, "elapsed_seconds": time.time() - started,
             "collocation_points": int(collocation["x"].size),
             "anchor_points_usable": int(anchor["usable"].sum())}, indent=2))
        if passed:
            print("\nall acceptance gates passed at round %d" % r)
            break
        failed = {k for k, g in gates.items() if not g["passed"]}
        cfg = replace(
            cfg,
            steps=int(cfg.steps * 1.5),
            learning_rate=cfg.learning_rate * 0.45,
            final_learning_rate=cfg.final_learning_rate * 0.35,
            warmup=200,
            weight_anchor=cfg.weight_anchor * (2.0 if {"traded_iv_rmse", "traded_iv_p99",
                                                       "full_box_iv_rmse"} & failed else 1.0),
            weight_pde=cfg.weight_pde * (2.0 if "pde_core_residual_rms" in failed else 1.0),
            weight_calendar=cfg.weight_calendar * (4.0 if "calendar_violation_fraction" in failed else 1.0),
            weight_butterfly=cfg.weight_butterfly * (4.0 if "butterfly_violation_fraction" in failed else 1.0),
        )
    else:
        print("\ngates still failing after %d rounds; best model saved" % args.max_rounds)

    np.savez_compressed(args.dir / ("pinn_collocation_%s.npz" % args.tag),
                        **{k: v for k, v in collocation.items() if k != "symbol"},
                        symbol=collocation["symbol"].astype("U16"))
    print("elapsed %.1f min" % ((time.time() - started) / 60))


if __name__ == "__main__":
    main()
