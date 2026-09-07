#!/usr/bin/env python3
"""All-quote development validation of frozen Double Heston checkpoints.

This corpus was used for checkpoint selection and includes inherited checkpoint
exposures. Its metrics are development evidence, not a fresh test or market forecast.
Fit quotes alone enter the encoder/refinement. Every holdout is retained, including
failed predictions. Final prices use 128-node Fourier quadrature; IV errors come from
Black-Scholes inversion, separately from the first-order price/vega approximation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.mentor_dh_pinn.run_finetune import load_real, validation_batches
from src.mentor_dh_pinn.nifty_panel import SIG_HI, SIG_LO, implied_vol
from src.mentor_dh_pinn.params_v2 import CANONICAL
from src.mentor_dh_pinn.torch_pricer import price_call
from src.mentor_dh_pinn.unified import UnifiedCalibrator

PRICING_NODES = 128


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def score_quote(prediction, observed, forward, strike, tau, vega):
    """Keep invalid prices unchanged. An invalid IV remains a failed observation."""
    lower, upper = max(forward - strike, 0.), forward
    finite_price = math.isfinite(prediction)
    noarb_bad = finite_price and (prediction < lower or prediction > upper)
    pred_iv = implied_vol(prediction, forward, strike, tau) if finite_price else math.nan
    obs_iv = implied_vol(observed, forward, strike, tau)
    iv_valid = math.isfinite(pred_iv) and math.isfinite(obs_iv)
    price_error = prediction - observed
    first_order = price_error / vega if math.isfinite(vega) and vega > 0 else math.nan
    status = ("nonfinite_price" if not finite_price else
              "no_arbitrage_violation" if noarb_bad else
              "source_iv_invalid" if not math.isfinite(obs_iv) else
              "model_iv_unavailable" if not math.isfinite(pred_iv) else "ok")
    return {"observed_price": observed, "model_price_128_nodes": prediction,
            "no_arbitrage_lower": lower, "no_arbitrage_upper": upper,
            "price_error": price_error, "market_iv_inverted": obs_iv,
            "model_iv_inverted": pred_iv,
            "iv_error": pred_iv - obs_iv if iv_valid else math.nan,
            "first_order_vega_error": first_order,
            "nonfinite_price": not finite_price, "no_arbitrage_violation": noarb_bad,
            "iv_valid": iv_valid, "status": status}


def summarize(rows):
    """An all-quote metric is unavailable if any required prediction failed."""
    def rmse(key):
        values = [r[key] for r in rows]
        return (math.sqrt(math.fsum(v * v for v in values) / len(values))
                if values and all(math.isfinite(v) for v in values) else None)
    return {"quotes": len(rows), "surfaces": len({r["label"] for r in rows}),
            "iv_rmse_all_quotes": rmse("iv_error"),
            "price_rmse_all_quotes_normalized": rmse("price_error"),
            "first_order_vega_rmse_all_quotes": rmse("first_order_vega_error"),
            "invalid_iv_quotes": sum(not r["iv_valid"] for r in rows),
            "nonfinite_price_quotes": sum(r["nonfinite_price"] for r in rows),
            "no_arbitrage_violations": sum(r["no_arbitrage_violation"] for r in rows),
            "inference_failure_quotes": sum(bool(r.get("inference_error")) for r in rows),
            "failed_quote_policy": "Keep all rows; an incomplete all-quote metric is null."}


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty evaluation table")
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_checkpoint(checkpoint, data, *, refine=3, batch_size=16):
    """Return all quote/surface records; never route target prices through the encoder."""
    sha_before = digest(checkpoint)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = ck["config"]
    seed = ck.get("seed", config.get("seed", "not_recorded"))
    name = f"{checkpoint.parent.name}/{checkpoint.stem} (seed {seed})"
    model = UnifiedCalibrator(d_model=config["d_model"], rounds=config["rounds"],
                              node_count=config["nodes"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    quotes, surfaces = [], []
    started = time.perf_counter()
    for start in range(0, len(data["label"]), batch_size):
        ids = np.arange(start, min(start + batch_size, len(data["label"])))
        fit, target = validation_batches(data, ids)
        before = time.perf_counter()
        error = ""
        try:
            with torch.no_grad():
                output = model(fit, refine_steps=refine)
                params = output["params"]
                # All holdout geometry is used only AFTER parameter calibration finishes.
                prediction = price_call(params, target["spot"], target["strike"],
                                        target["tau"], target["rate"], target["carry"],
                                        node_count=PRICING_NODES).numpy()
                p = params.numpy()
                objective = model.refinement_objective(
                    output["z"], output["mu_z"], output["L"], fit).numpy()
        except (RuntimeError, ValueError, FloatingPointError, OverflowError) as exc:
            # A batch failure is visible on every affected row, not silently excluded.
            error = f"{type(exc).__name__}: {exc}"
            prediction = np.full(tuple(target["price"].shape), np.nan)
            p = np.full((len(ids), 10), np.nan)
            objective = np.full(len(ids), np.nan)
        batch_seconds = time.perf_counter() - before
        for j, row in enumerate(ids):
            label = str(data["label"][row])
            symbol, trade_date = label.split("|", 1)
            common = {"model": name, "checkpoint_seed": seed, "surface_index": int(row),
                      "label": label, "symbol": symbol, "trade_date": trade_date}
            own = []
            for q in torch.where(target["mask"][j] > .5)[0].tolist():
                f, k, t, obs, vega = (float(target[key][j, q])
                                      for key in ("spot", "strike", "tau", "price", "vega"))
                rec = {**common, "quote_index": q, "forward_normalized": f,
                       "strike_normalized": k, "tau_years": t,
                       **score_quote(float(prediction[j, q]), obs, f, k, t, vega),
                       "inference_error": error}
                quotes.append(rec)
                own.append(rec)
            metrics = summarize(own)
            surfaces.append({**common, "fit_quotes": int(fit["n_quotes"][j]),
                             "holdout_quotes": len(own),
                             "fit_expiries": int(torch.unique(fit["tau"][j][fit["mask"][j] > .5]).numel()),
                             **{k: metrics[k] for k in ("iv_rmse_all_quotes", "price_rmse_all_quotes_normalized",
                                                       "first_order_vega_rmse_all_quotes", "invalid_iv_quotes",
                                                       "nonfinite_price_quotes", "no_arbitrage_violations")},
                             **{param: float(p[j, i]) for i, param in enumerate(CANONICAL)},
                             "fit_penalized_objective": float(objective[j]),
                             "batch_amortized_calibration_and_pricing_seconds": batch_seconds / len(ids),
                             "inference_error": error})
        print(f"{name}: {min(start+batch_size, len(data['label']))}/{len(data['label'])} surfaces", flush=True)
    if digest(checkpoint) != sha_before:
        raise RuntimeError(f"Checkpoint changed during evaluation; freeze it first: {checkpoint}")
    by_symbol = defaultdict(list)
    for row in quotes:
        by_symbol[row["symbol"]].append(row)
    summary = {"name": name, "checkpoint": str(checkpoint), "checkpoint_sha256": sha_before,
               "seed": seed, "checkpoint_step": ck.get("step"), "refinement_steps": refine,
               "refinement_nodes": config["nodes"], "evaluation_nodes": PRICING_NODES,
               "seconds": time.perf_counter() - started, **summarize(quotes),
               "by_symbol": {k: summarize(v) for k, v in sorted(by_symbol.items())}}
    return quotes, surfaces, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--real", type=Path, default=ROOT/"outputs/calibration_repair/real_corpus/real_validation.npz")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--refine", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.real.name != "real_validation.npz":
        raise ValueError("This evaluator accepts development real_validation.npz only")
    if args.refine < 0 or args.batch_size < 1 or args.threads < 1:
        raise ValueError("refine must be nonnegative; batch-size and threads must be positive")
    checkpoints = [p.resolve(strict=True) for p in args.checkpoint]
    if len(set(checkpoints)) != len(checkpoints):
        raise ValueError("Duplicate checkpoint paths")
    if args.out.exists():
        raise FileExistsError("Use a new output directory; prior evaluation evidence is preserved")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    data = load_real(args.real)
    active = data["mask"] > .5
    if any(np.any(data[key][active] != 0.) for key in ("rate", "carry")):
        raise ValueError("This corpus must be in the undiscounted forward measure (r=q=0)")
    if len(set(map(str, data["label"]))) != len(data["label"]):
        raise ValueError("Duplicate surface labels")
    expected_quotes = int((active & (data["holdout_mask"] > .5)).sum())
    sources = [Path(__file__), ROOT/"scripts/mentor_dh_pinn/run_finetune.py",
               *(ROOT/f"src/mentor_dh_pinn/{p}.py" for p in
                 ("unified", "params_v2", "torch_pricer", "nifty_panel", "collate"))]
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    corpus_hash = digest(args.real)
    summary = {"evidence": "Development validation used for checkpoint selection; not fresh test evidence.",
               "inherited_exposure": "Earlier checkpoints may already have seen these dates; rebuilding does not erase exposure.",
               "pricing": "Double Heston Fourier inversion with 128-node Gauss-Laguerre quadrature; finite numerical precision.",
               "iv_error": "Difference between BSM-inverted predicted and observed call prices; no price clipping.",
               "iv_bracket": [SIG_LO, SIG_HI],
               "first_order_vega_error": "Price difference / corpus calibration-derived vega; only a first-order IV approximation.",
               "price_units": "Undiscounted forward-call prices in the corpus normalization, not raw rupees.",
               "parameter_interpretation": "Calibration estimates, not known true market parameters; uniqueness is not established.",
               "covariance_scope": "Encoder L concerns mu_z before refinement; no refined posterior claim.",
               "corpus": str(args.real.resolve()), "corpus_sha256": corpus_hash,
               "surfaces_per_model": len(data["label"]), "holdout_quotes_per_model": expected_quotes,
               "source_sha256": source_hashes,
               "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                           "numpy": np.__version__, "scipy": scipy.__version__,
                           "platform": platform.platform(), "threads": args.threads},
               "models": []}
    manifest = args.real.parent/"manifest.json"
    if manifest.exists():
        summary["corpus_manifest_sha256"] = digest(manifest)
        summary["corpus_notes"] = json.loads(manifest.read_text()).get("notes", [])
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/"evaluation_protocol.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    all_quotes, all_surfaces = [], []
    for checkpoint in checkpoints:
        quotes, surfaces, model = evaluate_checkpoint(checkpoint, data, refine=args.refine,
                                                      batch_size=args.batch_size)
        if len(quotes) != expected_quotes or len(surfaces) != len(data["label"]):
            raise AssertionError("Evaluation dropped observations")
        all_quotes.extend(quotes)
        all_surfaces.extend(surfaces)
        summary["models"].append(model)
    if digest(args.real) != corpus_hash or any(digest(ROOT/p) != sha for p, sha in source_hashes.items()):
        raise RuntimeError("Corpus or implementation changed during evaluation; evidence is incomplete")
    write_csv(args.out/"holdout_predictions.csv", all_quotes)
    write_csv(args.out/"surface_parameters_and_metrics.csv", all_surfaces)
    summary["output_sha256"] = {p.name: digest(p) for p in
                               (args.out/"holdout_predictions.csv", args.out/"surface_parameters_and_metrics.csv")}
    summary["status"] = "complete"
    (args.out/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps({m["name"]: {k: m[k] for k in ("surfaces", "quotes", "iv_rmse_all_quotes",
                                                  "invalid_iv_quotes", "no_arbitrage_violations")}
                      for m in summary["models"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
