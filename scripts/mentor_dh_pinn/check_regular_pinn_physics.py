#!/usr/bin/env python3
"""Independent collocation/terminal/numerical checks of a frozen regular PINN.

This script makes no parameter-recovery claim and does not fit parameters or
use Fourier prices. It reports failures and both wide-domain and smile-region
residuals; weak results are not filtered out of the all-point diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.mentor_dh_pinn.assess_regular_pinn import load_checkpoint, sha256
from src.mentor_dh_pinn.regular_pinn import residual
from src.mentor_dh_pinn.regular_pinn_data import DOMAIN, coordinates, draw_points
from src.mentor_dh_pinn.regular_pinn_torch import TorchRegularVariancePINN


def array_hash(array):
    canonical = np.ascontiguousarray(array, dtype="<f8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def distribution(values):
    """All-point moments become null if any value failed; finite-only tails are explicit."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    good = values[finite]
    result = {"count": len(values), "nonfinite": int((~finite).sum()),
              "mean_all": float(np.mean(values)) if finite.all() and len(values) else None,
              "rmse_all": float(np.sqrt(np.mean(values**2))) if finite.all() and len(values) else None}
    result["absolute_quantiles_finite_subset"] = {
        str(q): float(np.quantile(np.abs(good), q)) if len(good) else None
        for q in (0.5, 0.9, 0.95, 0.99, 1.0)
    }
    return result


def evaluate_points(model, points, chunk=128):
    """Read-only diagnostics, sharing one float64 copy of the same network weights."""
    if chunk < 1 or len(points) < 1:
        raise ValueError("positive chunk and at least one point required")
    adapter = TorchRegularVariancePINN.from_mlx(model)
    blocks = []
    started = time.perf_counter()
    for start in range(0, len(points), chunk):
        q = points[start:start + chunk]
        states, structural = coordinates(mx.array(q, dtype=mx.float32), model.factors, mx)
        value, diag = residual(model, states, structural)
        iv = model.iv(states, structural)
        terminal = mx.concatenate([states[..., :-1], mx.zeros_like(states[..., -1:])], axis=-1)
        terminal_value = model.price(terminal, structural)
        mx.eval(value, diag, iv, terminal_value)
        with torch.no_grad():
            torch_states, torch_structural = coordinates(torch.tensor(q, dtype=torch.float64), model.factors, torch)
            reference_iv = adapter.iv(torch_states, torch_structural).numpy()
        x = np.asarray(states[..., 0], dtype=float)
        payoff = np.maximum(np.expm1(x), 0.0)
        block = {"residual": np.asarray(value, dtype=float), "iv_mlx": np.asarray(iv, dtype=float),
                 "iv_torch64": reference_iv, "terminal_payoff": payoff,
                 "terminal_error": np.asarray(terminal_value, dtype=float) - payoff,
                 **{name: np.asarray(value, dtype=float) for name, value in diag.items()}}
        blocks.append(block)
    arrays = {name: np.concatenate([block[name] for block in blocks]) for name in blocks[0]}
    finite_by_name = {name: int((~np.isfinite(value)).sum()) for name, value in arrays.items()}
    row_finite = np.logical_and.reduce([np.isfinite(value) for value in arrays.values()])
    wide = np.arange(len(points)) % 4 == 0
    r, convexity = arrays["residual"], arrays["convexity"]
    d2 = points[:, 0] / np.sqrt(arrays["w"]) - 0.5 * np.sqrt(arrays["w"])
    weight = np.maximum(np.exp(-d2**2 / 2), 0.01)
    parity_error = arrays["iv_mlx"] - arrays["iv_torch64"]
    terminal_scaled_error = arrays["terminal_error"] / (1.0 + arrays["terminal_payoff"])
    calendar = arrays["w"] * arrays["l_tau"]
    summary = {
        "points": len(points), "seconds": time.perf_counter() - started, "chunk": chunk,
        "all_diagnostics_finite": bool(row_finite.all()),
        "nonfinite_rows": int((~row_finite).sum()), "nonfinite_by_quantity": finite_by_name,
        "dimensionless_pde_residual": {
            "all_points": distribution(r), "smile_region": distribution(r[~wide]),
            "wide_domain": distribution(r[wide]),
            "vega_relevance_weighted_rmse_all": float(np.sqrt(np.sum(weight * r**2) / np.sum(weight)))
                if np.isfinite(r).all() and np.isfinite(weight).all() else None,
            "weight_definition": "max(exp(-d2^2/2), .01); unweighted full distribution is also retained",
        },
        "convexity": {
            "negative_count": int((convexity < 0).sum()),
            "negative_fraction_all_rows": float(np.mean(convexity < 0)),
            "material_negative_count": int((convexity < -1e-5).sum()), "material_threshold": -1e-5,
            "minimum_finite_subset": float(np.min(convexity[np.isfinite(convexity)])) if np.isfinite(convexity).any() else None,
            "nonfinite_count": int((~np.isfinite(convexity)).sum()),
        },
        "calendar_total_variance_derivative": {
            "negative_count": int((calendar < 0).sum()),
            "negative_fraction_all_rows": float(np.mean(calendar < 0)),
            "nonfinite_count": int((~np.isfinite(calendar)).sum()),
        },
        "terminal_condition": {
            "definition": "tau=0, normalized call payoff max(exp(x)-1, 0), imposed analytically",
            "errors": distribution(arrays["terminal_error"]),
            "max_error_divided_by_one_plus_payoff": float(np.max(np.abs(terminal_scaled_error)))
                if np.isfinite(terminal_scaled_error).all() else None,
            "scaled_error_tolerance": 5e-7,
            "passed_roundoff_tolerance": bool(np.isfinite(terminal_scaled_error).all()
                                               and np.max(np.abs(terminal_scaled_error)) <= 5e-7),
        },
        "same_weights_float32_float64_iv_parity": {
            "errors": distribution(parity_error), "atol": 2e-6, "rtol": 2e-6,
            "passed": bool(np.isfinite(parity_error).all() and np.allclose(arrays["iv_mlx"], arrays["iv_torch64"], atol=2e-6, rtol=2e-6)),
            "description": "same learned float32 weights promoted to float64; neither copy is retrained",
        },
    }
    order = np.argsort(np.where(np.isfinite(r), np.abs(r), np.inf))[::-1][:10]
    summary["largest_absolute_residual_points"] = [
        {"index": int(i), "q": points[i].tolist(), "wide_domain": bool(wide[i]),
         "residual": float(r[i]) if np.isfinite(r[i]) else None,
         "convexity": float(convexity[i]) if np.isfinite(convexity[i]) else None}
        for i in order
    ]
    return summary


def dataset_provenance(points, checkpoint_info, factors):
    config = checkpoint_info["config"]
    if not config.get("data"):
        return {"status": "dataset_path_not_recorded", "exact_overlap_fully_checked": False}
    directory = Path(config["data"])
    if not directory.is_absolute():
        directory = ROOT / directory
    point_rows = {row.tobytes() for row in np.ascontiguousarray(points, dtype="<f8")}
    records = []
    for name in ("train.npz", "validation.npz", "collocation.npz"):
        path = directory / name
        if not path.exists():
            records.append({"path": str(path), "status": "missing"})
            continue
        with np.load(path) as data:
            rows = np.ascontiguousarray(data["q"], dtype="<f8")
        overlap = sum(row.tobytes() in point_rows for row in rows)
        records.append({"path": str(path), "sha256": sha256(path), "points": len(rows),
                        "exact_overlap_rows": overlap, "status": "checked"})
    # Replay the deterministic resampling schedule of the current trainer, in
    # addition to checking the serialized initial pool and anchor/validation data.
    replay = []
    interval, steps = config.get("resample_every", 0), config.get("steps", 0)
    if interval:
        for step in range(interval, steps + 1, interval):
            seed = config["seed"] + 100000 + step
            q = draw_points(18000, factors, seed, collocation=True)
            rows = np.ascontiguousarray(q, dtype="<f8")
            replay.append({"step": step, "seed": seed, "array_sha256": array_hash(q),
                           "exact_overlap_rows": sum(row.tobytes() in point_rows for row in rows)})
    complete = all(row["status"] == "checked" for row in records)
    overlaps = sum(row.get("exact_overlap_rows", 0) for row in records + replay)
    return {"status": "checked" if complete else "incomplete", "datasets": records,
            "resampled_pool_replay": replay, "exact_overlap_rows": overlaps,
            "exact_overlap_fully_checked": complete,
            "scope": "Exact coordinate rows against saved datasets and replayed configured resampling seeds; not a statistical independence proof"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="New JSON artifact path")
    parser.add_argument("--seed", type=int, default=907611)
    parser.add_argument("--points", type=int, default=4096)
    parser.add_argument("--chunk", type=int, default=128)
    args = parser.parse_args()
    if args.points < 1 or args.chunk < 1:
        parser.error("positive --points and --chunk required")
    if args.out.exists():
        raise FileExistsError("Use a new output path to preserve prior evidence")
    torch.set_num_threads(1)
    checks = []
    for path in args.checkpoint:
        model, info = load_checkpoint(path)
        points = draw_points(args.points, model.factors, args.seed, collocation=True)
        result = evaluate_points(model, points, args.chunk)
        result.update(checkpoint=info, collocation_seed=args.seed,
                      collocation_array_sha256=array_hash(points),
                      dataset_provenance=dataset_provenance(points, info, model.factors))
        checks.append(result)
    artifact = {
        "status": "complete", "purpose": "Fresh collocation PDE and numerical diagnostics; not parameter recovery or a new training objective",
        "domain": DOMAIN,
        "sampling": "LHS parameter/state drivers; half continuous maturities and half monthly/rich slices; every fourth point uses x in [-3,3]",
        "limitations": "Small sampled residuals do not establish uniqueness, recovery, or global PDE correctness; no recovery-success gate is inferred",
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in (
            Path(__file__), ROOT / "src/mentor_dh_pinn/regular_pinn.py",
            ROOT / "src/mentor_dh_pinn/regular_pinn_torch.py", ROOT / "src/mentor_dh_pinn/regular_pinn_data.py",
            ROOT / "scripts/mentor_dh_pinn/assess_regular_pinn.py", ROOT / "scripts/mentor_dh_pinn/train_regular_pinn.py")},
        "checks": checks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, allow_nan=False))
    print(json.dumps({"out": str(args.out), "checks": len(checks),
                      "all_finite": all(row["all_diagnostics_finite"] for row in checks)}, indent=2))


if __name__ == "__main__":
    main()
