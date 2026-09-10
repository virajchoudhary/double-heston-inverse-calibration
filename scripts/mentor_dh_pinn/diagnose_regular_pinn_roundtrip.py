#!/usr/bin/env python3
"""Artificial neural self-consistency diagnostic; not exact-reference recovery.

Generate IVs from a frozen regular PINN, then recover its generating parameters
through the same neural function using blind starts and heldout strikes.
No Fourier price is generated, fitted, or scored by this diagnostic.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import scripts.mentor_dh_pinn.assess_regular_pinn as assessment
import src.mentor_dh_pinn.regular_pinn_data as data
import src.mentor_dh_pinn.torch_pricer as pricer
from scripts.mentor_dh_pinn.assess_regular_pinn import _network_iv, fit_network, load_checkpoint, sha256
from src.mentor_dh_pinn.regular_pinn_data import decode_unit
from src.mentor_dh_pinn.regular_pinn_torch import TorchRegularVariancePINN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/regular_pinn_recovery/double_sobolev_s17")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/regular_pinn_recovery/neural_roundtrip908171")
    parser.add_argument("--seed", type=int, default=908171)
    parser.add_argument("--cases", type=int, default=12)
    parser.add_argument("--starts", type=int, default=5)
    parser.add_argument("--max-nfev", type=int, default=400)
    args = parser.parse_args()
    if min(args.cases, args.starts, args.max_nfev) < 1:
        parser.error("Cases and optimization budgets must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    model, checkpoint_info = load_checkpoint(args.checkpoint)
    if model.factors != 2:
        raise ValueError("This diagnostic is specified for Double Heston only")
    network = model if isinstance(model, TorchRegularVariancePINN) else TorchRegularVariancePINN.from_mlx(model)
    sources = [Path(__file__), ROOT / "scripts/mentor_dh_pinn/assess_regular_pinn.py",
               ROOT / "src/mentor_dh_pinn/regular_pinn.py", ROOT / "src/mentor_dh_pinn/regular_pinn_torch.py",
               ROOT / "src/mentor_dh_pinn/regular_pinn_data.py"]
    manifest = {
        "status": "running", "evidence_level": "ARTIFICIAL NEURAL SELF-CONSISTENCY DEVELOPMENT DIAGNOSTIC",
        "interpretation": "Tests inverse optimization of a network against its own outputs; not exact-reference Heston recovery, market recovery, or research success",
        "checkpoint": checkpoint_info,
        "seed": args.seed, "cases": args.cases, "truth_sampling": "New independent uniform unit coordinates in [.1,.9]",
        "starts": args.starts, "max_nfev_per_start": args.max_nfev,
        "initialization": "Blind LHS starts generated with seed+100+case index, independent of generating parameters",
        "target": "Frozen network's own float64 IV values; no noise and no Fourier reference",
        "geometry": {"spot": 1.0, "strikes": np.linspace(0.8, 1.2, 21).tolist(),
                     "expiry_days": [30, 60, 90, 180, 365, 730], "tau_units": "years, day/365",
                     "holdout": "zero-based strike index % 3 == 2 at every expiry"},
        "parameter_gate": "All eight positive parameters within5% of their generating values, both correlations within.05 absolute",
        "data_access": "Generating parameters used only for target generation and post-fit scoring; calibration sees x,tau,observedIV at calibration strikes",
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    rng = np.random.default_rng(args.seed)
    truths = rng.uniform(0.1, 0.9, (args.cases, 10))
    x = np.tile(-np.log(np.linspace(0.8, 1.2, 21)), 6)
    tau = np.repeat(np.asarray([30, 60, 90, 180, 365, 730]) / 365.0, 21)
    holdout = np.tile(np.arange(21) % 3 == 2, 6)
    cases = []
    started = time.perf_counter()

    def forbidden(*args, **kwargs):
        raise AssertionError("Fourier evaluation is forbidden in this neural round-trip diagnostic")

    # Runtime guard supplements the existing no-exact-pricer calibration tests.
    with ExitStack() as guards:
        for module, name in ((assessment, "_exact"), (assessment, "exact_prices"),
                             (data, "exact_prices"), (data, "teacher_labels"),
                             (pricer, "price_call"), (pricer, "price_call_single")):
            guards.enter_context(patch.object(module, name, forbidden))
        for index, truth in enumerate(truths):
            true_physical = decode_unit(truth, 2)
            targets = _network_iv(network, x, tau, truth)
            record = {"case": index, "true_unit": truth.tolist(), "true_physical": true_physical.tolist(),
                      "target_iv": targets.tolist(), "all_parameter_pass": False}
            if not np.isfinite(targets).all() or (targets <= 0).any():
                record.update(status="invalid_neural_targets")
            else:
                fit = fit_network(network, x, tau, targets, fit_mask=~holdout,
                                  starts=args.starts, seed=args.seed + 100 + index, max_nfev=args.max_nfev)
                record.update(status=fit["status"], fit=fit)
                if fit["status"] == "fitted":
                    estimates = np.asarray(fit["physical"])
                    predictions = _network_iv(network, x, tau, np.asarray(fit["unit"]))
                    parameter_rows = []
                    for j, (actual, estimated) in enumerate(zip(true_physical, estimates)):
                        correlation = j % 5 == 3
                        error = float(abs(estimated - actual))
                        tolerance = 0.05 if correlation else 0.05 * actual
                        parameter_rows.append({"factor": j // 5 + 1,
                                               "parameter": ("kappa", "theta", "sigma", "rho", "v0")[j % 5],
                                               "truth": float(actual), "estimate": float(estimated),
                                               "absolute_error": error, "relative_error": None if correlation else error / actual,
                                               "gate_units": error / tolerance, "passed": bool(error <= tolerance)})
                    record.update(parameters=parameter_rows, all_parameter_pass=all(row["passed"] for row in parameter_rows),
                                  max_gate_units=max(row["gate_units"] for row in parameter_rows),
                                  calibration_iv_rmse=float(np.sqrt(np.mean((predictions[~holdout] - targets[~holdout])**2))),
                                  holdout_iv_rmse=float(np.sqrt(np.mean((predictions[holdout] - targets[holdout])**2))),
                                  prediction_iv=predictions.tolist())
            cases.append(record)
            (args.out / "cases_and_all_starts.json").write_text(json.dumps(cases, indent=2, allow_nan=False))
            print(json.dumps({"case": index, "status": record["status"],
                              "all_parameter_pass": record["all_parameter_pass"],
                              "holdout_iv_rmse": record.get("holdout_iv_rmse"),
                              "max_gate_units": record.get("max_gate_units")}), flush=True)
    if sha256(checkpoint_info["checkpoint"]) != checkpoint_info["sha256"]:
        raise RuntimeError("Checkpoint changed during the diagnostic")
    for path in sources:
        if sha256(path) != manifest["source_sha256"][str(path.relative_to(ROOT))]:
            raise RuntimeError(f"Source changed during diagnostic: {path}")
    summary = {
        "status": "complete", "evidence_level": manifest["evidence_level"],
        "cases": args.cases, "fitted_cases": sum(row["status"] == "fitted" for row in cases),
        "all_parameter_passes": sum(row["all_parameter_pass"] for row in cases),
        "parameter_pass_fraction": sum(row["all_parameter_pass"] for row in cases) / args.cases,
        "optimizer_successes": sum(row.get("fit", {}).get("optimizer_success", False) for row in cases),
        "seconds": time.perf_counter() - started,
        "largest_holdout_iv_rmse": max((row.get("holdout_iv_rmse", 0) for row in cases), default=None),
        "interpretation": "Passing diagnoses successful inversion of this frozen neural function on new noiseless examples; it does not measure accuracy against Double Heston Fourier targets",
    }
    manifest.update(status="complete", frozen_source_and_checkpoint_hashes_rechecked=True,
                    forbidden_fourier_runtime_guard_passed=True, seconds=summary["seconds"])
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
