#!/usr/bin/env python3
"""Evaluate one regular Double Heston checkpoint on the fixed development cases."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.mentor_dh_pinn.assess_regular_pinn import fit_network, sha256
from scripts.mentor_dh_pinn.finetune_regular_pinn_lbfgs import load_torch_checkpoint
from src.mentor_dh_pinn.regular_pinn_data import decode_unit, teacher_labels


def development_surfaces():
    rng = np.random.default_rng(906311)
    units = rng.uniform(0.1, 0.9, (4, 10))
    x = np.tile(-np.log(np.linspace(0.8, 1.2, 21)), 6)
    tau = np.repeat(np.asarray([30, 60, 90, 180, 365, 730]) / 365.0, 21)
    surfaces = []
    for unit in units:
        q = np.column_stack([x, np.log(tau), np.broadcast_to(unit, (len(x), len(unit)))])
        labels = teacher_labels(q, 2, gradients=False)
        if not labels["usable"].all():
            raise ValueError("Fixed development surface contains invalid numerical labels")
        surfaces.append((q, labels, decode_unit(unit, 2)))
    return surfaces


def evaluate(model, *, starts=3, max_nfev=200):
    errors, records, passes = [], [], 0
    for case, (q, labels, truth) in enumerate(development_surfaces()):
        tau = np.exp(q[:, 1])
        observed_iv = np.sqrt(labels["w"] / tau)
        fit_mask = np.tile(np.arange(21) % 3 != 2, 6)
        result = fit_network(
            model, q[:, 0], tau, observed_iv, fit_mask=fit_mask,
            starts=starts, seed=906777, max_nfev=max_nfev,
        )
        record = {"case": case, "true": truth.tolist(), "fit": result}
        if result["status"] != "fitted":
            errors.extend([math.inf] * 10)
            record.update(status=result["status"], parameter_gate=False)
            records.append(record)
            continue
        estimated = np.asarray(result["physical"])
        scale = np.abs(truth).copy()
        scale[3::5] = 0.5
        scaled_error = (estimated - truth) / scale
        gate_units = np.abs(scaled_error) / 0.05
        passed = bool(np.all(gate_units <= 1.0))
        passes += int(passed)
        errors.extend(scaled_error.tolist())
        record.update(
            status="fitted",
            estimated=estimated.tolist(),
            scaled_error=scaled_error.tolist(),
            gate_units=gate_units.tolist(),
            parameter_gate=passed,
        )
        records.append(record)
    return {
        "validation_parameter_rmse": float(np.sqrt(np.mean(np.asarray(errors) ** 2))),
        "all_parameter_passes": passes,
        "cases": len(records),
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--max-nfev", type=int, default=200)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Preserve previous development evidence; choose a new output")
    if min(args.starts, args.max_nfev) < 1:
        parser.error("starts and max-nfev must be positive")
    model, config, weights = load_torch_checkpoint(args.checkpoint)
    model.eval().requires_grad_(False)
    result = evaluate(model, starts=args.starts, max_nfev=args.max_nfev)
    output = {
        "scope": "four fixed previously declared Double Heston development surfaces",
        "truth_seed": 906311,
        "fit_seed": 906777,
        "parameter_gate": "all positive relative errors <= 5%; both rho absolute errors <= .05",
        "checkpoint": str(weights),
        "checkpoint_sha256": sha256(weights),
        "checkpoint_config": config,
        "script_sha256": sha256(Path(__file__)),
        **result,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    print(json.dumps({
        "validation_parameter_rmse": output["validation_parameter_rmse"],
        "all_parameter_passes": output["all_parameter_passes"],
        "cases": output["cases"],
    }))


if __name__ == "__main__":
    main()
