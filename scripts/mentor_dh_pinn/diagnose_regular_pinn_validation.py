#!/usr/bin/env python3
"""Development-only Jacobian diagnosis on the four fixed validation surfaces.

Truth is used to diagnose the forward surrogate after training, never to alter
calibration inputs or fit starts. This script does not fit parameters or select
a model. The first-order shifts are local diagnostics, not recovered parameters.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.mentor_dh_pinn.assess_regular_pinn import load_checkpoint, sha256
from scripts.mentor_dh_pinn.train_regular_pinn import make_validation_surfaces
from src.mentor_dh_pinn.regular_pinn_data import coordinates, decode_unit, exact_prices, invert_total_variance
from src.mentor_dh_pinn.regular_pinn_torch import TorchRegularVariancePINN


def exact_iv_jacobian(q, factors, nodes):
    query = torch.tensor(q, dtype=torch.float64, requires_grad=True)
    price = exact_prices(query, factors, nodes)
    dp = torch.autograd.grad(price.sum(), query)[0].detach().numpy()[:, 2:]
    x, tau = q[:, 0], np.exp(q[:, 1])
    w = invert_total_variance(price.detach().numpy(), x)
    iv = np.sqrt(w / tau)
    d2 = x / np.sqrt(w) - .5 * np.sqrt(w)
    price_vega = np.exp(-.5 * d2**2) / math.sqrt(2 * math.pi) * np.sqrt(tau)
    return iv, dp / price_vega[:, None]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Preserve previous diagnostics; choose a new output")
    torch.set_num_threads(1)
    model, info = load_checkpoint(args.checkpoint)
    network = model if isinstance(model, TorchRegularVariancePINN) else TorchRegularVariancePINN.from_mlx(model)
    rows = []
    for case, (q, _, truth) in enumerate(make_validation_surfaces(model.factors)):
        mask = np.tile(np.arange(21) % 3 != 2, 6)
        q = q[mask]
        true_iv, true_jac = exact_iv_jacobian(q, model.factors, 128)
        other_iv, other_jac = exact_iv_jacobian(q, model.factors, 96)
        query = torch.tensor(q, dtype=torch.float64, requires_grad=True)
        state, structural = coordinates(query, model.factors, torch)
        neural_iv = network.iv(state, structural)
        neural_jac = torch.autograd.grad(neural_iv.sum(), query)[0].detach().numpy()[:, 2:]
        error = neural_iv.detach().numpy() - true_iv
        unit = torch.tensor(q[0, 2:], dtype=torch.float64, requires_grad=True)
        physical_jac = torch.autograd.functional.jacobian(lambda u: decode_unit(u, model.factors, torch), unit).numpy()
        tolerance = .05 * truth
        tolerance[3::5] = .05
        # One scaled-coordinate unit is precisely the declared per-parameter
        # recovery tolerance: 5% for positives, absolute .05 for correlations.
        transform = np.linalg.solve(physical_jac, np.diag(tolerance))
        j = true_jac @ transform
        j_ann = neural_jac @ transform
        left, singular, right = np.linalg.svd(j, full_matrices=False)
        ann_singular = np.linalg.svd(j_ann, compute_uv=False)
        projected = left.T @ error
        local_shift = -right.T @ (projected / singular)
        rows.append({
            "case": case, "truth": truth.tolist(), "calibration_quotes": len(q),
            "iv_rmse_at_generating_parameters": float(np.sqrt(np.mean(error**2))),
            "iv_max_at_generating_parameters": float(np.max(np.abs(error))),
            "exact_iv_quadrature_max_difference": float(np.max(np.abs(true_iv - other_iv))),
            "exact_iv_jacobian_quadrature_max_difference": float(np.max(np.abs(true_jac - other_jac))),
            "unit_jacobian_relative_frobenius_error": float(np.linalg.norm(neural_jac - true_jac) / np.linalg.norm(true_jac)),
            "exact_tolerance_scaled_singular_values": singular.tolist(),
            "neural_tolerance_scaled_singular_values": ann_singular.tolist(),
            "exact_tolerance_scaled_condition_number": float(singular[0] / singular[-1]),
            "iv_error_projected_on_exact_left_singular_vectors": projected.tolist(),
            "linearized_parameter_shift_in_tolerance_units": local_shift.tolist(),
            "linearized_max_parameter_shift_in_tolerance_units": float(np.max(np.abs(local_shift))),
            "interpretation": "Local linear surrogate-bias diagnostic; large shifts can leave the valid box and are not fitted parameters.",
        })
    output = {
        "scope": "four previously declared synthetic validation cases; development only; no new final assessment",
        "checkpoint": info, "script_sha256": sha256(__file__),
        "parameter_coordinates": "canonical physical parameters, scaled by 5% of truth for positives and .05 for rho",
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    for row in rows:
        print(json.dumps({k: row[k] for k in ("case", "iv_rmse_at_generating_parameters",
              "exact_tolerance_scaled_condition_number", "linearized_max_parameter_shift_in_tolerance_units")}))


if __name__ == "__main__":
    main()
