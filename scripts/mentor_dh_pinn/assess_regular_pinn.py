#!/usr/bin/env python3
"""Frozen regular-PINN recovery assessment; numerical fitting uses only its ANN.

Example: python scripts/mentor_dh_pinn/assess_regular_pinn.py \
    --checkpoint outputs/regular_pinn/run --out outputs/regular_pinn/assessment --seed 73121

Exact synthetic prices are generated before fitting and exact fitted-parameter
prices are evaluated afterwards. Neither exact prices at trial parameters nor
generating parameters can enter fit_network's objective or start selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.mentor_dh_pinn.regular_pinn import RegularVariancePINN
from src.mentor_dh_pinn.regular_pinn_torch import TorchRegularVariancePINN
from src.mentor_dh_pinn.regular_pinn_data import (
    black_call, coordinates, decode_unit, exact_prices, invert_total_variance,
)

PARAMETERS = ["kappa", "theta", "sigma", "rho", "v0"]


def fit_network(model, x, tau, observed_iv, *, fit_mask=None, starts=5,
                seed=71, max_nfev=400):
    """Fit unit parameters with ANN IVs and analytic float64 Jacobians only.

    The mask is applied before accessing values, validating quotes, making a
    starting surface or computing an objective. No holdout or true parameter
    argument exists. Candidate selection is minimum calibration ANN-IV SSE.
    Every attempted start, including failures, is retained.
    """
    x, tau, observed_iv = [np.asarray(a, dtype=float) for a in (x, tau, observed_iv)]
    if x.ndim != 1 or x.shape != tau.shape or x.shape != observed_iv.shape:
        raise ValueError("x, tau and observed_iv must be equally sized one-dimensional arrays")
    mask = np.ones(x.shape, dtype=bool) if fit_mask is None else np.asarray(fit_mask, dtype=bool)
    if mask.shape != x.shape:
        raise ValueError("fit_mask shape must equal x shape")
    x, tau, observed_iv = x[mask], tau[mask], observed_iv[mask]
    if not len(x) or not np.isfinite(np.stack([x, tau, observed_iv])).all():
        raise ValueError("all calibration quotes must be finite")
    if (tau <= 0).any() or (observed_iv <= 0).any() or starts < 1 or max_nfev < 1:
        raise ValueError("positive maturities, IVs and optimizer budgets required")
    started = time.perf_counter()
    network = model if isinstance(model, torch.nn.Module) else TorchRegularVariancePINN.from_mlx(model)
    factors = network.factors
    dimension = 5 * factors
    geometry = np.column_stack([x, np.log(tau)])
    cached_u = None
    cached = None

    def residual_and_jacobian(unit):
        nonlocal cached_u, cached
        if cached_u is None or not np.array_equal(unit, cached_u):
            query = torch.tensor(np.column_stack([geometry, np.broadcast_to(unit, (len(x), dimension))]),
                                 dtype=torch.float64, requires_grad=True)
            state, structural = coordinates(query, factors, torch)
            value = network.iv(state, structural)
            # Quote rows are independent. Their sum-gradient gives one complete
            # parameter Jacobian row per quote without a dense batch Jacobian.
            jacobian = torch.autograd.grad(value.sum(), query)[0][..., 2:]
            value, jacobian = value.detach().numpy(), jacobian.detach().numpy()
            if not np.isfinite(value).all() or not np.isfinite(jacobian).all():
                raise FloatingPointError("nonfinite neural price or analytic parameter derivative")
            cached_u, cached = unit.copy(), (value - observed_iv, jacobian)
        return cached

    initial_units = 0.05 + 0.9 * qmc.LatinHypercube(d=dimension, seed=seed).random(starts)
    records = []
    candidates = []
    for index, initial in enumerate(initial_units):
        record = {"start": index, "initial_unit": initial.tolist()}
        start_time = time.perf_counter()
        try:
            initial_residual = residual_and_jacobian(initial)[0]
            record["initial_sse"] = float(initial_residual @ initial_residual)
            result = least_squares(
                lambda unit: residual_and_jacobian(unit)[0], initial,
                jac=lambda unit: residual_and_jacobian(unit)[1],
                bounds=(1e-5, 1.0 - 1e-5), method="trf", x_scale="jac",
                max_nfev=max_nfev, ftol=1e-12, xtol=1e-8, gtol=1e-12,
            )
            final_residual = residual_and_jacobian(result.x)[0]
            sse = float(final_residual @ final_residual)
            record.update(unit=result.x.tolist(), physical=decode_unit(result.x, factors).tolist(),
                          sse=sse, nfev=int(result.nfev), njev=int(result.njev or 0),
                          optimizer_success=bool(result.success), status=int(result.status),
                          message=str(result.message))
            if np.isfinite(sse):
                candidates.append((sse, index, result.x.copy(), bool(result.success)))
        except (FloatingPointError, ValueError, RuntimeError) as exc:
            record.update(sse=None, optimizer_success=False, error=f"{type(exc).__name__}: {exc}")
        record["seconds"] = time.perf_counter() - start_time
        records.append(record)
    if not candidates:
        return {"status": "all_starts_failed", "starts": records,
                "inference": "identical learned weights evaluated in float64 PyTorch; no retraining or exact pricing",
                "seconds": time.perf_counter() - started, "fit_quotes": len(x)}
    sse, chosen, unit, success = min(candidates, key=lambda row: row[0])
    return {"status": "fitted", "unit": unit.tolist(), "physical": decode_unit(unit, factors).tolist(),
            "inference": "identical learned weights evaluated in float64 PyTorch; no retraining or exact pricing",
            "calibration_iv_sse": sse, "selected_start": chosen,
            "optimizer_success": success, "starts": records,
            "seconds": time.perf_counter() - started, "fit_quotes": len(x),
            "total_nfev": sum(row.get("nfev", 0) for row in records)}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(path):
    path = Path(path).resolve()
    directory = path if path.is_dir() else path.parent
    config = json.loads((directory / "config.json").read_text())
    keys = {"factors", "width", "depth", "tau_min", "tau_max", "x_half_width", "correction_limit"}
    architecture = {key: value for key, value in config.items() if key in keys}
    checkpoint_format = config.get("checkpoint_format", "mlx_safetensors")
    if checkpoint_format == "convolution_price_pinn_float64":
        from src.mentor_dh_pinn.convolution_pinn import ConvolutionPINN
        weights = directory / "model.pt" if path.is_dir() else path
        state = torch.load(weights, map_location="cpu", weights_only=True)
        if not state or any(v.dtype != torch.float64 or not torch.isfinite(v).all() for v in state.values()):
            raise ValueError('Expected finite float64 composition checkpoint tensors')
        component = TorchRegularVariancePINN(**config['component_architecture'])
        model = ConvolutionPINN(component, nodes=config['quadrature_nodes'])
        model.load_state_dict(state, strict=True)
        model.eval().requires_grad_(False)
    elif checkpoint_format == "torch_state_dict_float64":
        architecture['residual_blocks'] = config.get('residual_blocks', 0)
        weights = directory / "model.pt" if path.is_dir() else path
        state = torch.load(weights, map_location="cpu", weights_only=True)
        if not state or any(v.dtype != torch.float64 or not torch.isfinite(v).all()
                            for v in state.values()):
            raise ValueError("Expected finite float64 regular-PINN checkpoint tensors")
        model = TorchRegularVariancePINN(**architecture)
        model.load_state_dict(state, strict=True)
        model.eval().requires_grad_(False)
    elif checkpoint_format == "mlx_safetensors":
        weights = directory / "model.safetensors" if path.is_dir() else path
        model = RegularVariancePINN(**architecture)
        model.load_weights(str(weights))
        model.eval()
        mx.eval(model.parameters())
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_format}")
    return model, {"label": directory.name, "checkpoint": str(weights),
                   "sha256": sha256(weights), "config": config,
                   "config_sha256": sha256(directory / "config.json")}


def _query(x, tau, unit):
    return np.column_stack([x, np.log(tau), np.broadcast_to(unit, (len(x), len(unit)))])


def _exact(x, tau, unit, factors, nodes=128):
    with torch.no_grad():
        return exact_prices(torch.tensor(_query(x, tau, unit), dtype=torch.float64), factors, nodes).numpy()


def _network_iv(model, x, tau, unit):
    network = model if isinstance(model, torch.nn.Module) else TorchRegularVariancePINN.from_mlx(model)
    with torch.no_grad():
        state, structural = coordinates(torch.tensor(_query(x, tau, unit), dtype=torch.float64), network.factors, torch)
        return network.iv(state, structural).detach().numpy()


def _rmse(value):
    value = np.asarray(value, float)
    return float(np.sqrt(np.mean(value**2))) if np.isfinite(value).all() and len(value) else None


def _score_price_iv(predicted, true_price, true_iv, x, tau, holdout):
    implied = np.sqrt(invert_total_variance(predicted, x) / tau)
    spot_error = (predicted - true_price) * np.exp(-x)
    all_price = _rmse(spot_error)
    heldout_price = _rmse(spot_error[holdout])
    return {"all_price_rmse_spot": all_price,
            "holdout_price_rmse_spot": heldout_price,
            "all_iv_rmse": _rmse(implied - true_iv),
            "holdout_iv_rmse": _rmse((implied - true_iv)[holdout]),
            "invalid_iv_quotes": int((~np.isfinite(implied)).sum()),
            "price_gate": bool(heldout_price is not None and heldout_price <= 1e-5)}, implied


def _write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=73121)
    parser.add_argument("--cases", type=int, default=12)
    parser.add_argument("--max-nfev", type=int, default=400)
    parser.add_argument("--starts", type=int, default=5)
    parser.add_argument("--geometry", choices=("monthly", "rich"), action="append",
                        help="Restrict geometries; default evaluates both")
    parser.add_argument("--noise", type=float, choices=(0.0, 0.01), action="append",
                        help="Restrict noise conditions; default evaluates both")
    parser.add_argument("--development", action="store_true",
                        help="Label reused/exposed cases as development, never unseen assessment")
    args = parser.parse_args()
    if args.cases < 1:
        parser.error("--cases must be positive")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("Output must be empty; completed or partial evidence will not be overwritten")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    loaded = [load_checkpoint(path) for path in args.checkpoint]
    labels = [info["label"] for _, info in loaded]
    if len(labels) != len(set(labels)):
        raise ValueError("Checkpoint run directory names must be unique")
    manifest = {
        "status": "running", "seed": args.seed, "cases_per_model_family": args.cases,
        "evidence_level": "development on exposed cases" if args.development else "frozen assessment",
        "evaluated_geometries": sorted(set(args.geometry or ("monthly", "rich"))),
        "evaluated_noise_levels": sorted(set(args.noise or (0.0, 0.01))),
        "truth_sampling": "Independent uniform unit coordinates in [0.1,0.9]; central domain only, not full-boundary coverage",
        "scope": "Frozen-model synthetic assessment; one training initialization does not establish seed-wise reliability or market-data recovery",
        "noise": "independent Gaussian standard deviation 1% of true option time value above intrinsic; no truncation/resampling",
        "selection": "minimum ANN IV calibration SSE across fixed seeded LHS starts",
        "fit_information": "calibration strikes and observed IV only; no truth, holdout, exact-pricer trial evaluations",
        "geometry": {"strikes": np.linspace(0.8, 1.2, 21).tolist(),
                     "monthly_days": [30, 60, 90], "rich_days": [30, 60, 90, 180, 365, 730],
                     "holdout": "zero-based strike index % 3 == 2 at every expiry", "spot": 1.0},
        "parameter_gate": "all positive-parameter relative errors <= 5%; all rho absolute errors <= .05",
        "price_gate": "RMSE over heldout quotes <= 1e-5 of spot; original exact synthetic prices are the scoring target",
        "invalid_policy": "Any noninvertible generated or noisy observation invalidates the entire case; counts remain in denominators",
        "quadrature_gate": "true and fitted 96-vs-128 node prices differ by <= 1e-8 of spot",
        "optimizer": {"max_nfev_per_start": args.max_nfev, "starts": args.starts,
                      "algorithm": "bounded SciPy TRF with analytic PyTorch Jacobian; frozen checkpoint weights evaluated in float64",
                      "ftol": 1e-12, "gtol": 1e-12, "xtol": 1e-8,
                      "unit_parameter_bounds": [1e-5, 1.0 - 1e-5]},
        "checkpoints": [info for _, info in loaded],
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in (
            Path(__file__), ROOT / "src/mentor_dh_pinn/regular_pinn.py",
            ROOT / "src/mentor_dh_pinn/regular_pinn_torch.py",
            ROOT / "src/mentor_dh_pinn/regular_pinn_data.py", ROOT / "src/mentor_dh_pinn/torch_pricer.py")},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    metrics, parameters, fits, quote_rows, truths = [], [], [], [], []
    started = time.perf_counter()
    for factors in sorted({model.factors for model, _ in loaded}):
        rng = np.random.default_rng(args.seed + factors * 10000)
        truth_units = rng.uniform(0.1, 0.9, (args.cases, 5 * factors))
        relevant = [(model, info) for model, info in loaded if model.factors == factors]
        for case_number, truth_unit in enumerate(truth_units):
            case_id = f"f{factors}_{args.seed}_{case_number:03d}"
            physical_truth = decode_unit(truth_unit, factors)
            truths.append({"case": case_id, "factors": factors, "unit": truth_unit.tolist(),
                           "physical": physical_truth.tolist()})
            for geometry, days in (("monthly", [30, 60, 90]), ("rich", [30, 60, 90, 180, 365, 730])):
                strike = np.tile(np.linspace(0.8, 1.2, 21), len(days))
                tau = np.repeat(np.asarray(days) / 365.0, 21)
                x = np.log(1.0 / strike)
                holdout = np.tile(np.arange(21) % 3 == 2, len(days))
                truth_price = _exact(x, tau, truth_unit, factors)
                other_price = _exact(x, tau, truth_unit, factors, 96)
                quadrature_error = float(np.max(np.abs((truth_price - other_price) * strike)))
                true_iv = np.sqrt(invert_total_variance(truth_price, x) / tau)
                for noise in (0.0, 0.01):
                    intrinsic = np.maximum(np.expm1(x), 0.0)
                    observed = intrinsic + (truth_price - intrinsic) * (1.0 + noise * rng.normal(size=len(x)))
                    observed_iv = np.sqrt(invert_total_variance(observed, x) / tau)
                    # Consume the same RNG draws even for excluded conditions,
                    # so filtering does not silently change seeded observations.
                    if args.geometry and geometry not in args.geometry:
                        continue
                    if args.noise is not None and noise not in args.noise:
                        continue
                    invalid = ~np.isfinite(observed_iv) | ~np.isfinite(true_iv)
                    tag = {"case": case_id, "factors": factors, "geometry": geometry, "noise": noise,
                           "quotes": len(x), "calibration_quotes": int((~holdout).sum()),
                           "holdout_quotes": int(holdout.sum()), "invalid_observed_quotes": int(invalid.sum()),
                           "true_quadrature_max_error_spot": quadrature_error}
                    valid = not invalid.any() and quadrature_error <= 1e-8
                    # Preserve every observed value, including observations that
                    # invalidate a case before any calibration takes place.
                    for quote in range(len(x)):
                        quote_rows.append({**tag, "quote": quote, "holdout": bool(holdout[quote]),
                                           "strike": strike[quote], "tau": tau[quote],
                                           "true_price_spot": truth_price[quote] * strike[quote],
                                           "observed_price_spot": observed[quote] * strike[quote],
                                           "true_iv": true_iv[quote], "observed_iv": observed_iv[quote]})
                    for model, info in relevant:
                        row = {**tag, "model": info["label"], "parameter_gate": False,
                               "neural_price_gate": False, "exact_price_gate": False,
                               "joint_recovery_gate": False}
                        if not valid:
                            row["status"] = "invalid_observations" if invalid.any() else "true_quadrature_failure"
                            metrics.append(row)
                            continue
                        # Only training-visible quotes reach fit_network.
                        fit = fit_network(model, x, tau, observed_iv, fit_mask=~holdout,
                                          starts=args.starts, seed=args.seed + 100 + case_number,
                                          max_nfev=args.max_nfev)
                        fits.append({**tag, "model": info["label"], **fit})
                        row.update(status=fit["status"], seconds=fit["seconds"],
                                   optimizer_success=fit.get("optimizer_success", False),
                                   total_nfev=fit.get("total_nfev", 0))
                        if fit["status"] != "fitted":
                            metrics.append(row)
                            continue
                        unit = np.asarray(fit["unit"])
                        fitted_parameters = np.asarray(fit["physical"])
                        neural_iv = _network_iv(model, x, tau, unit)
                        neural_price = black_call(x, neural_iv**2 * tau)
                        # First post-fit exact evaluation of the selected vector.
                        exact_price = _exact(x, tau, unit, factors)
                        exact_other = _exact(x, tau, unit, factors, 96)
                        fitted_quad = float(np.max(np.abs((exact_price - exact_other) * strike)))
                        gates = []
                        for j, (actual, recovered) in enumerate(zip(physical_truth, fitted_parameters)):
                            name = PARAMETERS[j % 5]
                            absolute = float(abs(recovered - actual))
                            relative = None if name == "rho" else absolute / float(actual)
                            gate = absolute <= 0.05 if name == "rho" else relative <= 0.05
                            gates.append(gate)
                            parameters.append({**tag, "model": info["label"], "factor": j // 5 + 1,
                                               "parameter": name, "truth": actual, "estimate": recovered,
                                               "absolute_error": absolute, "relative_error": relative,
                                               "parameter_pass": bool(gate)})
                        row.update(parameter_gate=bool(all(gates)),
                                   max_positive_parameter_relative_error=float(max(abs(fitted_parameters[j] / physical_truth[j] - 1)
                                       for j in range(len(physical_truth)) if j % 5 != 3)),
                                   max_rho_absolute_error=float(max(abs(fitted_parameters[j] - physical_truth[j])
                                       for j in range(len(physical_truth)) if j % 5 == 3)),
                                   fitted_quadrature_max_error_spot=fitted_quad)
                        for prefix, prices in (("neural", neural_price), ("exact", exact_price)):
                            scored, _ = _score_price_iv(prices, truth_price, true_iv, x, tau, holdout)
                            row.update({f"{prefix}_{key}": value for key, value in scored.items()})
                        row["exact_price_gate"] = row["exact_price_gate"] and fitted_quad <= 1e-8
                        row["joint_recovery_gate"] = bool(row["parameter_gate"] and row["exact_price_gate"] and row["neural_price_gate"])
                        metrics.append(row)
                    # On DH surfaces this is explicitly a misspecified BS fit.
                    # It has one IV parameter and no ten-parameter recovery claim.
                    if factors == 2:
                        bs = {**tag, "model": "black_scholes_on_double_heston",
                              "status": "fitted" if valid else "invalid_observations",
                              "parameter_gate": None, "joint_recovery_gate": None,
                              "interpretation": "misspecified one-volatility price benchmark; no DH parameter recovery target"}
                        if valid:
                            sigma = float(np.mean(observed_iv[~holdout]))
                            scored, _ = _score_price_iv(black_call(x, sigma**2 * tau), truth_price, true_iv, x, tau, holdout)
                            bs.update(sigma=sigma, **{f"neural_{key}": value for key, value in scored.items()})
                        metrics.append(bs)
            _write_csv(args.out / "metrics.csv", metrics)
            _write_csv(args.out / "parameter_recovery.csv", parameters)
            _write_csv(args.out / "observations.csv", quote_rows)
            (args.out / "fits_and_starts.json").write_text(json.dumps(fits, indent=2))
            (args.out / "synthetic_truth.json").write_text(json.dumps(truths, indent=2))
            print(f"completed {case_id}; {len(metrics)} metric rows", flush=True)
    summary = []
    for model_name, factors, geometry, noise in sorted({(r["model"], r["factors"], r["geometry"], r["noise"]) for r in metrics}):
        rows = [r for r in metrics if (r["model"], r["factors"], r["geometry"], r["noise"]) == (model_name, factors, geometry, noise)]
        summary.append({"model": model_name, "factors": factors, "geometry": geometry, "noise": noise,
                        "cases": len(rows), "fitted": sum(r["status"] == "fitted" for r in rows),
                        "invalid_cases": sum(r["status"] != "fitted" for r in rows),
                        "all_parameter_passes": sum(r.get("parameter_gate") is True for r in rows),
                        "neural_price_passes": sum(r.get("neural_price_gate") is True for r in rows),
                        "exact_price_passes": sum(r.get("exact_price_gate") is True for r in rows),
                        "joint_recovery_passes": sum(r.get("joint_recovery_gate") is True for r in rows)})
    for _,info in loaded:
        if sha256(info['checkpoint']) != info['sha256']:
            raise RuntimeError('Checkpoint changed during frozen assessment')
    for relative,digest in manifest['source_sha256'].items():
        if sha256(ROOT/relative) != digest:
            raise RuntimeError(f'Assessment source changed during execution: {relative}')
    manifest.update(status="complete", frozen_checkpoint_and_source_hashes_rechecked=True,
                    seconds=time.perf_counter() - started,
                    metric_rows=len(metrics), parameter_rows=len(parameters), observation_rows=len(quote_rows))
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
