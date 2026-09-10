#!/usr/bin/env python3
"""Report selected regular-PINN development-validation evidence without retraining.

Usage: python scripts/mentor_dh_pinn/report_regular_pinn.py --run RUN_DIR \
    --run ANOTHER_RUN_DIR --out NEW_REPORT_DIR
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PARAMETERS = ("kappa", "theta", "sigma", "rho", "v0")
LABEL = "DEVELOPMENT VALIDATION — NOT AN UNSEEN FINAL TEST"


def finite(value):
    return isinstance(value, (float, int)) and math.isfinite(value)


def show(value, digits=5):
    return f"{value:.{digits}g}" if finite(value) else "unavailable"


def clean_json(value):
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def read_json(path, warnings, hashes, default=None):
    if not path.exists():
        return default
    raw = path.read_bytes()
    hashes[str(path)] = hashlib.sha256(raw).hexdigest()
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        warnings.append(f"Could not read {path.name}: {exc}; the run may be updating it.")
        return default


def collect_run(directory):
    directory = directory.resolve()
    hashes, warnings = {}, []
    config = read_json(directory / "config.json", warnings, hashes, {})
    correction = read_json(directory / "effective_optimizer_settings.json", warnings, hashes, {})
    if correction:
        config = {**config, **correction.get("effective_config", {})}
        if "correction" in correction.get("artifact_type", ""):
            warnings.append("Effective L-BFGS settings overlay the explicitly preserved inherited-Adam metadata; see effective_optimizer_settings.json.")
    history = read_json(directory / "history.json", warnings, hashes, [])
    # Display both optimizer types without equating their computational budgets.
    history = [{**row, "step": row.get("step", row.get("iteration"))} for row in history]
    selection = read_json(directory / "selection.json", warnings, hashes, {})
    complete = read_json(directory / "complete.json", warnings, hashes)
    failure = read_json(directory / "FAILED.json", warnings, hashes)
    manifest = read_json(directory / "manifest.json", warnings, hashes, {})
    is_lbfgs = "iteration" in selection
    step = selection.get("step", selection.get("iteration"))
    selected = next((row for row in history if row.get("step") == step), {})
    if step is not None and not selected:
        warnings.append("Selected checkpoint step has no matching history entry; selected IV is unavailable.")
    recovery = []
    if step is not None:
        recovery = read_json(directory / f"recovery_validation_{step:06d}.json", warnings, hashes, [])
    factors = config.get("factors")
    label = directory.name
    expected_cases = 0 if config.get("skip_recovery", False) else 4
    if expected_cases and not recovery:
        warnings.append("No selected-checkpoint recovery-validation records are available.")
    if recovery and len(recovery) != expected_cases:
        warnings.append(f"Found {len(recovery)} recovery records; the configured study expects {expected_cases}.")
    case_ids = [row.get("case") for row in recovery]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError(f"Duplicate recovery case ids in {directory}")
    if any(not isinstance(i, int) or i not in range(4) for i in case_ids):
        raise ValueError(f"Unexpected validation case ids in {directory}")
    parameter_rows, case_rows = [], []
    for case in recovery:
        truth, estimates = case.get("true", []), case.get("estimated", [])
        if factors and len(truth) != factors * 5:
            warnings.append(f"Case {case.get('case')} has an unexpected truth-vector length.")
        gates, training_errors = [], []
        for j, actual in enumerate(truth):
            name = PARAMETERS[j % 5]
            estimate = estimates[j] if j < len(estimates) else None
            known = finite(actual) and finite(estimate)
            error = estimate - actual if known else None
            threshold = 0.05 if name == "rho" else abs(actual) * 0.05
            training_scale = 0.5 if name == "rho" else abs(actual)
            scaled = error / training_scale if known and training_scale > 0 else None
            gate_error = abs(error) / threshold if known and threshold > 0 else None
            passed = gate_error is not None and gate_error <= 1.0
            gates.append(passed)
            training_errors.append(scaled)
            parameter_rows.append({
                "run": label, "selected_step": step, "case": case.get("case"),
                "factor": j // 5 + 1, "parameter": name, "truth": actual,
                "estimate": estimate, "signed_error": error,
                "relative_error": error / actual if known and name != "rho" and actual != 0 else None,
                "training_scaled_error": scaled, "absolute_gate_tolerance": threshold,
                "absolute_error_in_gate_units": gate_error, "parameter_pass": passed,
                "optimizer_success": case.get("success", False),
            })
        available = factors is not None and len(estimates) == len(truth) == 5 * factors
        case_rows.append({"run": label, "selected_step": step, "case": case.get("case"),
                          "estimates_available": available, "optimizer_success": case.get("success", False),
                          "parameter_passes": sum(gates), "parameter_count": 5 * factors if factors else None,
                          "all_parameter_pass": bool(available and all(gates)),
                          "recomputed_training_scaled_rmse": float(np.sqrt(np.mean(np.square(training_errors))))
                              if available and all(finite(e) for e in training_errors) else None})
    prefix = "iteration" if is_lbfgs else "step"
    source_weights = directory / f"{prefix}_{step:06d}.safetensors" if step is not None else directory / "model.safetensors"
    if not source_weights.exists() and step is not None:
        warnings.append("The named selected-step weights are missing; no checkpoint hash is inferred from another step.")
    weights_hash = hashlib.sha256(source_weights.read_bytes()).hexdigest() if source_weights.exists() else None
    recomputed_errors = [row["training_scaled_error"] for row in parameter_rows]
    recomputed = (float(np.sqrt(np.mean(np.square(recomputed_errors))))
                  if expected_cases and len(recovery) == expected_cases and recomputed_errors
                  and all(finite(e) for e in recomputed_errors) else None)
    stored = selected.get("validation_parameter_rmse")
    if finite(stored) and finite(recomputed) and not np.isclose(stored, recomputed, atol=1e-10, rtol=1e-8):
        warnings.append("Stored and recomputed selected parameter RMSE differ; inspect input artifacts.")
    data_manifest = {}
    if config.get("data"):
        data_dir = Path(config["data"])
        if not data_dir.is_absolute():
            data_dir = ROOT / data_dir
        data_manifest = read_json(data_dir / "manifest.json", warnings, hashes, {})
    status = "failed" if failure is not None else "complete" if complete is not None else "incomplete / live status not checked"
    summary = {
        "run": label, "directory": str(directory), "status": status,
        "factors": factors, "training_seed": config.get("seed"),
        "sensitivity_weight": config.get("sensitivity"), "weight_decay": config.get("weight_decay"),
        "resumed_from": config.get("resume"), "independent_initialization": not bool(config.get("resume")),
        "optimizer": config.get("optimizer", "AdamW"),
        "update_kind": "L-BFGS iteration" if is_lbfgs else "AdamW step",
        "requested_steps": config.get("iterations") if is_lbfgs else config.get("steps"),
        "latest_recorded_step": history[-1].get("step") if history else None,
        "selected_step": step, "selection_metric": selection.get("metric", config.get("selection")), "selection_score": selection.get("score"),
        "selected_validation_iv_rmse": selected.get("validation_iv_rmse"),
        "selected_validation_parameter_rmse": stored, "recomputed_validation_parameter_rmse": recomputed,
        "selected_training_loss": selected.get("loss", selected.get("objective")),
        "selected_checkpoint_path": str(source_weights) if weights_hash else None,
        "selected_checkpoint_sha256": weights_hash,
        "validation_cases_expected": expected_cases, "validation_case_records": len(recovery),
        "selected_all_parameter_passes": sum(row["all_parameter_pass"] for row in case_rows) if recovery else None,
        "selected_case_pass_fraction": sum(row["all_parameter_pass"] for row in case_rows) / expected_cases
            if recovery and expected_cases else None,
        "selected_individual_parameter_passes": sum(row["parameter_pass"] for row in parameter_rows) if recovery else None,
        "individual_parameter_denominator": expected_cases * factors * 5 if factors and expected_cases else None,
        "optimizer_successes": sum(bool(row["optimizer_success"]) for row in case_rows) if recovery else None,
        "nonfinite_history_iv_values": sum("validation_iv_rmse" in row and not finite(row["validation_iv_rmse"]) for row in history),
        "nonfinite_history_recovery_values": sum("validation_parameter_rmse" in row and not finite(row["validation_parameter_rmse"]) for row in history),
        "training_seconds": complete.get("seconds") if isinstance(complete, dict) else (history[-1].get("seconds") if history else None),
        "training_candidates": data_manifest.get("splits", {}).get("train", {}).get("candidates"),
        "training_usable": data_manifest.get("splits", {}).get("train", {}).get("usable"),
        "fixed_training_subset": config.get("anchors") if is_lbfgs else None,
        "additional_surface_training_candidates": manifest.get('surface_training',{}).get('candidates',
            manifest.get('surface_training',{}).get('candidate_surfaces',0)),
        "additional_surface_training_usable": manifest.get('surface_training',{}).get('usable',
            manifest.get('surface_training',{}).get('usable_surfaces',0)),
        "parameter_aware_weight":manifest.get('surface_training',{}).get('bias_coefficient',config.get('surface_weight',0.)),
        "parameter_aware_loss":manifest.get('surface_training',{}).get('bias_loss',manifest.get('surface_training',{}).get('loss')),
        "jacobian_consistency_weight":manifest.get('surface_training',{}).get('jacobian_consistency',{}).get('jacobian_coefficient',0.),
        "jacobian_consistency_loss":manifest.get('surface_training',{}).get('jacobian_consistency',{}).get('loss'),
        "validation_usable": data_manifest.get("splits", {}).get("validation", {}).get("usable"),
        "collocation_pool": config.get("collocation") if is_lbfgs else manifest.get("collocation_pool", data_manifest.get("collocation_points")),
        "warnings": warnings,
    }
    return {"summary": summary, "config": config, "history": history,
            "selected_case_records": case_rows, "selected_parameters": parameter_rows,
            "input_sha256": hashes, "training_manifest": manifest, "data_manifest": data_manifest,
            "effective_optimizer_correction": correction}


def csv_write(path, rows):
    if not rows:
        path.write_text("")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(clean_json(rows))


def plot_history(runs, field, title, ylabel, destination):
    figure, axis = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    drawn = False
    for run in runs:
        summary = run["summary"]
        rows = [row for row in run["history"] if finite(row.get(field)) and row[field] > 0]
        if not rows:
            continue
        line, = axis.plot([row["step"] for row in rows], [row[field] for row in rows],
                          label=summary["run"], marker="o" if field.endswith("parameter_rmse") else None,
                          markersize=3, linewidth=1.5)
        selected = next((row for row in rows if row["step"] == summary["selected_step"]), None)
        if selected:
            axis.scatter([selected["step"]], [selected[field]], marker="*", s=150,
                         c=[line.get_color()], edgecolors="black", linewidths=0.7, zorder=4)
        drawn = True
    axis.set(title=title, xlabel="Optimizer update within this run (Adam step or L-BFGS iteration)", ylabel=ylabel)
    if drawn:
        axis.set_yscale("log")
        axis.legend(loc="best", fontsize=8)
    else:
        axis.text(0.5, 0.5, "No finite positive history values available", transform=axis.transAxes, ha="center")
    axis.grid(alpha=0.2)
    figure.suptitle(LABEL, fontsize=10)
    figure.savefig(destination, dpi=170)
    plt.close(figure)


def build_report(runs, output):
    summaries = [run["summary"] for run in runs]
    selected_parameters = [row for run in runs for row in run["selected_parameters"]]
    cases = [row for run in runs for row in run["selected_case_records"]]
    csv_write(output / "run_summary.csv", summaries)
    csv_write(output / "selected_parameter_details.csv", selected_parameters)
    csv_write(output / "selected_case_gates.csv", cases)
    plot_history(runs, "validation_iv_rmse", "Synthetic validation IV error", "IV RMSE (decimal annual volatility)", output / "validation_iv_history.png")
    plot_history(runs, "validation_parameter_rmse", "Four-case validation parameter recovery", "Scaled physical-parameter RMSE", output / "validation_recovery_history.png")
    artifact = {"label": LABEL, "generated_utc": datetime.now(timezone.utc).isoformat(),
                "report_generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "gate": {"positive_parameter_relative_error": 0.05, "rho_absolute_error": 0.05,
                         "case_pass": "all five or ten parameters pass, failed/missing cases stay in the four-case denominator"},
                "runs": runs}
    (output / "evidence.json").write_text(json.dumps(clean_json(artifact), indent=2, allow_nan=False))
    lines = ["# Regular Heston PINN development evidence", "", f"**{LABEL}**", "",
             "All results here use synthetic Heston development data. These four recovery-validation cases are repeatedly used to select checkpoints, so their scores do not establish unseen-test generalisation. Single and Double Heston each use their own generating model and parameter vectors; the model families are not fitted to a common NSE dataset in this report.", "",
             "The regular PINN learns a smooth implied-total-variance function constrained by the pricing PDE. Inverse calibration optimizes the frozen network's IV predictions. Exact Fourier prices supply synthetic targets and training sensitivities; the calibration objective evaluates the neural network with its learned weights promoted to float64.", "",
             "## Selected checkpoints", "",
             "Each IV/recovery value below belongs to the selected checkpoint step. The latest recorded step is shown separately. An incomplete status means no completion artifact was found; this report does not inspect whether a process is running.", "",
             "| Run | Status | Latest / selected step | Selected IV RMSE | Selected parameter RMSE | All-parameter cases passing |", "|---|---|---:|---:|---:|---:|"]
    for row in summaries:
        fraction = (f"{row['selected_all_parameter_passes']}/{row['validation_cases_expected']}"
                    if row["selected_all_parameter_passes"] is not None else "unavailable")
        lines.append(f"| {row['run']} | {row['status']} | {row['latest_recorded_step']} / {row['selected_step']} | {show(row['selected_validation_iv_rmse'])} | {show(row['selected_validation_parameter_rmse'])} | {fraction} |")
    lines += ["", "The case gate requires every positive parameter to be within 5% of its generating value and each correlation to be within 0.05 in absolute units. Optimizer success is recorded separately and does not imply that recovery passed. The training selection score divides positive-parameter errors by their true magnitudes and correlation errors by 0.5; it is therefore different from the gate.", "",
              "## Training histories", "", "![Synthetic validation IV error](validation_iv_history.png)", "",
              "Interpretation: lower IV error means the learned forward function matches synthetic option-implied volatility more closely. A star marks the selected checkpoint. This alone does not show that the inverse problem recovers its generating parameters.", "",
              "![Validation parameter recovery](validation_recovery_history.png)", "",
              "Interpretation: lower values mean smaller parameter errors on the same four development surfaces. A star marks checkpoint selection. Repeated use of these cases makes this development evidence; independent final cases are required to assess reliability. Missing/nonfinite history values are counted in the summary and cannot appear on a logarithmic plot.", "",
              "## Training data and continuation", "",
              "| Run | Usable training / candidates | Usable IV validation | Collocation pool | Sensitivity weight | Resumed weights |", "|---|---:|---:|---:|---:|---|"]
    for row in summaries:
        resume = row["resumed_from"] or "No"
        lines.append(f"| {row['run']} | {row['training_usable']} / {row['training_candidates']} | {row['validation_usable']} | {row['collocation_pool']} | {show(row['sensitivity_weight'])} | {resume} |")
    lines += ["", "A resumed run continues existing network weights and is not an independent initialization. AdamW runs use their declared weight decay; L-BFGS runs use a fixed deterministic objective without weight decay, clipping or resampling. A plotted L-BFGS iteration is not computationally equivalent to an Adam step. PDE/shape penalties and optional parameter-sensitivity supervision are recorded in each effective configuration.", "",
              "L-BFGS may use a fixed subset of the available training labels; fixed_training_subset in run_summary.csv gives the actual anchor count. The table's training count describes the source dataset, not necessarily all labels used in that fine-tuning phase.", "",
              "Grouped-surface runs additionally use 1,024 independent synthetic training parameter surfaces (126 quotes each, 129,024 extra quotes). The control and parameter-aware arm use identical extra data. Their parameter-aware weights are recorded in run_summary.csv. These labels are not supplied to the inverse calibrator. Double Heston factor 1 is stored as the slow factor and factor 2 as the fast factor throughout this experiment.", "",
              "## Every selected validation parameter", "",
              "The full machine-readable values are in [selected_parameter_details.csv](selected_parameter_details.csv). Gate units are absolute error divided by the allowed tolerance: values at or below 1 pass. The signed scaled-error column uses the training selection scale described above.", ""]
    for run in runs:
        label = run["summary"]["run"]
        lines += [f"<details><summary>{label}: all selected parameter estimates</summary>", "",
                  "| Case | Factor | Parameter | Truth | Estimate | Signed scaled error | Gate units | Pass |", "|---:|---:|---|---:|---:|---:|---:|---|"]
        for row in run["selected_parameters"]:
            lines.append(f"| {row['case']} | {row['factor']} | {row['parameter']} | {show(row['truth'],7)} | {show(row['estimate'],7)} | {show(row['training_scaled_error'])} | {show(row['absolute_error_in_gate_units'])} | {'Yes' if row['parameter_pass'] else 'No'} |")
        if not run["selected_parameters"]:
            lines.append("| — | — | Recovery records unavailable | — | — | — | — | — |")
        lines += ["", "</details>", ""]
    warnings = [(run["summary"]["run"], warning) for run in runs for warning in run["summary"]["warnings"]]
    if warnings:
        lines += ["## Incomplete or inconsistent evidence", ""]
        lines += [f"- {label}: {warning}" for label, warning in warnings]
        lines.append("")
    lines += ["## Reproducibility", "",
              "[evidence.json](evidence.json) contains the input snapshots, selected-checkpoint hashes, training manifests and exact report inputs. [run_summary.csv](run_summary.csv) and [selected_case_gates.csv](selected_case_gates.csv) retain run and case-level outcomes, including missing or failed recovery records.", ""]
    (output / "REPORT.md").write_text("\n".join(lines))
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    labels = [path.resolve().name for path in args.run]
    if len(labels) != len(set(labels)):
        raise ValueError("Run directory names must be unique")
    for path in args.run:
        if not path.is_dir():
            raise FileNotFoundError(path)
    args.out.mkdir(parents=True, exist_ok=False)
    summaries = build_report([collect_run(path) for path in args.run], args.out)
    print(json.dumps({"report": str(args.out / "REPORT.md"), "label": LABEL,
                      "runs": [{"run": row["run"], "status": row["status"], "selected_step": row["selected_step"]}
                               for row in summaries]}, indent=2))


if __name__ == "__main__":
    main()
