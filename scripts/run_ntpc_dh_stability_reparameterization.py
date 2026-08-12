"""Run the predeclared NTPC Double Heston optimization-geometry experiment.

This consumes the reviewed pilot's frozen local evidence.  It does not acquire
data and never rewrites the pilot outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import least_squares
from scipy.spatial.distance import pdist, squareform

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import run_ntpc_single_stock_pilot as pilot
from src.calibrate_double_heston import (
    boundary_diagnostics,
    load_hard_safety_bounds,
    unconstrained_to_parameters,
)
from src.constants import PARAMETER_NAMES
from src.ntpc_dh_reparameterization import (
    TRANSFORMED_NAMES,
    canonical_diagnostics,
    canonical_to_structured,
    derived_coordinates,
    structured_to_canonical,
)


BASE_COMMIT = "dd539150898bf5ca4d168c5dba3f3a33c69628e2"
OUTPUT_ROOT = (
    REPOSITORY_ROOT
    / "market_data_audit"
    / "stage_a"
    / "derived"
    / "ntpc_dh_stability_reparameterization"
)
FIGURE_ROOT = OUTPUT_ROOT / "figures"
REPORT_PATH = REPOSITORY_ROOT / "docs" / "NTPC_DH_STABILITY_REPARAMETERIZATION.md"
BASELINE_ROOT = pilot.OUTPUT_ROOT
BASELINE_MANIFEST = pilot.MANIFEST_PATH
BOUNDS_PATH = pilot.BOUNDS_PATH

START_COUNT = pilot.DOUBLE_HESTON_STARTS
NODE_COUNT = pilot.NODE_COUNT
MAX_NFEV = pilot.MAX_NFEV
ANALYSIS_SEED = pilot.ANALYSIS_SEED
MATERIAL_DISTANCE = 0.05
CLUSTER_DISTANCE = 0.05
STRONG_DISPERSION_REDUCTION = 0.25
PARTIAL_DISPERSION_REDUCTION = 0.10

PROTECTED_ARTIFACTS = (
    "selected_options.csv",
    "carry_contract.csv",
    "model_comparison.csv",
    "double_heston_multistart.csv",
    "parameter_stability.json",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def protected_hashes() -> dict[str, str]:
    paths = {name: BASELINE_ROOT / name for name in PROTECTED_ARTIFACTS}
    paths[BASELINE_MANIFEST.relative_to(REPOSITORY_ROOT).as_posix()] = BASELINE_MANIFEST
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing reviewed baseline evidence: {missing}")
    return {name: sha256(path) for name, path in paths.items()}


def verify_baseline_contract() -> dict[str, Any]:
    manifest = json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))
    hashes = protected_hashes()
    expected = manifest["artifact_hashes"]
    for name in PROTECTED_ARTIFACTS:
        if hashes[name] != expected[name]:
            raise RuntimeError(f"protected NTPC artifact hash changed: {name}")
    if manifest["tracked_artifact_hashes"]["scripts/run_ntpc_single_stock_pilot.py"] != sha256(
        REPOSITORY_ROOT / "scripts" / "run_ntpc_single_stock_pilot.py"
    ):
        raise RuntimeError("reviewed NTPC pilot implementation hash changed")
    exact = {
        "valuation_date": "2026-07-15",
        "spot": 344.35,
        "calibration_rows": 12,
        "holdout_rows": 7,
        "primary_price_field": "ClsPric",
        "risk_free_simple_yield": 0.053324,
        "node_count": 64,
        "max_nfev": 160,
    }
    observed = {
        "valuation_date": manifest["valuation_date"],
        "spot": manifest["spot"],
        "calibration_rows": manifest["selection"]["calibration_rows"],
        "holdout_rows": manifest["selection"]["holdout_rows"],
        "primary_price_field": manifest["selection"]["primary_price_field"],
        "risk_free_simple_yield": manifest["carry_contract"]["risk_free_simple_yield"],
        "node_count": manifest["optimizer"]["node_count"],
        "max_nfev": manifest["optimizer"]["max_nfev"],
    }
    if observed != exact:
        raise RuntimeError(f"frozen NTPC contract mismatch: {observed}")
    return {"protected_hashes": hashes, "contract": observed}


def paired_start_population(
    hard_bounds: dict[str, tuple[float, float]],
) -> tuple[list[np.ndarray], list[np.ndarray], pd.DataFrame]:
    rng = np.random.default_rng(ANALYSIS_SEED + 200)
    baseline_coordinates = [np.zeros(10)] + [
        rng.normal(0.0, 1.25, 10) for _ in range(START_COUNT - 1)
    ]
    canonical_starts = [unconstrained_to_parameters(value, hard_bounds) for value in baseline_coordinates]
    structured_starts = [canonical_to_structured(value, hard_bounds) for value in canonical_starts]
    rows = []
    for start_id, (baseline_z, canonical, structured_z) in enumerate(
        zip(baseline_coordinates, canonical_starts, structured_starts, strict=True)
    ):
        recovered = structured_to_canonical(structured_z, hard_bounds)
        max_error = float(np.max(np.abs(recovered - canonical)))
        row: dict[str, Any] = {
            "start_id": start_id,
            "canonical_start_sha256": hashlib.sha256(canonical.tobytes()).hexdigest().upper(),
            "paired_max_abs_error": max_error,
        }
        row.update({f"baseline_z_{index}": float(value) for index, value in enumerate(baseline_z)})
        row.update({f"start_{name}": float(value) for name, value in zip(PARAMETER_NAMES, canonical, strict=True)})
        row.update({name: float(value) for name, value in zip(TRANSFORMED_NAMES, structured_z, strict=True)})
        rows.append(row)
    frame = pd.DataFrame(rows)
    if len(frame) != START_COUNT or frame["start_id"].tolist() != list(range(START_COUNT)):
        raise RuntimeError("paired start population mismatch")
    if float(frame["paired_max_abs_error"].max()) > 2e-12:
        raise RuntimeError("paired start round trip exceeds tolerance")
    return canonical_starts, structured_starts, frame


def equivalence_audit(
    hard_bounds: dict[str, tuple[float, float]],
    best_parameters: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame]:
    rng = np.random.default_rng(2026081201)
    cases: list[tuple[str, np.ndarray]] = [("existing_ntpc_best_fit", best_parameters)]
    for _ in range(2000):
        canonical = unconstrained_to_parameters(rng.normal(0.0, 2.0, 10), hard_bounds)
        while True:
            rho_slow = rng.uniform(*hard_bounds["rho_slow"])
            rho_fast = rng.uniform(*hard_bounds["rho_fast"])
            if rho_slow**2 + rho_fast**2 < 1.0:
                canonical[3], canonical[8] = rho_slow, rho_fast
                break
        cases.append(("random_interior", canonical))
    boundary_coordinates = []
    for index in range(10):
        for sign in (-1.0, 1.0):
            coordinate = np.zeros(10)
            coordinate[index] = sign * 10.0
            boundary_coordinates.append(coordinate)
    cases.extend(
        ("near_boundary", unconstrained_to_parameters(value, hard_bounds))
        for value in boundary_coordinates
    )
    for rho_slow, rho_fast in (
        (0.70, 0.68),
        (0.70, -0.68),
        (-0.70, 0.68),
        (-0.70, -0.68),
        (0.94, 0.30),
        (0.94, -0.30),
        (-0.94, 0.30),
        (-0.94, -0.30),
    ):
        canonical = unconstrained_to_parameters(np.zeros(10), hard_bounds)
        canonical[3], canonical[8] = rho_slow, rho_fast
        cases.append(("correlation_annulus", canonical))

    rows = []
    failures = 0
    for case_id, (case_type, canonical) in enumerate(cases):
        try:
            transformed = canonical_to_structured(canonical, hard_bounds)
            recovered = structured_to_canonical(transformed, hard_bounds)
            diagnostics = canonical_diagnostics(recovered, hard_bounds)
            error = float(np.max(np.abs(recovered - canonical)))
            valid = bool(diagnostics["is_valid"] and error <= 2e-12)
        except Exception as exception:
            transformed = np.full(10, np.nan)
            error = np.nan
            valid = False
            diagnostics = {"violations": [f"{type(exception).__name__}: {exception}"]}
        failures += int(not valid)
        rows.append(
            {
                "case_id": case_id,
                "case_type": case_type,
                "round_trip_valid": valid,
                "max_abs_error": error,
                "violations": ";".join(diagnostics["violations"]),
                **{name: float(value) for name, value in zip(PARAMETER_NAMES, canonical, strict=True)},
                **{name: float(value) for name, value in zip(TRANSFORMED_NAMES, transformed, strict=True)},
            }
        )
    frame = pd.DataFrame(rows)
    summary = {
        "analytical_equivalence": (
            "Exact bijection on the same numerical interior: rectangle interiors for v0/theta are "
            "expressed by total plus conditional allocation; kappa uses the identical conditional "
            "ordering interval; sigma uses the identical Feller-safe conditional interval; correlations "
            "use a one-to-one conditional parameterization of the full intersection between the "
            "individual hard bounds and the unit disk."
        ),
        "sample_count": len(frame),
        "existing_best_fit_count": int((frame["case_type"] == "existing_ntpc_best_fit").sum()),
        "random_interior_count": int((frame["case_type"] == "random_interior").sum()),
        "near_boundary_count": int((frame["case_type"] == "near_boundary").sum()),
        "correlation_annulus_count": int((frame["case_type"] == "correlation_annulus").sum()),
        "round_trip_failure_count": failures,
        "empirical_lost_fraction": float(failures / len(frame)),
        "maximum_abs_round_trip_error": float(frame["max_abs_error"].max()),
        "search_space_changed": bool(failures),
    }
    return summary, frame


def price_rows(frame: pd.DataFrame, parameters: Sequence[float]) -> np.ndarray:
    return pilot._price_rows(frame, lambda row: pilot._double_heston_row_price(row, parameters))


def boundary_margins(
    parameters: np.ndarray,
    hard_bounds: dict[str, tuple[float, float]],
) -> dict[str, float]:
    normalized = []
    for name, value in zip(PARAMETER_NAMES, parameters, strict=True):
        lower, upper = hard_bounds[name]
        normalized.append(min((value - lower) / (upper - lower), (upper - value) / (upper - lower)))
    return {
        "minimum_hard_bound_fraction_margin": float(min(normalized)),
        "kappa_ordering_margin": float(parameters[5] - parameters[0]),
        "slow_feller_margin": float(2 * parameters[0] * parameters[1] - parameters[2] ** 2),
        "fast_feller_margin": float(2 * parameters[5] * parameters[6] - parameters[7] ** 2),
        "correlation_disk_margin": float(1.0 - parameters[3] ** 2 - parameters[8] ** 2),
    }


def run_transformed_calibrations(
    calibration: pd.DataFrame,
    holdout: pd.DataFrame,
    structured_starts: list[np.ndarray],
    hard_bounds: dict[str, tuple[float, float]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    observed = calibration["observed_price"].to_numpy(float)
    rows = []

    def residual(z: np.ndarray) -> np.ndarray:
        return price_rows(calibration, structured_to_canonical(z, hard_bounds)) - observed

    for start_id, start in enumerate(structured_starts):
        started = time.perf_counter()
        row: dict[str, Any] = {"model": "DOUBLE_HESTON_REPARAMETERIZED", "start_id": start_id}
        try:
            result = least_squares(
                residual,
                start,
                method="trf",
                max_nfev=MAX_NFEV,
                ftol=1e-9,
                xtol=1e-9,
                gtol=1e-9,
                diff_step=2e-5,
            )
            parameters = structured_to_canonical(result.x, hard_bounds)
            cal_prices = price_rows(calibration, parameters)
            hold_prices = price_rows(holdout, parameters)
            reasons = boundary_diagnostics(parameters, hard_bounds)
            row.update(
                {
                    "optimizer_success": bool(result.success),
                    "optimizer_status": int(result.status),
                    "optimizer_message": str(result.message),
                    "nfev": int(result.nfev),
                    "valid": True,
                    "boundary_reasons": ";".join(reasons),
                    "boundary_hit": bool(reasons),
                    **{f"calibration_{key}": value for key, value in pilot._metrics(calibration, cal_prices).items()},
                    **{f"holdout_{key}": value for key, value in pilot._metrics(holdout, hold_prices).items()},
                    **{name: float(value) for name, value in zip(PARAMETER_NAMES, parameters, strict=True)},
                    **{name: float(value) for name, value in zip(TRANSFORMED_NAMES, result.x, strict=True)},
                    **derived_coordinates(parameters),
                    **boundary_margins(parameters, hard_bounds),
                }
            )
        except Exception as exception:
            row.update(
                {
                    "optimizer_success": False,
                    "optimizer_status": -1,
                    "optimizer_message": f"{type(exception).__name__}: {exception}",
                    "nfev": 0,
                    "valid": False,
                    "boundary_reasons": "",
                    "boundary_hit": False,
                }
            )
        row["runtime_seconds"] = time.perf_counter() - started
        rows.append(row)
    frame = pd.DataFrame(rows)
    valid = frame.loc[frame["valid"]].copy()
    if valid.empty:
        raise RuntimeError("all transformed starts failed")
    best = valid.sort_values(["calibration_price_rmse", "start_id"]).iloc[0]
    summary = {
        "best_start_id": int(best["start_id"]),
        "runtime_seconds": float(frame["runtime_seconds"].sum()),
        **{name: float(best[name]) for name in PARAMETER_NAMES},
        **{key: float(best[key]) for key in best.index if key.startswith("calibration_") or key.startswith("holdout_")},
    }
    return frame, summary


def stability_metrics(
    starts: pd.DataFrame,
    hard_bounds: dict[str, tuple[float, float]],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    valid = starts.loc[starts["valid"].astype(bool)].copy()
    best = valid.sort_values(["calibration_price_rmse", "start_id"]).iloc[0]
    threshold = max(float(best["calibration_price_rmse"]) * 1.05, float(best["calibration_price_rmse"]) + 0.01)
    near = valid.loc[valid["calibration_price_rmse"] <= threshold].copy()
    widths = np.asarray([hard_bounds[name][1] - hard_bounds[name][0] for name in PARAMETER_NAMES])
    scaled = near[list(PARAMETER_NAMES)].to_numpy(float) / widths / math.sqrt(len(PARAMETER_NAMES))
    best_scaled = best[list(PARAMETER_NAMES)].to_numpy(float) / widths / math.sqrt(len(PARAMETER_NAMES))
    from_best = np.linalg.norm(scaled - best_scaled, axis=1)
    pairwise = pdist(scaled, metric="euclidean") if len(near) > 1 else np.asarray([], dtype=float)
    if len(near) > 1:
        labels = fcluster(linkage(scaled, method="complete", metric="euclidean"), CLUSTER_DISTANCE, criterion="distance")
    else:
        labels = np.ones(len(near), dtype=int)
    near["distance_from_best"] = from_best
    near["cluster_id"] = labels
    pair_rows = []
    matrix = squareform(pairwise) if len(near) > 1 else np.zeros((len(near), len(near)))
    for left in range(len(near)):
        for right in range(left + 1, len(near)):
            pair_rows.append(
                {
                    "left_start_id": int(near.iloc[left]["start_id"]),
                    "right_start_id": int(near.iloc[right]["start_id"]),
                    "range_scaled_distance": float(matrix[left, right]),
                }
            )
    parameter_stats = {}
    for name in PARAMETER_NAMES:
        values = near[name].to_numpy(float)
        mean = float(np.mean(values))
        parameter_stats[name] = {
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "range": float(np.ptp(values)),
            "coefficient_of_variation": float(np.std(values, ddof=0) / abs(mean)) if abs(mean) > 1e-12 else None,
        }
    derived_frame = pd.DataFrame(
        [derived_coordinates(row) for row in near[list(PARAMETER_NAMES)].to_numpy(float)]
    )
    derived_stats = {}
    for name in derived_frame.columns:
        values = derived_frame[name].to_numpy(float)
        mean = float(np.mean(values))
        derived_stats[name] = {
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "range": float(np.ptp(values)),
            "coefficient_of_variation": float(np.std(values, ddof=0) / abs(mean)) if abs(mean) > 1e-12 else None,
        }
    cap_rate = float(np.mean(valid["nfev"].astype(int) >= MAX_NFEV))
    boundary_rate = float(np.mean(valid["boundary_reasons"].fillna("").astype(str).str.len() > 0))
    metrics = {
        "near_equivalent_threshold_price_rmse": threshold,
        "valid_start_count": int(len(valid)),
        "near_equivalent_start_count": int(len(near)),
        "materially_displaced_start_count": int(np.sum(from_best >= MATERIAL_DISTANCE)),
        "cluster_count": int(len(set(labels))),
        "median_pairwise_range_scaled_distance": float(np.median(pairwise)) if len(pairwise) else 0.0,
        "maximum_pairwise_range_scaled_distance": float(np.max(pairwise)) if len(pairwise) else 0.0,
        "maximum_range_scaled_distance_from_best": float(np.max(from_best)) if len(from_best) else 0.0,
        "boundary_hit_rate": boundary_rate,
        "optimizer_cap_rate": cap_rate,
        "parameter_statistics": parameter_stats,
        "derived_coordinate_statistics": derived_stats,
    }
    return metrics, near, pd.DataFrame(pair_rows)


def classify_experiment(
    baseline_summary: dict[str, float],
    transformed_summary: dict[str, float],
    baseline_stability: dict[str, Any],
    transformed_stability: dict[str, Any],
    *,
    equivalence_passed: bool,
    contract_passed: bool,
    matched_population: bool,
) -> dict[str, Any]:
    price_ratios = {
        name: transformed_summary[name] / baseline_summary[name]
        for name in ("calibration_price_rmse", "holdout_price_rmse", "holdout_iv_rmse")
    }
    pricing_preserved = all(value <= 1.05 for value in price_ratios.values())
    median_reduction = 1.0 - transformed_stability["median_pairwise_range_scaled_distance"] / max(
        baseline_stability["median_pairwise_range_scaled_distance"], 1e-15
    )
    maximum_reduction = 1.0 - transformed_stability["maximum_pairwise_range_scaled_distance"] / max(
        baseline_stability["maximum_pairwise_range_scaled_distance"], 1e-15
    )
    fewer_clusters = transformed_stability["cluster_count"] < baseline_stability["cluster_count"]
    no_extra_clusters = transformed_stability["cluster_count"] <= baseline_stability["cluster_count"]
    invalid_reasons = []
    if not equivalence_passed:
        invalid_reasons.append("SEARCH_SPACE_CHANGED_OR_ROUND_TRIP_FAILED")
    if not contract_passed:
        invalid_reasons.append("DATA_OBJECTIVE_OR_PRICING_CONTRACT_CHANGED")
    if not matched_population:
        invalid_reasons.append("COMPARISON_POPULATION_MISMATCHED")
    if invalid_reasons:
        classification = "INVALID"
    elif not pricing_preserved:
        classification = "INSUFFICIENT"
    elif (
        transformed_stability["materially_displaced_start_count"] <= 3
        and median_reduction >= STRONG_DISPERSION_REDUCTION
        and maximum_reduction >= STRONG_DISPERSION_REDUCTION
        and fewer_clusters
    ):
        classification = "STRONG_STABILITY_IMPROVEMENT"
    elif (
        4 <= transformed_stability["materially_displaced_start_count"] <= 6
        and median_reduction >= PARTIAL_DISPERSION_REDUCTION
        and maximum_reduction >= PARTIAL_DISPERSION_REDUCTION
        and no_extra_clusters
    ):
        classification = "PARTIAL_STABILITY_IMPROVEMENT"
    else:
        classification = "INSUFFICIENT"
    heston_reference = 0.910569
    return {
        "classification": classification,
        "invalid_reasons": invalid_reasons,
        "pricing_preserved": pricing_preserved,
        "pricing_ratios": price_ratios,
        "median_pairwise_dispersion_reduction": median_reduction,
        "maximum_pairwise_dispersion_reduction": maximum_reduction,
        "materially_beats_heston_holdout_by_5_percent": bool(
            transformed_summary["holdout_price_rmse"] <= 0.95 * heston_reference
        ),
        "heston_holdout_reference": heston_reference,
        "heston_five_percent_threshold": 0.95 * heston_reference,
    }


def _plot_outputs(
    baseline: pd.DataFrame,
    transformed: pd.DataFrame,
    baseline_stability: dict[str, Any],
    transformed_stability: dict[str, Any],
    baseline_near: pd.DataFrame,
    transformed_near: pd.DataFrame,
    baseline_pairs: pd.DataFrame,
    transformed_pairs: pd.DataFrame,
) -> list[Path]:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    figures: list[Path] = []

    best_base = baseline.sort_values(["calibration_price_rmse", "start_id"]).iloc[0]
    best_new = transformed.loc[transformed["valid"]].sort_values(["calibration_price_rmse", "start_id"]).iloc[0]
    labels = ["Calibration price", "Holdout price", "Calibration IV", "Holdout IV"]
    keys = ["calibration_price_rmse", "holdout_price_rmse", "calibration_iv_rmse", "holdout_iv_rmse"]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.18, [best_base[key] for key in keys], 0.36, label="Baseline")
    ax.bar(x + 0.18, [best_new[key] for key in keys], 0.36, label="Reparameterized")
    ax.set_xticks(x, labels, rotation=12)
    ax.set_ylabel("RMSE")
    ax.set_title("Best-fit pricing quality (unchanged observations and pricer)")
    ax.legend()
    figures.append(FIGURE_ROOT / "01_baseline_vs_transformed_rmse.png")
    fig.tight_layout(); fig.savefig(figures[-1], dpi=170); plt.close(fig)

    widths = np.asarray([load_hard_safety_bounds(BOUNDS_PATH)[name][1] - load_hard_safety_bounds(BOUNDS_PATH)[name][0] for name in PARAMETER_NAMES])
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, frame, title in zip(axes, [baseline_near, transformed_near], ["Baseline", "Reparameterized"], strict=True):
        normalized = frame[list(PARAMETER_NAMES)].to_numpy(float) / widths
        ax.boxplot([normalized[:, index] for index in range(10)], tick_labels=PARAMETER_NAMES, showfliers=True)
        ax.set_ylabel("parameter / hard-range width")
        ax.set_title(f"{title} near-equivalent canonical parameter dispersion")
    axes[-1].tick_params(axis="x", rotation=35)
    figures.append(FIGURE_ROOT / "02_multistart_parameter_dispersion.png")
    fig.tight_layout(); fig.savefig(figures[-1], dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(baseline_pairs["range_scaled_distance"], bins=12, alpha=0.6, label=f"Baseline ({baseline_stability['cluster_count']} clusters)")
    ax.hist(transformed_pairs["range_scaled_distance"], bins=12, alpha=0.6, label=f"Reparameterized ({transformed_stability['cluster_count']} clusters)")
    ax.axvline(CLUSTER_DISTANCE, color="black", linestyle="--", label="0.05 separation threshold")
    ax.set_xlabel("Pairwise full-range-scaled distance")
    ax.set_ylabel("Pair count")
    ax.set_title("Near-equivalent solution separation")
    ax.legend()
    figures.append(FIGURE_ROOT / "03_pairwise_cluster_separation.png")
    fig.tight_layout(); fig.savefig(figures[-1], dpi=170); plt.close(fig)

    for number, columns, title, filename in (
        (4, ["v0_total", "alpha_v"], "Initial variance total and allocation", "04_v0_total_and_allocation.png"),
        (5, ["theta_total", "alpha_theta"], "Long-run variance total and allocation", "05_theta_total_and_allocation.png"),
        (6, ["kappa_slow", "kappa_fast", "slow_half_life_days", "fast_half_life_days"], "Mean reversion and half-life", "06_kappa_and_half_life.png"),
    ):
        fig, axes = plt.subplots(1, len(columns), figsize=(5 * len(columns), 4.5))
        axes = np.atleast_1d(axes)
        for ax, column in zip(axes, columns, strict=True):
            ax.plot(transformed_near["start_id"], transformed_near[column], marker="o", linestyle="none")
            ax.set_xlabel("paired start ID")
            ax.set_ylabel(column)
        suffix = " (alphas are optimization coordinates, not scientific parameters)" if any("alpha" in column for column in columns) else ""
        fig.suptitle(title + suffix)
        figures.append(FIGURE_ROOT / filename)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92)); fig.savefig(figures[-1], dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    normalized = transformed_near[list(PARAMETER_NAMES)].to_numpy(float) / widths
    for row, start_id in zip(normalized, transformed_near["start_id"], strict=True):
        ax.plot(PARAMETER_NAMES, row, marker="o", alpha=0.7, label=f"start {int(start_id)}")
    ax.set_ylabel("canonical value / hard-range width")
    ax.set_title("Canonical true-output vectors from reparameterized near-equivalent solutions")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(ncol=3, fontsize=8)
    figures.append(FIGURE_ROOT / "07_canonical_near_equivalent_vectors.png")
    fig.tight_layout(); fig.savefig(figures[-1], dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(baseline_near["distance_from_best"], baseline_near["calibration_price_rmse"], label="Baseline")
    ax.scatter(transformed_near["distance_from_best"], transformed_near["calibration_price_rmse"], label="Reparameterized")
    ax.axvline(MATERIAL_DISTANCE, color="black", linestyle="--")
    ax.set_xlabel("Canonical displacement from best (full-range scaled)")
    ax.set_ylabel("Calibration price RMSE")
    ax.set_title("Pricing error versus parameter displacement")
    ax.legend()
    figures.append(FIGURE_ROOT / "08_pricing_error_vs_displacement.png")
    fig.tight_layout(); fig.savefig(figures[-1], dpi=170); plt.close(fig)
    return figures


def _summary_table(baseline: dict[str, Any], transformed: dict[str, Any]) -> str:
    keys = ("calibration_price_rmse", "calibration_iv_rmse", "holdout_price_rmse", "holdout_iv_rmse", "runtime_seconds")
    lines = ["| metric | baseline | reparameterized |", "|---|---:|---:|"]
    for key in keys:
        lines.append(f"| {key} | {baseline[key]:.9g} | {transformed[key]:.9g} |")
    return "\n".join(lines)


def render_report(
    contract: dict[str, Any],
    equivalence: dict[str, Any],
    baseline_summary: dict[str, Any],
    transformed_summary: dict[str, Any],
    baseline_stability: dict[str, Any],
    transformed_stability: dict[str, Any],
    classification: dict[str, Any],
    transformed_near: pd.DataFrame,
    figures: list[Path],
) -> None:
    unstable = [
        name for name, stats in transformed_stability["parameter_statistics"].items()
        if stats["range"] / (load_hard_safety_bounds(BOUNDS_PATH)[name][1] - load_hard_safety_bounds(BOUNDS_PATH)[name][0]) >= 0.05
    ]
    best = transformed_near.sort_values(["calibration_price_rmse", "start_id"]).iloc[0]
    derived_lines = ["| diagnostic | baseline range | reparameterized range | baseline CV | reparameterized CV |", "|---|---:|---:|---:|---:|"]
    for name in ("v0_total", "alpha_v", "theta_total", "alpha_theta", "delta_kappa", "slow_half_life_days", "fast_half_life_days"):
        base_stat = baseline_stability["derived_coordinate_statistics"][name]
        new_stat = transformed_stability["derived_coordinate_statistics"][name]
        derived_lines.append(
            f"| {name} | {base_stat['range']:.9g} | {new_stat['range']:.9g} | {base_stat['coefficient_of_variation']:.9g} | {new_stat['coefficient_of_variation']:.9g} |"
        )
    derived_table = "\n".join(derived_lines)
    median_change = classification["median_pairwise_dispersion_reduction"]
    maximum_change = classification["maximum_pairwise_dispersion_reduction"]
    median_phrase = f"{median_change:.3%} reduction" if median_change >= 0.0 else f"{-median_change:.3%} increase"
    maximum_phrase = f"{maximum_change:.3%} reduction" if maximum_change >= 0.0 else f"{-maximum_change:.3%} increase"
    report = f"""# NTPC Double Heston Stability Reparameterization

## Predeclared research question and decision rules

Does a structure-aware, one-to-one transformed coordinate system for the **same canonical ten-parameter Double Heston space** reduce NTPC multi-start calibration instability while preserving calibration and holdout pricing quality? This tests optimization geometry. It does not establish structural identification unless globally separated solutions disappear over equivalent attainable space.

Base commit: `{BASE_COMMIT}`. Starts: `{START_COUNT}` paired canonical starts. Pricer: unchanged canonical Double Heston at `{NODE_COUNT}` production nodes. Optimizer: unchanged SciPy `least_squares(method="trf")`, `max_nfev={MAX_NFEV}`, tolerances and `diff_step`. Primary loss: unchanged unweighted calibration price residual vector.

Near-equivalent and material displacement rules are unchanged from the reviewed pilot. Complete-linkage clusters use the predeclared full-range-scaled distance `{CLUSTER_DISTANCE}`. Strong dispersion reduction requires at least 25% reductions in both median and maximum pairwise separation plus fewer clusters; partial requires at least 10% reductions with no extra clusters. These rules were fixed before calibration.

## What changed and what remained identical

Only optimizer coordinates changed. The canonical scientific target/order, pricing model, NTPC observations and frozen row roles, price field, activity screen, valuation date, spot, maturity basis, RBI/futures carry contract, IV inversion, 64-node production pricer, primary loss, start population, optimizer settings, and thresholds remained identical. Protected baseline hashes were verified before and after the run: `{contract['protected_hashes']}`.

The exact transformed order is:

1. `z_v0_total`: bounded logistic over `(v0_slow.lower + v0_fast.lower, v0_slow.upper + v0_fast.upper)`.
2. `z_alpha_v`: bounded logistic selecting `v0_slow` from the total-conditional interval; `v0_fast=v0_total-v0_slow`, and reported `alpha_v=v0_slow/v0_total`.
3. `z_theta_total`: analogous total long-run variance.
4. `z_alpha_theta`: analogous conditional allocation, reported as `theta_slow/theta_total`.
5. `z_kappa_slow`: bounded logistic over the unchanged slow-kappa range.
6. `z_delta_kappa`: bounded logistic for `kappa_fast` over `(max(kappa_fast.lower,kappa_slow+1e-5), kappa_fast.upper)`.
7-8. `z_sigma_slow`, `z_sigma_fast`: bounded logistics over the unchanged lower bound and `min(configured_upper, sqrt(2*kappa*theta)*(1-1e-7))`, preserving Feller validity without clipping.
9. `z_rho_slow`: bounded logistic over the unchanged configured interval `(-0.95,0.95)`.
10. `z_rho_fast`: bounded logistic over `max(-0.95,-sqrt(1-rho_slow^2))` to `min(0.95,+sqrt(1-rho_slow^2))`, a one-to-one representation of the full unchanged hard-bound/unit-disk intersection.

The final output is always the original canonical ten-vector. Alpha values are optimization coordinates/derived diagnostics, **not new Double Heston scientific parameters**.

## Search-space equivalence

`EXPERIMENT_VALIDITY = {'PASSED_EQUIVALENT_SEARCH_SPACE' if not equivalence['search_space_changed'] else 'NOT_PASSED_SEARCH_SPACE_CHANGED'}`.

{equivalence['analytical_equivalence']} The audit covered `{equivalence['sample_count']}` vectors: the existing best fit, `{equivalence['random_interior_count']}` deterministic random interiors, `{equivalence['near_boundary_count']}` near-boundary cases, and `{equivalence['correlation_annulus_count']}` explicit valid correlation-annulus cases with radius at least 0.95. Failures: `{equivalence['round_trip_failure_count']}`; empirical lost fraction: `{equivalence['empirical_lost_fraction']:.3g}`; maximum absolute round-trip error: `{equivalence['maximum_abs_round_trip_error']:.3g}`. No silent clipping is used.

## Pricing and runtime

{_summary_table(baseline_summary, transformed_summary)}

Pricing preservation: **{classification['pricing_preserved']}**. The reparameterized holdout price RMSE {'does' if classification['materially_beats_heston_holdout_by_5_percent'] else 'does not'} beat the Standard Heston reference `{classification['heston_holdout_reference']}` by at least 5% (required `<= {classification['heston_five_percent_threshold']:.9g}`).

## Multi-start stability

| metric | baseline | reparameterized |
|---|---:|---:|
| valid starts | {baseline_stability['valid_start_count']} | {transformed_stability['valid_start_count']} |
| near-equivalent starts | {baseline_stability['near_equivalent_start_count']} | {transformed_stability['near_equivalent_start_count']} |
| materially displaced | {baseline_stability['materially_displaced_start_count']} | {transformed_stability['materially_displaced_start_count']} |
| complete-linkage clusters | {baseline_stability['cluster_count']} | {transformed_stability['cluster_count']} |
| median pairwise distance | {baseline_stability['median_pairwise_range_scaled_distance']:.9g} | {transformed_stability['median_pairwise_range_scaled_distance']:.9g} |
| maximum pairwise distance | {baseline_stability['maximum_pairwise_range_scaled_distance']:.9g} | {transformed_stability['maximum_pairwise_range_scaled_distance']:.9g} |
| maximum distance from best | {baseline_stability['maximum_range_scaled_distance_from_best']:.9g} | {transformed_stability['maximum_range_scaled_distance_from_best']:.9g} |
| boundary-hit rate | {baseline_stability['boundary_hit_rate']:.3g} | {transformed_stability['boundary_hit_rate']:.3g} |
| optimizer-cap rate | {baseline_stability['optimizer_cap_rate']:.3g} | {transformed_stability['optimizer_cap_rate']:.3g} |

Median pairwise dispersion: **{median_phrase}**. Maximum pairwise dispersion: **{maximum_phrase}**. Parameters retaining at least 0.05 of their configured full-range width across near-equivalent solutions: `{unstable}`. Full parameter-wise ranges and coefficients of variation are in `stability_comparison.json`.

## Allocation and mean-reversion diagnostics

Best transformed solution: `v0_total={best['v0_total']:.9g}`, `alpha_v={best['alpha_v']:.9g}`, `theta_total={best['theta_total']:.9g}`, `alpha_theta={best['alpha_theta']:.9g}`, `kappa_slow={best['kappa_slow']:.9g}`, `kappa_fast={best['kappa_fast']:.9g}`, slow half-life `{best['slow_half_life_days']:.9g}` days, fast half-life `{best['fast_half_life_days']:.9g}` days. Cross-start total/allocation and half-life values are recorded in `reparameterized_near_equivalent.csv` and Figures 4-6.

{derived_table}

Neither the variance totals nor slow/fast allocations may be called stable unless their ranges and coefficients of variation materially contract. The table shows the direct baseline-versus-reparameterized comparison; alpha values remain coordinate diagnostics only.

## Interpretation and decision

Classification: **{classification['classification']}**. Invalid reasons: `{classification['invalid_reasons']}`.

Total-plus-allocation coordinates were tested because short-maturity prices may constrain aggregate variance more directly than the factor split. The experiment changes the coordinate chart, not the attainable scientific model. Any remaining separated canonical solutions are therefore still consistent with structural non-identification; optimizer-coordinate improvement alone is not identification evidence.

This result does **not** alter the canonical ten-parameter project target. It does not by itself justify proceeding to regularization: regularization requires a separate predeclared experiment and must not be inferred from local geometry alone.

## Exact recommended next experiment

Run a separately predeclared **optimizer-cap sensitivity diagnostic**, not regularization: replay these same 12 paired canonical starts under both baseline and structure-aware charts at `max_nfev=160` and `320`, with the same frozen NTPC rows, objective, 64-node pricer, tolerances, and all existing stability thresholds. Predeclare that persistence of separated near-equivalent clusters after the cap rate materially falls supports continuing structural ambiguity, while a collapse under both charts indicates the 160-evaluation cap was materially numerical. Do not select a budget after seeing results and do not add data, priors, regularization, ANN, or PINN.

## Figures

""" + "\n".join(f"- `{path.relative_to(REPOSITORY_ROOT).as_posix()}`" for path in figures) + "\n"
    REPORT_PATH.write_text(report, encoding="utf-8", newline="\n")


def run() -> dict[str, Any]:
    before = verify_baseline_contract()
    hard_bounds = load_hard_safety_bounds(BOUNDS_PATH)
    selected = pd.read_csv(BASELINE_ROOT / "selected_options.csv")
    calibration = selected.loc[selected["sample_role"] == "CALIBRATION"].copy()
    holdout = selected.loc[selected["sample_role"] == "HOLDOUT"].copy()
    baseline_starts = pd.read_csv(BASELINE_ROOT / "double_heston_multistart.csv")
    baseline_comparison = pd.read_csv(BASELINE_ROOT / "model_comparison.csv")
    baseline_row = baseline_comparison.loc[baseline_comparison["model"] == "DOUBLE_HESTON"].iloc[0]
    baseline_summary = {
        key: float(baseline_row[key])
        for key in ("calibration_price_rmse", "calibration_iv_rmse", "holdout_price_rmse", "holdout_iv_rmse", "runtime_seconds")
    }
    baseline_best = baseline_row[list(PARAMETER_NAMES)].to_numpy(float)

    canonical_starts, structured_starts, paired = paired_start_population(hard_bounds)
    equivalence, equivalence_rows = equivalence_audit(hard_bounds, baseline_best)
    transformed_starts, transformed_summary = run_transformed_calibrations(
        calibration, holdout, structured_starts, hard_bounds
    )
    baseline_metrics, baseline_near, baseline_pairs = stability_metrics(baseline_starts, hard_bounds)
    transformed_metrics, transformed_near, transformed_pairs = stability_metrics(transformed_starts, hard_bounds)

    for frame in (baseline_near, transformed_near):
        derived_rows = pd.DataFrame(
            [derived_coordinates(row) for row in frame[list(PARAMETER_NAMES)].to_numpy(float)],
            index=frame.index,
        )
        for column in derived_rows.columns:
            frame[column] = derived_rows[column]

    after = protected_hashes()
    contract_passed = before["protected_hashes"] == after
    matched_population = bool(
        len(paired) == START_COUNT
        and baseline_metrics["valid_start_count"] == START_COUNT
        and transformed_metrics["valid_start_count"] == START_COUNT
        and set(paired["start_id"].astype(int)) == set(baseline_starts["start_id"].astype(int))
        and set(paired["start_id"].astype(int)) == set(transformed_starts["start_id"].astype(int))
    )
    classification = classify_experiment(
        baseline_summary,
        transformed_summary,
        baseline_metrics,
        transformed_metrics,
        equivalence_passed=not equivalence["search_space_changed"],
        contract_passed=contract_passed,
        matched_population=matched_population,
    )
    figures = _plot_outputs(
        baseline_starts,
        transformed_starts,
        baseline_metrics,
        transformed_metrics,
        baseline_near,
        transformed_near,
        baseline_pairs,
        transformed_pairs,
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    outputs = {
        "paired_starts.csv": paired,
        "equivalence_round_trip.csv": equivalence_rows,
        "reparameterized_multistart.csv": transformed_starts,
        "baseline_near_equivalent.csv": baseline_near,
        "reparameterized_near_equivalent.csv": transformed_near,
        "baseline_pairwise_distances.csv": baseline_pairs,
        "reparameterized_pairwise_distances.csv": transformed_pairs,
    }
    for name, frame in outputs.items():
        write_csv(OUTPUT_ROOT / name, frame)
    result = {
        "base_commit": BASE_COMMIT,
        "predeclared_contract": {
            "start_count": START_COUNT,
            "node_count": NODE_COUNT,
            "max_nfev": MAX_NFEV,
            "material_distance": MATERIAL_DISTANCE,
            "cluster_method": "complete_linkage",
            "cluster_distance": CLUSTER_DISTANCE,
            "strong_dispersion_reduction": STRONG_DISPERSION_REDUCTION,
            "partial_dispersion_reduction": PARTIAL_DISPERSION_REDUCTION,
        },
        "baseline_contract": before,
        "protected_hashes_after": after,
        "equivalence": equivalence,
        "matched_population": matched_population,
        "baseline_summary": baseline_summary,
        "transformed_summary": transformed_summary,
        "baseline_stability": baseline_metrics,
        "transformed_stability": transformed_metrics,
        "classification": classification,
        "figures": [path.relative_to(REPOSITORY_ROOT).as_posix() for path in figures],
    }
    write_json(OUTPUT_ROOT / "stability_comparison.json", result)
    render_report(
        before,
        equivalence,
        baseline_summary,
        transformed_summary,
        baseline_metrics,
        transformed_metrics,
        classification,
        transformed_near,
        figures,
    )
    final_hashes = protected_hashes()
    if final_hashes != after:
        raise RuntimeError("protected NTPC baseline mutated during publication")
    write_json(
        OUTPUT_ROOT / "artifact_manifest.json",
        {
            "artifact_hashes": {
                path.relative_to(OUTPUT_ROOT).as_posix(): sha256(path)
                for path in sorted(OUTPUT_ROOT.rglob("*"))
                if path.is_file() and path.name != "artifact_manifest.json"
            },
            "report_sha256": sha256(REPORT_PATH),
            "protected_baseline_hashes": final_hashes,
        },
    )
    return result


def render_existing_outputs() -> dict[str, Any]:
    """Recompute metrics/report/figures from completed optimizer evidence only."""
    before = verify_baseline_contract()
    hard_bounds = load_hard_safety_bounds(BOUNDS_PATH)
    required = {
        "paired": OUTPUT_ROOT / "paired_starts.csv",
        "equivalence": OUTPUT_ROOT / "equivalence_round_trip.csv",
        "transformed": OUTPUT_ROOT / "reparameterized_multistart.csv",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing completed experiment evidence: {missing}")
    paired = pd.read_csv(required["paired"])
    equivalence_rows = pd.read_csv(required["equivalence"])
    transformed_starts = pd.read_csv(required["transformed"])
    baseline_starts = pd.read_csv(BASELINE_ROOT / "double_heston_multistart.csv")
    baseline_comparison = pd.read_csv(BASELINE_ROOT / "model_comparison.csv")
    baseline_row = baseline_comparison.loc[baseline_comparison["model"] == "DOUBLE_HESTON"].iloc[0]
    baseline_summary = {
        key: float(baseline_row[key])
        for key in ("calibration_price_rmse", "calibration_iv_rmse", "holdout_price_rmse", "holdout_iv_rmse", "runtime_seconds")
    }
    valid_transformed = transformed_starts.loc[transformed_starts["valid"].astype(bool)]
    best = valid_transformed.sort_values(["calibration_price_rmse", "start_id"]).iloc[0]
    transformed_summary = {
        "best_start_id": int(best["start_id"]),
        "runtime_seconds": float(transformed_starts["runtime_seconds"].sum()),
        **{name: float(best[name]) for name in PARAMETER_NAMES},
        **{
            key: float(best[key])
            for key in best.index
            if key.startswith("calibration_") or key.startswith("holdout_")
        },
    }
    failures = int((~equivalence_rows["round_trip_valid"].astype(bool)).sum())
    equivalence = {
        "analytical_equivalence": (
            "Exact bijection on the same numerical interior: rectangle interiors for v0/theta are "
            "expressed by total plus conditional allocation; kappa uses the identical conditional "
            "ordering interval; sigma uses the identical Feller-safe conditional interval; correlations "
            "use a one-to-one conditional parameterization of the full intersection between the "
            "individual hard bounds and the unit disk."
        ),
        "sample_count": int(len(equivalence_rows)),
        "existing_best_fit_count": int((equivalence_rows["case_type"] == "existing_ntpc_best_fit").sum()),
        "random_interior_count": int((equivalence_rows["case_type"] == "random_interior").sum()),
        "near_boundary_count": int((equivalence_rows["case_type"] == "near_boundary").sum()),
        "correlation_annulus_count": int((equivalence_rows["case_type"] == "correlation_annulus").sum()),
        "round_trip_failure_count": failures,
        "empirical_lost_fraction": float(failures / len(equivalence_rows)),
        "maximum_abs_round_trip_error": float(equivalence_rows["max_abs_error"].max()),
        "search_space_changed": bool(failures),
    }
    baseline_metrics, baseline_near, baseline_pairs = stability_metrics(baseline_starts, hard_bounds)
    transformed_metrics, transformed_near, transformed_pairs = stability_metrics(transformed_starts, hard_bounds)
    for frame in (baseline_near, transformed_near):
        derived_rows = pd.DataFrame(
            [derived_coordinates(row) for row in frame[list(PARAMETER_NAMES)].to_numpy(float)],
            index=frame.index,
        )
        for column in derived_rows.columns:
            frame[column] = derived_rows[column]
    after = protected_hashes()
    matched_population = bool(
        len(paired) == START_COUNT
        and baseline_metrics["valid_start_count"] == START_COUNT
        and transformed_metrics["valid_start_count"] == START_COUNT
        and set(paired["start_id"].astype(int)) == set(baseline_starts["start_id"].astype(int))
        and set(paired["start_id"].astype(int)) == set(transformed_starts["start_id"].astype(int))
    )
    classification = classify_experiment(
        baseline_summary,
        transformed_summary,
        baseline_metrics,
        transformed_metrics,
        equivalence_passed=not equivalence["search_space_changed"],
        contract_passed=before["protected_hashes"] == after,
        matched_population=matched_population,
    )
    figures = _plot_outputs(
        baseline_starts, transformed_starts, baseline_metrics, transformed_metrics,
        baseline_near, transformed_near, baseline_pairs, transformed_pairs,
    )
    for name, frame in {
        "baseline_near_equivalent.csv": baseline_near,
        "reparameterized_near_equivalent.csv": transformed_near,
        "baseline_pairwise_distances.csv": baseline_pairs,
        "reparameterized_pairwise_distances.csv": transformed_pairs,
    }.items():
        write_csv(OUTPUT_ROOT / name, frame)
    result = {
        "base_commit": BASE_COMMIT,
        "predeclared_contract": {
            "start_count": START_COUNT,
            "node_count": NODE_COUNT,
            "max_nfev": MAX_NFEV,
            "material_distance": MATERIAL_DISTANCE,
            "cluster_method": "complete_linkage",
            "cluster_distance": CLUSTER_DISTANCE,
            "strong_dispersion_reduction": STRONG_DISPERSION_REDUCTION,
            "partial_dispersion_reduction": PARTIAL_DISPERSION_REDUCTION,
        },
        "baseline_contract": before,
        "protected_hashes_after": after,
        "equivalence": equivalence,
        "matched_population": matched_population,
        "baseline_summary": baseline_summary,
        "transformed_summary": transformed_summary,
        "baseline_stability": baseline_metrics,
        "transformed_stability": transformed_metrics,
        "classification": classification,
        "figures": [path.relative_to(REPOSITORY_ROOT).as_posix() for path in figures],
    }
    write_json(OUTPUT_ROOT / "stability_comparison.json", result)
    render_report(
        before, equivalence, baseline_summary, transformed_summary, baseline_metrics,
        transformed_metrics, classification, transformed_near, figures,
    )
    write_json(
        OUTPUT_ROOT / "artifact_manifest.json",
        {
            "artifact_hashes": {
                path.relative_to(OUTPUT_ROOT).as_posix(): sha256(path)
                for path in sorted(OUTPUT_ROOT.rglob("*"))
                if path.is_file() and path.name != "artifact_manifest.json"
            },
            "report_sha256": sha256(REPORT_PATH),
            "protected_baseline_hashes": protected_hashes(),
        },
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-only", action="store_true")
    arguments = parser.parse_args()
    result = render_existing_outputs() if arguments.render_only else run()
    print(json.dumps({
        "classification": result["classification"]["classification"],
        "baseline": result["baseline_summary"],
        "reparameterized": result["transformed_summary"],
        "baseline_stability": result["baseline_stability"],
        "reparameterized_stability": result["transformed_stability"],
    }, indent=2, default=str))
