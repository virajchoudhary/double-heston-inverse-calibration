"""Phase E: objective-landscape / profile analysis on the full 108 grid.

Two complementary views, both on clean true surfaces (no optimizer confound in
the first, controlled re-optimization in the second):

1. Valley scan: evaluate the exact least-squares objective along the weakest
   and strongest Jacobian singular directions and along the single worst
   parameter axis (physical space, validity-clipped).
2. Compensated profiles: fix the worst parameters at controlled offsets from
   truth and re-optimize the remaining parameters; record the profile
   objective floor and how far free parameters drift.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import scripts.node_b_toolkit as toolkit
from src.calibrate_double_heston import unconstrained_to_parameters
from src.constants import PARAMETER_NAMES
from src.constraints import validate_parameters

ARTIFACT_DIR = toolkit.EVIDENCE_ROOT / "artifacts"
FIGURE_DIR = toolkit.EVIDENCE_ROOT / "figures"
TABLE_DIR = toolkit.EVIDENCE_ROOT / "tables"

VALLEY_POINTS = 41
VALLEY_RANGE = 2.0
PROFILE_OFFSETS = np.linspace(-1.5, 1.5, 11)
PROFILE_PARAMETERS_COUNT = 3
PROFILE_REOPT_NFEV = 120


def objective_against_clean(
    candidate: np.ndarray, clean: np.ndarray, geometry: toolkit.Geometry
) -> float:
    predicted = toolkit.normalized_observables_fast(candidate, geometry)
    return float(np.mean((predicted - clean) ** 2))


def valley_scan(
    truth: np.ndarray,
    geometry: toolkit.Geometry,
    bounds: dict,
) -> pd.DataFrame:
    widths = toolkit.parameter_widths(bounds)
    jacobian = toolkit.scaled_parameter_jacobian_fast(truth, geometry, bounds)
    _, singular_values, right_vectors = np.linalg.svd(jacobian, full_matrices=False)
    directions = {
        "weakest_singular_direction": right_vectors[-1],
        "median_singular_direction": right_vectors[jacobian.shape[1] // 2],
        "strongest_singular_direction": right_vectors[0],
    }
    clean = toolkit.normalized_observables_fast(truth, geometry)
    rows = []
    for label, direction in directions.items():
        for t in np.linspace(-VALLEY_RANGE, VALLEY_RANGE, VALLEY_POINTS):
            candidate = truth + t * direction * widths
            if not validate_parameters(candidate)["is_valid"]:
                status, objective = "invalid", np.nan
            else:
                status = "valid"
                objective = objective_against_clean(candidate, clean, geometry)
            rows.append(
                {
                    "scan": label,
                    "t_scaled_widths": float(t),
                    "objective_mse_normalized": objective,
                    "status": status,
                    "parameter_rmse_full_range": float(
                        np.sqrt(np.mean((t * direction) ** 2))
                    ),
                }
            )
    return pd.DataFrame(rows)


def compensated_profiles(
    case_id: str,
    truth: np.ndarray,
    worst_parameters: list[int],
    geometry: toolkit.Geometry,
    bounds: dict,
    seed: int,
) -> pd.DataFrame:
    widths = toolkit.parameter_widths(bounds)
    clean = toolkit.normalized_observables_fast(truth, geometry)
    rng = np.random.default_rng(seed)
    rows = []
    for index in worst_parameters:
        name = PARAMETER_NAMES[index]
        for offset in PROFILE_OFFSETS:
            fixed_value = truth[index] + offset * widths[index]
            lower = bounds[name][0]

            def residuals(latent: np.ndarray) -> np.ndarray:
                candidate = unconstrained_to_parameters(latent, bounds)
                candidate[index] = fixed_value
                predicted = toolkit.normalized_observables_fast(candidate, geometry)
                return predicted - clean

            # Two starts: transform midpoint and a random broad start.
            starts = [
                ("midpoint", np.zeros(10)),
                ("broad", rng.normal(0.0, 1.0, 10)),
            ]
            best: dict | None = None
            for strategy, start in starts:
                try:
                    result = least_squares(
                        residuals, start, method="trf", max_nfev=PROFILE_REOPT_NFEV,
                        ftol=1e-10, xtol=1e-10, gtol=1e-10, diff_step=2e-5,
                    )
                    candidate = unconstrained_to_parameters(result.x, bounds)
                    candidate[index] = fixed_value
                    if not validate_parameters(candidate)["is_valid"]:
                        continue
                    objective = objective_against_clean(candidate, clean, geometry)
                    free_displacement = np.delete(candidate - truth, index) / np.delete(widths, index)
                    record = {
                        "objective_mse_normalized": objective,
                        "free_parameter_rmse": float(np.sqrt(np.mean(free_displacement**2))),
                        "runtime_nfev": int(result.nfev),
                    }
                    if best is None or record["objective_mse_normalized"] < best["objective_mse_normalized"]:
                        best = record
                except Exception:
                    continue
            rows.append(
                {
                    "case_id": case_id,
                    "fixed_parameter": name,
                    "offset_scaled_widths": float(offset),
                    "fixed_value": float(fixed_value),
                    **(best or {"objective_mse_normalized": np.nan, "free_parameter_rmse": np.nan, "runtime_nfev": 0}),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    bounds = toolkit.load_hard_safety_bounds(toolkit.BOUNDS_PATH)
    geometry = toolkit.full_108_geometry()
    exemplars = json.loads(
        (ARTIFACT_DIR / "phase_f_exemplars.json").read_text(encoding="utf-8")
    )["exemplars"]
    exemplar = max(exemplars, key=lambda item: item["parameter_rmse_full_range"])
    case_id = exemplar["case_id"]
    truth = np.asarray(exemplar["true_parameters"], dtype=np.float64)

    started = time.perf_counter()
    valley = valley_scan(truth, geometry, bounds)
    displacement_vector = np.asarray(
        [
            np.asarray(exemplar["recovered_parameters"])[index]
            - np.asarray(exemplar["true_parameters"])[index]
            for index in range(len(PARAMETER_NAMES))
        ]
    ) / toolkit.parameter_widths(bounds)
    worst_parameters = list(np.argsort(-np.abs(displacement_vector))[:PROFILE_PARAMETERS_COUNT])
    profiles = compensated_profiles(case_id, truth, worst_parameters, geometry, bounds, seed=20260822)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    valley.to_csv(TABLE_DIR / "phase_e_valley_scan.csv", index=False)
    profiles.to_csv(TABLE_DIR / "phase_e_compensated_profiles.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, group in valley.groupby("scan"):
        axes[0].semilogy(
            group.t_scaled_widths,
            group.objective_mse_normalized,
            marker=".",
            label=label.replace("_singular_direction", ""),
        )
    axes[0].set_xlabel("displacement along singular direction (full-range widths)")
    axes[0].set_ylabel("objective (MSE, normalized prices)")
    axes[0].set_title(f"Objective valley, {case_id} (no re-optimization)")
    axes[0].legend()
    for name, group in profiles.groupby("fixed_parameter"):
        axes[1].semilogy(
            group.offset_scaled_widths,
            group.objective_mse_normalized,
            marker="o",
            label=f"fixed {name}",
        )
    axes[1].set_xlabel("fixed-parameter offset from truth (full-range widths)")
    axes[1].set_ylabel("profile objective floor (MSE)")
    axes[1].set_title("Compensated profiles (other 9 re-optimized)")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "phase_e_landscape.png", dpi=150)

    valley_summary = (
        valley[valley.status == "valid"]
        .groupby("scan")
        .objective_mse_normalized.apply(
            lambda values: float(np.nanmax(values) / max(np.nanmin(values), np.finfo(float).tiny))
        )
    )
    print(valley_summary.to_string())
    print(profiles.head(24).to_string(index=False))
    toolkit.log_experiment({
        "experiment_id": "node_b_phase_e_landscape_profiles",
        "hypothesis": "The full-108 objective is flat along the weakest scaled directions (valley) and the worst parameters have shallow compensated profiles, i.e. near-equivalence is a property of the landscape, not of optimizer failure.",
        "code": "scripts/run_node_b_phase_e_profiles.py",
        "configuration": {
            "case": case_id,
            "valley_points": VALLEY_POINTS,
            "valley_range_scaled": VALLEY_RANGE,
            "profile_offsets": PROFILE_OFFSETS.tolist(),
            "profile_parameters": [PARAMETER_NAMES[i] for i in worst_parameters],
            "reopt": f"TRF max_nfev={PROFILE_REOPT_NFEV}",
        },
        "sample_size": {"valley_points": int(len(valley)), "profile_points": int(len(profiles))},
        "runtime_seconds": time.perf_counter() - started,
        "result": {
            "valley_dynamic_range": json.loads(valley_summary.to_json()),
            "profile_min_objective": float(np.nanmin(profiles.objective_mse_normalized)),
        },
        "artifacts": [
            "tables/phase_e_valley_scan.csv",
            "tables/phase_e_compensated_profiles.csv",
            "figures/phase_e_landscape.png",
        ],
        "failure_status": "completed",
    })


if __name__ == "__main__":
    main()
