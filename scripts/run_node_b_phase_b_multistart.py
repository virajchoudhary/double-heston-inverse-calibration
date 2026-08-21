"""Phases B + C + F: multi-start recovery dispersion, noise robustness, and
near-equivalent-solution search on the FULL provisional 108-quote grid.

Frozen contract (predeclared before running):
- Four representative truth vectors (same as committed global-ambiguity runs).
- 12 deterministic starts per case and noise level: transform midpoint plus
  11 broad N(0, 1.25^2) unconstrained draws.
- Noise levels 0 / 0.5% / 1% / 2% multiplicative on prices, lognormal.
- TRF least-squares, max_nfev=120, ftol=xtol=gtol=1e-10, diff_step=2e-5,
  residuals on spot-normalized prices (committed G2 convention).
- Near-equivalence: normalized price RMSE <= 2.5e-7; material displacement:
  range-scaled parameter RMSE >= 0.05; complete-linkage clusters at 0.10.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import scripts.node_b_toolkit as toolkit
import scripts.run_g2_global_ambiguity_analysis as ambiguity
from src.calibrate_double_heston import boundary_diagnostics, unconstrained_to_parameters
from src.constants import PARAMETER_NAMES
from src.constraints import validate_parameters
from src.double_heston import propagate_variance_state

ARTIFACT_DIR = toolkit.EVIDENCE_ROOT / "artifacts"
TABLE_DIR = toolkit.EVIDENCE_ROOT / "tables"
FIGURE_DIR = toolkit.EVIDENCE_ROOT / "figures"

ANALYSIS_SEED_B = 27182818
START_COUNT = 12
NOISE_LEVELS = (0.0, 0.005, 0.01, 0.02)
MAX_NFEV = 120
OPTIMIZER_TOLERANCE = 1.0e-10
DIFF_STEP = 2.0e-5
CLUSTER_DISTANCE_CUTOFF = 0.10


def case_seed(case_index: int, noise_level: float, purpose: int) -> int:
    noise_code = int(round(noise_level * 10_000))
    return ANALYSIS_SEED_B + 100_000 * case_index + 100 * noise_code + purpose


def deterministic_starts(seed: int, count: int) -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    starts = [("neutral_transform_midpoint", np.zeros(len(PARAMETER_NAMES)))]
    starts.extend(
        (f"deterministic_broad_{index}", rng.normal(0.0, 1.25, len(PARAMETER_NAMES)))
        for index in range(1, count)
    )
    return starts


def recover_one_case(
    case_row,
    geometry: toolkit.Geometry,
    bounds: dict,
    *,
    noise_level: float,
) -> list[dict]:
    truth = toolkit.case_vector(case_row)
    widths = toolkit.parameter_widths(bounds)
    strikes, maturities, options, _, _ = geometry.build()
    clean = toolkit.normalized_observables_fast(truth, geometry)
    observed = toolkit.multiplicative_noise(clean, case_seed(int(case_row.case_index), noise_level, 1), noise_level)

    jacobian = toolkit.scaled_parameter_jacobian_fast(truth, geometry, bounds)
    _, _, right_vectors = np.linalg.svd(jacobian, full_matrices=False)
    weakest_direction = right_vectors[-1]

    def residuals(latent: np.ndarray) -> np.ndarray:
        candidate = unconstrained_to_parameters(latent, bounds)
        return (
            toolkit.normalized_observables_fast(candidate, geometry) - observed
        )

    records: list[dict] = []
    for start_index, (strategy, start) in enumerate(
        deterministic_starts(case_seed(int(case_row.case_index), noise_level, 11), START_COUNT)
    ):
        record: dict = {
            "case_id": case_row.case_id,
            "case_index": int(case_row.case_index),
            "distribution": case_row.distribution,
            "noise_level": noise_level,
            "start_index": start_index,
            "start_strategy": strategy,
        }
        run_started = time.perf_counter()
        try:
            result = least_squares(
                residuals, start, method="trf", max_nfev=MAX_NFEV,
                ftol=OPTIMIZER_TOLERANCE, xtol=OPTIMIZER_TOLERANCE,
                gtol=OPTIMIZER_TOLERANCE, diff_step=DIFF_STEP,
            )
            recovered = unconstrained_to_parameters(result.x, bounds)
            predicted = toolkit.normalized_observables_fast(recovered, geometry)
            displacement = (recovered - truth) / widths
            parameter_rmse = float(np.sqrt(np.mean(displacement**2)))
            validation = validate_parameters(recovered)
            norm = float(np.linalg.norm(displacement))
            record.update({
                "optimizer_success": bool(result.success),
                "optimizer_status": int(result.status),
                "nfev": int(result.nfev),
                "price_rmse_normalized": float(np.sqrt(np.mean((predicted - observed) ** 2))),
                "price_rmse_clean_normalized": float(np.sqrt(np.mean((predicted - clean) ** 2))),
                "parameter_rmse_full_range": parameter_rmse,
                "constraint_valid": bool(validation["is_valid"]),
                "finite_solution": bool(np.isfinite(recovered).all()),
                "bound_hit": bool(boundary_diagnostics(recovered, bounds)),
                "bound_reasons": ";".join(boundary_diagnostics(recovered, bounds)),
                "material_displacement": bool(parameter_rmse >= toolkit.MATERIAL_DISPLACEMENT_RMSE),
                "near_equivalent": bool(
                    math.sqrt(np.mean((predicted - observed) ** 2))
                    <= toolkit.NEAR_PRICE_EQUIVALENCE_RMSE
                ),
                "weakest_direction_absolute_cosine": (
                    float(abs(np.dot(displacement, weakest_direction)) / norm) if norm else math.nan
                ),
                "runtime_seconds": time.perf_counter() - run_started,
            })
            for index, name in enumerate(PARAMETER_NAMES):
                record[f"true_{name}"] = float(truth[index])
                record[f"recovered_{name}"] = float(recovered[index])
                record[f"scaled_displacement_{name}"] = float(displacement[index])
        except Exception as error:  # retain failures as evidence
            record.update({
                "optimizer_success": False, "optimizer_status": -1, "nfev": 0,
                "price_rmse_normalized": math.nan, "price_rmse_clean_normalized": math.nan,
                "parameter_rmse_full_range": math.nan, "constraint_valid": False,
                "finite_solution": False, "bound_hit": False, "bound_reasons": "",
                "material_displacement": False, "near_equivalent": False,
                "weakest_direction_absolute_cosine": math.nan,
                "runtime_seconds": time.perf_counter() - run_started,
                "error": f"{type(error).__name__}: {error}",
            })
            for index, name in enumerate(PARAMETER_NAMES):
                record[f"true_{name}"] = float(truth[index])
                record[f"recovered_{name}"] = math.nan
                record[f"scaled_displacement_{name}"] = math.nan
        records.append(record)
    return records


def latent_factor_table(truth: np.ndarray, recovered: np.ndarray) -> dict:
    horizon = np.arange(0, 181, 5)
    true_slow = propagate_variance_state(truth[0], truth[1], truth[4], horizon)
    true_fast = propagate_variance_state(truth[5], truth[6], truth[9], horizon)
    rec_slow = propagate_variance_state(recovered[0], recovered[1], recovered[4], horizon)
    rec_fast = propagate_variance_state(recovered[5], recovered[6], recovered[9], horizon)
    return {
        "horizon_days": horizon.tolist(),
        "true_slow": np.asarray(true_slow).tolist(),
        "true_fast": np.asarray(true_fast).tolist(),
        "true_total": (np.asarray(true_slow) + np.asarray(true_fast)).tolist(),
        "recovered_slow": np.asarray(rec_slow).tolist(),
        "recovered_fast": np.asarray(rec_fast).tolist(),
        "recovered_total": (np.asarray(rec_slow) + np.asarray(rec_fast)).tolist(),
        "true_half_life_slow_days": float(math.log(2.0) / truth[0] * 365.0),
        "true_half_life_fast_days": float(math.log(2.0) / truth[5] * 365.0),
        "recovered_half_life_slow_days": float(math.log(2.0) / recovered[0] * 365.0),
        "recovered_half_life_fast_days": float(math.log(2.0) / recovered[5] * 365.0),
    }


def main() -> None:
    bounds = toolkit.load_hard_safety_bounds(toolkit.BOUNDS_PATH)
    geometry = toolkit.full_108_geometry()
    cases = toolkit.representative_cases()
    output_csv = ARTIFACT_DIR / "phase_b_multistart_full108.csv"

    import os

    started = time.perf_counter()
    if output_csv.exists() and os.environ.get("NODE_B_REUSE") == "1":
        frame = pd.read_csv(output_csv)
        print(f"[phase_b] reusing {len(frame)} saved runs from {output_csv}")
    else:
        records: list[dict] = []
        started = time.perf_counter()
        for case_row in cases.itertuples():
            for noise_level in NOISE_LEVELS:
                batch = recover_one_case(case_row, geometry, bounds, noise_level=noise_level)
                records.extend(batch)
                usable = [r for r in batch if np.isfinite(r["price_rmse_normalized"])]
                print(
                    f"[phase_b] {case_row.case_id} noise={noise_level:.2%}: "
                    f"median price RMSE={np.median([r['price_rmse_normalized'] for r in usable]):.3e} "
                    f"median param RMSE={np.median([r['parameter_rmse_full_range'] for r in usable]):.3f} "
                    f"boundary hits={sum(r['bound_hit'] for r in usable)}/{len(usable)}",
                    flush=True,
                )
        frame = pd.DataFrame(records)
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_csv, index=False)

    # ---- Phase F: near-equivalent analysis on clean runs ----
    clean = frame[(frame.noise_level == 0.0) & frame.finite_solution & frame.constraint_valid]
    near = clean[clean.price_rmse_normalized <= toolkit.NEAR_PRICE_EQUIVALENCE_RMSE].copy()
    cluster_counts: dict[str, int] = {}
    exemplars: list[dict] = []
    for case_id, group in near.groupby("case_id"):
        recovered_matrix = np.asarray([
            [getattr(row, f"recovered_{name}") for name in PARAMETER_NAMES] for row in group.itertuples()
        ])
        lower = np.asarray([bounds[name][0] for name in PARAMETER_NAMES])
        widths = toolkit.parameter_widths(bounds)
        coordinates = (recovered_matrix - lower) / widths
        labels = ambiguity.complete_linkage_clusters(coordinates, cutoff=CLUSTER_DISTANCE_CUTOFF)
        group = group.assign(cluster_id=labels)
        cluster_counts[case_id] = int(len(set(labels.tolist())))
        best = group.loc[group.parameter_rmse_full_range.idxmax()]
        truth = np.asarray([best[f"true_{name}"] for name in PARAMETER_NAMES])
        recovered = np.asarray([best[f"recovered_{name}"] for name in PARAMETER_NAMES])
        exemplars.append({
            "case_id": case_id,
            "start_index": int(best.start_index),
            "start_strategy": best.start_strategy,
            "price_rmse_normalized": float(best.price_rmse_normalized),
            "parameter_rmse_full_range": float(best.parameter_rmse_full_range),
            "true_parameters": truth.tolist(),
            "recovered_parameters": recovered.tolist(),
            "per_parameter_relative_error": (
                (np.abs(recovered - truth) / np.maximum(np.abs(truth), 1e-4))
            ).tolist(),
            "latent_comparison": latent_factor_table(truth, recovered),
        })

    noise_summary = (
        frame[frame.finite_solution]
        .groupby("noise_level")
        .agg(
            median_price_rmse_normalized=("price_rmse_normalized", "median"),
            median_clean_price_rmse=("price_rmse_clean_normalized", "median"),
            median_parameter_rmse=("parameter_rmse_full_range", "median"),
            max_parameter_rmse=("parameter_rmse_full_range", "max"),
            boundary_hit_fraction=("bound_hit", "mean"),
            optimizer_success_fraction=("optimizer_success", "mean"),
            median_runtime=("runtime_seconds", "median"),
        )
        .reset_index()
    )
    noise_summary.to_csv(TABLE_DIR / "phase_c_noise_summary.csv", index=False)
    (ARTIFACT_DIR / "phase_f_exemplars.json").write_text(
        json.dumps({"exemplars": exemplars, "cluster_counts": cluster_counts}, indent=2),
        encoding="utf-8",
    )

    print("\nNoise summary:")
    print(noise_summary.to_string(index=False))
    print(f"\nClean near-equivalent count: {len(near)}/{len(clean)}; clusters per case: {cluster_counts}")
    print(f"Total runtime: {time.perf_counter() - started:.1f}s")

    toolkit.log_experiment({
        "experiment_id": "node_b_phase_b_multistart_full108",
        "hypothesis": "On the full provisional 108-quote grid, multi-start recovery still yields materially displaced solutions at tiny price error (global near-equivalence), and noise degrades parameter recovery far faster than repricing.",
        "code": "scripts/run_node_b_phase_b_multistart.py",
        "configuration": {
            "geometry": "full108, constant carry r=0.06 q=0.02, 64-node pricing",
            "cases": "4 representative truth vectors (committed global-ambiguity selection)",
            "starts_per_case_noise": START_COUNT,
            "noise_levels": list(NOISE_LEVELS),
            "optimizer": "TRF least_squares, max_nfev=120, tol=1e-10, diff_step=2e-5",
            "residual_convention": "spot-normalized prices, unweighted absolute",
            "seed_base": ANALYSIS_SEED_B,
        },
        "sample_size": {"runs": int(len(frame))},
        "runtime_seconds": time.perf_counter() - started,
        "result": {
            "noise_summary": json.loads(noise_summary.to_json(orient="records")),
            "clean_near_equivalent": {
                "count": int(len(near)), "total_clean": int(len(clean)),
                "clusters_per_case": cluster_counts,
            },
        },
        "artifacts": [
            "artifacts/phase_b_multistart_full108.csv",
            "tables/phase_c_noise_summary.csv",
            "artifacts/phase_f_exemplars.json",
        ],
        "failure_status": "completed",
    })


if __name__ == "__main__":
    main()
