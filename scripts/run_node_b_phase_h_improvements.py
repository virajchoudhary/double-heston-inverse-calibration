"""Phase H: limited, scientifically defensible calibration-improvement arms.

All arms use the SAME four clean true surfaces on the full 108 grid and
comparable compute budgets (12 TRF starts, max_nfev=120):

- ``baseline``    — 12 deterministic broad starts, unweighted normalized-price
                    residuals (identical to Phase B clean arm).
- ``sobol``       — 12 quasi-random Sobol starts in [-2, 2]^10 unconstrained.
- ``relweight``   — same starts as baseline, relative-price residuals
                    (pred - obs) / obs (approximate vega/IV-style weighting).
- ``polish``      — best Phase B solution per case re-optimized with
                    max_nfev=800 (optimizer-capacity probe).
- ``prior_ranges``— identical optimizer and starts, but the sigmoid transform
                    uses the YAML ``empirical_sampling_ranges`` as bounds.
                    EXPLICITLY LABELLED prior-driven stabilization, NOT
                    data-driven identification.

No bound is tightened to make recovery look better; the prior arm is reported
separately and never pooled with data-driven arms.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import qmc

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import scripts.node_b_toolkit as toolkit
from src.calibrate_double_heston import (
    boundary_diagnostics,
    load_hard_safety_bounds,
    unconstrained_to_parameters,
)
from src.constants import PARAMETER_NAMES
from src.constraints import validate_parameters

ARTIFACT_DIR = toolkit.EVIDENCE_ROOT / "artifacts"
TABLE_DIR = toolkit.EVIDENCE_ROOT / "tables"
SEED = 20260823
START_COUNT = 12


def evaluate_arm(
    case_row,
    geometry: toolkit.Geometry,
    bounds: dict,
    *,
    arm: str,
    truth: np.ndarray,
) -> list[dict]:
    clean = toolkit.normalized_observables_fast(truth, geometry)
    observed = clean  # clean arm only in Phase H
    widths = toolkit.parameter_widths(bounds)

    if arm == "baseline" or arm == "relweight":
        rng = np.random.default_rng(SEED + 1000 * int(case_row.case_index))
        starts = [("midpoint", np.zeros(10))] + [
            (f"broad_{i}", rng.normal(0.0, 1.25, 10)) for i in range(1, START_COUNT)
        ]
    elif arm == "sobol":
        engine = qmc.Sobol(d=10, scramble=True, seed=SEED + 1000 * int(case_row.case_index))
        unit = engine.random(START_COUNT)
        starts = [
            (f"sobol_{i}", 4.0 * row - 2.0) for i, row in enumerate(unit)
        ]
    elif arm == "polish":
        frame = pd.read_csv(ARTIFACT_DIR / "phase_b_multistart_full108.csv")
        pool = frame[
            (frame.case_id == case_row.case_id)
            & (frame.noise_level == 0.0)
            & frame.finite_solution
        ]
        ordered = pool.sort_values("price_rmse_normalized").head(START_COUNT)
        starts = []
        for position, row in enumerate(ordered.itertuples()):
            recovered = np.asarray([getattr(row, f"recovered_{name}") for name in PARAMETER_NAMES])
            from src.calibrate_double_heston import parameters_to_unconstrained

            starts.append((f"polish_{position}", parameters_to_unconstrained(recovered, bounds)))
    elif arm == "prior_ranges":
        rng = np.random.default_rng(SEED + 1000 * int(case_row.case_index))
        starts = [("midpoint", np.zeros(10))] + [
            (f"broad_{i}", rng.normal(0.0, 1.25, 10)) for i in range(1, START_COUNT)
        ]
    else:
        raise ValueError(f"unknown arm {arm}")

    active_bounds = bounds
    if arm == "prior_ranges":
        import yaml

        payload = yaml.safe_load(toolkit.BOUNDS_PATH.read_text(encoding="utf-8"))
        active_bounds = {
            name: (
                float(payload["empirical_sampling_ranges"]["parameter_bounds"][name]["lower"]),
                float(payload["empirical_sampling_ranges"]["parameter_bounds"][name]["upper"]),
            )
            for name in PARAMETER_NAMES
        }

    relative = arm == "relweight"

    def residuals(latent: np.ndarray) -> np.ndarray:
        candidate = unconstrained_to_parameters(latent, active_bounds)
        predicted = toolkit.normalized_observables_fast(candidate, geometry)
        difference = predicted - observed
        return difference / observed if relative else difference

    max_nfev = 800 if arm == "polish" else 120
    records = []
    for start_index, (strategy, start) in enumerate(starts):
        record = {
            "case_id": case_row.case_id,
            "arm": arm,
            "start_index": start_index,
            "start_strategy": strategy,
        }
        run_started = time.perf_counter()
        try:
            result = least_squares(
                residuals, start, method="trf", max_nfev=max_nfev,
                ftol=1e-10, xtol=1e-10, gtol=1e-10, diff_step=2e-5,
            )
            recovered = unconstrained_to_parameters(result.x, active_bounds)
            predicted = toolkit.normalized_observables_fast(recovered, geometry)
            displacement = (recovered - truth) / widths  # widths from HARD bounds in all arms
            records.append(record | {
                "optimizer_success": bool(result.success),
                "nfev": int(result.nfev),
                "price_rmse_normalized": float(np.sqrt(np.mean((predicted - observed) ** 2))),
                "parameter_rmse_full_range": float(np.sqrt(np.mean(displacement**2))),
                "constraint_valid": bool(validate_parameters(recovered)["is_valid"]),
                "bound_hit": bool(boundary_diagnostics(recovered, bounds)),
                "runtime_seconds": time.perf_counter() - run_started,
            })
        except Exception as error:
            records.append(record | {
                "optimizer_success": False, "nfev": 0,
                "price_rmse_normalized": np.nan,
                "parameter_rmse_full_range": np.nan,
                "constraint_valid": False, "bound_hit": False,
                "runtime_seconds": time.perf_counter() - run_started,
                "error": f"{type(error).__name__}",
            })
    return records


def main() -> None:
    bounds = load_hard_safety_bounds(toolkit.BOUNDS_PATH)
    geometry = toolkit.full_108_geometry()
    cases = toolkit.representative_cases()

    records: list[dict] = []
    started = time.perf_counter()
    for case_row in cases.itertuples():
        truth = toolkit.case_vector(case_row)
        for arm in ("baseline", "sobol", "relweight", "polish", "prior_ranges"):
            batch = evaluate_arm(case_row, geometry, bounds, arm=arm, truth=truth)
            records.extend(batch)
            usable = [r for r in batch if np.isfinite(r["parameter_rmse_full_range"])]
            if usable:
                print(
                    f"[phase_h] {case_row.case_id} {arm}: "
                    f"median param RMSE={np.median([r['parameter_rmse_full_range'] for r in usable]):.3f} "
                    f"best={np.min([r['parameter_rmse_full_range'] for r in usable]):.3e} "
                    f"price RMSE median={np.median([r['price_rmse_normalized'] for r in usable]):.3e}",
                    flush=True,
                )
    frame = pd.DataFrame(records)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLE_DIR / "phase_h_improvement_arms.csv", index=False)
    summary = (
        frame.groupby("arm")
        .agg(
            median_parameter_rmse=("parameter_rmse_full_range", "median"),
            best_parameter_rmse=("parameter_rmse_full_range", "min"),
            median_price_rmse=("price_rmse_normalized", "median"),
            boundary_hit_fraction=("bound_hit", "mean"),
            optimizer_success_fraction=("optimizer_success", "mean"),
            failures=("parameter_rmse_full_range", lambda s: int(s.isna().sum())),
        )
        .reset_index()
    )
    summary.to_csv(TABLE_DIR / "phase_h_summary.csv", index=False)
    print(summary.to_string(index=False))
    toolkit.log_experiment({
        "experiment_id": "node_b_phase_h_improvement_arms",
        "hypothesis": "Quasi-random starts, relative-price weighting, and larger optimizer budgets do not materially fix clean recovery dispersion; the prior-ranges arm stabilizes only by shrinking the feasible set (prior-driven, not data-driven).",
        "code": "scripts/run_node_b_phase_h_improvements.py",
        "configuration": {
            "arms": ["baseline", "sobol", "relweight", "polish", "prior_ranges"],
            "starts": START_COUNT,
            "seed": SEED,
            "note": "prior_ranges uses YAML empirical_sampling_ranges as transform bounds; reported separately as prior-driven stabilization",
        },
        "sample_size": {"runs": int(len(frame))},
        "runtime_seconds": time.perf_counter() - started,
        "result": json.loads(summary.to_json(orient="records")),
        "artifacts": ["tables/phase_h_improvement_arms.csv", "tables/phase_h_summary.csv"],
        "failure_status": "completed",
    })


if __name__ == "__main__":
    main()
