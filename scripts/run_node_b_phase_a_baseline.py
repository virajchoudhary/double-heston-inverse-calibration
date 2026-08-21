"""Phase A: reproduce the canonical controlled-recovery baseline.

Uses the production entry point ``src.calibrate_double_heston`` unchanged on
the provisional full 108-quote grid for the four representative truth vectors,
clean and 1% multiplicative noise, three canonical deterministic starts each.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import scripts.node_b_toolkit as toolkit
from src.calibrate_double_heston import calibrate_double_heston

ARTIFACT_DIR = toolkit.EVIDENCE_ROOT / "artifacts"
NOISE_LEVELS = (0.0, 0.01)
PHASE_A_SEED = 42


def main() -> None:
    cases = toolkit.representative_cases()
    geometry = toolkit.full_108_geometry()
    bounds = toolkit.load_hard_safety_bounds(toolkit.BOUNDS_PATH)
    strikes, maturities, options, _, _ = geometry.build()

    all_frames: list[pd.DataFrame] = []
    started = time.perf_counter()
    for row in cases.itertuples():
        truth = toolkit.case_vector(row)
        clean_prices = toolkit.price_surface_fast(
            truth, strikes, maturities, options
        )
        for noise_level in NOISE_LEVELS:
            seed = 20260822 + 1000 * int(row.case_index) + int(round(noise_level * 10_000))
            observed = toolkit.multiplicative_noise(clean_prices, seed, noise_level)
            frame = calibrate_double_heston(
                toolkit.SPOT,
                strikes,
                maturities,
                toolkit.RISK_FREE_RATE,
                toolkit.DIVIDEND_YIELD,
                options,
                observed,
                truth,
                toolkit.BOUNDS_PATH,
                node_count=toolkit.NODE_COUNT,
                max_nfev=300,
                seed=PHASE_A_SEED,
            )
            frame.insert(0, "case_id", row.case_id)
            frame.insert(1, "case_index", int(row.case_index))
            frame.insert(2, "distribution", row.distribution)
            frame.insert(3, "noise_level", noise_level)
            frame.insert(4, "seed", seed)
            all_frames.append(frame)
            print(
                f"[phase_a] {row.case_id} noise={noise_level:.2%} "
                f"best_price_rmse={frame.price_rmse.min():.3e} "
                f"param_rmse range=[{frame.parameter_rmse.min():.3e}, {frame.parameter_rmse.max():.3e}]",
                flush=True,
            )

    combined = pd.concat(all_frames, ignore_index=True)
    output_path = ARTIFACT_DIR / "phase_a_canonical_baseline.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    runtime = time.perf_counter() - started

    summary = {
        "cases": sorted(combined.case_id.unique().tolist()),
        "starts_per_case_noise": 3,
        "noise_levels": list(NOISE_LEVELS),
        "total_starts": int(len(combined)),
        "clean": {
            "best_price_rmse": float(combined[combined.noise_level == 0.0].price_rmse.min()),
            "worst_price_rmse": float(combined[combined.noise_level == 0.0].price_rmse.max()),
            "parameter_rmse_min": float(combined[combined.noise_level == 0.0].parameter_rmse.min()),
            "parameter_rmse_max": float(combined[combined.noise_level == 0.0].parameter_rmse.max()),
            "boundary_near_fraction": float(
                combined[combined.noise_level == 0.0].boundary_near.mean()
            ),
        },
        "noisy_1pct": {
            "best_price_rmse": float(combined[combined.noise_level == 0.01].price_rmse.min()),
            "worst_price_rmse": float(combined[combined.noise_level == 0.01].price_rmse.max()),
            "parameter_rmse_min": float(combined[combined.noise_level == 0.01].parameter_rmse.min()),
            "parameter_rmse_max": float(combined[combined.noise_level == 0.01].parameter_rmse.max()),
            "boundary_near_fraction": float(
                combined[combined.noise_level == 0.01].boundary_near.mean()
            ),
        },
        "runtime_seconds_total": runtime,
        "artifact": str(output_path.relative_to(REPOSITORY_ROOT)),
    }
    (ARTIFACT_DIR / "phase_a_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    toolkit.log_experiment({
        "experiment_id": "node_b_phase_a_canonical_baseline",
        "hypothesis": "The production calibration entry point on the provisional 108-quote grid reproduces the committed pattern: near-exact clean recovery for informed starts but start sensitivity, and severe parameter instability at 1% noise.",
        "code": "scripts/run_node_b_phase_a_baseline.py",
        "configuration": {
            "entry_point": "src.calibrate_double_heston.calibrate_double_heston",
            "grid": "full108 (9 log-moneyness x 6 maturities x calls+puts)",
            "node_count": toolkit.NODE_COUNT,
            "spot": toolkit.SPOT,
            "carry": {"r": toolkit.RISK_FREE_RATE, "q": toolkit.DIVIDEND_YIELD},
            "optimizer": "scipy least_squares TRF, max_nfev=300, ftol=xtol=gtol=1e-10, diff_step=2e-5",
            "starts": "3 canonical deterministic starts (neutral/broad/disclosed perturbation)",
        },
        "random_seed": PHASE_A_SEED,
        "sample_size": {"cases": len(cases), "noise_levels": len(NOISE_LEVELS), "starts": int(len(combined))},
        "runtime_seconds": runtime,
        "result": summary,
        "failure_status": "completed",
    })


if __name__ == "__main__":
    main()
