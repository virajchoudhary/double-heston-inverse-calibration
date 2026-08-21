"""Validate the Node B vectorized diagnostic pricer against the frozen production engine.

Compares per-quote prices on the full 108-quote provisional grid and on the
committed G2 central-5 market geometry (per-maturity carry) for the four
representative truth vectors, and records single-surface runtimes.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import scripts.node_b_toolkit as toolkit
import scripts.run_g2_identifiability_analysis as baseline
from src.double_heston import price_double_heston_surface


def main() -> None:
    cases = toolkit.representative_cases()
    full = toolkit.full_108_geometry()
    bounds = toolkit.load_hard_safety_bounds(toolkit.BOUNDS_PATH)

    rows = []
    for row in cases.itertuples():
        truth = toolkit.case_vector(row)

        # Full 108 grid, constant carry.
        strikes, maturities, options, rates, dividends = full.build()
        canonical_prices = price_double_heston_surface(
            toolkit.SPOT,
            strikes,
            maturities,
            toolkit.RISK_FREE_RATE,
            toolkit.DIVIDEND_YIELD,
            options,
            truth,
            node_count=toolkit.NODE_COUNT,
        )
        fast_prices = toolkit.price_surface_fast(
            truth, strikes, maturities, options
        )
        full_max_diff = float(np.max(np.abs(canonical_prices - fast_prices)))

        # G2 central-5 market geometry, per-maturity carry, via normalized observables.
        profile_id, maturity_days = baseline.MATURITY_PROFILES[
            int(row.case_index) % len(baseline.MATURITY_PROFILES)
        ]
        representation = baseline.REPRESENTATIONS[0]
        canonical_norm = baseline.normalized_observables(
            truth, representation, maturity_days, node_count=toolkit.NODE_COUNT
        )
        geometry = toolkit.central5_market_geometry(tuple(maturity_days))
        fast_norm = toolkit.normalized_observables_fast(truth, geometry)
        c5_max_diff = float(np.max(np.abs(canonical_norm - fast_norm)))
        rows.append(
            {
                "case_id": row.case_id,
                "distribution": row.distribution,
                "maturity_profile": profile_id,
                "full108_max_abs_price_diff": full_max_diff,
                "central5_max_abs_normalized_diff": c5_max_diff,
                "full108_price_scale": float(np.max(canonical_prices)),
            }
        )

    # Timing on the full 108 grid.
    truth = toolkit.case_vector(cases.iloc[0])
    strikes, maturities, options, _, _ = full.build()
    started = time.perf_counter()
    for _ in range(5):
        price_double_heston_surface(
            toolkit.SPOT, strikes, maturities,
            toolkit.RISK_FREE_RATE, toolkit.DIVIDEND_YIELD, options, truth,
            node_count=toolkit.NODE_COUNT,
        )
    canonical_seconds = (time.perf_counter() - started) / 5
    started = time.perf_counter()
    for _ in range(200):
        toolkit.price_surface_fast(truth, strikes, maturities, options)
    fast_seconds = (time.perf_counter() - started) / 200

    summary = {
        "rows": rows,
        "max_abs_diff_any_case": max(
            max(r["full108_max_abs_price_diff"], r["central5_max_abs_normalized_diff"])
            for r in rows
        ),
        "canonical_surface_seconds": canonical_seconds,
        "fast_surface_seconds": fast_seconds,
        "speedup": canonical_seconds / fast_seconds,
    }
    print(np.array2string(np.array([summary["max_abs_diff_any_case"]])))
    for row in rows:
        print(row)
    print(f"canonical={canonical_seconds*1000:.2f} ms  fast={fast_seconds*1000:.3f} ms  speedup={summary['speedup']:.1f}x")

    toolkit.log_experiment({
        "experiment_id": "node_b_fast_pricer_validation",
        "hypothesis": "Vectorized broadcast re-implementation reproduces the frozen production Gauss-Laguerre pricer to near machine precision on both the full 108 grid and the G2 central-5 geometry.",
        "code": "scripts/run_node_b_validate_fast_pricer.py",
        "configuration": {"node_count": toolkit.NODE_COUNT, "spot": toolkit.SPOT},
        "sample_size": {"cases": len(rows), "quotes_full108": full.quote_count},
        "result": summary,
        "interpretation": (
            "PASS if max abs diff is at numerical-noise level (<1e-10 relative); "
            "the diagnostic pricer may then substitute for the production loop in "
            "high-volume diagnostics while the production pricer stays canonical."
        ),
        "failure_status": "pending_review",
    })


if __name__ == "__main__":
    main()
