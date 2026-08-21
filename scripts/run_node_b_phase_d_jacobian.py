"""Phases D + G: scaled Jacobian conditioning across provisional geometries.

Local identifiability evidence: singular spectra, practical rank, condition
numbers, weakest scaled directions, and column norms for the four
representative truth vectors (plus a small maximin interior sample on three
anchor geometries).  Also verifies the exact slow/fast factor-swap symmetry
excluded by the declared ordering constraint.
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
import scripts.run_g2_identifiability_analysis as baseline
from src.constants import CALL_OPTION, PARAMETER_NAMES, PUT_OPTION
from src.double_heston import price_double_heston_surface

ARTIFACT_DIR = toolkit.EVIDENCE_ROOT / "artifacts"
TABLE_DIR = toolkit.EVIDENCE_ROOT / "tables"
N_EXTRA_SAMPLE = 16
EXTRA_SAMPLE_SEED = 20260822


def build_geometries() -> dict[str, toolkit.Geometry]:
    def constant(moneyness, maturities, options, geometry_id):
        n = len(maturities)
        return toolkit.Geometry(
            geometry_id,
            tuple(moneyness),
            tuple(maturities),
            tuple(options),
            (toolkit.RISK_FREE_RATE,) * n,
            (toolkit.DIVIDEND_YIELD,) * n,
        )

    geometries: dict[str, toolkit.Geometry] = {
        "full108": toolkit.full_108_geometry(),
        "full108_calls": constant(
            toolkit.FULL_MONEYNESS, toolkit.FULL_MATURITY_DAYS, (CALL_OPTION,), "full108_calls"
        ),
        "full108_puts": constant(
            toolkit.FULL_MONEYNESS, toolkit.FULL_MATURITY_DAYS, (PUT_OPTION,), "full108_puts"
        ),
        "short3_full_moneyness": constant(
            toolkit.FULL_MONEYNESS, (7, 14, 30), (CALL_OPTION, PUT_OPTION), "short3_full_moneyness"
        ),
        "long3_full_moneyness": constant(
            toolkit.FULL_MONEYNESS, (60, 90, 180), (CALL_OPTION, PUT_OPTION), "long3_full_moneyness"
        ),
        "central5x6": constant(
            (-0.10, -0.05, 0.0, 0.05, 0.10),
            toolkit.FULL_MATURITY_DAYS,
            (CALL_OPTION, PUT_OPTION),
            "central5x6",
        ),
        "wings4x6": constant(
            (-0.30, -0.20, 0.20, 0.30),
            toolkit.FULL_MATURITY_DAYS,
            (CALL_OPTION, PUT_OPTION),
            "wings4x6",
        ),
        "single_7d": constant(toolkit.FULL_MONEYNESS, (7,), (CALL_OPTION, PUT_OPTION), "single_7d"),
        "single_30d": constant(toolkit.FULL_MONEYNESS, (30,), (CALL_OPTION, PUT_OPTION), "single_30d"),
        "single_90d": constant(toolkit.FULL_MONEYNESS, (90,), (CALL_OPTION, PUT_OPTION), "single_90d"),
        "single_180d": constant(toolkit.FULL_MONEYNESS, (180,), (CALL_OPTION, PUT_OPTION), "single_180d"),
        "central5_const_carry_27_55": constant(
            (-0.10, -0.05, 0.0, 0.05, 0.10), (27, 55), (CALL_OPTION, PUT_OPTION), "central5_const_carry_27_55"
        ),
    }
    geometries["central5_market_27_55"] = toolkit.central5_market_geometry((27, 55))
    geometries["central5_market_13_41"] = toolkit.central5_market_geometry((13, 41))
    geometries["central5_market_6_34"] = toolkit.central5_market_geometry((6, 34))
    return geometries


def analyse_jacobian(
    case_id: str,
    source: str,
    truth: np.ndarray,
    geometry: toolkit.Geometry,
    bounds: dict,
) -> tuple[dict, dict]:
    jacobian = toolkit.scaled_parameter_jacobian_fast(truth, geometry, bounds)
    summary = toolkit.jacobian_summary(jacobian)
    _, _, right_vectors = np.linalg.svd(jacobian, full_matrices=False)
    weakest = right_vectors[-1]
    second_weakest = right_vectors[-2]
    ranked = np.argsort(-np.abs(weakest))
    weakest_combo = ", ".join(
        f"{PARAMETER_NAMES[index]}:{weakest[index]:+.3f}" for index in ranked[:4]
    )
    column_norms = np.linalg.norm(jacobian, axis=0)
    header = {
        "case_id": case_id,
        "source": source,
        "geometry": geometry.geometry_id,
        "quotes": geometry.quote_count,
        "weakest_direction_combo": weakest_combo,
        **summary,
        **{f"column_norm_{name}": float(value) for name, value in zip(PARAMETER_NAMES, column_norms)},
        **{f"weakest_{name}": float(value) for name, value in zip(PARAMETER_NAMES, weakest)},
        **{f"second_weakest_{name}": float(value) for name, value in zip(PARAMETER_NAMES, second_weakest)},
    }
    return header, {"jacobian": jacobian, "weakest": weakest}


def factor_swap_check(truth: np.ndarray) -> dict:
    """Exact permutation symmetry: price(Theta) == price(swap_factors(Theta))."""
    swapped = truth.copy()
    swapped[0:5] = truth[5:10]
    swapped[5:10] = truth[0:5]
    strikes, maturities, options, _, _ = toolkit.full_108_geometry().build()
    base = price_double_heston_surface(
        toolkit.SPOT, strikes, maturities, toolkit.RISK_FREE_RATE, toolkit.DIVIDEND_YIELD,
        options, truth, node_count=toolkit.NODE_COUNT, enforce_ordering=False,
    )
    swapped_prices = price_double_heston_surface(
        toolkit.SPOT, strikes, maturities, toolkit.RISK_FREE_RATE, toolkit.DIVIDEND_YIELD,
        options, swapped, node_count=toolkit.NODE_COUNT, enforce_ordering=False,
    )
    return {
        "max_abs_price_difference": float(np.max(np.abs(base - swapped_prices))),
        "interpretation": (
            "Exact label-swap symmetry of the two factors; excluded from the "
            "declared parameter space only by kappa_slow < kappa_fast."
        ),
    }


def main() -> None:
    bounds = toolkit.load_hard_safety_bounds(toolkit.BOUNDS_PATH)
    geometries = build_geometries()
    cases = toolkit.representative_cases()

    # Extra maximin interior sample for distributional conditioning evidence.
    interior = pd.read_csv(
        REPOSITORY_ROOT / "outputs" / "reviewed_sampling_audit" / "interior_accepted.csv"
    )
    rng = np.random.default_rng(EXTRA_SAMPLE_SEED)
    chosen = interior.sample(n=N_EXTRA_SAMPLE, random_state=rng).reset_index(drop=True)

    rows: list[dict] = []
    jacobians: dict[tuple[str, str], np.ndarray] = {}
    started = time.perf_counter()
    for case in cases.itertuples():
        truth = toolkit.case_vector(case)
        for geometry_id, geometry in geometries.items():
            header, extras = analyse_jacobian(case.case_id, "representative", truth, geometry, bounds)
            rows.append(header)
            jacobians[(case.case_id, geometry_id)] = extras["jacobian"]
    for sample_index, row in chosen.iterrows():
        truth = np.asarray([row[name] for name in PARAMETER_NAMES], dtype=np.float64)
        for geometry_id in ("full108", "central5_market_27_55", "central5x6"):
            header, _ = analyse_jacobian(
                f"interior_sample_{sample_index:02d}", "interior_sample", truth, geometries[geometry_id], bounds
            )
            rows.append(header)

    swap_result = factor_swap_check(toolkit.case_vector(cases.iloc[0]))
    frame = pd.DataFrame(rows)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLE_DIR / "phase_d_jacobian_conditioning.csv", index=False)

    anchor = frame[frame.geometry.isin(["central5_market_27_55", "full108"])]
    pivot = (
        anchor.groupby("geometry")[
            ["condition_number", "smallest_singular_value", "practical_rank"]
        ]
        .median()
    )
    full_pivot = (
        frame[frame.source == "representative"]
        .groupby("geometry")[["condition_number", "smallest_singular_value", "practical_rank", "quotes"]]
        .median()
        .sort_values("condition_number")
    )
    full_pivot.to_csv(TABLE_DIR / "phase_d_geometry_medians.csv")
    print(full_pivot.to_string())
    print("\nSwap check:", swap_result)

    toolkit.log_experiment({
        "experiment_id": "node_b_phase_d_jacobian_conditioning",
        "hypothesis": "Local scaled-Jacobian conditioning on the full provisional 108-quote grid remains far from practical full rank; geometry subsets carry materially different information content.",
        "code": "scripts/run_node_b_phase_d_jacobian.py",
        "configuration": {
            "jacobian": "central differences, step=1e-4 * full-range width (validity-aware), spot-normalized prices, range-scaled parameters",
            "practical_rank_tolerance": toolkit.PRACTICAL_RANK_RELATIVE_TOLERANCE,
            "geometries": sorted(geometries),
        },
        "random_seed": EXTRA_SAMPLE_SEED,
        "sample_size": {
            "representative_cases": int(len(cases)),
            "interior_samples": N_EXTRA_SAMPLE,
            "jacobians_total": len(rows),
        },
        "runtime_seconds": time.perf_counter() - started,
        "result": {
            "geometry_medians": json.loads(full_pivot.to_json()),
            "factor_swap": swap_result,
        },
        "artifacts": [
            "tables/phase_d_jacobian_conditioning.csv",
            "tables/phase_d_geometry_medians.csv",
        ],
        "failure_status": "completed",
    })


if __name__ == "__main__":
    main()
