"""Linearized noise-propagation analysis: Jacobian predicts the noise collapse.

For each representative case on the full 108 grid, compute the scaled Jacobian
J (spot-normalized prices, range-scaled parameters) and the multiplicative-noise
covariance Sigma = diag((noise_level * normalized_price)^2).  The linearized
expected squared parameter displacement is trace(J^+ Sigma J^{+T}) (pseudo-
inverse).  Compare sqrt of that quantity against the observed median parameter
RMSE from the Phase B multi-start runs at the same noise levels.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import scripts.node_b_toolkit as toolkit

ARTIFACT_DIR = toolkit.EVIDENCE_ROOT / "artifacts"
TABLE_DIR = toolkit.EVIDENCE_ROOT / "tables"
NOISE_LEVELS = (0.005, 0.01, 0.02)


def main() -> None:
    bounds = toolkit.load_hard_safety_bounds(toolkit.BOUNDS_PATH)
    geometry = toolkit.full_108_geometry()
    cases = toolkit.representative_cases()
    observed = pd.read_csv(ARTIFACT_DIR / "phase_b_multistart_full108.csv")

    rows = []
    for case in cases.itertuples():
        truth = toolkit.case_vector(case)
        clean = toolkit.normalized_observables_fast(truth, geometry)
        jacobian = toolkit.scaled_parameter_jacobian_fast(truth, geometry, bounds)
        pinv = np.linalg.pinv(jacobian)
        for noise_level in NOISE_LEVELS:
            sigma = (noise_level * clean) ** 2
            predicted_rmse = float(np.sqrt(np.trace(pinv @ np.diag(sigma) @ pinv.T) / jacobian.shape[1]))
            mask = (
                (observed.case_id == case.case_id)
                & (np.isclose(observed.noise_level, noise_level))
                & observed.finite_solution
            )
            rows.append({
                "case_id": case.case_id,
                "noise_level": noise_level,
                "linearized_expected_parameter_rmse": predicted_rmse,
                "observed_median_parameter_rmse": float(observed[mask].parameter_rmse_full_range.median()),
                "observed_boundary_hit_fraction": float(observed[mask].bound_hit.mean()),
            })
    frame = pd.DataFrame(rows)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLE_DIR / "noise_propagation_vs_observed.csv", index=False)
    print(frame.to_string(index=False))
    toolkit.log_experiment({
        "experiment_id": "node_b_linearized_noise_propagation",
        "hypothesis": "The local scaled Jacobian already predicts O(0.1-10) full-range parameter displacement at 0.5-2% noise, matching the observed collapse; boundary saturation truncates the linearized prediction.",
        "code": "scripts/run_node_b_noise_propagation.py",
        "configuration": {"geometry": "full108", "formula": "sqrt(trace(J^+ diag(sigma) J^+T)/10)"},
        "sample_size": {"cases": int(len(cases)), "noise_levels": len(NOISE_LEVELS)},
        "result": json.loads(frame.to_json(orient="records")),
        "artifacts": ["tables/noise_propagation_vs_observed.csv"],
        "failure_status": "completed",
    })


if __name__ == "__main__":
    main()
