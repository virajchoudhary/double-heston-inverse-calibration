"""Development-only Model 3 PDE smoke check; never a research result."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model3_pde.collocation import CollocationDomain, sample_collocation_states
from src.model3_pde.operator import double_heston_pde_residual


def main() -> int:
    parameters = torch.tensor(
        [[0.80, 0.060, 0.250, -0.40, 0.050, 3.00, 0.030, 0.350, 0.20, 0.020]],
        dtype=torch.float64,
    ).repeat(4, 1)
    state, _ = sample_collocation_states(
        CollocationDomain(
            spot_minimum=50.0,
            spot_maximum=150.0,
            variance_slow_minimum=0.01,
            variance_slow_maximum=0.15,
            variance_fast_minimum=0.005,
            variance_fast_maximum=0.10,
            maturity_minimum=7.0 / 365.0,
            maturity_maximum=180.0 / 365.0,
        ),
        point_count=4,
        seed=42,
    )
    linear_price = (
        state.spot * torch.exp(-0.015 * state.maturity)
        - 100.0 * torch.exp(-0.03 * state.maturity)
        + 0.0 * state.spot.square()
        + 0.0 * state.spot.square() * state.variance_slow
        + 0.0 * state.spot.square() * state.variance_fast
        + 0.0 * state.variance_slow.square()
        + 0.0 * state.variance_fast.square()
    )
    residual = double_heston_pde_residual(
        linear_price,
        state,
        parameters,
        risk_free_rate=torch.full_like(state.spot, 0.03),
        dividend_yield=torch.full_like(state.spot, 0.015),
    )
    report = {
        "run_kind": "DEVELOPMENT_SMOKE_NOT_RESEARCH_RESULT",
        "point_count": int(residual.numel()),
        "residual_max_abs": float(residual.detach().abs().max()),
        "all_finite": bool(torch.isfinite(residual).all()),
        "status": (
            "PASSED"
            if (
                torch.isfinite(residual).all()
                and float(residual.detach().abs().max()) <= 1e-12
            )
            else "FAILED"
        ),
    }
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
