"""Tiny smoke run for the G2 R2/R3 harness (pre-matrix gate).

- truths: case_1, case_2
- representations: R2, R3
- noise: 0% and 0.5%
- starts: first 2 of the frozen schedule
- market-support construction: 2026-07-15 (full five-date audit is checkpoint C)

Writes evidence/g2_r2_r3_20260822/smoke.json for manual inspection.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.g2_r2r3 import frozen  # noqa: E402
from src.g2_r2r3.calibration import fit_cell  # noqa: E402
from src.g2_r2r3.geometry import profile_for_truth  # noqa: E402
from src.g2_r2r3.market import audit_date, date_profiles  # noqa: E402
from src.g2_r2r3.starts import start_schedule_for_cell  # noqa: E402
from src.g2_r2r3.truths import truth_panel  # noqa: E402

EVIDENCE_ROOT = REPOSITORY_ROOT / "evidence" / "g2_r2_r3_20260822"
SMOKE_PROFILE_DATE = "2026-07-15"


def run_smoke() -> dict:
    begun = time.perf_counter()
    panel = truth_panel()
    market_report = audit_date(SMOKE_PROFILE_DATE)
    slot_table = market_report.pop("slot_table")
    # Audit all five dates once so truth->profile assignment matches the matrix.
    all_reports = []
    for date_id in frozen.MARKET_DATES:
        report = audit_date(date_id)
        report.pop("slot_table", None)
        all_reports.append(report)
    profiles = date_profiles(all_reports)

    runs = []
    for truth_index, truth_id in ((0, "case_1"), (1, "case_2")):
        truth = frozen.STANDING_TRUTH_VECTORS[truth_id]
        profile = profile_for_truth(truth_index, profiles)
        for representation in frozen.REPRESENTATIONS:
            for noise_index, level in ((0, 0.0), (1, 0.005)):
                schedule = start_schedule_for_cell(truth_index, noise_index)[:2]
                rows = fit_cell(
                    truth_id, truth_index, truth, representation,
                    noise_index, level, profile, schedule,
                )
                runs.extend(rows)
    return {
        "smoke_id": "G2_R2_R3_SMOKE",
        "truths": ["case_1", "case_2"],
        "representations": list(frozen.REPRESENTATIONS),
        "noise_levels": [0.0, 0.005],
        "starts_used": 2,
        "market_construction_date": SMOKE_PROFILE_DATE,
        "market_summary": {
            key: market_report[key]
            for key in (
                "usable_slots", "mask_count", "mask_rate", "r2_usable",
                "r3_usable", "r2_completeness", "r3_completeness",
            )
        },
        "slot_table_head": slot_table.head(12).to_dict(orient="records"),
        "run_count": len(runs),
        "clean_best_repricing_by_cell": {
            f"{row['truth_id']}|{row['representation']}|{row['noise_level']}": row["repricing_rmse"]
            for row in runs
            if row["start_index"] == 0
        },
        "all_runs_have_diagnostics": all(
            {"parameter_rmse_scaled", "boundary_reasons", "runtime_seconds"} <= set(row)
            for row in runs
        ),
        "runtime_seconds": time.perf_counter() - begun,
    }


if __name__ == "__main__":
    result = run_smoke()
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_ROOT / "smoke.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=float) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=float))
