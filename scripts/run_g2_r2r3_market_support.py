"""Market-support audit across all five NTPC development dates (checkpoint C).

Reproduces the existing official-NSE support/activity/quote-selection contract
on every date, reports the predeclared per-date and aggregate diagnostics,
and writes the per-rank rate/carry conditioning profiles that the synthetic
matrix consumes.  Failures are recorded, never imputed.
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

from src.g2_r2r3 import frozen  # noqa: E402
from src.g2_r2r3.market import audit_all_dates, date_profiles  # noqa: E402

EVIDENCE_ROOT = REPOSITORY_ROOT / "evidence" / "g2_r2_r3_20260822"


def run_audit() -> dict:
    reports, slots = audit_all_dates()
    slots.to_csv(EVIDENCE_ROOT / "market_support_slots.csv", index=False, lineterminator="\n")

    profiles = date_profiles(reports)
    profile_records = [profile.summary() for profile in profiles]
    (EVIDENCE_ROOT / "market_profiles.json").write_text(
        json.dumps(profile_records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    per_date = []
    for report in reports:
        per_date.append(
            {
                key: report[key]
                for key in (
                    "date_id", "spot", "listed_expiry_ranks", "listed_expiry_count",
                    "eligible_expiry_ranks", "expiry_details", "rate_observation_date",
                    "rate_carry_forward", "rate_simple_yield", "total_slots",
                    "usable_slots", "usable_call_slots", "usable_put_slots",
                    "mask_count", "mask_rate", "r2_slots", "r2_usable", "r3_usable",
                    "r2_completeness", "r3_completeness", "constructible",
                    "futures_support_failures", "quote_selection_failures",
                    "slot_failure_breakdown",
                )
            }
        )
    aggregate = {
        "dates": list(frozen.MARKET_DATES),
        "total_r2_slots": sum(item["r2_slots"] for item in per_date),
        "total_r2_usable": sum(item["r2_usable"] for item in per_date),
        "total_r3_usable": sum(item["r3_usable"] for item in per_date),
        "aggregate_r2_completeness": sum(
            item["r2_usable"] for item in per_date
        ) / (frozen.R2_NOMINAL_SLOTS * len(per_date)),
        "aggregate_r3_completeness": sum(
            item["r3_usable"] for item in per_date
        ) / (frozen.R3_NOMINAL_SLOTS * len(per_date)),
        "dates_constructible": sum(1 for item in per_date if item["constructible"]),
        "dte_distribution": sorted(
            {
                detail["dte"]
                for item in per_date
                for detail in item["expiry_details"]
            }
        ),
        "rate_carry_forward_dates": [
            item["date_id"] for item in per_date if item["rate_carry_forward"]
        ],
    }
    summary = {
        "analysis_id": "G2_R2_R3_MARKET_SUPPORT_AUDIT",
        "contract": (
            "existing official-NSE UDiFF support/activity/quote-selection contract "
            "(scripts/run_ntpc_dh_multi_date_calibration.py), extended to three "
            "expiry ranks and five development dates; failures recorded not imputed"
        ),
        "rate_provenance": {
            "validated_observations": frozen.RATE_OBSERVATIONS,
            "source_by_valuation": frozen.RATE_SOURCE_BY_VALUATION,
            "note": (
                "07-08 and 07-29 use the committed carry-forward convention (latest "
                "preserved hash-sealed observation on or before the valuation date); "
                "no preserved RBI auction artifact exists for those dates and "
                "nothing was fabricated or newly acquired"
            ),
        },
        "per_date": per_date,
        "aggregate": aggregate,
        "all_dates_remain_development_excluded_from_g8": True,
    }
    (EVIDENCE_ROOT / "market_support_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary


if __name__ == "__main__":
    result = run_audit()
    for item in result["per_date"]:
        print(
            f"{item['date_id']}: spot={item['spot']:.2f} ranks={item['eligible_expiry_ranks']} "
            f"R2 {item['r2_usable']}/{frozen.R2_NOMINAL_SLOTS} "
            f"R3 {item['r3_usable']}/{frozen.R3_NOMINAL_SLOTS} "
            f"mask_rate={item['mask_rate']:.3f} constructible={item['constructible']}"
        )
    print(json.dumps(result["aggregate"], indent=2))
