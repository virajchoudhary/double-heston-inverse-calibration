#!/usr/bin/env python3
"""Default-safe G8 readiness runner; fixture replay is the only local mode."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.g8_readiness.pipeline import (
    CurrentDateAcquisitionLocked,
    FinalEvaluationLocked,
    assert_final_evaluation_gate,
    assert_future_acquisition_gate,
    run_synthetic_end_to_end_replay,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run-fixture", action="store_true")
    authorization = parser.add_mutually_exclusive_group()
    authorization.add_argument("--authorize-g8-acquisition", action="store_true")
    authorization.add_argument("--authorize-g8-final-evaluation", action="store_true")
    parser.add_argument("--valuation-date")
    args = parser.parse_args()

    if args.dry_run_fixture:
        payload = run_synthetic_end_to_end_replay(output_root=Path("outputs"))
        print(json.dumps(payload, sort_keys=True, indent=2))
        return 0

    if args.authorize_g8_acquisition:
        if not args.valuation_date:
            parser.error("--authorize-g8-acquisition requires --valuation-date")
        try:
            assert_future_acquisition_gate(
                authorize_g8_acquisition=True,
                valuation_date=args.valuation_date,
                current_date=date.today(),
            )
        except CurrentDateAcquisitionLocked as exc:
            print(json.dumps({"gate": "ACQUISITION_BLOCKED_CURRENT_DATE", "reason": str(exc)}, sort_keys=True))
            return 2
        print(json.dumps({"gate": "ACQUISITION_CALENDAR_PASSED_PREREQUISITE_CHECKS_STILL_REQUIRED"}, sort_keys=True))
        return 0

    if args.authorize_g8_final_evaluation:
        try:
            assert_final_evaluation_gate(
                authorize_g8_final_evaluation=True,
                selected_data_manifest={},
                current_date=date.today(),
            )
        except FinalEvaluationLocked as exc:
            print(json.dumps({"gate": "FINAL_EVALUATION_LOCKED", "reason": str(exc)}, sort_keys=True))
            return 2

    print(json.dumps({"mode": "DEFAULT_SAFE_READINESS_ONLY", "acquired_real_data": False, "final_evaluation_started": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
