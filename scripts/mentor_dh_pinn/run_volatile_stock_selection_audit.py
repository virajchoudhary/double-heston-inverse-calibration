"""Acquire official NSE history and run the pre-model volatile-stock audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mentor_dh_pinn.volatile_stock_selection import (
    acquire_official_histories,
    audit_candidates,
    load_spec,
    propose_split,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/mentor_dh_pinn/volatile_stock_selection_v1.yaml")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    spec = load_spec(args.config)
    history, _ = acquire_official_histories(spec, offline=args.offline)
    candidates, active = audit_candidates(history, spec)
    candidates.to_csv(spec.derived_root / "candidate_audit.csv", index=False, lineterminator="\n")
    eligible = candidates.loc[candidates["classification"] == "ELIGIBLE"].sort_values(
        ["rv_3m", "percentile_rank", "symbol", "option_valuation_date"],
        ascending=[False, False, True, True], kind="stable",
    )
    result: dict[str, object] = {"classification": "VOLATILE_STOCK_PHASE3B_BLOCKED_NO_ELIGIBLE_CANDIDATE"}
    if not eligible.empty:
        selected = eligible.iloc[0]
        key = f"{selected['symbol']}|{selected['option_valuation_date'].isoformat()}"
        split = propose_split(active[key], spec)
        split.to_csv(spec.derived_root / "proposed_call_split.csv", index=False, lineterminator="\n")
        result = {
            "classification": f"VOLATILE_STOCK_PHASE3B_READY_{selected['symbol']}",
            "symbol": selected["symbol"], "option_valuation_date": selected["option_valuation_date"].isoformat(),
            "window_start": selected["window_start"].isoformat(), "window_end": selected["window_end"].isoformat(),
            "rv_3m": selected["rv_3m"], "percentile_rank": selected["percentile_rank"],
            "calibration_count": int((split["role"] == "calibration").sum()),
            "holdout_count": int((split["role"] == "holdout").sum()),
        }
    (spec.derived_root / "selection_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
