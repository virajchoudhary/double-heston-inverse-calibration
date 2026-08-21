"""Run the authorized three-date official-NSE Stage A coverage screen."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.nse_stage_a import (
    AUTHORIZED_DATES,
    DEFAULT_DERIVED_ROOT,
    DEFAULT_RAW_ROOT,
    acquire_udiff_archive,
    analyze_stage_a,
    read_prior_acquisition_evidence,
    read_udiff_csv,
    write_stage_a_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--derived-root", type=Path, default=DEFAULT_DERIVED_ROOT)
    parser.add_argument("--offline", action="store_true", help="Require local raw archives; never download.")
    args = parser.parse_args()
    prior_evidence = read_prior_acquisition_evidence(args.derived_root)
    records = []
    raw_by_date = {}
    for valuation_date in AUTHORIZED_DATES:
        records_for_date = []
        for market in ("CM", "FO"):
            if args.offline:
                archive = args.raw_root / valuation_date.isoformat() / f"BhavCopy_NSE_{market}_0_0_0_{valuation_date:%Y%m%d}_F_0000.csv.zip"
                if not archive.is_file():
                    raise FileNotFoundError(f"Offline mode requires local archive: {archive}")
            record = acquire_udiff_archive(
                market,
                valuation_date,
                args.raw_root,
                prior_evidence=prior_evidence,
                allow_download=not args.offline,
            )
            records_for_date.append(record)
        records.extend(records_for_date)
        raw_by_date[valuation_date] = {
            record.market: read_udiff_csv(record.csv_path, valuation_date, record.market) for record in records_for_date
        }
    write_stage_a_outputs(analyze_stage_a(raw_by_date, records), args.derived_root)
    print(f"Wrote eight Stage A CSVs for {len(AUTHORIZED_DATES)} authorized dates under {args.derived_root}")


if __name__ == "__main__":
    main()
