"""Merge sharded matrix run logs into the canonical synthetic_runs.jsonl.

Validates that the merged log holds exactly the 1,920 unique predeclared run
keys (truth_id, representation, noise_index, start_index) with no duplicates,
and that every record is bit-identical when re-read.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.g2_r2r3 import frozen  # noqa: E402
from src.g2_r2r3.truths import truth_panel  # noqa: E402

EVIDENCE_ROOT = REPOSITORY_ROOT / "evidence" / "g2_r2_r3_20260822"
CANONICAL = EVIDENCE_ROOT / "synthetic_runs.jsonl"
EXPECTED_TOTAL = 20 * 2 * 4 * frozen.START_COUNT

# Duplicate-key conflict comparison ignores wall-clock runtime and treats NaN
# fields as equal to NaN (JSON round-trip form), so a crash-resume duplicate of
# a deterministic cell never aborts the merge while a true divergence still does.
NON_SCIENTIFIC_FIELDS = ("runtime_seconds",)


def _canonical_form(record: dict) -> str:
    trimmed = {
        key: value
        for key, value in record.items()
        if key not in NON_SCIENTIFIC_FIELDS
    }
    return json.dumps(trimmed, sort_keys=True, default=float)


def merge() -> dict:
    records: dict[tuple[str, str, int, int], dict] = {}
    # Shard files only: the canonical output is written fresh below.  A stale
    # single-threaded partial log (if present) double-covers cells recomputed
    # bit-identically by shard 0, differing only in wall-clock runtime; it is
    # preserved under tmp/ before the canonical file is rewritten.
    if CANONICAL.is_file():
        preserve = (
            REPOSITORY_ROOT / "tmp" / "synthetic_runs_single_threaded_partial.jsonl"
        )
        preserve.parent.mkdir(parents=True, exist_ok=True)
        preserved = preserve.read_text(encoding="utf-8") if preserve.is_file() else ""
        current = CANONICAL.read_text(encoding="utf-8")
        if current and current != preserved:
            with preserve.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(current if current.endswith("\n") else current + "\n")
    sources = sorted(EVIDENCE_ROOT.glob("synthetic_runs_shard*.jsonl"))
    for source in sources:
        if not source.is_file():
            continue
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("record_type") != "run":
                continue
            key = (
                record["truth_id"],
                record["representation"],
                record["noise_index"],
                record["start_index"],
            )
            if key in records and _canonical_form(records[key]) != _canonical_form(record):
                raise RuntimeError(f"conflicting records for {key}")
            records[key] = record
    expected_keys = set()
    panel = truth_panel()
    for row in panel.itertuples():
        for representation in frozen.REPRESENTATIONS:
            for noise_index in range(len(frozen.NOISE_LEVELS)):
                for start_index in range(frozen.START_COUNT):
                    expected_keys.add(
                        (row.truth_id, representation, noise_index, start_index)
                    )
    missing = expected_keys - set(records)
    unexpected = set(records) - expected_keys
    if missing or unexpected or len(records) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"merge incomplete: {len(records)} records, "
            f"{len(missing)} missing, {len(unexpected)} unexpected"
        )
    with CANONICAL.open("w", encoding="utf-8", newline="\n") as handle:
        for key in sorted(records):
            handle.write(json.dumps(records[key], default=float) + "\n")
    return {
        "merged_records": len(records),
        "sources": [source.name for source in sources],
        "complete": True,
    }


if __name__ == "__main__":
    print(json.dumps(merge(), indent=2))
