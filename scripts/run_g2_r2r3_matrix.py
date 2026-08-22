"""Full G2 R2/R3 synthetic calibration matrix (checkpoints D-G).

20 truths x 2 representations x 4 noise levels x 12 deterministic starts
= 1,920 calibration attempts, run only after HARNESS_READY.  Results append
incrementally to evidence/g2_r2_r3_20260822/synthetic_runs.jsonl (resume-safe);
nothing is kept only in memory.  Failures and boundary hits are retained.

Optional sharding (--shard i --shards N) splits cells by truth_index modulo N
into per-shard JSONL files.  Every cell is deterministic and keyed by cell
identity (never by execution order), so shard outputs merge losslessly and
bit-identically into the single canonical run log.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.constants import PARAMETER_NAMES  # noqa: E402
from src.g2_r2r3 import frozen  # noqa: E402
from src.g2_r2r3.calibration import ResultLog, fit_cell  # noqa: E402
from src.g2_r2r3.geometry import DateProfile, profile_for_truth  # noqa: E402
from src.g2_r2r3.starts import start_schedule_for_cell  # noqa: E402
from src.g2_r2r3.truths import truth_panel  # noqa: E402

EVIDENCE_ROOT = REPOSITORY_ROOT / "evidence" / "g2_r2_r3_20260822"
RUN_LOG = EVIDENCE_ROOT / "synthetic_runs.jsonl"
PROFILES_PATH = EVIDENCE_ROOT / "market_profiles.json"


def load_profiles() -> list[DateProfile]:
    records = json.loads(PROFILES_PATH.read_text(encoding="utf-8"))
    return [
        DateProfile(
            date_id=record["date_id"],
            spot=float(record["spot"]),
            expiry_dates=tuple(record["expiry_dates"]),
            dte=tuple(int(value) for value in record["dte"]),
            rates=tuple(float(value) for value in record["rates"]),
            carries=tuple(float(value) for value in record["carries"]),
        )
        for record in records
    ]


def run_matrix(shard: int = 0, shards: int = 1, log_path: Path = RUN_LOG) -> None:
    panel = truth_panel()
    profiles = load_profiles()
    log = ResultLog(log_path)
    total_cells = len(panel) * len(frozen.REPRESENTATIONS) * len(frozen.NOISE_LEVELS)
    cell_index = 0
    matrix_started = time.perf_counter()
    for row in panel.itertuples():
        truth_id = row.truth_id
        truth_index = int(row.truth_index)
        if truth_index % shards != shard:
            continue
        truth_vector = [getattr(row, name) for name in PARAMETER_NAMES]
        profile = profile_for_truth(truth_index, profiles)
        for representation in frozen.REPRESENTATIONS:
            for noise_index, level in enumerate(frozen.NOISE_LEVELS):
                cell_index += 1
                if log.cell_complete(truth_id, representation, noise_index):
                    continue
                schedule = start_schedule_for_cell(truth_index, noise_index)
                rows = fit_cell(
                    truth_id,
                    truth_index,
                    truth_vector,
                    representation,
                    noise_index,
                    level,
                    profile,
                    schedule,
                )
                log.append(rows)
                elapsed = time.perf_counter() - matrix_started
                done_runs = len(log.completed)
                print(
                    f"[{done_runs}/{total_cells * frozen.START_COUNT}] "
                    f"{truth_id} {representation} noise={level:.3f} done "
                    f"({elapsed:.0f}s elapsed)",
                    flush=True,
                )
    log.append_event(
        {
            "event": "matrix_complete",
            "total_runs": len(log.completed),
            "wall_seconds": time.perf_counter() - matrix_started,
        }
    )
    print("MATRIX COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.shards > 1:
        shard_path = EVIDENCE_ROOT / f"synthetic_runs_shard{args.shard}of{args.shards}.jsonl"
    else:
        shard_path = RUN_LOG
    run_matrix(shard=args.shard, shards=args.shards, log_path=shard_path)
