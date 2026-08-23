"""Frozen traditional-calibration execution on the R2 test split.

Runs the existing canonical calibration module
(``src/calibrate_double_heston.py``) with the EXACT frozen settings from
``configs/r2_primary_comparison_FINAL.yaml`` section TRADITIONAL_CALIBRATION
on the 1,250-surface test split, parallelized across surfaces (wall time
only; the per-surface budget is unchanged).  Every start outcome — including
failures — is retained.  The frozen representative-selection rule (lowest
final objective; ties by lowest start index) is applied only at evaluation
time, never during execution.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..calibrate_double_heston import calibrate_double_heston
from ..utils import write_json
from .dataset import R2PrimaryDataset, R2SurfaceItem

# Frozen protocol settings (mirror of r2_primary_comparison_FINAL.yaml
# traditional_calibration section; consistency asserted by tests).
FROZEN_SETTINGS: dict[str, Any] = {
    "module": "src/calibrate_double_heston.py",
    "pricer": "production pricer src/double_heston.py (unchanged)",
    "node_count": 64,
    "optimizer": "trf",
    "ftol": 1.0e-10,
    "xtol": 1.0e-10,
    "gtol": 1.0e-10,
    "diff_step": 2.0e-5,
    "max_nfev": 300,
    "start_seed": 42,
    "start_count": 3,
    "bounds_path": "configs/parameter_bounds_PROVISIONAL.yaml",
    "residual_scale": "max(observed_dollar_price, 1.0)",
    "representative_rule": "lowest final objective; ties by lowest start_index",
}


def calibrate_surface(item_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Calibrate ONE surface under the frozen settings (worker process)."""
    item: R2SurfaceItem = item_payload["item"]
    started = time.perf_counter()
    frame = calibrate_double_heston(
        spot=item.spot,
        strikes=item.strikes,
        maturities=item.maturities,
        risk_free_rate=item.rate,
        dividend_yield=item.carry,
        option_types=item.option_types,
        observed_prices=item.dollar_prices,
        known_parameters=item.targets,
        bounds_path=FROZEN_SETTINGS["bounds_path"],
        node_count=FROZEN_SETTINGS["node_count"],
        max_nfev=FROZEN_SETTINGS["max_nfev"],
        seed=FROZEN_SETTINGS["start_seed"],
    )
    wall_seconds = time.perf_counter() - started
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        record["surface_id"] = item.surface_id
        record["wall_seconds_all_starts"] = wall_seconds
        rows.append(record)
    return rows


def run_traditional_calibration(
    dataset: R2PrimaryDataset,
    output_csv: str | Path,
    *,
    workers: int = 10,
    limit: int | None = None,
    split: str = "test",
) -> pd.DataFrame:
    """Execute the frozen calibration on every surface of ``split``."""
    indices = dataset.indices_for_split(split)
    if limit is not None:
        indices = indices[:limit]
    payloads = [
        {"item": dataset.items[index], "index": position}
        for position, index in enumerate(indices)
    ]
    all_rows: list[dict[str, Any]] = []
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(calibrate_surface, payload): payload for payload in payloads
        }
        completed = 0
        for future in as_completed(futures):
            all_rows.extend(future.result())
            completed += 1
            if completed % 50 == 0 or completed == len(payloads):
                print(f"calibrated {completed}/{len(payloads)} surfaces", flush=True)
    frame = pd.DataFrame(all_rows)
    frame.to_csv(output_path, index=False)
    return frame


def select_representatives(starts_frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen representative-selection rule per surface.

    Lowest final objective (mean squared residual, column ``loss``) among all
    recorded starts of the surface, regardless of optimizer success flag;
    ties broken by lowest start_index.  Failed starts with NaN loss can never
    be selected (NaN compares false).
    """
    if "loss" not in starts_frame or "surface_id" not in starts_frame:
        raise ValueError("starts frame must carry surface_id and loss")
    representatives: list[pd.Series] = []
    for _, group in starts_frame.groupby("surface_id", sort=True):
        candidates = group.dropna(subset=["loss"])
        if candidates.empty:
            representatives.append(group.sort_values("start_index").iloc[0])
        else:
            ordered = candidates.sort_values(
                ["loss", "start_index"], ascending=[True, True]
            )
            representatives.append(ordered.iloc[0])
    return pd.DataFrame(representatives).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/final_r2_clean_10000/surfaces.jsonl")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evidence/r2_primary_comparison_20260823/traditional_calibration_starts.csv"),
    )
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument(
        "--limit", type=int, default=None, help="development subset sizing only"
    )
    parser.add_argument(
        "--smoke", action="store_true", help="DEVELOPMENT_SMOKE_NOT_RESEARCH_RESULT run"
    )
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    if args.smoke:
        print("DEVELOPMENT_SMOKE_NOT_RESEARCH_RESULT")
    dataset = R2PrimaryDataset.from_jsonl(args.dataset)
    frame = run_traditional_calibration(
        dataset, args.output, workers=args.workers, limit=args.limit, split=args.split
    )
    representatives = select_representatives(frame)
    summary = {
        "run_kind": (
            "DEVELOPMENT_SMOKE_NOT_RESEARCH_RESULT"
            if args.smoke
            else "RESEARCH"
        ),
        "frozen_settings": FROZEN_SETTINGS,
        "split": args.split,
        "surfaces_calibrated": int(frame["surface_id"].nunique()),
        "starts_recorded": int(len(frame)),
        "starts_flagged_success": int(frame["success"].sum()),
        "starts_failed": int((~frame["success"].astype(bool)).sum()),
        "total_wall_seconds": float(frame["wall_seconds_all_starts"].sum()),
        "output_csv": str(args.output),
    }
    write_json(
        Path(args.output).with_name(Path(args.output).stem + "_summary.json"), summary
    )
    write_json(
        Path(args.output).with_name(
            Path(args.output).stem + "_representatives.json"
        ),
        {
            "representative_count": int(len(representatives)),
            "rule": FROZEN_SETTINGS["representative_rule"],
        },
    )
    print(
        f"calibrated {summary['surfaces_calibrated']} surfaces "
        f"({summary['starts_recorded']} starts, {summary['starts_failed']} non-success)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
