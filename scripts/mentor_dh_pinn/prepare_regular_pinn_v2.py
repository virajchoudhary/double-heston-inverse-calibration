#!/usr/bin/env python3
"""Generate the frozen DOUBLE_DATA_V2_REPRODUCIBLE contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.mentor_dh_pinn.reproducible_regular_pinn_data import DEFAULT_SEED, generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--train", type=int, default=131072)
    parser.add_argument("--validation", type=int, default=16384)
    parser.add_argument("--collocation", type=int, default=18000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--allow-runtime-mismatch", action="store_true")
    args = parser.parse_args()
    manifest = generate_dataset(
        args.out,
        train_count=args.train,
        validation_count=args.validation,
        collocation_count=args.collocation,
        seed=args.seed,
        enforce_runtime=not args.allow_runtime_mismatch,
    )
    print(json.dumps({
        "contract_id": manifest["contract_id"],
        "splits": manifest["splits"],
        "artifact_sha256": {name: record["sha256"] for name, record in manifest["artifacts"].items()},
        "manifest_sha256": manifest["manifest_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()

