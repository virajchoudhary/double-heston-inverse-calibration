"""Generate the deterministic CALL dataset for Mentor Double Heston PINN V1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mentor_dh_pinn.config import load_baseline_config
from src.mentor_dh_pinn.synthetic_data import generate_synthetic_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/mentor_dh_pinn_baseline_v1"),
    )
    parser.add_argument("--train-count", type=int, default=None)
    parser.add_argument("--validation-count", type=int, default=None)
    parser.add_argument("--test-count", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_baseline_config(args.config)
    dataset = generate_synthetic_dataset(
        args.output_dir,
        config=config,
        train_count=args.train_count,
        validation_count=args.validation_count,
        test_count=args.test_count,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "point_count": dataset.size,
                "counts": dataset.manifest["counts"],
                "dataset_sha256": dataset.manifest.get("dataset_sha256"),
                "split_id_hashes": dataset.split_id_hashes(),
                "frozen_surfaces_sha256": dataset.parameter_source.dataset_sha256,
                "parameter_hash": dataset.parameter_source.parameter_hash,
                "surface_id": dataset.parameter_source.surface_id,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
