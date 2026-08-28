"""Evaluate a selected V1 checkpoint on the sealed test split exactly once."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mentor_dh_pinn.config import load_baseline_config
from src.mentor_dh_pinn.evaluation import evaluate_test_once
from src.mentor_dh_pinn.synthetic_data import load_synthetic_dataset
from src.mentor_dh_pinn.trainer import load_checkpoint_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/mentor_dh_pinn_baseline_v1/checkpoint.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/mentor_dh_pinn_baseline_v1"),
    )
    parser.add_argument("--config", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_baseline_config(args.config) if args.config else load_checkpoint_config(args.checkpoint)
    dataset = load_synthetic_dataset(args.output_dir, config=config)
    result = evaluate_test_once(
        args.checkpoint,
        dataset,
        args.output_dir,
        config=config,
    )
    print(json.dumps(result.metrics, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
