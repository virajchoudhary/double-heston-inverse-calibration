"""Purely summarize completed Phase 3A checkpoint metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mentor_dh_pinn.final_synthetic_eval import (
    DEFAULT_FINAL_EVAL_CONFIG,
    summarize_final_synthetic_eval,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_FINAL_EVAL_CONFIG)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = summarize_final_synthetic_eval(args.config, repo_root=REPO_ROOT)
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
