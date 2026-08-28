"""Summarize completed Phase 2B validation-only refinement runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mentor_dh_pinn.lambda_refinement import (
    DEFAULT_REFINEMENT_CONFIG,
    summarize_lambda_refinement,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_REFINEMENT_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = summarize_lambda_refinement(
        args.config,
        repo_root=REPO_ROOT,
        output_path=args.output,
    )
    print(json.dumps({"lambda_refinement_summary": str(path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
