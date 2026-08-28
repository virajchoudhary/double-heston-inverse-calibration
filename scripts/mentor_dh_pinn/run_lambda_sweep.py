"""Run the predeclared validation-only mentor PINN lambda sweep."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mentor_dh_pinn.lambda_sweep import DEFAULT_SWEEP_CONFIG, run_lambda_sweep


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_SWEEP_CONFIG)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--run-id", action="append", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = run_lambda_sweep(
        args.config,
        repo_root=REPO_ROOT,
        device=args.device,
        selected_run_ids=args.run_id,
    )
    print(json.dumps({"run_results": [str(path) for path in paths]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
