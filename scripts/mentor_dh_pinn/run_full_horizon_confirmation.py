"""Run the sealed Phase 2C full-horizon validation confirmation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mentor_dh_pinn.full_horizon_confirmation import (
    DEFAULT_CONFIRMATION_CONFIG,
    run_full_horizon_confirmation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIRMATION_CONFIG)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_full_horizon_confirmation(
        args.config,
        repo_root=REPO_ROOT,
        device=args.device,
    )
    print(json.dumps({"confirmation_result": str(result)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
