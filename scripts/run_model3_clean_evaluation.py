"""Future authorized clean-test runner; remains sealed without explicit flags."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model3_evaluation.contracts import REQUIRED_SEEDS
from src.model3_evaluation.harness import run_clean_evaluation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--authorize-frozen-test-evaluation",
        action="store_true",
        help="explicit second lock required only after all three seeds are verified",
    )
    parser.add_argument(
        "--exact-command",
        required=True,
        help="operator-provided portable command recorded byte-for-byte in evidence",
    )
    for seed in REQUIRED_SEEDS:
        parser.add_argument(f"--checkpoint-root-{seed}", type=Path, required=True)
    arguments = parser.parse_args(argv)
    roots = {seed: getattr(arguments, f"checkpoint_root_{seed}") for seed in REQUIRED_SEEDS}
    try:
        manifest = run_clean_evaluation(
            freeze_manifest_path=arguments.freeze_manifest,
            checkpoint_roots=roots,
            output_root=arguments.output_root,
            exact_command=arguments.exact_command,
            authorize_frozen_test_evaluation=arguments.authorize_frozen_test_evaluation,
            device=arguments.device,
        )
    except Exception as error:
        print(json.dumps({"status": "FAILED_CLOSED", "error": str(error)}), file=sys.stderr)
        return 1
    print(json.dumps({
        "status": manifest["completion_state"],
        "output_root": str(arguments.output_root),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
