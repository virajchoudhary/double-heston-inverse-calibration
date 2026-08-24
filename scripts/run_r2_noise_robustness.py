"""Operator entrypoint for the frozen R2 observation-noise robustness study."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.r2_noise.aggregation import evaluate_traditional_levels, paired_degradation
from src.r2_noise.execution import (
    CONFIG_PATH,
    EVIDENCE_ROOT,
    load_frozen_protocol,
)
from src.r2_noise.generator import generate_cohorts
from src.r2_noise.neural_evaluation import (
    evaluate_neural_levels,
    recheck_zero_percent_gate,
)
from src.r2_noise.traditional_runner import (
    compare_zero_percent_traditional_gate,
    run_traditional_subset,
)
from src.r2_noise.validation import validate_cohorts, validate_completed_bundle


def _parse_levels(value: str) -> tuple[float, ...]:
    if value == "gate":
        return (0.0,)
    if value == "all":
        return tuple(float(level) for level in load_frozen_protocol()["noise_levels"])
    raise ValueError("--levels must be 'gate' or 'all'")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("generate-cohorts")
    neural = subparsers.add_parser("evaluate-neural")
    neural.add_argument("--levels", choices=["gate", "all"], required=True)
    neural.add_argument("--output", type=Path, default=None)
    subparsers.add_parser("recheck-neural-gate")
    traditional = subparsers.add_parser("run-traditional")
    group = traditional.add_mutually_exclusive_group(required=True)
    group.add_argument("--level")
    group.add_argument("--all-levels", action="store_true")
    traditional.add_argument("--workers", type=int, default=10)
    traditional.add_argument("--gate-check", action="store_true")
    subparsers.add_parser("aggregate")
    validator = subparsers.add_parser("validate")
    validator.add_argument("--completed", action="store_true")
    args = parser.parse_args()

    if args.command == "generate-cohorts":
        print(generate_cohorts())
    elif args.command == "recheck-neural-gate":
        print(recheck_zero_percent_gate())
    elif args.command == "evaluate-neural":
        output = args.output
        if output is None:
            output = (
                EVIDENCE_ROOT / "zero_percent_neural"
                if args.levels == "gate"
                else EVIDENCE_ROOT / "neural"
            )
        print(evaluate_neural_levels(_parse_levels(args.levels), output_root=output))
    elif args.command == "run-traditional":
        protocol = load_frozen_protocol()
        pairs = list(zip(protocol["noise_levels"], protocol["noise_level_labels"]))
        if args.all_levels:
            for level, _ in pairs:
                print(run_traditional_subset(float(level), workers=args.workers))
        else:
            matches = [
                float(level)
                for level, label in pairs
                if label == args.level
            ]
            if len(matches) != 1:
                raise ValueError(f"unknown frozen level label: {args.level}")
            print(run_traditional_subset(matches[0], workers=args.workers))
        if args.gate_check:
            print(compare_zero_percent_traditional_gate())
    elif args.command == "aggregate":
        print(evaluate_traditional_levels(tuple(float(x) for x in load_frozen_protocol()["noise_levels"])))
        print(paired_degradation())
    elif args.command == "validate":
        print(validate_completed_bundle() if args.completed else validate_cohorts())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
