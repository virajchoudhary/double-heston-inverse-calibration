"""Verify the three-seed Model3 Stage-B freeze gate; never authorize evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model3_evaluation.contracts import (
    CheckpointContractError,
    build_freeze_manifest,
    render_report,
    verify_freeze_manifest,
)
from src.utils import write_json


def _seed_roots(values: list[str]) -> dict[int, Path]:
    roots: dict[int, Path] = {}
    for value in values:
        try:
            raw_seed, raw_path = value.split("=", 1)
            seed = int(raw_seed)
        except ValueError as error:
            raise ValueError("--run-root must use SEED=PATH") from error
        if seed in roots:
            raise ValueError(f"duplicate seed {seed}")
        roots[seed] = Path(raw_path)
    return roots


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-manifest", type=Path)
    parser.add_argument("--run-root", action="append", default=[])
    parser.add_argument("--experiment-id")
    parser.add_argument("--expected-train-population-sha256")
    parser.add_argument("--expected-validation-population-sha256")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--json-output", type=Path)
    arguments = parser.parse_args(argv)
    try:
        if arguments.run_root:
            if arguments.experiment_id is None:
                parser.error("--experiment-id is required with --run-root")
            if bool(arguments.expected_train_population_sha256) != bool(
                arguments.expected_validation_population_sha256
            ):
                parser.error("both expected population hashes are required together")
            roots = _seed_roots(arguments.run_root)
            manifest = build_freeze_manifest(
                roots,
                experiment_id=arguments.experiment_id,
                expected_train_population_sha256=arguments.expected_train_population_sha256,
                expected_validation_population_sha256=arguments.expected_validation_population_sha256,
            )
            verification = verify_freeze_manifest(manifest)
        else:
            if arguments.experiment_id is not None or not arguments.freeze_manifest:
                parser.error("use either --freeze-manifest or exactly three --run-root values")
            import json

            payload = json.loads(arguments.freeze_manifest.read_text(encoding="utf-8"))
            verification = verify_freeze_manifest(payload)
        report = render_report(verification)
        print(report, end="")
        if arguments.report is not None:
            if arguments.report.exists():
                raise FileExistsError(f"refusing to overwrite {arguments.report}")
            arguments.report.write_text(report, encoding="utf-8")
        if arguments.json_output is not None:
            if arguments.json_output.exists():
                raise FileExistsError(f"refusing to overwrite {arguments.json_output}")
            write_json(arguments.json_output, verification)
        return 0
    except (CheckpointContractError, OSError, ValueError) as error:
        print(f"MODEL3 THREE-SEED FREEZE: BLOCKED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
