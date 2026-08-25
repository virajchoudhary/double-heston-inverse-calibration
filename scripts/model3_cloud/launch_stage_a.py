"""Launch the exact frozen Stage-A command after a passing CUDA preflight."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from preflight import build_report


REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "model3_pde_development_pilot",
    )
    parser.add_argument(
        "--preflight-output",
        type=Path,
        default=None,
        help="optional preflight JSON path outside the run directory",
    )
    parser.add_argument("--expected-git-sha", required=True)
    arguments = parser.parse_args(argv)
    output_root = arguments.output_root.resolve()
    preflight_path = (
        arguments.preflight_output.resolve()
        if arguments.preflight_output is not None
        else output_root.with_name(output_root.name + ".cloud_preflight.json")
    )
    report = build_report(
        require_cuda=True,
        expected_git_sha=arguments.expected_git_sha,
    )
    if report["status"] != "PASS":
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit("Stage-A launch blocked by preflight")

    preflight_path.parent.mkdir(parents=True, exist_ok=True)
    preflight_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    command = [
        sys.executable,
        "scripts/run_model3_pde_pilot.py",
        "--dataset",
        "data/final_r2_clean_10000/surfaces.jsonl",
        "--output-root",
        output_root.as_posix(),
        "--train-limit",
        "240",
        "--validation-limit",
        "40",
        "--seed",
        "4207",
        "--epochs",
        "3",
        "--batch-size",
        "16",
        "--interior-points",
        "16",
        "--terminal-points",
        "8",
        "--learning-rate",
        "0.0002",
        "--weight-decay",
        "0.00001",
        "--device",
        "cuda",
    ]
    started = time.time()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [REPO_ROOT.as_posix(), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    console_path = output_root.with_name(output_root.name + ".launch_console.log")
    with console_path.open("x", encoding="utf-8", newline="") as console_handle:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            check=False,
            stdout=console_handle,
            stderr=subprocess.STDOUT,
        )
    finished = time.time()
    transcript = {
        "command": command,
        "return_code": completed.returncode,
        "started_unix_seconds": started,
        "finished_unix_seconds": finished,
        "duration_seconds": finished - started,
        "output_root": output_root.as_posix(),
        "console_log": console_path.as_posix(),
    }
    transcript_path = output_root.with_name(output_root.name + ".launch_transcript.json")
    transcript_path.write_text(
        json.dumps(transcript, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
