"""Fail-closed Model 3 cloud execution identity and GPU preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
INHERITED_GIT_SHA = "a01ddc1db854f823eb02b91193eecb4dc6698974"
EXPECTED_BRANCH = "research/model3-stage-a"
PROTOCOL_CONFIG = REPO_ROOT / "configs" / "model3_pde_protocol.yaml"
DATASET = REPO_ROOT / "data" / "final_r2_clean_10000" / "surfaces.jsonl"
EXPECTED_CONFIG_SHA256 = (
    "d38482381bd3021baff80333b40a0770941a79d80fd5e0da3b4bc314a4f10361"
)
EXPECTED_DATASET_SHA256 = (
    "148b579a4f6ce572e34796e872479c4c016c89bbcd20438c2bb62d6b6960f1f6"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower()


def run_git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def build_report(require_cuda: bool, *, expected_git_sha: str) -> dict[str, Any]:
    failures: list[str] = []
    try:
        git_sha = run_git("rev-parse", "HEAD")
        branch = run_git("branch", "--show-current")
        status_lines = run_git(
            "status", "--porcelain", "--untracked-files=no"
        ).splitlines()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Git identity checks failed") from error

    if len(expected_git_sha) != 40 or any(
        character not in "0123456789abcdef" for character in expected_git_sha.lower()
    ):
        failures.append("expected_git_sha_is_not_forty_hex_characters")
    if git_sha != expected_git_sha.lower():
        failures.append(f"git_sha: {git_sha}")
    if branch != EXPECTED_BRANCH:
        failures.append(f"branch: {branch}")
    if status_lines:
        failures.append(f"tracked_tree_dirty: {status_lines}")
    if not PROTOCOL_CONFIG.is_file():
        failures.append(f"missing_config: {PROTOCOL_CONFIG}")
    if not DATASET.is_file():
        failures.append(f"missing_dataset: {DATASET}")

    config_hash = sha256_file(PROTOCOL_CONFIG) if PROTOCOL_CONFIG.is_file() else None
    dataset_hash = sha256_file(DATASET) if DATASET.is_file() else None
    if config_hash != EXPECTED_CONFIG_SHA256:
        failures.append(f"config_sha256: {config_hash}")
    if dataset_hash != EXPECTED_DATASET_SHA256:
        failures.append(f"dataset_sha256: {dataset_hash}")

    report: dict[str, Any] = {
        "purpose": "MODEL3_CLOUD_PREFLIGHT_NOT_RESEARCH_EVIDENCE",
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "expected_git_sha": expected_git_sha.lower(),
        "git_sha": git_sha,
        "expected_branch": EXPECTED_BRANCH,
        "branch": branch,
        "tracked_git_status": status_lines,
        "expected_config_sha256": EXPECTED_CONFIG_SHA256,
        "config_sha256": config_hash,
        "expected_dataset_sha256": EXPECTED_DATASET_SHA256,
        "dataset_sha256": dataset_hash,
    }

    try:
        import numpy as np
        import pandas as pd
        import torch
        import yaml
    except Exception as error:
        failures.append(f"dependencies: {type(error).__name__}: {error}")
    else:
        cuda_available = bool(torch.cuda.is_available())
        device_name = None
        compute_capability = None
        total_memory_bytes = None
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            compute_capability = f"{major}.{minor}"
            total_memory_bytes = int(torch.cuda.get_device_properties(0).total_memory)
        elif require_cuda:
            failures.append("cuda_not_available")
        report.update(
            {
                "numpy_version": np.__version__,
                "pandas_version": pd.__version__,
                "torch_version": torch.__version__,
                "torch_cuda_build": str(torch.version.cuda),
                "cuda_available": cuda_available,
                "cuda_device_name": device_name,
                "cuda_compute_capability": compute_capability,
                "cuda_total_memory_bytes": total_memory_bytes,
                "yaml_version": yaml.__version__,
            }
        )

    report["failures"] = failures
    report["status"] = "PASS" if not failures else "FAIL"
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--output", type=Path, help="optional JSON copy; never overwrite")
    arguments = parser.parse_args(argv)
    report = build_report(
        require_cuda=arguments.require_cuda,
        expected_git_sha=arguments.expected_git_sha,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if arguments.output is not None:
        if arguments.output.exists():
            raise FileExistsError(f"refusing to overwrite preflight evidence: {arguments.output}")
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"PREFLIGHT_FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2) from error
