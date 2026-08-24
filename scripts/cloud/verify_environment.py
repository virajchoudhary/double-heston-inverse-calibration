"""Frozen-identity and provenance verification for cloud execution.

Fails closed unless the dataset hash, protocol config hash, and protocol
status marker match the frozen primary comparison.  Writes a provenance JSON
recording the execution environment (CPU/GPU/CUDA/torch/numpy/scipy/Python).

Usage: python scripts/cloud/verify_environment.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import socket
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = REPO_ROOT / "data" / "final_r2_clean_10000" / "surfaces.jsonl"
PROTOCOL_CONFIG = REPO_ROOT / "configs" / "r2_primary_comparison_FINAL.yaml"
EVIDENCE_DIR = REPO_ROOT / "evidence" / "r2_primary_comparison_20260823"

FROZEN_DATASET_SHA256 = (
    "148b579a4f6ce572e34796e872479c4c016c89bbcd20438c2bb62d6b6960f1f6"
)
FROZEN_PROTOCOL_CONFIG_SHA256 = (
    "33ca0f763ec10bb2424eefb02448c9c8e50021854b96a948e420f44bdba70781"
)
FROZEN_PROTOCOL_STATUS = "FROZEN_BEFORE_ANY_PRIMARY_RESEARCH_TRAINING"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        ).stdout.strip()
    except Exception:
        return "UNAVAILABLE_NOT_A_GIT_CHECKOUT"


def _provenance_output(host: str) -> Path:
    output = EVIDENCE_DIR / f"cloud_provenance_{host}.json"
    if output.exists():
        raise FileExistsError(
            "refusing to overwrite cloud provenance: "
            f"{output}"
        )
    return output


def main() -> int:
    failures: list[str] = []

    dataset_hash = _sha256(DATASET)
    if dataset_hash != FROZEN_DATASET_SHA256:
        failures.append(f"dataset hash mismatch: {dataset_hash}")

    config_text = PROTOCOL_CONFIG.read_text(encoding="utf-8")
    if FROZEN_PROTOCOL_STATUS not in config_text:
        failures.append("protocol config status marker missing")
    protocol_hash = _sha256(PROTOCOL_CONFIG)
    if protocol_hash != FROZEN_PROTOCOL_CONFIG_SHA256:
        failures.append(f"protocol config hash mismatch: {protocol_hash}")

    import numpy
    import scipy
    import torch
    import yaml

    config = yaml.safe_load(config_text)
    if config.get("protocol", {}).get("status") != FROZEN_PROTOCOL_STATUS:
        failures.append("protocol status is not the frozen marker")

    cuda_available = torch.cuda.is_available()
    device_name = None
    fp64_capability = None
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        fp64_capability = f"{major}.{minor}"

    provenance = {
        "purpose": "execution_environment_provenance_only_no_scientific_change",
        "host": socket.gethostname(),
        "git_sha": _git_sha(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "logical_cores": __import__("os").cpu_count(),
        "torch_version": torch.__version__,
        "torch_cuda_built": str(torch.version.cuda),
        "cuda_available": cuda_available,
        "cuda_device_name": device_name,
        "cuda_compute_capability": fp64_capability,
        "numpy_version": numpy.__version__,
        "scipy_version": scipy.__version__,
        "dataset_sha256": dataset_hash,
        "protocol_config_sha256": protocol_hash,
        "protocol_status": config.get("protocol", {}).get("status"),
    }

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    output = _provenance_output(socket.gethostname())
    output.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(provenance, indent=2, sort_keys=True))
    if failures:
        print("VERIFY FAILED:", failures, file=sys.stderr)
        return 1
    print("VERIFY PASSED: frozen dataset and protocol identity intact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
