#!/usr/bin/env python3
"""Fail-closed Apple-Silicon precheck for the frozen Double Heston V2 run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


EXPECTED = {
    "outputs/regular_pinn_recovery/double_data_v2_reproducible/train.npz":
        "92b5c70d8a66fc2996ed945b54ba9a80feb6768874151ab666aea54a3e2c6250",
    "outputs/regular_pinn_recovery/double_data_v2_reproducible/validation.npz":
        "479673a31342f2dd27eb05ecf6e72913f233e0b4ea41153fe74f6dfb6dca2bd1",
    "outputs/regular_pinn_recovery/double_data_v2_reproducible/collocation.npz":
        "730d7352df74075464c4443240f5de753371ef2bf6c2593a3f861727b3171c6f",
    "outputs/regular_pinn_recovery/double_data_v2_reproducible/rejections.npz":
        "49ce6c347f1982d33565299eee73aff71b212a78f51e5e68092bf96ef0ed3039",
    "outputs/regular_pinn_recovery/double_data_v2_reproducible/manifest.json":
        "c673e1bd190e3608ed0b5e544c48e73c36ea33a8276d074a69d57080be45bd3a",
    "outputs/regular_pinn_recovery/double_jacobian_surfaces_v2_927721/surfaces.npz":
        "4a6a8f91c3432262b34b77f01c989828b2ee2d0e1a2d29a57adf7d17b3946e84",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def command(*args: str) -> str:
    return subprocess.run(args, check=True, text=True, capture_output=True).stdout.strip()


def memory_status() -> dict[str, int]:
    total = int(command("sysctl", "-n", "hw.memsize"))
    lines = command("vm_stat").splitlines()
    page_size = int(lines[0].split("page size of ", 1)[1].split(" bytes", 1)[0])
    pages = {}
    for line in lines[1:]:
        if ":" not in line:
            continue
        name, value = line.split(":", 1)
        pages[name] = int(value.strip().rstrip("."))
    available_pages = sum(
        pages.get(name, 0)
        for name in ("Pages free", "Pages inactive", "Pages speculative")
    )
    return {"total_bytes": total, "available_bytes": available_pages * page_size}


def verify_artifacts(root: Path) -> dict[str, dict[str, str | bool]]:
    checks = {}
    for relative, expected in EXPECTED.items():
        path = root / relative
        actual = sha256(path) if path.is_file() else None
        checks[relative] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "passed": actual == expected,
        }
    failed = [name for name, result in checks.items() if not result["passed"]]
    if failed:
        raise RuntimeError(f"HARD STOP: V2 artifact hash mismatch or missing file: {failed}")
    return checks


def verify_dataset_structure(root: Path) -> dict[str, int | bool]:
    data = root / "outputs/regular_pinn_recovery/double_data_v2_reproducible"
    expected_rows = {"train": 131072, "validation": 16384, "collocation": 18000}
    result: dict[str, int | bool] = {}
    for split, rows in expected_rows.items():
        with np.load(data / f"{split}.npz", allow_pickle=False) as source:
            q = source["q"]
            if q.shape != (rows, 12) or not np.isfinite(q).all():
                raise RuntimeError(f"HARD STOP: invalid {split} coordinates")
            if split != "collocation":
                for name in ("price", "w", "g", "dg_du", "quadrature_difference"):
                    if not np.isfinite(source[name]).all():
                        raise RuntimeError(f"HARD STOP: non-finite {split}/{name}")
                if not source["usable"].all():
                    raise RuntimeError(f"HARD STOP: unusable rows in {split}")
            result[f"{split}_rows"] = rows
    with np.load(data / "rejections.npz", allow_pickle=False) as source:
        result["rejected_rows"] = len(source["q"])
    if result["rejected_rows"] != 0:
        raise RuntimeError("HARD STOP: frozen V2 rejection count changed")
    result["passed"] = True
    return result


def mlx_smoke(root: Path) -> dict[str, float | bool]:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    sys.path.insert(0, str(root))
    from src.mentor_dh_pinn.regular_pinn import RegularVariancePINN

    mx.random.seed(17)
    model = RegularVariancePINN(factors=2, width=16, depth=1)
    coords = mx.array(
        [[0.0, 0.04, 0.03, 0.5], [0.1, 0.05, 0.02, 1.0]], dtype=mx.float32
    )
    structural = mx.array(
        [
            [[0.8, 0.03, 0.12, -0.5], [4.0, 0.02, 0.20, -0.2]],
            [[1.0, 0.04, 0.15, -0.4], [5.0, 0.02, 0.22, -0.1]],
        ],
        dtype=mx.float32,
    )
    target = mx.array([0.21, 0.24], dtype=mx.float32)

    def loss():
        return mx.mean((model(coords, structural) - target) ** 2)

    value_and_grad = nn.value_and_grad(model, loss)
    optimizer = optim.Adam(learning_rate=1e-3)
    before, gradients = value_and_grad()
    optimizer.update(model, gradients)
    after = loss(model)
    mx.eval(model.parameters(), optimizer.state, before, after)
    if not np.isfinite([float(before), float(after)]).all():
        raise RuntimeError("HARD STOP: non-finite MLX smoke loss")

    with tempfile.TemporaryDirectory() as temporary:
        checkpoint = Path(temporary) / "smoke.safetensors"
        model.save_weights(str(checkpoint))
        restored = RegularVariancePINN(factors=2, width=16, depth=1)
        restored.load_weights(str(checkpoint))
        original = np.asarray(model(coords, structural))
        reloaded = np.asarray(restored(coords, structural))
        if not np.array_equal(original, reloaded):
            raise RuntimeError("HARD STOP: MLX checkpoint round trip changed predictions")
    return {
        "passed": True,
        "loss_before": float(before),
        "loss_after_one_step": float(after),
        "checkpoint_round_trip": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("HARD STOP: B0 requires macOS on Apple Silicon (arm64)")
    if args.report.exists():
        raise FileExistsError("Preserve prior precheck evidence; choose a new report path")

    report = {
        "status": "PASS",
        "git_head": command("git", "-C", str(root), "rev-parse", "HEAD"),
        "platform": platform.platform(),
        "macos_version": platform.mac_ver()[0],
        "architecture": platform.machine(),
        "memory": memory_status(),
        "versions": {
            "python": platform.python_version(),
            "mlx": importlib.metadata.version("mlx"),
            "numpy": np.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "torch": importlib.metadata.version("torch"),
        },
        "artifact_checks": verify_artifacts(root),
        "dataset_structure": verify_dataset_structure(root),
        "mlx_smoke": mlx_smoke(root),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
