"""Hash Model 3 outputs or generate a failure-preservation record."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import time
from pathlib import Path
from typing import Any


EXCLUDED_NAMES = {"artifact_manifest.json", "failure_record.json"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower()


def collect_files(output_root: Path) -> list[Path]:
    return sorted(
        path
        for path in output_root.rglob("*")
        if path.is_file() and path.name not in EXCLUDED_NAMES
    )


def manifest(output_root: Path, *, failed: bool) -> dict[str, Any]:
    paths = collect_files(output_root)
    artifacts: dict[str, Any] = {}
    for path in paths:
        relative = path.relative_to(output_root).as_posix()
        artifacts[relative] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return {
        "schema": "MODEL3_OUTPUT_MANIFEST_V1"
        if not failed
        else "MODEL3_FAILURE_PRESERVATION_MANIFEST_V1",
        "failed_run_preserved": failed,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "created_unix_seconds": time.time(),
        "output_root": output_root.resolve().as_posix(),
        "artifact_count": len(paths),
        "artifacts": artifacts,
    }


def verify(output_root: Path) -> list[str]:
    manifest_path = output_root / "artifact_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = payload["artifacts"]
    actual_paths = collect_files(output_root)
    actual = {path.relative_to(output_root).as_posix(): path for path in actual_paths}
    failures = []
    for relative, metadata in expected.items():
        path = actual.get(relative)
        if path is None:
            failures.append(f"missing: {relative}")
            continue
        if path.stat().st_size != metadata["bytes"]:
            failures.append(f"size_mismatch: {relative}")
        if sha256_file(path) != metadata["sha256"]:
            failures.append(f"hash_mismatch: {relative}")
    failures.extend(f"extra: {relative}" for relative in set(actual) - set(expected))
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--failed", action="store_true")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify an existing artifact_manifest.json without rewriting it",
    )
    arguments = parser.parse_args(argv)
    output_root = arguments.output_root.resolve()
    if arguments.verify_only:
        failures = verify(output_root)
        result = {
            "status": "PASS" if not failures else "FAIL",
            "failures": failures,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if not failures else 1
    payload = manifest(output_root, failed=arguments.failed)
    name = "failure_record.json" if arguments.failed else "artifact_manifest.json"
    destination = output_root / name
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite: {destination}")
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
