"""Explicit authorization boundary for the frozen clean synthetic test."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import EXPECTED_DATASET_SHA256, verify_freeze_manifest


AUTHORIZATION_FLAG = "--authorize-frozen-test-evaluation"


class FrozenTestLockedError(RuntimeError):
    """Raised before opening any frozen-test records when access is unauthorized."""


def _read_json(path: str | Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise FrozenTestLockedError(f"non-finite JSON literal {value}")

    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle, parse_constant=reject_constant)
    except json.JSONDecodeError as error:
        raise FrozenTestLockedError(f"invalid freeze manifest JSON: {error}") from error
    if not isinstance(value, dict):
        raise FrozenTestLockedError("freeze manifest must contain an object")
    return value


def require_frozen_test_authorization(
    freeze_manifest_path: str | Path,
    *,
    authorized: bool,
) -> dict[str, Any]:
    """Fail closed unless a valid freeze manifest and explicit flag are present."""
    if not authorized:
        raise FrozenTestLockedError(
            f"frozen Model3 clean-test evaluation is locked; pass {AUTHORIZATION_FLAG} "
            "with a verified three-seed freeze manifest"
        )
    verification = verify_freeze_manifest(_read_json(freeze_manifest_path))
    if verification["final_r2_dataset_sha256"] != EXPECTED_DATASET_SHA256:
        raise FrozenTestLockedError("freeze manifest does not pin the frozen clean dataset")
    return verification
