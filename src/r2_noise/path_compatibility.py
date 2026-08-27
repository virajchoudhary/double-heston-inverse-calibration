"""Fail-closed path relocation policy for frozen R2 provenance artifacts."""

from __future__ import annotations

import posixpath
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTED_REPOSITORY_ROOTS = (
    REPO_ROOT.resolve().as_posix(),
    "C:/ann_inverse_calibration",
)


def documented_repo_relative_path(path: str | Path) -> str | None:
    """Return a repository-relative path only for an exact approved root."""
    normalized = str(path)
    if "\\" in normalized:
        return None
    if posixpath.normpath(normalized) != normalized:
        return None
    for root in DOCUMENTED_REPOSITORY_ROOTS:
        root_normalized = root.replace("\\", "/").rstrip("/")
        prefix = f"{root_normalized}/"
        if not normalized.startswith(prefix):
            continue
        relative = normalized[len(prefix) :]
        parts = PurePosixPath(relative).parts
        if not relative or any(part in {"", ".", ".."} for part in parts):
            return None
        return PurePosixPath(*parts).as_posix()
    return None


def is_documented_worktree_relocation(
    stored_path: str | Path,
    current_path: str | Path,
) -> bool:
    """Require exact repo-relative identity under approved repository roots."""
    stored_relative = documented_repo_relative_path(stored_path)
    current_relative = documented_repo_relative_path(current_path)
    return (
        stored_relative is not None
        and current_relative is not None
        and stored_relative == current_relative
    )
