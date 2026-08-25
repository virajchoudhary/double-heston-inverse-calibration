"""Optional Model3 inclusion decision made before G8 acquisition."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROTOCOL_BASE_SHA = "7eecc7188c54f9d4505d32ccf5c51069a4c3a97c"
FROZEN_STATUS = "FROZEN_FINAL_MODEL3_RESEARCH_ARTIFACT"
NOT_FROZEN_LABEL = "MODEL3_NOT_FROZEN_NOT_EVALUATED"


@dataclass(frozen=True)
class Model3FreezeEvidence:
    final_status: str
    commit_sha: str
    artifact_path: str
    artifact_sha256: str
    committed_before_acquisition: bool


def evaluate_model3_inclusion(
    evidence: Model3FreezeEvidence | None = None,
    *,
    repository_root: Path | str = Path.cwd(),
    acquisition_has_begun: bool = False,
) -> dict[str, Any]:
    """Include Model3 only on complete, pre-acquisition, merged freeze proof."""
    if acquisition_has_begun:
        return {
            "decision": "MODEL3_POST_ACQUISITION_CHANGE_FORBIDDEN",
            "label": NOT_FROZEN_LABEL,
        }
    if evidence is None:
        return {
            "decision": "NO_FINAL_FROZEN_MODEL3_EVIDENCE",
            "label": NOT_FROZEN_LABEL,
            "remote_stage_a_commit_is_ancestor_of_protocol_base": False,
        }
    root = Path(repository_root)
    artifact = root / evidence.artifact_path
    checks = {
        "final_status_valid": evidence.final_status == FROZEN_STATUS,
        "commit_shape_valid": len(evidence.commit_sha) == 40
        and all(character in "0123456789abcdef" for character in evidence.commit_sha.lower()),
        "artifact_exists": artifact.is_file(),
        "committed_before_acquisition": evidence.committed_before_acquisition is True,
        "acquisition_not_started": acquisition_has_begun is False,
    }
    if artifact.is_file():
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        checks["artifact_sha_matches"] = digest == evidence.artifact_sha256.lower()
    else:
        checks["artifact_sha_matches"] = False
    try:
        subprocess.run(
            ["git", "cat-file", "-e", f"{evidence.commit_sha}^{{commit}}"],
            cwd=root,
            check=True,
            capture_output=True,
        )
        checks["commit_exists"] = True
        checks["commit_ancestor_of_execution_head"] = subprocess.run(
            ["git", "merge-base", "--is-ancestor", evidence.commit_sha, "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
        ).returncode == 0
    except (OSError, subprocess.CalledProcessError):
        checks["commit_exists"] = False
        checks["commit_ancestor_of_execution_head"] = False
    included = all(checks.values())
    return {
        "decision": "MODEL3_INCLUDED" if included else "FINAL_FREEZE_EVIDENCE_INCOMPLETE",
        "label": "MODEL3_INCLUDED" if included else NOT_FROZEN_LABEL,
        "checks": checks,
        "evidence": {
            "final_status": evidence.final_status,
            "commit_sha": evidence.commit_sha,
            "artifact_path": evidence.artifact_path,
            "artifact_sha256": evidence.artifact_sha256,
            "committed_before_acquisition": evidence.committed_before_acquisition,
        },
    }
