"""Freeze the G2 R2-vs-R3 truth panel (checkpoint A).

Persists the exact 20 truth vectors and identifiers to the experiment
manifest BEFORE any R2/R3 outcome is computed, then commits are made from the
shell.  Procedure is fully deterministic:

- 4 standing representative G2 truth cases (frozen literals in
  ``src/g2_r2r3/frozen.py``, exact committed maximin selection);
- 16 additional reviewed-interior truths = first 16 accepted rows of the
  existing ``interior_train`` sampler draw of 64 rows with truth-selection
  seed 20260822, using that contract's own margin gate.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.calibrate_double_heston import load_hard_safety_bounds  # noqa: E402
from src.constants import PARAMETER_NAMES  # noqa: E402
from src.g2_r2r3 import frozen  # noqa: E402
from src.g2_r2r3.truths import truth_panel  # noqa: E402

EVIDENCE_ROOT = REPOSITORY_ROOT / "evidence" / "g2_r2_r3_20260822"
TRUTH_PANEL_CSV = EVIDENCE_ROOT / "truth_panel.csv"
MANIFEST_PATH = EVIDENCE_ROOT / "manifest.json"


def environment_metadata() -> dict[str, str]:
    import torch

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "git_version": _git_version(),
    }


def _git_version() -> str:
    import subprocess

    return subprocess.run(
        ["git", "--version"], capture_output=True, text=True, check=True
    ).stdout.strip()


def freeze_panel() -> dict[str, object]:
    panel = truth_panel()
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    panel.to_csv(TRUTH_PANEL_CSV, index=False, lineterminator="\n")

    bounds = load_hard_safety_bounds(
        REPOSITORY_ROOT / "configs" / "parameter_bounds_PROVISIONAL.yaml"
    )
    manifest = {
        "analysis_id": "G2_R2_R3_REPRESENTATION_SELECTION",
        "protocol": "docs/G2_SELF_GOVERNED_REPRESENTATION_PROTOCOL.md",
        "phase": "TRUTH_PANEL_FROZEN",
        "expected_starting_main": "c64aaaf54f72a48768316c582d312dd8cf27a089",
        "execution_branch": "research/g2-r2-r3-selection",
        "randomization": {
            "truth_selection_seed": frozen.TRUTH_SELECTION_SEED,
            "multistart_seed": frozen.MULTISTART_SEED,
            "noise_base_seed": frozen.NOISE_BASE_SEED,
        },
        "truth_panel": {
            "size": int(len(panel)),
            "standing_cases": list(frozen.STANDING_TRUTH_VECTORS),
            "additional_truths_procedure": (
                "first 16 accepted rows (existing ordinary-training margin gate) "
                "of a single interior_train LHS draw of 64 rows with seed 20260822 "
                "via src/audit_reviewed_sampling.sample_distribution"
            ),
            "additional_draw_count": frozen.ADDITIONAL_TRUTH_DRAW_COUNT,
            "panel_csv": "evidence/g2_r2_r3_20260822/truth_panel.csv",
            "parameter_order": list(PARAMETER_NAMES),
        },
        "noise_levels": list(frozen.NOISE_LEVELS),
        "start_count": frozen.START_COUNT,
        "matrix_plan": {
            "truths": 20,
            "representations": list(frozen.REPRESENTATIONS),
            "noise_levels": len(frozen.NOISE_LEVELS),
            "starts": frozen.START_COUNT,
            "total_calibration_attempts": 20 * 2 * 4 * 12,
        },
        "hard_bounds": {
            name: [bounds[name][0], bounds[name][1]] for name in PARAMETER_NAMES
        },
        "environment": environment_metadata(),
        "notes": [
            "Truth panel frozen before any R2/R3 outcome computation.",
            "All five NTPC development dates remain excluded from final G8.",
        ],
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


if __name__ == "__main__":
    result = freeze_panel()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
