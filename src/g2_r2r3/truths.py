"""Deterministic truth panel for the G2 R2-vs-R3 study.

Panel composition (protocol section 5):
- the four standing representative G2 truth cases (``frozen.STANDING_TRUTH_VECTORS``);
- 16 additional deterministic reviewed-interior truths drawn with the frozen
  truth-selection seed 20260822 from the EXISTING reviewed sampling contract
  (``src/audit_reviewed_sampling.py`` ``interior_train`` population), filtered
  by that contract's own ordinary-training margin gate, first 16 accepted rows.

No new parameter distribution is introduced.  The draw size (64) is fixed in
``frozen.ADDITIONAL_TRUTH_DRAW_COUNT`` and the whole procedure is recorded in
the experiment manifest before any R2/R3 outcome is computed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..audit_reviewed_sampling import sample_distribution
from ..calibrate_double_heston import load_hard_safety_bounds
from ..constants import PARAMETER_NAMES
from ..constraints import validate_parameters
from . import frozen

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BOUNDS_PATH = REPOSITORY_ROOT / "configs" / "parameter_bounds_PROVISIONAL.yaml"


def additional_truths() -> pd.DataFrame:
    """Return the 16 seeded reviewed-interior truths in row order."""
    frame = sample_distribution(
        "interior_train",
        count=frozen.ADDITIONAL_TRUTH_DRAW_COUNT,
        seed=frozen.TRUTH_SELECTION_SEED,
    )
    accepted = frame.loc[frame["accepted"]].reset_index(drop=True)
    if len(accepted) < frozen.ADDITIONAL_TRUTH_COUNT:
        raise RuntimeError(
            "seeded reviewed-interior draw produced fewer accepted rows than "
            "required by the frozen panel size"
        )
    chosen = accepted.iloc[: frozen.ADDITIONAL_TRUTH_COUNT].copy()
    chosen["truth_id"] = [
        f"interior_seed{frozen.TRUTH_SELECTION_SEED}_{int(row.candidate_id):04d}"
        for row in chosen.itertuples()
    ]
    return chosen


def truth_panel() -> pd.DataFrame:
    """Build the frozen 20-truth panel with identifiers and provenance."""
    rows: list[dict[str, Any]] = []
    for truth_index, case_id in enumerate(
        ["case_1", "case_2", "case_3", "case_4"], start=0
    ):
        vector = frozen.STANDING_TRUTH_VECTORS[case_id].copy()
        rows.append(
            {
                "truth_index": truth_index,
                "truth_id": case_id,
                "source": "standing_representative_g2_case",
                "sample_id": frozen.STANDING_TRUTH_SAMPLE_IDS[case_id],
                **{
                    name: float(value) for name, value in zip(PARAMETER_NAMES, vector)
                },
            }
        )
    extras = additional_truths()
    for offset, row in enumerate(extras.itertuples(), start=frozen.STANDING_TRUTH_COUNT):
        vector = np.asarray(
            [getattr(row, name) for name in PARAMETER_NAMES], dtype=np.float64
        )
        rows.append(
            {
                "truth_index": offset,
                "truth_id": row.truth_id,
                "source": "reviewed_interior_seed_20260822",
                "sample_id": f"interior_train_{int(row.candidate_id)}",
                **{
                    name: float(value) for name, value in zip(PARAMETER_NAMES, vector)
                },
            }
        )
    panel = pd.DataFrame(rows)
    if len(panel) != frozen.TRUTH_PANEL_SIZE:
        raise RuntimeError("truth panel size mismatch")
    for row in panel.itertuples():
        diagnostics = validate_parameters(
            np.asarray([getattr(row, name) for name in PARAMETER_NAMES])
        )
        if not diagnostics["is_valid"]:
            raise RuntimeError(f"invalid truth vector in panel: {row.truth_id}")
    return panel


def panel_vectors(panel: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        row.truth_id: np.asarray(
            [getattr(row, name) for name in PARAMETER_NAMES], dtype=np.float64
        )
        for row in panel.itertuples()
    }


def load_bounds() -> dict[str, tuple[float, float]]:
    return load_hard_safety_bounds(BOUNDS_PATH)
