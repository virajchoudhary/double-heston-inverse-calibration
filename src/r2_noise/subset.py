"""Deterministic predeclared traditional-calibration subset (frozen design).

Selects exactly 250 test-split surfaces stratified by truth ``v0_total`` and
``kappa_slow`` terciles (3x3 = 9 cells, proportional largest-remainder
allocation), ordered within cells by ascending SHA-256 hex of the surface id.
A pure function of the clean dataset's known truths and ids: no RNG and no
model/calibration outcome participates.  Must be executed BEFORE any noisy
result exists.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np

SUBSET_SIZE = 250
STRATA_PARAMETERS = ("v0_total", "kappa_slow")


def _tercile_edges(values: np.ndarray) -> np.ndarray:
    """Tercile cut points (2 edges) of the given clean-truth values."""
    return np.percentile(values, [100.0 / 3.0, 200.0 / 3.0])


def _cell_index(value: float, edges: np.ndarray) -> int:
    return int(np.searchsorted(edges, value, side="left"))


def select_traditional_subset(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Select the frozen traditional subset from test-split truth records.

    ``records`` must be one mapping per TEST surface with keys ``surface_id``
    plus canonical parameters under ``parameters_canonical_order``.  Returns a
    frozen-shape payload including ids, strata metadata, allocation, and
    provenance; safe to serialize as the committed selection artifact.
    """
    if len(records) != SUBSET_SIZE * 5:
        raise ValueError(
            f"expected exactly {SUBSET_SIZE * 5} test surfaces, got {len(records)}"
        )
    v0_total = np.array(
        [r["parameters_canonical_order"]["v0_slow"] + r["parameters_canonical_order"]["v0_fast"]
         for r in records]
    )
    kappa_slow = np.array(
        [r["parameters_canonical_order"]["kappa_slow"] for r in records]
    )
    v0_edges = _tercile_edges(v0_total)
    kappa_edges = _tercile_edges(kappa_slow)

    cells: dict[tuple[int, int], list[str]] = {}
    for record, v0_value, kappa_value in zip(records, v0_total, kappa_slow):
        cell = (_cell_index(float(v0_value), v0_edges), _cell_index(float(kappa_value), kappa_edges))
        cells.setdefault(cell, []).append(record["surface_id"])

    for members in cells.values():
        members.sort(key=lambda sid: hashlib.sha256(sid.encode("utf-8")).hexdigest())

    # proportional allocation by largest remainder to reach exactly SUBSET_SIZE
    total = sum(len(members) for members in cells.values())
    exact = {cell: len(members) * SUBSET_SIZE / total for cell, members in cells.items()}
    floors = {cell: int(np.floor(value)) for cell, value in exact.items()}
    remaining = SUBSET_SIZE - sum(floors.values())
    order_by_remainder = sorted(
        cells, key=lambda cell: (-(exact[cell] - floors[cell]), cell)
    )
    allocation = dict(floors)
    for cell in order_by_remainder[:remaining]:
        allocation[cell] += 1

    selected: list[dict[str, Any]] = []
    for cell in sorted(cells):
        take = allocation.get(cell, 0)
        for surface_id in cells[cell][:take]:
            selected.append({"surface_id": surface_id, "v0_total_cell": cell[0], "kappa_slow_cell": cell[1]})

    return {
        "artifact_kind": "R2_NOISE_ROBUSTNESS_TRADITIONAL_SUBSET",
        "subset_size": SUBSET_SIZE,
        "population": "test_split_only_1250_surfaces",
        "strata_dimensions": list(STRATA_PARAMETERS),
        "tercile_edges": {
            "v0_total": [float(edge) for edge in v0_edges],
            "kappa_slow": [float(edge) for edge in kappa_edges],
        },
        "within_cell_ordering": "ascending_sha256_hex_of_surface_id",
        "allocation_proportional_largest_remainder": True,
        "selection_uses_outcomes": False,
        "selected_ids": [entry["surface_id"] for entry in selected],
        "selected_detail": selected,
        "cell_sizes": {f"v{cell[0]}_k{cell[1]}": len(members) for cell, members in sorted(cells.items())},
    }
