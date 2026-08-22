"""Deterministic multi-start schedule for the G2 R2-vs-R3 study.

Twelve starts per truth per noise level, generated from the frozen multi-start
seed 20260823 under the existing project convention:

- start 0 is the neutral transform midpoint (latent zeros);
- starts 1-11 are broad N(0, 1.25^2) latent draws.

The start seed is keyed by ``(truth_index, noise_level)`` ONLY — never by the
representation — so R2 and R3 receive bit-identical starts.
"""

from __future__ import annotations

import numpy as np

from . import frozen


def start_seed(truth_index: int, noise_index: int) -> int:
    return (
        frozen.MULTISTART_SEED
        + 100_000 * int(truth_index)
        + 100 * int(noise_index)
    )


def deterministic_starts(seed: int) -> list[tuple[str, np.ndarray]]:
    """Return the twelve named deterministic latent starts."""
    rng = np.random.default_rng(seed)
    starts: list[tuple[str, np.ndarray]] = [
        ("neutral_transform_midpoint", np.zeros(10, dtype=np.float64))
    ]
    for index in range(1, frozen.START_COUNT):
        starts.append(
            (f"deterministic_broad_{index}", rng.normal(0.0, frozen.BROAD_START_SCALE, size=10))
        )
    return starts


def start_schedule_for_cell(truth_index: int, noise_index: int) -> list[tuple[str, np.ndarray]]:
    return deterministic_starts(start_seed(truth_index, noise_index))
