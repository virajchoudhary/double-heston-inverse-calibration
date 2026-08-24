"""Keyed deterministic multiplicative observation noise (frozen contract).

Lineage: the G2 R2-vs-R3 donor contract in ``src/g2_r2r3/noise.py``
(multiplicative ``clean * (1 + level * z)``, SHA-256-keyed per-slot draws,
bit-replayable).  This module re-derives that documented functional form for
the R2 primary surfaces with a NEW dedicated base seed so realizations are
independent of every historical study.  Keys contain no method/model/seed
component: all three methods and all neural seeds observe byte-identical
noisy cohorts at each level.
"""

from __future__ import annotations

import hashlib

import numpy as np

NOISE_BASE_SEED = 20260825
NOISE_LEVELS: tuple[float, ...] = (0.0, 0.001, 0.0025, 0.005, 0.01)
_MAX_RESAMPLE_COUNTER = 64


def slot_seed(
    surface_id: str,
    expiry_rank: int,
    moneyness_k: float,
    option_type: str,
    noise_level: float,
) -> int:
    """Deterministic 63-bit seed from the frozen key format."""
    key = (
        f"{NOISE_BASE_SEED}|{surface_id}|rank{expiry_rank}"
        f"|k{moneyness_k:+.2f}|{option_type}|level{noise_level:.4f}"
    )
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) & ((1 << 63) - 1)


def slot_noise_factor(
    surface_id: str,
    expiry_rank: int,
    moneyness_k: float,
    option_type: str,
    noise_level: float,
) -> float:
    """Single multiplicative factor ``(1 + level * z)`` for one quote slot."""
    if noise_level == 0.0:
        return 1.0
    rng = np.random.default_rng(
        slot_seed(surface_id, expiry_rank, moneyness_k, option_type, noise_level)
    )
    z = float(rng.standard_normal())
    return 1.0 + noise_level * z


def perturb_slot(
    clean_price: float,
    surface_id: str,
    expiry_rank: int,
    moneyness_k: float,
    option_type: str,
    noise_level: float,
) -> tuple[float, int]:
    """Perturb one quote slot; returns ``(observed_price, resample_counter)``.

    Positivity is asserted rather than clamped.  A negative draw (requires
    ``z < -1/level``; effectively impossible at the frozen levels) triggers a
    deterministic counter-suffixed redraw, capped at 64 attempts.
    """
    if noise_level == 0.0:
        return float(clean_price), 0
    counter = 0
    while True:
        if counter == 0:
            seed = slot_seed(surface_id, expiry_rank, moneyness_k, option_type, noise_level)
        else:
            key = (
                f"{NOISE_BASE_SEED}|{surface_id}|rank{expiry_rank}"
                f"|k{moneyness_k:+.2f}|{option_type}|level{noise_level:.4f}"
                f"#r{counter}"
            )
            digest = hashlib.sha256(key.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], "big", signed=False) & ((1 << 63) - 1)
        z = float(np.random.default_rng(seed).standard_normal())
        observed = float(clean_price) * (1.0 + noise_level * z)
        if observed >= 0.0:
            return observed, counter
        counter += 1
        if counter > _MAX_RESAMPLE_COUNTER:
            raise RuntimeError(
                "deterministic noise could not produce a positive price for "
                f"({surface_id}, rank{expiry_rank}, k{moneyness_k}, "
                f"{option_type}, level {noise_level})"
            )


def perturb_surface_prices(
    clean_prices,
    surface_id: str,
    slot_keys,
    noise_level: float,
):
    """Perturb every slot of one surface.

    Returns ``(noisy_prices, realization_ids)`` where each realization id
    records the per-slot resample counter (identity of the draw).
    """
    noisy = []
    counters = []
    for index, ((expiry_rank, moneyness_k, option_type), price) in enumerate(
        zip(slot_keys, clean_prices)
    ):
        value, counter = perturb_slot(
            float(price),
            surface_id,
            int(expiry_rank),
            float(moneyness_k),
            str(option_type),
            float(noise_level),
        )
        noisy.append(value)
        counters.append(counter)
    return noisy, counters
