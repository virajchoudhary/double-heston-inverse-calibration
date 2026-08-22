"""Keyed deterministic observational noise for the G2 R2-vs-R3 study.

Noise is multiplicative lognormal in the existing project convention
(``observed = clean * (1 + level * z)``), but the draw is keyed per quote slot
by ``(truth_id, expiry_rank, moneyness, option_type, noise_level)`` so that:

- every slot shared by R2 and R3 receives the bit-identical perturbation
  (the seed depends only on slot identity, never on the representation);
- R3-only third-expiry slots receive deterministic additional draws from the
  same keyed scheme;
- reruns are bit-identical.

The per-slot RNG seed is derived by hashing a canonical key string with
SHA-256 (platform-independent), seeded under the frozen noise base seed
20260824.
"""

from __future__ import annotations

import hashlib

import numpy as np

from . import frozen


def slot_seed(
    truth_id: str,
    expiry_rank: int,
    moneyness: float,
    option_type: str,
    noise_level: float,
) -> int:
    key = (
        f"{frozen.NOISE_BASE_SEED}|{truth_id}|rank{expiry_rank}"
        f"|k{moneyness:+.2f}|{option_type}|level{noise_level:.4f}"
    )
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) & ((1 << 63) - 1)


def slot_noise_factor(
    truth_id: str,
    expiry_rank: int,
    moneyness: float,
    option_type: str,
    noise_level: float,
) -> float:
    """Return the single multiplicative factor (1 + level * z) for one slot."""
    if noise_level == 0.0:
        return 1.0
    rng = np.random.default_rng(
        slot_seed(truth_id, expiry_rank, moneyness, option_type, noise_level)
    )
    z = float(rng.standard_normal())
    return 1.0 + noise_level * z


def perturb_slot(
    clean_price: float,
    truth_id: str,
    expiry_rank: int,
    moneyness: float,
    option_type: str,
    noise_level: float,
) -> float:
    factor = slot_noise_factor(
        truth_id, expiry_rank, moneyness, option_type, noise_level
    )
    observed = float(clean_price) * factor
    if observed < 0.0:
        raise RuntimeError(
            "keyed multiplicative noise produced a negative price for "
            f"({truth_id}, rank{expiry_rank}, k{moneyness}, {option_type}, "
            f"{noise_level})"
        )
    return observed
