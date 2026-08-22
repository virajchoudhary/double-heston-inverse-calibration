"""R2/R3 representation geometry for the G2 selection study.

R2: 20 NOMINAL price slots — first TWO eligible listed expiry ranks x
central-five log-moneyness x calls/puts — spot-normalized prices, actual
time-to-maturity supplied explicitly, existing per-rank rate/carry
conditioning.  On real-market construction, unsupported or unusable slots
carry an explicit mask/missingness flag; a missing real quote is NEVER
imputed with a model price or any proxy.  The synthetic G2 panel is complete
by construction (no missing slots), which is a property of the synthetic
design, not an assumption about real surfaces.

R3: 30 NOMINAL price slots — first THREE eligible listed expiry ranks, same
contract and same explicit masking semantics.  In this study R3's synthetic
third-rank slots were complete while real third-rank support was 100% masked
on all five NTPC development dates (a limitation of the R3 comparison, not of
the frozen R2 decision).

Slot identity is ``(expiry_rank, moneyness, option_type)`` — the same key the
noise scheme uses — so common R2/R3 slots are aligned by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from . import frozen


@dataclass(frozen=True)
class DateProfile:
    """Actual first-ranks market geometry of one development date."""

    date_id: str
    spot: float
    expiry_dates: tuple[str, ...]
    dte: tuple[int, ...]
    rates: tuple[float, ...]  # continuous risk-free rate per rank (from T-bill discount)
    carries: tuple[float, ...]  # futures-implied carry per rank

    def summary(self) -> dict[str, object]:
        return {
            "date_id": self.date_id,
            "spot": self.spot,
            "expiry_dates": list(self.expiry_dates),
            "dte": list(self.dte),
            "rates": list(self.rates),
            "carries": list(self.carries),
        }


@dataclass(frozen=True)
class SlotSpec:
    rank: int  # 1-based listed expiry rank
    moneyness: float
    option_type: str
    maturity_years: float  # ACTUAL time to maturity supplied explicitly
    rate: float
    carry: float

    @property
    def key(self) -> tuple[int, float, str]:
        return (self.rank, self.moneyness, self.option_type)


def representation_slot_count(representation: str) -> int:
    if representation == "R2":
        return frozen.R2_NOMINAL_SLOTS
    if representation == "R3":
        return frozen.R3_NOMINAL_SLOTS
    raise ValueError(f"unknown representation: {representation}")


def representation_expiry_ranks(representation: str) -> tuple[int, ...]:
    if representation == "R2":
        return tuple(range(1, frozen.R2_EXPIRY_RANKS + 1))
    if representation == "R3":
        return tuple(range(1, frozen.R3_EXPIRY_RANKS + 1))
    raise ValueError(f"unknown representation: {representation}")


def representation_slots(
    profile: DateProfile, representation: str
) -> tuple[SlotSpec, ...]:
    """Canonical slot order: option-type major, then expiry rank, then moneyness."""
    ranks = representation_expiry_ranks(representation)
    slots: list[SlotSpec] = []
    for option_type in ("call", "put"):
        for rank in ranks:
            index = rank - 1
            for moneyness in frozen.CENTRAL_FIVE:
                slots.append(
                    SlotSpec(
                        rank=rank,
                        moneyness=moneyness,
                        option_type=option_type,
                        maturity_years=profile.dte[index] / 365.0,
                        rate=profile.rates[index],
                        carry=profile.carries[index],
                    )
                )
    if len(slots) != representation_slot_count(representation):
        raise RuntimeError("representation geometry size mismatch")
    return tuple(slots)


def build_geometry(
    slots: Sequence[SlotSpec], *, spot: float = frozen.SYNTHETIC_SPOT
) -> dict[str, np.ndarray]:
    """Quote-aligned arrays plus the slot-key list for noise alignment."""
    strikes = np.asarray(
        [spot * float(np.exp(slot.moneyness)) for slot in slots], dtype=np.float64
    )
    maturities = np.asarray([slot.maturity_years for slot in slots], dtype=np.float64)
    option_types = np.asarray([slot.option_type for slot in slots], dtype=str)
    rates = np.asarray([slot.rate for slot in slots], dtype=np.float64)
    dividends = np.asarray([slot.carry for slot in slots], dtype=np.float64)
    mask = np.ones(len(slots), dtype=bool)  # synthetic panel: no unusable slots
    return {
        "strikes": strikes,
        "maturities": maturities,
        "option_types": option_types,
        "rates": rates,
        "dividends": dividends,
        "mask": mask,
        "slot_keys": [slot.key for slot in slots],
    }


def profile_for_truth(truth_index: int, profiles: Sequence[DateProfile]) -> DateProfile:
    """Frozen deterministic truth-to-date-profile assignment (index mod 5)."""
    return profiles[truth_index % len(profiles)]
