"""The canonical R2 surface object: 20 nominal slots plus explicit mask.

One immutable :class:`R2Surface` is the single in-memory form shared by
synthetic generation, ANN/Model-2 dataset construction, and frozen
real-market evaluation.  Field semantics (see
``docs/R2_REPRESENTATION_CONTRACT.md``):

- ``prices`` -- 20 spot-normalized option prices (price / spot).  Valid slots
  carry strictly positive finite values; masked (unavailable) slots carry
  exactly ``0.0`` (``contract.MASKED_PRICE_PLACEHOLDER``).  NaN/Inf are never
  allowed anywhere.
- ``mask`` -- 20 validity flags.  ``True`` = real, usable observation (or
  synthetic slot, complete by construction).  ``False`` = unsupported or
  unusable slot; never imputed.
- ``maturities`` / ``rates`` / ``carries`` -- per-slot actual time-to-maturity
  (years) and the existing per-rank rate/carry conditioning.  These are known
  for every slot of an eligible rank, including masked quote slots, so they
  are always 20 finite values.
- ``spot`` -- the normalization spot for this surface.
- ``slot_keys`` -- must equal ``contract.CANONICAL_SLOT_KEYS`` exactly
  (identity and order).
- ``metadata`` -- JSON-safe provenance (source/date identifiers, raw-quote
  provenance for real surfaces, parameter vectors for synthetic surfaces).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .contract import (
    CANONICAL_SLOT_KEYS,
    MASKED_PRICE_PLACEHOLDER,
    NOMINAL_SLOT_COUNT,
    R2_EXPIRY_RANKS,
    RepresentationContractError,
    SlotKey,
    validate_slot_keys,
    validate_vector_length,
)

SOURCE_SYNTHETIC: str = "synthetic_canonical_double_heston_production_pricer"
SOURCE_REAL_NSE_DEVELOPMENT: str = "real_nse_official_contract_development_date"


@dataclass(frozen=True)
class R2Conditioning:
    """Actual two-rank market geometry conditioning of one R2 surface.

    ``dte``/``rates``/``carries`` are indexed by representation expiry rank
    (first eligible listed expiry = rank 1, second = rank 2); the values are
    identical for every moneyness and option type within a rank.
    """

    date_id: str
    spot: float
    expiry_dates: tuple[str, str]
    dte: tuple[int, int]
    rates: tuple[float, float]
    carries: tuple[float, float]

    def __post_init__(self) -> None:
        if not str(self.date_id):
            raise RepresentationContractError("conditioning date_id must be non-empty")
        _require_finite_positive(self.spot, "conditioning spot")
        expiry_dates = tuple(str(value) for value in self.expiry_dates)
        dte = tuple(int(value) for value in self.dte)
        rates = tuple(float(value) for value in self.rates)
        carries = tuple(float(value) for value in self.carries)
        for name, values in (("expiry_dates", expiry_dates), ("dte", dte), ("rates", rates), ("carries", carries)):
            if len(values) != len(R2_EXPIRY_RANKS):
                raise RepresentationContractError(
                    f"conditioning {name} must hold exactly {len(R2_EXPIRY_RANKS)} "
                    f"values (one per representation rank), received {len(values)}"
                )
        if len(set(expiry_dates)) != len(expiry_dates):
            raise RepresentationContractError("conditioning expiry dates must be distinct")
        if dte[0] <= 0 or dte[1] <= 0:
            raise RepresentationContractError("conditioning dte values must be positive")
        if dte[1] <= dte[0]:
            raise RepresentationContractError(
                "conditioning dte must be strictly increasing across ranks "
                "(listed expiry ranks are chronological)"
            )
        for name, values in (("rates", rates), ("carries", carries)):
            for value in values:
                if not math.isfinite(value):
                    raise RepresentationContractError(f"conditioning {name} must be finite")
        object.__setattr__(self, "expiry_dates", expiry_dates)
        object.__setattr__(self, "dte", dte)
        object.__setattr__(self, "rates", rates)
        object.__setattr__(self, "carries", carries)

    @property
    def maturities(self) -> tuple[float, float]:
        """Actual time-to-maturity in years per representation rank."""
        return (self.dte[0] / 365.0, self.dte[1] / 365.0)

    @classmethod
    def from_date_profile(cls, profile: Any) -> "R2Conditioning":
        """Adapt a G2 ``DateProfile`` (or equivalent attribute carrier).

        Uses the first two eligible ranks of the profile.  The synthetic spot
        normalization is the G2 convention ``spot = 100`` (Black-Scholes
        homogeneity makes spot-normalized target-moneyness prices independent
        of the normalization spot), NOT the profile's market spot, matching
        ``src/g2_r2r3/geometry.build_geometry``'s synthetic convention.
        """
        return cls(
            date_id=str(profile.date_id),
            spot=SYNTHETIC_NORMALIZATION_SPOT,
            expiry_dates=(str(profile.expiry_dates[0]), str(profile.expiry_dates[1])),
            dte=(int(profile.dte[0]), int(profile.dte[1])),
            rates=(float(profile.rates[0]), float(profile.rates[1])),
            carries=(float(profile.carries[0]), float(profile.carries[1])),
        )


# Equal by test to src.g2_r2r3.frozen.SYNTHETIC_SPOT; normalization-only spot.
SYNTHETIC_NORMALIZATION_SPOT: float = 100.0


@dataclass(frozen=True)
class R2Surface:
    """One canonical frozen-R2 option surface (20 nominal slots + mask)."""

    prices: tuple[float, ...]
    mask: tuple[bool, ...]
    maturities: tuple[float, ...]
    rates: tuple[float, ...]
    carries: tuple[float, ...]
    spot: float
    surface_id: str
    source: str
    slot_keys: tuple[SlotKey, ...] = CANONICAL_SLOT_KEYS
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        canonical = validate_slot_keys(self.slot_keys)
        object.__setattr__(self, "slot_keys", canonical)
        for name in ("prices", "mask", "maturities", "rates", "carries"):
            validate_vector_length(getattr(self, name), name)
        if not str(self.surface_id):
            raise RepresentationContractError("surface_id must be non-empty")
        if not str(self.source):
            raise RepresentationContractError("source must be non-empty")
        _require_finite_positive(self.spot, "spot")
        prices = tuple(float(value) for value in self.prices)
        mask = tuple(bool(value) for value in self.mask)
        maturities = tuple(float(value) for value in self.maturities)
        rates = tuple(float(value) for value in self.rates)
        carries = tuple(float(value) for value in self.carries)
        for name, values in (("prices", prices), ("maturities", maturities), ("rates", rates), ("carries", carries)):
            for value in values:
                if not math.isfinite(value):
                    raise RepresentationContractError(
                        f"{name} must be finite (NaN/Inf never encode missingness; "
                        "masked slots are 0.0 with mask=False)"
                    )
        for index, (price, valid) in enumerate(zip(prices, mask, strict=True)):
            if valid and price <= 0.0:
                raise RepresentationContractError(
                    f"valid slot {index} carries non-positive price {price}; valid "
                    "observed/synthetic prices are strictly positive, so 0.0 is "
                    "reserved for masked slots"
                )
            if not valid and price != MASKED_PRICE_PLACEHOLDER:
                raise RepresentationContractError(
                    f"masked slot {index} carries {price}; masked slots must hold "
                    "exactly the 0.0 placeholder and must never carry an imputed, "
                    "interpolated, or neighboring value"
                )
        for value in maturities:
            if value <= 0.0:
                raise RepresentationContractError("maturities must be strictly positive")
        per_rank = self._per_rank_rank_slices()
        for rank, indices in per_rank.items():
            for name, values in (("maturities", maturities), ("rates", rates), ("carries", carries)):
                first = values[indices[0]]
                for index in indices:
                    if values[index] != first:
                        raise RepresentationContractError(
                            f"{name} must be constant within expiry rank {rank} "
                            f"(position {index} differs from position {indices[0]})"
                        )
        if per_rank and maturities[per_rank[2][0]] <= maturities[per_rank[1][0]]:
            raise RepresentationContractError(
                "rank-2 maturity must exceed rank-1 maturity (chronological ranks)"
            )
        object.__setattr__(self, "prices", prices)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "maturities", maturities)
        object.__setattr__(self, "rates", rates)
        object.__setattr__(self, "carries", carries)

    @staticmethod
    def _per_rank_rank_slices() -> dict[int, list[int]]:
        return {
            rank: [
                index
                for index, key in enumerate(CANONICAL_SLOT_KEYS)
                if key.expiry_rank == rank
            ]
            for rank in R2_EXPIRY_RANKS
        }

    # -- array accessors (consumers must gate prices by mask) -----------------

    def prices_array(self) -> np.ndarray:
        return np.asarray(self.prices, dtype=np.float64)

    def mask_array(self) -> np.ndarray:
        return np.asarray(self.mask, dtype=bool)

    def maturities_array(self) -> np.ndarray:
        return np.asarray(self.maturities, dtype=np.float64)

    def rates_array(self) -> np.ndarray:
        return np.asarray(self.rates, dtype=np.float64)

    def carries_array(self) -> np.ndarray:
        return np.asarray(self.carries, dtype=np.float64)

    def valid_prices_array(self) -> np.ndarray:
        """Prices at valid slots only (never includes masked placeholders)."""
        return self.prices_array()[self.mask_array()]

    def denormalized_prices_array(self) -> np.ndarray:
        """Prices x spot, with masked slots remaining exactly 0.0."""
        return np.where(self.mask_array(), self.prices_array() * self.spot, 0.0)

    def valid_slot_keys(self) -> tuple[SlotKey, ...]:
        return tuple(key for key, valid in zip(self.slot_keys, self.mask, strict=True) if valid)

    def masked_slot_keys(self) -> tuple[SlotKey, ...]:
        return tuple(key for key, valid in zip(self.slot_keys, self.mask, strict=True) if not valid)

    def usable_slot_count(self) -> int:
        return int(sum(self.mask))

    @property
    def slot_count(self) -> int:
        return NOMINAL_SLOT_COUNT


def _require_finite_positive(value: float, name: str) -> None:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise RepresentationContractError(f"{name} must be finite and strictly positive, got {value}")


def surface_from_vectors(
    prices: Sequence[float],
    mask: Sequence[bool],
    maturities: Sequence[float],
    rates: Sequence[float],
    carries: Sequence[float],
    *,
    spot: float,
    surface_id: str,
    source: str,
    slot_keys: Sequence[SlotKey] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> R2Surface:
    """Explicit flat-vector constructor with full contract validation.

    Vectors whose length is 108 (rejected legacy grid) or 30 (rejected R3
    study representation) are rejected with a diagnostic message rather than
    silently reinterpreted.
    """
    return R2Surface(
        prices=tuple(prices),
        mask=tuple(mask),
        maturities=tuple(maturities),
        rates=tuple(rates),
        carries=tuple(carries),
        spot=spot,
        surface_id=surface_id,
        source=source,
        slot_keys=tuple(slot_keys) if slot_keys is not None else CANONICAL_SLOT_KEYS,
        metadata=dict(metadata or {}),
    )


__all__ = [
    "R2Conditioning",
    "R2Surface",
    "SOURCE_REAL_NSE_DEVELOPMENT",
    "SOURCE_SYNTHETIC",
    "SYNTHETIC_NORMALIZATION_SPOT",
    "surface_from_vectors",
]
