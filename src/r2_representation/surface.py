"""The canonical R2 surface object: 20 nominal slots plus explicit mask.

One immutable :class:`R2Surface` is the single in-memory form shared by
synthetic generation, ANN/Model-2 dataset construction, and frozen
real-market evaluation.  Field semantics (see
``docs/R2_REPRESENTATION_CONTRACT.md``):

- ``prices`` -- 20 spot-normalized option prices (price / spot).  Valid slots
  carry strictly positive finite values; masked (unavailable) slots carry
  exactly ``0.0`` (``contract.MASKED_PRICE_PLACEHOLDER``).  NaN/Inf are never
  allowed anywhere.
- ``mask`` -- 20 validity flags, each a genuine boolean (``bool`` or
  ``numpy.bool_``).  Strings, numbers, and other truthy/falsy objects are
  REJECTED, never coerced — e.g. the string ``"False"`` must not silently
  become a valid observation.  ``True`` = real, usable observation (or
  synthetic slot, complete by construction); ``False`` = unsupported or
  unusable slot, never imputed.
- ``maturities`` / ``rates`` / ``carries`` -- per-slot actual time-to-maturity
  (years) and the existing per-rank rate/carry conditioning.  These are known
  for every slot of an eligible rank, including masked quote slots, so they
  are always 20 finite values.
- ``spot`` -- the normalization spot for this surface.
- ``slot_keys`` -- must equal ``contract.CANONICAL_SLOT_KEYS`` exactly
  (identity and order).
- ``metadata`` -- JSON-safe provenance, validated and defensively normalized
  at construction: a mapping of finite-number JSON-compatible values only
  (NaN/Infinity and unsupported Python objects raise
  ``RepresentationContractError``).  The surface stores a deeply immutable,
  normalized copy, so neither the caller's input nor a returned nested value
  can mutate recorded provenance.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
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
        mask = tuple(
            _require_mask_boolean(value, index)
            for index, value in enumerate(self.mask)
        )
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
        metadata = _prepared_stored_metadata(self.metadata)
        object.__setattr__(self, "prices", prices)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "maturities", maturities)
        object.__setattr__(self, "rates", rates)
        object.__setattr__(self, "carries", carries)
        object.__setattr__(self, "metadata", metadata)

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


def _require_mask_boolean(value: Any, index: int) -> bool:
    """Accept only genuine booleans; never coerce truthy/falsy objects.

    ``bool`` covers Python literals and JSON round-trips; ``np.bool_``
    covers numpy mask arrays used by normal callers (note ``np.bool_`` is
    NOT a subclass of ``bool``).  Everything else — strings such as
    ``"False"``/``"True"``, ints 0/1, floats, None, arbitrary objects — is
    rejected so a malformed value can never silently flip a missing quote
    into a valid observation.
    """
    if isinstance(value, bool) or isinstance(value, np.bool_):
        return bool(value)
    raise RepresentationContractError(
        f"mask[{index}] must be a genuine boolean (bool or numpy.bool_), got "
        f"{type(value).__name__} ({value!r}); truthy/falsy coercion of mask "
        "values is forbidden (e.g. the string \"False\" must never become a "
        "valid observation)"
    )


def normalize_metadata_mapping(metadata: Any) -> dict[str, Any]:
    """Validate metadata as finite-number JSON-safe and return a plain copy."""
    if not isinstance(metadata, Mapping):
        raise RepresentationContractError(
            f"metadata must be a mapping/JSON object, got {type(metadata).__name__}"
        )

    def normalize(value: Any) -> Any:
        if isinstance(value, (Mapping, list, tuple)):
            marker = id(value)
            if marker in active_containers:
                raise RepresentationContractError(
                    "metadata must be acyclic; circular references are forbidden"
                )
            active_containers.add(marker)
            try:
                if isinstance(value, Mapping):
                    result: dict[str, Any] = {}
                    for key, item in value.items():
                        if not isinstance(key, str):
                            raise RepresentationContractError(
                                f"metadata keys must be strings, got {type(key).__name__}"
                            )
                        result[key] = normalize(item)
                    return result
                return [normalize(item) for item in value]
            finally:
                active_containers.discard(marker)
        if value is None or type(value) in (bool, int, float, str):
            if isinstance(value, float) and not math.isfinite(value):
                raise RepresentationContractError(
                    "metadata numbers must be finite (NaN/Infinity are forbidden)"
                )
            return value
        raise RepresentationContractError(
            f"metadata contains unsupported non-JSON value of type "
            f"{type(value).__name__}; values are never stringified or lossily coerced"
        )

    active_containers: set[int] = set()
    try:
        normalized = normalize(metadata)
        try:
            json.dumps(normalized, allow_nan=False)
        except (TypeError, ValueError, RecursionError) as error:
            raise RepresentationContractError(
                f"metadata is not finite-number JSON-serializable: {error}"
            ) from error
        return normalized
    except RecursionError as error:
        raise RepresentationContractError(
            "metadata is too deeply nested or circular"
        ) from error


def _immutable_metadata(value: Any) -> Any:
    """Freeze validated JSON structure without changing its meaning."""
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _immutable_metadata(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_immutable_metadata(item) for item in value)
    return value


def _prepared_stored_metadata(metadata: Any) -> Any:
    """Validate then freeze metadata with one contract-typed depth boundary."""
    try:
        return _immutable_metadata(normalize_metadata_mapping(metadata))
    except RecursionError as error:
        raise RepresentationContractError(
            "metadata is too deeply nested"
        ) from error


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
        metadata={} if metadata is None else metadata,
    )


__all__ = [
    "R2Conditioning",
    "R2Surface",
    "SOURCE_REAL_NSE_DEVELOPMENT",
    "SOURCE_SYNTHETIC",
    "SYNTHETIC_NORMALIZATION_SPOT",
    "normalize_metadata_mapping",
    "surface_from_vectors",
]
