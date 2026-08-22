"""Canonical slot-identity contract for the frozen post-G2 R2 representation.

G2 (executed and sealed 22 August 2026) froze R2 as the primary research
representation:

    G2_FINAL_REPRESENTATION = FROZEN_R2_RANKED_TWO_EXPIRY_CENTRAL_FIVE
    G2 = PASSED_REPRESENTATION_FROZEN_WITH_PRACTICAL_NON_IDENTIFIABILITY

R2 is the first TWO eligible listed expiry ranks x central-five log-moneyness
``[-0.10, -0.05, 0.00, +0.05, +0.10]`` x calls and puts = exactly 20 NOMINAL
spot-normalized price slots, with actual time-to-maturity supplied explicitly
and existing per-rank rate/carry conditioning.  Real-market surfaces may
contain unsupported or unusable slots; those slots carry an explicit
mask/missingness flag and are NEVER imputed with a model price, interpolated,
or extrapolated.  Synthetic surfaces are complete by construction under the
frozen generator design.

This module is the single source of truth for slot identity, slot order, and
slot count for every downstream consumer (final synthetic-data generation,
ANN/Model-2 dataset loading, and frozen real-market evaluation).  The
constants deliberately mirror the sealed G2 study constants
(``src/g2_r2r3/frozen.py`` / ``src/g2_r2r3/geometry.py``); equality is
enforced by ``tests/test_r2_representation.py`` so the interface stays
self-contained without a runtime dependency on the sealed study module.

Canonical slot order (the reviewed G2 ordering, reused unchanged):

    option-type major (all calls, then all puts), then expiry rank, then
    target log-moneyness ascending:

    call/rank1/-0.10 .. call/rank1/+0.10   (slots  0- 4)
    call/rank2/-0.10 .. call/rank2/+0.10   (slots  5- 9)
    put /rank1/-0.10 .. put /rank1/+0.10   (slots 10-14)
    put /rank2/-0.10 .. put /rank2/+0.10   (slots 15-19)
"""

from __future__ import annotations

from typing import Final, NamedTuple

REPRESENTATION_NAME: Final[str] = "FROZEN_R2_RANKED_TWO_EXPIRY_CENTRAL_FIVE"
REPRESENTATION_VERSION: Final[str] = "1.0"

# The central-five targets of the frozen G2 contract.  Equal by test to
# src.g2_r2r3.frozen.CENTRAL_FIVE; do not change one without the other.
CENTRAL_FIVE_LOG_MONEYNESS: Final[tuple[float, ...]] = (
    -0.10, -0.05, 0.0, 0.05, 0.10,
)
R2_EXPIRY_RANKS: Final[tuple[int, ...]] = (1, 2)
NOMINAL_SLOT_COUNT: Final[int] = len(R2_EXPIRY_RANKS) * len(CENTRAL_FIVE_LOG_MONEYNESS) * 2  # 20

CALL: Final[str] = "call"
PUT: Final[str] = "put"
OPTION_TYPE_ORDER: Final[tuple[str, str]] = (CALL, PUT)

# The rejected historical grid.  Vectors or slot tables of this length can
# never be silently reinterpreted as canonical R2.
LEGACY_108_INPUT_SIZE: Final[int] = 108
# The rejected G2 candidate representation size (R3), rejected for the same
# reason: it is not the frozen primary representation.
REJECTED_R3_INPUT_SIZE: Final[int] = 30

# Numeric placeholder stored at masked (unavailable) slots.  Valid observed
# prices are strictly positive under the official-NSE activity contract and
# synthetic canonical prices are strictly positive by construction, so 0.0 can
# never collide with a real observation.  A masked slot's 0.0 is a dense
# serialization placeholder only -- never an observation; consumers must gate
# every use of ``prices`` by ``mask``.  NaN is not used for missingness.
MASKED_PRICE_PLACEHOLDER: Final[float] = 0.0


class RepresentationContractError(ValueError):
    """Raised when data violates the canonical R2 representation contract."""


class SlotKey(NamedTuple):
    """Canonical identity of one nominal R2 slot."""

    expiry_rank: int  # 1-based rank among the first two eligible listed expiries
    target_log_moneyness: float
    option_type: str  # "call" or "put"


def canonical_slot_keys() -> tuple[SlotKey, ...]:
    """The 20 canonical slot keys in canonical order (fresh tuple per call)."""
    keys: list[SlotKey] = []
    for option_type in OPTION_TYPE_ORDER:
        for rank in R2_EXPIRY_RANKS:
            for moneyness in CENTRAL_FIVE_LOG_MONEYNESS:
                keys.append(
                    SlotKey(
                        expiry_rank=rank,
                        target_log_moneyness=moneyness,
                        option_type=option_type,
                    )
                )
    if len(keys) != NOMINAL_SLOT_COUNT:
        raise RepresentationContractError("canonical slot construction size mismatch")
    return tuple(keys)


CANONICAL_SLOT_KEYS: Final[tuple[SlotKey, ...]] = canonical_slot_keys()

_SLOT_INDEX: Final[dict[SlotKey, int]] = {key: index for index, key in enumerate(CANONICAL_SLOT_KEYS)}


def slot_index(key: SlotKey) -> int:
    """Canonical position of ``key``; raises if the key is not canonical."""
    try:
        return _SLOT_INDEX[key]
    except KeyError:
        known = ", ".join(
            f"({item.expiry_rank}, {item.target_log_moneyness:+.2f}, {item.option_type})"
            for item in CANONICAL_SLOT_KEYS
        )
        raise RepresentationContractError(
            f"non-canonical slot key {key}; canonical keys are {known}"
        ) from None


def _length_diagnostic(length: int) -> str:
    if length == LEGACY_108_INPUT_SIZE:
        return (
            f"length {length} matches the rejected legacy 108-input grid "
            "(9 log-moneyness x 6 fixed-DTE maturity x calls/puts). Legacy "
            "108-feature data cannot be silently reinterpreted as canonical "
            "R2; it is historical/provisional evidence only "
            "(CURRENT_108_GRID = REJECTED_AS_FINAL_UNCHANGED_GRID)."
        )
    if length == REJECTED_R3_INPUT_SIZE:
        return (
            f"length {length} matches the rejected R3 study representation "
            "(three expiry ranks). Only R2 (20 slots) is canonical."
        )
    return f"expected exactly {NOMINAL_SLOT_COUNT} slots, received {length}."


def validate_slot_keys(keys: object) -> tuple[SlotKey, ...]:
    """Validate a slot-key sequence against the canonical order.

    Returns the canonical tuple on success; raises
    ``RepresentationContractError`` on any count, identity, or order mismatch.
    """
    if isinstance(keys, (str, bytes)):
        raise RepresentationContractError("slot keys must be a sequence of key triples, not text")
    try:
        materialized = tuple(keys)  # type: ignore[arg-type]
    except TypeError as error:
        raise RepresentationContractError(f"slot keys are not iterable: {error}") from None
    if len(materialized) != NOMINAL_SLOT_COUNT:
        raise RepresentationContractError(_length_diagnostic(len(materialized)))
    for index, (received, expected) in enumerate(zip(materialized, CANONICAL_SLOT_KEYS, strict=True)):
        try:
            mismatch = bool(received != expected)
        except (ValueError, TypeError):
            raise RepresentationContractError(
                f"slot key at position {index} is not comparable to a canonical "
                f"SlotKey: {received!r}"
            ) from None
        if mismatch:
            raise RepresentationContractError(
                "slot ordering violates the canonical R2 contract at position "
                f"{index}: received {received}, expected {expected}. The "
                "canonical order is option-type major (call, put), then expiry "
                "rank, then target log-moneyness ascending."
            )
    return materialized


def validate_vector_length(values: object, field_name: str) -> None:
    """Reject per-slot vectors whose length is not exactly the nominal count."""
    if isinstance(values, (str, bytes)):
        raise RepresentationContractError(f"{field_name} must be numeric, not text")
    try:
        length = len(values)  # type: ignore[arg-type]
    except TypeError as error:
        raise RepresentationContractError(f"{field_name} has no length: {error}") from None
    if length != NOMINAL_SLOT_COUNT:
        raise RepresentationContractError(
            f"{field_name}: {_length_diagnostic(length)}"
        )


__all__ = [
    "CALL",
    "CANONICAL_SLOT_KEYS",
    "CENTRAL_FIVE_LOG_MONEYNESS",
    "LEGACY_108_INPUT_SIZE",
    "MASKED_PRICE_PLACEHOLDER",
    "NOMINAL_SLOT_COUNT",
    "OPTION_TYPE_ORDER",
    "PUT",
    "REJECTED_R3_INPUT_SIZE",
    "REPRESENTATION_NAME",
    "REPRESENTATION_VERSION",
    "R2_EXPIRY_RANKS",
    "RepresentationContractError",
    "SlotKey",
    "canonical_slot_keys",
    "slot_index",
    "validate_slot_keys",
    "validate_vector_length",
]
