"""Canonical frozen-R2 representation interface (post-G2 production layer).

G2 froze R2 as the primary research representation on 22 August 2026
(``G2_FINAL_REPRESENTATION = FROZEN_R2_RANKED_TWO_EXPIRY_CENTRAL_FIVE``).
This package is the single canonical software contract for that
representation, shared by final synthetic-data generation, ANN/Model-2
dataset construction, and frozen real-market evaluation.

Scientific contract (see ``docs/R2_REPRESENTATION_CONTRACT.md``):

- 20 NOMINAL slots: first two eligible listed expiry ranks x central-five
  log-moneyness [-0.10, -0.05, 0.00, +0.05, +0.10] x calls and puts;
- spot-normalized prices; actual time-to-maturity supplied explicitly;
- existing per-rank rate/carry conditioning;
- explicit validity mask — a missing real quote is never imputed with a
  model price, interpolation, extrapolation, or neighboring observations;
- synthetic surfaces complete by construction;
- canonical ten-parameter order/constraints and the production pricer are
  unchanged;
- the rejected legacy 108-input grid and the rejected R3 study
  representation can never pass as canonical R2.
"""

from __future__ import annotations

from .contract import (
    CANONICAL_SLOT_KEYS,
    CENTRAL_FIVE_LOG_MONEYNESS,
    LEGACY_108_INPUT_SIZE,
    MASKED_PRICE_PLACEHOLDER,
    NOMINAL_SLOT_COUNT,
    OPTION_TYPE_ORDER,
    REJECTED_R3_INPUT_SIZE,
    REPRESENTATION_NAME,
    REPRESENTATION_VERSION,
    R2_EXPIRY_RANKS,
    RepresentationContractError,
    SlotKey,
    canonical_slot_keys,
    slot_index,
    validate_slot_keys,
    validate_vector_length,
)
from .real import RealSurfaceNotConstructibleError, build_real_surface
from .serialization import (
    dataset_manifest,
    manifest_from_payload,
    payload_to_surface,
    read_manifest_json,
    read_surface_json,
    surface_to_payload,
    validate_payload,
    write_manifest_json,
    write_surface_json,
)
from .surface import (
    R2Conditioning,
    R2Surface,
    SOURCE_REAL_NSE_DEVELOPMENT,
    SOURCE_SYNTHETIC,
    SYNTHETIC_NORMALIZATION_SPOT,
    surface_from_vectors,
)
from .synthetic import build_synthetic_surface

__all__ = [
    "CANONICAL_SLOT_KEYS",
    "CENTRAL_FIVE_LOG_MONEYNESS",
    "LEGACY_108_INPUT_SIZE",
    "MASKED_PRICE_PLACEHOLDER",
    "NOMINAL_SLOT_COUNT",
    "OPTION_TYPE_ORDER",
    "R2Conditioning",
    "R2Surface",
    "REJECTED_R3_INPUT_SIZE",
    "REPRESENTATION_NAME",
    "REPRESENTATION_VERSION",
    "R2_EXPIRY_RANKS",
    "RealSurfaceNotConstructibleError",
    "RepresentationContractError",
    "SOURCE_REAL_NSE_DEVELOPMENT",
    "SOURCE_SYNTHETIC",
    "SYNTHETIC_NORMALIZATION_SPOT",
    "SlotKey",
    "build_real_surface",
    "build_synthetic_surface",
    "canonical_slot_keys",
    "dataset_manifest",
    "manifest_from_payload",
    "payload_to_surface",
    "read_manifest_json",
    "read_surface_json",
    "slot_index",
    "surface_from_vectors",
    "surface_to_payload",
    "validate_payload",
    "validate_slot_keys",
    "validate_vector_length",
    "write_manifest_json",
    "write_surface_json",
]
