"""Canonical constructor for synthetic R2 surfaces.

Synthetic R2 surfaces are complete by construction: exactly 20 nominal slots,
all mask values valid/true, actual maturity and per-rank rate/carry
conditioning preserved, and prices produced by the frozen production Double
Heston pricer (``src/double_heston.price_double_heston_surface`` via the
``src/pricing_interface`` adapter) — the canonical production pricing
contract, unchanged.

Because the production surface pricer accepts scalar rate/carry per call,
the 20 target-moneyness quotes are priced per constant-conditioning rank
piece (the same piece-wise approach the committed G2 diagnostics used); the
two pieces are scattered back into canonical slot positions.

This module deliberately does NOT depend on the sealed G2 study package:
parameter validation uses the canonical constraint contract, pricing uses the
production engine, and slot identity comes from :mod:`.contract` (kept equal
to the frozen study constants by test).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ..constants import PARAMETER_NAMES
from ..constraints import validate_parameters
from ..double_heston import price_double_heston_surface
from .contract import (
    CANONICAL_SLOT_KEYS,
    CENTRAL_FIVE_LOG_MONEYNESS,
    R2_EXPIRY_RANKS,
    RepresentationContractError,
    SlotKey,
)
from .surface import (
    R2Conditioning,
    R2Surface,
    SOURCE_SYNTHETIC,
    normalize_metadata_mapping,
)


def build_synthetic_surface(
    parameters: Sequence[float] | Mapping[str, float],
    conditioning: R2Conditioning,
    *,
    surface_id: str,
    metadata: Mapping[str, Any] | None = None,
    node_count: int = 64,
) -> R2Surface:
    """Build one complete synthetic R2 surface with the production pricer.

    ``parameters`` is the canonical ten-parameter vector in the frozen order
    and must satisfy the canonical structural constraints (validated — no
    silent clamping).  Prices are spot-normalized (price / conditioning.spot)
    at the five target moneyness strikes ``spot * exp(k)`` per rank.

    Caller-supplied ``metadata`` is provenance-protected: it is stored under
    the ``user_metadata`` namespace and can never overwrite the authoritative
    scientific fields (``synthetic``, ``parameters_canonical_order``,
    ``pricing_engine``, ``node_count``, ``target_moneyness_strikes``,
    ``date_conditioning_id``, ``expiry_dates``, ``dte``, ``imputation``).
    """
    vector = np.asarray(
        [parameters[name] for name in PARAMETER_NAMES]
        if isinstance(parameters, Mapping)
        else parameters,
        dtype=np.float64,
    )
    diagnostics = validate_parameters(vector)
    if not diagnostics["is_valid"]:
        raise RepresentationContractError(
            f"synthetic surface parameters violate the canonical constraints: "
            f"{diagnostics['violations']}"
        )

    spot = conditioning.spot
    prices = np.zeros(len(CANONICAL_SLOT_KEYS), dtype=np.float64)
    for rank in R2_EXPIRY_RANKS:
        indices = [
            index
            for index, key in enumerate(CANONICAL_SLOT_KEYS)
            if key.expiry_rank == rank
        ]
        rank_keys: list[SlotKey] = [CANONICAL_SLOT_KEYS[index] for index in indices]
        strikes = np.asarray(
            [spot * float(np.exp(key.target_log_moneyness)) for key in rank_keys],
            dtype=np.float64,
        )
        maturities = np.full(
            len(rank_keys), conditioning.maturities[rank - 1], dtype=np.float64
        )
        option_types = [key.option_type for key in rank_keys]
        rank_prices = price_double_heston_surface(
            spot,
            strikes,
            maturities,
            conditioning.rates[rank - 1],
            conditioning.carries[rank - 1],
            option_types,
            vector,
            node_count=node_count,
        )
        if rank_prices.shape != (len(rank_keys),) or not np.isfinite(rank_prices).all():
            raise RepresentationContractError(
                f"production pricer failed shape/finite checks for rank {rank}"
            )
        prices[np.asarray(indices, dtype=int)] = rank_prices

    normalized = prices / spot
    parameters_record: dict[str, float] = {
        name: float(value) for name, value in zip(PARAMETER_NAMES, vector, strict=True)
    }
    surface_metadata: dict[str, Any] = {
        "synthetic": True,
        "parameters_canonical_order": parameters_record,
        "pricing_engine": "production_double_heston_unchanged",
        "node_count": int(node_count),
        "target_moneyness_strikes": True,
        "date_conditioning_id": conditioning.date_id,
        "expiry_dates": list(conditioning.expiry_dates),
        "dte": list(conditioning.dte),
        "imputation": "NONE_COMPLETE_BY_CONSTRUCTION",
    }
    # Authoritative provenance protection: caller metadata can never
    # overwrite the reserved scientific fields above.  Caller-supplied
    # metadata is stored under a separate "user_metadata" namespace; the
    # namespace key itself is reserved so a caller cannot nest a conflicting
    # namespace inside itself.
    user_metadata = {} if metadata is None else normalize_metadata_mapping(metadata)
    if "user_metadata" in user_metadata:
        raise RepresentationContractError(
            "caller metadata key 'user_metadata' is reserved; caller metadata "
            "is stored under that namespace and cannot redefine it"
        )
    if user_metadata:
        surface_metadata["user_metadata"] = user_metadata
    return R2Surface(
        prices=tuple(float(value) for value in normalized),
        mask=tuple(True for _ in CANONICAL_SLOT_KEYS),
        maturities=tuple(
            conditioning.maturities[key.expiry_rank - 1] for key in CANONICAL_SLOT_KEYS
        ),
        rates=tuple(
            conditioning.rates[key.expiry_rank - 1] for key in CANONICAL_SLOT_KEYS
        ),
        carries=tuple(
            conditioning.carries[key.expiry_rank - 1] for key in CANONICAL_SLOT_KEYS
        ),
        spot=spot,
        surface_id=surface_id,
        source=SOURCE_SYNTHETIC,
        metadata=surface_metadata,
    )


__all__ = ["build_synthetic_surface", "CENTRAL_FIVE_LOG_MONEYNESS"]
