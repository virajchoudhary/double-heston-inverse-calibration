"""Stable serialized form of canonical R2 surfaces (versioned JSON schema).

Serialized payload (one object per surface; see
``docs/R2_REPRESENTATION_CONTRACT.md`` section "Serialization schema"):

    representation_name      "FROZEN_R2_RANKED_TWO_EXPIRY_CENTRAL_FIVE"
    representation_version   "1.0"
    slot_keys                20 triples [expiry_rank, target_log_moneyness, option_type]
    prices                   20 spot-normalized prices (masked slots exactly 0.0)
    mask                     20 booleans (true = usable observation)
    maturities               20 actual times-to-maturity in years
    rates                    20 per-slot rate-conditioning values
    carries                  20 per-slot carry-conditioning values
    spot                     normalization spot
    surface_id               unique surface identifier
    source                   provenance label (synthetic / real development)
    metadata                 JSON-safe provenance dict

Round-trip guarantees (tested): ``payload_to_surface(surface_to_payload(s))``
reproduces every contract field of ``s`` exactly — slot identity and order,
prices, mask, maturities, rates, carries, spot — with float64 values
bit-identical (Python's JSON float repr round-trips float64 exactly) and
``allow_nan=False`` everywhere so NaN/Inf can never enter a payload.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contract import (
    CANONICAL_SLOT_KEYS,
    MASKED_PRICE_PLACEHOLDER,
    NOMINAL_SLOT_COUNT,
    REPRESENTATION_NAME,
    REPRESENTATION_VERSION,
    RepresentationContractError,
    SlotKey,
    validate_slot_keys,
    validate_vector_length,
)
from .surface import R2Surface

_PAYLOAD_FIELDS: tuple[str, ...] = (
    "representation_name",
    "representation_version",
    "slot_keys",
    "prices",
    "mask",
    "maturities",
    "rates",
    "carries",
    "spot",
    "surface_id",
    "source",
    "metadata",
)


def surface_to_payload(surface: R2Surface) -> dict[str, Any]:
    """Serialize one surface to a JSON-safe payload dict."""
    payload: dict[str, Any] = {
        "representation_name": REPRESENTATION_NAME,
        "representation_version": REPRESENTATION_VERSION,
        "slot_keys": [
            [int(key.expiry_rank), float(key.target_log_moneyness), str(key.option_type)]
            for key in surface.slot_keys
        ],
        "prices": [float(value) for value in surface.prices],
        "mask": [bool(value) for value in surface.mask],
        "maturities": [float(value) for value in surface.maturities],
        "rates": [float(value) for value in surface.rates],
        "carries": [float(value) for value in surface.carries],
        "spot": float(surface.spot),
        "surface_id": str(surface.surface_id),
        "source": str(surface.source),
        "metadata": _normalize_metadata(surface.metadata),
    }
    _require_serializable(payload)
    return payload


def _normalize_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """JSON-normalize metadata (tuples -> lists) with contract error typing."""
    try:
        return json.loads(json.dumps(dict(metadata), allow_nan=False))
    except (TypeError, ValueError) as error:
        raise RepresentationContractError(
            f"surface metadata is not finite-number JSON-serializable: {error}"
        ) from None


def validate_payload(payload: Mapping[str, Any]) -> None:
    """Validate a payload dict against the versioned schema; raise on drift."""
    if not isinstance(payload, Mapping):
        raise RepresentationContractError("payload must be a JSON object/dict")
    missing = [field for field in _PAYLOAD_FIELDS if field not in payload]
    if missing:
        raise RepresentationContractError(f"payload is missing fields: {missing}")
    name = payload["representation_name"]
    if name != REPRESENTATION_NAME:
        raise RepresentationContractError(
            f"payload representation_name {name!r} is not the canonical R2 "
            f"representation {REPRESENTATION_NAME!r}"
        )
    version = payload["representation_version"]
    if version != REPRESENTATION_VERSION:
        raise RepresentationContractError(
            f"payload representation_version {version!r} is not supported; this "
            f"interface implements {REPRESENTATION_VERSION!r} (an explicit version "
            "migration is required for any other schema)"
        )
    slot_keys = _payload_slot_keys(payload["slot_keys"])
    validate_slot_keys(slot_keys)
    for field in ("prices", "mask", "maturities", "rates", "carries"):
        validate_vector_length(payload[field], f"payload {field}")
    mask = payload["mask"]
    for index, value in enumerate(mask):
        if not isinstance(value, bool):
            raise RepresentationContractError(
                f"payload mask[{index}] must be a JSON boolean, got {type(value).__name__}"
            )
    prices = payload["prices"]
    for index, (price, valid) in enumerate(zip(prices, mask, strict=True)):
        if not isinstance(price, (int, float)) or isinstance(price, bool) or not math.isfinite(price):
            raise RepresentationContractError(
                f"payload prices[{index}] must be a finite number"
            )
        if valid and price <= 0.0:
            raise RepresentationContractError(
                f"payload prices[{index}] is valid but non-positive; valid prices "
                "are strictly positive"
            )
        if not valid and price != MASKED_PRICE_PLACEHOLDER:
            raise RepresentationContractError(
                f"payload prices[{index}] is masked but carries {price}; masked "
                "slots must hold exactly 0.0 and never an imputed value"
            )
    for field in ("maturities", "rates", "carries"):
        for index, value in enumerate(payload[field]):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise RepresentationContractError(
                    f"payload {field}[{index}] must be a finite number"
                )
    if any(value <= 0.0 for value in payload["maturities"]):
        raise RepresentationContractError("payload maturities must be strictly positive")
    if not isinstance(payload["spot"], (int, float)) or isinstance(payload["spot"], bool) or payload["spot"] <= 0.0 or not math.isfinite(payload["spot"]):
        raise RepresentationContractError("payload spot must be finite and strictly positive")
    if not str(payload["surface_id"]):
        raise RepresentationContractError("payload surface_id must be non-empty")
    if not str(payload["source"]):
        raise RepresentationContractError("payload source must be non-empty")
    if not isinstance(payload["metadata"], Mapping):
        raise RepresentationContractError("payload metadata must be a JSON object/dict")
    # Metadata VALUES must also be finite-number JSON-safe: NaN/Inf can enter
    # a JSON file through Python's permissive NaN/Infinity literals, and the
    # contract forbids non-finite values anywhere in a surface.
    try:
        json.dumps(dict(payload["metadata"]), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise RepresentationContractError(
            f"payload metadata is not finite-number JSON-serializable: {error}"
        ) from None


def payload_to_surface(payload: Mapping[str, Any]) -> R2Surface:
    """Deserialize a validated payload into a canonical R2Surface."""
    validate_payload(payload)
    return R2Surface(
        prices=tuple(payload["prices"]),
        mask=tuple(payload["mask"]),
        maturities=tuple(payload["maturities"]),
        rates=tuple(payload["rates"]),
        carries=tuple(payload["carries"]),
        spot=float(payload["spot"]),
        surface_id=str(payload["surface_id"]),
        source=str(payload["source"]),
        slot_keys=_payload_slot_keys(payload["slot_keys"]),
        metadata=dict(payload["metadata"]),
    )


def write_surface_json(surface: R2Surface, path: str | Path) -> None:
    """Write one surface as deterministic JSON (sorted keys, indent 2)."""
    _write_json(surface_to_payload(surface), path)


def read_surface_json(path: str | Path) -> R2Surface:
    """Read one surface from JSON, validating the full contract."""
    payload = _load_json(path)
    return payload_to_surface(payload)


def dataset_manifest(surfaces: Sequence[R2Surface]) -> dict[str, Any]:
    """Build a versioned dataset manifest payload over canonical surfaces."""
    payloads = [surface_to_payload(surface) for surface in surfaces]
    if len(payloads) != len({payload["surface_id"] for payload in payloads}):
        raise RepresentationContractError("dataset manifest requires unique surface_ids")
    manifest = {
        "representation_name": REPRESENTATION_NAME,
        "representation_version": REPRESENTATION_VERSION,
        "slot_keys": payloads[0]["slot_keys"] if payloads else
        [[int(k.expiry_rank), float(k.target_log_moneyness), str(k.option_type)] for k in CANONICAL_SLOT_KEYS],
        "surface_count": len(payloads),
        "surfaces": payloads,
    }
    for payload in payloads:
        if payload["slot_keys"] != manifest["slot_keys"]:
            raise RepresentationContractError(
                "manifest contains a surface whose slot keys deviate from the "
                "canonical order"
            )
    _require_serializable(manifest)
    return manifest


def manifest_from_payload(payload: Mapping[str, Any]) -> list[R2Surface]:
    """Reconstruct surfaces from a dataset manifest payload."""
    if not isinstance(payload, Mapping):
        raise RepresentationContractError("manifest must be a JSON object/dict")
    for field in ("representation_name", "representation_version", "slot_keys", "surfaces"):
        if field not in payload:
            raise RepresentationContractError(f"manifest is missing field {field!r}")
    if payload["representation_name"] != REPRESENTATION_NAME:
        raise RepresentationContractError(
            f"manifest representation_name {payload['representation_name']!r} is "
            f"not canonical"
        )
    if payload["representation_version"] != REPRESENTATION_VERSION:
        raise RepresentationContractError(
            f"manifest representation_version {payload['representation_version']!r} "
            "is not supported"
        )
    manifest_slot_keys = _payload_slot_keys(payload["slot_keys"])
    validate_slot_keys(manifest_slot_keys)
    surfaces = [payload_to_surface(entry) for entry in payload["surfaces"]]
    ids = [surface.surface_id for surface in surfaces]
    if len(ids) != len(set(ids)):
        raise RepresentationContractError("manifest contains duplicate surface_ids")
    if payload.get("surface_count") is not None and int(payload["surface_count"]) != len(surfaces):
        raise RepresentationContractError("manifest surface_count does not match surfaces")
    return surfaces


def write_manifest_json(surfaces: Sequence[R2Surface], path: str | Path) -> None:
    """Write a dataset manifest as deterministic JSON."""
    _write_json(dataset_manifest(surfaces), path)


def read_manifest_json(path: str | Path) -> list[R2Surface]:
    """Read a dataset manifest, validating every surface."""
    return manifest_from_payload(_load_json(path))


def _load_json(path: str | Path) -> Any:
    """Load JSON, refusing the permissive NaN/Infinity literals outright."""
    def _reject_constant(name: str) -> float:
        raise RepresentationContractError(
            f"{name} literal is not allowed anywhere in an R2 payload"
        )

    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle, parse_constant=_reject_constant)
    except json.JSONDecodeError as error:
        raise RepresentationContractError(f"invalid JSON: {error}") from None


def _payload_slot_keys(raw: Any) -> tuple[SlotKey, ...]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise RepresentationContractError("slot_keys must be a sequence of triples")
    keys: list[SlotKey] = []
    for entry in raw:
        if isinstance(entry, (str, bytes)) or not isinstance(entry, Sequence) or len(entry) != 3:
            raise RepresentationContractError(
                f"slot key {entry!r} must be a [rank, moneyness, option_type] triple"
            )
        rank, moneyness, option_type = entry
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise RepresentationContractError(f"slot key rank must be an integer: {entry!r}")
        if not isinstance(moneyness, (int, float)) or isinstance(moneyness, bool):
            raise RepresentationContractError(f"slot key moneyness must be numeric: {entry!r}")
        if not isinstance(option_type, str):
            raise RepresentationContractError(f"slot key option_type must be a string: {entry!r}")
        keys.append(SlotKey(int(rank), float(moneyness), option_type))
    return tuple(keys)


def _require_serializable(payload: Any) -> None:
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise RepresentationContractError(
            f"payload is not finite-number JSON-serializable: {error}"
        ) from None


def _write_json(payload: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


__all__ = [
    "NOMINAL_SLOT_COUNT",
    "dataset_manifest",
    "manifest_from_payload",
    "payload_to_surface",
    "read_manifest_json",
    "read_surface_json",
    "surface_to_payload",
    "validate_payload",
    "write_manifest_json",
    "write_surface_json",
]
