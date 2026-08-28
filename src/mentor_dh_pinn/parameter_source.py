"""Frozen-data parameter provenance for the mentor baseline.

The baseline does not infer parameters from prices.  It selects the first
stored TRAIN record from the frozen synthetic surfaces file, validates its
canonical vector with :mod:`src.constraints`, and uses that record's eight
structural values as the forward-pricing source.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from src.constants import PARAMETER_NAMES
from src.constraints import validate_parameters, vector_to_dictionary

CANONICAL_PARAMETER_ORDER = tuple(PARAMETER_NAMES)
FROZEN_SURFACES_SHA256 = (
    "148b579a4f6ce572e34796e872479c4c016c89bbcd20438c2bb62d6b6960f1f6"
)
EXPECTED_FIRST_PARAMETER_HASH = (
    "b262584d6c2e76f7b46d635f580a5a00c7cb195e28a3536f18303e856704e8cd"
)
EXPECTED_FIRST_SURFACE_ID = "R2_FINAL_interior_train_000000"
EXPECTED_FIRST_PARAMETER_VECTOR = np.asarray(
    [
        1.3674937560587115,
        0.11602756036897444,
        0.3618711734277465,
        -0.14696793790549395,
        0.10372660942325253,
        2.6470842008378765,
        0.07208681877078564,
        0.18961705430448939,
        -0.3792663779420725,
        0.07926603504504037,
    ],
    dtype=np.float64,
)


def sha256_file(path: str | Path) -> str:
    """Return a lowercase SHA256 digest without altering the file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_surface_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield JSONL records while rejecting malformed/non-object rows."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at line {line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"surface record at line {line_number} is not an object")
            yield record


def _metadata(record: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("surface record has no metadata mapping")
    return metadata


def _user_metadata(record: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _metadata(record)
    user_metadata = metadata.get("user_metadata")
    return user_metadata if isinstance(user_metadata, Mapping) else {}


def parameter_vector_from_record(record: Mapping[str, Any]) -> np.ndarray:
    """Extract a canonical float64 vector from frozen record metadata."""
    metadata_parameters = _metadata(record).get("parameters_canonical_order")
    if not isinstance(metadata_parameters, Mapping):
        raise ValueError("record metadata lacks parameters_canonical_order")
    missing = [name for name in CANONICAL_PARAMETER_ORDER if name not in metadata_parameters]
    if missing:
        raise ValueError(f"record parameter metadata missing {missing}")
    vector = np.asarray(
        [metadata_parameters[name] for name in CANONICAL_PARAMETER_ORDER],
        dtype=np.float64,
    )
    diagnostics = validate_parameters(vector)
    if not diagnostics["is_valid"]:
        raise ValueError(f"invalid stored parameter vector: {diagnostics['violations']}")
    return vector


@dataclass(frozen=True)
class ParameterSource:
    """Validated selected record and its provenance identities."""

    vector: np.ndarray
    record: dict[str, Any]
    dataset_path: str
    dataset_sha256: str
    parameter_hash: str
    surface_id: str
    split: str

    def __post_init__(self) -> None:
        vector = np.asarray(self.vector, dtype=np.float64)
        vector.setflags(write=False)
        object.__setattr__(self, "vector", vector)
        if vector.shape != (10,):
            raise ValueError("parameter source vector must have shape (10,)")
        if self.split != "train":
            raise ValueError("parameter source must come from the stored TRAIN split")
        diagnostics = validate_parameters(vector)
        if not diagnostics["is_valid"]:
            raise ValueError(f"parameter source is invalid: {diagnostics['violations']}")

    @property
    def mapping(self) -> dict[str, float]:
        return vector_to_dictionary(self.vector)

    @property
    def structural_vector(self) -> np.ndarray:
        """Return the eight non-v0 entries in canonical structural order."""
        indices = (0, 1, 2, 3, 5, 6, 7, 8)
        values = np.asarray(self.vector[list(indices)], dtype=np.float64)
        values.setflags(write=False)
        return values

    def parameters_for_state(self, variance_slow: float, variance_fast: float) -> np.ndarray:
        """Copy structural values and substitute current variance states as v0."""
        if not np.isfinite(variance_slow) or not np.isfinite(variance_fast):
            raise ValueError("variance states must be finite")
        if variance_slow <= 0 or variance_fast <= 0:
            raise ValueError("variance states must be strictly positive")
        values = self.vector.copy()
        values[4] = variance_slow
        values[9] = variance_fast
        diagnostics = validate_parameters(values)
        if not diagnostics["is_valid"]:
            raise ValueError(f"state-conditioned parameters are invalid: {diagnostics['violations']}")
        return values

    def provenance(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "dataset_sha256": self.dataset_sha256,
            "surface_id": self.surface_id,
            "stored_split": self.split,
            "parameter_hash": self.parameter_hash,
            "canonical_parameter_order": list(CANONICAL_PARAMETER_ORDER),
            "parameter_vector": [float(value) for value in self.vector],
            "selection_rule": "first eligible stored TRAIN record passing validate_parameters",
        }


def select_first_eligible_train_record(
    dataset_path: str | Path,
    *,
    expected_sha256: str = FROZEN_SURFACES_SHA256,
    expected_parameter_hash: str | None = None,
) -> ParameterSource:
    """Select and validate the first stored TRAIN record.

    Eligibility is determined solely from the record's persisted split or
    distribution metadata.  The file hash is verified before any row is used.
    """
    path = Path(dataset_path)
    actual_sha256 = sha256_file(path)
    if actual_sha256.lower() != expected_sha256.lower():
        raise ValueError(
            "frozen surfaces SHA256 mismatch: "
            f"expected {expected_sha256.lower()}, got {actual_sha256}"
        )
    for record in iter_surface_records(path):
        user_metadata = _user_metadata(record)
        metadata = _metadata(record)
        split = str(user_metadata.get("split", metadata.get("split", ""))).lower()
        if split != "train":
            continue
        try:
            vector = parameter_vector_from_record(record)
        except (TypeError, ValueError):
            continue
        parameter_hash = str(user_metadata.get("parameter_vector_hash", ""))
        if not parameter_hash:
            raise ValueError("eligible train record lacks parameter_vector_hash")
        if expected_parameter_hash is not None and parameter_hash.lower() != expected_parameter_hash.lower():
            raise ValueError(
                "first eligible train parameter hash mismatch: "
                f"expected {expected_parameter_hash.lower()}, got {parameter_hash.lower()}"
            )
        surface_id = str(record.get("surface_id", ""))
        return ParameterSource(
            vector=vector,
            record=dict(record),
            dataset_path=str(path),
            dataset_sha256=actual_sha256,
            parameter_hash=parameter_hash,
            surface_id=surface_id,
            split=split,
        )
    raise ValueError("no eligible stored TRAIN record passed validate_parameters")


def select_first_train_record(
    dataset_path: str | Path,
    *,
    expected_sha256: str = FROZEN_SURFACES_SHA256,
) -> ParameterSource:
    """Short compatibility alias for the explicit selection API."""
    return select_first_eligible_train_record(
        dataset_path,
        expected_sha256=expected_sha256,
    )


__all__ = [
    "CANONICAL_PARAMETER_ORDER",
    "EXPECTED_FIRST_PARAMETER_HASH",
    "EXPECTED_FIRST_PARAMETER_VECTOR",
    "EXPECTED_FIRST_SURFACE_ID",
    "FROZEN_SURFACES_SHA256",
    "ParameterSource",
    "iter_surface_records",
    "parameter_vector_from_record",
    "select_first_eligible_train_record",
    "select_first_train_record",
    "sha256_file",
]
