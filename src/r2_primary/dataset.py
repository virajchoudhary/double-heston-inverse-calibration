"""Canonical R2 dataset loading and the frozen primary-comparison features.

This module is the single shared data path for the frozen primary comparison
(``docs/R2_PRIMARY_COMPARISON_PROTOCOL.md``): it loads the sealed final-10k
JSONL, validates every record through the canonical R2 representation
contract, and builds the ONE frozen 100-dimensional feature vector consumed
identically by Model 1 (ordinary ANN) and Model 2 (constraint +
differentiable-repricing-informed inverse model).

Feature blocks (canonical slot order, float32, raw values — no input
normalization; see the frozen protocol INPUT section):

    [0:20)   prices_masked      spot-normalized price, masked slots exactly 0.0
    [20:40)  mask               1.0 / 0.0
    [40:60)  maturities_years   actual per-slot time to maturity
    [60:80)  rates              per-slot risk-free rate
    [80:100) carries            per-slot carry (pricer dividend-yield slot)

Legacy-108 and rejected-R3 geometries are structurally unreachable here: the
loader only accepts canonical R2 payloads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from ..constants import PARAMETER_NAMES
from ..r2_representation.contract import (
    LEGACY_108_INPUT_SIZE,
    NOMINAL_SLOT_COUNT,
    REJECTED_R3_INPUT_SIZE,
)
from ..r2_representation.serialization import payload_to_surface

R2_PRIMARY_FEATURE_SIZE: int = 5 * NOMINAL_SLOT_COUNT  # 100

SYNTHETIC_SOURCE: str = "synthetic_canonical_double_heston_production_pricer"


class R2DatasetError(ValueError):
    """Raised when a record violates the frozen primary-comparison data path."""


def build_r2_features(record: Mapping[str, Any]) -> np.ndarray:
    """Build the frozen 100-dimensional feature vector from ONE R2 payload."""
    try:
        surface = payload_to_surface(record)
    except ValueError as error:
        raise R2DatasetError(f"record rejected by the R2 contract: {error}") from error
    mask = np.asarray(surface.mask, dtype=bool)
    prices = np.asarray(surface.prices, dtype=np.float64)
    masked_prices = np.where(mask, prices, 0.0)
    features = np.concatenate(
        [
            masked_prices,
            mask.astype(np.float64),
            np.asarray(surface.maturities, dtype=np.float64),
            np.asarray(surface.rates, dtype=np.float64),
            np.asarray(surface.carries, dtype=np.float64),
        ]
    )
    if features.shape != (R2_PRIMARY_FEATURE_SIZE,):
        raise R2DatasetError(
            f"R2 primary feature vector must have shape ({R2_PRIMARY_FEATURE_SIZE},), "
            f"got {features.shape}"
        )
    if not np.isfinite(features).all():
        raise R2DatasetError("R2 primary features must be finite")
    return features.astype(np.float32)


def _surface_split(record: Mapping[str, Any]) -> str:
    user_metadata = record.get("metadata", {}).get("user_metadata", {})
    split = str(user_metadata.get("split", ""))
    allowed = {"train", "validation", "test"}
    if split not in allowed:
        raise R2DatasetError(
            f"record {record.get('surface_id')!r} carries no stored train/"
            f"validation/test split label (found {split!r})"
        )
    return split


def _surface_targets(record: Mapping[str, Any]) -> np.ndarray:
    stored = record["metadata"]["parameters_canonical_order"]
    try:
        vector = np.asarray(
            [float(stored[name]) for name in PARAMETER_NAMES], dtype=np.float64
        )
    except KeyError as error:
        raise R2DatasetError(
            f"record {record.get('surface_id')!r} is missing canonical "
            f"parameter {error}"
        ) from error
    if not np.isfinite(vector).all():
        raise R2DatasetError("canonical parameters must be finite")
    return vector


@dataclass(frozen=True)
class R2SurfaceItem:
    """One frozen-dataset surface with everything the comparison needs."""

    surface_id: str
    split: str
    features: np.ndarray  # (100,) float32
    targets: np.ndarray  # (10,) float64 canonical order
    mask: np.ndarray  # (20,) bool
    dollar_prices: np.ndarray  # (20,) float64 normalized price * spot
    normalized_prices: np.ndarray  # (20,) float64 (masked slots 0.0)
    strikes: np.ndarray  # (20,) float64 spot * exp(target log-moneyness)
    maturities: np.ndarray  # (20,) float64 years
    option_types: list[str]  # 20 canonical call/put labels
    spot: float
    rate: float  # rank-constant within a final-dataset surface (asserted)
    carry: float  # rank-constant within a final-dataset surface (asserted)
    parameter_vector_hash: str


def _record_to_item(record: Mapping[str, Any]) -> R2SurfaceItem:
    if record.get("representation_name") != "FROZEN_R2_RANKED_TWO_EXPIRY_CENTRAL_FIVE":
        raise R2DatasetError("non-R2 representation rejected by the primary path")
    if record.get("source") != SYNTHETIC_SOURCE:
        raise R2DatasetError("only synthetic canonical surfaces are loadable here")
    metadata = record.get("metadata", {})
    if metadata.get("synthetic") is not True:
        raise R2DatasetError("real-market records cannot enter the primary path")
    user_metadata = metadata.get("user_metadata", {})
    if user_metadata.get("real_market_inputs_used") is not False:
        raise R2DatasetError("records with real-market inputs are rejected")

    surface = payload_to_surface(record)
    rates = np.asarray(surface.rates, dtype=np.float64)
    carries = np.asarray(surface.carries, dtype=np.float64)
    if not np.allclose(rates, rates[0], rtol=0.0, atol=0.0):
        raise R2DatasetError(
            f"surface {record['surface_id']!r} has non-constant rates across slots; "
            "the frozen final dataset is rank-constant by construction"
        )
    if not np.allclose(carries, carries[0], rtol=0.0, atol=0.0):
        raise R2DatasetError(
            f"surface {record['surface_id']!r} has non-constant carries across slots; "
            "the frozen final dataset is rank-constant by construction"
        )
    strikes = np.asarray(
        [surface.spot * float(np.exp(key.target_log_moneyness)) for key in surface.slot_keys],
        dtype=np.float64,
    )
    mask = np.asarray(surface.mask, dtype=bool)
    normalized_prices = np.where(mask, np.asarray(surface.prices, dtype=np.float64), 0.0)
    return R2SurfaceItem(
        surface_id=str(record["surface_id"]),
        split=_surface_split(record),
        features=build_r2_features(record),
        targets=_surface_targets(record),
        mask=mask,
        dollar_prices=normalized_prices * surface.spot,
        normalized_prices=normalized_prices,
        strikes=strikes,
        maturities=np.asarray(surface.maturities, dtype=np.float64),
        option_types=[key.option_type for key in surface.slot_keys],
        spot=float(surface.spot),
        rate=float(rates[0]),
        carry=float(carries[0]),
        parameter_vector_hash=str(user_metadata.get("parameter_vector_hash", "")),
    )


def iter_r2_jsonl(path: str | Path) -> Iterator[Mapping[str, Any]]:
    """Stream validated R2 payload records from the sealed JSONL."""
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise R2DatasetError(f"line {line_number}: invalid JSON") from error
            yield record


class R2PrimaryDataset(Dataset):
    """Frozen R2 surfaces with shared features, targets, and split labels."""

    def __init__(self, items: list[R2SurfaceItem]) -> None:
        if not items:
            raise R2DatasetError("dataset must contain at least one surface")
        self.items = items
        self.features = torch.as_tensor(
            np.stack([item.features for item in items]), dtype=torch.float32
        )
        self.targets = torch.as_tensor(
            np.stack([item.targets for item in items]), dtype=torch.float64
        )
        self.masks = torch.as_tensor(
            np.stack([item.mask for item in items]), dtype=torch.bool
        )
        if self.features.shape[1] in (LEGACY_108_INPUT_SIZE, REJECTED_R3_INPUT_SIZE):
            raise R2DatasetError("legacy geometry cannot reach the R2 primary path")
        identifiers = [item.surface_id for item in items]
        if len(set(identifiers)) != len(identifiers):
            raise R2DatasetError("surface ids must be unique")

    @classmethod
    def from_jsonl(cls, path: str | Path, *, splits: set[str] | None = None) -> "R2PrimaryDataset":
        """Load every record (optionally restricted to specific splits)."""
        items = [
            item
            for record in iter_r2_jsonl(path)
            if splits is None or _surface_split(record) in splits
            for item in [_record_to_item(record)]
        ]
        return cls(items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str, R2SurfaceItem]:
        return self.features[index], self.targets[index], self.items[index].surface_id, self.items[index]

    def indices_for_split(self, split: str) -> list[int]:
        return [index for index, item in enumerate(self.items) if item.split == split]

    def split_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.items:
            counts[item.split] = counts.get(item.split, 0) + 1
        return counts

    def subset(self, indices: list[int]) -> "R2PrimaryDataset":
        return R2PrimaryDataset([self.items[index] for index in indices])


def assert_split_isolation(dataset: R2PrimaryDataset) -> None:
    """Fail unless surface ids and parameter hashes are disjoint across splits."""
    by_split: dict[str, list[R2SurfaceItem]] = {}
    for item in dataset.items:
        by_split.setdefault(item.split, []).append(item)
    names = sorted(by_split)
    for left_index in range(len(names)):
        for right_index in range(left_index + 1, len(names)):
            left, right = names[left_index], names[right_index]
            left_ids = {item.surface_id for item in by_split[left]}
            right_ids = {item.surface_id for item in by_split[right]}
            assert not left_ids & right_ids, f"surface id overlap between {left} and {right}"
            left_hashes = {
                item.parameter_vector_hash
                for item in by_split[left]
                if item.parameter_vector_hash
            }
            right_hashes = {
                item.parameter_vector_hash
                for item in by_split[right]
                if item.parameter_vector_hash
            }
            assert not left_hashes & right_hashes, (
                f"parameter-vector-hash overlap between {left} and {right}"
            )
