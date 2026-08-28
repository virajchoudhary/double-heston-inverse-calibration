"""Deterministic state-conditioned synthetic data for the forward PINN."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.double_heston import price_double_heston_call

from .config import BaselineConfig, DomainConfig, load_baseline_config, variance_state_bounds
from .parameter_source import ParameterSource, select_first_eligible_train_record

FEATURE_NAMES = ("spot", "variance_slow", "variance_fast", "tau", "strike", "rate", "carry")
SPLIT_NAMES = ("train", "validation", "test")


def _stable_hash_strings(values: np.ndarray) -> str:
    payload = "\n".join(str(value) for value in values.tolist()).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve_path(path: str | Path, *, base: Path | None = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (base or Path.cwd()) / candidate


@dataclass(frozen=True)
class SyntheticDataset:
    """Immutable in-memory representation of generated quote points."""

    features: np.ndarray
    reference_prices: np.ndarray
    split_names: np.ndarray
    split_ids: np.ndarray
    sample_ids: np.ndarray
    parameter_source: ParameterSource
    manifest: dict[str, Any]

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=np.float64)
        prices = np.asarray(self.reference_prices, dtype=np.float64)
        split_names = np.asarray(self.split_names, dtype=str)
        split_ids = np.asarray(self.split_ids, dtype=str)
        sample_ids = np.asarray(self.sample_ids, dtype=str)
        if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"features must have shape (points, {len(FEATURE_NAMES)})")
        count = features.shape[0]
        if any(array.shape != (count,) for array in (prices, split_names, split_ids, sample_ids)):
            raise ValueError("dataset arrays must contain one row per point")
        if not np.isfinite(features).all() or not np.isfinite(prices).all():
            raise ValueError("dataset arrays must be finite")
        if not np.all(features[:, 0] > 0) or not np.all(features[:, 4] > 0):
            raise ValueError("spot and strike must be strictly positive")
        if not np.all(features[:, 1:3] > 0) or not np.all(features[:, 3] >= 0):
            raise ValueError("variance states must be positive and tau non-negative")
        if not set(split_names.tolist()).issubset(set(SPLIT_NAMES)):
            raise ValueError("unknown dataset split")
        for array in (features, prices, split_names, split_ids, sample_ids):
            array.setflags(write=False)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "reference_prices", prices)
        object.__setattr__(self, "split_names", split_names)
        object.__setattr__(self, "split_ids", split_ids)
        object.__setattr__(self, "sample_ids", sample_ids)

    @property
    def size(self) -> int:
        return int(self.features.shape[0])

    @property
    def spot(self) -> np.ndarray:
        return self.features[:, 0]

    @property
    def variance_slow(self) -> np.ndarray:
        return self.features[:, 1]

    @property
    def variance_fast(self) -> np.ndarray:
        return self.features[:, 2]

    @property
    def tau(self) -> np.ndarray:
        return self.features[:, 3]

    @property
    def strike(self) -> np.ndarray:
        return self.features[:, 4]

    @property
    def rate(self) -> np.ndarray:
        return self.features[:, 5]

    @property
    def carry(self) -> np.ndarray:
        return self.features[:, 6]

    def indices(self, split: str) -> np.ndarray:
        name = str(split).lower()
        if name not in SPLIT_NAMES:
            raise ValueError(f"split must be one of {SPLIT_NAMES}")
        return np.flatnonzero(self.split_names == name)

    def split_id_hashes(self) -> dict[str, str]:
        return {
            name: _stable_hash_strings(self.split_ids[self.split_names == name])
            for name in SPLIT_NAMES
        }

    def to_tensors(self) -> tuple[Any, Any]:
        """Convert features and references to float64 torch tensors lazily."""
        import torch

        return (
            torch.from_numpy(np.asarray(self.features, dtype=np.float64)),
            torch.from_numpy(np.asarray(self.reference_prices, dtype=np.float64)),
        )


def _sample_split(
    *,
    split_name: str,
    count: int,
    seed: int,
    domain: DomainConfig,
    source: ParameterSource,
    node_count: int,
    split_id_prefix: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if count <= 0:
        raise ValueError("split count must be strictly positive")
    # A separate SeedSequence child per split makes split membership invariant
    # to changes in the other split sizes.
    split_index = SPLIT_NAMES.index(split_name)
    generator = np.random.default_rng(np.random.SeedSequence([seed, split_index]))
    spot = generator.uniform(domain.spot_min, domain.spot_max, count)
    moneyness = generator.uniform(domain.moneyness_min, domain.moneyness_max, count)
    tau = generator.uniform(domain.tau_min, domain.tau_max, count)
    rate = generator.uniform(domain.rate_min, domain.rate_max, count)
    carry = generator.uniform(domain.carry_min, domain.carry_max, count)

    theta_slow = float(source.vector[1])
    theta_fast = float(source.vector[6])
    slow_minimum, slow_maximum = variance_state_bounds(theta_slow, domain)
    fast_minimum, fast_maximum = variance_state_bounds(theta_fast, domain)
    variance_slow = generator.uniform(slow_minimum, slow_maximum, count)
    variance_fast = generator.uniform(fast_minimum, fast_maximum, count)
    strike = spot * moneyness

    prices = np.empty(count, dtype=np.float64)
    for index in range(count):
        conditioned_parameters = source.parameters_for_state(
            float(variance_slow[index]), float(variance_fast[index])
        )
        prices[index] = price_double_heston_call(
            float(spot[index]),
            float(strike[index]),
            float(tau[index]),
            float(rate[index]),
            float(carry[index]),
            conditioned_parameters,
            node_count=node_count,
        )
    features = np.column_stack((spot, variance_slow, variance_fast, tau, strike, rate, carry))
    split_ids = np.asarray(
        [f"{split_id_prefix}_{split_name}_{index:06d}" for index in range(count)],
        dtype=str,
    )
    sample_ids = np.asarray(
        [f"{split_name}_{index:06d}" for index in range(count)], dtype=str
    )
    split_labels = np.full(count, split_name, dtype=f"<U{len(split_name)}")
    return features, prices, split_labels, split_ids, sample_ids


def generate_synthetic_dataset(
    output_dir: str | Path,
    *,
    config: BaselineConfig | None = None,
    frozen_surfaces_path: str | Path | None = None,
    train_count: int | None = None,
    validation_count: int | None = None,
    test_count: int | None = None,
) -> SyntheticDataset:
    """Generate, persist, and return the deterministic CALL dataset."""
    baseline = config or load_baseline_config()
    overrides = {
        name: value
        for name, value in (
            ("train_count", train_count),
            ("validation_count", validation_count),
            ("test_count", test_count),
        )
        if value is not None
    }
    if overrides:
        baseline = baseline.with_overrides(**overrides)
    baseline.validate()
    repo_root = Path(__file__).resolve().parents[2]
    source_path = _resolve_path(
        frozen_surfaces_path or baseline.data.frozen_surfaces_path,
        base=repo_root,
    )
    source = select_first_eligible_train_record(
        source_path,
        expected_sha256=baseline.data.frozen_surfaces_sha256,
        expected_parameter_hash=baseline.expected_parameter_hash,
    )
    counts = {
        "train": baseline.data.train_count,
        "validation": baseline.data.validation_count,
        "test": baseline.data.test_count,
    }
    if any(value <= 0 for value in counts.values()):
        raise ValueError("all generated split counts must be strictly positive")

    pieces = [
        _sample_split(
            split_name=name,
            count=counts[name],
            seed=baseline.seed,
            domain=baseline.domain,
            source=source,
            node_count=baseline.pricing_node_count,
            split_id_prefix=baseline.data.split_id_prefix,
        )
        for name in SPLIT_NAMES
    ]
    features = np.concatenate([piece[0] for piece in pieces], axis=0)
    prices = np.concatenate([piece[1] for piece in pieces], axis=0)
    split_names = np.concatenate([piece[2] for piece in pieces], axis=0)
    split_ids = np.concatenate([piece[3] for piece in pieces], axis=0)
    sample_ids = np.concatenate([piece[4] for piece in pieces], axis=0)
    dataset_manifest: dict[str, Any] = {
        "schema_version": "mentor_dh_pinn_dataset_v1",
        "seed": baseline.seed,
        "counts": counts,
        "feature_names": list(FEATURE_NAMES),
        "domain": {
            "spot": [baseline.domain.spot_min, baseline.domain.spot_max],
            "moneyness": [baseline.domain.moneyness_min, baseline.domain.moneyness_max],
            "tau_years": [baseline.domain.tau_min, baseline.domain.tau_max],
            "rate": [baseline.domain.rate_min, baseline.domain.rate_max],
            "carry": [baseline.domain.carry_min, baseline.domain.carry_max],
            "variance_state_rule": baseline.domain.variance_state_rule,
            "variance_theta_min_multiplier": baseline.domain.variance_theta_min_multiplier,
            "variance_theta_max_multiplier": baseline.domain.variance_theta_max_multiplier,
            "variance_floor": baseline.domain.variance_floor,
            "variance_ceiling": baseline.domain.variance_ceiling,
        },
        "option_type": baseline.option_type,
        "pricing_node_count": baseline.pricing_node_count,
        "rate_range": [baseline.domain.rate_min, baseline.domain.rate_max],
        "carry_range": [baseline.domain.carry_min, baseline.domain.carry_max],
        "frozen_surfaces_sha256": source.dataset_sha256,
        "parameter_hash": source.parameter_hash,
        "surface_id": source.surface_id,
        "parameter_source": source.provenance(),
        "split_id_hashes": {
            name: _stable_hash_strings(split_ids[split_names == name])
            for name in SPLIT_NAMES
        },
        "split_disjoint": True,
        "config": baseline.to_dict(),
        "config_sha256": _json_sha256(baseline.to_dict()),
    }
    dataset = SyntheticDataset(
        features=features,
        reference_prices=prices,
        split_names=split_names,
        split_ids=split_ids,
        sample_ids=sample_ids,
        parameter_source=source,
        manifest=dataset_manifest,
    )
    _persist_dataset(
        dataset,
        output_dir,
        baseline.data.dataset_filename,
        baseline.data.manifest_filename,
        baseline.data.parameter_provenance_filename,
    )
    return dataset


def _persist_dataset(
    dataset: SyntheticDataset,
    output_dir: str | Path,
    dataset_filename: str,
    manifest_filename: str,
    parameter_provenance_filename: str,
) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    dataset_path = root / dataset_filename
    np.savez_compressed(
        dataset_path,
        features=dataset.features,
        reference_prices=dataset.reference_prices,
        split_names=dataset.split_names,
        split_ids=dataset.split_ids,
        sample_ids=dataset.sample_ids,
    )
    dataset_hash = _sha256_file(dataset_path)
    manifest = dict(dataset.manifest)
    dataset.manifest["dataset_sha256"] = dataset_hash
    manifest["dataset_filename"] = dataset_filename
    manifest["dataset_sha256"] = dataset_hash
    with (root / manifest_filename).open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (root / parameter_provenance_filename).open("w", encoding="utf-8") as handle:
        json.dump(dataset.parameter_source.provenance(), handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_synthetic_dataset(
    output_dir: str | Path,
    *,
    config: BaselineConfig | None = None,
    verify_manifest: bool = True,
) -> SyntheticDataset:
    """Load a persisted dataset and verify its split/parameter provenance."""
    baseline = config or load_baseline_config()
    root = Path(output_dir)
    with (root / baseline.data.manifest_filename).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    dataset_path = root / str(manifest.get("dataset_filename", baseline.data.dataset_filename))
    if verify_manifest and str(manifest.get("dataset_sha256", "")).lower() != _sha256_file(dataset_path):
        raise ValueError("generated dataset SHA256 mismatch")
    arrays = np.load(dataset_path, allow_pickle=False)
    source = select_first_eligible_train_record(
        Path(__file__).resolve().parents[2] / baseline.data.frozen_surfaces_path,
        expected_sha256=baseline.data.frozen_surfaces_sha256,
        expected_parameter_hash=baseline.expected_parameter_hash,
    )
    dataset = SyntheticDataset(
        features=arrays["features"],
        reference_prices=arrays["reference_prices"],
        split_names=arrays["split_names"],
        split_ids=arrays["split_ids"],
        sample_ids=arrays["sample_ids"],
        parameter_source=source,
        manifest=manifest,
    )
    if verify_manifest:
        if dataset.split_id_hashes() != manifest.get("split_id_hashes"):
            raise ValueError("split ID hashes do not match the dataset manifest")
        if dataset.size != sum(int(value) for value in manifest["counts"].values()):
            raise ValueError("dataset size does not match manifest counts")
        validate_dataset_identity(dataset, baseline)
        with (root / baseline.data.parameter_provenance_filename).open("r", encoding="utf-8") as handle:
            persisted_provenance = json.load(handle)
        if persisted_provenance != source.provenance() or manifest.get("parameter_source") != persisted_provenance:
            raise ValueError("parameter provenance identities do not match")
    return dataset


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_dataset_identity(dataset: SyntheticDataset, config: BaselineConfig) -> None:
    """Fail closed unless dataset, config, parameter, and split identities agree."""
    expected_counts = {
        "train": config.data.train_count,
        "validation": config.data.validation_count,
        "test": config.data.test_count,
    }
    manifest = dataset.manifest
    checks = (
        (manifest.get("schema_version") == "mentor_dh_pinn_dataset_v1", "dataset schema"),
        (manifest.get("counts") == expected_counts, "dataset/config split counts"),
        (manifest.get("option_type") == config.option_type, "option type"),
        (manifest.get("pricing_node_count") == config.pricing_node_count, "pricing node count"),
        (manifest.get("frozen_surfaces_sha256") == config.data.frozen_surfaces_sha256, "frozen source hash"),
        (manifest.get("parameter_hash") == config.expected_parameter_hash, "parameter hash"),
        (manifest.get("surface_id") == dataset.parameter_source.surface_id, "surface identity"),
        (manifest.get("split_id_hashes") == dataset.split_id_hashes(), "split hashes"),
        (manifest.get("config") == config.to_dict(), "embedded config"),
        (manifest.get("config_sha256") == _json_sha256(config.to_dict()), "config hash"),
    )
    for passed, name in checks:
        if not passed:
            raise ValueError(f"dataset identity mismatch: {name}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "FEATURE_NAMES",
    "SPLIT_NAMES",
    "SyntheticDataset",
    "generate_synthetic_dataset",
    "load_synthetic_dataset",
    "validate_dataset_identity",
]
