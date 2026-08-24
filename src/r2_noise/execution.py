"""Execution-only paths for the frozen R2 observation-noise study.

The frozen draw and subset-selection contracts live in ``perturbation.py``
and ``subset.py``.  This module adds deterministic cohort serialization and
the shared provenance checks needed by the research runners.  It deliberately
contains no metric or model-selection policy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from .perturbation import NOISE_BASE_SEED, perturb_surface_prices

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "r2_noise_robustness_FINAL.yaml"
PRIMARY_CONFIG_PATH = REPO_ROOT / "configs" / "r2_primary_comparison_FINAL.yaml"
CLEAN_DATASET_PATH = REPO_ROOT / "data" / "final_r2_clean_10000" / "surfaces.jsonl"
DATA_ROOT = REPO_ROOT / "data" / "r2_noise_robustness"
EVIDENCE_ROOT = REPO_ROOT / "evidence" / "r2_noise_robustness"
FROZEN_PROTOCOL_SHA256 = (
    "2fa49b3eb885d3427c01ab0cfe447fc6ddd7f19957db73c4b4ed782476c57c5a"
)
FROZEN_CLEAN_DATASET_SHA256 = (
    "148b579a4f6ce572e34796e872479c4c016c89bbcd20438c2bb62d6b6960f1f6"
)
PRIMARY_PROTOCOL_SHA256 = (
    "33ca0f763ec10bb2424eefb02448c9c8e50021854b96a948e420f44bdba70781"
)


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_protocol() -> dict[str, Any]:
    """Load the immutable YAML protocol after checking its sealed identity."""
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["protocol"]["name"] != "R2_OBSERVATION_NOISE_ROBUSTNESS":
        raise ValueError("unexpected R2 noise protocol")
    if config["protocol"]["status"] != "FROZEN_BEFORE_ANY_NOISY_RESEARCH_RESULT":
        raise ValueError("R2 noise protocol is not frozen")
    actual_protocol = sha256_path(CONFIG_PATH)
    if actual_protocol != FROZEN_PROTOCOL_SHA256:
        raise ValueError(
            f"frozen protocol hash changed: {actual_protocol} != "
            f"{FROZEN_PROTOCOL_SHA256}"
        )
    if config["canonical_baseline"]["frozen_r2_dataset_sha256"] != (
        FROZEN_CLEAN_DATASET_SHA256
    ):
        raise ValueError("protocol no longer pins the canonical clean dataset")
    return config


def assert_clean_dataset_identity(path: str | Path = CLEAN_DATASET_PATH) -> str:
    digest = sha256_path(path)
    if digest != FROZEN_CLEAN_DATASET_SHA256:
        raise ValueError(
            f"clean dataset hash mismatch: {digest} != "
            f"{FROZEN_CLEAN_DATASET_SHA256}"
        )
    return digest


def level_label(level: float, labels: list[str] | tuple[str, ...]) -> str:
    exact = {float(value): label for value, label in zip(_levels(labels), labels)}
    try:
        return exact[float(level)]
    except KeyError as error:
        raise ValueError(f"noise level is not frozen: {level!r}") from error


def _levels(labels: list[str] | tuple[str, ...]) -> tuple[float, ...]:
    configured = load_frozen_protocol()["noise_levels"]
    if len(configured) != len(labels):
        raise ValueError("noise levels and labels are misaligned")
    return tuple(float(value) for value in configured)


def iter_test_records(path: str | Path = CLEAN_DATASET_PATH):
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            metadata = record.get("metadata", {}).get("user_metadata", {})
            if metadata.get("split") != "test":
                continue
            yield line_number, record


def derivation_provenance(clean_sha256: str) -> dict[str, Any]:
    perturbation_path = Path(__file__).with_name("perturbation.py")
    return {
        "config_path": CONFIG_PATH.as_posix(),
        "config_sha256": FROZEN_PROTOCOL_SHA256,
        "clean_dataset_path": CLEAN_DATASET_PATH.as_posix(),
        "clean_dataset_sha256": clean_sha256,
        "module_path": Path(__file__).as_posix(),
        "module_sha256": sha256_path(Path(__file__)),
        "primary_protocol_sha256": PRIMARY_PROTOCOL_SHA256,
        "perturbation_module_path": perturbation_path.as_posix(),
        "perturbation_module_sha256": sha256_path(perturbation_path),
    }


def derive_noisy_record(
    record: Mapping[str, Any],
    *,
    noise_level: float,
    label: str,
    clean_sha256: str,
) -> dict[str, Any]:
    """Return a validated derived payload without mutating the clean input."""
    noisy_prices, resample_counters = perturb_surface_prices(
        record["prices"],
        record["surface_id"],
        record["slot_keys"],
        noise_level,
    )
    realizations = []
    for index, key in enumerate(record["slot_keys"]):
        expiry_rank, moneyness_k, option_type = key
        realizations.append(
            {
                "resample_counter": int(resample_counters[index]),
                "slot_index": index,
                "slot_key": [int(expiry_rank), float(moneyness_k), str(option_type)],
            }
        )
    derived = json.loads(json.dumps(record, allow_nan=False))
    derived["prices"] = noisy_prices
    derived["observation_noise"] = {
        "base_seed": NOISE_BASE_SEED,
        "derivation": derivation_provenance(clean_sha256),
        "level": float(noise_level),
        "level_label": label,
        "realizations": realizations,
    }
    return derived


def serialize_record(record: Mapping[str, Any]) -> str:
    return json.dumps(
        record, sort_keys=True, separators=(",", ":"), allow_nan=False
    ) + "\n"
