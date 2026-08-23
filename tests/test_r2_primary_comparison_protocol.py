"""Pre-training checkpoint contract tests for the frozen R2 primary comparison.

These tests gate the primary comparison: they must pass, and the protocol
commit must be pushed and remote-verified, BEFORE any research training or
calibration run (docs/R2_PRIMARY_COMPARISON_PROTOCOL.md section 10).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from src.constants import PARAMETER_NAMES
from src.dheston.real_market_policy import (
    RealMarketWeightUpdateQuarantineError,
    resolve_real_market_epochs,
)
from src.r2_representation.contract import (
    CANONICAL_SLOT_KEYS,
    LEGACY_108_INPUT_SIZE,
    NOMINAL_SLOT_COUNT,
    REPRESENTATION_NAME,
    REJECTED_R3_INPUT_SIZE,
    SlotKey,
    slot_index,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "data" / "final_r2_clean_10000" / "surfaces.jsonl"
PROTOCOL_CONFIG_PATH = REPO_ROOT / "configs" / "r2_primary_comparison_FINAL.yaml"
PROTOCOL_DOC_PATH = REPO_ROOT / "docs" / "R2_PRIMARY_COMPARISON_PROTOCOL.md"
AUDIT_DOC_PATH = REPO_ROOT / "docs" / "R2_PRIMARY_COMPARISON_PRE_TRAINING_AUDIT.md"
CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / "r2_primary_comparison"

EXPECTED_SURFACES_SHA256 = (
    "148b579a4f6ce572e34796e872479c4c016c89bbcd20438c2bb62d6b6960f1f6"
)
EXPECTED_TOTAL = 10_000
EXPECTED_SPLIT_COUNTS = {"train": 7_500, "validation": 1_250, "test": 1_250}


def _iter_records() -> iter:
    with DATASET_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            yield __import__("json").loads(line)


def _protocol_config() -> dict:
    return yaml.safe_load(PROTOCOL_CONFIG_PATH.read_text(encoding="utf-8"))


def test_frozen_dataset_hash_is_exact() -> None:
    digest = hashlib.sha256()
    with DATASET_PATH.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    assert digest.hexdigest() == EXPECTED_SURFACES_SHA256


def test_dataset_counts_and_stored_split_labels() -> None:
    split_counts: dict[str, int] = {}
    split_surface_ids: dict[str, set[str]] = {}
    split_param_hashes: dict[str, set[str]] = {}
    total = 0
    for record in _iter_records():
        total += 1
        metadata = record["metadata"]["user_metadata"]
        split = metadata["split"]
        split_counts[split] = split_counts.get(split, 0) + 1
        split_surface_ids.setdefault(split, set()).add(record["surface_id"])
        split_param_hashes.setdefault(split, set()).add(
            metadata["parameter_vector_hash"]
        )
        assert split in EXPECTED_SPLIT_COUNTS
    assert total == EXPECTED_TOTAL
    assert split_counts == EXPECTED_SPLIT_COUNTS
    # zero cross-split overlap by surface id and by parameter-vector hash
    for left in split_surface_ids:
        for right in split_surface_ids:
            if left < right:
                assert not split_surface_ids[left] & split_surface_ids[right]
                assert not split_param_hashes[left] & split_param_hashes[right]


def test_dataset_is_r2_only_canonical_slot_identity() -> None:
    for record in _iter_records():
        assert record["representation_name"] == REPRESENTATION_NAME
        assert record["representation_version"] == "1.0"
        assert len(record["prices"]) == NOMINAL_SLOT_COUNT == 20
        assert len(record["mask"]) == NOMINAL_SLOT_COUNT
        assert len(record["maturities"]) == NOMINAL_SLOT_COUNT
        assert len(record["rates"]) == NOMINAL_SLOT_COUNT
        assert len(record["carries"]) == NOMINAL_SLOT_COUNT
        assert len(record["slot_keys"]) == NOMINAL_SLOT_COUNT
        assert all(isinstance(flag, bool) for flag in record["mask"])
        for key, expected in zip(record["slot_keys"], CANONICAL_SLOT_KEYS, strict=True):
            assert SlotKey(*key) == expected


def test_no_legacy_108_or_rejected_r3_anywhere_in_representation() -> None:
    config = _protocol_config()
    assert config["input_representation"]["canonical_r2_only"] is True
    assert config["input_representation"]["legacy_108_forbidden"] is True
    assert config["input_representation"]["input_size"] == 100
    assert NOMINAL_SLOT_COUNT * 5 == 100
    assert 100 not in (LEGACY_108_INPUT_SIZE, REJECTED_R3_INPUT_SIZE)
    assert slot_index(SlotKey(1, -0.10, "call")) == 0
    assert slot_index(SlotKey(2, 0.10, "put")) == NOMINAL_SLOT_COUNT - 1


def test_canonical_parameter_order_matches_dataset_records() -> None:
    for record in _iter_records():
        stored = record["metadata"]["parameters_canonical_order"]
        # the stored mapping is a name -> value dict (serialized with sorted
        # keys); canonical ORDER is defined by PARAMETER_NAMES name lookup
        assert set(stored) == set(PARAMETER_NAMES)
        vector = [float(stored[name]) for name in PARAMETER_NAMES]
        assert len(vector) == 10
        break
    config = _protocol_config()
    assert config["output"]["parameter_order"] == PARAMETER_NAMES


def test_dataset_contains_no_real_market_records() -> None:
    for record in _iter_records():
        assert record["metadata"]["synthetic"] is True
        assert record["source"] == "synthetic_canonical_double_heston_production_pricer"
        assert record["metadata"]["user_metadata"]["real_market_inputs_used"] is False


def test_real_market_weight_update_quarantine_is_fail_closed() -> None:
    # synthetic-only usage resolves unchanged...
    assert (
        resolve_real_market_epochs(
            config_real_epochs=0,
            continuous_requested=False,
            allow_noncanonical_real_weight_updates=False,
            continuous_epoch_limit=0,
        )
        == 0
    )
    # ...while any positive real-epoch request without the explicit
    # noncanonical opt-in flag raises (Issue #20 guard live on this branch).
    with pytest.raises(RealMarketWeightUpdateQuarantineError):
        resolve_real_market_epochs(
            config_real_epochs=1,
            continuous_requested=False,
            allow_noncanonical_real_weight_updates=False,
            continuous_epoch_limit=0,
        )


def test_protocol_config_freezes_seeds_metrics_and_methods() -> None:
    config = _protocol_config()
    assert config["protocol"]["status"] == "FROZEN_BEFORE_ANY_PRIMARY_RESEARCH_TRAINING"
    assert config["seeds"]["neural_research_seeds"] == [11, 22, 33]
    assert config["seeds"]["rule"].startswith("all seeds trained")
    assert config["hyperparameter_policy"]["synthetic_test_optimization"] == (
        "ABSOLUTELY_FORBIDDEN"
    )
    assert config["metrics"]["frozen_before_results"] is True
    for family in (
        "parameter_recovery",
        "constraint_validity",
        "repricing",
        "identifiability_aware",
        "stability",
        "runtime",
    ):
        assert config["metrics"][family], family
    model1 = config["model1_ordinary_ann"]
    model2 = config["model2_constraint_repricing_informed"]
    assert model1["class"] == "models.ann_model.ANNInverseCalibrator"
    assert model2["class"] == "models.pinn_model.PhysicsInformedInverseCalibrator"
    assert model1["hidden_sizes"] == [512, 256, 128, 64]
    assert model2["hidden_sizes"] == [512, 512, 256, 256, 128]
    assert model2["loss_weights"] == {
        "parameter_loss_weight": 1.0,
        "repricing_loss_weight": 1.0,
    }
    assert config["input_representation"]["shared_by_model1_and_model2"] == (
        "ONE_SHARED_FEATURE_BUILDER"
    )
    traditional = config["traditional_calibration"]
    assert traditional["module"] == "src/calibrate_double_heston.py"
    assert traditional["max_nfev"] == 300
    assert traditional["starts"]["count"] == 3


def test_protocol_documents_exist_and_declare_freeze() -> None:
    doc = PROTOCOL_DOC_PATH.read_text(encoding="utf-8")
    audit = AUDIT_DOC_PATH.read_text(encoding="utf-8")
    assert "FROZEN_BEFORE_ANY_PRIMARY_RESEARCH_TRAINING" in doc
    assert EXPECTED_SURFACES_SHA256 in doc
    assert "11, 22, 33" in doc
    for question in "ABCDEFGHIJ":
        assert f"## {question}." in audit


def test_research_checkpoint_directories_absent_before_training() -> None:
    if CHECKPOINT_ROOT.exists():
        existing = sorted(
            path.name
            for path in CHECKPOINT_ROOT.iterdir()
            if path.name.startswith(("model1_", "model2_"))
        )
        assert not existing, f"unexpected pre-existing research runs: {existing}"
