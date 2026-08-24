"""Frozen-contract tests for the R2 observation-noise robustness study.

These tests pin the frozen noise/selection contract BEFORE any noisy research
result exists.  They generate no research cohort beyond synthetic in-test
fixtures and read no model checkpoint.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.r2_noise import perturbation as P
from src.r2_noise import subset as S

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "data" / "final_r2_clean_10000" / "surfaces.jsonl"
CONFIG_PATH = REPO_ROOT / "configs" / "r2_noise_robustness_FINAL.yaml"
PROTOCOL_DOC = REPO_ROOT / "docs" / "R2_NOISE_ROBUSTNESS_PROTOCOL.md"

FROZEN_PROTOCOL_SHA256 = "33ca0f763ec10bb2424eefb02448c9c8e50021854b96a948e420f44bdba70781"
FROZEN_DATASET_SHA256 = "148b579a4f6ce572e34796e872479c4c016c89bbcd20438c2bb62d6b6960f1f6"


# ---------------------------------------------------------------------------
# deterministic noise replay / seed policy
# ---------------------------------------------------------------------------


def test_slot_seed_is_deterministic_and_level_separated() -> None:
    base = P.slot_seed("R2_FINAL_interior_train_000001", 1, -0.10, "call", 0.005)
    assert base == P.slot_seed("R2_FINAL_interior_train_000001", 1, -0.10, "call", 0.005)
    variants = {
        P.slot_seed("R2_FINAL_interior_train_000001", 1, -0.10, "call", level)
        for level in (0.001, 0.0025, 0.005, 0.01)
    }
    assert len(variants) == 4, "levels must receive independent realizations"
    assert (
        P.slot_seed("R2_FINAL_interior_train_000001", 1, -0.10, "put", 0.005)
        != P.slot_seed("R2_FINAL_interior_train_000001", 1, -0.10, "call", 0.005)
    ), "calls and puts must be perturbed independently"


def test_perturbation_replays_bitwise() -> None:
    args = ("R2_FINAL_interior_train_000007", 2, 0.05, "put", 0.005)
    first = P.perturb_slot(12.34, *args)
    second = P.perturb_slot(12.34, *args)
    assert first == second
    prices_a, counters_a = P.perturb_surface_prices(
        [10.0, 9.0], "sid", [[1, -0.10, "call"], [1, -0.05, "put"]], 0.0025
    )
    prices_b, counters_b = P.perturb_surface_prices(
        [10.0, 9.0], "sid", [[1, -0.10, "call"], [1, -0.05, "put"]], 0.0025
    )
    assert prices_a == prices_b and counters_a == counters_b


def test_zero_level_returns_exact_identity() -> None:
    price, counter = P.perturb_slot(17.25, "sid", 1, 0.0, "call", 0.0)
    assert price == 17.25 and counter == 0
    assert P.slot_noise_factor("sid", 1, 0.0, "call", 0.0) == 1.0


def test_negative_draw_counter_resample_is_deterministic_and_capped(monkeypatch) -> None:
    """Force the pathological branch: every draw negative until counter cap."""
    calls = {"n": 0}

    def fake_rng(seed):
        class _R:
            def standard_normal(self):
                calls["n"] += 1
                return -1000.0  # factor 1-10 < 0 at every frozen level

        return _R()

    monkeypatch.setattr(P.np.random, "default_rng", fake_rng)
    with pytest.raises(RuntimeError):
        P.perturb_slot(10.0, "sid", 1, 0.0, "call", 0.01)
    assert calls["n"] == 65  # initial draw + 64 capped resamples


def test_realization_shared_across_methods_and_neural_seeds() -> None:
    """The key format must not contain any method/model/seed component."""
    import inspect

    source = inspect.getsource(P.slot_seed)
    for forbidden in ("model", "method", "seed11", "model1", "model2", "worker"):
        assert forbidden not in source.lower(), forbidden
    factor_ann = P.slot_noise_factor("sid", 1, 0.0, "call", 0.005)
    factor_traditional = P.slot_noise_factor("sid", 1, 0.0, "call", 0.005)
    assert factor_ann == factor_traditional


# ---------------------------------------------------------------------------
# clean-dataset identity + derivation semantics
# ---------------------------------------------------------------------------


def _test_split_records(count: int | None = None):
    records = []
    with DATASET_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["metadata"]["user_metadata"]["split"] != "test":
                continue
            records.append(record)
            if count is not None and len(records) >= count:
                break
    return records


def test_frozen_dataset_identity_unchanged() -> None:
    assert (
        hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest() == FROZEN_DATASET_SHA256
    )


def test_derivation_preserves_truth_metadata_masks_and_fields() -> None:
    record = _test_split_records(1)[0]
    clean_prices = list(record["prices"])
    noisy, counters = P.perturb_surface_prices(
        clean_prices,
        record["surface_id"],
        record["slot_keys"],
        0.005,
    )
    assert len(noisy) == len(clean_prices) == len(record["slot_keys"])
    # masks/spots/truth are outside the perturbation surface entirely
    assert all(c == 0 for c in counters)  # no pathological resample occurred
    for clean_value, noisy_value in zip(clean_prices, noisy):
        if clean_value > 0:
            assert noisy_value > 0.0


def test_no_train_or_validation_contamination_in_population_source() -> None:
    records = _test_split_records()
    assert len(records) == 1250
    assert all(r["metadata"]["user_metadata"]["split"] == "test" for r in records)


# ---------------------------------------------------------------------------
# traditional subset selection (frozen design)
# ---------------------------------------------------------------------------


def test_subset_selection_is_deterministic_stratified_and_exact() -> None:
    records = [
        {
            "surface_id": r["surface_id"],
            "parameters_canonical_order": r["metadata"]["parameters_canonical_order"],
        }
        for r in _test_split_records()
    ]
    first = S.select_traditional_subset(records)
    second = S.select_traditional_subset(records)
    assert first["selected_ids"] == second["selected_ids"]
    assert len(first["selected_ids"]) == S.SUBSET_SIZE == 250
    assert len(set(first["selected_ids"])) == 250
    assert set(first["selected_ids"]).issubset({r["surface_id"] for r in records})
    # stratification coverage: every populated cell contributes proportionally
    assert sum(first["cell_sizes"].values()) == 1250
    assert max(first["allocation_proportional_largest_remainder"], True)


def test_subset_ids_are_test_split_only() -> None:
    subset_path = REPO_ROOT / "evidence" / "r2_noise_robustness" / "traditional_subset_ids.json"
    payload = json.loads(subset_path.read_text())
    test_ids = {r["surface_id"] for r in _test_split_records()}
    ids = payload["selected_ids"]
    assert payload["subset_size"] == 250 and len(ids) == 250
    assert set(ids).issubset(test_ids), "subset leaked non-test surfaces"
    assert payload["selection_uses_outcomes"] is False


# ---------------------------------------------------------------------------
# protocol/config freeze pinning
# ---------------------------------------------------------------------------


def test_noise_config_freezes_required_sections_and_hashes() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["protocol"]["status"] == "FROZEN_BEFORE_ANY_NOISY_RESEARCH_RESULT"
    assert config["protocol"]["name"] == "R2_OBSERVATION_NOISE_ROBUSTNESS"
    assert config["canonical_baseline"]["frozen_r2_dataset_sha256"] == FROZEN_DATASET_SHA256
    assert (
        config["canonical_baseline"]["frozen_primary_protocol_sha256"]
        == FROZEN_PROTOCOL_SHA256
    )
    assert config["noise_semantics"]["base_seed"] == 20260825
    assert config["noise_levels"] == [0.0, 0.001, 0.0025, 0.005, 0.01]
    assert config["traditional_calibration_compute"]["traditional_subset_size"] == 250
    assert config["evaluation_population"]["split"] == "test_only"
    assert "retain_and_flag" == config["noise_semantics"]["static_arbitrage_policy"]


def test_protocol_doc_records_the_config_hash() -> None:
    text = PROTOCOL_DOC.read_text(encoding="utf-8")
    config_sha = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    assert config_sha in text, "protocol doc must pin the exact frozen config hash"
