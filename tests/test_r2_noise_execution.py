"""Execution-path guards for the frozen R2 noise robustness study."""

from __future__ import annotations

import json
import inspect

import numpy as np

from src.r2_noise.execution import (
    CLEAN_DATASET_PATH,
    load_frozen_protocol,
    assert_clean_dataset_identity,
    derive_noisy_record,
    iter_test_records,
    serialize_record,
)
from src.r2_noise.generator import generate_cohorts
from src.r2_noise.neural_evaluation import static_arbitrage_diagnostics
from src.r2_noise.perturbation import perturb_surface_prices


def _first_test_record() -> dict:
    return next(record for _, record in iter_test_records(CLEAN_DATASET_PATH))


def test_derived_record_preserves_clean_payload_and_replays_exactly() -> None:
    clean = _first_test_record()
    derived = derive_noisy_record(
        clean,
        noise_level=0.005,
        label="0.50%",
        clean_sha256=assert_clean_dataset_identity(),
    )
    expected_prices, counters = perturb_surface_prices(
        clean["prices"], clean["surface_id"], clean["slot_keys"], 0.005
    )
    assert derived["prices"] == expected_prices
    assert [item["resample_counter"] for item in derived["observation_noise"]["realizations"]] == counters
    assert derived["metadata"]["user_metadata"]["noise_level"] == clean[
        "metadata"
    ]["user_metadata"]["noise_level"] == 0.0
    for field in ("spot", "rates", "carries", "maturities", "mask", "slot_keys"):
        assert derived[field] == clean[field]
    assert derived["metadata"] == clean["metadata"]
    assert serialize_record(derived) == serialize_record(
        json.loads(serialize_record(derived))
    )


def test_generator_refuses_an_existing_manifest() -> None:
    protocol = load_frozen_protocol()
    positive_labels = [
        label
        for level, label in zip(protocol["noise_levels"], protocol["noise_level_labels"])
        if float(level) > 0.0
    ]
    assert "0.10%" in positive_labels
    assert "refusing to overwrite cohort" in inspect.getsource(generate_cohorts)


def _surface_record(prices: list[float]) -> dict:
    keys = [
        [1, -0.10, "call"],
        [1, -0.05, "call"],
        [1, -0.10, "put"],
        [1, -0.05, "put"],
    ]
    return {
        "surface_id": "diagnostic",
        "prices": prices,
        "mask": [True] * 4,
        "slot_keys": keys,
    }


def test_arbitrage_diagnostics_flag_without_repair() -> None:
    clean = _surface_record([3.0, 2.0, 1.0, 2.0])
    unchanged = json.loads(json.dumps(clean))
    result = static_arbitrage_diagnostics([clean], [unchanged])
    assert result["parity_violation_slot_count_total"] == 0
    assert result["vertical_violation_pair_count_total"] == 0

    violating = json.loads(json.dumps(clean))
    violating["prices"] = [1.0, 2.5, 1.5, 2.0]
    result = static_arbitrage_diagnostics([clean], [violating])
    assert result["parity_violation_slot_count_total"] == 2
    # The changed call at k=-0.10 also creates a raw upward call spread.
    assert result["vertical_violation_pair_count_total"] == 1
