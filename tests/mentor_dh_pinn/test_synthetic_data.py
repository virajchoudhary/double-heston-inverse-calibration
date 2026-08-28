from __future__ import annotations

from pathlib import Path

import numpy as np

import src.mentor_dh_pinn.synthetic_data as synthetic_module
from src.mentor_dh_pinn.config import load_baseline_config
from src.mentor_dh_pinn.synthetic_data import (
    generate_synthetic_dataset,
    load_synthetic_dataset,
)


def test_small_dataset_is_deterministic_and_manifested(tmp_path: Path) -> None:
    config = load_baseline_config().with_overrides(
        train_count=4, validation_count=3, test_count=2, max_epochs=1, patience=1
    )
    left = generate_synthetic_dataset(tmp_path / "left", config=config)
    right = generate_synthetic_dataset(tmp_path / "right", config=config)
    assert left.size == 9
    assert np.array_equal(left.features, right.features)
    assert np.array_equal(left.reference_prices, right.reference_prices)
    assert np.array_equal(left.split_ids, right.split_ids)
    assert left.split_id_hashes() == right.split_id_hashes()
    restored = load_synthetic_dataset(tmp_path / "left", config=config)
    assert np.array_equal(restored.features, left.features)
    assert restored.manifest["option_type"] == "call"
    assert restored.manifest["pricing_node_count"] == 64
    assert np.isfinite(left.reference_prices).all()
    assert np.all(left.reference_prices >= 0.0)
    assert (tmp_path / "left" / "dataset_manifest.json").exists()
    assert (tmp_path / "left" / "parameter_provenance.json").exists()
    assert left.manifest["counts"] == {"train": 4, "validation": 3, "test": 2}
    split_sets = {
        name: set(left.split_ids[left.split_names == name])
        for name in ("train", "validation", "test")
    }
    assert split_sets["train"].isdisjoint(split_sets["validation"])
    assert split_sets["train"].isdisjoint(split_sets["test"])
    assert split_sets["validation"].isdisjoint(split_sets["test"])


def test_variance_states_follow_theta_rule(tmp_path: Path) -> None:
    config = load_baseline_config().with_overrides(
        train_count=8, validation_count=2, test_count=2, max_epochs=1, patience=1
    )
    dataset = generate_synthetic_dataset(tmp_path, config=config)
    source = dataset.parameter_source.vector
    assert np.all(dataset.variance_slow >= max(config.domain.variance_floor, 0.25 * source[1]))
    assert np.all(dataset.variance_slow <= min(config.domain.variance_ceiling, 2.0 * source[1]))
    assert np.all(dataset.variance_fast >= max(config.domain.variance_floor, 0.25 * source[6]))
    assert np.all(dataset.variance_fast <= min(config.domain.variance_ceiling, 2.0 * source[6]))


def test_pricer_receives_exact_sampled_variance_state_substitution(monkeypatch, tmp_path: Path) -> None:
    calls: list[np.ndarray] = []

    def fake_pricer(*args, **kwargs):
        calls.append(np.asarray(args[5], dtype=np.float64).copy())
        return 0.25

    monkeypatch.setattr(synthetic_module, "price_double_heston_call", fake_pricer)
    config = load_baseline_config().with_overrides(
        train_count=1, validation_count=1, test_count=1, max_epochs=1, patience=1
    )
    dataset = generate_synthetic_dataset(tmp_path, config=config)
    assert len(calls) == dataset.size
    structural = [0, 1, 2, 3, 5, 6, 7, 8]
    for row, parameters in zip(dataset.features, calls, strict=True):
        assert parameters[4] == row[1]
        assert parameters[9] == row[2]
        np.testing.assert_array_equal(
            parameters[structural], dataset.parameter_source.vector[structural]
        )
