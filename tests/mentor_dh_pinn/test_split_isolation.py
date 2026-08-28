from __future__ import annotations

from pathlib import Path

import numpy as np

from src.mentor_dh_pinn.config import load_baseline_config
from src.mentor_dh_pinn.synthetic_data import generate_synthetic_dataset
from src.mentor_dh_pinn.model import DoubleHestonForwardPINN
from src.mentor_dh_pinn.synthetic_data import SyntheticDataset
from src.mentor_dh_pinn.trainer import train_baseline


def test_split_id_streams_are_disjoint_and_stable_when_other_counts_change(tmp_path: Path) -> None:
    base = load_baseline_config().with_overrides(
        train_count=4, validation_count=3, test_count=2, max_epochs=1, patience=1
    )
    changed = base.with_overrides(
        train_count=7, validation_count=3, test_count=2, max_epochs=1, patience=1
    )
    left = generate_synthetic_dataset(tmp_path / "left", config=base)
    right = generate_synthetic_dataset(tmp_path / "right", config=changed)
    sets = [set(left.split_ids[left.split_names == name]) for name in ("train", "validation", "test")]
    assert all(sets[index].isdisjoint(sets[other]) for index in range(3) for other in range(index + 1, 3))
    for name in ("validation", "test"):
        assert np.array_equal(
            left.split_ids[left.split_names == name],
            right.split_ids[right.split_names == name],
        )


def test_trainer_never_requests_test_indices(monkeypatch, tmp_path: Path) -> None:
    config = load_baseline_config().with_overrides(
        train_count=2, validation_count=2, test_count=2, max_epochs=1, patience=1
    )
    dataset = generate_synthetic_dataset(tmp_path / "data", config=config)
    original = SyntheticDataset.indices

    def guarded(self: SyntheticDataset, split: str):
        if split == "test":
            raise AssertionError("trainer requested the sealed test split")
        return original(self, split)

    monkeypatch.setattr(SyntheticDataset, "indices", guarded)
    model = DoubleHestonForwardPINN(
        feature_min=config.domain.feature_min, feature_max=config.domain.feature_max
    )
    train_baseline(model, dataset, tmp_path / "run", config=config)
