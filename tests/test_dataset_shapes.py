from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ann_inverse_calibration.src.constants import NOT_RESEARCH_DATA
from ann_inverse_calibration.src.dataset import SurfaceParameterDataset
from ann_inverse_calibration.src.surface_grid import (
    build_surface_grid,
    expected_input_size,
    flatten_surface_prices,
)
from ann_inverse_calibration.src.synthetic_dataset import generate_smoke_test_dataset


def test_surface_grid_and_flattening_are_deterministic() -> None:
    first = build_surface_grid(100.0)
    second = build_surface_grid(100.0)
    assert first.equals(second)
    assert len(first) == expected_input_size() == 108
    assert set(first.iloc[:54]["option_type"]) == {"call"}
    assert set(first.iloc[54:]["option_type"]) == {"put"}
    calls = np.arange(54.0)
    puts = np.arange(100.0, 154.0)
    flattened = flatten_surface_prices(calls, puts)
    np.testing.assert_array_equal(flattened[:54], calls)
    np.testing.assert_array_equal(flattened[54:], puts)


def test_dataset_split_keeps_complete_surfaces_together(tmp_path: Path) -> None:
    output = tmp_path / "smoke_test" / "dataset"
    frame = generate_smoke_test_dataset(output, n_surfaces=12, seed=42)
    assert set(frame.groupby("surface_id")["split"].nunique()) == {1}
    split_ids = {
        split: set(frame.loc[frame["split"] == split, "surface_id"])
        for split in ("train", "validation", "test")
    }
    assert not (split_ids["train"] & split_ids["validation"])
    assert not (split_ids["train"] & split_ids["test"])
    assert not (split_ids["validation"] & split_ids["test"])
    with pytest.raises(ValueError, match="allow_not_research_data"):
        SurfaceParameterDataset.from_surface_frame(frame)
    dataset = SurfaceParameterDataset.from_surface_frame(
        frame, allow_not_research_data=True
    )
    assert len(dataset) == 12
    assert dataset.features.shape == (12, 108)
    assert dataset.targets.shape == (12, 10)


def test_smoke_outputs_contain_not_research_marker(tmp_path: Path) -> None:
    output = tmp_path / "smoke_test"
    frame = generate_smoke_test_dataset(output, n_surfaces=6, seed=7)
    metadata = (output / "dataset_metadata.json").read_text(encoding="utf-8")
    assert set(frame["data_status"]) == {NOT_RESEARCH_DATA}
    assert NOT_RESEARCH_DATA in metadata
