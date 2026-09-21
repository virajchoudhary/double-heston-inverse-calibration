"""Contract tests for DOUBLE_DATA_V2_REPRODUCIBLE."""

import hashlib
import json

import numpy as np

from src.mentor_dh_pinn.reproducible_regular_pinn_data import (
    CONTRACT_ID,
    PROPOSAL_BLOCK_SIZE,
    generate_dataset,
    lhs_unit_block,
    write_deterministic_npz,
)


def test_deterministic_npz_is_byte_identical(tmp_path) -> None:
    arrays = {
        "z": np.array([1.25, -2.5], dtype=np.float64),
        "a": np.arange(12, dtype=np.int64).reshape(3, 4),
    }
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    write_deterministic_npz(first, arrays)
    write_deterministic_npz(second, arrays)
    assert first.read_bytes() == second.read_bytes()


def test_block_latin_hypercube_is_repeatable_and_stratified() -> None:
    first = lhs_unit_block(906210, 0, 12)
    second = lhs_unit_block(906210, 0, 12)
    assert np.array_equal(first, second)
    assert first.shape == (PROPOSAL_BLOCK_SIZE, 12)
    expected = np.arange(PROPOSAL_BLOCK_SIZE)
    for dimension in range(first.shape[1]):
        np.testing.assert_array_equal(
            np.sort(np.floor(first[:, dimension] * PROPOSAL_BLOCK_SIZE).astype(int)),
            expected,
        )


def test_small_dataset_replays_with_exact_hashes_and_counts(tmp_path) -> None:
    first_path = tmp_path / "first"
    second_path = tmp_path / "second"
    first = generate_dataset(
        first_path,
        train_count=16,
        validation_count=8,
        collocation_count=12,
        enforce_runtime=False,
    )
    second = generate_dataset(
        second_path,
        train_count=16,
        validation_count=8,
        collocation_count=12,
        enforce_runtime=False,
    )

    assert first["contract_id"] == CONTRACT_ID
    assert first["splits"]["train"]["accepted"] == 16
    assert first["splits"]["validation"]["accepted"] == 8
    assert first["splits"]["collocation"]["rows"] == 12
    assert {name: record["sha256"] for name, record in first["artifacts"].items()} == {
        name: record["sha256"] for name, record in second["artifacts"].items()
    }
    assert first["manifest_sha256"] == second["manifest_sha256"]

    for name in ("train.npz", "validation.npz"):
        with np.load(first_path / name) as data:
            assert data["usable"].all()
            assert np.isfinite(data["q"]).all()
            assert np.isfinite(data["price"]).all()
            assert np.isfinite(data["w"]).all()
            assert np.isfinite(data["g"]).all()
            assert np.isfinite(data["dg_du"]).all()
            assert np.all(np.diff(data["proposal_index"]) > 0)

    manifest_bytes = (first_path / "manifest.json").read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == first["manifest_sha256"]
    manifest = json.loads(manifest_bytes)
    assert manifest["generator_base_commit"]
    assert manifest["rng"]["algorithm"].startswith("numpy.random.PCG64")
    assert manifest["device_policy"].startswith("CPU only")
