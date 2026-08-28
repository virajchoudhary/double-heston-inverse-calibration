from __future__ import annotations

from pathlib import Path

import numpy as np

from src.constraints import validate_parameters
from src.mentor_dh_pinn.parameter_source import (
    EXPECTED_FIRST_PARAMETER_HASH,
    EXPECTED_FIRST_PARAMETER_VECTOR,
    EXPECTED_FIRST_SURFACE_ID,
    FROZEN_SURFACES_SHA256,
    select_first_eligible_train_record,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SURFACES = REPO_ROOT / "data/final_r2_clean_10000/surfaces.jsonl"


def test_first_train_parameter_source_is_hash_verified_and_valid() -> None:
    source = select_first_eligible_train_record(
        SURFACES,
        expected_sha256=FROZEN_SURFACES_SHA256,
        expected_parameter_hash=EXPECTED_FIRST_PARAMETER_HASH,
    )
    assert source.surface_id == EXPECTED_FIRST_SURFACE_ID
    assert source.split == "train"
    assert source.parameter_hash == EXPECTED_FIRST_PARAMETER_HASH
    assert validate_parameters(source.vector)["is_valid"]
    assert tuple(source.mapping) == (
        "kappa_slow", "theta_slow", "sigma_slow", "rho_slow", "v0_slow",
        "kappa_fast", "theta_fast", "sigma_fast", "rho_fast", "v0_fast",
    )
    assert np.allclose(source.vector, EXPECTED_FIRST_PARAMETER_VECTOR, rtol=0.0, atol=2.0e-14)


def test_parameter_source_substitutes_only_v0_state_fields() -> None:
    source = select_first_eligible_train_record(SURFACES)
    conditioned = source.parameters_for_state(0.25 * source.vector[1], 0.25 * source.vector[6])
    assert np.array_equal(conditioned[[0, 1, 2, 3, 5, 6, 7, 8]], source.structural_vector)
    assert conditioned[4] != source.vector[4]
    assert conditioned[9] != source.vector[9]
    assert validate_parameters(conditioned)["is_valid"]


def test_parameter_selection_is_deterministic() -> None:
    left = select_first_eligible_train_record(SURFACES)
    right = select_first_eligible_train_record(SURFACES)
    assert left.surface_id == right.surface_id
    assert left.parameter_hash == right.parameter_hash
    assert np.array_equal(left.vector, right.vector)
