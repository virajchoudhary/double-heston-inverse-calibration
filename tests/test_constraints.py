from __future__ import annotations

import pytest

from src.constants import PARAMETER_INDICES
from src.constraints import validate_parameters


@pytest.fixture
def valid_vector() -> list[float]:
    return [1.0, 0.10, 0.30, -0.30, 0.08, 3.0, 0.08, 0.50, -0.20, 0.07]


def test_valid_vector_passes(valid_vector: list[float]) -> None:
    diagnostics = validate_parameters(valid_vector)
    assert diagnostics["is_valid"]
    assert diagnostics["violations"] == []


def test_negative_positive_only_parameter_fails(valid_vector: list[float]) -> None:
    valid_vector[PARAMETER_INDICES["v0_slow"]] = -0.01
    diagnostics = validate_parameters(valid_vector)
    assert not diagnostics["is_valid"]
    assert not diagnostics["positive_valid"]


def test_invalid_slow_fast_order_fails(valid_vector: list[float]) -> None:
    valid_vector[PARAMETER_INDICES["kappa_slow"]] = 3.0
    diagnostics = validate_parameters(valid_vector)
    assert not diagnostics["ordering_valid"]
    assert not diagnostics["is_valid"]


def test_negative_slow_feller_gap_fails(valid_vector: list[float]) -> None:
    valid_vector[PARAMETER_INDICES["sigma_slow"]] = 1.0
    diagnostics = validate_parameters(valid_vector)
    assert diagnostics["slow_feller_gap"] < 0.0
    assert not diagnostics["is_valid"]


def test_negative_fast_feller_gap_fails(valid_vector: list[float]) -> None:
    valid_vector[PARAMETER_INDICES["sigma_fast"]] = 1.0
    diagnostics = validate_parameters(valid_vector)
    assert diagnostics["fast_feller_gap"] < 0.0
    assert not diagnostics["is_valid"]


def test_invalid_correlation_disk_fails(valid_vector: list[float]) -> None:
    valid_vector[PARAMETER_INDICES["rho_slow"]] = 0.80
    valid_vector[PARAMETER_INDICES["rho_fast"]] = 0.80
    diagnostics = validate_parameters(valid_vector)
    assert diagnostics["correlation_disk_value"] > 1.0
    assert not diagnostics["is_valid"]
