from __future__ import annotations

import numpy as np

from src.constants import PARAMETER_NAMES
from src.constraints import dictionary_to_vector, vector_to_dictionary


EXPECTED_PARAMETER_NAMES = [
    "kappa_slow",
    "theta_slow",
    "sigma_slow",
    "rho_slow",
    "v0_slow",
    "kappa_fast",
    "theta_fast",
    "sigma_fast",
    "rho_fast",
    "v0_fast",
]


def test_exact_ten_parameter_order() -> None:
    assert PARAMETER_NAMES == EXPECTED_PARAMETER_NAMES


def test_parameter_dictionary_round_trip_uses_locked_order() -> None:
    mapping = {name: float(index + 1) for index, name in enumerate(PARAMETER_NAMES)}
    vector = dictionary_to_vector(mapping)
    np.testing.assert_array_equal(vector, np.arange(1.0, 11.0))
    assert vector_to_dictionary(vector) == mapping
