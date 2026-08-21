"""Validation for the canonical ten-parameter Double Heston contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .constants import PARAMETER_COUNT, PARAMETER_INDICES, PARAMETER_NAMES


def feller_gap(kappa: float, theta: float, sigma: float) -> float:
    """Return the strict Feller diagnostic ``2*kappa*theta - sigma**2``."""
    return float(2.0 * kappa * theta - sigma**2)


def positivity_is_valid(parameters: Sequence[float]) -> bool:
    """Check all positive-only structural and state parameters."""
    vector = _coerce_vector(parameters)
    positive_names = (
        "kappa_slow",
        "theta_slow",
        "sigma_slow",
        "v0_slow",
        "kappa_fast",
        "theta_fast",
        "sigma_fast",
        "v0_fast",
    )
    return all(vector[PARAMETER_INDICES[name]] > 0.0 for name in positive_names)


def correlations_are_bounded(rho_slow: float, rho_fast: float) -> bool:
    """Require both correlations to lie strictly inside ``(-1, 1)``."""
    return bool(-1.0 < rho_slow < 1.0 and -1.0 < rho_fast < 1.0)


def correlation_disk_value(rho_slow: float, rho_fast: float) -> float:
    """Return the squared radius of the joint-correlation disk."""
    return float(rho_slow**2 + rho_fast**2)


def joint_correlation_disk_is_valid(rho_slow: float, rho_fast: float) -> bool:
    """Require ``rho_slow**2 + rho_fast**2 < 1``."""
    return correlation_disk_value(rho_slow, rho_fast) < 1.0


def slow_fast_ordering_is_valid(kappa_slow: float, kappa_fast: float) -> bool:
    """Require the slow factor to mean-revert more slowly than the fast factor."""
    return bool(kappa_slow < kappa_fast)


def dictionary_to_vector(parameters: Mapping[str, float]) -> np.ndarray:
    """Convert a complete named parameter mapping to canonical vector order."""
    missing = [name for name in PARAMETER_NAMES if name not in parameters]
    extras = sorted(set(parameters) - set(PARAMETER_NAMES))
    if missing or extras:
        raise ValueError(f"Parameter keys mismatch; missing={missing}, extras={extras}")
    return _coerce_vector([parameters[name] for name in PARAMETER_NAMES])


def vector_to_dictionary(parameters: Sequence[float]) -> dict[str, float]:
    """Convert a canonical parameter vector to a named dictionary."""
    vector = _coerce_vector(parameters)
    return {name: float(vector[index]) for index, name in enumerate(PARAMETER_NAMES)}


def validate_parameters(parameters: Sequence[float] | Mapping[str, float]) -> dict[str, Any]:
    """Return complete validity diagnostics without hiding individual failures."""
    vector = (
        dictionary_to_vector(parameters)
        if isinstance(parameters, Mapping)
        else _coerce_vector(parameters)
    )
    values = vector_to_dictionary(vector)
    positive_valid = positivity_is_valid(vector)
    ordering_valid = slow_fast_ordering_is_valid(
        values["kappa_slow"], values["kappa_fast"]
    )
    slow_gap = feller_gap(
        values["kappa_slow"], values["theta_slow"], values["sigma_slow"]
    )
    fast_gap = feller_gap(
        values["kappa_fast"], values["theta_fast"], values["sigma_fast"]
    )
    correlations_valid = correlations_are_bounded(
        values["rho_slow"], values["rho_fast"]
    )
    disk_value = correlation_disk_value(values["rho_slow"], values["rho_fast"])

    violations: list[str] = []
    if not positive_valid:
        violations.append("positive-only parameters must be strictly positive")
    if not ordering_valid:
        violations.append("kappa_slow must be strictly less than kappa_fast")
    if slow_gap <= 0.0:
        violations.append("slow-factor Feller gap must be strictly positive")
    if fast_gap <= 0.0:
        violations.append("fast-factor Feller gap must be strictly positive")
    if not correlations_valid:
        violations.append("rho_slow and rho_fast must each lie strictly inside (-1, 1)")
    if disk_value >= 1.0:
        violations.append("rho_slow^2 + rho_fast^2 must be strictly less than 1")

    return {
        "is_valid": not violations,
        "positive_valid": positive_valid,
        "ordering_valid": ordering_valid,
        "slow_feller_gap": slow_gap,
        "fast_feller_gap": fast_gap,
        "correlation_disk_value": disk_value,
        "violations": violations,
    }


def _coerce_vector(parameters: Sequence[float]) -> np.ndarray:
    vector = np.asarray(parameters, dtype=np.float64)
    if vector.shape != (PARAMETER_COUNT,):
        raise ValueError(
            f"Expected parameter shape ({PARAMETER_COUNT},), got {vector.shape}"
        )
    if not np.isfinite(vector).all():
        raise ValueError("Parameters must all be finite")
    return vector
