"""Structure-aware coordinates for the unchanged canonical Double Heston target.

The transform is one-to-one on the numerical interior of the same finite
canonical search envelope used by the reviewed NTPC pilot.  Total/allocation
coordinates change optimization geometry only; the returned scientific vector
always retains the canonical ten-parameter order.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.special import expit

from .constants import PARAMETER_NAMES
from .constraints import validate_parameters


TRANSFORMED_NAMES = (
    "z_v0_total",
    "z_alpha_v",
    "z_theta_total",
    "z_alpha_theta",
    "z_kappa_slow",
    "z_delta_kappa",
    "z_sigma_slow",
    "z_sigma_fast",
    "z_rho_slow",
    "z_rho_fast",
)
ORDERING_EPSILON = 1.0e-5
FELLER_INTERIOR_FACTOR = 1.0 - 1.0e-7
Z_LIMIT = 35.0
_UNIT_EPSILON = float(expit(-Z_LIMIT))


def _unit(value: float) -> float:
    return float(expit(np.clip(value, -Z_LIMIT, Z_LIMIT)))


def _logit_unit(value: float) -> float:
    unit = float(np.clip(value, _UNIT_EPSILON, 1.0 - _UNIT_EPSILON))
    return float(math.log(unit) - math.log1p(-unit))


def _bounded(value: float, lower: float, upper: float) -> float:
    if not lower < upper:
        raise ValueError(f"empty transformed interval [{lower}, {upper}]")
    return float(lower + _unit(value) * (upper - lower))


def _inverse_bounded(value: float, lower: float, upper: float) -> float:
    if not lower < value < upper:
        raise ValueError(f"value {value} is not in numerical interior ({lower}, {upper})")
    return _logit_unit((value - lower) / (upper - lower))


def _total_pair_from_coordinates(
    total_coordinate: float,
    allocation_coordinate: float,
    slow_bounds: tuple[float, float],
    fast_bounds: tuple[float, float],
) -> tuple[float, float, float, float]:
    slow_lower, slow_upper = slow_bounds
    fast_lower, fast_upper = fast_bounds
    total = _bounded(
        total_coordinate,
        slow_lower + fast_lower,
        slow_upper + fast_upper,
    )
    conditional_slow_lower = max(slow_lower, total - fast_upper)
    conditional_slow_upper = min(slow_upper, total - fast_lower)
    slow = _bounded(
        allocation_coordinate,
        conditional_slow_lower,
        conditional_slow_upper,
    )
    fast = float(total - slow)
    return total, float(slow / total), slow, fast


def _total_pair_to_coordinates(
    slow: float,
    fast: float,
    slow_bounds: tuple[float, float],
    fast_bounds: tuple[float, float],
) -> tuple[float, float]:
    slow_lower, slow_upper = slow_bounds
    fast_lower, fast_upper = fast_bounds
    total = float(slow + fast)
    total_coordinate = _inverse_bounded(
        total,
        slow_lower + fast_lower,
        slow_upper + fast_upper,
    )
    conditional_slow_lower = max(slow_lower, total - fast_upper)
    conditional_slow_upper = min(slow_upper, total - fast_lower)
    allocation_coordinate = _inverse_bounded(
        slow,
        conditional_slow_lower,
        conditional_slow_upper,
    )
    return total_coordinate, allocation_coordinate


def _sigma_upper(
    name: str,
    kappa: float,
    theta: float,
    hard_bounds: Mapping[str, tuple[float, float]],
) -> float:
    return float(
        min(
            hard_bounds[name][1],
            math.sqrt(2.0 * kappa * theta) * FELLER_INTERIOR_FACTOR,
        )
    )


def structured_to_canonical(
    transformed: Sequence[float],
    hard_bounds: Mapping[str, tuple[float, float]],
) -> np.ndarray:
    """Map structure-aware coordinates to canonical scientific order."""
    z = np.asarray(transformed, dtype=np.float64)
    if z.shape != (10,) or not np.isfinite(z).all():
        raise ValueError("transformed coordinate must have shape (10,) and be finite")

    v0_total, alpha_v, v0_slow, v0_fast = _total_pair_from_coordinates(
        z[0], z[1], hard_bounds["v0_slow"], hard_bounds["v0_fast"]
    )
    theta_total, alpha_theta, theta_slow, theta_fast = _total_pair_from_coordinates(
        z[2], z[3], hard_bounds["theta_slow"], hard_bounds["theta_fast"]
    )

    kappa_slow = _bounded(z[4], *hard_bounds["kappa_slow"])
    fast_lower = max(
        hard_bounds["kappa_fast"][0],
        kappa_slow + ORDERING_EPSILON,
    )
    kappa_fast = _bounded(z[5], fast_lower, hard_bounds["kappa_fast"][1])

    sigma_slow = _bounded(
        z[6],
        hard_bounds["sigma_slow"][0],
        _sigma_upper("sigma_slow", kappa_slow, theta_slow, hard_bounds),
    )
    sigma_fast = _bounded(
        z[7],
        hard_bounds["sigma_fast"][0],
        _sigma_upper("sigma_fast", kappa_fast, theta_fast, hard_bounds),
    )

    rho_slow = _bounded(z[8], *hard_bounds["rho_slow"])
    disk_limit = math.sqrt(max(1.0 - rho_slow**2, 0.0))
    rho_fast_lower = max(hard_bounds["rho_fast"][0], -disk_limit)
    rho_fast_upper = min(hard_bounds["rho_fast"][1], disk_limit)
    rho_fast = _bounded(z[9], rho_fast_lower, rho_fast_upper)

    vector = np.asarray(
        [
            kappa_slow,
            theta_slow,
            sigma_slow,
            rho_slow,
            v0_slow,
            kappa_fast,
            theta_fast,
            sigma_fast,
            rho_fast,
            v0_fast,
        ],
        dtype=np.float64,
    )
    diagnostics = canonical_diagnostics(vector, hard_bounds)
    if not diagnostics["is_valid"]:
        raise RuntimeError(f"structure-aware transform failed: {diagnostics['violations']}")
    if not np.isclose(v0_total, vector[4] + vector[9], rtol=0.0, atol=2e-15):
        raise RuntimeError("v0 total decomposition failed")
    if not np.isclose(theta_total, vector[1] + vector[6], rtol=0.0, atol=2e-15):
        raise RuntimeError("theta total decomposition failed")
    if not (0.0 < alpha_v < 1.0 and 0.0 < alpha_theta < 1.0):
        raise RuntimeError("allocation coordinate left the open unit interval")
    return vector


def canonical_to_structured(
    parameters: Sequence[float],
    hard_bounds: Mapping[str, tuple[float, float]],
) -> np.ndarray:
    """Invert an interior canonical vector into structure-aware coordinates."""
    vector = np.asarray(parameters, dtype=np.float64)
    diagnostics = canonical_diagnostics(vector, hard_bounds)
    if not diagnostics["is_valid"]:
        raise ValueError(f"canonical vector is outside the experiment envelope: {diagnostics['violations']}")

    z = np.empty(10, dtype=np.float64)
    z[0], z[1] = _total_pair_to_coordinates(
        vector[4], vector[9], hard_bounds["v0_slow"], hard_bounds["v0_fast"]
    )
    z[2], z[3] = _total_pair_to_coordinates(
        vector[1], vector[6], hard_bounds["theta_slow"], hard_bounds["theta_fast"]
    )
    z[4] = _inverse_bounded(vector[0], *hard_bounds["kappa_slow"])
    fast_lower = max(hard_bounds["kappa_fast"][0], vector[0] + ORDERING_EPSILON)
    z[5] = _inverse_bounded(vector[5], fast_lower, hard_bounds["kappa_fast"][1])
    z[6] = _inverse_bounded(
        vector[2],
        hard_bounds["sigma_slow"][0],
        _sigma_upper("sigma_slow", vector[0], vector[1], hard_bounds),
    )
    z[7] = _inverse_bounded(
        vector[7],
        hard_bounds["sigma_fast"][0],
        _sigma_upper("sigma_fast", vector[5], vector[6], hard_bounds),
    )
    z[8] = _inverse_bounded(vector[3], *hard_bounds["rho_slow"])
    disk_limit = math.sqrt(max(1.0 - vector[3] ** 2, 0.0))
    rho_fast_lower = max(hard_bounds["rho_fast"][0], -disk_limit)
    rho_fast_upper = min(hard_bounds["rho_fast"][1], disk_limit)
    z[9] = _inverse_bounded(vector[8], rho_fast_lower, rho_fast_upper)
    return z


def canonical_diagnostics(
    parameters: Sequence[float],
    hard_bounds: Mapping[str, tuple[float, float]],
) -> dict[str, Any]:
    """Validate structural rules plus the exact pilot numerical envelope."""
    vector = np.asarray(parameters, dtype=np.float64)
    try:
        structural = validate_parameters(vector)
    except ValueError as error:
        return {"is_valid": False, "violations": [str(error)]}
    violations = list(structural["violations"])
    for name, value in zip(PARAMETER_NAMES, vector, strict=True):
        lower, upper = hard_bounds[name]
        if not lower < float(value) < upper:
            violations.append(f"{name} must be in numerical interior ({lower}, {upper})")
    if vector[5] - vector[0] <= ORDERING_EPSILON:
        violations.append(f"kappa ordering gap must exceed {ORDERING_EPSILON}")
    radius = float(math.hypot(vector[3], vector[8]))
    return {
        **structural,
        "is_valid": not violations,
        "correlation_radius": radius,
        "violations": violations,
    }


def derived_coordinates(parameters: Sequence[float]) -> dict[str, float]:
    """Return interpretable totals/allocations; alphas are not model parameters."""
    p = np.asarray(parameters, dtype=np.float64)
    if p.shape != (10,):
        raise ValueError("canonical vector must have shape (10,)")
    v0_total = float(p[4] + p[9])
    theta_total = float(p[1] + p[6])
    return {
        "v0_total": v0_total,
        "alpha_v": float(p[4] / v0_total),
        "theta_total": theta_total,
        "alpha_theta": float(p[1] / theta_total),
        "delta_kappa": float(p[5] - p[0]),
        "slow_half_life_days": float(365.0 * math.log(2.0) / p[0]),
        "fast_half_life_days": float(365.0 * math.log(2.0) / p[5]),
    }
