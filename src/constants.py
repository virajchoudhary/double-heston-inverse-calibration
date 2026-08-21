"""Immutable contracts shared by the ordinary ANN baseline."""

from __future__ import annotations

from typing import Final

PARAMETER_NAMES: Final[list[str]] = [
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
PARAMETER_INDICES: Final[dict[str, int]] = {
    name: index for index, name in enumerate(PARAMETER_NAMES)
}
PARAMETER_COUNT: Final[int] = len(PARAMETER_NAMES)

DEFAULT_SEED: Final[int] = 42
CALL_OPTION: Final[str] = "call"
PUT_OPTION: Final[str] = "put"
OPTION_TYPES: Final[tuple[str, str]] = (CALL_OPTION, PUT_OPTION)

INFRASTRUCTURE_TEST_STAGE: Final[str] = "infrastructure_test"
RESEARCH_STAGE: Final[str] = "research"
PROJECT_STAGE_LABELS: Final[tuple[str, str]] = (
    INFRASTRUCTURE_TEST_STAGE,
    RESEARCH_STAGE,
)
NOT_RESEARCH_DATA: Final[str] = "NOT_RESEARCH_DATA"
GENUINE_CANONICAL_DOUBLE_HESTON_SYNTHETIC_DATA: Final[str] = (
    "GENUINE_CANONICAL_DOUBLE_HESTON_SYNTHETIC_DATA"
)

LOG_MONEYNESS_GRID: Final[list[float]] = [
    -0.30,
    -0.20,
    -0.10,
    -0.05,
    0.00,
    0.05,
    0.10,
    0.20,
    0.30,
]
MATURITY_DAYS_GRID: Final[list[int]] = [7, 14, 30, 60, 90, 180]
