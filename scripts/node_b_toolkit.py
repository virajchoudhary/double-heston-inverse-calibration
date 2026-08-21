"""Node B overnight diagnostic toolkit (identifiability/calibration research).

This module provides a vectorized re-implementation of the canonical Double
Heston Gauss-Laguerre pricing arithmetic for DIAGNOSTIC use only.  It replicates
the frozen production algorithm (``src/double_heston.py``) quote-by-quote but
evaluates all quotes of a surface in a single broadcast, and it is validated
against the production pricer before any experiment uses it.  The production
pricer remains the canonical reference; nothing in this file alters it.

Conventions reused verbatim from the committed G2 diagnostics:
- spot-normalized observables (price / spot);
- full-range-scaled parameters ((theta - lower) / width);
- central-difference Jacobians with validity-aware step reduction
  (``JACOBIAN_RELATIVE_STEP = 1e-4``);
- multiplicative lognormal observational noise;
- TRF least-squares on the unconstrained constraint transform from
  ``src.calibrate_double_heston``.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.calibrate_double_heston import (  # noqa: E402
    boundary_diagnostics,
    load_hard_safety_bounds,
    unconstrained_to_parameters,
)
from src.constants import CALL_OPTION, LOG_MONEYNESS_GRID, MATURITY_DAYS_GRID, PARAMETER_NAMES, PUT_OPTION  # noqa: E402
from src.constraints import validate_parameters  # noqa: E402

BOUNDS_PATH = REPOSITORY_ROOT / "configs" / "parameter_bounds_PROVISIONAL.yaml"
EVIDENCE_ROOT = REPOSITORY_ROOT / ".ai-research" / "overnight" / "2026-08-22" / "node-B"
EXPERIMENTS_LOG = EVIDENCE_ROOT / "EXPERIMENTS.jsonl"

# Frozen diagnostic contract for the full-grid work tonight.  The G2 market
# carry contract used per-maturity rates (0.0600, 0.0625) and dividends
# (0.0200, 0.0225); the provisional 108-grid synthetic pipeline uses a single
# constant carry, so we declare one and hold it fixed everywhere.
SPOT = 100.0
RISK_FREE_RATE = 0.06
DIVIDEND_YIELD = 0.02
NODE_COUNT = 64

JACOBIAN_RELATIVE_STEP = 1.0e-4
PRACTICAL_RANK_RELATIVE_TOLERANCE = 1.0e-6
NEAR_PRICE_EQUIVALENCE_RMSE = 2.5e-7
MATERIAL_DISPLACEMENT_RMSE = 0.05

FULL_MATURITY_DAYS = tuple(MATURITY_DAYS_GRID)
FULL_MONEYNESS = tuple(LOG_MONEYNESS_GRID)


def _gauss_laguerre(node_count: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.laguerre.laggauss(int(node_count))
    return nodes.astype(np.float64), weights.astype(np.float64)


def price_surface_fast(
    parameters: Sequence[float],
    strikes: np.ndarray,
    maturities_years: np.ndarray,
    option_types: Sequence[str],
    *,
    spot: float = SPOT,
    risk_free_rate: float = RISK_FREE_RATE,
    dividend_yield: float = DIVIDEND_YIELD,
    node_count: int = NODE_COUNT,
) -> np.ndarray:
    """Vectorized canonical Gauss-Laguerre Double Heston surface pricing.

    Replicates ``src.double_heston.price_double_heston_surface`` arithmetic
    exactly: same Little-Heston-Trap exponent, same ``d`` branch convention,
    same ``log1p`` forms, same Gauss-Laguerre rule, puts via put-call parity.
    All quotes are priced in one broadcast; inputs are assumed pre-validated
    (use ``validate_parameters`` upstream in experiments that need it).
    """
    vector = np.asarray(parameters, dtype=np.float64)
    strikes = np.asarray(strikes, dtype=np.float64)
    maturities = np.asarray(maturities_years, dtype=np.float64)
    quote_count = strikes.shape[0]
    if maturities.shape != (quote_count,) or len(option_types) != quote_count:
        raise ValueError("strikes, maturities_years, option_types must be quote-aligned")
    nodes, weights = _gauss_laguerre(node_count)

    # (quotes, nodes) broadcasts; maturity/strike vary per quote.
    u = nodes[None, :]                        # (1, N)
    T = maturities[:, None]                   # (Q, 1)
    log_k = np.log(strikes)[:, None]          # (Q, 1)

    def _exponent_for(b: np.ndarray, offset: int, u_values: np.ndarray) -> np.ndarray:
        kappa, theta, sigma, rho, v0 = (
            float(vector[offset + index]) for index in range(5)
        )
        iu = 1j * u_values
        discriminant = b * b + sigma * sigma * (u_values * u_values + iu)
        d = np.sqrt(discriminant)
        d = np.where(np.real(d) < 0.0, -d, d)
        denominator = b + d
        g = (b - d) / denominator
        exp_minus_dt = np.exp(-d * T)
        numerator = 1.0 - g * exp_minus_dt
        log_ratio = np.log1p(-g * exp_minus_dt) - np.log1p(-g)
        c_term = (kappa * theta / sigma**2) * ((b - d) * T - 2.0 * log_ratio)
        d_term = ((b - d) / sigma**2) * ((-np.expm1(-d * T)) / numerator)
        return c_term + d_term * v0

    def characteristic(u_values: np.ndarray) -> np.ndarray:
        iu = 1j * u_values
        b_slow = vector[0] - vector[3] * vector[2] * iu
        b_fast = vector[5] - vector[8] * vector[7] * iu
        exponent = (
            iu * (np.log(spot) + (risk_free_rate - dividend_yield) * T)
            + _exponent_for(b_slow, 0, u_values)
            + _exponent_for(b_fast, 5, u_values)
        )
        return np.exp(exponent)

    phi_u = characteristic(u)
    phi_shifted = characteristic(u - 1j)
    # phi(-i): characteristic function at u = -i, one scalar per quote.
    phi_minus_i = characteristic_full_scalar(
        vector, maturities, spot, risk_free_rate, dividend_yield
    )

    oscillation = np.exp(-1j * u * log_k)
    inverse_iu = 1.0 / (1j * u)
    laguerre_compensation = np.exp(u)
    p1_integrand = np.real(oscillation * phi_shifted * inverse_iu / phi_minus_i[:, None])
    p2_integrand = np.real(oscillation * phi_u * inverse_iu)
    p1 = 0.5 + np.sum(weights[None, :] * laguerre_compensation * p1_integrand, axis=1) / np.pi
    p2 = 0.5 + np.sum(weights[None, :] * laguerre_compensation * p2_integrand, axis=1) / np.pi
    calls = (
        spot * np.exp(-dividend_yield * maturities) * p1
        - strikes * np.exp(-risk_free_rate * maturities) * p2
    )
    puts = calls - spot * np.exp(-dividend_yield * maturities) + strikes * np.exp(
        -risk_free_rate * maturities
    )
    option_array = np.asarray(option_types, dtype=str)
    return np.where(option_array == CALL_OPTION, calls, puts)


def characteristic_full_scalar(
    vector: np.ndarray,
    maturities: np.ndarray,
    spot: float,
    risk_free_rate: float,
    dividend_yield: float,
) -> np.ndarray:
    """Characteristic function at ``u = -i`` per quote (production form)."""
    u_value = -1j
    iu = 1j * u_value
    result = np.empty(maturities.shape[0], dtype=np.complex128)
    for offset in (0, 5):
        pass  # placeholder replaced below; kept single implementation
    exponents = []
    for offset in (0, 5):
        kappa, theta, sigma, rho, v0 = (
            float(vector[offset + index]) for index in range(5)
        )
        b = kappa - rho * sigma * iu
        discriminant = b * b + sigma * sigma * (u_value * u_value + iu)
        d = np.sqrt(discriminant)
        if np.real(d) < 0.0:
            d = -d
        denominator = b + d
        g = (b - d) / denominator
        exp_minus_dt = np.exp(-d * maturities)
        numerator = 1.0 - g * exp_minus_dt
        log_ratio = np.log1p(-g * exp_minus_dt) - np.log1p(-g)
        c_term = (kappa * theta / sigma**2) * ((b - d) * maturities - 2.0 * log_ratio)
        d_term = ((b - d) / sigma**2) * ((-np.expm1(-d * maturities)) / numerator)
        exponents.append(c_term + d_term * v0)
    exponent = (
        iu * (np.log(spot) + (risk_free_rate - dividend_yield) * maturities)
        + exponents[0]
        + exponents[1]
    )
    result[:] = np.exp(exponent)
    return result


# ---------------------------------------------------------------------------
# Grids and geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Geometry:
    geometry_id: str
    moneyness_nodes: tuple[float, ...]
    maturity_days: tuple[int, ...]
    option_types: tuple[str, ...]
    rates: tuple[float, ...]
    dividends: tuple[float, ...]

    def build(self, spot: float = SPOT) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return strikes, maturities_years, option_types, rates, dividends aligned per quote."""
        strikes: list[float] = []
        maturities: list[float] = []
        options: list[str] = []
        rate_list: list[float] = []
        dividend_list: list[float] = []
        for option_type in self.option_types:
            for index, days in enumerate(self.maturity_days):
                for node in self.moneyness_nodes:
                    strikes.append(spot * math.exp(node))
                    maturities.append(float(days) / 365.0)
                    options.append(option_type)
                    rate_list.append(self.rates[index])
                    dividend_list.append(self.dividends[index])
        return (
            np.asarray(strikes),
            np.asarray(maturities),
            np.asarray(options, dtype=str),
            np.asarray(rate_list),
            np.asarray(dividend_list),
        )

    @property
    def quote_count(self) -> int:
        return len(self.option_types) * len(self.maturity_days) * len(self.moneyness_nodes)


def full_108_geometry() -> Geometry:
    n = len(FULL_MATURITY_DAYS)
    return Geometry(
        "full108_calls_puts",
        FULL_MONEYNESS,
        FULL_MATURITY_DAYS,
        (CALL_OPTION, PUT_OPTION),
        (RISK_FREE_RATE,) * n,
        (DIVIDEND_YIELD,) * n,
    )


def central5_market_geometry(maturity_days: tuple[int, ...]) -> Geometry:
    """The committed G2 market-supported geometry for comparability."""
    n = len(maturity_days)
    return Geometry(
        f"central5_{maturity_days[0]}_{maturity_days[1]}",
        (-0.10, -0.05, 0.0, 0.05, 0.10),
        maturity_days,
        (CALL_OPTION, PUT_OPTION),
        (0.0600, 0.0625)[:n],
        (0.0200, 0.0225)[:n],
    )


def normalized_observables_fast(
    parameters: Sequence[float],
    geometry: Geometry,
    *,
    node_count: int = NODE_COUNT,
    spot: float = SPOT,
) -> np.ndarray:
    """Spot-normalized prices in G2 convention (calls then puts, per geometry)."""
    strikes, maturities, options, rates, dividends = geometry.build(spot)
    unique_carry = sorted(set(zip(rates.tolist(), dividends.tolist(), strict=True)))
    if len(unique_carry) == 1:
        prices = price_surface_fast(
            parameters,
            strikes,
            maturities,
            options,
            spot=spot,
            risk_free_rate=unique_carry[0][0],
            dividend_yield=unique_carry[0][1],
            node_count=node_count,
        )
        return prices / spot
    pieces: list[np.ndarray] = []
    for rate, dividend in unique_carry:
        mask = (rates == rate) & (dividends == dividend)
        piece = price_surface_fast(
            parameters,
            strikes[mask],
            maturities[mask],
            options[mask],
            spot=spot,
            risk_free_rate=rate,
            dividend_yield=dividend,
            node_count=node_count,
        )
        pieces.append((mask, piece / spot))
    combined = np.zeros(strikes.shape[0], dtype=np.float64)
    for mask, piece in pieces:
        combined[mask] = piece
    # Reorder to the geometry's declared order (option-major, maturity-major).
    return combined


def parameter_widths(bounds: dict[str, tuple[float, float]]) -> np.ndarray:
    return np.asarray([bounds[name][1] - bounds[name][0] for name in PARAMETER_NAMES])


def scaled_coordinates(
    parameters: np.ndarray, bounds: dict[str, tuple[float, float]]
) -> np.ndarray:
    lower = np.asarray([bounds[name][0] for name in PARAMETER_NAMES])
    return (np.asarray(parameters, dtype=np.float64) - lower) / parameter_widths(bounds)


def scaled_parameter_jacobian_fast(
    parameters: Sequence[float],
    geometry: Geometry,
    bounds: dict[str, tuple[float, float]],
    *,
    node_count: int = NODE_COUNT,
    relative_step: float = JACOBIAN_RELATIVE_STEP,
) -> np.ndarray:
    """Central-difference Jacobian of spot-normalized prices w.r.t. range-scaled parameters."""
    vector = np.asarray(parameters, dtype=np.float64)
    widths = parameter_widths(bounds)
    columns: list[np.ndarray] = []
    for index, width in enumerate(widths):
        step = relative_step * width
        for _ in range(12):
            lower = vector.copy()
            upper = vector.copy()
            lower[index] -= step
            upper[index] += step
            if (
                validate_parameters(lower)["is_valid"]
                and validate_parameters(upper)["is_valid"]
            ):
                break
            step *= 0.5
        else:
            raise RuntimeError(
                f"Could not form a valid central difference for {PARAMETER_NAMES[index]}"
            )
        lower_prices = normalized_observables_fast(lower, geometry, node_count=node_count)
        upper_prices = normalized_observables_fast(upper, geometry, node_count=node_count)
        columns.append((upper_prices - lower_prices) * width / (2.0 * step))
    return np.column_stack(columns)


def jacobian_summary(jacobian: np.ndarray) -> dict[str, Any]:
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    largest = float(singular_values[0])
    smallest = float(singular_values[-1])
    numerical_tolerance = largest * singular_values.size * np.finfo(np.float64).eps
    return {
        "largest_singular_value": largest,
        "smallest_singular_value": smallest,
        "condition_number": largest / smallest,
        "numerical_rank": int(np.sum(singular_values > numerical_tolerance)),
        "practical_rank": int(
            np.sum(singular_values > PRACTICAL_RANK_RELATIVE_TOLERANCE * largest)
        ),
        **{
            f"singular_value_{index + 1:02d}": float(value)
            for index, value in enumerate(singular_values)
        },
    }


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def representative_cases() -> "np.ndarray":
    """Return the same four representative truth vectors as the committed
    global-ambiguity diagnostic (first two maximin representatives per
    distribution), preserving case order."""
    import scripts.run_g2_identifiability_analysis as baseline

    bounds = load_hard_safety_bounds(BOUNDS_PATH)
    selected = baseline.select_representative_parameters(bounds, per_distribution=4)
    cases = (
        selected.groupby("distribution", sort=True, group_keys=False)
        .head(2)
        .reset_index(drop=True)
        .copy()
    )
    cases.insert(0, "case_index", np.arange(len(cases), dtype=int))
    cases.insert(1, "case_id", [f"case_{index + 1}" for index in range(len(cases))])
    return cases


def case_vector(row: Any) -> np.ndarray:
    return np.asarray([getattr(row, name) for name in PARAMETER_NAMES], dtype=np.float64)


# ---------------------------------------------------------------------------
# Experiment logging
# ---------------------------------------------------------------------------


def log_experiment(record: dict[str, Any]) -> None:
    EXPERIMENTS_LOG.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **record}
    with EXPERIMENTS_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=float) + "\n")


def multiplicative_noise(
    values: np.ndarray, seed: int, noise_level: float
) -> np.ndarray:
    if noise_level == 0.0:
        return np.asarray(values, dtype=np.float64).copy()
    rng = np.random.default_rng(seed)
    observed = np.asarray(values, dtype=np.float64) * (
        1.0 + rng.normal(0.0, noise_level, size=len(values))
    )
    if np.any(observed < 0.0):
        raise RuntimeError("multiplicative noise produced a negative price")
    return observed
