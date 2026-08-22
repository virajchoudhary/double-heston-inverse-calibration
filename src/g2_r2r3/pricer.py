"""Fast vectorized diagnostic pricer for the G2 R2-vs-R3 study.

Adapted (not merged) from the validated Node B overnight toolkit
(``archive/overnight-20260822-node-b``, commits 77b8f2e/61905d0,
``scripts/node_b_toolkit.py``), which itself replicates the frozen production
pricer ``src/double_heston.py`` quote-by-quote (same Little-Heston-Trap
exponent, same ``d``-branch, ``log1p`` forms, same Gauss-Laguerre rule, puts by
parity).  The production pricer remains the canonical source of truth; this
module exists for diagnostic throughput only and MUST be validated against the
production pricer (see ``tests/test_g2_r2r3_harness.py`` and the recorded
validation evidence) before full-matrix use.

Parameter order is the canonical ten-vector order everywhere in this module.
The extension beyond the Node B original is per-quote rate/carry support via
piece-wise pricing (the production surface pricer accepts scalar rate/carry
only, so per-rank market conditioning is priced per constant-carry piece —
the same approach the committed G2 diagnostics used).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..constants import CALL_OPTION


def _gauss_laguerre(node_count: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.laguerre.laggauss(int(node_count))
    return nodes.astype(np.float64), weights.astype(np.float64)


def price_surface_fast(
    parameters: Sequence[float],
    strikes: np.ndarray,
    maturities_years: np.ndarray,
    option_types: Sequence[str],
    *,
    spot: float,
    risk_free_rate: float,
    dividend_yield: float,
    node_count: int = 64,
) -> np.ndarray:
    """Vectorized canonical Gauss-Laguerre Double Heston surface pricing."""
    vector = np.asarray(parameters, dtype=np.float64)
    strikes = np.asarray(strikes, dtype=np.float64)
    maturities = np.asarray(maturities_years, dtype=np.float64)
    quote_count = strikes.shape[0]
    if maturities.shape != (quote_count,) or len(option_types) != quote_count:
        raise ValueError("strikes, maturities_years, option_types must be quote-aligned")
    nodes, weights = _gauss_laguerre(node_count)

    u = nodes[None, :]
    T = maturities[:, None]
    log_k = np.log(strikes)[:, None]

    def _exponent_for(b: np.ndarray, offset: int, u_values: np.ndarray) -> np.ndarray:
        kappa, theta, sigma, rho, v0 = (
            float(vector[offset + index]) for index in range(5)
        )
        iu = 1j * u_values
        discriminant = b * b + sigma * sigma * (u_values * u_values + iu)
        d = np.sqrt(discriminant)
        d = np.where(np.real(d) < 0.0, -d, d)
        g = (b - d) / (b + d)
        exp_minus_dt = np.exp(-d * T)
        log_ratio = np.log1p(-g * exp_minus_dt) - np.log1p(-g)
        c_term = (kappa * theta / sigma**2) * ((b - d) * T - 2.0 * log_ratio)
        d_term = ((b - d) / sigma**2) * ((-np.expm1(-d * T)) / (1.0 - g * exp_minus_dt))
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

    # phi(-i) per quote, evaluated in the production scalar form.
    def characteristic_minus_i() -> np.ndarray:
        u_value = -1j
        iu = 1j * u_value
        total = iu * (np.log(spot) + (risk_free_rate - dividend_yield) * maturities)
        for offset in (0, 5):
            kappa, theta, sigma, rho, v0 = (
                float(vector[offset + index]) for index in range(5)
            )
            b = kappa - rho * sigma * iu
            discriminant = b * b + sigma * sigma * (u_value * u_value + iu)
            d = np.sqrt(discriminant)
            if np.real(d) < 0.0:
                d = -d
            g = (b - d) / (b + d)
            exp_minus_dt = np.exp(-d * maturities)
            log_ratio = np.log1p(-g * exp_minus_dt) - np.log1p(-g)
            c_term = (kappa * theta / sigma**2) * ((b - d) * maturities - 2.0 * log_ratio)
            d_term = ((b - d) / sigma**2) * ((-np.expm1(-d * maturities)) / (1.0 - g * exp_minus_dt))
            total = total + c_term + d_term * v0
        return np.exp(total)

    phi_u = characteristic(u)
    phi_shifted = characteristic(u - 1j)
    phi_minus_i = characteristic_minus_i()

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


def price_surface_per_carry(
    parameters: Sequence[float],
    strikes: np.ndarray,
    maturities_years: np.ndarray,
    option_types: Sequence[str],
    rates: np.ndarray,
    dividends: np.ndarray,
    *,
    spot: float,
    node_count: int = 64,
) -> np.ndarray:
    """Price a surface whose quotes carry per-rank rate/carry conditioning.

    Groups quotes by their constant-carry piece (exactly how the committed G2
    diagnostics priced per-maturity carry) and evaluates one broadcast per
    distinct (rate, dividend) pair.
    """
    rates = np.asarray(rates, dtype=np.float64)
    dividends = np.asarray(dividends, dtype=np.float64)
    strikes = np.asarray(strikes, dtype=np.float64)
    maturities = np.asarray(maturities_years, dtype=np.float64)
    combined = np.zeros(strikes.shape[0], dtype=np.float64)
    for rate, dividend in sorted(set(zip(rates.tolist(), dividends.tolist()))):
        mask = (rates == rate) & (dividends == dividend)
        combined[mask] = price_surface_fast(
            parameters,
            strikes[mask],
            maturities[mask],
            np.asarray(option_types, dtype=str)[mask],
            spot=spot,
            risk_free_rate=rate,
            dividend_yield=dividend,
            node_count=node_count,
        )
    return combined


def normalized_observables(
    parameters: Sequence[float],
    strikes: np.ndarray,
    maturities_years: np.ndarray,
    option_types: Sequence[str],
    rates: np.ndarray,
    dividends: np.ndarray,
    *,
    spot: float,
    node_count: int = 64,
) -> np.ndarray:
    """Spot-normalized prices (price / spot) in the G2 convention."""
    prices = price_surface_per_carry(
        parameters,
        strikes,
        maturities_years,
        option_types,
        rates,
        dividends,
        spot=spot,
        node_count=node_count,
    )
    return prices / spot
