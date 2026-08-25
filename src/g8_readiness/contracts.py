"""Frozen G8 boundaries shared by all readiness components."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

from ..r2_representation.contract import CANONICAL_SLOT_KEYS

DATE_FLOOR = date(2026, 9, 30)
SCAN_START = DATE_FLOOR
SCAN_END = date(2026, 12, 31)
PRIMARY_SYMBOLS = ("NTPC", "CIPLA", "INFY", "HDFCBANK")
BACKUP_SYMBOLS_BY_PRIMARY = {
    "NTPC": "POWERGRID",
    "CIPLA": "SUNPHARMA",
    "INFY": "TCS",
    "HDFCBANK": "ICICIBANK",
}
TARGET_MONEYNESS = (-0.10, -0.05, 0.00, 0.05, 0.10)
CALIBRATION_MONEYNESS = (-0.05, 0.00, 0.05)
HOLDOUT_MONEYNESS = (-0.10, 0.10)
MONEYNESS_TOLERANCE = 1e-12
MAX_TARGET_DISTANCE = 0.05 + MONEYNESS_TOLERANCE
MONEYNESS_LIMIT = 0.10 + MONEYNESS_TOLERANCE


class G8ReadinessError(ValueError):
    """Base error for violations of the frozen G8 readiness contract."""


def validate_g8_valuation_date(value: date | str) -> date:
    """Reject malformed, pre-floor, and post-window valuation dates."""
    if type(value) is date:
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = date.fromisoformat(value)
        except ValueError as exc:
            raise G8ReadinessError(f"invalid ISO G8 valuation date: {value!r}") from exc
    else:
        raise G8ReadinessError("valuation date must be datetime.date or ISO text")
    if parsed < DATE_FLOOR:
        raise G8ReadinessError(
            f"G8 valuation date {parsed.isoformat()} precedes frozen floor {DATE_FLOOR.isoformat()}"
        )
    if parsed > SCAN_END:
        raise G8ReadinessError(
            f"G8 scan end {SCAN_END.isoformat()} cannot be extended to {parsed.isoformat()}"
        )
    return parsed


def discount_factor(simple_yield: float, maturity: float) -> float:
    if not math.isfinite(simple_yield) or simple_yield < 0.0:
        raise G8ReadinessError("simple yield must be finite and non-negative")
    if not math.isfinite(maturity) or maturity <= 0.0:
        raise G8ReadinessError("maturity must be finite and positive")
    return 1.0 / (1.0 + simple_yield * maturity)


def continuous_rate(simple_yield: float, maturity: float) -> float:
    return -math.log(discount_factor(simple_yield, maturity)) / maturity


def futures_implied_carry(
    spot: float,
    forward: float,
    maturity: float,
    simple_yield: float,
) -> tuple[float, float]:
    """Return continuous rate then carry from the matched actual futures price."""
    if not all(math.isfinite(value) and value > 0.0 for value in (spot, forward, maturity)):
        raise G8ReadinessError("spot, forward, and maturity must be positive")
    rate = continuous_rate(simple_yield, maturity)
    carry = rate - math.log(forward / spot) / maturity
    return rate, carry


def forward_black_price(
    forward: float,
    strike: float,
    maturity: float,
    discount: float,
    volatility: float,
    option_type: str,
) -> float:
    if not all(math.isfinite(v) and v > 0.0 for v in (forward, strike, maturity, discount, volatility)):
        raise G8ReadinessError("forward Black inputs must be positive and finite")
    root_maturity = math.sqrt(maturity)
    d1 = (
        math.log(forward / strike) + 0.5 * volatility * volatility * maturity
    ) / (volatility * root_maturity)
    d2 = d1 - volatility * root_maturity
    normalized = ndtr(d1), ndtr(d2)
    if option_type == "call":
        return float(discount * (forward * normalized[0] - strike * normalized[1]))
    if option_type == "put":
        return float(discount * (strike * (1.0 - normalized[1]) - forward * (1.0 - normalized[0])))
    raise G8ReadinessError("option_type must be call or put")


def no_arbitrage_bounds(
    forward: float,
    strike: float,
    discount: float,
    option_type: str,
) -> tuple[float, float]:
    if option_type == "call":
        return discount * max(forward - strike, 0.0), discount * forward
    if option_type == "put":
        return discount * max(strike - forward, 0.0), discount * strike
    raise G8ReadinessError("option_type must be call or put")


def implied_volatility(
    price: float,
    forward: float,
    strike: float,
    maturity: float,
    discount: float,
    option_type: str,
) -> float:
    lower, upper = no_arbitrage_bounds(forward, strike, discount, option_type)
    tolerance = 1e-12 * max(1.0, upper)
    if not math.isfinite(price) or price < lower + tolerance or price > upper - tolerance:
        raise G8ReadinessError("price does not admit a strictly bracketed forward Black IV")
    intrinsic = forward_black_price(forward, strike, maturity, discount, 1e-7, option_type)
    if abs(price - intrinsic) <= tolerance:
        return 1e-7

    def objective(sigma: float) -> float:
        return forward_black_price(forward, strike, maturity, discount, sigma, option_type) - price

    low, high = 1e-7, 5.0
    if objective(low) * objective(high) > 0.0:
        raise G8ReadinessError("forward Black IV root is not bracketed on [1e-7, 5]")
    return float(brentq(objective, low, high, xtol=1e-13, rtol=1e-13, maxiter=200))


def canonical_slot_roles() -> dict[str, np.ndarray]:
    """Mask-gated nominal roles aligned exactly with canonical R2 order."""
    calibration = np.array(
        [key.target_log_moneyness in CALIBRATION_MONEYNESS for key in CANONICAL_SLOT_KEYS],
        dtype=bool,
    )
    holdout = np.array(
        [key.target_log_moneyness in HOLDOUT_MONEYNESS for key in CANONICAL_SLOT_KEYS],
        dtype=bool,
    )
    inverse = np.ones(len(CANONICAL_SLOT_KEYS), dtype=bool)
    if calibration.sum() != 12 or holdout.sum() != 8 or int((calibration & holdout).sum()) != 0:
        raise AssertionError("canonical pricing-role partition drifted")
    return {
        "pricing_family_calibration": calibration,
        "pricing_family_holdout": holdout,
        "inverse_method_full_r2": inverse,
    }
