"""Operationally executable readiness layer for the frozen G8 protocol."""

from .contracts import (
    DATE_FLOOR,
    SCAN_END,
    G8ReadinessError,
    canonical_slot_roles,
    continuous_rate,
    discount_factor,
    forward_black_price,
    futures_implied_carry,
    implied_volatility,
    validate_g8_valuation_date,
)
from .acquisition import (
    CurrentDateAcquisitionLocked,
    G8AcquisitionLocked,
    NSEArchiveRecord,
    RbiRateRecord,
    intake_official_nse,
    normalize_rbi_auction,
    verify_acquisition_gate,
)

__all__ = [
    "CurrentDateAcquisitionLocked",
    "DATE_FLOOR",
    "G8AcquisitionLocked",
    "G8ReadinessError",
    "NSEArchiveRecord",
    "RbiRateRecord",
    "SCAN_END",
    "canonical_slot_roles",
    "continuous_rate",
    "discount_factor",
    "forward_black_price",
    "futures_implied_carry",
    "implied_volatility",
    "intake_official_nse",
    "normalize_rbi_auction",
    "validate_g8_valuation_date",
    "verify_acquisition_gate",
]
