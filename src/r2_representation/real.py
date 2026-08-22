"""Canonical constructor/adaptor for real-market R2 surfaces (NTPC).

Reuses the merged, reviewed official-NSE quote-selection contract exactly as
executed for the sealed G2 market-support audit
(``src/g2_r2r3/market.audit_date``): raw UDiFF archives, Hungarian
assignment with the 0.05 target gate, activity/moneyness/Black-IV
eligibility on the matched-futures forward, and hash-sealed rate
observations.  No new market-data logic is introduced here — this module only
re-expresses that contract's output as the canonical :class:`R2Surface`.

Contract guarantees enforced here:

- the same 20 nominal positions as synthetic R2 (canonical slot order);
- unavailable observations keep their canonical positions with
  ``mask=False`` and price exactly ``0.0`` — never imputed, interpolated,
  extrapolated, or filled from neighboring observations;
- actual listed expiry/DTE, per-rank rate and futures-implied carry, and the
  market spot are retained;
- raw-quote provenance (selected strike, realized log-moneyness, raw close
  price, failure reason) is retained where practical.

The five NTPC development dates are the only dates wired into the sealed
raw-archive mapping; they remain DEVELOPMENT / DIAGNOSTIC and permanently
excluded from final G8.  Constructing G8-date surfaces is a later,
separately controlled milestone and must not reuse this development mapping
silently.
"""

from __future__ import annotations

from typing import Any

from .contract import (
    CANONICAL_SLOT_KEYS,
    CENTRAL_FIVE_LOG_MONEYNESS,
    RepresentationContractError,
    SlotKey,
)
from .surface import R2Surface, SOURCE_REAL_NSE_DEVELOPMENT


class RealSurfaceNotConstructibleError(RepresentationContractError):
    """The date cannot produce a real R2 surface under the official contract."""


def build_real_surface(date_id: str, *, audit_report: dict[str, Any] | None = None) -> R2Surface:
    """Build one masked real R2 surface for an NTPC development date.

    ``audit_report`` may be supplied to avoid re-running the raw-file audit;
    it must be the report dict returned by ``src.g2_r2r3.market.audit_date``.
    """
    from ..g2_r2r3 import frozen, market  # sealed official-NSE contract, reused as-is

    if date_id not in frozen.MARKET_DATES:
        raise RealSurfaceNotConstructibleError(
            f"date {date_id} is not one of the five frozen NTPC development dates "
            f"{list(frozen.MARKET_DATES)}; final G8 dates are selected in a later, "
            "separately controlled milestone"
        )
    report = audit_report if audit_report is not None else market.audit_date(date_id)
    if report.get("date_id") != date_id:
        raise RealSurfaceNotConstructibleError(
            f"audit_report is for date {report.get('date_id')!r}, not the "
            f"requested date {date_id!r}; a supplied report must be that "
            "date's own audit output"
        )
    if not report.get("constructible", False):
        raise RealSurfaceNotConstructibleError(
            f"date {date_id} is not R2-constructible under the official-NSE "
            f"contract: {report.get('hard_failure', 'no usable R2 support')}"
        )

    details = sorted(report["expiry_details"], key=lambda item: item["rank"])
    if len(details) < 2:
        raise RealSurfaceNotConstructibleError(
            f"date {date_id} has {len(details)} eligible expiry ranks; a real R2 "
            "surface requires the first two eligible listed ranks (rank absence is "
            "not slot masking)"
        )
    selected = details[:2]
    listed_ranks = [int(item["rank"]) for item in selected]
    listed_rank_of_representation_rank = {
        representation_rank: listed_rank
        for representation_rank, listed_rank in enumerate(listed_ranks, start=1)
    }

    spot = float(report["spot"])
    maturities = tuple(float(item["dte"]) / 365.0 for item in selected)
    rates = tuple(float(item["rate"]) for item in selected)
    carries = tuple(float(item["carry"]) for item in selected)

    slot_rows = report["slot_table"]
    row_by_identity = {
        (int(row.expiry_rank), float(row.target_log_moneyness), str(row.option_type)): row
        for row in slot_rows.itertuples()
    }

    prices: list[float] = []
    mask: list[bool] = []
    actual_strikes: list[float | None] = []
    actual_log_moneyness: list[float | None] = []
    observed_raw_prices: list[float | None] = []
    failure_reasons: list[str] = []
    for key in CANONICAL_SLOT_KEYS:
        listed_rank = listed_rank_of_representation_rank[key.expiry_rank]
        row = row_by_identity.get((listed_rank, key.target_log_moneyness, key.option_type))
        if row is None:
            raise RealSurfaceNotConstructibleError(
                f"audit slot table lacks the canonical slot "
                f"(listed rank {listed_rank}, k={key.target_log_moneyness:+.2f}, "
                f"{key.option_type}) for date {date_id}"
            )
        if bool(row.usable):
            raw_price = float(row.observed_price)
            prices.append(raw_price / spot)
            mask.append(True)
            actual_strikes.append(float(row.strike))
            actual_log_moneyness.append(float(row.log_moneyness_actual))
            observed_raw_prices.append(raw_price)
            failure_reasons.append("")
        else:
            prices.append(0.0)
            mask.append(False)
            actual_strikes.append(None)
            actual_log_moneyness.append(None)
            observed_raw_prices.append(None)
            failure_reasons.append(str(row.failure_reason))

    metadata: dict[str, Any] = {
        "synthetic": False,
        "ticker": market.TICKER,
        "date_id": date_id,
        "valuation_date": date_id,
        "development_date_excluded_from_g8": True,
        "rate_observation_date": report.get("rate_observation_date"),
        "rate_simple_yield": report.get("rate_simple_yield"),
        "rate_carry_forward": report.get("rate_carry_forward"),
        "listed_expiry_ranks_used": listed_ranks,
        "expiry_dates": [str(item["expiry_date"]) for item in selected],
        "dte": [int(item["dte"]) for item in selected],
        "quote_selection_contract": "official_nse_udiff_hungarian_0.05_gate_as_sealed_g2_audit",
        "imputation": "NONE_MASKED_EXPLICITLY",
        "usable_slot_count": int(sum(mask)),
        "provenance": {
            "actual_strikes": actual_strikes,
            "actual_log_moneyness": actual_log_moneyness,
            "observed_raw_prices": observed_raw_prices,
            "failure_reasons": failure_reasons,
        },
    }
    return R2Surface(
        prices=tuple(prices),
        mask=tuple(mask),
        maturities=tuple(
            maturities[key.expiry_rank - 1] for key in CANONICAL_SLOT_KEYS
        ),
        rates=tuple(rates[key.expiry_rank - 1] for key in CANONICAL_SLOT_KEYS),
        carries=tuple(carries[key.expiry_rank - 1] for key in CANONICAL_SLOT_KEYS),
        spot=spot,
        surface_id=f"{market.TICKER}_{date_id}_R2",
        source=SOURCE_REAL_NSE_DEVELOPMENT,
        metadata=metadata,
    )


def canonical_key_for_slots() -> tuple[SlotKey, ...]:
    """Expose the canonical keys for callers verifying real/synthetic parity."""
    return CANONICAL_SLOT_KEYS


__all__ = [
    "CENTRAL_FIVE_LOG_MONEYNESS",
    "RealSurfaceNotConstructibleError",
    "build_real_surface",
    "canonical_key_for_slots",
]
