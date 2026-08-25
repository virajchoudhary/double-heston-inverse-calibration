"""Frozen G8 R2 construction from validated official-source frames/fixtures."""

from __future__ import annotations

from datetime import date
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from ..r2_representation.contract import CANONICAL_SLOT_KEYS
from ..r2_representation.surface import R2Surface, surface_from_vectors
from .acquisition import RbiRateRecord
from .contracts import (
    G8ReadinessError,
    MONEYNESS_LIMIT,
    MAX_TARGET_DISTANCE,
    MONEYNESS_TOLERANCE,
    TARGET_MONEYNESS,
    canonical_slot_roles,
    discount_factor,
    futures_implied_carry,
    implied_volatility,
    validate_g8_valuation_date,
)

SYNTHETIC_FIXTURE_SOURCE = "SYNTHETIC_G8_PIPELINE_FIXTURE_R2"
REAL_G8_SOURCE = "REAL_G8_OFFICIAL_NSE_R2"


class G8SurfaceConstructionError(G8ReadinessError):
    pass


def _positive_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    return values.where(np.isfinite(values) & values.gt(0.0))


def _activity_mask(frame: pd.DataFrame) -> pd.Series:
    mask = None
    for column in ("ClsPric", "TtlTradgVol", "OpnIntrst", "TtlNbOfTxsExctd"):
        positive = _positive_numeric(frame, column).notna()
        mask = positive if mask is None else mask & positive
    assert mask is not None
    return mask.fillna(False)


def _latest_rate(rates: Mapping[date, RbiRateRecord], valuation_date: date) -> RbiRateRecord:
    parsed: dict[date, RbiRateRecord] = {}
    for observed, record in rates.items():
        value = observed if type(observed) is date else date.fromisoformat(str(observed))
        if value > valuation_date:
            raise G8SurfaceConstructionError(
                f"future RBI observation rejected: {value.isoformat()} > {valuation_date.isoformat()}"
            )
        parsed[value] = record
    if not parsed:
        raise G8SurfaceConstructionError("MISSING_RBI_RATE_HARD_SURFACE_FAILURE")
    return parsed[max(parsed)]


def build_g8_r2_surface(
    symbol: str,
    valuation_date: date | str,
    cm: pd.DataFrame,
    fo: pd.DataFrame,
    rates_by_observation_date: Mapping[date, RbiRateRecord],
    *,
    data_classification: str = "SYNTHETIC_G8_PIPELINE_FIXTURE",
) -> tuple[R2Surface, dict[str, Any]]:
    """Build one masked surface using the frozen central-five Hungarian rules."""
    value = validate_g8_valuation_date(valuation_date)
    normalized_symbol = symbol.upper()
    if data_classification != "SYNTHETIC_G8_PIPELINE_FIXTURE":
        raise G8SurfaceConstructionError(
            "real G8 construction requires the separately sealed selected-data path"
        )
    required = {"TckrSymb", "FinInstrmTp", "FinInstrmId"}
    if not required.issubset(cm.columns) or not required.issubset(fo.columns):
        raise G8SurfaceConstructionError("market frame lacks required UDiFF columns")

    equity = cm.loc[
        cm["SctySrs"].eq("EQ") & cm["TckrSymb"].eq(normalized_symbol)
    ]
    equity_prices = _positive_numeric(equity, "ClsPric").dropna()
    if len(equity) != 1 or len(equity_prices) != 1:
        raise G8SurfaceConstructionError("SPOT_MUST_BE_EXACTLY_ONE_POSITIVE_CM_EQ_CLOSE")
    spot = float(equity_prices.iloc[0])
    underlying_rows = fo.loc[fo["TckrSymb"].eq(normalized_symbol)]
    underlying_prices = pd.to_numeric(underlying_rows["UndrlygPric"], errors="coerce").dropna().unique()
    if len(underlying_prices) != 1 or float(underlying_prices[0]) != spot:
        raise G8SurfaceConstructionError("CM_EQ_CLOSE_AND_FO_UNDERLYGPRIC_EXACT_EQUALITY_FAILED")

    options_all = underlying_rows.loc[underlying_rows["FinInstrmTp"].eq("STO")].copy()
    futures = underlying_rows.loc[underlying_rows["FinInstrmTp"].eq("STF")].copy()
    if options_all.empty or futures.empty:
        raise G8SurfaceConstructionError("MISSING_STOCK_OPTIONS_OR_FUTURES")
    options_all["actual_expiry"] = pd.to_datetime(
        options_all["FininstrmActlXpryDt"], format="%d-%b-%Y", errors="coerce"
    ).dt.date
    futures["actual_expiry"] = pd.to_datetime(
        futures["FininstrmActlXpryDt"], format="%d-%b-%Y", errors="coerce"
    ).dt.date
    if options_all["actual_expiry"].isna().any() or futures["actual_expiry"].isna().any():
        raise G8SurfaceConstructionError("INVALID_ACTUAL_EXPIRY")
    futures["active"] = _activity_mask(futures).fillna(False)
    expiry_candidates = []
    for expiry in sorted(options_all["actual_expiry"].unique()):
        if expiry <= value:
            continue
        matching_future = futures.loc[
            futures["actual_expiry"].eq(expiry) & futures["active"]
        ]
        if len(matching_future) == 1:
            expiry_candidates.append(expiry)
        if len(expiry_candidates) == 2:
            break
    if len(expiry_candidates) < 2:
        raise G8SurfaceConstructionError("FEWER_THAN_TWO_ELIGIBLE_EXPIRIES_WITH_ACTIVE_MATCHED_FUTURE")

    rate_record = _latest_rate(rates_by_observation_date, value)
    simple_yield = rate_record.yield_percent / 100.0
    selected_rows: list[dict[str, Any]] = []
    rejection_log: list[dict[str, Any]] = []
    rank_rates: list[float] = []
    rank_carries: list[float] = []
    for rank, expiry in enumerate(expiry_candidates, start=1):
        maturity = (expiry - value).days / 365.0
        future_rows = futures.loc[futures["actual_expiry"].eq(expiry) & futures["active"]]
        forward = float(_positive_numeric(future_rows, "ClsPric").iloc[0])
        continuous, carry = futures_implied_carry(spot, forward, maturity, simple_yield)
        discount = discount_factor(simple_yield, maturity)
        rank_rates.append(continuous)
        rank_carries.append(carry)
        option_rows = options_all.loc[
            options_all["actual_expiry"].eq(expiry) & _activity_mask(options_all.loc[options_all["actual_expiry"].eq(expiry)])
        ].copy()
        strikes = _positive_numeric(option_rows, "StrkPric")
        option_rows = option_rows.loc[strikes.notna()].copy()
        option_rows["strike"] = strikes.dropna()
        log_moneyness = np.log(option_rows["strike"] / spot)
        option_rows = option_rows.loc[
            log_moneyness.abs().le(MONEYNESS_LIMIT)
        ]
        option_rows["log_moneyness"] = log_moneyness.loc[option_rows.index]
        ivs: list[float | None] = []
        for row in option_rows.itertuples():
            option_type = {"CE": "call", "PE": "put"}.get(str(row.OptnTp).upper())
            price = float(row.ClsPric)
            if option_type is None:
                ivs.append(None)
                continue
            try:
                iv = implied_volatility(price, forward, float(row.strike), maturity, discount, option_type)
                ivs.append(iv)
            except Exception as exc:
                ivs.append(None)
                rejection_log.append(
                    {
                        "FinInstrmId": str(row.FinInstrmId),
                        "expiry": expiry.isoformat(),
                        "reason": f"BLACK_IV_{type(exc).__name__}",
                    }
                )
        option_rows["market_iv"] = ivs
        eligible = option_rows.loc[pd.Series(ivs, index=option_rows.index).notna()].copy()
        eligible["option_type"] = eligible["OptnTp"].map({"CE": "call", "PE": "put"})
        eligible = eligible.sort_values(["strike", "FinInstrmId"], kind="mergesort")
        for option_type in ("call", "put"):
            group = eligible.loc[eligible["option_type"].eq(option_type)]
            cost = np.full((len(group), len(TARGET_MONEYNESS)), np.inf)
            for row_index, candidate in enumerate(group.itertuples()):
                for target_index, target in enumerate(TARGET_MONEYNESS):
                    distance = abs(float(candidate.log_moneyness) - target)
                    if distance <= MAX_TARGET_DISTANCE + MONEYNESS_TOLERANCE:
                        cost[row_index, target_index] = distance
            if group.empty:
                continue
            assigned_rows, assigned_columns = linear_sum_assignment(cost)
            for row_index, column_index in zip(assigned_rows, assigned_columns, strict=True):
                if not np.isfinite(cost[row_index, column_index]):
                    continue
                candidate = group.loc[group.index[int(row_index)]]
                selected_rows.append(
                    {
                        "rank": rank,
                        "target": TARGET_MONEYNESS[int(column_index)],
                        "option_type": option_type,
                        "price": float(candidate["ClsPric"]),
                        "strike": float(candidate["strike"]),
                        "actual_log_moneyness": float(candidate["log_moneyness"]),
                        "iv": float(candidate["market_iv"]),
                        "FinInstrmId": str(candidate["FinInstrmId"]),
                        "expiry": expiry.isoformat(),
                        "forward": forward,
                        "discount": discount,
                    }
                )

    selection = {(int(row["rank"]), float(row["target"]), str(row["option_type"])): row for row in selected_rows}
    prices: list[float] = []
    masks: list[bool] = []
    maturities: list[float] = []
    rates: list[float] = []
    carries: list[float] = []
    failure_reasons: list[str] = []
    for key in CANONICAL_SLOT_KEYS:
        row = selection.get((key.expiry_rank, key.target_log_moneyness, key.option_type))
        maturity_rank = key.expiry_rank - 1
        maturities.append((expiry_candidates[maturity_rank] - value).days / 365.0)
        rates.append(rank_rates[maturity_rank])
        carries.append(rank_carries[maturity_rank])
        if row is None:
            prices.append(0.0)
            masks.append(False)
            failure_reasons.append("NO_ELIGIBLE_HUNGARIAN_ASSIGNMENT")
        else:
            prices.append(float(row["price"]) / spot)
            masks.append(True)
            failure_reasons.append("")

    roles = canonical_slot_roles()
    calibration_support = int((np.asarray(masks) & roles["pricing_family_calibration"]).sum())
    holdout_support = int((np.asarray(masks) & roles["pricing_family_holdout"]).sum())
    rank_counts = [
        sum(mask for key, mask in zip(CANONICAL_SLOT_KEYS, masks, strict=True) if key.expiry_rank == rank)
        for rank in (1, 2)
    ]
    call_count = sum(mask for key, mask in zip(CANONICAL_SLOT_KEYS, masks, strict=True) if key.option_type == "call")
    put_count = len(masks) - call_count
    usable = sum(masks)
    if usable < 12 or calibration_support < 6 or holdout_support < 3 or min(rank_counts) < 1 or call_count < 1 or put_count < 1:
        raise G8SurfaceConstructionError(
            "INSUFFICIENT_FROZEN_R2_SUPPORT:"
            f"total={usable},calibration={calibration_support},holdout={holdout_support},"
            f"ranks={rank_counts},calls={call_count},puts={put_count}"
        )

    metadata = {
        "synthetic": data_classification.startswith("SYNTHETIC"),
        "data_classification": data_classification,
        "symbol": normalized_symbol,
        "valuation_date": value.isoformat(),
        "rate_record": {
            "observation_date": rate_record.observation_date,
            "release_identifier": rate_record.release_identifier,
            "normalized_extract_sha256": rate_record.normalized_extract_sha256,
            "simple_yield": simple_yield,
        },
        "listed_expiry_ranks_used": [expiry.isoformat() for expiry in expiry_candidates],
        "selected_contracts": selected_rows,
        "rejected_nonselected_examples": rejection_log[:100],
        "support": {
            "total": usable,
            "calibration": calibration_support,
            "holdout": holdout_support,
            "rank": rank_counts,
            "calls": call_count,
            "puts": put_count,
        },
        "imputation_or_interpolation": "NONE",
    }
    surface = surface_from_vectors(
        prices,
        masks,
        maturities,
        rates,
        carries,
        spot=spot,
        surface_id=f"G8_{normalized_symbol}_{value.isoformat()}_R2",
        source=SYNTHETIC_FIXTURE_SOURCE,
        metadata=metadata,
    )
    report = {
        "surface_id": surface.surface_id,
        "usable_slots": usable,
        "mask_hash_input": [bool(item) for item in masks],
        "rejection_count": len(rejection_log),
        "rejection_reasons": rejection_log,
        "construction_status": "CONSTRUCTED_SYNTHETIC_FIXTURE_ONLY",
    }
    return surface, report
