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


def development_contract_key(
    instrument_kind: str,
    valuation_date: str,
    expiry_date: str,
    strike: str,
    fin_instrm_id: str,
) -> str:
    if instrument_kind not in {"FUTURE", "CALL", "PUT"}:
        raise G8SurfaceConstructionError("INVALID_DEVELOPMENT_CONTRACT_KIND")
    date.fromisoformat(valuation_date)
    date.fromisoformat(expiry_date)
    float(strike)
    import math
    if not math.isfinite(float(strike)) or repr(float(strike)) != strike:
        raise G8SurfaceConstructionError("NONCANONICAL_DEVELOPMENT_CONTRACT_STRIKE")
    if not fin_instrm_id:
        raise G8SurfaceConstructionError("INVALID_DEVELOPMENT_CONTRACT_ID")
    return f"{instrument_kind}|{valuation_date}|{expiry_date}|{strike}|{fin_instrm_id}"


def validate_development_contract_keys(keys: frozenset[str] | set[str]) -> frozenset[str]:
    for item in keys:
        parts = item.split("|")
        if len(parts) != 5:
            raise G8SurfaceConstructionError("INVALID_DEVELOPMENT_CONTRACT_REGISTRY_SCHEMA")
        try:
            development_contract_key(*parts)
        except (ValueError, TypeError) as exc:
            raise G8SurfaceConstructionError("INVALID_DEVELOPMENT_CONTRACT_REGISTRY_SCHEMA") from exc
    return frozenset(keys)


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


def _latest_rate(
    rates: Mapping[date, RbiRateRecord],
    valuation_date: date,
    official_release_calendar: Mapping[str, date],
) -> RbiRateRecord:
    parsed: dict[date, RbiRateRecord] = {}
    for observed, record in rates.items():
        value = observed if type(observed) is date else date.fromisoformat(str(observed))
        if value > valuation_date:
            raise G8SurfaceConstructionError(
                f"future RBI observation rejected: {value.isoformat()} > {valuation_date.isoformat()}"
            )
        if record.observation_date != value.isoformat():
            raise G8SurfaceConstructionError("RBI_RECORD_DATE_NOT_BOUND_TO_LOOKUP_KEY")
        if official_release_calendar.get(record.release_identifier) != value:
            raise G8SurfaceConstructionError("RBI_CALENDAR_DATE_NOT_BOUND_TO_RECORD")
        parsed[value] = record
    if not parsed:
        raise G8SurfaceConstructionError("MISSING_RBI_RATE_HARD_SURFACE_FAILURE")
    record_ids = {record.release_identifier for record in rates.values()}
    calendar_ids = set(official_release_calendar)
    if record_ids != calendar_ids:
        raise G8SurfaceConstructionError(
            "RBI_RELEASE_CALENDAR_MISMATCH:"
            f"missing_records={sorted(calendar_ids - record_ids)},"
            f"unregistered_records={sorted(record_ids - calendar_ids)}"
        )
    return parsed[max(parsed)]


def build_g8_r2_surface(
    symbol: str,
    valuation_date: date | str,
    cm: pd.DataFrame,
    fo: pd.DataFrame,
    rates_by_observation_date: Mapping[date, RbiRateRecord],
    *,
    data_classification: str = "SYNTHETIC_G8_PIPELINE_FIXTURE",
    official_release_calendar: Mapping[str, date] | None = None,
    development_contract_keys: frozenset[str] | set[str] | None = None,
    authorize_real_surface_construction: bool = False,
) -> tuple[R2Surface, dict[str, Any]]:
    """Build one masked surface using the frozen central-five Hungarian rules."""
    value = validate_g8_valuation_date(valuation_date)
    normalized_symbol = symbol.upper()
    is_real = data_classification == "REAL_G8_SELECTED_DATA"
    if data_classification not in {"SYNTHETIC_G8_PIPELINE_FIXTURE", "REAL_G8_SELECTED_DATA"} or (
        is_real and not authorize_real_surface_construction
    ):
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

    if official_release_calendar is None:
        raise G8SurfaceConstructionError("RBI_OFFICIAL_RELEASE_CALENDAR_REQUIRED")
    rate_record = _latest_rate(rates_by_observation_date, value, official_release_calendar)
    development_registry = validate_development_contract_keys(set(development_contract_keys or ()))
    development_overlap_checked = development_contract_keys is not None
    if data_classification != "SYNTHETIC_G8_PIPELINE_FIXTURE" and not development_registry:
        raise G8SurfaceConstructionError("REAL_G8_REQUIRES_NONEMPTY_DEVELOPMENT_CONTRACT_REGISTRY")
    simple_yield = rate_record.yield_percent / 100.0
    selected_rows: list[dict[str, Any]] = []
    rejection_by_index: dict[Any, dict[str, Any]] = {}
    selected_indices: set[Any] = set()
    rank_rates: list[float] = []
    rank_carries: list[float] = []
    for rank, expiry in enumerate(expiry_candidates, start=1):
        selected_expiry = expiry in expiry_candidates
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
        for raw_index, row in options_all.loc[options_all["actual_expiry"].ne(expiry)].iterrows():
            rejection_by_index.setdefault(raw_index, {
                "FinInstrmId": str(row["FinInstrmId"]),
                "expiry": str(row["actual_expiry"]),
                "reason": "EXPIRY_NOT_SELECTED",
            })
        inactive_mask = ~_activity_mask(options_all.loc[options_all["actual_expiry"].eq(expiry)])
        for raw_index in options_all.loc[options_all["actual_expiry"].eq(expiry)][inactive_mask].index:
            row = options_all.loc[raw_index]
            rejection_by_index[raw_index] = {
                "FinInstrmId": str(row["FinInstrmId"]),
                "expiry": expiry.isoformat(),
                "reason": "INACTIVE_OPTION",
            }
        strikes = _positive_numeric(option_rows, "StrkPric")
        for raw_index in option_rows.index[strikes.isna()]:
            row = options_all.loc[raw_index]
            rejection_by_index[raw_index] = {
                "FinInstrmId": str(row["FinInstrmId"]),
                "expiry": expiry.isoformat(),
                "reason": "INVALID_STRIKE",
            }
        option_rows = option_rows.loc[strikes.notna()].copy()
        option_rows["strike"] = strikes.dropna()
        log_moneyness = np.log(option_rows["strike"] / spot)
        outside_moneyness = ~log_moneyness.abs().le(MONEYNESS_LIMIT)
        for raw_index in option_rows.index[outside_moneyness]:
            row = options_all.loc[raw_index]
            rejection_by_index[raw_index] = {
                "FinInstrmId": str(row["FinInstrmId"]),
                "expiry": expiry.isoformat(),
                "reason": "OUTSIDE_MONEYNESS_SCREEN",
            }
        option_rows = option_rows.loc[
            log_moneyness.abs().le(MONEYNESS_LIMIT)
        ]
        option_rows["log_moneyness"] = log_moneyness.loc[option_rows.index]
        ivs: list[float | None] = []
        for row in option_rows.itertuples():
            raw_index = row.Index
            option_type = {"CE": "call", "PE": "put"}.get(str(row.OptnTp).upper())
            price = float(row.ClsPric)
            if option_type is None:
                if True:
                    ivs.append(None)
                    rejection_by_index[raw_index] = {
                        "FinInstrmId": str(row.FinInstrmId),
                        "expiry": expiry.isoformat(),
                        "reason": "INVALID_OPTION_TYPE",
                    }
                    continue
                ivs.append(None)
                continue
            try:
                iv = implied_volatility(price, forward, float(row.strike), maturity, discount, option_type)
                ivs.append(iv)
            except Exception as exc:
                ivs.append(None)
                rejection_by_index[raw_index] = {
                    "FinInstrmId": str(row.FinInstrmId),
                    "expiry": expiry.isoformat(),
                    "reason": f"BLACK_IV_{type(exc).__name__}",
                }
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
                    if distance <= MAX_TARGET_DISTANCE:
                        cost[row_index, target_index] = distance
            if group.empty:
                continue
            assigned_rows, assigned_columns = linear_sum_assignment(cost)
            assigned_candidate_positions = set()
            for row_index, column_index in zip(assigned_rows, assigned_columns, strict=True):
                if not np.isfinite(cost[row_index, column_index]):
                    continue
                assigned_candidate_positions.add(int(row_index))
                candidate = group.loc[group.index[int(row_index)]]
                selected_indices.add(candidate.name)
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
            group_candidates = list(group.itertuples())
            for position, candidate in enumerate(group_candidates):
                raw_index = candidate.Index
                if position not in assigned_candidate_positions:
                    distance_reason = (
                        "TARGET_DISTANCE_REJECTED"
                        if all(not np.isfinite(cost[position, target_index]) for target_index in range(len(TARGET_MONEYNESS)))
                        else "NOT_ASSIGNED_HUNGARIAN"
                    )
                    rejection_by_index[raw_index] = {
                        "FinInstrmId": str(candidate.FinInstrmId),
                        "expiry": expiry.isoformat(),
                        "reason": distance_reason,
                    }

    selection = {(int(row["rank"]), float(row["target"]), str(row["option_type"])): row for row in selected_rows}
    rejection_log = [
        item
        for index, item in sorted(rejection_by_index.items(), key=lambda pair: str(pair[0]))
        if index not in selected_indices
    ]
    selected_contract_keys = {
        development_contract_key(
            row["option_type"].upper(),
            value.isoformat(),
            row["expiry"],
            str(row["strike"]),
            row["FinInstrmId"],
        )
        for row in selected_rows
    }
    development_overlap_found = bool(development_registry.intersection(selected_contract_keys))
    if development_overlap_checked and development_overlap_found:
        raise G8SurfaceConstructionError("DEVELOPMENT_CONTRACT_KEY_OVERLAP_FAIL_CLOSED")
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
        "rejected_nonselected_rows": rejection_log,
        "development_contract_overlap_checked": development_overlap_checked,
        "development_contract_overlap_found": development_overlap_found if development_overlap_checked else None,
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
        source=REAL_G8_SOURCE if data_classification == "REAL_G8_SELECTED_DATA" else SYNTHETIC_FIXTURE_SOURCE,
        metadata=metadata,
    )
    report = {
        "surface_id": surface.surface_id,
        "usable_slots": usable,
        "mask_hash_input": [bool(item) for item in masks],
        "rejection_count": len(rejection_log),
        "rejection_reasons": rejection_log,
        "construction_status": (
            "CONSTRUCTED_SYNTHETIC_FIXTURE_ONLY"
            if data_classification == "SYNTHETIC_G8_PIPELINE_FIXTURE"
            else "CONSTRUCTED_REAL_G8_UNDER_SEALED_PATH"
        ),
    }
    return surface, report
