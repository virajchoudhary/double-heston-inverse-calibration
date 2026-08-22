"""Market-support audit for the five NTPC development dates (G2 R2/R3 study).

Reproduces the existing official-NSE support/activity/quote-selection contract
from ``scripts/run_ntpc_dh_multi_date_calibration.py`` / the single-stock pilot,
extended to the first THREE listed expiry ranks and all five development
dates, with one deliberate difference required by the audit's purpose: where
the committed calibration script raised on inactive futures or missing
matches, this audit RECORDS the failure and masks the affected slots (fail
closed, never impute).

Rate conditioning: the committed, hash-sealed RBI 91-day T-bill observations
(5.2521% on 2026-07-01, 5.3324% on 2026-07-15) are used, extended to 07-08 and
07-29 by that contract's own carry-forward convention (latest preserved
observation on or before the valuation date).  No new acquisition is performed
and nothing is fabricated; the absence of preserved auction artifacts for
those two dates is documented in the audit output.

Outputs per date and aggregate: listed expiry ranks, actual DTE, eligible
expiries, usable central-five call/put slots, total usable slots, mask
count/rate, support/activity failures, quote-selection failures, and R2/R3
representation completeness, plus the per-rank rate/carry conditioning used by
the synthetic geometry.
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from ..nse_stage_a import read_udiff_csv
from . import frozen
from .geometry import DateProfile

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TICKER = "NTPC"
TARGETS = frozen.CENTRAL_FIVE
MONEYNESS_LIMIT = 0.10 + 1e-12
TARGET_GATE = 0.05 + 1e-12
RANK_COUNT = frozen.R3_EXPIRY_RANKS


def _raw_paths(date_id: str) -> tuple[Path, Path]:
    stamp = date_id.replace("-", "")
    root = REPOSITORY_ROOT / frozen.MARKET_RAW_ROOTS[date_id] / date_id
    return (
        root / f"BhavCopy_NSE_CM_0_0_0_{stamp}_F_0000.csv",
        root / f"BhavCopy_NSE_FO_0_0_0_{stamp}_F_0000.csv",
    )


def _implied_volatility(
    price: float, forward: float, strike: float, maturity: float,
    discount: float, option_type: str,
) -> float:
    """Black IV on the matched-futures forward — the committed pilot contract."""
    import sys

    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))
    from scripts.run_ntpc_single_stock_pilot import implied_volatility

    try:
        return float(
            implied_volatility(price, forward, strike, maturity, discount, option_type)
        )
    except ValueError:
        return float("nan")


def _assign_targets(group: pd.DataFrame) -> dict[int, float]:
    """Hungarian assignment with the 0.05 target-distance gate (existing contract)."""
    eligible = group.loc[
        group["activity_eligible"] & group["within_moneyness"] & group["iv_eligible"]
    ].copy()
    eligible = eligible.sort_values(["strike", "FinInstrmId"]).drop_duplicates(
        "strike", keep="first"
    )
    candidates = eligible["log_moneyness"].to_numpy(float)
    if not len(candidates):
        return {}
    costs = np.abs(np.asarray(TARGETS)[:, None] - candidates[None, :])
    costs += 1e-12 * np.arange(len(candidates))[None, :]
    costs[costs > TARGET_GATE] = 1e6
    dummy = np.full((len(TARGETS), len(TARGETS)), 1e6)
    np.fill_diagonal(dummy, 0.0500000001)
    left, right = linear_sum_assignment(np.concatenate([costs, dummy], axis=1))
    return {
        int(eligible.index[j]): float(TARGETS[i])
        for i, j in zip(left, right, strict=True)
        if j < len(eligible)
    }


def audit_date(date_id: str) -> dict[str, Any]:
    """Run the full per-date audit; every failure is recorded, none imputed."""
    report: dict[str, Any] = {
        "date_id": date_id,
        "raw_root": frozen.MARKET_RAW_ROOTS[date_id],
        "futures_support_failures": [],
        "quote_selection_failures": [],
    }
    cm_path, fo_path = _raw_paths(date_id)
    for path in (cm_path, fo_path):
        if not path.is_file():
            report["hard_failure"] = f"missing raw artifact: {path.name}"
            report["constructible"] = False
            return report
    valuation = date.fromisoformat(date_id)
    cm = read_udiff_csv(cm_path, valuation, "CM")
    fo = read_udiff_csv(fo_path, valuation, "FO")

    spot_rows = cm.loc[(cm["TckrSymb"] == TICKER) & (cm["SctySrs"] == "EQ")]
    if len(spot_rows) != 1:
        report["hard_failure"] = "NTPC EQ spot row missing"
        report["constructible"] = False
        return report
    spot = float(spot_rows.iloc[0]["ClsPric"])
    report["spot"] = spot

    futures = fo.loc[(fo["TckrSymb"] == TICKER) & (fo["FinInstrmTp"] == "STF")].copy()
    futures["expiry"] = pd.to_datetime(futures["FininstrmActlXpryDt"]).dt.date
    listed_expiries = tuple(sorted(futures["expiry"].unique()))
    report["listed_expiry_ranks"] = [str(value) for value in listed_expiries[:RANK_COUNT]]
    report["listed_expiry_count"] = len(listed_expiries)

    rate_observation = frozen.RATE_SOURCE_BY_VALUATION[date_id]
    rate_yield = float(frozen.RATE_OBSERVATIONS[rate_observation]["yield"])
    report["rate_observation_date"] = rate_observation
    report["rate_simple_yield"] = rate_yield
    report["rate_carry_forward"] = rate_observation != date_id

    expiry_info: dict[date, dict[str, Any]] = {}
    for rank, expiry in enumerate(listed_expiries[:RANK_COUNT], start=1):
        rows = futures.loc[futures["expiry"] == expiry]
        if len(rows) != 1:
            report["futures_support_failures"].append(
                f"rank{rank}: expected one matched future, found {len(rows)}"
            )
            continue
        row = rows.iloc[0]
        activity = (
            float(row["ClsPric"]), float(row["TtlTradgVol"]),
            float(row["OpnIntrst"]), float(row["TtlNbOfTxsExctd"]),
        )
        if min(activity) <= 0.0:
            report["futures_support_failures"].append(
                f"rank{rank}: inactive matched future (close/volume/OI/trades not all positive)"
            )
            continue
        dte = (expiry - valuation).days
        maturity = dte / 365.0
        discount = 1.0 / (1.0 + rate_yield * maturity)
        rate = -math.log(discount) / maturity
        forward = float(row["ClsPric"])
        carry = rate - math.log(forward / spot) / maturity
        expiry_info[expiry] = {
            "rank": rank, "expiry_date": expiry.isoformat(), "dte": dte,
            "forward": forward, "discount": discount, "rate": rate, "carry": carry,
        }

    eligible_expiries = sorted(expiry_info)
    report["eligible_expiry_ranks"] = [
        expiry_info[item]["rank"] for item in eligible_expiries
    ]
    report["expiry_details"] = [expiry_info[item] for item in eligible_expiries]

    options = fo.loc[(fo["TckrSymb"] == TICKER) & (fo["FinInstrmTp"] == "STO")].copy()
    options["expiry_obj"] = pd.to_datetime(options["FininstrmActlXpryDt"]).dt.date
    options = options.loc[options["expiry_obj"].isin(eligible_expiries)].copy()
    options["option_type"] = options["OptnTp"].map({"CE": "call", "PE": "put"})
    options["strike"] = pd.to_numeric(options["StrkPric"], errors="raise")
    options["log_moneyness"] = np.log(options["strike"] / spot)
    options["observed_price"] = pd.to_numeric(options["ClsPric"], errors="coerce")
    options["traded_volume"] = pd.to_numeric(options["TtlTradgVol"], errors="coerce")
    options["open_interest"] = pd.to_numeric(options["OpnIntrst"], errors="coerce")
    options["trade_count"] = pd.to_numeric(options["TtlNbOfTxsExctd"], errors="coerce")
    options["activity_eligible"] = (
        options["observed_price"].gt(0) & options["traded_volume"].gt(0)
        & options["open_interest"].gt(0) & options["trade_count"].gt(0)
    )
    options["within_moneyness"] = options["log_moneyness"].abs() <= MONEYNESS_LIMIT
    options["matched_futures_price"] = options["expiry_obj"].map(
        lambda item: expiry_info[item]["forward"]
    )
    options["discount_factor"] = options["expiry_obj"].map(
        lambda item: expiry_info[item]["discount"]
    )
    market_iv: list[float] = []
    for row in options.itertuples():
        info = expiry_info[row.expiry_obj]
        market_iv.append(
            _implied_volatility(
                float(row.observed_price), info["forward"], float(row.strike),
                info["dte"] / 365.0, info["discount"], str(row.option_type),
            )
        )
    options["market_implied_volatility"] = market_iv
    options["iv_eligible"] = np.isfinite(options["market_implied_volatility"])
    options["target_log_moneyness"] = np.nan
    for (_, _), group in options.groupby(["expiry_obj", "option_type"], sort=True):
        for index, target in _assign_targets(group).items():
            options.loc[index, "target_log_moneyness"] = target

    slot_rows: list[dict[str, Any]] = []
    for info in report["expiry_details"]:
        for option_type in ("call", "put"):
            for target in TARGETS:
                selected = options.loc[
                    (options["expiry_obj"] == pd.Timestamp(
                        date.fromisoformat(info["expiry_date"])
                    ).date())
                    & (options["option_type"] == option_type)
                    & (options["target_log_moneyness"] == target)
                ]
                usable = len(selected) == 1
                reason = "SELECTED"
                if not usable:
                    universe = options.loc[
                        (options["expiry_obj"] == pd.Timestamp(
                            date.fromisoformat(info["expiry_date"])
                        ).date())
                        & (options["option_type"] == option_type)
                    ]
                    active = universe.loc[
                        universe["activity_eligible"] & universe["within_moneyness"]
                    ]
                    if not len(universe):
                        reason = "NO_OPTION_ROWS_FOR_EXPIRY"
                    elif not universe["activity_eligible"].any():
                        reason = "SUPPORT_ACTIVITY_FAILURE"
                    elif not len(active):
                        reason = "NO_ACTIVE_ROW_WITHIN_MONEYNESS"
                    elif not active["iv_eligible"].any():
                        reason = "IV_ELIGIBILITY_FAILURE"
                    else:
                        reason = "QUOTE_SELECTION_FAILURE_NO_TARGET_WITHIN_GATE"
                    report["quote_selection_failures"].append(
                        f"rank{info['rank']}/{option_type}/k{target:+.2f}: {reason}"
                    )
                slot_rows.append(
                    {
                        "date_id": date_id,
                        "expiry_rank": info["rank"],
                        "expiry_date": info["expiry_date"],
                        "dte": info["dte"],
                        "option_type": option_type,
                        "target_log_moneyness": target,
                        "usable": usable,
                        "failure_reason": reason if not usable else "",
                        "strike": float(selected.iloc[0]["strike"]) if usable else None,
                        "observed_price": float(selected.iloc[0]["observed_price"]) if usable else None,
                        "log_moneyness_actual": float(selected.iloc[0]["log_moneyness"]) if usable else None,
                    }
                )
    slots = pd.DataFrame(slot_rows)
    report["slot_table"] = slots
    total = len(slots)
    usable = int(slots["usable"].sum())
    report["total_slots"] = total
    report["usable_slots"] = usable
    report["mask_count"] = total - usable
    report["mask_rate"] = (total - usable) / total if total else float("nan")
    report["usable_call_slots"] = int(
        slots.loc[slots["option_type"] == "call", "usable"].sum()
    )
    report["usable_put_slots"] = int(
        slots.loc[slots["option_type"] == "put", "usable"].sum()
    )
    report["r2_slots"] = int(slots.loc[slots["expiry_rank"] <= 2].shape[0])
    report["r2_usable"] = int(
        slots.loc[(slots["expiry_rank"] <= 2) & slots["usable"]].shape[0]
    )
    report["r3_usable"] = usable
    report["r2_completeness"] = (
        report["r2_usable"] / frozen.R2_NOMINAL_SLOTS
    )
    report["r3_completeness"] = usable / frozen.R3_NOMINAL_SLOTS
    report["constructible"] = bool(eligible_expiries) and report["r2_usable"] > 0
    report["slot_failure_breakdown"] = (
        slots.loc[~slots["usable"], "failure_reason"].value_counts().to_dict()
    )
    return report


def audit_all_dates() -> tuple[list[dict[str, Any]], pd.DataFrame]:
    reports = [audit_date(date_id) for date_id in frozen.MARKET_DATES]
    slot_tables = [report.pop("slot_table") for report in reports]
    return reports, pd.concat(slot_tables, ignore_index=True)


def date_profiles(reports: list[dict[str, Any]]) -> list[DateProfile]:
    """Build the synthetic-geometry conditioning profiles from the audit."""
    profiles: list[DateProfile] = []
    for report in reports:
        details = sorted(report["expiry_details"], key=lambda item: item["rank"])
        if len(details) < frozen.R3_EXPIRY_RANKS:
            raise RuntimeError(
                f"date {report['date_id']} lacks three eligible expiry ranks; "
                "R3 synthetic geometry cannot be built (recorded, not imputed)"
            )
        profiles.append(
            DateProfile(
                date_id=report["date_id"],
                spot=report["spot"],
                expiry_dates=tuple(item["expiry_date"] for item in details),
                dte=tuple(int(item["dte"]) for item in details),
                rates=tuple(float(item["rate"]) for item in details),
                carries=tuple(float(item["carry"]) for item in details),
            )
        )
    return profiles
