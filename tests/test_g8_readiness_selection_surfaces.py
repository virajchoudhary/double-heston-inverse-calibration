from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.g8_readiness.acquisition import RbiRateRecord
from src.g8_readiness.contracts import canonical_slot_roles
from src.g8_readiness.contracts import continuous_rate, discount_factor, forward_black_price
from src.g8_readiness.harness import pricing_rows_from_surface
from src.g8_readiness.scanner import full_window_backup_replacements, scan_common_dates
from src.g8_readiness.surfaces import build_g8_r2_surface
from src.r2_representation.surface import surface_from_vectors


def test_scanner_stops_after_two_common_dates_without_models() -> None:
    support = {
        date(2026, 9, 30): {"NTPC": True, "CIPLA": True, "INFY": True, "HDFCBANK": False},
        date(2026, 10, 1): dict.fromkeys(("NTPC", "CIPLA", "INFY", "HDFCBANK"), True),
        date(2026, 10, 2): dict.fromkeys(("NTPC", "CIPLA", "INFY", "HDFCBANK"), False),
        date(2026, 10, 3): dict.fromkeys(("NTPC", "CIPLA", "INFY", "HDFCBANK"), True),
    }
    result = scan_common_dates(support)
    assert result.selected_dates == (date(2026, 10, 1), date(2026, 10, 3))
    assert result.reached_target is True
    assert [(item.valuation_date.isoformat(), item.symbol) for item in result.failures[:2]] == [
        ("2026-09-30", "HDFCBANK"),
        ("2026-10-02", "NTPC"),
    ]


def test_backup_only_after_complete_window_zero_support() -> None:
    one_failure = {
        date(2026, 9, 30): {"NTPC": False, "CIPLA": True, "INFY": True, "HDFCBANK": True},
        date(2026, 12, 31): {"NTPC": True, "CIPLA": True, "INFY": True, "HDFCBANK": True},
    }
    expected = tuple(one_failure)
    decisions, replacements = full_window_backup_replacements(one_failure, expected_scanned_dates=expected)
    assert decisions == () and replacements == {}
    zero = {
        date(2026, 9, 30): {"NTPC": False, "CIPLA": True, "INFY": True, "HDFCBANK": True},
        date(2026, 12, 31): {"NTPC": False, "CIPLA": True, "INFY": True, "HDFCBANK": True},
    }
    decisions, replacements = full_window_backup_replacements(zero, expected_scanned_dates=tuple(zero))
    assert len(decisions) == 1
    assert replacements == {"NTPC": "POWERGRID"}


def test_backup_replacement_requires_complete_calendar_support() -> None:
    sparse = {
        date(2026, 9, 30): {"NTPC": False, "CIPLA": True, "INFY": True, "HDFCBANK": True},
        date(2026, 12, 31): {"NTPC": False, "CIPLA": True, "INFY": True, "HDFCBANK": True},
    }
    with pytest.raises(Exception, match="complete official-calendar coverage"):
        full_window_backup_replacements(
            {next(iter(sparse)): next(iter(sparse.values()))},
            expected_scanned_dates=tuple(sparse),
        )


def _market_frames(spot: float, valuation: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    from src.nse_stage_a import UDIFF_COLUMNS

    cm_row = {column: "" for column in UDIFF_COLUMNS}
    cm_row.update({
        "TradDt": f"{valuation:%d-%b-%Y}", "BizDt": f"{valuation:%d-%b-%Y}", "Sgmt": "CM",
        "Src": "NSE", "FinInstrmTp": "EQ", "TckrSymb": "TEST", "SctySrs": "EQ",
        "ClsPric": f"{spot:.2f}",
    })
    rows = []
    identifier = 1
    for offset in (30, 60):
        expiry = valuation + timedelta(days=offset)
        maturity = offset / 365.0
        forward = spot * math.exp((continuous_rate(0.0525, 1.0) - 0.01) * maturity)
        future = {column: "" for column in UDIFF_COLUMNS}
        future.update({
            "TradDt": f"{valuation:%d-%b-%Y}", "BizDt": f"{valuation:%d-%b-%Y}", "Sgmt": "FO",
            "Src": "NSE", "FinInstrmTp": "STF", "FinInstrmId": str(identifier), "TckrSymb": "TEST",
            "XpryDt": f"{expiry:%d-%b-%Y}", "FininstrmActlXpryDt": f"{expiry:%d-%b-%Y}",
            "ClsPric": f"{forward:.6f}", "UndrlygPric": f"{spot:.2f}",
            "OpnIntrst": "1", "TtlTradgVol": "1", "TtlNbOfTxsExctd": "1",
        })
        rows.append(future)
        identifier += 1
        for target in (-0.10, -0.05, 0.0, 0.05, 0.10):
            strike = spot * math.exp(target)
            for suffix, option_type in (("CE", "call"), ("PE", "put")):
                price = forward_black_price(
                    forward,
                    strike,
                    maturity,
                    discount_factor(0.0525, maturity),
                    0.25,
                    option_type,
                )
                option = {column: "" for column in UDIFF_COLUMNS}
                option.update({
                    "TradDt": f"{valuation:%d-%b-%Y}", "BizDt": f"{valuation:%d-%b-%Y}", "Sgmt": "FO",
                    "Src": "NSE", "FinInstrmTp": "STO", "FinInstrmId": str(identifier), "TckrSymb": "TEST",
                    "XpryDt": f"{expiry:%d-%b-%Y}", "FininstrmActlXpryDt": f"{expiry:%d-%b-%Y}",
                    "StrkPric": f"{strike:.10f}", "OptnTp": suffix,
                    "ClsPric": f"{price:.10f}", "UndrlygPric": f"{spot:.2f}",
                    "OpnIntrst": "1", "TtlTradgVol": "1", "TtlNbOfTxsExctd": "1",
                })
                rows.append(option)
                identifier += 1
    cm_frame = pd.DataFrame([cm_row], columns=list(UDIFF_COLUMNS))
    fo_frame = pd.DataFrame(rows, columns=list(UDIFF_COLUMNS))
    return cm_frame, fo_frame


def _rates(valuation: date) -> dict[date, RbiRateRecord]:
    observation = valuation - timedelta(days=3)
    record = RbiRateRecord(
        official_url="https://www.rbi.org.in/x", release_identifier=f"R-{observation:%Y%m%d}",
        observation_date=observation.isoformat(), cutoff_price=98.7, yield_percent=5.25,
        source_sha256="0" * 64, normalized_extract_sha256="1" * 64,
    )
    return {observation: record}


def _release_calendar(rates: dict[date, RbiRateRecord]) -> dict[str, date]:
    return {record.release_identifier: observed for observed, record in rates.items()}


def test_surface_builder_masks_ties_roles_and_minimums() -> None:
    valuation = date(2026, 10, 1)
    cm, fo = _market_frames(100.0, valuation)
    rates = _rates(valuation)
    surface, report = build_g8_r2_surface(
        "TEST", valuation, cm, fo, rates,
        official_release_calendar=_release_calendar(rates),
        development_contract_keys=set(),
    )
    assert report["usable_slots"] == 20, report
    assert all(surface.mask)
    roles_mask = np.asarray(surface.mask)
    calibration_keys = [key for key in surface.slot_keys if key.target_log_moneyness in (-0.05, 0.0, 0.05)]
    holdout_keys = [key for key in surface.slot_keys if key.target_log_moneyness in (-0.10, 0.10)]
    assert len(calibration_keys) == 12 and len(holdout_keys) == 8
    rows = pricing_rows_from_surface(surface)
    assert set(rows.loc[rows.sample_role.eq("CALIBRATION"), "target_log_moneyness"]) == {-0.05, 0.0, 0.05}
    assert set(rows.loc[rows.sample_role.eq("HOLDOUT"), "target_log_moneyness"]) == {-0.10, 0.10}


def test_spot_mismatch_fails_closed() -> None:
    valuation = date(2026, 10, 1)
    cm, fo = _market_frames(100.0, valuation)
    fo.loc[fo.FinInstrmTp.eq("STO"), "UndrlygPric"] = "101"
    with pytest.raises(Exception, match="EXACT_EQUALITY"):
        build_g8_r2_surface(
            "TEST", valuation, cm, fo, _rates(valuation),
            official_release_calendar=_release_calendar(_rates(valuation)),
            development_contract_keys=set(),
        )


def test_future_rate_information_fails_closed() -> None:
    valuation = date(2026, 10, 1)
    cm, fo = _market_frames(100.0, valuation)
    future_date = valuation + timedelta(days=1)
    rates = {
        future_date: RbiRateRecord(
            official_url="https://www.rbi.org.in/future", release_identifier="FUTURE",
            observation_date=future_date.isoformat(), cutoff_price=98, yield_percent=5,
            source_sha256="a" * 64, normalized_extract_sha256="b" * 64,
        )
    }
    with pytest.raises(Exception, match="future RBI"):
        build_g8_r2_surface(
            "TEST", valuation, cm, fo, rates,
            official_release_calendar={"FUTURE": future_date},
            development_contract_keys=set(),
        )


def test_surface_builder_requires_rbi_completeness_calendar() -> None:
    valuation = date(2026, 10, 1)
    cm, fo = _market_frames(100.0, valuation)
    with pytest.raises(Exception, match="RBI_OFFICIAL_RELEASE_CALENDAR_REQUIRED"):
        build_g8_r2_surface(
            "TEST", valuation, cm, fo, _rates(valuation), development_contract_keys=set(),
        )


def test_partial_mask_pricing_rows_emit_only_valid_slots() -> None:
    valuation = date(2026, 10, 1)
    cm, fo = _market_frames(100.0, valuation)
    rates = _rates(valuation)
    surface, _report = build_g8_r2_surface(
        "TEST", valuation, cm, fo, rates,
        official_release_calendar=_release_calendar(rates),
        development_contract_keys=set(),
    )
    masked_index = 0
    key = surface.slot_keys[masked_index]
    retained_contracts = [
        row for row in surface.metadata["selected_contracts"]
        if not (
            int(row["rank"]) == key.expiry_rank
            and float(row["target"]) == key.target_log_moneyness
            and str(row["option_type"]) == key.option_type
        )
    ]
    masked_surface = surface_from_vectors(
        list(surface.prices),
        list(surface.mask),
        list(surface.maturities),
        list(surface.rates),
        list(surface.carries),
        spot=surface.spot,
        surface_id=surface.surface_id + "_MASKED",
        source=surface.source,
        metadata={**surface.metadata, "selected_contracts": retained_contracts},
    )
    object.__setattr__(masked_surface, "mask", tuple(bool(value) if index != masked_index else False for index, value in enumerate(surface.mask)))
    prices = list(surface.prices)
    prices[masked_index] = 0.0
    object.__setattr__(masked_surface, "prices", tuple(prices))
    rows = pricing_rows_from_surface(masked_surface)
    assert len(rows) == 19
    assert 0 not in rows.slot_index.tolist()


def test_extra_distance_candidate_is_explicitly_rejected() -> None:
    valuation = date(2026, 10, 1)
    cm, fo = _market_frames(100.0, valuation)
    extra = fo.iloc[2].copy()
    extra["FinInstrmId"] = "999"
    extra["StrkPric"] = fo.iloc[2]["StrkPric"]
    extra["ClsPric"] = fo.iloc[2]["ClsPric"]
    extra2 = extra.copy()
    extra2["FinInstrmId"] = "998"
    fo = pd.concat([fo, pd.DataFrame([extra, extra2])], ignore_index=True)
    rates = _rates(valuation)
    _surface, report = build_g8_r2_surface(
        "TEST", valuation, cm, fo, rates,
        official_release_calendar=_release_calendar(rates),
        development_contract_keys=set(),
    )
    reasons = {row["reason"] for row in report["rejection_reasons"]}
    assert reasons & {"NOT_ASSIGNED_HUNGARIAN", "TARGET_DISTANCE_REJECTED"}


def test_real_surface_overlap_check_fails_closed_without_authorization() -> None:
    valuation = date(2026, 10, 1)
    cm, fo = _market_frames(100.0, valuation)
    rates = _rates(valuation)
    with pytest.raises(Exception, match="separately sealed selected-data path"):
        build_g8_r2_surface(
            "TEST",
            valuation,
            cm,
            fo,
            rates,
            data_classification="REAL_G8_SELECTED_DATA",
            official_release_calendar=_release_calendar(rates),
            development_contract_keys={"call|2026-10-01|2026-10-31|90|X"},
        )
