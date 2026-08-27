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


from src.g8_readiness.acquisition import NSEArchiveRecord, RbiRateRecord
from src.g8_readiness.manifests import build_selected_data_freeze, sha256_payload


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


def _synthetic_dataset(symbols: tuple[str, ...], dates: tuple[date, date]) -> tuple[list[R2Surface], list[NSEArchiveRecord], list[RbiRateRecord], dict[str, list[int]]]:
    surfaces: list[R2Surface] = []
    archives: list[NSEArchiveRecord] = []
    rates: list[RbiRateRecord] = []
    roles = {
        name: [int(index) for index, valid in enumerate(mask_array) if valid]
        for name, mask_array in canonical_slot_roles().items()
    }
    for date_idx, val_date in enumerate(dates):
        rate_record = RbiRateRecord(
            official_url=f"https://www.rbi.org.in/release-{date_idx}",
            release_identifier=f"RBI-{val_date:%Y%m%d}",
            observation_date=val_date.isoformat(),
            cutoff_price=98.7,
            yield_percent=5.25,
            source_sha256="a" * 64,
            normalized_extract_sha256="b" * 64,
        )
        rates.append(rate_record)
        rates_map = {val_date: rate_record}
        cal = {rate_record.release_identifier: val_date}
        for sym_idx, sym in enumerate(symbols):
            cm, fo = _market_frames(100.0 + sym_idx, val_date)
            cm.loc[cm.TckrSymb.eq("TEST"), "TckrSymb"] = sym
            fo.loc[fo.TckrSymb.eq("TEST"), "TckrSymb"] = sym
            surface, _ = build_g8_r2_surface(
                sym, val_date, cm, fo, rates_map,
                official_release_calendar=cal,
                development_contract_keys=set(),
            )
            surfaces.append(surface)
        for market in ("CM", "FO"):
            fn = f"BhavCopy_NSE_{market}_0_0_0_{val_date:%Y%m%d}_F_0000.csv.zip"
            archives.append(
                NSEArchiveRecord(
                    market=market,
                    trading_date=val_date.isoformat(),
                    official_url=f"https://nsearchives.nseindia.com/content/{market.lower()}/{fn}",
                    retrieval_timestamp_utc="2026-10-01T00:00:00Z",
                    original_filename=fn,
                    byte_size=1000,
                    zip_sha256="c" * 64,
                    zip_integrity_result=True,
                    member_filename=fn.removesuffix(".zip"),
                    csv_sha256="d" * 64,
                    encoding="UTF-8",
                    delimiter=",",
                    archive_path=Path(fn),
                    extracted_csv_path=Path(fn.removesuffix(".zip")),
                )
            )
    return surfaces, archives, rates, roles


def test_primary_scan_sealing_and_backup_rescan_share_canonical_schema() -> None:
    dates = (date(2026, 10, 1), date(2026, 10, 2))
    primary_symbols = ("NTPC", "CIPLA", "INFY", "HDFCBANK")

    # 1. Primary Scan Seal
    surfaces_p, archives_p, rates_p, roles = _synthetic_dataset(primary_symbols, dates)
    support_p = {d: dict.fromkeys(primary_symbols, True) for d in dates}
    scan_p = scan_common_dates(support_p, expected_calendar_dates=dates)
    primary_seal = build_selected_data_freeze(
        surfaces=surfaces_p,
        archive_records=archives_p,
        rate_records=rates_p,
        scan_result=scan_p,
        backup_decisions=(),
        observation_role_mappings=roles,
    )
    assert primary_seal["schema_version"] == "g8.selected_data_freeze/1"
    assert primary_seal["scan_mode"] == "PRIMARY_SCAN"
    assert primary_seal["status"] == "SYNTHETIC_G8_SELECTED_DATA_FIXTURE_SEALED"
    assert primary_seal["backup_decisions"] == []
    assert primary_seal["backup_activation_reason"] == "NONE_PRIMARY_SCAN"
    assert primary_seal["selected_symbols"] == sorted(primary_symbols)
    assert primary_seal["manifest_sha256"] == sha256_payload({**primary_seal, "manifest_sha256": ""})

    # 2. Backup Rescan Seal
    failed_primary_support = {d: {"NTPC": False, "CIPLA": True, "INFY": True, "HDFCBANK": True} for d in dates}
    failed_primary_scan = scan_common_dates(failed_primary_support, expected_calendar_dates=dates)
    decisions, replacements = full_window_backup_replacements(
        failed_primary_support,
        expected_scanned_dates=dates,
        primary_scan_result=failed_primary_scan,
    )
    assert replacements == {"NTPC": "POWERGRID"}
    backup_symbols = ("POWERGRID", "CIPLA", "INFY", "HDFCBANK")
    backup_support = {d: dict.fromkeys(backup_symbols, True) for d in dates}
    backup_scan = scan_common_dates(
        backup_support,
        active_symbols=backup_symbols,
        expected_calendar_dates=dates,
    )
    surfaces_b, archives_b, rates_b, _ = _synthetic_dataset(backup_symbols, dates)
    backup_seal = build_selected_data_freeze(
        surfaces=surfaces_b,
        archive_records=archives_b,
        rate_records=rates_b,
        scan_result=backup_scan,
        backup_decisions=decisions,
        observation_role_mappings=roles,
        primary_scan_result=failed_primary_scan,
    )
    assert backup_seal["schema_version"] == "g8.selected_data_freeze/1"
    assert backup_seal["scan_mode"] == "BACKUP_RESCAN"
    assert backup_seal["status"] == "SYNTHETIC_G8_SELECTED_DATA_FIXTURE_SEALED"
    assert len(backup_seal["backup_decisions"]) == 1
    assert backup_seal["selected_symbols"] == sorted(backup_symbols)
    assert backup_seal["primary_scan_provenance"]["complete_window_scanned"] is True
    assert backup_seal["manifest_sha256"] == sha256_payload({**backup_seal, "manifest_sha256": ""})

    # 3. Both share identical top-level schema keys
    assert set(primary_seal.keys()) == set(backup_seal.keys())

    # 4. Different content produces different deterministic hashes
    assert primary_seal["manifest_sha256"] != backup_seal["manifest_sha256"]

    # 5. Deterministic replay yields identical hash
    replay_seal = build_selected_data_freeze(
        surfaces=surfaces_b,
        archive_records=archives_b,
        rate_records=rates_b,
        scan_result=backup_scan,
        backup_decisions=decisions,
        observation_role_mappings=roles,
        primary_scan_result=failed_primary_scan,
    )
    assert replay_seal["manifest_sha256"] == backup_seal["manifest_sha256"]


def test_backup_seal_invariants_and_masquerade_prevention() -> None:
    dates = (date(2026, 10, 1), date(2026, 10, 2))
    primary_symbols = ("NTPC", "CIPLA", "INFY", "HDFCBANK")
    backup_symbols = ("POWERGRID", "CIPLA", "INFY", "HDFCBANK")
    surfaces_p, archives, rates, roles = _synthetic_dataset(primary_symbols, dates)
    support = {d: dict.fromkeys(primary_symbols, True) for d in dates}
    scan_p = scan_common_dates(support, expected_calendar_dates=dates)

    # Cannot mix primary surfaces with backup decisions
    decisions, _ = full_window_backup_replacements(
        {d: {"NTPC": False, "CIPLA": True, "INFY": True, "HDFCBANK": True} for d in dates},
        expected_scanned_dates=dates,
    )
    with pytest.raises(ValueError, match="unexpected symbol composition"):
        build_selected_data_freeze(
            surfaces=surfaces_p,
            archive_records=archives,
            rate_records=rates,
            scan_result=scan_p,
            backup_decisions=decisions,
            observation_role_mappings=roles,
        )

    # Primary scan mode cannot have backup decisions
    with pytest.raises(ValueError, match="cannot contain backup decisions"):
        build_selected_data_freeze(
            surfaces=surfaces_p,
            archive_records=archives,
            rate_records=rates,
            scan_result=scan_p,
            backup_decisions=decisions,
            scan_mode="PRIMARY_SCAN",
            observation_role_mappings=roles,
        )

    # Backup rescan mode cannot have empty backup decisions
    surfaces_b, _, _, _ = _synthetic_dataset(backup_symbols, dates)
    scan_b = scan_common_dates(
        {d: dict.fromkeys(backup_symbols, True) for d in dates},
        active_symbols=backup_symbols,
        expected_calendar_dates=dates,
    )
    with pytest.raises(ValueError, match="requires non-empty backup decisions"):
        build_selected_data_freeze(
            surfaces=surfaces_b,
            archive_records=archives,
            rate_records=rates,
            scan_result=scan_b,
            backup_decisions=(),
            scan_mode="BACKUP_RESCAN",
            observation_role_mappings=roles,
        )


def test_sealed_state_cannot_be_mutated_in_place() -> None:
    dates = (date(2026, 10, 1), date(2026, 10, 2))
    symbols = ("NTPC", "CIPLA", "INFY", "HDFCBANK")
    surfaces, archives, rates, roles = _synthetic_dataset(symbols, dates)
    support = {d: dict.fromkeys(symbols, True) for d in dates}
    scan = scan_common_dates(support, expected_calendar_dates=dates)
    seal = build_selected_data_freeze(
        surfaces=surfaces,
        archive_records=archives,
        rate_records=rates,
        scan_result=scan,
        backup_decisions=(),
        observation_role_mappings=roles,
    )
    original_hash = seal["manifest_sha256"]

    # Tampering with symbols
    tampered = dict(seal)
    tampered["selected_symbols"] = ["TAMPERED"]
    computed_tampered = sha256_payload({**tampered, "manifest_sha256": ""})
    assert computed_tampered != original_hash

    # Tampering with classification
    tampered2 = dict(seal)
    tampered2["classification"] = "REAL_G8_SELECTED_DATA"
    computed_tampered2 = sha256_payload({**tampered2, "manifest_sha256": ""})
    assert computed_tampered2 != original_hash
