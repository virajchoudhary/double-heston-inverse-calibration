from __future__ import annotations

import io
import hashlib
import math
import zipfile
import csv
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.nse_stage_a import (
    AUTHORIZED_DATES,
    ArchiveIntegrityError,
    UDIFF_COLUMNS,
    acquire_udiff_archive,
    analyze_stage_a,
    derive_option_observations,
    nse_archive_filename,
    nse_archive_url,
    read_prior_acquisition_evidence,
    validate_udiff_schema,
    write_stage_a_outputs,
)
import src.nse_stage_a as nse_stage_a


DAY = date(2026, 7, 1)


def _raw_row(**overrides: object) -> dict[str, str]:
    row = {column: "" for column in UDIFF_COLUMNS}
    row.update({"TradDt": "01-Jul-2026", "BizDt": "01-Jul-2026", "Sgmt": "FO", "Src": "NSE", "FinInstrmTp": "STO", "FinInstrmId": "1", "TckrSymb": "INFY", "XpryDt": "31-Jul-2026", "FininstrmActlXpryDt": "31-Jul-2026", "StrkPric": "100", "OptnTp": "CE", "ClsPric": "4", "LastPric": "3", "SttlmPric": "4", "UndrlygPric": "101", "OpnIntrst": "0", "TtlTradgVol": "0", "TtlTrfVal": "123.4", "TtlNbOfTxsExctd": "0", "NewBrdLotQty": "100"})
    row.update({key: str(value) for key, value in overrides.items()})
    return row


def _frame(rows: list[dict[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=UDIFF_COLUMNS)


def _cm(value: date = DAY) -> pd.DataFrame:
    return _frame([_raw_row(TradDt=value.strftime("%d-%b-%Y"), BizDt=value.strftime("%d-%b-%Y"), Sgmt="CM", FinInstrmTp="EQUITY", SctySrs="EQ", TckrSymb="INFY", ClsPric="100")])


def test_filename_and_official_url_construction() -> None:
    assert nse_archive_filename("FO", DAY) == "BhavCopy_NSE_FO_0_0_0_20260701_F_0000.csv.zip"
    assert nse_archive_url("CM", "2026-07-01") == "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_20260701_F_0000.csv.zip"


def test_exact_schema_rejects_missing_or_reordered_column() -> None:
    validate_udiff_schema(UDIFF_COLUMNS)
    with pytest.raises(ValueError): validate_udiff_schema(UDIFF_COLUMNS[:-1])
    with pytest.raises(ValueError): validate_udiff_schema(tuple(reversed(UDIFF_COLUMNS)))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("TradDt", "02-Jul-2026", "TradDt"),
        ("BizDt", "02-Jul-2026", "BizDt"),
        ("Src", "OTHER", "Src"),
        ("Sgmt", "CM", "Sgmt"),
    ],
)
def test_raw_identity_requires_traddt_bizdt_nse_source_and_expected_segment(
    field: str, value: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        derive_option_observations(_frame([_raw_row(**{field: value})]), _cm(), DAY)


def test_raw_names_expiry_fallback_mismatch_and_metrics_do_not_mutate_input() -> None:
    raw = _frame([
        _raw_row(XpryDt="30-Jul-2026", FininstrmActlXpryDt="31-Jul-2026", StrkPric="110"),
        _raw_row(FinInstrmId="2", XpryDt="31-Jul-2026", FininstrmActlXpryDt=""),
    ])
    before = raw.copy(deep=True)
    result = derive_option_observations(raw, _cm(), DAY)
    row = result.iloc[0]
    assert_frame_equal(raw, before)
    assert "FininstrmActlXpryDt" in result.columns and row["XpryDt"] == "30-Jul-2026"
    assert row["actual_expiry"] == date(2026, 7, 31) and not bool(row["expiry_fields_match"])
    assert result.iloc[1]["actual_expiry"] == date(2026, 7, 31) and not bool(result.iloc[1]["expiry_fields_match"])
    assert row["DTE"] == 30 and math.isclose(row["T"], 30 / 365)
    assert math.isclose(row["K_over_S"], 1.1) and math.isclose(row["log_K_over_S"], math.log(1.1))


def test_one_underlying_date_is_one_surface_and_expiry_slots_do_not_force_three() -> None:
    fo = _frame([
        _raw_row(XpryDt="31-Jul-2026", FininstrmActlXpryDt="31-Jul-2026"),
        _raw_row(FinInstrmId="2", XpryDt="28-Aug-2026", FininstrmActlXpryDt="28-Aug-2026"),
        _raw_row(FinInstrmId="3", XpryDt="25-Sep-2026", FininstrmActlXpryDt="25-Sep-2026"),
    ])
    raw_by_date = {value: {"CM": _cm(value), "FO": fo if value == DAY else _frame([_raw_row(BizDt=value.strftime("%d-%b-%Y"), TradDt=value.strftime("%d-%b-%Y"), XpryDt=(value.replace(day=28)).strftime("%d-%b-%Y"), FininstrmActlXpryDt=(value.replace(day=28)).strftime("%d-%b-%Y"))])} for value in AUTHORIZED_DATES}
    outputs = analyze_stage_a(raw_by_date)
    summary = outputs["surface_summary"]
    assert len(summary) == 3 and set(summary["surface_count"]) == {1}
    slots = derive_option_observations(fo, _cm(), DAY).set_index("actual_expiry")["expiry_slot"].to_dict()
    assert slots[date(2026, 7, 31)] == "near" and slots[date(2026, 8, 28)] == "mid"
    assert slots[date(2026, 9, 25)] == "far"


def test_price_activity_and_unit_semantics_are_separate() -> None:
    row = derive_option_observations(_frame([_raw_row()]), _cm(), DAY).iloc[0]
    assert bool(row["close_positive"]) and bool(row["last_positive"]) and bool(row["settlement_positive"])
    assert not bool(row["traded_qty_positive"]) and not bool(row["open_interest_positive"])
    assert not bool(row["bid_available"]) and not bool(row["ask_available"])
    assert row["traded_qty_label"] == "NSE Total Traded Qty" and "contract" not in row["traded_qty_label"].lower()


def test_grid_support_classification_and_nifty_ranking_exclusion() -> None:
    fo = _frame([
        _raw_row(StrkPric="100", XpryDt="08-Jul-2026", FininstrmActlXpryDt="08-Jul-2026"),
        _raw_row(FinInstrmId="2", StrkPric="110", XpryDt="08-Jul-2026", FininstrmActlXpryDt="08-Jul-2026"),
        _raw_row(FinInstrmId="3", TckrSymb="NIFTY", FinInstrmTp="IDO", StrkPric="100", XpryDt="08-Jul-2026", FininstrmActlXpryDt="08-Jul-2026"),
    ])
    raw_by_date = {value: {"CM": _cm(value), "FO": fo.assign(BizDt=value.strftime("%d-%b-%Y"), TradDt=value.strftime("%d-%b-%Y"))} for value in AUTHORIZED_DATES}
    outputs = analyze_stage_a(raw_by_date)
    support = outputs["candidate_grid_support"]
    assert "NIFTY" not in set(support["underlying"])
    assert set(support.loc[support["grid_axis"] == "maturity_days", "classification"]) >= {"DIRECT", "OUTSIDE_OBSERVED_SUPPORT"}
    moneyness = outputs["moneyness_coverage"]
    assert "DIRECT" in set(moneyness["classification"]) and "OUTSIDE_OBSERVED_SUPPORT" in set(moneyness["classification"])


def _archive_bytes(filename: str, csv_text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(filename.removesuffix(".zip"), csv_text)
    return buffer.getvalue()


def _valid_csv_text() -> str:
    return ",".join(UDIFF_COLUMNS) + "\n" + ",".join(_raw_row().get(column, "") for column in UDIFF_COLUMNS) + "\n"


def test_existing_raw_csv_conflict_stops_without_overwriting(tmp_path: Path) -> None:
    filename = nse_archive_filename("FO", DAY)
    csv_text = _valid_csv_text()
    archive_path = tmp_path / DAY.isoformat() / filename
    archive_path.parent.mkdir(parents=True); archive_path.write_bytes(_archive_bytes(filename, csv_text))
    csv_path = archive_path.with_suffix(""); csv_path.write_text("conflicting raw CSV", encoding="utf-8")
    before = archive_path.read_bytes()
    with pytest.raises(ArchiveIntegrityError): acquire_udiff_archive("FO", DAY, tmp_path)
    assert archive_path.read_bytes() == before and csv_path.read_text(encoding="utf-8") == "conflicting raw CSV"


def test_valid_raw_reuse_preserves_bytes_timestamp_and_encoding(tmp_path: Path) -> None:
    filename = nse_archive_filename("FO", DAY)
    archive_path = tmp_path / DAY.isoformat() / filename
    archive_path.parent.mkdir(parents=True)
    archive_path.write_bytes(_archive_bytes(filename, _valid_csv_text()))
    before_bytes, before_mtime = archive_path.read_bytes(), archive_path.stat().st_mtime_ns
    record = acquire_udiff_archive("FO", DAY, tmp_path)
    assert record.current_run_action == "REUSED" and record.first_acquisition_status == "PREEXISTING_NO_DURABLE_RETRIEVAL_LOG"
    assert record.encoding == "UTF-8" and record.delimiter == ","
    assert archive_path.read_bytes() == before_bytes and archive_path.stat().st_mtime_ns == before_mtime


def test_invalid_download_is_validated_before_raw_paths_are_published(tmp_path: Path) -> None:
    filename = nse_archive_filename("FO", DAY)
    invalid = _archive_bytes(filename, "wrong,header\nnot,udiff\n")
    with pytest.raises(ValueError):
        acquire_udiff_archive("FO", DAY, tmp_path, downloader=lambda _: invalid)
    assert not (tmp_path / DAY.isoformat() / filename).exists()
    assert not (tmp_path / DAY.isoformat() / filename.removesuffix(".zip")).exists()


@pytest.mark.parametrize("extra_fields", [-1, 1])
def test_malformed_csv_row_width_is_rejected_before_raw_paths_are_published(
    tmp_path: Path, extra_fields: int
) -> None:
    filename = nse_archive_filename("FO", DAY)
    values = [value for value in _raw_row().values()]
    if extra_fields < 0:
        values = values[:-1]
    else:
        values.append("unexpected")
    malformed_csv = ",".join(UDIFF_COLUMNS) + "\n" + ",".join(values) + "\n"
    with pytest.raises(ValueError, match="fields"):
        acquire_udiff_archive("FO", DAY, tmp_path, downloader=lambda _: _archive_bytes(filename, malformed_csv))
    assert not (tmp_path / DAY.isoformat() / filename).exists()
    assert not (tmp_path / DAY.isoformat() / filename.removesuffix(".zip")).exists()


def test_offline_acquisition_never_calls_downloader_when_archive_is_missing(tmp_path: Path) -> None:
    called = False

    def downloader(_: str) -> bytes:
        nonlocal called
        called = True
        return b"unexpected"

    with pytest.raises(FileNotFoundError, match="Offline acquisition"):
        acquire_udiff_archive("FO", DAY, tmp_path, downloader=downloader, allow_download=False)
    assert not called


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("TradDt", "02-Jul-2026", "TradDt"),
        ("BizDt", "02-Jul-2026", "BizDt"),
        ("Src", "OTHER", "Src"),
        ("Sgmt", "CM", "Sgmt"),
    ],
)
def test_identity_conflict_download_is_rejected_before_raw_paths_are_published(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    filename = nse_archive_filename("FO", DAY)
    row = _raw_row(**{field: value})
    conflicting_csv = ",".join(UDIFF_COLUMNS) + "\n" + ",".join(row[column] for column in UDIFF_COLUMNS) + "\n"
    with pytest.raises(ValueError, match=message):
        acquire_udiff_archive("FO", DAY, tmp_path, downloader=lambda _: _archive_bytes(filename, conflicting_csv))
    assert not (tmp_path / DAY.isoformat() / filename).exists()
    assert not (tmp_path / DAY.isoformat() / filename.removesuffix(".zip")).exists()


class _MockResponse:
    def __init__(self, payload: bytes, status: int = 200, content_type: str | None = "application/zip") -> None:
        self.payload = payload
        self.status = status
        self.headers = {} if content_type is None else {"Content-Type": content_type}

    def __enter__(self) -> "_MockResponse": return self
    def __exit__(self, *_: object) -> None: return None
    def read(self) -> bytes: return self.payload


def test_official_download_request_is_transparent_and_validates_response(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request: object, timeout: int) -> _MockResponse:
        captured["request"], captured["timeout"] = request, timeout
        return _MockResponse(b"zip-bytes")

    monkeypatch.setattr(nse_stage_a.urllib.request, "urlopen", fake_urlopen)
    assert nse_stage_a._download_official(nse_archive_url("FO", DAY)) == b"zip-bytes"
    request = captured["request"]
    assert captured["timeout"] == 180
    assert request.get_header("User-agent") == "ann-inverse-calibration-stage-a-audit/1.0"
    assert request.get_header("Accept") == "application/zip, application/octet-stream"

    monkeypatch.setattr(nse_stage_a.urllib.request, "urlopen", lambda *_args, **_kwargs: _MockResponse(b"zip", status=503))
    with pytest.raises(ValueError, match="HTTP 503"):
        nse_stage_a._download_official(nse_archive_url("CM", DAY))
    monkeypatch.setattr(nse_stage_a.urllib.request, "urlopen", lambda *_args, **_kwargs: _MockResponse(b"", content_type="text/html"))
    with pytest.raises(ValueError, match="Content-Type"):
        nse_stage_a._download_official(nse_archive_url("CM", DAY))
    monkeypatch.setattr(nse_stage_a.urllib.request, "urlopen", lambda *_args, **_kwargs: _MockResponse(b""))
    with pytest.raises(ValueError, match="empty"):
        nse_stage_a._download_official(nse_archive_url("CM", DAY))


def test_live_instrument_codes_include_stock_and_index_futures() -> None:
    fo = _frame([
        _raw_row(),
        _raw_row(FinInstrmId="2", FinInstrmTp="STF", XpryDt="08-Jul-2026", FininstrmActlXpryDt="08-Jul-2026"),
        _raw_row(FinInstrmId="3", TckrSymb="NIFTY", FinInstrmTp="IDF", XpryDt="08-Jul-2026", FininstrmActlXpryDt="08-Jul-2026"),
        _raw_row(FinInstrmId="4", XpryDt="08-Jul-2026", FininstrmActlXpryDt="08-Jul-2026"),
    ])
    raw_by_date = {value: {"CM": _cm(value), "FO": fo.assign(BizDt=value.strftime("%d-%b-%Y"), TradDt=value.strftime("%d-%b-%Y"))} for value in AUTHORIZED_DATES}
    outputs = analyze_stage_a(raw_by_date)
    futures = outputs["futures_availability"]
    assert set(futures["FinInstrmTp"]) == {"STF", "IDF"}
    assert set(futures["underlying"]) == {"INFY", "NIFTY"}
    infy_surface = outputs["surface_summary"].query("underlying == 'INFY'").iloc[0]
    assert infy_surface["option_expiry_without_matching_futures_count"] == 1
    assert infy_surface["option_to_futures_alignment_status"] == "OPTION_EXPIRIES_WITHOUT_FUTURES"


def test_surface_activity_atm_counts_and_spot_variation_are_auditable() -> None:
    fo = _frame([
        _raw_row(StrkPric="90", UndrlygPric="101", XpryDt="08-Jul-2026", FininstrmActlXpryDt="08-Jul-2026"),
        _raw_row(FinInstrmId="2", StrkPric="100", UndrlygPric="102", XpryDt="08-Jul-2026", FininstrmActlXpryDt="08-Jul-2026"),
        _raw_row(FinInstrmId="3", StrkPric="110", UndrlygPric="101", XpryDt="08-Jul-2026", FininstrmActlXpryDt="08-Jul-2026"),
    ])
    raw_by_date = {value: {"CM": _cm(value), "FO": fo.assign(BizDt=value.strftime("%d-%b-%Y"), TradDt=value.strftime("%d-%b-%Y"))} for value in AUTHORIZED_DATES}
    outputs = analyze_stage_a(raw_by_date)
    summary = outputs["surface_summary"].iloc[0]
    expiry = outputs["expiry_coverage"].iloc[0]
    spot = outputs["spot_consistency"].query("underlying == 'INFY'").iloc[0]
    assert summary["close_reported_count"] == 3 and summary["close_positive_count"] == 3
    assert summary["traded_qty_reported_count"] == 3 and summary["traded_qty_positive_count"] == 0
    assert summary["bid_available_count"] == 0 and summary["ask_size_available_count"] == 0
    assert expiry["strikes_below_spot_count"] == 1 and expiry["strikes_at_spot_count"] == 1
    assert expiry["strikes_above_spot_count"] == 1 and expiry["strikes_bracketing_ATM_pair_count"] == 1
    assert spot["fo_underlying_price_unique_count"] == 2 and spot["spot_status"] == "FO_UNDERLYING_PRICE_VARIATION"
    assert pd.isna(spot["fo_underlying_price"]) and pd.isna(spot["cm_minus_fo_underlying_price"])


def test_empty_selected_option_universe_produces_zero_presence_outputs() -> None:
    fo = _frame([_raw_row(FinInstrmTp="OTHER", TckrSymb="OTHER")])
    raw_by_date = {
        value: {"CM": _cm(value), "FO": fo.assign(BizDt=value.strftime("%d-%b-%Y"), TradDt=value.strftime("%d-%b-%Y"))}
        for value in AUTHORIZED_DATES
    }
    outputs = analyze_stage_a(raw_by_date)
    presence = outputs["universe_presence"]
    assert outputs["surface_summary"].empty
    assert len(presence) == len(AUTHORIZED_DATES) * 9
    assert set(presence.loc[presence["underlying"] == "INFY", "fo_option_rows"]) == {0}


def test_replay_preserves_first_acquisition_provenance_and_raw_immutability(tmp_path: Path) -> None:
    filename = nse_archive_filename("FO", DAY)
    archive = _archive_bytes(filename, _valid_csv_text())
    first = acquire_udiff_archive("FO", DAY, tmp_path, downloader=lambda _: archive)
    manifest_path = tmp_path / "derived" / "acquisition_manifest.csv"
    manifest_path.parent.mkdir()
    first_row = {key: str(value) for key, value in first.__dict__.items()}
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(first_row))
        writer.writeheader()
        writer.writerow(first_row)
    prior = read_prior_acquisition_evidence(manifest_path.parent)
    archive_path = tmp_path / DAY.isoformat() / filename
    csv_path = archive_path.with_suffix("")
    before_archive = archive_path.read_bytes(), archive_path.stat().st_mtime_ns
    before_csv = csv_path.read_bytes(), csv_path.stat().st_mtime_ns

    replay = acquire_udiff_archive("FO", DAY, tmp_path, prior_evidence=prior)

    assert replay.current_run_action == "REUSED"
    assert replay.first_acquisition_status == "DOWNLOADED"
    assert replay.retrieval_timestamp_utc == first.retrieval_timestamp_utc
    assert replay.retrieval_timestamp_source == first.retrieval_timestamp_source
    assert list(first_row) == list(replay.__dict__)
    assert (archive_path.read_bytes(), archive_path.stat().st_mtime_ns) == before_archive
    assert (csv_path.read_bytes(), csv_path.stat().st_mtime_ns) == before_csv


def _complete_prior_evidence(archive: bytes, filename: str) -> dict[str, str]:
    csv_bytes = _valid_csv_text().encode("utf-8")
    return {
        "archive_sha256": hashlib.sha256(archive).hexdigest(),
        "archive_size_bytes": str(len(archive)),
        "archive_member_name": filename.removesuffix(".zip"),
        "csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "zip_integrity": "True",
    }


@pytest.mark.parametrize(
    ("field", "bad_value", "remove", "message"),
    [
        ("archive_sha256", "0" * 64, False, "archive_sha256"),
        ("archive_size_bytes", "0", False, "archive_size_bytes"),
        ("archive_member_name", "wrong.csv", False, "archive_member_name"),
        ("csv_sha256", "0" * 64, False, "csv_sha256"),
        ("archive_sha256", "", False, "missing required archive_sha256"),
        ("archive_size_bytes", "", False, "missing required archive_size_bytes"),
        ("archive_member_name", "", False, "missing required archive_member_name"),
        ("csv_sha256", "", False, "missing required csv_sha256"),
        ("archive_sha256", "", True, "missing required archive_sha256"),
        ("archive_size_bytes", "", True, "missing required archive_size_bytes"),
        ("archive_member_name", "", True, "missing required archive_member_name"),
        ("csv_sha256", "", True, "missing required csv_sha256"),
        ("zip_integrity", "false", False, "zip_integrity"),
        ("zip_integrity", "", False, "zip_integrity"),
        ("zip_integrity", "unknown", False, "zip_integrity"),
        ("zip_integrity", "", True, "zip_integrity"),
    ],
)
def test_prior_evidence_mismatch_download_is_rejected_before_raw_publication(
    tmp_path: Path, field: str, bad_value: str, remove: bool, message: str
) -> None:
    filename = nse_archive_filename("FO", DAY)
    archive = _archive_bytes(filename, _valid_csv_text())
    prior = _complete_prior_evidence(archive, filename)
    if remove:
        prior.pop(field)
    else:
        prior[field] = bad_value
    with pytest.raises(ValueError, match=message):
        acquire_udiff_archive("FO", DAY, tmp_path, prior_evidence={("FO", DAY.isoformat()): prior}, downloader=lambda _: archive)
    assert not (tmp_path / DAY.isoformat() / filename).exists()
    assert not (tmp_path / DAY.isoformat() / filename.removesuffix(".zip")).exists()


def test_prior_evidence_mismatch_reuse_with_missing_csv_does_not_extract(tmp_path: Path) -> None:
    filename = nse_archive_filename("FO", DAY)
    archive = _archive_bytes(filename, _valid_csv_text())
    archive_path = tmp_path / DAY.isoformat() / filename
    archive_path.parent.mkdir(parents=True)
    archive_path.write_bytes(archive)
    prior = _complete_prior_evidence(archive, filename)
    prior["csv_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="csv_sha256"):
        acquire_udiff_archive("FO", DAY, tmp_path, prior_evidence={("FO", DAY.isoformat()): prior})
    assert archive_path.read_bytes() == archive
    assert not archive_path.with_suffix("").exists()


def test_empty_prior_evidence_download_is_rejected_before_raw_publication(tmp_path: Path) -> None:
    filename = nse_archive_filename("FO", DAY)
    archive = _archive_bytes(filename, _valid_csv_text())
    with pytest.raises(ValueError, match="missing required archive_sha256"):
        acquire_udiff_archive("FO", DAY, tmp_path, prior_evidence={("FO", DAY.isoformat()): {}}, downloader=lambda _: archive)
    assert not (tmp_path / DAY.isoformat() / filename).exists()
    assert not (tmp_path / DAY.isoformat() / filename.removesuffix(".zip")).exists()


def test_empty_prior_evidence_reuse_with_missing_csv_does_not_extract(tmp_path: Path) -> None:
    filename = nse_archive_filename("FO", DAY)
    archive = _archive_bytes(filename, _valid_csv_text())
    archive_path = tmp_path / DAY.isoformat() / filename
    archive_path.parent.mkdir(parents=True)
    archive_path.write_bytes(archive)
    with pytest.raises(ValueError, match="missing required archive_sha256"):
        acquire_udiff_archive("FO", DAY, tmp_path, prior_evidence={("FO", DAY.isoformat()): {}})
    assert archive_path.read_bytes() == archive
    assert not archive_path.with_suffix("").exists()


def test_new_archive_rolls_back_when_concurrent_csv_winner_blocks_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    filename = nse_archive_filename("FO", DAY)
    archive = _archive_bytes(filename, _valid_csv_text())
    csv_path = tmp_path / DAY.isoformat() / filename.removesuffix(".zip")
    real_atomic_write = nse_stage_a._atomic_write_new

    def csv_winner(target: Path, content: bytes) -> None:
        if target == csv_path:
            target.write_bytes(b"concurrent-winner")
            raise ArchiveIntegrityError("CSV winner published first")
        real_atomic_write(target, content)

    monkeypatch.setattr(nse_stage_a, "_atomic_write_new", csv_winner)
    with pytest.raises(ArchiveIntegrityError, match="winner"):
        acquire_udiff_archive("FO", DAY, tmp_path, downloader=lambda _: archive)
    assert not (tmp_path / DAY.isoformat() / filename).exists()
    assert csv_path.read_bytes() == b"concurrent-winner"


def test_atomic_raw_publication_never_overwrites_a_concurrent_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "raw.zip"

    def concurrent_winner(_temporary: object, destination: str | Path) -> None:
        Path(destination).write_bytes(b"winner-bytes")
        raise FileExistsError("winner published first")

    monkeypatch.setattr(nse_stage_a.os, "link", concurrent_winner)
    with pytest.raises(ArchiveIntegrityError, match="concurrently published"):
        nse_stage_a._atomic_write_new(target, b"loser-bytes")
    assert target.read_bytes() == b"winner-bytes"
    assert not list(tmp_path.glob("*.partial"))


def test_atomic_raw_publication_cleans_temp_after_durable_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(nse_stage_a.os, "fsync", lambda _: (_ for _ in ()).throw(OSError("fsync failed")))
    with pytest.raises(OSError, match="fsync failed"):
        nse_stage_a._atomic_write_new(tmp_path / "raw.zip", b"bytes")
    assert not (tmp_path / "raw.zip").exists()
    assert not list(tmp_path.glob("*.partial"))


def _eight_outputs(label: str) -> dict[str, pd.DataFrame]:
    names = {
        "acquisition_manifest", "surface_summary", "expiry_coverage", "moneyness_coverage",
        "candidate_grid_support", "futures_availability", "spot_consistency", "universe_presence",
    }
    return {name: pd.DataFrame({"generation": [label]}) for name in names}


def test_derived_output_publish_rolls_back_on_commit_failure_and_cleans_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in _eight_outputs("old"):
        (tmp_path / f"{name}.csv").write_text("old-generation\n", encoding="utf-8")
    real_replace = nse_stage_a.os.replace

    def fail_one_publish(source: str | Path, destination: str | Path) -> None:
        if Path(source).name == "surface_summary.csv" and Path(destination) == tmp_path / "surface_summary.csv":
            raise OSError("commit failure")
        real_replace(source, destination)

    monkeypatch.setattr(nse_stage_a.os, "replace", fail_one_publish)
    with pytest.raises(OSError, match="commit failure"):
        write_stage_a_outputs(_eight_outputs("new"), tmp_path)
    assert all((tmp_path / f"{name}.csv").read_text(encoding="utf-8") == "old-generation\n" for name in _eight_outputs("old"))
    assert not list(tmp_path.glob(".stage_a_outputs_*"))
    assert not list(tmp_path.glob("*.backup"))
    assert not (tmp_path / ".stage_a_outputs.lock").exists()


def test_derived_output_writer_does_not_publish_when_another_writer_holds_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = tmp_path / ".stage_a_outputs.lock"
    lock.write_text("held", encoding="utf-8")
    original_acquire = nse_stage_a._acquire_output_lock
    monkeypatch.setattr(nse_stage_a, "_acquire_output_lock", lambda path: original_acquire(path, attempts=1))
    with pytest.raises(ValueError, match="writer lock"):
        write_stage_a_outputs(_eight_outputs("new"), tmp_path)
    assert not list(tmp_path.glob(".stage_a_outputs_*"))
    assert not list(tmp_path.glob("*.backup"))
    lock.unlink()


def test_output_staging_cleanup_refuses_unexpected_contents(tmp_path: Path) -> None:
    staging = tmp_path / ".stage_a_outputs_test"
    staging.mkdir()
    unexpected = staging / "unexpected.txt"
    unexpected.write_text("retain", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected output-staging"):
        nse_stage_a._cleanup_output_staging(staging, tmp_path, sorted(_eight_outputs("x")))
    assert unexpected.read_text(encoding="utf-8") == "retain"
    assert staging.is_dir()
