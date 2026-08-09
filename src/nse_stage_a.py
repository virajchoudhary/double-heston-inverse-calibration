"""Deterministic, integrity-first NSE UDiFF Stage A acquisition and screening.

This module deliberately keeps the official daily files as raw evidence.  It does
not infer bid/ask quotes, convert NSE trading quantities into contracts, rank
candidates, interpolate a surface, or change the ANN representation.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import os
import tempfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = REPOSITORY_ROOT / "market_data_audit" / "stage_a" / "raw" / "nse"
DEFAULT_DERIVED_ROOT = REPOSITORY_ROOT / "market_data_audit" / "stage_a" / "derived"
OFFICIAL_NSE_ARCHIVE_ROOT = "https://nsearchives.nseindia.com/content"
AUTHORIZED_DATES = (date(2026, 7, 1), date(2026, 7, 15), date(2026, 7, 22))
CANDIDATES = ("NTPC", "POWERGRID", "SUNPHARMA", "CIPLA", "INFY", "TCS", "ICICIBANK", "HDFCBANK")
REFERENCE_UNDERLYING = "NIFTY"
MONEYNESS_NODES = (-0.30, -0.20, -0.10, -0.05, 0.00, 0.05, 0.10, 0.20, 0.30)
MATURITY_NODES = (7, 14, 30, 60, 90, 180)
DIRECT_MONEYNESS_TOLERANCE = 1e-12
MATURITY_NEAR_MATCH_DAYS = 2

# This is the ordered, 34-column UDiFF header observed in both official CM and
# FO daily archives.  Header order is part of the archive contract.
UDIFF_COLUMNS = (
    "TradDt", "BizDt", "Sgmt", "Src", "FinInstrmTp", "FinInstrmId", "ISIN",
    "TckrSymb", "SctySrs", "XpryDt", "FininstrmActlXpryDt", "StrkPric",
    "OptnTp", "FinInstrmNm", "OpnPric", "HghPric", "LwPric", "ClsPric",
    "LastPric", "PrvsClsgPric", "UndrlygPric", "SttlmPric", "OpnIntrst",
    "ChngInOpnIntrst", "TtlTradgVol", "TtlTrfVal", "TtlNbOfTxsExctd",
    "SsnId", "NewBrdLotQty", "Rmks", "Rsvd1", "Rsvd2", "Rsvd3", "Rsvd4",
)


class NSEStageAError(ValueError):
    """Base error for deterministic Stage A failures."""


class ArchiveIntegrityError(NSEStageAError):
    """An existing raw archive/CSV is invalid or conflicts with recorded evidence."""


@dataclass(frozen=True)
class AcquisitionRecord:
    """Provenance for one raw archive and its extracted CSV."""

    market: str
    valuation_date: str
    official_url: str
    original_filename: str
    archive_path: Path
    archive_size_bytes: int
    archive_sha256: str
    zip_integrity: bool
    archive_member_name: str
    csv_path: Path
    csv_sha256: str
    encoding: str
    delimiter: str
    current_run_action: str
    first_acquisition_status: str
    retrieval_timestamp_utc: str
    retrieval_timestamp_source: str


def nse_archive_filename(market: str, valuation_date: date | str) -> str:
    """Construct the official CM/FO UDiFF filename without network access."""
    market_code = _market_code(market)
    value = _as_date(valuation_date)
    return f"BhavCopy_NSE_{market_code}_0_0_0_{value:%Y%m%d}_F_0000.csv.zip"


def nse_archive_url(market: str, valuation_date: date | str) -> str:
    """Return the only permitted official NSE archive URL for a daily UDiFF file."""
    market_code = _market_code(market)
    directory = "fo" if market_code == "FO" else "cm"
    return f"{OFFICIAL_NSE_ARCHIVE_ROOT}/{directory}/{nse_archive_filename(market_code, valuation_date)}"


def validate_udiff_schema(columns: Sequence[str]) -> None:
    """Fail closed unless a UDiFF CSV has the exact observed 34-column schema."""
    actual = tuple(columns)
    if actual != UDIFF_COLUMNS:
        raise NSEStageAError(
            "Unexpected UDiFF schema; expected the exact ordered 34-column header "
            f"but received {len(actual)} columns."
        )


def read_udiff_csv(
    path: str | Path, valuation_date: date | str, expected_market: str
) -> pd.DataFrame:
    """Read raw UDiFF CSV data while preserving every NSE field name unchanged."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Missing UDiFF CSV: {source}")
    frame, _, _ = _read_udiff_bytes(
        source.read_bytes(), _as_date(valuation_date), _market_code(expected_market)
    )
    return frame


def acquire_udiff_archive(
    market: str,
    valuation_date: date | str,
    raw_root: str | Path = DEFAULT_RAW_ROOT,
    known_hashes: Mapping[tuple[str, str], str] | None = None,
    prior_evidence: Mapping[tuple[str, str], Mapping[str, str]] | None = None,
    downloader: Callable[[str], bytes] | None = None,
    allow_download: bool = True,
) -> AcquisitionRecord:
    """Reuse a valid raw archive or download it atomically from official NSE only.

    Existing evidence is never replaced.  A previously recorded SHA-256 conflict,
    an invalid ZIP, an unexpected member, or a CSV/archive mismatch stops safely.
    """
    market_code = _market_code(market)
    value = _as_date(valuation_date)
    filename = nse_archive_filename(market_code, value)
    official_url = nse_archive_url(market_code, value)
    directory = Path(raw_root) / value.isoformat()
    archive_path = directory / filename
    csv_path = directory / filename.removesuffix(".zip")
    identity = (market_code, value.isoformat())
    prior = (prior_evidence or {}).get(identity)
    known = prior.get("archive_sha256") if prior is not None else (known_hashes or {}).get(identity)
    was_reused = archive_path.exists()
    created_archive = False
    created_csv = False
    try:
        if was_reused:
            archive_bytes = archive_path.read_bytes()
            archive_sha = _sha256(archive_bytes)
            member_name, csv_bytes = _validated_zip_member(archive_bytes, filename.removesuffix(".zip"))
            _, encoding, delimiter = _read_udiff_bytes(csv_bytes, value, market_code)
            if prior is not None:
                _validate_prior_evidence(prior, archive_sha, len(archive_bytes), member_name, _sha256(csv_bytes))
            if known is not None and known != archive_sha:
                raise ArchiveIntegrityError(
                    f"Existing {archive_path} conflicts with recorded SHA-256; it was not overwritten."
                )
        else:
            if not allow_download:
                raise FileNotFoundError(
                    f"Offline acquisition requires local archive: {archive_path}"
                )
            download = downloader or _download_official
            archive_bytes = download(official_url)
            archive_sha = _sha256(archive_bytes)
            # Validate every archive and CSV property before either raw path exists.
            member_name, csv_bytes = _validated_zip_member(archive_bytes, filename.removesuffix(".zip"))
            _, encoding, delimiter = _read_udiff_bytes(csv_bytes, value, market_code)
            if prior is not None:
                _validate_prior_evidence(prior, archive_sha, len(archive_bytes), member_name, _sha256(csv_bytes))
            if known is not None and known != archive_sha:
                raise ArchiveIntegrityError(
                    "Downloaded archive conflicts with prior recorded SHA-256; it was not published."
                )
            if csv_path.exists() and _sha256(csv_path.read_bytes()) != _sha256(csv_bytes):
                raise ArchiveIntegrityError(
                    f"Existing extracted CSV {csv_path} conflicts with the downloaded archive; neither raw file was overwritten."
                )
            directory.mkdir(parents=True, exist_ok=True)
            _atomic_write_new(archive_path, archive_bytes)
            created_archive = True
            retrieval_timestamp = datetime.now(timezone.utc).isoformat()
            timestamp_source = "download_time"
        if csv_path.exists():
            existing_csv_sha = _sha256(csv_path.read_bytes())
            archive_csv_sha = _sha256(csv_bytes)
            if existing_csv_sha != archive_csv_sha:
                raise ArchiveIntegrityError(
                    f"Existing extracted CSV {csv_path} conflicts with its archive; it was not overwritten."
                )
        else:
            _atomic_write_new(csv_path, csv_bytes)
            created_csv = True

        # For reuse the CSV could be independently corrupted even when the archive is valid.
        _read_udiff_bytes(csv_path.read_bytes(), value, market_code)
        if was_reused and prior is not None and prior.get("retrieval_timestamp_utc") and prior.get("retrieval_timestamp_source"):
            retrieval_timestamp = prior["retrieval_timestamp_utc"]
            timestamp_source = prior["retrieval_timestamp_source"]
            first_acquisition_status = _prior_first_acquisition_status(prior)
        elif was_reused:
            retrieval_timestamp, timestamp_source = _filesystem_timestamp(archive_path)
            first_acquisition_status = "PREEXISTING_NO_DURABLE_RETRIEVAL_LOG"
        else:
            first_acquisition_status = "DOWNLOADED"
        return AcquisitionRecord(
            market=market_code,
            valuation_date=value.isoformat(),
            official_url=official_url,
            original_filename=filename,
            archive_path=archive_path,
            archive_size_bytes=len(archive_bytes),
            archive_sha256=archive_sha,
            zip_integrity=True,
            archive_member_name=member_name,
            csv_path=csv_path,
            csv_sha256=_sha256(csv_bytes),
            encoding=encoding,
            delimiter=delimiter,
            current_run_action="REUSED" if was_reused else "DOWNLOADED",
            first_acquisition_status=first_acquisition_status,
            retrieval_timestamp_utc=retrieval_timestamp,
            retrieval_timestamp_source=timestamp_source,
        )
    except Exception:
        if created_csv:
            _remove_created_raw_path(csv_path, _sha256(csv_bytes))
        if created_archive:
            _remove_created_raw_path(archive_path, archive_sha)
        raise


def derive_option_observations(
    fo_raw: pd.DataFrame,
    cm_raw: pd.DataFrame,
    valuation_date: date | str,
) -> pd.DataFrame:
    """Attach Stage A option diagnostics without renaming or mutating raw fields."""
    value = _as_date(valuation_date)
    _require_raw_frame(fo_raw, value, "FO")
    _require_raw_frame(cm_raw, value, "CM")
    instrument_type = fo_raw["FinInstrmTp"].astype(str).str.upper()
    ticker = fo_raw["TckrSymb"].astype(str)
    options = fo_raw.loc[
        ((ticker.isin(CANDIDATES)) & (instrument_type == "STO"))
        | ((ticker == REFERENCE_UNDERLYING) & (instrument_type == "IDO"))
    ].copy(deep=True)
    if options.empty:
        return _empty_option_frame()

    options["valuation_date"] = value.isoformat()
    options["underlying"] = options["TckrSymb"].astype(str)
    options["original_expiry"] = options["XpryDt"].map(_optional_nse_date)
    options["actual_expiry_raw"] = options["FininstrmActlXpryDt"].map(_optional_nse_date)
    options["actual_expiry"] = options["actual_expiry_raw"].fillna(options["original_expiry"])
    if options["actual_expiry"].isna().any():
        raise NSEStageAError("Option row is missing both XpryDt and FininstrmActlXpryDt")
    options["expiry_fields_match"] = (
        options["actual_expiry_raw"].notna()
        & options["original_expiry"].notna()
        & (options["actual_expiry_raw"] == options["original_expiry"])
    )
    options["DTE"] = options["actual_expiry"].map(lambda expiry: (expiry - value).days)
    options["T"] = options["DTE"] / 365.0
    options["expiry_slot"] = _expiry_slots(options)
    options["option_type"] = options["OptnTp"].astype(str)
    options["strike"] = _numeric(options["StrkPric"])
    options["cm_spot"] = options["underlying"].map(_cm_spot_by_underlying(cm_raw))
    options["fo_underlying_price"] = _numeric(options["UndrlygPric"])
    options["cm_minus_fo_underlying_price"] = options["cm_spot"] - options["fo_underlying_price"]
    valid_spot = (options["strike"] > 0) & (options["cm_spot"] > 0)
    options["K_over_S"] = float("nan")
    options.loc[valid_spot, "K_over_S"] = options.loc[valid_spot, "strike"] / options.loc[valid_spot, "cm_spot"]
    options["log_K_over_S"] = options["K_over_S"].map(
        lambda ratio: math.log(ratio) if pd.notna(ratio) and ratio > 0 else float("nan")
    )
    for raw_name, prefix in (("ClsPric", "close"), ("LastPric", "last"), ("SttlmPric", "settlement")):
        values = _numeric(options[raw_name])
        options[f"{prefix}_reported"] = values.notna()
        options[f"{prefix}_positive"] = values > 0
        options[f"normalized_{prefix}"] = values.where(options["cm_spot"] > 0) / options["cm_spot"]
    _attach_activity_flags(options)
    options["bid_available"] = False
    options["ask_available"] = False
    options["bid_size_available"] = False
    options["ask_size_available"] = False
    options["traded_qty_label"] = "NSE Total Traded Qty"
    options["traded_value_label"] = "NSE Total Traded Value"
    options["market_lot_label"] = "Market Lot Size"
    options["ranking_status"] = _ranking_status(options["underlying"])
    options["representation_decision"] = "PENDING_REVIEW"
    return options


def analyze_stage_a(
    raw_by_date: Mapping[date, Mapping[str, pd.DataFrame]],
    acquisitions: Iterable[AcquisitionRecord] = (),
) -> dict[str, pd.DataFrame]:
    """Produce the eight small Stage A audit tables from validated local raw data."""
    option_frames: list[pd.DataFrame] = []
    futures_frames: list[pd.DataFrame] = []
    spot_frames: list[pd.DataFrame] = []
    for value in AUTHORIZED_DATES:
        if value not in raw_by_date:
            raise NSEStageAError(f"Missing authorized Stage A date: {value:%Y-%m-%d}")
        tables = raw_by_date[value]
        if set(tables) != {"CM", "FO"}:
            raise NSEStageAError(f"{value:%Y-%m-%d} requires exactly CM and FO raw tables")
        option_frames.append(derive_option_observations(tables["FO"], tables["CM"], value))
        futures_frames.append(_derive_futures(tables["FO"], value, option_frames[-1]))
        spot_frames.append(_spot_consistency(tables["CM"], option_frames[-1], value))
    options = pd.concat(option_frames, ignore_index=True) if option_frames else _empty_option_frame()
    futures = pd.concat(futures_frames, ignore_index=True) if futures_frames else pd.DataFrame()
    spots = pd.concat(spot_frames, ignore_index=True) if spot_frames else pd.DataFrame()
    return {
        "acquisition_manifest": pd.DataFrame([_acquisition_dict(record) for record in acquisitions]),
        "surface_summary": _surface_summary(options, futures),
        "expiry_coverage": _expiry_coverage(options),
        "moneyness_coverage": _moneyness_coverage(options),
        "candidate_grid_support": _candidate_grid_support(options),
        "futures_availability": futures,
        "spot_consistency": spots,
        "universe_presence": _universe_presence(options, futures, spots),
    }


def write_stage_a_outputs(outputs: Mapping[str, pd.DataFrame], derived_root: str | Path = DEFAULT_DERIVED_ROOT) -> None:
    """Publish exactly eight pre-serialized CSVs as one lock-serialized generation."""
    expected = {
        "acquisition_manifest", "surface_summary", "expiry_coverage", "moneyness_coverage",
        "candidate_grid_support", "futures_availability", "spot_consistency", "universe_presence",
    }
    if set(outputs) != expected:
        raise NSEStageAError("Stage A outputs must be exactly the prescribed eight CSV tables")
    root = Path(derived_root)
    root.mkdir(parents=True, exist_ok=True)
    serialized = {name: outputs[name].to_csv(index=False).encode("utf-8") for name in sorted(expected)}
    staging = Path(tempfile.mkdtemp(prefix=".stage_a_outputs_", dir=root))
    lock_fd: int | None = None
    lock_path = root / ".stage_a_outputs.lock"
    try:
        for name, content in serialized.items():
            _write_complete_file(staging / f"{name}.csv", content)
        lock_fd = _acquire_output_lock(lock_path)
        _commit_output_generation(staging, root, sorted(expected))
    finally:
        if lock_fd is not None:
            _release_output_lock(lock_fd, lock_path)
        _cleanup_output_staging(staging, root, sorted(expected))


def read_known_hashes(derived_root: str | Path = DEFAULT_DERIVED_ROOT) -> dict[tuple[str, str], str]:
    """Read prior output provenance, if present, for reuse conflict detection."""
    return {
        identity: row["archive_sha256"]
        for identity, row in read_prior_acquisition_evidence(derived_root).items()
        if row.get("archive_sha256")
    }


def read_prior_acquisition_evidence(
    derived_root: str | Path = DEFAULT_DERIVED_ROOT,
) -> dict[tuple[str, str], dict[str, str]]:
    """Read prior acquisition-manifest rows without discarding durable provenance."""
    path = Path(derived_root) / "acquisition_manifest.csv"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            (row["market"], row["valuation_date"]): dict(row)
            for row in csv.DictReader(handle)
            if row.get("market") and row.get("valuation_date")
        }


def _surface_summary(options: pd.DataFrame, futures: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (underlying, valuation_date), group in options.groupby(["underlying", "valuation_date"], sort=True):
        expiries = sorted(group["actual_expiry"].unique())
        matching_futures = futures.loc[
            (futures["underlying"] == underlying) & (futures["valuation_date"] == valuation_date),
            "actual_expiry",
        ] if not futures.empty else pd.Series(dtype="object")
        option_expiries_without_futures = set(expiries).difference(matching_futures.dropna())
        row = {
            "underlying": underlying, "valuation_date": valuation_date, "surface_count": 1,
            "actual_option_expiries": "|".join(expiry.isoformat() for expiry in expiries),
            "actual_DTE_values": "|".join(str((expiry - _as_date(valuation_date)).days) for expiry in expiries),
            "option_row_count": len(group), "expiry_count": len(expiries),
            "option_expiry_without_matching_futures_count": len(option_expiries_without_futures),
            "option_to_futures_alignment_status": "ALL_OPTION_EXPIRIES_HAVE_FUTURES" if not option_expiries_without_futures else "OPTION_EXPIRIES_WITHOUT_FUTURES",
            "expiry_field_mismatch_count": int((~group["expiry_fields_match"]).sum()),
            "ranking_status": "REFERENCE_ONLY" if underlying == REFERENCE_UNDERLYING else "PENDING_QUOTE_QUALITY_LAYER",
            "representation_decision": "PENDING_REVIEW",
        }
        for field in (
            "close_reported", "close_positive", "last_reported", "last_positive",
            "settlement_reported", "settlement_positive", "traded_qty_reported",
            "traded_qty_positive", "open_interest_reported", "open_interest_positive",
            "transactions_reported", "transactions_positive", "bid_available", "ask_available",
            "bid_size_available", "ask_size_available",
        ):
            row[f"{field}_count"] = int(group[field].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _expiry_coverage(options: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in options.groupby(["underlying", "valuation_date", "actual_expiry", "DTE", "expiry_slot"], sort=True):
        positive = group.loc[group["close_positive"] | group["last_positive"] | group["settlement_positive"]]
        active = group.loc[group["traded_qty_positive"] | group["open_interest_positive"] | group["transactions_positive"]]
        logs = _numeric(group["log_K_over_S"])
        strikes = sorted(_numeric(group["strike"]).dropna().unique())
        rows.append({
            "underlying": keys[0], "valuation_date": keys[1], "actual_expiry": keys[2].isoformat(), "DTE": keys[3], "expiry_slot": keys[4],
            "call_count": int((group["OptnTp"].astype(str).str.upper() == "CE").sum()),
            "put_count": int((group["OptnTp"].astype(str).str.upper() == "PE").sum()),
            "unique_strike_count": len(strikes), "observed_min_log_moneyness": logs.min(), "observed_max_log_moneyness": logs.max(),
            "price_positive_min_log_moneyness": _numeric(positive["log_K_over_S"]).min(), "price_positive_max_log_moneyness": _numeric(positive["log_K_over_S"]).max(),
            "active_min_log_moneyness": _numeric(active["log_K_over_S"]).min(), "active_max_log_moneyness": _numeric(active["log_K_over_S"]).max(),
            **_atm_counts(strikes, group["cm_spot"]),
        })
    return pd.DataFrame(rows)


def _moneyness_coverage(options: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in options.groupby(["underlying", "valuation_date", "actual_expiry", "DTE"], sort=True):
        observed = _numeric(group["log_K_over_S"]).dropna().tolist()
        for node in MONEYNESS_NODES:
            rows.append({
                "underlying": keys[0], "valuation_date": keys[1], "actual_expiry": keys[2].isoformat(), "DTE": keys[3],
                "log_moneyness_node": node, "classification": _moneyness_classification(observed, node),
                "direct_tolerance": DIRECT_MONEYNESS_TOLERANCE, "interpolation_performed": False,
            })
    return pd.DataFrame(rows)


def _candidate_grid_support(options: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (underlying, valuation_date), group in options.groupby(["underlying", "valuation_date"], sort=True):
        if underlying == REFERENCE_UNDERLYING:
            continue
        dtes = sorted({int(value) for value in group["DTE"].dropna()})
        for node in MATURITY_NODES:
            rows.append({
                "underlying": underlying, "valuation_date": valuation_date, "grid_axis": "maturity_days", "node": node,
                "classification": _maturity_classification(dtes, node), "near_match_tolerance_days": MATURITY_NEAR_MATCH_DAYS,
                "interpolation_performed": False, "ranking_status": "PENDING_QUOTE_QUALITY_LAYER", "representation_decision": "PENDING_REVIEW",
            })
        for expiry, expiry_group in group.groupby("actual_expiry", sort=True):
            observed = _numeric(expiry_group["log_K_over_S"]).dropna().tolist()
            for node in MONEYNESS_NODES:
                rows.append({
                    "underlying": underlying, "valuation_date": valuation_date, "grid_axis": "log_moneyness", "node": node,
                    "actual_expiry": expiry.isoformat(), "classification": _moneyness_classification(observed, node),
                    "direct_tolerance": DIRECT_MONEYNESS_TOLERANCE, "interpolation_performed": False,
                    "ranking_status": "PENDING_QUOTE_QUALITY_LAYER", "representation_decision": "PENDING_REVIEW",
                })
    return pd.DataFrame(rows)


def _derive_futures(fo_raw: pd.DataFrame, value: date, options: pd.DataFrame) -> pd.DataFrame:
    _require_raw_frame(fo_raw, value, "FO")
    instrument_type = fo_raw["FinInstrmTp"].astype(str).str.upper()
    ticker = fo_raw["TckrSymb"].astype(str)
    futures = fo_raw.loc[
        ((ticker.isin(CANDIDATES)) & (instrument_type == "STF"))
        | ((ticker == REFERENCE_UNDERLYING) & (instrument_type == "IDF"))
    ].copy(deep=True)
    if futures.empty:
        return pd.DataFrame()
    futures["underlying"] = futures["TckrSymb"].astype(str)
    futures["valuation_date"] = value.isoformat()
    futures["actual_expiry_raw"] = futures["FininstrmActlXpryDt"].map(_optional_nse_date)
    futures["actual_expiry"] = futures["actual_expiry_raw"].fillna(futures["XpryDt"].map(_optional_nse_date))
    if futures["actual_expiry"].isna().any():
        raise NSEStageAError("Futures row is missing both XpryDt and FininstrmActlXpryDt")
    futures["DTE"] = futures["actual_expiry"].map(lambda expiry: (expiry - value).days)
    futures["expiry_fields_match"] = futures["actual_expiry_raw"].notna() & futures["XpryDt"].map(_optional_nse_date).notna() & (futures["actual_expiry_raw"] == futures["XpryDt"].map(_optional_nse_date))
    for raw_name, prefix in (("ClsPric", "close"), ("SttlmPric", "settlement"), ("LastPric", "last")):
        values = _numeric(futures[raw_name]); futures[f"{prefix}_reported"] = values.notna(); futures[f"{prefix}_positive"] = values > 0
    _attach_activity_flags(futures)
    option_expiries = set(zip(options.get("underlying", []), options.get("actual_expiry", [])))
    futures["matching_option_expiry_available"] = [(underlying, expiry) in option_expiries for underlying, expiry in zip(futures["underlying"], futures["actual_expiry"], strict=True)]
    futures["ranking_status"] = _ranking_status(futures["underlying"])
    return futures


def _spot_consistency(cm_raw: pd.DataFrame, options: pd.DataFrame, value: date) -> pd.DataFrame:
    _require_raw_frame(cm_raw, value, "CM")
    spots = _cm_spot_by_underlying(cm_raw)
    rows: list[dict[str, object]] = []
    for underlying in (*CANDIDATES, REFERENCE_UNDERLYING):
        if underlying == REFERENCE_UNDERLYING:
            rows.append({"underlying": underlying, "valuation_date": value.isoformat(), "spot_status": "INDEPENDENT_OFFICIAL_NSE_INDEX_SOURCE_REQUIRED", "ranking_status": "REFERENCE_ONLY"})
            continue
        option_values = _numeric(options.loc[options["underlying"] == underlying, "fo_underlying_price"]).dropna()
        unique_values = sorted(option_values.unique())
        fo_value = unique_values[0] if len(unique_values) == 1 else float("nan")
        cm_value = spots.get(underlying, float("nan"))
        if not unique_values:
            status = "FO_UNDERLYING_PRICE_MISSING"
        elif len(unique_values) > 1:
            status = "FO_UNDERLYING_PRICE_VARIATION"
        elif pd.isna(cm_value):
            status = "CM_EQ_CLOSE_MISSING"
        else:
            status = "CM_EQ_CLOSE_AND_FO_UNDERLYING_PRICE"
        rows.append({"underlying": underlying, "valuation_date": value.isoformat(), "cm_close": cm_value, "fo_underlying_price_unique_count": len(unique_values), "fo_underlying_price_min": min(unique_values) if unique_values else float("nan"), "fo_underlying_price_max": max(unique_values) if unique_values else float("nan"), "fo_underlying_price": fo_value, "cm_minus_fo_underlying_price": cm_value - fo_value if pd.notna(fo_value) else float("nan"), "spot_status": status, "ranking_status": "PENDING_QUOTE_QUALITY_LAYER"})
    return pd.DataFrame(rows)


def _universe_presence(options: pd.DataFrame, futures: pd.DataFrame, spots: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for value in AUTHORIZED_DATES:
        for underlying in (*CANDIDATES, REFERENCE_UNDERLYING):
            option_count = int(((options.get("underlying") == underlying) & (options.get("valuation_date") == value.isoformat())).sum()) if not options.empty else 0
            futures_count = int(((futures.get("underlying") == underlying) & (futures.get("valuation_date") == value.isoformat())).sum()) if not futures.empty else 0
            spot_row = spots.loc[(spots["underlying"] == underlying) & (spots["valuation_date"] == value.isoformat())]
            rows.append({"underlying": underlying, "valuation_date": value.isoformat(), "reference_only": underlying == REFERENCE_UNDERLYING, "fo_option_rows": option_count, "fo_futures_rows": futures_count, "cm_spot_status": spot_row.iloc[0]["spot_status"] if not spot_row.empty else "MISSING", "ranking_status": "REFERENCE_ONLY" if underlying == REFERENCE_UNDERLYING else "PENDING_QUOTE_QUALITY_LAYER", "representation_decision": "PENDING_REVIEW"})
    return pd.DataFrame(rows)


def _cm_spot_by_underlying(cm_raw: pd.DataFrame) -> dict[str, float]:
    equity = cm_raw.loc[(cm_raw["SctySrs"].astype(str) == "EQ") & cm_raw["TckrSymb"].astype(str).isin(CANDIDATES)]
    result: dict[str, float] = {}
    for underlying, group in equity.groupby("TckrSymb", sort=True):
        if len(group) != 1:
            raise NSEStageAError(f"CM has multiple EQ rows for {underlying}")
        result[str(underlying)] = float(_numeric(group["ClsPric"]).iloc[0])
    return result


def _attach_activity_flags(frame: pd.DataFrame) -> None:
    for raw_name, prefix in (("TtlTradgVol", "traded_qty"), ("OpnIntrst", "open_interest"), ("TtlNbOfTxsExctd", "transactions")):
        values = _numeric(frame[raw_name]); frame[f"{prefix}_reported"] = values.notna(); frame[f"{prefix}_positive"] = values > 0


def _expiry_slots(options: pd.DataFrame) -> pd.Series:
    result = pd.Series(index=options.index, dtype="object")
    for _, group in options.groupby("underlying", sort=False):
        expiries = sorted(group["actual_expiry"].dropna().unique())
        names = (["near", "mid", "far"] + [f"additional_{index}" for index in range(4, len(expiries) + 1)])[:len(expiries)]
        result.loc[group.index] = group["actual_expiry"].map(dict(zip(expiries, names, strict=True)))
    return result


def _moneyness_classification(observed: Sequence[float], node: float) -> str:
    if any(abs(value - node) <= DIRECT_MONEYNESS_TOLERANCE for value in observed): return "DIRECT"
    if observed and min(observed) <= node <= max(observed): return "BRACKETED_WITHIN_OBSERVED_SUPPORT"
    return "OUTSIDE_OBSERVED_SUPPORT"


def _maturity_classification(dtes: Sequence[int], node: int) -> str:
    if node in dtes: return "DIRECT"
    if any(abs(dte - node) <= MATURITY_NEAR_MATCH_DAYS for dte in dtes): return "NEAR_MATCH"
    if dtes and min(dtes) <= node <= max(dtes): return "BRACKETED_BY_OBSERVED_EXPIRIES"
    return "OUTSIDE_OBSERVED_SUPPORT"


def _atm_counts(strikes: Sequence[float], spots: pd.Series) -> dict[str, int]:
    spot = _numeric(spots).dropna()
    if spot.empty or not strikes:
        return {"strikes_below_spot_count": 0, "strikes_at_spot_count": 0, "strikes_above_spot_count": 0, "strikes_bracketing_ATM_pair_count": 0}
    value = float(spot.iloc[0])
    below = sum(strike < value for strike in strikes)
    at = sum(math.isclose(strike, value, rel_tol=0.0, abs_tol=DIRECT_MONEYNESS_TOLERANCE) for strike in strikes)
    above = sum(strike > value for strike in strikes)
    return {"strikes_below_spot_count": below, "strikes_at_spot_count": at, "strikes_above_spot_count": above, "strikes_bracketing_ATM_pair_count": min(below, above)}


def _empty_option_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        *UDIFF_COLUMNS, "valuation_date", "underlying", "original_expiry", "actual_expiry_raw",
        "actual_expiry", "expiry_fields_match", "DTE", "T", "expiry_slot", "option_type", "strike",
        "cm_spot", "fo_underlying_price", "cm_minus_fo_underlying_price", "K_over_S", "log_K_over_S",
    ])


def _require_raw_frame(frame: pd.DataFrame, value: date, expected_market: str) -> None:
    validate_udiff_schema(tuple(frame.columns[:len(UDIFF_COLUMNS)]))
    _validate_udiff_identity(frame, value, _market_code(expected_market))


def _acquisition_dict(record: AcquisitionRecord) -> dict[str, object]:
    return {key: (str(value) if isinstance(value, Path) else value) for key, value in record.__dict__.items()}


def _validate_prior_evidence(
    prior: Mapping[str, str], archive_sha: str, archive_size: int, member_name: str, csv_sha: str
) -> None:
    checks = {
        "archive_sha256": archive_sha,
        "archive_size_bytes": str(archive_size),
        "archive_member_name": member_name,
        "csv_sha256": csv_sha,
    }
    for field, actual in checks.items():
        recorded = prior.get(field)
        if recorded is None or not str(recorded).strip():
            raise ArchiveIntegrityError(f"Prior acquisition evidence is missing required {field}.")
        if recorded != actual:
            raise ArchiveIntegrityError(f"Reused raw archive conflicts with prior {field}; it was not overwritten.")
    if str(prior.get("zip_integrity", "")).strip().lower() != "true":
        raise ArchiveIntegrityError("Prior acquisition evidence must record zip_integrity as true.")


def _prior_first_acquisition_status(prior: Mapping[str, str]) -> str:
    durable = prior.get("first_acquisition_status")
    if durable:
        return durable
    if prior.get("retrieval_timestamp_source") == "filesystem_mtime_no_durable_retrieval_log":
        return "PREEXISTING_NO_DURABLE_RETRIEVAL_LOG"
    return prior.get("acquisition_status") or prior.get("current_run_action") or "UNKNOWN_PRIOR_ORIGIN"


def _read_udiff_bytes(
    content: bytes, expected: date, expected_market: str
) -> tuple[pd.DataFrame, str, str]:
    """Validate UDiFF bytes and return raw rows plus honest encoding provenance."""
    has_bom = content.startswith(b"\xef\xbb\xbf")
    encoding = "UTF-8 with BOM" if has_bom else "UTF-8"
    try:
        text = content.decode("utf-8-sig" if has_bom else "utf-8")
    except UnicodeDecodeError as exc:
        raise NSEStageAError("UDiFF CSV is not valid UTF-8") from exc
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=",")
    try:
        header = next(reader)
    except StopIteration as exc:
        raise NSEStageAError("UDiFF CSV has no header")
    validate_udiff_schema(header)
    rows: list[dict[str, str]] = []
    for row_number, row in enumerate(reader, start=2):
        if len(row) != len(UDIFF_COLUMNS):
            raise NSEStageAError(
                f"UDiFF CSV row {row_number} has {len(row)} fields; expected {len(UDIFF_COLUMNS)}"
            )
        rows.append(dict(zip(UDIFF_COLUMNS, row, strict=True)))
    frame = pd.DataFrame(rows, columns=list(UDIFF_COLUMNS))
    _validate_udiff_identity(frame, expected, expected_market)
    return frame, encoding, ","


def _validate_udiff_identity(frame: pd.DataFrame, expected: date, expected_market: str) -> None:
    if frame.empty:
        raise NSEStageAError("UDiFF raw table has no data rows")
    for field in ("TradDt", "BizDt"):
        observed_dates = {_parse_nse_date(item) for item in frame[field]}
        if observed_dates != {expected}:
            raise NSEStageAError(
                f"UDiFF {field} does not match requested date {expected:%Y-%m-%d}: {observed_dates}"
            )
    if not frame["Src"].eq("NSE").all():
        raise NSEStageAError("UDiFF Src must be exactly NSE for every row")
    if not frame["Sgmt"].eq(expected_market).all():
        raise NSEStageAError(
            f"UDiFF Sgmt must be exactly {expected_market} for every row"
        )


def _download_official(url: str) -> bytes:
    if not url.startswith(OFFICIAL_NSE_ARCHIVE_ROOT + "/"):
        raise NSEStageAError("Only official nsearchives.nseindia.com URLs are permitted")
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "ann-inverse-calibration-stage-a-audit/1.0",
            "Accept": "application/zip, application/octet-stream",
        },
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        status = getattr(response, "status", None)
        if status is None and hasattr(response, "getcode"):
            status = response.getcode()
        if status is not None and not (200 <= int(status) < 300):
            raise NSEStageAError(f"Official NSE archive request returned HTTP {status}")
        content_type = response.headers.get("Content-Type") if getattr(response, "headers", None) else None
        if content_type is not None:
            media_type = content_type.split(";", 1)[0].strip().lower()
            if media_type not in {"application/zip", "application/x-zip-compressed", "application/octet-stream"}:
                raise NSEStageAError(f"Official NSE archive returned incompatible Content-Type: {content_type}")
        content = response.read()
    if not content:
        raise NSEStageAError("Official NSE archive response was empty")
    return content


def _validated_zip_member(archive_bytes: bytes, expected_name: str) -> tuple[str, bytes]:
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            bad_member = archive.testzip()
            if bad_member is not None: raise ArchiveIntegrityError(f"ZIP CRC failure in member {bad_member}")
            members = [member for member in archive.infolist() if not member.is_dir()]
            if len(members) != 1 or members[0].filename != expected_name: raise ArchiveIntegrityError("ZIP must contain exactly the expected CSV member")
            return members[0].filename, archive.read(members[0])
    except zipfile.BadZipFile as exc:
        raise ArchiveIntegrityError("Invalid NSE ZIP archive") from exc


def _atomic_write_new(target: Path, content: bytes) -> None:
    if target.exists(): raise ArchiveIntegrityError(f"Refusing to overwrite existing raw evidence: {target}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", delete=False, dir=target.parent, suffix=".partial") as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        # `link` creates the final directory entry atomically and fails if another
        # writer created it first.  The temporary file is in the same directory,
        # so a successful hard-link publish is complete before it becomes visible.
        os.link(temporary, target)
    except FileExistsError as exc:
        raise ArchiveIntegrityError(
            f"Refusing to overwrite concurrently published raw evidence: {target}"
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _remove_created_raw_path(path: Path, expected_sha256: str) -> None:
    """Rollback only a path whose bytes still prove this invocation created it."""
    try:
        if path.is_file() and _sha256(path.read_bytes()) == expected_sha256:
            path.unlink()
    except OSError:
        return


def _write_complete_file(target: Path, content: bytes) -> None:
    with target.open("wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _acquire_output_lock(lock_path: Path, attempts: int = 200) -> int:
    for attempt in range(attempts):
        try:
            return os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if attempt + 1 == attempts:
                break
            time.sleep(0.05)
    raise NSEStageAError(f"Timed out waiting for Stage A output writer lock: {lock_path}")


def _release_output_lock(lock_fd: int, lock_path: Path) -> None:
    try:
        os.close(lock_fd)
    finally:
        lock_path.unlink(missing_ok=True)


def _commit_output_generation(staging: Path, root: Path, names: Sequence[str]) -> None:
    states: list[dict[str, object]] = []
    try:
        for name in names:
            target = root / f"{name}.csv"
            backup = staging / f"{name}.backup"
            state: dict[str, object] = {
                "target": target, "backup": backup, "moved_old": False, "published_new": False,
            }
            states.append(state)
            if target.exists():
                os.replace(target, backup)
                state["moved_old"] = True
            os.replace(staging / f"{name}.csv", target)
            state["published_new"] = True
    except Exception:
        for state in reversed(states):
            target = state["target"]
            backup = state["backup"]
            if state["published_new"]:
                Path(target).unlink(missing_ok=True)
            if state["moved_old"] and Path(backup).exists():
                os.replace(backup, target)
        raise


def _cleanup_output_staging(staging: Path, root: Path, names: Sequence[str]) -> None:
    """Remove only this transaction's known files, never recursively delete."""
    resolved_root = root.resolve()
    resolved_staging = staging.resolve()
    if resolved_staging.parent != resolved_root or not staging.name.startswith(".stage_a_outputs_"):
        raise NSEStageAError("Refusing to clean a staging directory outside this output transaction")
    allowed = {f"{name}.csv" for name in names} | {f"{name}.backup" for name in names}
    contents = {path.name for path in staging.iterdir()}
    unexpected = contents.difference(allowed)
    if unexpected:
        raise NSEStageAError(
            f"Refusing to clean unexpected output-staging contents: {sorted(unexpected)}"
        )
    for name in sorted(allowed):
        path = staging / name
        if path.exists() or path.is_symlink():
            path.unlink()
    staging.rmdir()


def _filesystem_timestamp(path: Path) -> tuple[str, str]:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(), "filesystem_mtime_no_durable_retrieval_log"


def _sha256(content: bytes) -> str: return hashlib.sha256(content).hexdigest()
def _numeric(values: pd.Series) -> pd.Series: return pd.to_numeric(values, errors="coerce")
def _market_code(market: str) -> str:
    value = market.upper()
    if value not in {"CM", "FO"}: raise NSEStageAError("market must be CM or FO")
    return value
def _as_date(value: date | str) -> date:
    if isinstance(value, datetime): return value.date()
    if isinstance(value, date): return value
    return _parse_nse_date(value)
def _optional_nse_date(value: object) -> date | None:
    if value is None or pd.isna(value) or str(value).strip() == "": return None
    return _parse_nse_date(str(value))
def _parse_nse_date(value: object) -> date:
    text = str(value).strip()
    for pattern in ("%Y-%m-%d", "%d-%b-%Y", "%d-%b-%y", "%d/%m/%Y"):
        try: return datetime.strptime(text, pattern).date()
        except ValueError: pass
    raise NSEStageAError(f"Invalid NSE date: {value!r}")
def _ranking_status(underlyings: pd.Series) -> pd.Series:
    return underlyings.map(lambda value: "REFERENCE_ONLY" if value == REFERENCE_UNDERLYING else "PENDING_QUOTE_QUALITY_LAYER")


__all__ = [
    "AUTHORIZED_DATES", "CANDIDATES", "DIRECT_MONEYNESS_TOLERANCE", "MATURITY_NEAR_MATCH_DAYS", "UDIFF_COLUMNS",
    "AcquisitionRecord", "ArchiveIntegrityError", "NSEStageAError", "acquire_udiff_archive", "analyze_stage_a",
    "derive_option_observations", "nse_archive_filename", "nse_archive_url", "read_known_hashes", "read_prior_acquisition_evidence", "read_udiff_csv",
    "validate_udiff_schema", "write_stage_a_outputs",
]
