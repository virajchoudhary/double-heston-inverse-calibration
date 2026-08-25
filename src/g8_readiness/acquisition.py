"""Future official-source acquisition interfaces with hard current locks."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ..nse_stage_a import UDIFF_COLUMNS
from .contracts import G8ReadinessError, validate_g8_valuation_date

OFFICIAL_NSE_ARCHIVE_ROOT = "https://nsearchives.nseindia.com/content"
OFFICIAL_RBI_ROOT = "https://www.rbi.org.in/"


class G8AcquisitionLocked(G8ReadinessError):
    """Raised unless the caller supplies an explicit future authorization flag."""


@dataclass(frozen=True)
class NSEArchiveRecord:
    market: str
    trading_date: str
    official_url: str
    retrieval_timestamp_utc: str
    original_filename: str
    byte_size: int
    zip_sha256: str
    zip_integrity_result: bool
    member_filename: str
    csv_sha256: str
    encoding: str
    delimiter: str
    archive_path: Path
    extracted_csv_path: Path


@dataclass(frozen=True)
class RbiRateRecord:
    official_url: str
    release_identifier: str
    observation_date: str
    cutoff_price: float
    yield_percent: float
    source_sha256: str
    normalized_extract_sha256: str
    source_path: Path | None = None
    normalized_path: Path | None = None


Downloader = Callable[[str], bytes]


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _atomic_new(target: Path, content: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    except FileExistsError as exc:
        raise G8AcquisitionLocked(f"immutable artifact already exists: {target}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def nse_archive_identity(market: str, valuation_date: date | str) -> tuple[str, str, str, str]:
    value = validate_g8_valuation_date(valuation_date)
    code = {"CM": "CM", "FO": "FO"}.get(market.upper())
    if code is None:
        raise G8ReadinessError("market must be CM or FO")
    directory = "cm" if code == "CM" else "fo"
    filename = f"BhavCopy_NSE_{code}_0_0_0_{value:%Y%m%d}_F_0000.csv.zip"
    url = f"{OFFICIAL_NSE_ARCHIVE_ROOT}/{directory}/{filename}"
    return value.isoformat(), code, filename, url


def _validate_udiff_bytes(content: bytes, expected_date: date, market: str) -> tuple[str, str]:
    has_bom = content.startswith(b"\xef\xbb\xbf")
    encoding = "UTF-8 with BOM" if has_bom else "UTF-8"
    try:
        text = content.decode("utf-8-sig" if has_bom else "utf-8")
    except UnicodeDecodeError as exc:
        raise G8ReadinessError("UDiFF CSV is not UTF-8") from exc
    rows = list(csv.reader(io.StringIO(text, newline=""), delimiter=","))
    if not rows:
        raise G8ReadinessError("UDiFF CSV has no header")
    if tuple(rows[0]) != UDIFF_COLUMNS:
        raise G8ReadinessError("UDiFF schema differs from the frozen 34-column contract")
    if any(len(row) != len(UDIFF_COLUMNS) for row in rows[1:]):
        raise G8ReadinessError("UDiFF row width differs from schema")
    frame = pd.DataFrame(rows[1:], columns=list(UDIFF_COLUMNS))
    if frame.empty:
        raise G8ReadinessError("UDiFF CSV contains no data row")
    observed_dates = {_parse_iso_or_nse(row) for row in frame["TradDt"]}
    if observed_dates != {expected_date}:
        raise G8ReadinessError("UDiFF trading-date identity mismatch")
    if frame["Src"].ne("NSE").any() or frame["Sgmt"].ne(market).any():
        raise G8ReadinessError("UDiFF source/segment identity mismatch")
    return encoding, ","


def _validated_zip_member(archive_bytes: bytes, expected_member: str) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise G8ReadinessError(f"ZIP CRC failure at member {bad_member}")
            members = [item.filename for item in archive.infolist() if not item.is_dir()]
            if members != [expected_member]:
                raise G8ReadinessError("ZIP must contain exactly one expected CSV member")
    except zipfile.BadZipFile as exc:
        raise G8ReadinessError("invalid ZIP archive") from exc
    return expected_member


def _retain_failed_archive(root: Path, identity: dict[str, Any], reason: str) -> Path:
    rejected_root = root / "rejected"
    rejected_root.mkdir(parents=True, exist_ok=True)
    safe_name = f"{identity['trading_date']}_{identity['original_filename']}.rejected.json"
    payload = {
        **{key: value for key, value in identity.items() if key != "archive_bytes"},
        "archive_sha256": _sha256_bytes(identity["archive_bytes"]),
        "byte_size": len(identity["archive_bytes"]),
        "failure_reason": reason,
        "retained": True,
    }
    target = rejected_root / safe_name
    if not target.exists():
        _atomic_new(target, (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode())
    return target


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise G8ReadinessError(f"invalid immutable provenance JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise G8ReadinessError(f"provenance must be a JSON object: {path}")
    return payload


def intake_official_nse(
    market: str,
    valuation_date: date | str,
    *,
    store_root: Path | str,
    authorize_acquisition: bool,
    downloader: Downloader | None = None,
    retrieval_timestamp_utc: str | None = None,
) -> NSEArchiveRecord:
    """Acquire one authorized CM/FO archive later; tonight callers use fixtures only."""
    if authorize_acquisition is not True:
        raise G8AcquisitionLocked(
            "current run lacks --authorize-g8-acquisition; real network acquisition is forbidden"
        )
    trading_date, market_code, filename, url = nse_archive_identity(market, valuation_date)
    directory = Path(store_root) / "official_nse" / trading_date
    archive_path = directory / filename
    extracted_path = directory / filename.removesuffix(".zip")
    provenance_path = directory / f".provenance.{filename}.json"

    def read_existing() -> tuple[bytes, str, str]:
        archive_bytes = archive_path.read_bytes()
        try:
            member = _validated_zip_member(archive_bytes, filename.removesuffix(".zip"))
            with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
                csv_bytes = archive.read(member)
            encoding, delimiter = _validate_udiff_bytes(csv_bytes, date.fromisoformat(trading_date), market_code)
        except Exception as exc:
            _retain_failed_archive(Path(store_root), {
                "trading_date": trading_date,
                "original_filename": filename,
                "official_url": url,
                "archive_bytes": archive_bytes,
            }, str(exc))
            raise
        return csv_bytes, encoding, delimiter

    if archive_path.exists():
        csv_bytes, encoding, delimiter = read_existing()
        record = NSEArchiveRecord(
            market=market_code,
            trading_date=trading_date,
            official_url=url,
            retrieval_timestamp_utc=retrieval_timestamp_utc or "",
            original_filename=filename,
            byte_size=len(archive_path.read_bytes()),
            zip_sha256=_sha256_bytes(archive_path.read_bytes()),
            zip_integrity_result=True,
            member_filename=filename.removesuffix(".zip"),
            csv_sha256=_sha256_bytes(csv_bytes),
            encoding=encoding,
            delimiter=delimiter,
            archive_path=archive_path,
            extracted_csv_path=extracted_path,
        )
        expected_provenance = {
            "official_url": url,
            "original_filename": filename,
            "byte_size": record.byte_size,
            "zip_sha256": record.zip_sha256,
            "zip_integrity_result": True,
            "member_filename": record.member_filename,
            "csv_sha256": record.csv_sha256,
            "encoding": encoding,
            "delimiter": delimiter,
            "trading_date": trading_date,
        }
        stored_provenance = _read_json(provenance_path)
        mismatched = [
            key for key, value in expected_provenance.items()
            if stored_provenance.get(key) != value
        ]
        if mismatched:
            raise G8ReadinessError(
                f"immutable NSE provenance mismatch for {filename}: {mismatched}"
            )
        if (
            retrieval_timestamp_utc is not None
            and stored_provenance["retrieval_timestamp_utc"] != retrieval_timestamp_utc
        ):
            raise G8ReadinessError("immutable NSE retrieval timestamp mismatch")
        return NSEArchiveRecord(
            **{
                **record.__dict__,
                "retrieval_timestamp_utc": stored_provenance["retrieval_timestamp_utc"],
            }
        )

    if downloader is None:
        raise G8AcquisitionLocked("explicit future downloader required; no hidden fallback exists")
    archive_bytes = downloader(url)
    identity = {
        "trading_date": trading_date,
        "original_filename": filename,
        "official_url": url,
        "archive_bytes": archive_bytes,
    }
    try:
        member = _validated_zip_member(archive_bytes, filename.removesuffix(".zip"))
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            csv_bytes = archive.read(member)
        encoding, delimiter = _validate_udiff_bytes(csv_bytes, date.fromisoformat(trading_date), market_code)
    except Exception as exc:
        _retain_failed_archive(Path(store_root), identity, str(exc))
        raise
    timestamp = retrieval_timestamp_utc or datetime.now(timezone.utc).isoformat()
    provenance = {
        "schema_version": "g8.nse_archive_provenance/1",
        "official_url": url,
        "retrieval_timestamp_utc": timestamp,
        "original_filename": filename,
        "byte_size": len(archive_bytes),
        "zip_sha256": _sha256_bytes(archive_bytes),
        "zip_integrity_result": True,
        "member_filename": member,
        "csv_sha256": _sha256_bytes(csv_bytes),
        "encoding": encoding,
        "delimiter": delimiter,
        "trading_date": trading_date,
        "data_classification": "REAL_G8_MARKET_CANDIDATE_NOT_SEALED",
    }
    _atomic_new(archive_path, archive_bytes)
    _atomic_new(extracted_path, csv_bytes)
    _atomic_new(provenance_path, (json.dumps(provenance, sort_keys=True, indent=2) + "\n").encode())
    return NSEArchiveRecord(**{
        "market": market_code,
        "trading_date": trading_date,
        "official_url": url,
        "retrieval_timestamp_utc": timestamp,
        "original_filename": filename,
        "byte_size": len(archive_bytes),
        "zip_sha256": provenance["zip_sha256"],
        "zip_integrity_result": True,
        "member_filename": member,
        "csv_sha256": provenance["csv_sha256"],
        "encoding": encoding,
        "delimiter": delimiter,
        "archive_path": archive_path,
        "extracted_csv_path": extracted_path,
    })


def normalize_rbi_auction(
    html: str,
    *,
    official_url: str,
    latest_permitted_observation_date: date | str,
) -> RbiRateRecord:
    """Parse a deliberately explicit fixture/future result without backward leakage."""
    permitted = latest_permitted_observation_date if isinstance(latest_permitted_observation_date, date) else date.fromisoformat(latest_permitted_observation_date)
    fields = {
        name: _html_attribute(html, name)
        for name in ("release-id", "observation-date", "cutoff-price", "yield-percent")
    }
    missing = sorted(name for name, value in fields.items() if value is None)
    if missing:
        raise G8ReadinessError(f"RBI result missing required attributes: {missing}")
    observation = date.fromisoformat(fields["observation-date"])
    if observation > permitted:
        raise G8ReadinessError("future RBI observation is forbidden for this valuation date")
    if not official_url.startswith(OFFICIAL_RBI_ROOT):
        raise G8ReadinessError("RBI URL is outside the official domain")
    cutoff = float(fields["cutoff-price"])
    yield_percent = float(fields["yield-percent"])
    if cutoff <= 0.0 or yield_percent < 0.0:
        raise G8ReadinessError("RBI cutoff/yield must be valid non-negative values")
    normalized_payload = {
        "official_url": official_url,
        "release_identifier": fields["release-id"],
        "observation_date": observation.isoformat(),
        "cutoff_price": cutoff,
        "yield_percent": yield_percent,
    }
    normalized_json = json.dumps(normalized_payload, sort_keys=True, separators=(",", ":")).encode()
    return RbiRateRecord(
        official_url=official_url,
        release_identifier=fields["release-id"],
        observation_date=observation.isoformat(),
        cutoff_price=cutoff,
        yield_percent=yield_percent,
        source_sha256=hashlib.sha256(html.encode()).hexdigest(),
        normalized_extract_sha256=_sha256_bytes(normalized_json),
    )


def intake_official_rbi(
    html: str,
    *,
    store_root: Path | str,
    official_url: str,
    latest_permitted_observation_date: date | str,
    authorize_acquisition: bool,
) -> RbiRateRecord:
    if authorize_acquisition is not True:
        raise G8AcquisitionLocked("RBI acquisition requires --authorize-g8-acquisition")
    record = normalize_rbi_auction(
        html,
        official_url=official_url,
        latest_permitted_observation_date=latest_permitted_observation_date,
    )
    directory = Path(store_root) / "official_rbi"
    source_path = directory / f"{record.release_identifier}.html"
    normalized_path = directory / f"{record.release_identifier}.normalized.json"
    if not source_path.exists():
        _atomic_new(source_path, html.encode())
    if not normalized_path.exists():
        payload = {
            "official_url": record.official_url,
            "release_identifier": record.release_identifier,
            "observation_date": record.observation_date,
            "cutoff_price": record.cutoff_price,
            "yield_percent": record.yield_percent,
            "source_sha256": record.source_sha256,
            "normalized_extract_sha256": record.normalized_extract_sha256,
            "latest_permitted_observation_date": str(latest_permitted_observation_date),
        }
        _atomic_new(normalized_path, (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode())
    object.__setattr__(record, "source_path", source_path)
    object.__setattr__(record, "normalized_path", normalized_path)
    return record


def _html_attribute(html: str, name: str) -> str | None:
    marker = f'data-{name}="'
    start = html.find(marker)
    if start < 0:
        return None
    start += len(marker)
    end = html.find('"', start)
    return None if end < 0 else html[start:end]


def _parse_iso_or_nse(value: object) -> date:
    text = str(value).strip()
    for pattern in ("%Y-%m-%d", "%d-%b-%Y"):
        try:
            return datetime.strptime(text, pattern).date()
        except ValueError:
            pass
    raise G8ReadinessError(f"invalid UDiFF date: {text!r}")
