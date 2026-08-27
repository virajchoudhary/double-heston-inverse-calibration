"""Future official-source acquisition interfaces with hard current locks."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from ..nse_stage_a import UDIFF_COLUMNS
from .contracts import DATE_FLOOR, SCAN_END, G8ReadinessError, validate_g8_valuation_date

OFFICIAL_NSE_ARCHIVE_ROOT = "https://nsearchives.nseindia.com/content"
OFFICIAL_RBI_ROOT = "https://www.rbi.org.in/"
FROZEN_PROTOCOL_COMMIT = "7eecc7188c54f9d4505d32ccf5c51069a4c3a97c"
FROZEN_CONFIG_SHA256 = "d6107bf7c1b5404e59130d99b5e0f12aef4352c1452b83235187caa7628d4f37"
REQUIRED_TOOLS = {"acquisition", "surface_builder", "evaluation_harness"}


class G8AcquisitionLocked(G8ReadinessError, RuntimeError):
    """Raised unless the caller supplies an explicit future authorization flag."""


class CurrentDateAcquisitionLocked(G8AcquisitionLocked):
    """Raised when calendar floor 2026-09-30 has not been reached or future valuation is requested."""


def verify_acquisition_gate(
    *,
    valuation_date: date | str,
    authorize_acquisition: bool,
    current_date: date | None = None,
    protocol_commit: str = FROZEN_PROTOCOL_COMMIT,
    config_path: Path | str = Path("configs/g8_final_real_market.yaml"),
    pre_acquisition_freeze: Mapping[str, Any] | None = None,
    independent_review_verdict: str | None = None,
    checkpoint_manifest: Mapping[str, Any] | None = None,
    model3_decision: Mapping[str, Any] | None = None,
    tool_identities: Mapping[str, Mapping[str, Any]] | None = None,
    protocol_frozen: bool = True,
) -> date:
    """Authoritative fail-closed acquisition gate enforcing all required preconditions."""
    if authorize_acquisition is not True:
        raise G8AcquisitionLocked(
            "default invocation cannot acquire G8 market data; explicit future authorization is required"
        )
    val_date = validate_g8_valuation_date(valuation_date)
    today = current_date or date.today()
    if today < DATE_FLOOR:
        raise CurrentDateAcquisitionLocked(
            f"calendar blocker remains: current date {today.isoformat()} precedes {DATE_FLOOR.isoformat()}"
        )
    if val_date > today:
        raise CurrentDateAcquisitionLocked(
            f"future valuation evidence is unavailable: requested {val_date.isoformat()}, current {today.isoformat()}"
        )

    if pre_acquisition_freeze is not None:
        if not isinstance(pre_acquisition_freeze, Mapping):
            raise G8AcquisitionLocked("pre_acquisition_freeze must be a mapping")
        if pre_acquisition_freeze.get("schema_version") != "g8.pre_acquisition_freeze/1":
            raise G8AcquisitionLocked("invalid pre-acquisition freeze schema version")
        if pre_acquisition_freeze.get("status") != "G8_PRE_ACQUISITION_FREEZE_READY":
            raise G8AcquisitionLocked(
                f"pre-acquisition freeze status is not ready: {pre_acquisition_freeze.get('status')}"
            )
        if pre_acquisition_freeze.get("protocol_commit") != FROZEN_PROTOCOL_COMMIT:
            raise G8AcquisitionLocked("pre-acquisition freeze protocol commit mismatch")
        if pre_acquisition_freeze.get("protocol_identity_verified") is not True:
            raise G8AcquisitionLocked("pre-acquisition freeze protocol identity not verified")
        if pre_acquisition_freeze.get("config_identity_verified") is not True:
            raise G8AcquisitionLocked("pre-acquisition freeze config identity not verified")
        if pre_acquisition_freeze.get("config", {}).get("sha256") != FROZEN_CONFIG_SHA256:
            raise G8AcquisitionLocked("pre-acquisition freeze config hash mismatch")
        if pre_acquisition_freeze.get("checkpoint_readiness", {}).get("all_checks_passed") is not True:
            raise G8AcquisitionLocked("pre-acquisition freeze checkpoint gate not passed")
        if pre_acquisition_freeze.get("independent_review_verdict") != "APPROVED":
            raise G8AcquisitionLocked("pre-acquisition freeze independent review not approved")
        if pre_acquisition_freeze.get("model3_evidence_bound") is not True:
            raise G8AcquisitionLocked("pre-acquisition freeze Model3 evidence not bound")
        if pre_acquisition_freeze.get("tool_identities_verified") is not True:
            raise G8AcquisitionLocked("pre-acquisition freeze tool identities not verified")
        if pre_acquisition_freeze.get("sealed"):
            from .manifests import sha256_payload
            computed_hash = sha256_payload({**pre_acquisition_freeze, "manifest_sha256": ""})
            if pre_acquisition_freeze.get("manifest_sha256") != computed_hash:
                raise G8AcquisitionLocked("pre-acquisition freeze manifest SHA-256 mismatch")
        return val_date

    if protocol_commit != FROZEN_PROTOCOL_COMMIT or protocol_frozen is not True:
        raise G8AcquisitionLocked(
            f"protocol identity mismatch: expected {FROZEN_PROTOCOL_COMMIT}, got {protocol_commit}"
        )
    cfg_file = Path(config_path)
    if not cfg_file.is_file():
        raise G8AcquisitionLocked(f"G8 config file missing: {cfg_file}")
    from .manifests import artifact_identity
    cfg_hash = artifact_identity(cfg_file)["sha256"]
    if cfg_hash != FROZEN_CONFIG_SHA256:
        raise G8AcquisitionLocked(
            f"config hash mismatch: expected {FROZEN_CONFIG_SHA256}, got {cfg_hash}"
        )
    from .checkpoints import checkpoint_readiness_manifest
    checkpoints = checkpoint_manifest or checkpoint_readiness_manifest(config_path=cfg_file)
    if (
        checkpoints.get("all_checks_passed") is not True
        or checkpoints.get("overall_status") != "CHECKPOINT_ARTIFACTS_STAGED_AND_VERIFIED"
    ):
        raise G8AcquisitionLocked("checkpoint readiness gate failed: 6 canonical checkpoints must pass")
    if independent_review_verdict != "APPROVED":
        raise G8AcquisitionLocked(
            f"independent review verdict must be 'APPROVED', got {independent_review_verdict!r}"
        )
    from .model3 import evaluate_model3_inclusion
    m3_dec = model3_decision or evaluate_model3_inclusion(None, acquisition_has_begun=False)
    m3_label = m3_dec.get("label")
    m3_ok = (
        m3_label in {"MODEL3_NOT_FROZEN_NOT_EVALUATED", "MODEL3_NOT_YET_ELIGIBLE_FOR_G8_INCLUSION"}
        or (
            m3_label == "MODEL3_INCLUDED"
            and m3_dec.get("decision") == "MODEL3_INCLUDED"
            and isinstance(m3_dec.get("checks"), Mapping)
            and all(m3_dec["checks"].values())
        )
    )
    if not m3_ok:
        raise G8AcquisitionLocked(
            f"model3 participation decision incomplete or unbound: label={m3_label}"
        )
    if tool_identities is not None:
        tools_ok = set(tool_identities) == REQUIRED_TOOLS and all(
            isinstance(ident, Mapping)
            and Path(str(ident.get("path", ""))).is_file()
            and artifact_identity(ident["path"])["sha256"] == ident.get("sha256")
            for ident in tool_identities.values()
        )
        if not tools_ok:
            raise G8AcquisitionLocked("tool identity verification failed")
    return val_date


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
    if target.exists():
        try:
            stored = json.loads(target.read_text(encoding="utf-8"))
        except Exception as exc:
            raise G8ReadinessError(f"invalid retained failed-archive record: {target}") from exc
        if stored.get("archive_sha256") != payload["archive_sha256"] or stored.get("failure_reason") != reason:
            raise G8ReadinessError(f"conflicting retained failed-archive record: {target}")
    else:
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
    current_date: date | None = None,
    protocol_commit: str = FROZEN_PROTOCOL_COMMIT,
    config_path: Path | str = Path("configs/g8_final_real_market.yaml"),
    pre_acquisition_freeze: Mapping[str, Any] | None = None,
    independent_review_verdict: str | None = None,
    checkpoint_manifest: Mapping[str, Any] | None = None,
    model3_decision: Mapping[str, Any] | None = None,
    tool_identities: Mapping[str, Mapping[str, Any]] | None = None,
    protocol_frozen: bool = True,
) -> NSEArchiveRecord:
    """Acquire one authorized CM/FO archive; enforces the central fail-closed acquisition gate first."""
    verify_acquisition_gate(
        valuation_date=valuation_date,
        authorize_acquisition=authorize_acquisition,
        current_date=current_date,
        protocol_commit=protocol_commit,
        config_path=config_path,
        pre_acquisition_freeze=pre_acquisition_freeze,
        independent_review_verdict=independent_review_verdict,
        checkpoint_manifest=checkpoint_manifest,
        model3_decision=model3_decision,
        tool_identities=tool_identities,
        protocol_frozen=protocol_frozen,
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
    current_date: date | None = None,
    protocol_commit: str = FROZEN_PROTOCOL_COMMIT,
    config_path: Path | str = Path("configs/g8_final_real_market.yaml"),
    pre_acquisition_freeze: Mapping[str, Any] | None = None,
    independent_review_verdict: str | None = None,
    checkpoint_manifest: Mapping[str, Any] | None = None,
    model3_decision: Mapping[str, Any] | None = None,
    tool_identities: Mapping[str, Mapping[str, Any]] | None = None,
    protocol_frozen: bool = True,
) -> RbiRateRecord:
    verify_acquisition_gate(
        valuation_date=latest_permitted_observation_date,
        authorize_acquisition=authorize_acquisition,
        current_date=current_date,
        protocol_commit=protocol_commit,
        config_path=config_path,
        pre_acquisition_freeze=pre_acquisition_freeze,
        independent_review_verdict=independent_review_verdict,
        checkpoint_manifest=checkpoint_manifest,
        model3_decision=model3_decision,
        tool_identities=tool_identities,
        protocol_frozen=protocol_frozen,
    )
    record = normalize_rbi_auction(
        html,
        official_url=official_url,
        latest_permitted_observation_date=latest_permitted_observation_date,
    )
    if re.fullmatch(r"[A-Za-z0-9._-]+", record.release_identifier) is None:
        raise G8ReadinessError("RBI release identifier contains unsafe path characters")
    directory = Path(store_root) / "official_rbi"
    source_path = directory / f"{record.release_identifier}.html"
    normalized_path = directory / f"{record.release_identifier}.normalized.json"
    if source_path.exists():
        existing_hash = _sha256_bytes(source_path.read_bytes())
        if existing_hash != record.source_sha256:
            raise G8AcquisitionLocked(f"immutable RBI source conflict: {source_path}")
    else:
        _atomic_new(source_path, html.encode())
    normalized_payload = {
        "official_url": record.official_url,
        "release_identifier": record.release_identifier,
        "observation_date": record.observation_date,
        "cutoff_price": record.cutoff_price,
        "yield_percent": record.yield_percent,
        "source_sha256": record.source_sha256,
        "normalized_extract_sha256": record.normalized_extract_sha256,
        "latest_permitted_observation_date": str(latest_permitted_observation_date),
    }
    if normalized_path.exists():
        try:
            stored = json.loads(normalized_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise G8ReadinessError(f"invalid immutable RBI normalized artifact: {normalized_path}") from exc
        for key in ("official_url", "release_identifier", "observation_date", "cutoff_price", "yield_percent", "source_sha256", "normalized_extract_sha256"):
            if key not in ("cutoff_price", "yield_percent") and stored.get(key) != normalized_payload[key]:
                raise G8ReadinessError(f"immutable RBI normalized conflict: {key}")
            if key in ("cutoff_price", "yield_percent") and float(stored.get(key)) != normalized_payload[key]:
                raise G8ReadinessError(f"immutable RBI normalized conflict: {key}")
    else:
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
