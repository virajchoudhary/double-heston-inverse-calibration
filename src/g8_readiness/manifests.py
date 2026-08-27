"""Strict pre-acquisition, selected-data, and evidence manifest schemas."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..r2_representation.serialization import surface_to_payload
from ..r2_representation.surface import R2Surface
from .acquisition import NSEArchiveRecord, RbiRateRecord
from .contracts import BACKUP_SYMBOLS_BY_PRIMARY, DATE_FLOOR, PRIMARY_SYMBOLS, SCAN_END
from .scanner import BackupDecision, ScanResult


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def artifact_identity(path: Path | str) -> dict[str, Any]:
    value = Path(path)
    content = value.read_bytes()
    return {
        "path": str(value),
        "byte_size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def build_pre_acquisition_freeze(
    *,
    protocol_commit: str,
    config_path: Path | str,
    checkpoint_manifest: Mapping[str, Any],
    independent_review_verdict: str | None,
    model3_decision: Mapping[str, Any],
    tool_identities: Mapping[str, Mapping[str, Any]],
    current_date: date,
    protocol_frozen: bool,
    seal: bool = False,
) -> dict[str, Any]:
    """Never seal while any scientific, review, calendar, or identity prerequisite is missing."""
    checkpoints_ready = checkpoint_manifest.get('all_checks_passed') is True
    review_approved = independent_review_verdict == 'APPROVED'
    model3_label = model3_decision.get('label')
    model3_evidence_bound = (
        model3_label in {'MODEL3_NOT_FROZEN_NOT_EVALUATED', 'MODEL3_NOT_YET_ELIGIBLE_FOR_G8_INCLUSION'}
        or (
            model3_label == 'MODEL3_INCLUDED'
            and model3_decision.get('decision') == 'MODEL3_INCLUDED'
            and isinstance(model3_decision.get('checks'), Mapping)
            and all(model3_decision['checks'].values())
        )
    )
    protocol_identity_ok = (
        protocol_frozen is True
        and protocol_commit == '7eecc7188c54f9d4505d32ccf5c51069a4c3a97c'
    )
    config_identity_ok = artifact_identity(config_path)['sha256'] == 'd6107bf7c1b5404e59130d99b5e0f12aef4352c1452b83235187caa7628d4f37'
    date_floor_reached = current_date >= DATE_FLOOR
    required_tools = {'acquisition', 'surface_builder', 'evaluation_harness'}
    complete_tools = set(tool_identities) == required_tools and all(
        isinstance(identity, Mapping)
        and Path(str(identity.get('path', ''))).is_file()
        and artifact_identity(identity['path'])['sha256'] == identity.get('sha256')
        for identity in tool_identities.values()
    )
    ready = all((checkpoints_ready, review_approved, model3_evidence_bound, protocol_identity_ok, config_identity_ok, date_floor_reached, complete_tools))
    waiting: list[str] = []
    if not checkpoints_ready:
        waiting.append('WAITING_FOR_CHECKPOINT')
    if not review_approved:
        waiting.append('WAITING_FOR_INDEPENDENT_REVIEW')
    if not model3_evidence_bound:
        waiting.append('WAITING_FOR_MODEL3_FREEZE_DECISION')
    if not date_floor_reached:
        waiting.append('WAITING_FOR_DATE_FLOOR')
    if not protocol_identity_ok:
        waiting.append('WAITING_FOR_PROTOCOL_IDENTITY_VERIFICATION')
    if not config_identity_ok:
        waiting.append('WAITING_FOR_CONFIG_IDENTITY_VERIFICATION')
    if not complete_tools:
        waiting.append('WAITING_FOR_TOOL_IDENTITY_VERIFICATION')
    payload = {
        'schema_version': 'g8.pre_acquisition_freeze/1',
        'status': 'G8_PRE_ACQUISITION_FREEZE_READY' if ready else 'G8_READINESS_PREFLIGHT_NOT_SEALED',
        'sealed': bool(ready and seal),
        'protocol_commit': protocol_commit,
        'protocol_identity_verified': protocol_identity_ok,
        'config_identity_verified': config_identity_ok,
        'config': artifact_identity(config_path),
        'participating_inverse_methods': ['TRADITIONAL', 'MODEL1_ANN', 'MODEL2_CONSTRAINT_REPRICING_INFORMED'],
        'optional_model3': model3_decision,
        'model3_evidence_bound': model3_evidence_bound,
        'checkpoint_readiness': checkpoint_manifest,
        'independent_review_verdict': independent_review_verdict,
        'tool_identities': dict(tool_identities),
        'tool_identities_verified': complete_tools,
        'date_floor': DATE_FLOOR.isoformat(),
        'scan_end': SCAN_END.isoformat(),
        'date_floor_reached': date_floor_reached,
        'outstanding_prerequisites': waiting,
        'source_contract': 'OFFICIAL_NSE_UDIFF_BHAVCOPY_ONLY_PLUS_OFFICIAL_RBI_91D_TBILL',
        'real_market_data_acquired': False,
        'model_output_exists': False,
    }
    if payload['sealed']:
        payload['manifest_sha256'] = sha256_payload({**payload, 'manifest_sha256': ''})
    return payload


def _archive_identity(record: NSEArchiveRecord) -> dict[str, Any]:
    return {
        "market": record.market,
        "trading_date": record.trading_date,
        "official_url": record.official_url,
        "retrieval_timestamp_utc": record.retrieval_timestamp_utc,
        "original_filename": record.original_filename,
        "byte_size": record.byte_size,
        "zip_sha256": record.zip_sha256,
        "member_filename": record.member_filename,
        "csv_sha256": record.csv_sha256,
        "encoding": record.encoding,
        "delimiter": record.delimiter,
    }


def _rate_identity(record: RbiRateRecord) -> dict[str, Any]:
    return {
        "official_url": record.official_url,
        "release_identifier": record.release_identifier,
        "observation_date": record.observation_date,
        "cutoff_price": record.cutoff_price,
        "yield_percent": record.yield_percent,
        "source_sha256": record.source_sha256,
        "normalized_extract_sha256": record.normalized_extract_sha256,
    }


def build_selected_data_freeze(
    *,
    surfaces: Iterable[R2Surface],
    archive_records: Iterable[NSEArchiveRecord],
    rate_records: Iterable[RbiRateRecord],
    scan_result: ScanResult,
    backup_decisions: Iterable[BackupDecision] = (),
    observation_role_mappings: Mapping[str, list[int]],
    data_classification: str = "SYNTHETIC_G8_PIPELINE_FIXTURE",
    authorize_real_selected_data_seal: bool = False,
    protocol_commit: str = "7eecc7188c54f9d4505d32ccf5c51069a4c3a97c",
    config_sha256: str = "d6107bf7c1b5404e59130d99b5e0f12aef4352c1452b83235187caa7628d4f37",
    pre_acquisition_freeze_sha256: str | None = None,
    pre_acquisition_freeze: Mapping[str, Any] | None = None,
    primary_scan_result: ScanResult | None = None,
    scan_mode: str | None = None,
    search_window: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Second seal after deterministic structural selection and before models."""
    if protocol_commit != "7eecc7188c54f9d4505d32ccf5c51069a4c3a97c":
        raise ValueError("selected-data freeze protocol commit mismatch")
    if config_sha256 != "d6107bf7c1b5404e59130d99b5e0f12aef4352c1452b83235187caa7628d4f37":
        raise ValueError("selected-data freeze config SHA-256 mismatch")

    predecessor_hash = pre_acquisition_freeze_sha256 or ""
    if pre_acquisition_freeze is not None:
        if not isinstance(pre_acquisition_freeze, Mapping):
            raise ValueError("pre_acquisition_freeze must be a mapping")
        if pre_acquisition_freeze.get("schema_version") != "g8.pre_acquisition_freeze/1":
            raise ValueError("invalid pre-acquisition freeze schema version")
        if pre_acquisition_freeze.get("status") != "G8_PRE_ACQUISITION_FREEZE_READY":
            raise ValueError(f"pre-acquisition freeze status not ready: {pre_acquisition_freeze.get('status')}")
        predecessor_hash = (
            pre_acquisition_freeze.get("manifest_sha256")
            or sha256_payload({**pre_acquisition_freeze, "manifest_sha256": ""})
        )

    surface_list = list(surfaces)
    archives = [_archive_identity(record) for record in archive_records]
    rates = [_rate_identity(record) for record in rate_records]
    backups = [
        {
            "primary_symbol": item.primary_symbol,
            "backup_symbol": item.backup_symbol,
            "primary_support_count": item.primary_support_count,
            "trigger": item.trigger,
        }
        for item in backup_decisions
    ]

    effective_scan_mode = scan_mode or ("BACKUP_RESCAN" if backups else "PRIMARY_SCAN")
    if effective_scan_mode not in ("PRIMARY_SCAN", "BACKUP_RESCAN"):
        raise ValueError(f"unknown scan mode {effective_scan_mode}")

    if effective_scan_mode == "PRIMARY_SCAN":
        if backups:
            raise ValueError("primary scan freeze cannot contain backup decisions")
        if scan_result.scan_mode not in ("PRIMARY_SCAN", None):
            raise ValueError("scan_result mode mismatch for primary scan")
        backup_activation_reason = "NONE_PRIMARY_SCAN"
        primary_scan_provenance = None
    else:
        if not backups:
            raise ValueError("backup rescan freeze requires non-empty backup decisions")
        if any(b["primary_support_count"] != 0 or b["trigger"] != "PRIMARY_ZERO_ELIGIBLE_SURFACES_COMPLETE_WINDOW" for b in backups):
            raise ValueError("backup decisions require zero eligible surfaces across complete window")
        if primary_scan_result is not None:
            if primary_scan_result.scan_mode != "PRIMARY_SCAN":
                raise ValueError("primary_scan_result must have scan_mode == 'PRIMARY_SCAN'")
            if not primary_scan_result.complete_window_scanned:
                raise ValueError("primary scan must have covered the complete window")
            if primary_scan_result.reached_target:
                raise ValueError("primary scan reached target; backup rescan is forbidden")
            primary_scan_provenance = {
                "scan_mode": "PRIMARY_SCAN",
                "complete_window_scanned": True,
                "reached_target": False,
                "failure_count": len(primary_scan_result.failures),
                "failures": [
                    {
                        "valuation_date": f.valuation_date.isoformat(),
                        "symbol": f.symbol,
                        "reason": f.reason,
                    }
                    for f in primary_scan_result.failures
                ],
            }
        else:
            primary_scan_provenance = {
                "scan_mode": "PRIMARY_SCAN",
                "complete_window_scanned": True,
                "reached_target": False,
            }
        backup_activation_reason = "; ".join(
            f"{b['primary_symbol']}->{b['backup_symbol']}:{b['trigger']}"
            for b in backups
        )

    if scan_result.reached_target is not True:
        raise ValueError("selected-data freeze requires a completed deterministic scan that reached target")
    if len(scan_result.selected_dates) != 2:
        raise ValueError("selected-data freeze requires exactly two common dates")
    if len(surface_list) != 8:
        raise ValueError("selected-data freeze requires four surfaces on each of two dates")
    selected_dates = sorted({surface.metadata["valuation_date"] for surface in surface_list})
    if [value.isoformat() for value in scan_result.selected_dates] != selected_dates:
        raise ValueError("surface dates do not match deterministic scan result")
    identities = [(str(surface.metadata["valuation_date"]), str(surface.metadata["symbol"])) for surface in surface_list]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate symbol/date surface in selected-data freeze")
    date_counts = Counter(date_value for date_value, _symbol in identities)
    if set(date_counts.values()) != {4}:
        raise ValueError("selected-data freeze requires exactly four surfaces per common date")

    replacements = {item["primary_symbol"]: item["backup_symbol"] for item in backups}
    sector_backups = {"NTPC": "POWERGRID", "CIPLA": "SUNPHARMA", "INFY": "TCS", "HDFCBANK": "ICICIBANK"}
    if any(sector_backups.get(primary) != backup for primary, backup in replacements.items()):
        raise ValueError("invalid fixed-order backup replacement")
    expected_symbols = {
        sector_backups[primary] if primary in replacements else primary
        for primary in ("NTPC", "CIPLA", "INFY", "HDFCBANK")
    }
    expected_source = (
        "REAL_G8_OFFICIAL_NSE_R2"
        if data_classification == "REAL_G8_SELECTED_DATA"
        else "SYNTHETIC_G8_PIPELINE_FIXTURE_R2"
    )
    archive_list = list(archives)
    rate_list = list(rates)
    for selected_date in selected_dates:
        symbols_on_date = {symbol for date_value, symbol in identities if date_value == selected_date}
        if symbols_on_date != expected_symbols:
            raise ValueError(f"unexpected symbol composition on {selected_date}")
        date_archives = [record for record in archive_list if record["trading_date"] == selected_date]
        markets = sorted({record["market"] for record in date_archives})
        if markets != ["CM", "FO"]:
            raise ValueError(f"selected-data freeze requires CM and FO archives on {selected_date}")
        for archive in date_archives:
            if sum(item["market"] == archive["market"] for item in date_archives) != 1:
                raise ValueError(f"duplicate {archive['market']} archive for {selected_date}")
            expected_name = (
                f"BhavCopy_NSE_{archive['market']}_0_0_0_{selected_date.replace('-', '')}_F_0000.csv.zip"
            )
            if archive["original_filename"] != expected_name or not str(archive["official_url"]).endswith(expected_name):
                raise ValueError(f"official archive identity mismatch: {expected_name}")
        surface_rate_ids = {
            str(surface.metadata.get("rate_record", {}).get("release_identifier"))
            for surface in surface_list
            if surface.metadata["valuation_date"] == selected_date
        }
        eligible_rates = [
            rate for rate in rate_list
            if date.fromisoformat(rate["observation_date"]) <= date.fromisoformat(selected_date)
        ]
        rate_identities = Counter(
            (rate["release_identifier"], rate["observation_date"], rate["source_sha256"], rate["normalized_extract_sha256"])
            for rate in rate_list
        )
        if any(count != 1 for count in rate_identities.values()):
            raise ValueError("conflicting duplicate RBI release records")
        if not eligible_rates:
            raise ValueError(f"missing eligible RBI observation for {selected_date}")
        latest_rate = max(eligible_rates, key=lambda rate: rate["observation_date"])
        if surface_rate_ids != {latest_rate["release_identifier"]}:
            raise ValueError(f"surface RBI chronology mismatch on {selected_date}")
        for surface in surface_list:
            if surface.metadata["valuation_date"] != selected_date:
                continue
            if surface.metadata.get("data_classification") != data_classification:
                raise ValueError(f"surface classification does not match seal for {surface.surface_id}")
            if surface.source != expected_source:
                raise ValueError(f"surface source does not match seal for {surface.surface_id}")

    surface_hashes = []
    for surface in surface_list:
        payload = surface_to_payload(surface)
        surface_hashes.append(
            {
                "surface_id": surface.surface_id,
                "payload_sha256": sha256_payload(payload),
                "mask_sha256": hashlib.sha256(
                    json.dumps([bool(value) for value in surface.mask], separators=(",", ":")).encode()
                ).hexdigest(),
            }
        )
    if data_classification == "SYNTHETIC_G8_PIPELINE_FIXTURE":
        status = "SYNTHETIC_G8_SELECTED_DATA_FIXTURE_SEALED"
        not_real_market_data = True
    elif (
        data_classification == "REAL_G8_SELECTED_DATA"
        and authorize_real_selected_data_seal is True
    ):
        status = "REAL_G8_SELECTED_DATA_FROZEN"
        not_real_market_data = False
    else:
        raise ValueError("real selected-data seal requires explicit separate authorization")

    window = search_window or {"search_start": DATE_FLOOR.isoformat(), "search_end": SCAN_END.isoformat()}
    payload = {
        "schema_version": "g8.selected_data_freeze/1",
        "classification": data_classification,
        "status": status,
        "scan_mode": effective_scan_mode,
        "protocol_commit": protocol_commit,
        "config_sha256": config_sha256,
        "pre_acquisition_freeze_sha256": predecessor_hash,
        "search_window": dict(window),
        "not_real_market_data": not_real_market_data,
        "selected_dates": selected_dates,
        "selected_symbols": sorted({str(surface.metadata["symbol"]) for surface in surface_list}),
        "primary_symbols": list(PRIMARY_SYMBOLS),
        "active_symbols": sorted(expected_symbols),
        "raw_archive_hashes": archives,
        "rate_hashes": rates,
        "surface_hashes": surface_hashes,
        "complete_scan_failure_log": [
            {
                "valuation_date": item.valuation_date.isoformat(),
                "symbol": item.symbol,
                "reason": item.reason,
            }
            for item in scan_result.failures
        ],
        "scan_reached_target": scan_result.reached_target,
        "complete_window_scanned_for_backup_policy": (
            True if effective_scan_mode == "BACKUP_RESCAN" else scan_result.complete_window_scanned
        ),
        "backup_decisions": backups,
        "backup_activation_reason": backup_activation_reason,
        "primary_scan_provenance": primary_scan_provenance,
        "backup_policy_resolution": (
            "COMPLETE_WINDOW_BACKUP_SCAN_COMPLETE"
            if effective_scan_mode == "BACKUP_RESCAN"
            else "NOT_REQUIRED_TWO_COMMON_DATES_REACHED_WITHOUT_SUBSTITUTION"
        ),
        "observation_role_mappings": {key: list(value) for key, value in observation_role_mappings.items()},
        "model_prediction_or_calibration_present": False,
    }
    payload["manifest_sha256"] = sha256_payload({**payload, "manifest_sha256": ""})
    return payload


EVIDENCE_SCHEMAS: dict[str, tuple[str, ...]] = {
    "raw_acquisition": (
        "official_url", "retrieval_timestamp_utc", "original_filename", "byte_size",
        "zip_sha256", "zip_integrity_result", "member_filename", "csv_sha256",
        "encoding", "delimiter", "trading_date",
    ),
    "rate_acquisition": (
        "official_url", "release_identifier", "observation_date", "cutoff_price",
        "yield_percent", "source_sha256", "normalized_extract_sha256",
    ),
    "scan_log": ("selected_dates", "failures", "backup_decisions"),
    "surface_construction": ("surface_id", "support", "imputation_or_interpolation"),
    "pre_model_selected_data_freeze": ("surface_hashes", "raw_archive_hashes", "rate_hashes"),
    "pricing_family_run": ("family", "surfaces", "metrics", "winner_label"),
    "inverse_method_run": ("method", "surfaces", "metrics", "parameter_truth"),
    "final_aggregation": ("pricing_family", "inverse_methods", "partial_status"),
}


def validate_evidence_schema(kind: str, payload: Mapping[str, Any], *, partial: bool = False) -> None:
    if kind not in EVIDENCE_SCHEMAS:
        raise ValueError(f"unknown evidence schema {kind}")
    missing = [field for field in EVIDENCE_SCHEMAS[kind] if field not in payload]
    if missing:
        raise ValueError(f"{kind} evidence missing fields: {missing}")
    if partial and str(payload.get("status", "")).upper() != "PARTIAL":
        raise ValueError("partial evidence must carry status PARTIAL")
