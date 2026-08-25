"""Strict pre-acquisition, selected-data, and evidence manifest schemas."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..r2_representation.serialization import surface_to_payload
from ..r2_representation.surface import R2Surface
from .acquisition import NSEArchiveRecord, RbiRateRecord
from .contracts import DATE_FLOOR, SCAN_END
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
    model3_decision: Mapping[str, Any],
    tool_identities: Mapping[str, Mapping[str, Any]],
    seal: bool = False,
) -> dict[str, Any]:
    """Never return a sealed manifest while checkpoint prerequisites are absent."""
    checkpoints_ready = checkpoint_manifest.get("all_checks_passed") is True
    model3_label = model3_decision.get("label")
    model3_valid_label = model3_label in {
        "MODEL3_NOT_FROZEN_NOT_EVALUATED",
        "MODEL3_INCLUDED",
    }
    missing_tools = sorted(set(("acquisition", "surface_builder", "evaluation_harness")) - set(tool_identities))
    ready = checkpoints_ready and model3_valid_label and not missing_tools
    payload = {
        "schema_version": "g8.pre_acquisition_freeze/1",
        "status": (
            "G8_PRE_ACQUISITION_FREEZE_READY" if ready else "G8_READINESS_PREFLIGHT_NOT_SEALED"
        ),
        "sealed": bool(ready and seal),
        "protocol_commit": protocol_commit,
        "config": artifact_identity(config_path),
        "participating_inverse_methods": [
            "TRADITIONAL",
            "MODEL1_ANN",
            "MODEL2_CONSTRAINT_REPRICING_INFORMED",
        ],
        "optional_model3": model3_decision,
        "checkpoint_readiness": checkpoint_manifest,
        "tool_identities": dict(tool_identities),
        "date_floor": DATE_FLOOR.isoformat(),
        "scan_end": SCAN_END.isoformat(),
        "source_contract": "OFFICIAL_NSE_UDIFF_BHAVCOPY_ONLY_PLUS_OFFICIAL_RBI_91D_TBILL",
        "real_market_data_acquired": False,
        "model_output_exists": False,
    }
    if payload["sealed"]:
        payload["manifest_sha256"] = sha256_payload({**payload, "manifest_sha256": ""})
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
    backup_decisions: Iterable[BackupDecision],
    observation_role_mappings: Mapping[str, list[int]],
    data_classification: str = "SYNTHETIC_G8_PIPELINE_FIXTURE",
    authorize_real_selected_data_seal: bool = False,
) -> dict[str, Any]:
    """Second seal after deterministic structural selection and before models."""
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
    if not surface_list or len(surface_list) > 8:
        raise ValueError("selected-data freeze requires one through eight surfaces")
    selected_dates = sorted({surface.metadata["valuation_date"] for surface in surface_list})
    if [value.isoformat() for value in scan_result.selected_dates] != selected_dates:
        raise ValueError("surface dates do not match deterministic scan result")
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
    payload = {
        "schema_version": "g8.selected_data_freeze/1",
        "classification": data_classification,
        "status": status,
        "not_real_market_data": not_real_market_data,
        "selected_dates": selected_dates,
        "selected_symbols": sorted({str(surface.metadata["symbol"]) for surface in surface_list}),
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
        "complete_window_scanned_for_backup_policy": scan_result.complete_window_scanned,
        "backup_decisions": backups,
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
