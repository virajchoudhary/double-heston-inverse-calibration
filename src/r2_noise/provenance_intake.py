"""Fail-closed newline-variant intake for the frozen Traditional subset.

The frozen experiment pins the original CRLF artifact byte-for-byte.  One
historical worktree serialized the identical JSON payload with LF endings;
Git's ``* text=auto eol=lf`` policy then committed that normalized form.
Only those two exact byte identities are eligible for intake.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .subset import select_traditional_subset

CANONICAL_SUBSET_SHA256 = (
    "f856ef5ffcc33782a115180ca7cb7b1f4cfa4ebeb8fd1af45c7cde242c85aba7"
)
HISTORICAL_LF_SUBSET_SHA256 = (
    "db3a4917dcbf4b26a5e9bfa8103b8a3beffe439c8ad6f232330908f615d18880"
)
INTAKE_VERSION = "r2-traditional-subset-intake-v1"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLEAN_DATASET_PATH = (
    REPO_ROOT / "data" / "final_r2_clean_10000" / "surfaces.jsonl"
)
DEFAULT_SUBSET_PATH = (
    REPO_ROOT
    / "evidence"
    / "r2_noise_robustness"
    / "traditional_subset_ids.json"
)
DEFAULT_REPAIR_DIR = DEFAULT_SUBSET_PATH.parent / "provenance_repair"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _byte_identity(data: bytes) -> dict[str, Any]:
    crlf_count = data.count(b"\r\n")
    lf_count = data.count(b"\n")
    return {
        "byte_size": len(data),
        "sha256": _sha256(data),
        "crlf_count": crlf_count,
        "lf_count": lf_count,
        "line_ending_form": (
            "CRLF" if lf_count == crlf_count else "LF" if crlf_count == 0 else "mixed"
        ),
        "has_final_trailing_newline": data.endswith(b"\n"),
        "has_utf8_bom": data.startswith(b"\xef\xbb\xbf"),
    }


def _load_json_object(data: bytes) -> dict[str, Any]:
    if data.startswith(b"\xef\xbb\xbf"):
        raise ValueError("subset intake rejected UTF-8 BOM")
    payload = json.loads(data.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("subset intake rejected non-object JSON")
    return payload


def canonical_record_digest(payload: Mapping[str, Any]) -> str:
    """Digest the parsed scientific payload independent of newline encoding."""
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256(encoded + b"\n")


def assert_regenerated_selection(
    payload: Mapping[str, Any],
    *,
    clean_dataset_path: str | Path,
) -> None:
    """Require every selector-produced field to equal the frozen clean truth."""
    records: list[dict[str, Any]] = []
    with Path(clean_dataset_path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            if record["metadata"]["user_metadata"]["split"] != "test":
                continue
            records.append(
                {
                    "surface_id": record["surface_id"],
                    "parameters_canonical_order": record["metadata"][
                        "parameters_canonical_order"
                    ],
                }
            )
    if len(records) != 1250:
        raise ValueError(
            f"clean test population changed: found {len(records)}, expected 1250"
        )
    regenerated = select_traditional_subset(records)
    differences = [key for key in regenerated if payload.get(key) != regenerated[key]]
    if differences:
        raise ValueError(f"subset scientific payload differs: {differences}")


def canonicalize_known_subset_representation(
    data: bytes,
    *,
    clean_dataset_path: str | Path = DEFAULT_CLEAN_DATASET_PATH,
) -> tuple[bytes, dict[str, Any]]:
    """Canonicalize only the exact canonical or known historical byte form."""
    input_identity = _byte_identity(data)
    if input_identity["sha256"] == CANONICAL_SUBSET_SHA256:
        if input_identity["line_ending_form"] != "CRLF":
            raise ValueError("canonical hash collision rejected: form is not CRLF")
        payload = _load_json_object(data)
        assert_regenerated_selection(payload, clean_dataset_path=clean_dataset_path)
        return data, {
            "input": input_identity,
            "transformation": "none_exact_canonical",
            "output": input_identity,
            "semantic_payload_sha256": canonical_record_digest(payload),
        }

    if input_identity["sha256"] != HISTORICAL_LF_SUBSET_SHA256:
        raise ValueError(
            "subset intake accepts only canonical "
            f"{CANONICAL_SUBSET_SHA256} or historical "
            f"{HISTORICAL_LF_SUBSET_SHA256}; got {input_identity['sha256']}"
        )
    if input_identity["line_ending_form"] != "LF":
        raise ValueError("known historical variant must be LF-only")
    if input_identity["has_utf8_bom"]:
        raise ValueError("known historical variant must not contain a UTF-8 BOM")

    payload = _load_json_object(data)
    assert_regenerated_selection(payload, clean_dataset_path=clean_dataset_path)
    canonical = data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    output_identity = _byte_identity(canonical)
    if output_identity["sha256"] != CANONICAL_SUBSET_SHA256:
        raise ValueError("LF-to-CRLF reconstruction did not produce frozen identity")
    if canonical_record_digest(payload) != canonical_record_digest(
        _load_json_object(canonical)
    ):
        raise ValueError("newline reconstruction changed the parsed payload")
    return canonical, {
        "input": input_identity,
        "transformation": "replace_each_LF_with_CRLF_after_removing_any_CRLF",
        "output": output_identity,
        "semantic_payload_sha256": canonical_record_digest(payload),
    }


def _validate_existing_intake_evidence(
    record: Mapping[str, Any],
    *,
    input_identity: Mapping[str, Any],
    output_identity: Mapping[str, Any],
) -> None:
    expected_pairs = (
        ("candidate_sha256", input_identity["sha256"]),
        ("canonical_sha256", output_identity["sha256"]),
        ("transformation", "replace_each_LF_with_CRLF_after_removing_any_CRLF"),
        ("repair_status", "CANONICALIZED"),
    )
    mismatches = [
        f"{key}: {record.get(key)!r} != {expected!r}"
        for key, expected in expected_pairs
        if record.get(key) != expected
    ]
    if mismatches:
        raise ValueError(f"existing subset intake record mismatch: {mismatches}")


def ensure_canonical_subset(
    subset_path: str | Path = DEFAULT_SUBSET_PATH,
    *,
    clean_dataset_path: str | Path = DEFAULT_CLEAN_DATASET_PATH,
    repair_dir: str | Path = DEFAULT_REPAIR_DIR,
) -> dict[str, Any]:
    """Materialize the frozen CRLF artifact, preserving any historical input."""
    target = Path(subset_path)
    evidence_dir = Path(repair_dir)
    backup_path = evidence_dir / "original_candidate_traditional_subset_ids.json"
    record_path = evidence_dir / "TRADITIONAL_SUBSET_INTAKE_RECORD.json"
    data = target.read_bytes()
    canonical, proof = canonicalize_known_subset_representation(
        data, clean_dataset_path=clean_dataset_path
    )
    if proof["transformation"] == "none_exact_canonical":
        return {"status": "ALREADY_CANONICAL", **proof}

    evidence_dir.mkdir(parents=True, exist_ok=True)
    if backup_path.exists():
        backup_data = backup_path.read_bytes()
        if _sha256(backup_data) != proof["input"]["sha256"]:
            raise ValueError("existing historical subset backup identity mismatch")
    else:
        backup_path.write_bytes(data)

    record_exists = record_path.exists()
    record: dict[str, Any] = {}
    if record_exists:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        _validate_existing_intake_evidence(
            record,
            input_identity=proof["input"],
            output_identity=proof["output"],
        )
    else:
        record = {
            "artifact_kind": "R2_NOISE_TRADITIONAL_SUBSET_PROVENANCE_INTAKE",
            "canonical_sha256": CANONICAL_SUBSET_SHA256,
            "candidate_sha256": HISTORICAL_LF_SUBSET_SHA256,
            "candidate_source_path": target.resolve().as_posix(),
            "candidate_timestamp_utc": datetime.fromtimestamp(
                target.stat().st_mtime, timezone.utc
            ).isoformat(),
            "frozen_expected_hash_preserved": True,
            "intake_version": INTAKE_VERSION,
            "original_backup_path": backup_path.resolve().as_posix(),
            "proof": proof,
            "repair_status": "CANONICALIZED",
            "scientific_fields_changed": False,
            "tool": "src/r2_noise/provenance_intake.py",
            "transformation": proof["transformation"],
        }
        record_path.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    temporary_path = target.with_name(target.name + ".intake.tmp")
    try:
        temporary_path.write_bytes(canonical)
        os.replace(temporary_path, target)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    if _sha256(target.read_bytes()) != CANONICAL_SUBSET_SHA256:
        raise RuntimeError("post-materialization canonical subset identity failed")
    return {
        "status": "MATERIALIZED_FROM_KNOWN_HISTORICAL_VARIANT",
        "backup_path": backup_path.resolve().as_posix(),
        "record_path": record_path.resolve().as_posix(),
        **proof,
    }


if __name__ == "__main__":
    print(json.dumps(ensure_canonical_subset(), indent=2, sort_keys=True))
