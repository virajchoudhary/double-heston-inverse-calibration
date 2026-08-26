"""Fail-closed tests for the frozen Traditional-subset provenance intake."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.r2_noise.provenance_intake import (
    CANONICAL_SUBSET_SHA256,
    HISTORICAL_LF_SUBSET_SHA256,
    canonical_record_digest,
    canonicalize_known_subset_representation,
    ensure_canonical_subset,
)
from src.r2_noise.traditional_runner import (
    _assert_resume_provenance_compatible,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CLEAN_DATASET_PATH = REPO_ROOT / "data" / "final_r2_clean_10000" / "surfaces.jsonl"
SUBSET_PATH = (
    REPO_ROOT / "evidence" / "r2_noise_robustness" / "traditional_subset_ids.json"
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_bytes() -> bytes:
    data = SUBSET_PATH.read_bytes()
    if _sha256(data) != CANONICAL_SUBSET_SHA256:
        data = data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    assert _sha256(data) == CANONICAL_SUBSET_SHA256
    return data


def _historical_bytes() -> bytes:
    data = _canonical_bytes().replace(b"\r\n", b"\n")
    assert _sha256(data) == HISTORICAL_LF_SUBSET_SHA256
    return data


def _changed_payload_bytes() -> bytes:
    payload = json.loads(_historical_bytes())
    payload["subset_size"] = 249
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8") + b"\n"


def _reordered_payload_bytes() -> bytes:
    payload = json.loads(_historical_bytes())
    payload["selected_ids"] = list(reversed(payload["selected_ids"]))
    payload["selected_detail"] = list(reversed(payload["selected_detail"]))
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8") + b"\n"


def _missing_row_payload_bytes() -> bytes:
    payload = json.loads(_historical_bytes())
    payload["subset_size"] -= 1
    payload["selected_ids"].pop()
    payload["selected_detail"].pop()
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8") + b"\n"


def _extra_row_payload_bytes() -> bytes:
    payload = json.loads(_historical_bytes())
    extra = dict(payload["selected_detail"][-1])
    extra["surface_id"] += "_EXTRA"
    payload["subset_size"] += 1
    payload["selected_ids"].append(extra["surface_id"])
    payload["selected_detail"].append(extra)
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8") + b"\n"


def test_exact_canonical_crlf_artifact_passes() -> None:
    canonical, proof = canonicalize_known_subset_representation(
        _canonical_bytes(), clean_dataset_path=CLEAN_DATASET_PATH
    )
    assert proof["transformation"] == "none_exact_canonical"
    assert _sha256(canonical) == CANONICAL_SUBSET_SHA256


def test_known_lf_variant_recognized_and_reconstructed_exactly() -> None:
    historical = _historical_bytes()
    canonical, proof = canonicalize_known_subset_representation(
        historical, clean_dataset_path=CLEAN_DATASET_PATH
    )
    assert proof["transformation"] == (
        "replace_each_LF_with_CRLF_after_removing_any_CRLF"
    )
    assert proof["input"]["sha256"] == HISTORICAL_LF_SUBSET_SHA256
    assert proof["output"]["sha256"] == CANONICAL_SUBSET_SHA256
    assert canonical_record_digest(json.loads(historical)) == (
        canonical_record_digest(json.loads(canonical))
    )


def test_unknown_or_changed_subsets_fail_closed() -> None:
    cases = [
        _changed_payload_bytes(),
        _reordered_payload_bytes(),
        _missing_row_payload_bytes(),
        _extra_row_payload_bytes(),
        b"not-json",
        b"\xef\xbb\xbf" + _historical_bytes(),
        _historical_bytes()[:-1],
        _historical_bytes() + b"\n",
    ]
    for data in cases:
        with pytest.raises((ValueError, UnicodeDecodeError)):
            canonicalize_known_subset_representation(
                data, clean_dataset_path=CLEAN_DATASET_PATH
            )


def test_transformation_that_cannot_reach_frozen_hash_fails() -> None:
    payload = json.loads(_historical_bytes())
    # Preserve every scientific field while changing nonsemantic whitespace;
    # this is still rejected because it is not the known historical identity.
    unrelated = json.dumps(payload, indent=4, sort_keys=True).encode("utf-8") + b"\n"
    assert _sha256(unrelated) not in {CANONICAL_SUBSET_SHA256, HISTORICAL_LF_SUBSET_SHA256}
    with pytest.raises(ValueError):
        canonicalize_known_subset_representation(
            unrelated, clean_dataset_path=CLEAN_DATASET_PATH
        )


def test_materialization_preserves_candidate_and_writes_intake_record() -> None:
    base = REPO_ROOT / ".tmp_intake_test"
    if base.exists():
        raise RuntimeError(f"unexpected stale test directory: {base}")
    base.mkdir()
    target = base / "traditional_subset_ids.json"
    repair_dir = base / "provenance_repair"
    target.write_bytes(_historical_bytes())

    first = ensure_canonical_subset(
        target,
        clean_dataset_path=CLEAN_DATASET_PATH,
        repair_dir=repair_dir,
    )
    assert first["status"] == "MATERIALIZED_FROM_KNOWN_HISTORICAL_VARIANT"
    assert _sha256(target.read_bytes()) == CANONICAL_SUBSET_SHA256
    backup = repair_dir / "original_candidate_traditional_subset_ids.json"
    record_path = repair_dir / "TRADITIONAL_SUBSET_INTAKE_RECORD.json"
    assert _sha256(backup.read_bytes()) == HISTORICAL_LF_SUBSET_SHA256
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["candidate_sha256"] == HISTORICAL_LF_SUBSET_SHA256
    assert record["canonical_sha256"] == CANONICAL_SUBSET_SHA256
    assert record["scientific_fields_changed"] is False
    assert record["frozen_expected_hash_preserved"] is True

    second = ensure_canonical_subset(
        target,
        clean_dataset_path=CLEAN_DATASET_PATH,
        repair_dir=repair_dir,
    )
    assert second["status"] == "ALREADY_CANONICAL"

    record_path.unlink()
    backup.unlink()
    target.unlink()
    repair_dir.rmdir()
    base.rmdir()


def test_resume_gate_accepts_only_documented_worktree_relocation() -> None:
    stored = {
        "source_path": "C:/ann_inverse_calibration/data/r2_noise_robustness/levels/0.10%/noisy_surfaces.jsonl",
        "source_sha256": "same",
    }
    current = {
        "source_path": "C:/dh_noise_overnight/data/r2_noise_robustness/levels/0.10%/noisy_surfaces.jsonl",
        "source_sha256": "same",
    }
    _assert_resume_provenance_compatible(stored, current)

    relocated_wrong_file = dict(stored)
    relocated_wrong_file["source_path"] = (
        "C:/ann_inverse_calibration/data/r2_noise_robustness/levels/0.25%/noisy_surfaces.jsonl"
    )
    with pytest.raises(ValueError):
        _assert_resume_provenance_compatible(relocated_wrong_file, current)

    changed_identity = dict(stored)
    changed_identity["source_sha256"] = "different"
    with pytest.raises(ValueError):
        _assert_resume_provenance_compatible(changed_identity, current)

    arbitrary_root = dict(stored)
    arbitrary_root["source_path"] = (
        "C:/arbitrary/data/r2_noise_robustness/levels/0.10%/noisy_surfaces.jsonl"
    )
    with pytest.raises(ValueError):
        _assert_resume_provenance_compatible(arbitrary_root, current)

    unapproved_current = dict(current)
    unapproved_current["source_path"] = (
        "C:/unapproved/data/r2_noise_robustness/levels/0.10%/noisy_surfaces.jsonl"
    )
    with pytest.raises(ValueError):
        _assert_resume_provenance_compatible(stored, unapproved_current)

    traversal_current = dict(current)
    traversal_current["source_path"] = (
        "C:/ann_inverse_calibration/../unapproved/data/r2_noise_robustness/levels/0.10%/noisy_surfaces.jsonl"
    )
    with pytest.raises(ValueError):
        _assert_resume_provenance_compatible(stored, traversal_current)

    traversal_same_root = dict(stored)
    traversal_same_root["source_path"] = (
        "C:/ann_inverse_calibration/../unapproved/data/r2_noise_robustness/levels/0.10%/noisy_surfaces.jsonl"
    )
    with pytest.raises(ValueError):
        _assert_resume_provenance_compatible(traversal_same_root, traversal_same_root)
