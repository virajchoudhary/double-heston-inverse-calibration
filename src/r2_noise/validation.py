"""Fail-closed validation for the completed R2 noise-robustness bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .execution import (
    CLEAN_DATASET_PATH,
    DATA_ROOT,
    EVIDENCE_ROOT,
    assert_clean_dataset_identity,
    derive_noisy_record,
    iter_test_records,
    load_frozen_protocol,
    serialize_record,
    sha256_path,
)
from .neural_evaluation import NEURAL_SEEDS


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_cohorts() -> dict[str, Any]:
    protocol = load_frozen_protocol()
    clean_hash = assert_clean_dataset_identity()
    labels = [label for _, label in zip(protocol["noise_levels"], protocol["noise_level_labels"]) if float(_) > 0.0]
    manifest = _read_json(DATA_ROOT / "MANIFEST.json")
    if manifest["levels_in_manifest"] != labels or manifest["clean_dataset_sha256"] != clean_hash:
        raise ValueError("cohort manifest identity mismatch")
    clean_by_id = {
        record["surface_id"]: record
        for _, record in iter_test_records(CLEAN_DATASET_PATH)
    }
    for label in labels:
        path = DATA_ROOT / "levels" / label / "noisy_surfaces.jsonl"
        relative = f"levels/{label}/noisy_surfaces.jsonl"
        if sha256_path(path) != manifest["files"][relative]["sha256"]:
            raise ValueError(f"cohort file hash mismatch: {path}")
        level = next(float(level) for level, name in zip(protocol["noise_levels"], protocol["noise_level_labels"]) if name == label)
        count = 0
        clean_iter = iter(clean_by_id.values())
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                clean = next(clean_iter)
                derived = json.loads(line)
                expected = derive_noisy_record(
                    clean,
                    noise_level=level,
                    label=label,
                    clean_sha256=clean_hash,
                )
                if line != serialize_record(expected):
                    raise ValueError(f"non-replayable cohort line {path}:{line_number}")
                if derived["surface_id"] != clean["surface_id"]:
                    raise ValueError("cohort population/order mismatch")
                count += 1
        if count != 1250:
            raise ValueError(f"unexpected cohort count at {label}: {count}")
    return {"status": "PASSED", "levels": labels, "records_per_level": 1250}


def validate_completed_bundle() -> dict[str, Any]:
    protocol = load_frozen_protocol()
    cohort_check = validate_cohorts()
    neural_gate = _read_json(
        EVIDENCE_ROOT / "zero_percent_neural" / "ZERO_PERCENT_GATE.json"
    )
    traditional_gate = _read_json(
        EVIDENCE_ROOT / "traditional" / "level_0pct" / "ZERO_PERCENT_GATE.json"
    )
    if neural_gate["status"] != "PASSED" or traditional_gate["status"] != "PASSED":
        raise ValueError("one or more 0% reproduction gates did not pass")

    labels = list(protocol["noise_level_labels"])
    neural_manifest = _read_json(EVIDENCE_ROOT / "neural" / "MANIFEST.json")
    if neural_manifest["levels"] != labels or neural_manifest["population"] != "FULL_TEST_NEURAL_N1250":
        raise ValueError("neural bundle is incomplete")
    for relative, expected_hash in neural_manifest["files"].items():
        path = EVIDENCE_ROOT / "neural" / relative
        if sha256_path(path) != expected_hash:
            raise ValueError(f"neural artifact hash mismatch: {path}")

    traditional_summaries = []
    for label in labels:
        safe = label.replace(".", "_").replace("%", "pct")
        summary = _read_json(
            EVIDENCE_ROOT / "traditional" / f"level_{safe}" / "run_summary.json"
        )
        if summary["surfaces_calibrated"] != 250 or summary["starts_recorded"] != 750:
            raise ValueError(f"incomplete traditional run: {label}")
        traditional_summaries.append(summary)
    required_aggregates = [
        "paired_degradation.csv",
        "identifiability_conditioned_degradation.csv",
    ]
    for name in required_aggregates:
        if not (EVIDENCE_ROOT / name).exists():
            raise ValueError(f"missing aggregate: {name}")
    return {
        "R2_NOISE_ROBUSTNESS": "COMPLETE",
        "ZERO_PERCENT_REPRODUCTION_GATE": "PASSED",
        "ALL_FROZEN_NOISE_LEVELS": "COMPLETE",
        "NEURAL_FULL_N1250": "COMPLETE",
        "TRADITIONAL_SUBSET_N250": "COMPLETE",
        "cohorts": cohort_check,
        "neural_manifest": neural_manifest,
        "traditional_runs": traditional_summaries,
    }
