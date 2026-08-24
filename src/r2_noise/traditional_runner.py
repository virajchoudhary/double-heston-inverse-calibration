"""Crash-safe traditional calibration on the predeclared N=250 subset."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.r2_primary.calibration import FROZEN_SETTINGS, run_traditional_calibration, select_representatives
from src.r2_primary.dataset import R2PrimaryDataset
from src.utils import write_json

from .execution import (
    CLEAN_DATASET_PATH,
    DATA_ROOT,
    EVIDENCE_ROOT,
    REPO_ROOT,
    assert_clean_dataset_identity,
    level_label,
    load_frozen_protocol,
    sha256_path,
)
from .neural_evaluation import safe_level_label

SUBSET_PATH = (
    REPO_ROOT / "evidence" / "r2_noise_robustness" / "traditional_subset_ids.json"
)
CANONICAL_TRADITIONAL_STARTS = (
    REPO_ROOT
    / "evidence"
    / "r2_primary_comparison_20260823"
    / "traditional_calibration_starts.csv"
)


def _subset_dataset(source_path: Path) -> tuple[R2PrimaryDataset, list[str]]:
    subset_payload = json.loads(SUBSET_PATH.read_text(encoding="utf-8"))
    selected_ids = subset_payload["selected_ids"]
    if len(selected_ids) != 250 or len(set(selected_ids)) != 250:
        raise ValueError("frozen traditional subset is not exactly N=250 unique")
    dataset = R2PrimaryDataset.from_jsonl(source_path, splits={"test"})
    by_id = {item.surface_id: index for index, item in enumerate(dataset.items)}
    missing = [surface_id for surface_id in selected_ids if surface_id not in by_id]
    if missing:
        raise ValueError(f"cohort is missing {len(missing)} subset surfaces")
    items = [dataset.items[by_id[surface_id]] for surface_id in selected_ids]
    return R2PrimaryDataset(items), selected_ids


def _provenance(
    *,
    label: str,
    source_path: Path,
    selected_ids: list[str],
) -> dict[str, object]:
    return {
        "artifact_kind": "R2_NOISE_ROBUSTNESS_TRADITIONAL_JOURNAL_PROVENANCE",
        "clean_dataset_sha256": assert_clean_dataset_identity(CLEAN_DATASET_PATH),
        "frozen_settings": FROZEN_SETTINGS,
        "noise_level_label": label,
        "population": "THREE_WAY_FROZEN_SUBSET_N250",
        "selected_ids": selected_ids,
        "source_path": source_path.as_posix(),
        "source_sha256": sha256_path(source_path),
        "subset_artifact_sha256": sha256_path(SUBSET_PATH),
    }


def run_traditional_subset(
    noise_level: float,
    *,
    workers: int = 10,
) -> dict[str, object]:
    """Run or resume exactly one frozen level; never overwrite a final bundle."""
    protocol = load_frozen_protocol()
    allowed = {float(value): label for value, label in zip(protocol["noise_levels"], protocol["noise_level_labels"])}
    if noise_level not in allowed:
        raise ValueError("requested traditional level outside frozen protocol")
    label = level_label(noise_level, protocol["noise_level_labels"])
    source_path = (
        CLEAN_DATASET_PATH
        if noise_level == 0.0
        else DATA_ROOT / "levels" / label / "noisy_surfaces.jsonl"
    )
    if noise_level > 0.0:
        cohort_manifest = json.loads((DATA_ROOT / "MANIFEST.json").read_text())
        relative = f"levels/{label}/noisy_surfaces.jsonl"
        expected_hash = cohort_manifest["files"][relative]["sha256"]
        if sha256_path(source_path) != expected_hash:
            raise ValueError(f"cohort hash mismatch: {source_path}")

    subset_dataset, selected_ids = _subset_dataset(source_path)
    output_root = EVIDENCE_ROOT / "traditional" / f"level_{safe_level_label(label)}"
    output_root.mkdir(parents=True, exist_ok=True)
    starts_path = output_root / "traditional_calibration_starts.csv"
    journal_path = output_root / "traditional_calibration_starts_journal.jsonl"
    provenance_path = output_root / "JOURNAL_PROVENANCE.json"
    current_provenance = _provenance(label=label, source_path=source_path, selected_ids=selected_ids)
    if provenance_path.exists():
        stored = json.loads(provenance_path.read_text(encoding="utf-8"))
        if stored != current_provenance:
            raise ValueError(f"journal provenance mismatch at {output_root}")
    else:
        if journal_path.exists():
            raise ValueError("orphan calibration journal found without provenance")
        write_json(provenance_path, current_provenance)
    if starts_path.exists():
        raise FileExistsError(f"final traditional evidence already exists: {starts_path}")

    starts_frame = run_traditional_calibration(
        subset_dataset,
        starts_path,
        workers=workers,
        split="test",
    )
    representatives = select_representatives(starts_frame)
    representatives_path = output_root / "traditional_calibration_results.csv"
    representatives.to_csv(representatives_path, index=False)
    summary = {
        "artifact_kind": "R2_NOISE_ROBUSTNESS_TRADITIONAL_SUBSET_RUN",
        "failed_starts_retained": int((~starts_frame["success"].astype(bool)).sum()),
        "noise_level": float(noise_level),
        "noise_level_label": label,
        "population": "THREE_WAY_FROZEN_SUBSET_N250",
        "representative_count": int(len(representatives)),
        "source_sha256": sha256_path(source_path),
        "starts_failed": int((~starts_frame["success"].astype(bool)).sum()),
        "starts_flagged_success": int(starts_frame["success"].sum()),
        "starts_recorded": int(len(starts_frame)),
        "status": "COMPLETE_FAILED_STARTS_RETAINED",
        "surfaces_calibrated": int(starts_frame["surface_id"].nunique()),
        "total_wall_seconds_all_starts": float(
            starts_frame["wall_seconds_all_starts"].sum()
        ),
    }
    write_json(output_root / "run_summary.json", summary)
    return summary


def compare_zero_percent_traditional_gate() -> dict[str, object]:
    """Compare scientific start-row values for the same 250 canonical IDs."""
    zero_dir = EVIDENCE_ROOT / "traditional" / "level_0pct"
    produced = pd.read_csv(zero_dir / "traditional_calibration_starts.csv")
    subset_ids = set(json.loads(SUBSET_PATH.read_text())["selected_ids"])
    canonical = pd.read_csv(CANONICAL_TRADITIONAL_STARTS)
    canonical_subset = canonical[canonical["surface_id"].isin(subset_ids)].copy()
    runtime_columns = ["runtime_seconds", "wall_seconds_all_starts"]
    science_columns = [
        column for column in canonical_subset.columns if column not in runtime_columns
    ]
    left = produced.sort_values(["surface_id", "start_index"]).reset_index(drop=True)[science_columns]
    right = canonical_subset.sort_values(["surface_id", "start_index"]).reset_index(drop=True)[science_columns]
    exact = left.equals(right)
    report = {
        "artifact_kind": "R2_NOISE_ROBUSTNESS_ZERO_PERCENT_TRADITIONAL_GATE",
        "canonical_rows_compared": int(len(right)),
        "checks": {
            "exactly_750_frozen_subset_starts": len(produced) == 750,
            "scientific_start_rows_bitwise_after_pandas_round_trip": bool(exact),
        },
        "runtime_columns_excluded": runtime_columns,
        "status": "PASSED" if len(produced) == 750 and exact else "FAILED_STOP_BEFORE_POSITIVE_NOISE_INTERPRETATION",
    }
    write_json(zero_dir / "ZERO_PERCENT_GATE.json", report)
    if report["status"] != "PASSED":
        raise RuntimeError("0% traditional reproduction gate failed")
    return report
