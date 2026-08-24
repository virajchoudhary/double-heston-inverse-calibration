"""Deterministic generator for the frozen R2 positive-noise cohorts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils import write_json

from .execution import (
    CLEAN_DATASET_PATH,
    DATA_ROOT,
    assert_clean_dataset_identity,
    derive_noisy_record,
    iter_test_records,
    level_label,
    load_frozen_protocol,
    serialize_record,
    sha256_path,
)


def generate_cohorts(
    *,
    output_root: str | Path = DATA_ROOT,
    clean_dataset: str | Path = CLEAN_DATASET_PATH,
) -> dict[str, Any]:
    """Generate only the four positive-level cohorts; 0% remains canonical."""
    protocol = load_frozen_protocol()
    clean_sha256 = assert_clean_dataset_identity(clean_dataset)
    root = Path(output_root)
    if (root / "MANIFEST.json").exists() and not root.joinpath(
        "levels", "0.10%", "noisy_surfaces.jsonl"
    ).exists():
        raise FileExistsError(
            "refusing ambiguous partial cohort root containing a manifest"
        )

    positive = [
        (float(level), label)
        for level, label in zip(protocol["noise_levels"], protocol["noise_level_labels"])
        if float(level) > 0.0
    ]
    files: dict[str, dict[str, Any]] = {}
    temporary_paths: list[Path] = []
    try:
        for noise_level, label in positive:
            final_path = root / "levels" / label / "noisy_surfaces.jsonl"
            if final_path.exists():
                raise FileExistsError(f"refusing to overwrite cohort: {final_path}")
            temporary_path = final_path.with_name(final_path.name + ".tmp")
            temporary_paths.append(temporary_path)
            temporary_path.parent.mkdir(parents=True, exist_ok=True)
            records = 0
            resamples = 0
            with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
                for _, clean_record in iter_test_records(clean_dataset):
                    derived = derive_noisy_record(
                        clean_record,
                        noise_level=noise_level,
                        label=label,
                        clean_sha256=clean_sha256,
                    )
                    handle.write(serialize_record(derived))
                    records += 1
                    resamples += sum(
                        item["resample_counter"]
                        for item in derived["observation_noise"]["realizations"]
                    )
            if records != int(protocol["evaluation_population"]["test_surface_count"]):
                raise RuntimeError(f"unexpected test count at {label}: {records}")
            temporary_path.replace(final_path)
            files[f"levels/{label}/noisy_surfaces.jsonl"] = {
                "record_count": records,
                "sha256": sha256_path(final_path),
                "positive_price_resample_events": resamples,
            }
    finally:
        for path in temporary_paths:
            if path.exists():
                path.unlink()

    manifest = {
        "artifact_kind": "R2_NOISE_ROBUSTNESS_IMMUTABLE_COHORTS",
        "base_seed": protocol["noise_semantics"]["base_seed"],
        "clean_dataset_sha256": clean_sha256,
        "files": files,
        "generation_status": "COMPLETE_BYTE_REPLAYABLE",
        "levels_in_manifest": [label for _, label in positive],
        "population": "test_split_only_N1250",
        "protocol_config_sha256": sha256_path(
            Path(__file__).resolve().parents[2]
            / "configs"
            / "r2_noise_robustness_FINAL.yaml"
        ),
    }
    write_json(root / "MANIFEST.json", manifest)
    return manifest
