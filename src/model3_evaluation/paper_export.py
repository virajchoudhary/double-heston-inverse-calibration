"""Deterministic publication-safe export from a sealed Model3 result manifest."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd

from .contracts import read_json, sha256_file


class PaperExportError(ValueError):
    """Export refused because evidence is absent, partial, or tampered."""


_TABLES = {
    "clean_comparison_table": ("model3_vs_frozen_baselines.csv", "csv"),
    "seed_stability_table": ("model3_seed_results.csv", "csv"),
}
_JSON_EVIDENCE = {
    "parameter_recovery_evidence": "parameter_metrics.json",
    "repricing_evidence": "repricing_metrics.json",
    "seed_dispersion_evidence": "three_seed_aggregation.json",
}


def export_publication_tables(result_root: str | Path, output_root: str | Path) -> list[Path]:
    """Copy only verified COMPLETE result artifacts; never invent placeholders."""
    source = Path(result_root)
    destination = Path(output_root)
    if destination.exists():
        raise PaperExportError(f"refusing to overwrite export output: {destination}")
    manifest_path = source / "final_evaluation_manifest.json"
    if not manifest_path.is_file():
        raise PaperExportError("sealed result manifest is missing")
    manifest = read_json(manifest_path)
    if manifest.get("schema") != "MODEL3_CLEAN_EVALUATION_RESULT_MANIFEST_V1":
        raise PaperExportError("unsupported result-manifest schema")
    if manifest.get("completion_state") != "COMPLETE":
        raise PaperExportError("partial results are not publication-exportable")
    status = read_json(source / "evaluation_status.json")
    if status.get("status") != "COMPLETE" or status.get("research_metrics_complete") is not True:
        raise PaperExportError("result status is not COMPLETE")
    for relative, expected in manifest["artifact_hashes"].items():
        path = source / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise PaperExportError(f"sealed artifact tampered or missing: {relative}")
    destination.mkdir(parents=True)
    exported: list[Path] = []
    try:
        for label, (relative, kind) in {**_TABLES}.items():
            source_path = source / relative
            if kind == "csv":
                frame = pd.read_csv(source_path)
                target = destination / f"{label}.csv"
                frame.to_csv(target, index=False, lineterminator="\n")
            else:
                target = destination / f"{label}.json"
                target.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
            exported.append(target)
        for label, relative in _JSON_EVIDENCE.items():
            target = destination / f"{label}.json"
            shutil.copyfile(source / relative, target)
            exported.append(target)
        export_manifest = {
            "schema": "MODEL3_PAPER_EXPORT_MANIFEST_V1",
            "source_result_manifest_sha256": sha256_file(manifest_path),
            "source_completion_state": "COMPLETE",
            "exports": {
                path.name: sha256_file(path) for path in exported
            },
        }
        write_path = destination / "paper_export_manifest.json"
        write_path.write_text(
            json.dumps(export_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        exported.append(write_path)
    except BaseException:
        # A failed export remains visibly incomplete rather than half-trusted.
        marker = destination / "EXPORT_PARTIAL_FAILED_CLOSED"
        try:
            descriptor = os.open(str(marker), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise PaperExportError(
                "refusing to overwrite existing partial-export marker"
            ) from error
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write("do_not_publish\n")
        raise
    return exported


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    parser.add_argument("output_root", type=Path)
    arguments = parser.parse_args(argv)
    paths = export_publication_tables(arguments.result_root, arguments.output_root)
    print("\n".join(str(path) for path in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
