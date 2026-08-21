"""Create ignored Stage A drop folders and pending manifests, never fake data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "market_data_audit_stage_a.yaml"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "market_data_audit" / "stage_a"
EXPECTED_FILES = [
    "options_raw.xlsx",
    "futures_raw.xlsx",
    "spot_raw.xlsx",
    "collection_manifest.yaml",
]


def create_structure(config_path: Path, output_root: Path) -> list[Path]:
    """Create all Stage A directories and only missing pending manifests."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dates = [str(value) for value in config["valuation_dates"]]
    created_manifests: list[Path] = []

    for sector_config in config["universe"]["sectors"].values():
        sector = sector_config["directory_name"]
        for underlying in sector_config["candidates"]:
            for valuation_date in dates:
                directory = (
                    output_root / "candidates" / sector / underlying / valuation_date
                )
                manifest = _manifest(
                    underlying=underlying,
                    valuation_date=valuation_date,
                    sector=sector,
                    reference_only=False,
                )
                if _write_missing_manifest(directory, manifest):
                    created_manifests.append(directory / "collection_manifest.yaml")

    reference = config["universe"]["reference"]
    for valuation_date in dates:
        underlying = reference["underlying"]
        directory = output_root / "reference" / underlying / valuation_date
        manifest = _manifest(
            underlying=underlying,
            valuation_date=valuation_date,
            sector=None,
            reference_only=bool(reference["reference_only"]),
        )
        if _write_missing_manifest(directory, manifest):
            created_manifests.append(directory / "collection_manifest.yaml")

    return created_manifests


def _manifest(
    underlying: str,
    valuation_date: str,
    sector: str | None,
    reference_only: bool,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "stage": "A",
        "collection_status": "NOT_COLLECTED",
        "underlying": underlying,
        "valuation_date": valuation_date,
        "sector": sector,
        "reference_only": reference_only,
        "surface_definition": "one_underlying_date_with_all_near_mid_far_expiry_slices",
        "expected_files": EXPECTED_FILES,
        "bloomberg_export_pending": True,
        "observations_present": False,
        "notes": None,
    }


def _write_missing_manifest(directory: Path, manifest: dict[str, Any]) -> bool:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "collection_manifest.yaml"
    if path.exists():
        return False
    path.write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create ignored Stage A drop folders without fake Bloomberg rows."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    created = create_structure(args.config, args.output_root)
    print(f"Created {len(created)} pending manifests under {args.output_root}")


if __name__ == "__main__":
    main()
