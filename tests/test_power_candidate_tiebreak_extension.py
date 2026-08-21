from __future__ import annotations

import csv
import hashlib
import io
import sys
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

import scripts.run_power_candidate_tiebreak_extension as runner
from src.nse_stage_a import UDIFF_COLUMNS, nse_archive_filename, nse_archive_url


MANIFEST_FIELDS = (
    "market",
    "valuation_date",
    "official_url",
    "original_filename",
    "archive_path",
    "archive_size_bytes",
    "archive_sha256",
    "zip_integrity",
    "archive_member_name",
    "csv_path",
    "csv_sha256",
    "encoding",
    "delimiter",
    "current_run_action",
    "first_acquisition_status",
    "retrieval_timestamp_utc",
    "retrieval_timestamp_source",
)


def _udiff_bytes(value: date, market: str) -> bytes:
    row = {column: "" for column in UDIFF_COLUMNS}
    row.update(
        {
            "TradDt": value.strftime("%d-%b-%Y"),
            "BizDt": value.strftime("%d-%b-%Y"),
            "Sgmt": market,
            "Src": "NSE",
        }
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=UDIFF_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerow(row)
    return output.getvalue().encode("utf-8")


def _archive_bytes(member_name: str, csv_bytes: bytes) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member_name, csv_bytes)
    return output.getvalue()


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _original_fixture(
    tmp_path: Path,
    *,
    member_override: tuple[str, str] | None = None,
    dates: tuple[date, ...] | None = None,
) -> tuple[Path, Path, list[dict[str, str]]]:
    raw_root = tmp_path / "raw" / "nse"
    derived_root = tmp_path / "derived"
    rows: list[dict[str, str]] = []
    for value in dates or runner.ORIGINAL_STAGE_A_DATES:
        for market in ("CM", "FO"):
            filename = nse_archive_filename(market, value)
            expected_member = filename.removesuffix(".zip")
            member_name = expected_member
            if member_override == (market, value.isoformat()):
                member_name = "unexpected-member.csv"
            csv_bytes = _udiff_bytes(value, market)
            archive_bytes = _archive_bytes(member_name, csv_bytes)
            directory = raw_root / value.isoformat()
            directory.mkdir(parents=True, exist_ok=True)
            archive_path = directory / filename
            csv_path = archive_path.with_suffix("")
            archive_path.write_bytes(archive_bytes)
            csv_path.write_bytes(csv_bytes)
            rows.append(
                {
                    "market": market,
                    "valuation_date": value.isoformat(),
                    "official_url": nse_archive_url(market, value),
                    "original_filename": filename,
                    "archive_path": str(archive_path),
                    "archive_size_bytes": str(len(archive_bytes)),
                    "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
                    "zip_integrity": "True",
                    "archive_member_name": expected_member,
                    "csv_path": str(csv_path),
                    "csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
                    "encoding": "UTF-8",
                    "delimiter": ",",
                    "current_run_action": "REUSED",
                    "first_acquisition_status": "DOWNLOADED",
                    "retrieval_timestamp_utc": "2026-08-10T00:00:00+00:00",
                    "retrieval_timestamp_source": "download_time",
                }
            )
    _write_manifest(derived_root / "acquisition_manifest.csv", rows)
    return raw_root, derived_root, rows


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_original_manifest_and_raw_files_pass_provenance_preflight(tmp_path: Path) -> None:
    raw_root, derived_root, _ = _original_fixture(tmp_path)

    records = runner._validate_original_stage_a_provenance(raw_root, derived_root)

    assert len(records) == 6
    assert {(record.market, record.valuation_date) for record in records} == {
        (market, value.isoformat())
        for value in runner.ORIGINAL_STAGE_A_DATES
        for market in ("CM", "FO")
    }


def test_modified_original_csv_with_unchanged_manifest_fails_closed(tmp_path: Path) -> None:
    raw_root, derived_root, _ = _original_fixture(tmp_path)
    csv_path = raw_root / "2026-07-01" / nse_archive_filename("CM", date(2026, 7, 1)).removesuffix(".zip")
    csv_path.write_bytes(csv_path.read_bytes() + b"tampered")

    with pytest.raises(runner.ArchiveIntegrityError, match="extracted CSV"):
        runner._validate_original_stage_a_provenance(raw_root, derived_root)


def test_modified_original_archive_with_unchanged_manifest_fails_closed(tmp_path: Path) -> None:
    raw_root, derived_root, _ = _original_fixture(tmp_path)
    archive_path = raw_root / "2026-07-01" / nse_archive_filename("CM", date(2026, 7, 1))
    archive_path.write_bytes(archive_path.read_bytes() + b"tampered")

    with pytest.raises(runner.ArchiveIntegrityError, match="archive_sha256|ZIP"):
        runner._validate_original_stage_a_provenance(raw_root, derived_root)


def test_missing_original_manifest_row_fails_closed(tmp_path: Path) -> None:
    raw_root, derived_root, rows = _original_fixture(tmp_path)
    _write_manifest(derived_root / "acquisition_manifest.csv", rows[1:])

    with pytest.raises(runner.ArchiveIntegrityError, match="missing CM/2026-07-01"):
        runner._validate_original_stage_a_provenance(raw_root, derived_root)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("official_url", "https://example.invalid/not-nse.zip"),
        ("original_filename", "wrong.csv.zip"),
        ("archive_member_name", "wrong.csv"),
    ],
)
def test_wrong_original_manifest_identity_fails_closed(
    tmp_path: Path, field: str, value: str
) -> None:
    raw_root, derived_root, rows = _original_fixture(tmp_path)
    rows[0][field] = value
    _write_manifest(derived_root / "acquisition_manifest.csv", rows)

    with pytest.raises(runner.ArchiveIntegrityError, match="unexpected"):
        runner._validate_original_stage_a_provenance(raw_root, derived_root)


def test_wrong_archive_member_identity_fails_closed(tmp_path: Path) -> None:
    raw_root, derived_root, _ = _original_fixture(
        tmp_path, member_override=("CM", "2026-07-01")
    )

    with pytest.raises(runner.ArchiveIntegrityError, match="expected CSV member"):
        runner._validate_original_stage_a_provenance(raw_root, derived_root)


def test_exact_five_wednesday_power_only_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    assert runner.ORIGINAL_STAGE_A_DATES == (
        date(2026, 7, 1),
        date(2026, 7, 15),
        date(2026, 7, 22),
    )
    assert runner.ADDED_DATES == (date(2026, 7, 8), date(2026, 7, 29))
    assert runner.PANEL_DATES == (
        date(2026, 7, 1),
        date(2026, 7, 8),
        date(2026, 7, 15),
        date(2026, 7, 22),
        date(2026, 7, 29),
    )
    assert runner.POWER_CANDIDATES == ("NTPC", "POWERGRID")

    daily = {value.isoformat(): 100.0 for value in runner.PANEL_DATES}
    monkeypatch.setattr(
        runner,
        "_metric_payloads",
        lambda *_args: {
            "synthetic metric": (100.0, daily, "synthetic", "higher_is_better", 0)
        },
    )
    monkeypatch.setattr(
        runner,
        "_expiry_lists",
        lambda _options: {value.isoformat(): ["2026-08-27"] for value in runner.PANEL_DATES},
    )
    options = pd.DataFrame(
        {"underlying": ["NTPC", "POWERGRID", "NIFTY", "INFY"]}
    )

    rows = runner._evidence_rows(options, {}, {}, {})

    assert rows
    assert all(row["candidate_a"] == "NTPC" for row in rows)
    assert all(row["candidate_b"] == "POWERGRID" for row in rows)
    assert all(row["ranking_universe"] == "NTPC|POWERGRID" for row in rows)
    assert all(row["panel_dates"] == "2026-07-01|2026-07-08|2026-07-15|2026-07-22|2026-07-29" for row in rows)
    assert all(row["interpolation_performed"] is False and row["nifty_ranked"] is False for row in rows)


def test_minimum_symmetric_reach_is_zero_for_one_sided_surface() -> None:
    rows = []
    for value in runner.PANEL_DATES:
        log_moneyness = (0.10, 0.20) if value == runner.PANEL_DATES[0] else (-0.20, 0.20)
        rows.extend(
            {
                "valuation_date": value.isoformat(),
                "actual_expiry": date(2026, 8, 27),
                "log_K_over_S": node,
            }
            for node in log_moneyness
        )

    overall, daily = runner._minimum_symmetric_reach(pd.DataFrame(rows))

    assert overall == 0.0
    assert daily[runner.PANEL_DATES[0].isoformat()] == 0.0


def test_replay_and_evidence_manifest_serialization_are_byte_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    daily = {value.isoformat(): 100.0 for value in runner.PANEL_DATES}
    monkeypatch.setattr(
        runner,
        "_metric_payloads",
        lambda *_args: {
            "synthetic metric": (100.0, daily, "synthetic", "higher_is_better", 0)
        },
    )
    monkeypatch.setattr(
        runner,
        "_expiry_lists",
        lambda _options: {value.isoformat(): ["2026-08-27"] for value in runner.PANEL_DATES},
    )
    rows = runner._evidence_rows(
        pd.DataFrame({"underlying": ["NTPC", "POWERGRID"]}), {}, {}, {}
    )
    raw_root, derived_root, _ = _original_fixture(tmp_path)
    records = runner._validate_original_stage_a_provenance(raw_root, derived_root)
    manifest_rows = runner._manifest_rows(records)

    extension_raw_root, extension_derived_root, _ = _original_fixture(
        tmp_path / "extension", dates=runner.ADDED_DATES
    )
    first_replay = runner._acquire_extension_records(
        extension_raw_root, extension_derived_root, offline=True
    )
    first_manifest = runner._manifest_rows(first_replay)
    runner._write_csv_atomic(
        extension_derived_root / "acquisition_manifest.csv",
        first_manifest,
        MANIFEST_FIELDS,
    )
    second_replay = runner._acquire_extension_records(
        extension_raw_root, extension_derived_root, offline=True
    )
    second_manifest = runner._manifest_rows(second_replay)
    assert first_manifest == second_manifest

    evidence_paths = [tmp_path / "evidence-a.csv", tmp_path / "evidence-b.csv"]
    manifest_paths = [tmp_path / "manifest-a.csv", tmp_path / "manifest-b.csv"]
    for path in evidence_paths:
        runner._write_csv_atomic(path, rows, runner.EVIDENCE_FIELDS)
    for path in manifest_paths:
        runner._write_csv_atomic(path, manifest_rows, MANIFEST_FIELDS)

    assert _sha(evidence_paths[0]) == _sha(evidence_paths[1])
    assert _sha(manifest_paths[0]) == _sha(manifest_paths[1])


def test_provenance_preflight_preserves_candidate_and_stage_a_outputs(tmp_path: Path) -> None:
    raw_root, derived_root, _ = _original_fixture(tmp_path)
    preserved = {
        "candidate_pairwise_evidence.csv": tmp_path / "candidate_pairwise_evidence.csv",
        "power_candidate_tiebreak_evidence.csv": tmp_path / "power_candidate_tiebreak_evidence.csv",
    }
    preserved["candidate_pairwise_evidence.csv"].write_text(
        "row\n" + "\n".join(str(index) for index in range(128)) + "\n",
        encoding="utf-8",
    )
    preserved["power_candidate_tiebreak_evidence.csv"].write_text(
        "row\n" + "\n".join(str(index) for index in range(43)) + "\n",
        encoding="utf-8",
    )
    stage_a_outputs = {
        name: tmp_path / f"{name}.csv"
        for name in (
            "acquisition_manifest",
            "surface_summary",
            "expiry_coverage",
            "moneyness_coverage",
            "candidate_grid_support",
            "futures_availability",
            "spot_consistency",
            "universe_presence",
        )
    }
    for path in stage_a_outputs.values():
        path.write_text("preserved\n", encoding="utf-8")
    before = {path: _sha(path) for path in (*preserved.values(), *stage_a_outputs.values())}
    raw_before = {
        path: _sha(path)
        for path in raw_root.rglob("*")
        if path.is_file()
    }

    runner._validate_original_stage_a_provenance(raw_root, derived_root)

    assert sum(1 for _ in preserved["candidate_pairwise_evidence.csv"].read_text().splitlines()) - 1 == 128
    assert sum(1 for _ in preserved["power_candidate_tiebreak_evidence.csv"].read_text().splitlines()) - 1 == 43
    assert {path: _sha(path) for path in (*preserved.values(), *stage_a_outputs.values())} == before
    assert {
        path: _sha(path)
        for path in raw_root.rglob("*")
        if path.is_file()
    } == raw_before


def test_provenance_failure_does_not_publish_partial_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_root, derived_root, _ = _original_fixture(tmp_path)
    csv_path = raw_root / "2026-07-01" / nse_archive_filename("CM", date(2026, 7, 1)).removesuffix(".zip")
    csv_path.write_bytes(csv_path.read_bytes() + b"tampered")
    extension_raw_root = tmp_path / "extension-raw"
    extension_derived_root = tmp_path / "extension-derived"
    evidence_path = tmp_path / "power_candidate_tiebreak_evidence.csv"
    extension_derived_root.mkdir()
    manifest_path = extension_derived_root / "acquisition_manifest.csv"
    manifest_path.write_bytes(b"previous manifest\n")
    evidence_path.write_bytes(b"previous evidence\n")
    before_manifest = manifest_path.read_bytes()
    before_evidence = evidence_path.read_bytes()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_power_candidate_tiebreak_extension.py",
            "--canonical-raw-root",
            str(raw_root),
            "--canonical-derived-root",
            str(derived_root),
            "--extension-raw-root",
            str(extension_raw_root),
            "--extension-derived-root",
            str(extension_derived_root),
            "--evidence-path",
            str(evidence_path),
            "--offline",
        ],
    )

    with pytest.raises(runner.ArchiveIntegrityError):
        runner.main()

    assert manifest_path.read_bytes() == before_manifest
    assert evidence_path.read_bytes() == before_evidence
    assert not list(extension_derived_root.glob("*.tmp"))
    if extension_raw_root.exists():
        assert not list(extension_raw_root.rglob("*"))


def test_output_publication_rolls_back_both_files_when_second_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "extension" / "acquisition_manifest.csv"
    evidence_path = tmp_path / "power_candidate_tiebreak_evidence.csv"
    manifest_path.parent.mkdir()
    manifest_path.write_bytes(b"previous manifest\n")
    evidence_path.write_bytes(b"previous evidence\n")
    before_manifest = manifest_path.read_bytes()
    before_evidence = evidence_path.read_bytes()
    real_replace = runner.os.replace
    failure_injected = False

    def fail_once_on_evidence_install(source: str | Path, destination: str | Path) -> None:
        nonlocal failure_injected
        if (
            Path(destination) == evidence_path
            and Path(source).suffix == ".tmp"
            and not failure_injected
        ):
            failure_injected = True
            raise OSError("injected evidence publication failure")
        real_replace(source, destination)

    monkeypatch.setattr(runner.os, "replace", fail_once_on_evidence_install)

    with pytest.raises(OSError, match="injected evidence publication failure"):
        runner._publish_csv_outputs_atomically(
            manifest_path,
            [{"market": "CM"}],
            ("market",),
            evidence_path,
            [{"metric": "new"}],
            ("metric",),
        )

    assert failure_injected
    assert manifest_path.read_bytes() == before_manifest
    assert evidence_path.read_bytes() == before_evidence
    assert sorted(
        path.name for path in tmp_path.rglob("*") if path.is_file()
    ) == ["acquisition_manifest.csv", "power_candidate_tiebreak_evidence.csv"]


def test_output_publication_retains_backup_when_rollback_also_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "extension" / "acquisition_manifest.csv"
    evidence_path = tmp_path / "power_candidate_tiebreak_evidence.csv"
    manifest_path.parent.mkdir()
    manifest_path.write_bytes(b"previous manifest\n")
    evidence_path.write_bytes(b"previous evidence\n")
    real_replace = runner.os.replace

    def fail_evidence_install_and_restore(
        source: str | Path, destination: str | Path
    ) -> None:
        source_path = Path(source)
        if Path(destination) == evidence_path and source_path.suffix in {".tmp", ".bak"}:
            raise OSError(f"injected evidence {source_path.suffix} failure")
        real_replace(source, destination)

    monkeypatch.setattr(runner.os, "replace", fail_evidence_install_and_restore)

    with pytest.raises(RuntimeError, match="retained backup"):
        runner._publish_csv_outputs_atomically(
            manifest_path,
            [{"market": "CM"}],
            ("market",),
            evidence_path,
            [{"metric": "new"}],
            ("metric",),
        )

    assert manifest_path.read_bytes() == b"previous manifest\n"
    assert not evidence_path.exists()
    retained_backups = list(tmp_path.rglob("*.bak"))
    assert len(retained_backups) == 1
    assert retained_backups[0].read_bytes() == b"previous evidence\n"
    assert not list(tmp_path.rglob("*.tmp"))
