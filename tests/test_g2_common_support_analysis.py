from __future__ import annotations

import csv
import hashlib
import io
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

import scripts.run_g2_common_support_analysis as g2
from src.nse_stage_a import ArchiveIntegrityError, nse_archive_filename, nse_archive_url


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


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _provenance_fixture(tmp_path: Path) -> tuple[Path, Path, list[dict[str, str]]]:
    raw_root = tmp_path / "raw" / "nse"
    derived_root = tmp_path / "derived"
    rows: list[dict[str, str]] = []
    for value in g2.G2_DATES:
        for market in ("CM", "FO"):
            filename = nse_archive_filename(market, value)
            member = filename.removesuffix(".zip")
            csv_bytes = f"synthetic-{market}-{value.isoformat()}\n".encode()
            archive_buffer = io.BytesIO()
            with zipfile.ZipFile(archive_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(member, csv_bytes)
            archive_bytes = archive_buffer.getvalue()
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
                    "archive_member_name": member,
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


def _balanced_panel() -> pd.DataFrame:
    schedules = {
        "2026-07-01": (27, 55, 90),
        "2026-07-15": (13, 41, 76),
        "2026-07-22": (6, 34, 69),
    }
    slots = ("near", "mid", "far")
    rows: list[dict[str, object]] = []
    for underlying in g2.PRIMARY_UNDERLYINGS:
        for valuation_date, dtes in schedules.items():
            start = date.fromisoformat(valuation_date)
            for slot, dte in zip(slots, dtes, strict=True):
                for option_type in g2.PROPOSED_OPTION_TYPES:
                    for node in (-0.12, -0.10, -0.05, 0.00, 0.05, 0.10, 0.12):
                        active = slot != "far" or node == 0.00
                        rows.append(
                            {
                                "underlying": underlying,
                                "valuation_date": valuation_date,
                                "actual_expiry": start + timedelta(days=dte),
                                "DTE": dte,
                                "T": dte / 365.0,
                                "expiry_slot": slot,
                                "OptnTp": option_type,
                                "log_K_over_S": node,
                                "strike": 100.0 * (2.718281828459045**node),
                                "traded_qty_positive": active,
                                "open_interest_positive": active,
                                "transactions_positive": active,
                                "close_positive": True,
                                "last_positive": active,
                                "settlement_positive": True,
                            }
                        )
    panel = pd.DataFrame(rows)
    panel["active_positive"] = (
        panel["traded_qty_positive"]
        | panel["open_interest_positive"]
        | panel["transactions_positive"]
    )
    panel["price_observed_positive"] = (
        panel["close_positive"] | panel["last_positive"] | panel["settlement_positive"]
    )
    return panel


def test_exact_balanced_g2_contract_excludes_power_extension_backups_and_nifty() -> None:
    assert g2.PRIMARY_UNDERLYINGS == ("NTPC", "CIPLA", "INFY", "HDFCBANK")
    assert g2.G2_DATES == (date(2026, 7, 1), date(2026, 7, 15), date(2026, 7, 22))
    assert g2.POWER_TIEBREAK_DATES == (date(2026, 7, 8), date(2026, 7, 29))
    assert not set(g2.PRIMARY_UNDERLYINGS).intersection(g2.BACKUP_UNDERLYINGS)
    assert g2.REFERENCE_UNDERLYING not in g2.PRIMARY_UNDERLYINGS
    g2.validate_balanced_panel(_balanced_panel())


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("underlying", "POWERGRID", "exactly NTPC"),
        ("underlying", "NIFTY", "exactly NTPC"),
        ("valuation_date", "2026-07-08", "exactly the three"),
        ("valuation_date", "2026-07-29", "exactly the three"),
    ],
)
def test_balanced_panel_rejects_excluded_security_or_date(
    column: str, value: str, message: str
) -> None:
    panel = _balanced_panel()
    panel.loc[panel.index[0], column] = value
    with pytest.raises(ValueError, match=message):
        g2.validate_balanced_panel(panel)


def test_provenance_enforces_exact_six_identities_and_current_hashes(tmp_path: Path) -> None:
    raw_root, derived_root, rows = _provenance_fixture(tmp_path)
    assert len(g2.validate_canonical_stage_a_provenance(raw_root, derived_root)) == 6

    csv_path = Path(rows[0]["csv_path"])
    csv_path.write_bytes(csv_path.read_bytes() + b"tampered")
    with pytest.raises(ArchiveIntegrityError, match="CSV hash"):
        g2.validate_canonical_stage_a_provenance(raw_root, derived_root)


def test_provenance_rejects_extra_or_missing_identity(tmp_path: Path) -> None:
    raw_root, derived_root, rows = _provenance_fixture(tmp_path)
    _write_manifest(derived_root / "acquisition_manifest.csv", rows[:-1])
    with pytest.raises(ArchiveIntegrityError, match="identities must be exactly"):
        g2.validate_canonical_stage_a_provenance(raw_root, derived_root)

    _write_manifest(derived_root / "acquisition_manifest.csv", [*rows, dict(rows[0])])
    with pytest.raises(ArchiveIntegrityError, match="duplicate identities"):
        g2.validate_canonical_stage_a_provenance(raw_root, derived_root)


def test_support_classifiers_never_silently_extrapolate() -> None:
    observed = g2._node_support(pd.Series([-0.10, 0.00, 0.10]).to_numpy(), 0.05)
    outside = g2._node_support(pd.Series([-0.10, 0.00, 0.10]).to_numpy(), 0.20)
    assert observed["classification"] == "INTERPOLATED"
    assert observed["inside_observed_bounds"] is True
    assert observed["extrapolation_required"] is False
    assert outside["classification"] == "UNSUPPORTED"
    assert outside["inside_observed_bounds"] is False
    assert outside["extrapolation_required"] is True

    maturity = g2._fixed_maturity_support(pd.Series([27, 55, 90]).to_numpy(), 60)
    maturity_outside = g2._fixed_maturity_support(pd.Series([27, 55, 90]).to_numpy(), 180)
    assert maturity["classification"] == "INTERPOLATED"
    assert maturity["extrapolation_required"] is False
    assert maturity_outside["classification"] == "UNSUPPORTED"
    assert maturity_outside["extrapolation_required"] is True


def test_proposed_geometry_is_exact_but_g2_stays_open_for_carry() -> None:
    panel = _balanced_panel()
    maturity = g2.build_maturity_support(panel)
    moneyness = g2.build_moneyness_support(panel)
    candidates = g2.build_representation_candidates(maturity, moneyness)
    proposed = candidates.loc[candidates["decision"] == "PROPOSED_GEOMETRY"]
    assert list(proposed["candidate_id"]) == ["relative_near_mid_central5"]
    row = proposed.iloc[0]
    assert row["maturity_nodes"] == "near|mid"
    assert row["moneyness_nodes"] == "-0.10|-0.05|+0.00|+0.05|+0.10"
    assert row["option_types"] == "CE|PE"
    assert row["normalized_price_feature_count"] == 20
    assert row["coordinate_feature_count"] == 2
    assert row["total_input_dimension"] == 22
    assert row["maturity_unsupported_pct"] == 0.0
    assert bool(row["no_maturity_extrapolation_pass"])
    assert bool(row["no_strike_extrapolation_pass"])
    assert not bool(row["carry_conditioning_rule_pass"])
    assert not bool(row["g2_gate_pass"])
    assert g2.G2_VERDICT == "NOT_PASSED"
    assert g2.FINAL_TOTAL_INPUT_DIMENSION is None


def test_maximal_geometry_selection_is_order_invariant_and_data_driven() -> None:
    panel = _balanced_panel()
    maturity = g2.build_maturity_support(panel)
    moneyness = g2.build_moneyness_support(panel)
    candidates = g2.build_representation_candidates(maturity, moneyness)

    shuffled = candidates.sample(frac=1.0, random_state=20260810).reset_index(drop=True)
    assert g2.select_maximal_passing_geometry(
        shuffled, tuple(reversed(g2.CANDIDATE_SPECS))
    ) == "relative_near_mid_central5"

    enlarged = candidates.copy()
    enlarged.loc[
        enlarged["candidate_id"] == "relative_near_mid_central7",
        "decision_rule_pass",
    ] = True
    assert g2.select_maximal_passing_geometry(
        enlarged
    ) == "relative_near_mid_central7"


def test_evidence_serialization_and_publication_are_deterministic(tmp_path: Path) -> None:
    panel = _balanced_panel()
    frames = {
        "surface": g2.build_surface_support(panel),
        "moneyness": g2.build_moneyness_support(panel),
        "maturity": g2.build_maturity_support(panel),
    }
    frames["candidates"] = g2.build_representation_candidates(
        frames["maturity"], frames["moneyness"]
    )
    first = g2.publish_evidence_atomically(frames, tmp_path / "first")
    second = g2.publish_evidence_atomically(frames, tmp_path / "second")
    assert {
        key: hashlib.sha256(first[key].read_bytes()).hexdigest() for key in first
    } == {
        key: hashlib.sha256(second[key].read_bytes()).hexdigest() for key in second
    }


def test_g2_publication_preserves_exact_eight_stage_a_outputs(tmp_path: Path) -> None:
    derived = tmp_path / "derived"
    derived.mkdir()
    for name in g2.CANONICAL_STAGE_A_OUTPUTS:
        (derived / name).write_bytes(f"canonical-{name}\n".encode())
    before = g2.snapshot_canonical_outputs(derived)
    panel = _balanced_panel()
    maturity = g2.build_maturity_support(panel)
    moneyness = g2.build_moneyness_support(panel)
    g2.publish_evidence_atomically(
        {
            "surface": g2.build_surface_support(panel),
            "moneyness": moneyness,
            "maturity": maturity,
            "candidates": g2.build_representation_candidates(maturity, moneyness),
        },
        derived,
    )
    g2.assert_canonical_outputs_preserved(derived, before)
    assert g2.snapshot_canonical_outputs(derived) == before
