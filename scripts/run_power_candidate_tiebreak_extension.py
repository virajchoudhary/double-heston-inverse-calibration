"""Run the predeclared five-Wednesday official-NSE Power tie-break extension.

This extension ranks only NTPC and POWERGRID.  It reuses the preserved original
Stage A files for 01/15/22 July 2026 and acquires only 08/29 July 2026 into a
separate raw tree.  It does not alter the canonical three-date Stage A contract,
rank NIFTY, interpolate support, compare activity magnitudes, or choose a neural
representation.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import tempfile
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.nse_stage_a import (  # noqa: E402
    AcquisitionRecord,
    ArchiveIntegrityError,
    acquire_udiff_archive,
    derive_option_observations,
    nse_archive_filename,
    nse_archive_url,
    read_prior_acquisition_evidence,
    read_udiff_csv,
)


EXTENSION_ID = "POWER_CANDIDATE_TIEBREAK_EXTENSION"
ORIGINAL_STAGE_A_DATES = (date(2026, 7, 1), date(2026, 7, 15), date(2026, 7, 22))
ADDED_DATES = (date(2026, 7, 8), date(2026, 7, 29))
PANEL_DATES = tuple(sorted((*ORIGINAL_STAGE_A_DATES, *ADDED_DATES)))
POWER_CANDIDATES = ("NTPC", "POWERGRID")
CENTRAL_NODES = (-0.10, -0.05, 0.00, 0.05, 0.10)

CANONICAL_RAW_ROOT = REPOSITORY_ROOT / "market_data_audit" / "stage_a" / "raw" / "nse"
CANONICAL_DERIVED_ROOT = REPOSITORY_ROOT / "market_data_audit" / "stage_a" / "derived"
EXTENSION_ROOT = REPOSITORY_ROOT / "market_data_audit" / "stage_a" / "power_tiebreak"
EXTENSION_RAW_ROOT = EXTENSION_ROOT / "raw" / "nse"
EXTENSION_DERIVED_ROOT = EXTENSION_ROOT / "derived"
DEFAULT_EVIDENCE_PATH = (
    REPOSITORY_ROOT
    / "market_data_audit"
    / "stage_a"
    / "derived"
    / "power_candidate_tiebreak_evidence.csv"
)

EVIDENCE_FIELDS = (
    "extension_id",
    "priority",
    "metric",
    "direction",
    "candidate_a",
    "candidate_a_value",
    "candidate_b",
    "candidate_b_value",
    "candidate_a_dates_better",
    "candidate_b_dates_better",
    "date_ties",
    "overall_better_or_tie",
    "definition",
    "source",
    "original_stage_a_dates",
    "added_dates",
    "panel_dates",
    "ranking_universe",
    "interpolation_performed",
    "nifty_ranked",
)


def _date_text(values: Iterable[date]) -> str:
    return "|".join(value.isoformat() for value in values)


def _pct(value: float) -> str:
    return f"{value:.3f}%"


def _number(value: float) -> str:
    return f"{value:.6f}"


def _summary(
    overall: float,
    daily: Mapping[str, float],
    formatter: Callable[[float], str],
    range_formatter: Callable[[float], str] | None = None,
) -> str:
    ordered = [daily[value.isoformat()] for value in PANEL_DATES]
    format_range = range_formatter or formatter
    return (
        f"overall={formatter(overall)}; "
        f"by_date[{';'.join(f'{value.isoformat()}={formatter(daily[value.isoformat()])}' for value in PANEL_DATES)}]; "
        f"min={formatter(min(ordered))}; median={formatter(statistics.median(ordered))}; "
        f"mean={formatter(statistics.mean(ordered))}; range={format_range(max(ordered) - min(ordered))}"
    )


def _expiry_summary(expiries: Mapping[str, Sequence[str]]) -> str:
    counts = {value: float(len(expiries[value])) for value in expiries}
    return (
        f"panel_min={int(min(counts.values()))}; "
        f"by_date[{';'.join(f'{value.isoformat()}={('|'.join(expiries[value.isoformat()]))}' for value in PANEL_DATES)}]"
    )


def _dominance(
    candidate_a_daily: Mapping[str, float], candidate_b_daily: Mapping[str, float]
) -> tuple[int, int, int]:
    a_better = b_better = ties = 0
    for value in PANEL_DATES:
        key = value.isoformat()
        difference = candidate_a_daily[key] - candidate_b_daily[key]
        if abs(difference) <= 1e-12:
            ties += 1
        elif difference > 0:
            a_better += 1
        else:
            b_better += 1
    return a_better, b_better, ties


def _overall_better(candidate_a: float, candidate_b: float) -> str:
    if abs(candidate_a - candidate_b) <= 1e-12:
        return "tie"
    return POWER_CANDIDATES[0] if candidate_a > candidate_b else POWER_CANDIDATES[1]


def _support(rows: pd.DataFrame, mask: pd.Series) -> dict[float, bool]:
    values = pd.to_numeric(rows.loc[mask, "log_K_over_S"], errors="coerce").dropna()
    if values.empty:
        return {node: False for node in CENTRAL_NODES}
    lower, upper = float(values.min()), float(values.max())
    return {node: lower <= node <= upper for node in CENTRAL_NODES}


def _slice_support_metrics(
    candidate_options: pd.DataFrame, mask_builder: Callable[[pd.DataFrame], pd.Series]
) -> tuple[
    float,
    dict[str, float],
    float,
    dict[str, float],
    dict[float, tuple[float, dict[str, float]]],
]:
    all_five_by_date: dict[str, float] = {}
    node_support_by_date: dict[str, float] = {}
    node_counts = {node: 0 for node in CENTRAL_NODES}
    node_daily_counts = {node: {} for node in CENTRAL_NODES}
    all_five_total = node_total = slice_total = 0
    for value in PANEL_DATES:
        date_text = value.isoformat()
        date_rows = candidate_options.loc[candidate_options["valuation_date"] == date_text]
        date_all_five = date_node_total = date_slices = 0
        for _, expiry_rows in date_rows.groupby("actual_expiry", sort=True):
            support = _support(expiry_rows, mask_builder(expiry_rows))
            date_slices += 1
            date_all_five += int(all(support.values()))
            date_node_total += sum(support.values())
        if date_slices == 0:
            raise ValueError(f"Missing option surface for {date_text}")
        all_five_by_date[date_text] = 100.0 * date_all_five / date_slices
        node_support_by_date[date_text] = 100.0 * date_node_total / (date_slices * len(CENTRAL_NODES))
        for node in CENTRAL_NODES:
            supported = sum(
                _support(expiry_rows, mask_builder(expiry_rows))[node]
                for _, expiry_rows in date_rows.groupby("actual_expiry", sort=True)
            )
            node_counts[node] += supported
            node_daily_counts[node][date_text] = 100.0 * supported / date_slices
        all_five_total += date_all_five
        node_total += date_node_total
        slice_total += date_slices
    node_profiles = {
        node: (100.0 * node_counts[node] / slice_total, node_daily_counts[node])
        for node in CENTRAL_NODES
    }
    return (
        100.0 * all_five_total / slice_total,
        all_five_by_date,
        100.0 * node_total / (slice_total * len(CENTRAL_NODES)),
        node_support_by_date,
        node_profiles,
    )


def _row_coverage(candidate_options: pd.DataFrame, field: str) -> tuple[float, dict[str, float]]:
    overall = 100.0 * float(candidate_options[field].sum()) / len(candidate_options)
    daily = {}
    for value in PANEL_DATES:
        date_rows = candidate_options.loc[candidate_options["valuation_date"] == value.isoformat()]
        daily[value.isoformat()] = 100.0 * float(date_rows[field].sum()) / len(date_rows)
    return overall, daily


def _reported_coverage(candidate_options: pd.DataFrame) -> tuple[float, dict[str, float]]:
    fields = (
        "close_reported",
        "settlement_reported",
        "last_reported",
        "traded_qty_reported",
        "open_interest_reported",
        "transactions_reported",
    )
    overall = 100.0 * sum(float(candidate_options[field].sum()) for field in fields) / (
        len(candidate_options) * len(fields)
    )
    daily = {}
    for value in PANEL_DATES:
        date_rows = candidate_options.loc[candidate_options["valuation_date"] == value.isoformat()]
        daily[value.isoformat()] = 100.0 * sum(float(date_rows[field].sum()) for field in fields) / (
            len(date_rows) * len(fields)
        )
    return overall, daily


def _minimum_symmetric_reach(candidate_options: pd.DataFrame) -> tuple[float, dict[str, float]]:
    daily: dict[str, float] = {}
    panel_values: list[float] = []
    for value in PANEL_DATES:
        date_rows = candidate_options.loc[candidate_options["valuation_date"] == value.isoformat()]
        reaches = []
        for _, expiry_rows in date_rows.groupby("actual_expiry", sort=True):
            values = pd.to_numeric(expiry_rows["log_K_over_S"], errors="coerce").dropna()
            lower, upper = float(values.min()), float(values.max())
            reaches.append(min(abs(lower), upper) if lower <= 0.0 <= upper else 0.0)
        daily[value.isoformat()] = min(reaches)
        panel_values.extend(reaches)
    return min(panel_values), daily


def _metric_payloads(
    candidate_options: pd.DataFrame,
    futures_expiries: Mapping[tuple[str, str], set[str]],
    cm_spots: Mapping[tuple[str, str], float],
    futures_expiry_agreement: Mapping[tuple[str, str], float],
) -> dict[str, tuple[float, dict[str, float], str, str, int]]:
    payloads: dict[str, tuple[float, dict[str, float], str, str, int]] = {}
    selectors = {
        "Observed": lambda rows: pd.Series(True, index=rows.index),
        "Trade-count-positive": lambda rows: rows["transactions_positive"].astype(bool),
        "OI-positive": lambda rows: rows["open_interest_positive"].astype(bool),
        "Traded-qty-positive": lambda rows: rows["traded_qty_positive"].astype(bool),
    }
    priority = {"Observed": 1, "Trade-count-positive": 2, "OI-positive": 2, "Traded-qty-positive": 2}
    for label, selector in selectors.items():
        all_five, all_five_daily, node_support, node_daily, node_profiles = _slice_support_metrics(
            candidate_options, selector
        )
        payloads[f"{label} slices supporting all five central nodes"] = (
            all_five,
            all_five_daily,
            "Percent of expiry slices whose selected rows span all five central nodes; no interpolation.",
            "higher_is_better",
            priority[label],
        )
        payloads[f"{label} central node-slice support"] = (
            node_support,
            node_daily,
            "Percent of central node-slice combinations inside the selected-row support range; no interpolation.",
            "higher_is_better",
            priority[label],
        )
        for node in CENTRAL_NODES:
            node_overall, node_by_date = node_profiles[node]
            payloads[f"{label} support at log-moneyness {node:+.2f}"] = (
                node_overall,
                node_by_date,
                f"Percent of expiry slices whose selected-row range contains the {node:+.2f} central node; no interpolation.",
                "higher_is_better",
                priority[label],
            )

    for label, field, priority_value in (
        ("Trade-count-positive row coverage", "transactions_positive", 5),
        ("OI-positive row coverage", "open_interest_positive", 6),
        ("Positive NSE Total Traded Qty row coverage", "traded_qty_positive", 5),
        ("Close-positive row coverage", "close_positive", 7),
        ("Settlement-positive row coverage", "settlement_positive", 7),
        ("Last-positive row coverage", "last_positive", 7),
    ):
        overall, daily = _row_coverage(candidate_options, field)
        if field in {"transactions_positive", "open_interest_positive", "traded_qty_positive"}:
            definition = f"Percent of option rows with {field.replace('_', ' ')}; activity magnitudes are not compared."
        else:
            definition = f"Percent of option rows with {field.replace('_', ' ')}; availability is not interpreted as quote quality."
        payloads[label] = (
            overall,
            daily,
            definition,
            "higher_is_better",
            priority_value,
        )

    reach, reach_daily = _minimum_symmetric_reach(candidate_options)
    payloads["Minimum symmetric observed reach around ATM"] = (
        reach,
        reach_daily,
        "Minimum across expiry slices of the smaller absolute observed left/right log-moneyness reach.",
        "higher_is_better",
        8,
    )

    surface_daily = {
        value.isoformat(): 100.0 * float(
            candidate_options.loc[candidate_options["valuation_date"] == value.isoformat(), "actual_expiry"].notna().any()
        )
        for value in PANEL_DATES
    }
    payloads["Logical surface present"] = (
        statistics.mean(surface_daily.values()),
        surface_daily,
        "Presence of one logical option surface on each panel date.",
        "higher_is_better",
        0,
    )

    both_daily: dict[str, float] = {}
    futures_daily: dict[str, float] = {}
    spot_daily: dict[str, float] = {}
    expiry_match_daily: dict[str, float] = {}
    futures_expiry_match_daily: dict[str, float] = {}
    for value in PANEL_DATES:
        date_text = value.isoformat()
        date_rows = candidate_options.loc[candidate_options["valuation_date"] == date_text]
        slice_both = []
        slice_futures = []
        for expiry, expiry_rows in date_rows.groupby("actual_expiry", sort=True):
            types = set(expiry_rows["OptnTp"].astype(str).str.upper())
            slice_both.append("CE" in types and "PE" in types)
            slice_futures.append(expiry.isoformat() in futures_expiries[(str(date_rows.iloc[0]["underlying"]), date_text)])
        both_daily[date_text] = 100.0 * sum(slice_both) / len(slice_both)
        futures_daily[date_text] = 100.0 * sum(slice_futures) / len(slice_futures)
        observed_underlyings = set(pd.to_numeric(date_rows["fo_underlying_price"], errors="coerce").dropna())
        spot = cm_spots[(str(date_rows.iloc[0]["underlying"]), date_text)]
        spot_daily[date_text] = 100.0 if spot > 0 and observed_underlyings == {spot} else 0.0
        expiry_match_daily[date_text] = 100.0 * float(date_rows["expiry_fields_match"].sum()) / len(date_rows)
        futures_expiry_match_daily[date_text] = futures_expiry_agreement[
            (str(date_rows.iloc[0]["underlying"]), date_text)
        ]
    for label, daily, definition in (
        ("Expiry slices with both calls and puts", both_daily, "Percent of expiry slices containing CE and PE rows."),
        ("Option expiries with corresponding futures", futures_daily, "Percent of option expiry slices with a matching stock-futures expiry."),
        ("Positive CM spot and F&O underlying consistency", spot_daily, "Percent of panel dates with a positive CM EQ close matching the unique F&O underlying price."),
        ("Option expiry-field agreement", expiry_match_daily, "Percent of option rows whose XpryDt and FininstrmActlXpryDt agree."),
        ("Futures expiry-field agreement", futures_expiry_match_daily, "Percent of stock-futures rows whose XpryDt and FininstrmActlXpryDt agree."),
    ):
        payloads[label] = (statistics.mean(daily.values()), daily, definition, "higher_is_better", 0)

    reported, reported_daily = _reported_coverage(candidate_options)
    payloads["Required observation-field reporting"] = (
        reported,
        reported_daily,
        "Percent reported across close, settlement, last, traded quantity, OI, and trade-count fields.",
        "higher_is_better",
        0,
    )
    return payloads


def _expiry_lists(candidate_options: pd.DataFrame) -> dict[str, list[str]]:
    return {
        value.isoformat(): sorted(
            expiry.isoformat()
            for expiry in candidate_options.loc[
                candidate_options["valuation_date"] == value.isoformat(), "actual_expiry"
            ].dropna().unique()
        )
        for value in PANEL_DATES
    }


def _evidence_rows(
    options: pd.DataFrame,
    futures_expiries: Mapping[tuple[str, str], set[str]],
    cm_spots: Mapping[tuple[str, str], float],
    futures_expiry_agreement: Mapping[tuple[str, str], float],
) -> list[dict[str, object]]:
    candidates = {
        candidate: options.loc[options["underlying"] == candidate].copy()
        for candidate in POWER_CANDIDATES
    }
    payloads = {
        candidate: _metric_payloads(
            candidate_options, futures_expiries, cm_spots, futures_expiry_agreement
        )
        for candidate, candidate_options in candidates.items()
    }
    rows: list[dict[str, object]] = []
    a, b = POWER_CANDIDATES
    source = f"official NSE {EXTENSION_ID}: {_date_text(PANEL_DATES)}"
    common = {
        "extension_id": EXTENSION_ID,
        "candidate_a": a,
        "candidate_b": b,
        "source": source,
        "original_stage_a_dates": _date_text(ORIGINAL_STAGE_A_DATES),
        "added_dates": _date_text(ADDED_DATES),
        "panel_dates": _date_text(PANEL_DATES),
        "ranking_universe": "|".join(POWER_CANDIDATES),
        "interpolation_performed": False,
        "nifty_ranked": False,
    }
    for metric in payloads[a]:
        a_overall, a_daily, definition, direction, priority = payloads[a][metric]
        b_overall, b_daily, _, _, _ = payloads[b][metric]
        a_better, b_better, ties = _dominance(a_daily, b_daily)
        formatter = _number if metric == "Minimum symmetric observed reach around ATM" else _pct
        range_formatter = _number if metric == "Minimum symmetric observed reach around ATM" else lambda value: f"{value:.3f}pp"
        rows.append({
            **common,
            "priority": priority,
            "metric": metric,
            "direction": direction,
            "candidate_a_value": _summary(a_overall, a_daily, formatter, range_formatter),
            "candidate_b_value": _summary(b_overall, b_daily, formatter, range_formatter),
            "candidate_a_dates_better": a_better,
            "candidate_b_dates_better": b_better,
            "date_ties": ties,
            "overall_better_or_tie": _overall_better(a_overall, b_overall),
            "definition": definition,
        })

    expiry_lists = {candidate: _expiry_lists(candidate_options) for candidate, candidate_options in candidates.items()}
    a_daily = {value: float(len(expiry_lists[a][value])) for value in expiry_lists[a]}
    b_daily = {value: float(len(expiry_lists[b][value])) for value in expiry_lists[b]}
    a_better, b_better, ties = _dominance(a_daily, b_daily)
    rows.append({
        **common,
        "priority": 0,
        "metric": "Actual expiries per surface",
        "direction": "higher_is_better",
        "candidate_a_value": _expiry_summary(expiry_lists[a]),
        "candidate_b_value": _expiry_summary(expiry_lists[b]),
        "candidate_a_dates_better": a_better,
        "candidate_b_dates_better": b_better,
        "date_ties": ties,
        "overall_better_or_tie": _overall_better(min(a_daily.values()), min(b_daily.values())),
        "definition": "Actual option expiries observed on each logical surface; no maturity node is inferred.",
    })
    return sorted(rows, key=lambda row: (int(row["priority"]), str(row["metric"])))


def _stage_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        return temporary
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _write_csv_atomic(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    temporary = _stage_csv(path, rows, fields)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _publish_csv_outputs_atomically(
    manifest_path: Path,
    manifest_rows: Sequence[Mapping[str, object]],
    manifest_fields: Sequence[str],
    evidence_path: Path,
    evidence_rows: Sequence[Mapping[str, object]],
    evidence_fields: Sequence[str],
) -> None:
    """Publish the manifest/evidence pair with rollback on replacement failure."""
    outputs = (
        (manifest_path, manifest_rows, manifest_fields),
        (evidence_path, evidence_rows, evidence_fields),
    )
    if manifest_path.resolve() == evidence_path.resolve():
        raise ValueError("Manifest and evidence outputs must use different paths")

    staged: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    installed: set[Path] = set()
    retained_backups: set[Path] = set()
    try:
        for target, rows, fields in outputs:
            staged[target] = _stage_csv(target, rows, fields)
        for target, _, _ in outputs:
            if target.exists():
                descriptor, backup_name = tempfile.mkstemp(
                    prefix=f".{target.name}.", suffix=".bak", dir=target.parent
                )
                os.close(descriptor)
                backup = Path(backup_name)
                backup.unlink()
                backups[target] = backup
                os.replace(target, backup)
            os.replace(staged[target], target)
            installed.add(target)
    except BaseException as publication_error:
        rollback_errors: list[tuple[Path, OSError]] = []
        for target, _, _ in reversed(outputs):
            try:
                if target in installed and target.exists():
                    target.unlink()
                backup = backups.get(target)
                if backup is not None and backup.exists():
                    os.replace(backup, target)
            except OSError as rollback_error:
                backup = backups.get(target)
                if backup is not None and backup.exists():
                    retained_backups.add(backup)
                rollback_errors.append((target, rollback_error))
        if rollback_errors:
            retained = ", ".join(str(path) for path in sorted(retained_backups))
            detail = f"; retained backups: {retained}" if retained else ""
            raise RuntimeError(
                f"CSV publication failed and rollback was incomplete{detail}"
            ) from publication_error
        raise
    finally:
        for temporary in staged.values():
            if temporary.exists():
                temporary.unlink()
        for backup in backups.values():
            if backup.exists() and backup not in retained_backups:
                backup.unlink()


def _manifest_rows(records: Iterable[AcquisitionRecord]) -> list[dict[str, object]]:
    rows = []
    for record in records:
        row = asdict(record)
        row["archive_path"] = str(record.archive_path)
        row["csv_path"] = str(record.csv_path)
        rows.append(row)
    return rows


def _validate_original_manifest_row(
    row: Mapping[str, str], market: str, value: date, canonical_raw_root: Path
) -> None:
    filename = nse_archive_filename(market, value)
    archive_path = canonical_raw_root / value.isoformat() / filename
    csv_path = archive_path.with_suffix("")
    expected = {
        "market": market,
        "valuation_date": value.isoformat(),
        "official_url": nse_archive_url(market, value),
        "original_filename": filename,
        "archive_path": str(archive_path),
        "csv_path": str(csv_path),
        "archive_member_name": filename.removesuffix(".zip"),
    }
    for field, expected_value in expected.items():
        observed = str(row.get(field, "")).strip()
        if field in {"archive_path", "csv_path"}:
            if not observed or Path(observed).resolve() != Path(expected_value).resolve():
                raise ArchiveIntegrityError(
                    f"Canonical Stage A manifest {market}/{value.isoformat()} has unexpected {field}."
                )
        elif observed != expected_value:
            raise ArchiveIntegrityError(
                f"Canonical Stage A manifest {market}/{value.isoformat()} has unexpected {field}."
            )
    if str(row.get("zip_integrity", "")).strip().lower() != "true":
        raise ArchiveIntegrityError(
            f"Canonical Stage A manifest {market}/{value.isoformat()} does not record ZIP integrity as true."
        )
    if not archive_path.is_file() or not csv_path.is_file():
        raise ArchiveIntegrityError(
            f"Canonical Stage A manifest {market}/{value.isoformat()} points to missing raw evidence."
        )


def _validate_original_stage_a_provenance(
    canonical_raw_root: Path, canonical_derived_root: Path
) -> tuple[AcquisitionRecord, ...]:
    """Validate every preserved three-date source before extension work begins."""
    manifest_path = canonical_derived_root / "acquisition_manifest.csv"
    if not manifest_path.is_file():
        raise ArchiveIntegrityError(
            f"Canonical Stage A acquisition manifest is missing: {manifest_path}"
        )
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    identities = [
        (str(row.get("market", "")).strip(), str(row.get("valuation_date", "")).strip())
        for row in manifest_rows
    ]
    if len(identities) != len(set(identities)):
        raise ArchiveIntegrityError("Canonical Stage A acquisition manifest contains duplicate identities.")
    prior = read_prior_acquisition_evidence(canonical_derived_root)
    records: list[AcquisitionRecord] = []
    for value in ORIGINAL_STAGE_A_DATES:
        for market in ("CM", "FO"):
            identity = (market, value.isoformat())
            row = prior.get(identity)
            if row is None:
                raise ArchiveIntegrityError(
                    f"Canonical Stage A acquisition manifest is missing {market}/{value.isoformat()}."
                )
            _validate_original_manifest_row(row, market, value, canonical_raw_root)
            records.append(
                acquire_udiff_archive(
                    market,
                    value,
                    canonical_raw_root,
                    prior_evidence=prior,
                    allow_download=False,
                )
            )
    return tuple(records)


def _acquire_extension_records(
    raw_root: Path, derived_root: Path, offline: bool
) -> list[AcquisitionRecord]:
    prior = read_prior_acquisition_evidence(derived_root)
    first_pass = [
        acquire_udiff_archive(
            market,
            value,
            raw_root,
            prior_evidence=prior,
            allow_download=not offline,
        )
        for value in ADDED_DATES
        for market in ("CM", "FO")
    ]
    stable_prior = {
        (record.market, record.valuation_date): {
            key: str(value)
            for key, value in _manifest_rows([record])[0].items()
        }
        for record in first_pass
    }
    return [
        acquire_udiff_archive(
            market,
            value,
            raw_root,
            prior_evidence=stable_prior,
            allow_download=False,
        )
        for value in ADDED_DATES
        for market in ("CM", "FO")
    ]


def _read_panel(
    canonical_raw_root: Path, extension_raw_root: Path
) -> tuple[
    pd.DataFrame,
    dict[tuple[str, str], set[str]],
    dict[tuple[str, str], float],
    dict[tuple[str, str], float],
]:
    option_frames = []
    futures_expiries: dict[tuple[str, str], set[str]] = {}
    cm_spots: dict[tuple[str, str], float] = {}
    futures_expiry_agreement: dict[tuple[str, str], float] = {}
    for value in PANEL_DATES:
        raw_root = canonical_raw_root if value in ORIGINAL_STAGE_A_DATES else extension_raw_root
        paths = {
            market: raw_root / value.isoformat() / nse_archive_filename(market, value).removesuffix(".zip")
            for market in ("CM", "FO")
        }
        cm = read_udiff_csv(paths["CM"], value, "CM")
        fo = read_udiff_csv(paths["FO"], value, "FO")
        power_fo = fo.loc[fo["TckrSymb"].astype(str).isin(POWER_CANDIDATES)].copy()
        options = derive_option_observations(power_fo, cm, value)
        if set(options["underlying"]) != set(POWER_CANDIDATES):
            raise ValueError(f"Power-only option universe is incomplete on {value.isoformat()}")
        option_frames.append(options)
        for candidate in POWER_CANDIDATES:
            cm_rows = cm.loc[
                (cm["SctySrs"].astype(str) == "EQ") & (cm["TckrSymb"].astype(str) == candidate)
            ]
            if len(cm_rows) != 1:
                raise ValueError(f"Expected one CM EQ row for {candidate} on {value.isoformat()}")
            cm_spots[(candidate, value.isoformat())] = float(pd.to_numeric(cm_rows["ClsPric"]).iloc[0])
            candidate_futures = power_fo.loc[
                (power_fo["TckrSymb"].astype(str) == candidate)
                & (power_fo["FinInstrmTp"].astype(str).str.upper() == "STF")
            ]
            if candidate_futures.empty:
                raise ValueError(f"Missing stock futures for {candidate} on {value.isoformat()}")
            expiry_values = pd.to_datetime(
                candidate_futures["FininstrmActlXpryDt"].fillna(candidate_futures["XpryDt"])
            ).dt.date
            futures_expiries[(candidate, value.isoformat())] = {
                expiry.isoformat() for expiry in expiry_values
            }
            original_expiry = pd.to_datetime(candidate_futures["XpryDt"], errors="coerce")
            actual_expiry = pd.to_datetime(candidate_futures["FininstrmActlXpryDt"], errors="coerce")
            futures_expiry_agreement[(candidate, value.isoformat())] = (
                100.0 * float((original_expiry.notna() & actual_expiry.notna() & (original_expiry == actual_expiry)).sum())
                / len(candidate_futures)
            )
    combined = pd.concat(option_frames, ignore_index=True)
    if set(combined["valuation_date"]) != {value.isoformat() for value in PANEL_DATES}:
        raise ValueError("Power panel date membership differs from the predeclared five Wednesdays")
    if set(combined["underlying"]) != set(POWER_CANDIDATES):
        raise ValueError("Tie-break ranking universe must be exactly NTPC and POWERGRID")
    return combined, futures_expiries, cm_spots, futures_expiry_agreement


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-raw-root", type=Path, default=CANONICAL_RAW_ROOT)
    parser.add_argument("--canonical-derived-root", type=Path, default=CANONICAL_DERIVED_ROOT)
    parser.add_argument("--extension-raw-root", type=Path, default=EXTENSION_RAW_ROOT)
    parser.add_argument("--extension-derived-root", type=Path, default=EXTENSION_DERIVED_ROOT)
    parser.add_argument("--evidence-path", type=Path, default=DEFAULT_EVIDENCE_PATH)
    parser.add_argument("--offline", action="store_true", help="Require the two extension archives locally.")
    args = parser.parse_args()

    _validate_original_stage_a_provenance(
        args.canonical_raw_root, args.canonical_derived_root
    )
    records = _acquire_extension_records(
        args.extension_raw_root, args.extension_derived_root, args.offline
    )
    options, futures_expiries, cm_spots, futures_expiry_agreement = _read_panel(
        args.canonical_raw_root, args.extension_raw_root
    )
    evidence = _evidence_rows(
        options, futures_expiries, cm_spots, futures_expiry_agreement
    )
    manifest_rows = _manifest_rows(records)
    manifest_fields = tuple(manifest_rows[0])
    _publish_csv_outputs_atomically(
        args.extension_derived_root / "acquisition_manifest.csv",
        manifest_rows,
        manifest_fields,
        args.evidence_path,
        evidence,
        EVIDENCE_FIELDS,
    )
    print(f"{EXTENSION_ID} dates={_date_text(PANEL_DATES)} candidates={'|'.join(POWER_CANDIDATES)}")
    print(f"added_archives={len(records)} option_rows={len(options)} evidence_rows={len(evidence)}")
    print(f"manifest={args.extension_derived_root / 'acquisition_manifest.csv'}")
    print(f"evidence={args.evidence_path}")


if __name__ == "__main__":
    main()
