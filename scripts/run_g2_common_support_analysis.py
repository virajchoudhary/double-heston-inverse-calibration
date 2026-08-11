"""Determine the balanced four-stock G2 common option-surface representation.

The analysis is deliberately confined to the canonical Stage A CM/FO files for
01, 15, and 22 July 2026.  It validates their manifest identities and hashes,
keeps the four securities separate, and writes only new ignored ``g2_*``
evidence.  It never acquires data, rewrites Stage A outputs, generates synthetic
surfaces, or trains a model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.nse_stage_a import (  # noqa: E402
    ArchiveIntegrityError,
    derive_option_observations,
    nse_archive_filename,
    nse_archive_url,
    read_udiff_csv,
)


ANALYSIS_ID = "G2_COMMON_SUPPORT"
G2_DATES = (date(2026, 7, 1), date(2026, 7, 15), date(2026, 7, 22))
PRIMARY_UNDERLYINGS = ("NTPC", "CIPLA", "INFY", "HDFCBANK")
POWER_TIEBREAK_DATES = (date(2026, 7, 8), date(2026, 7, 29))
BACKUP_UNDERLYINGS = ("POWERGRID", "SUNPHARMA", "TCS", "ICICIBANK")
REFERENCE_UNDERLYING = "NIFTY"

MONEYNESS_CANDIDATE_NODES = (
    -0.30,
    -0.20,
    -0.15,
    -0.10,
    -0.05,
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
)
G2_VERDICT = "NOT_PASSED"
PROPOSED_MONEYNESS_NODES = (-0.10, -0.05, 0.00, 0.05, 0.10)
PROPOSED_EXPIRY_POSITIONS = ("near", "mid")
PROPOSED_OPTION_TYPES = ("CE", "PE")
PROPOSED_PRICE_INPUT_DIMENSION = 20
PROPOSED_MATURITY_COORDINATE_DIMENSION = 2
PROPOSED_SURFACE_INPUT_DIMENSION = 22
FINAL_TOTAL_INPUT_DIMENSION: int | None = None
CARRY_CONDITIONING_STATUS = "UNRESOLVED"

# The rule is declared independently of any convenient network width.  A price
# node must be bracketed everywhere, use no wider than one 0.05 grid interval,
# and have activity-positive bracketing on at least three quarters of every
# balanced support view.  Maturity positions must be direct listed expiries;
# at least two are retained so the inverse problem sees term structure.
MIN_ACTIVE_SUPPORT_PCT = 75.0
MAX_LOG_MONEYNESS_BRACKET_WIDTH = 0.05
MIN_TERM_STRUCTURE_POSITIONS = 2
DIRECT_MONEYNESS_TOLERANCE = 1e-12

FIXED_DTE_TARGETS = (7, 14, 27, 30, 34, 41, 45, 55, 60, 69, 76, 90, 180)
CANONICAL_RAW_ROOT = REPOSITORY_ROOT / "market_data_audit" / "stage_a" / "raw" / "nse"
CANONICAL_DERIVED_ROOT = REPOSITORY_ROOT / "market_data_audit" / "stage_a" / "derived"
DEFAULT_REPORT_PATH = REPOSITORY_ROOT / "docs" / "G2_COMMON_SUPPORT_ANALYSIS.md"
DEFAULT_FIGURE_ROOT = CANONICAL_DERIVED_ROOT / "g2_figures"

CANONICAL_STAGE_A_OUTPUTS = (
    "acquisition_manifest.csv",
    "surface_summary.csv",
    "expiry_coverage.csv",
    "moneyness_coverage.csv",
    "candidate_grid_support.csv",
    "futures_availability.csv",
    "spot_consistency.csv",
    "universe_presence.csv",
)

CSV_OUTPUT_NAMES = {
    "maturity": "g2_maturity_support.csv",
    "moneyness": "g2_moneyness_support.csv",
    "surface": "g2_surface_support.csv",
    "candidates": "g2_representation_candidates.csv",
}


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    representation_type: str
    maturity_nodes: tuple[int | str, ...]
    moneyness_nodes: tuple[float, ...]
    option_types: tuple[str, ...]
    include_maturity_coordinates: bool
    maturity_interpolation_contract_available: bool
    rationale: str


CANDIDATE_SPECS = (
    CandidateSpec(
        "provisional_108",
        "FIXED_DTE",
        (7, 14, 30, 60, 90, 180),
        (-0.30, -0.20, -0.10, -0.05, 0.00, 0.05, 0.10, 0.20, 0.30),
        ("CE", "PE"),
        False,
        False,
        "Rejected Stage A grid retained as the baseline comparison.",
    ),
    CandidateSpec(
        "fixed_30_60_central5",
        "FIXED_DTE",
        (30, 60),
        PROPOSED_MONEYNESS_NODES,
        PROPOSED_OPTION_TYPES,
        False,
        False,
        "No extrapolation, but every maturity is interpolated and no validated maturity-price interpolation contract exists.",
    ),
    CandidateSpec(
        "fixed_34_central5",
        "FIXED_DTE",
        (34,),
        PROPOSED_MONEYNESS_NODES,
        PROPOSED_OPTION_TYPES,
        False,
        False,
        "Best-supported single fixed DTE is too small to retain term-structure information.",
    ),
    CandidateSpec(
        "fixed_34_69_central5",
        "FIXED_DTE",
        (34, 69),
        PROPOSED_MONEYNESS_NODES,
        PROPOSED_OPTION_TYPES,
        False,
        False,
        "Data-derived fixed nodes avoid extrapolation but still require maturity interpolation on two thirds of surfaces.",
    ),
    CandidateSpec(
        "relative_near_central5",
        "RELATIVE_EXPIRY",
        ("near",),
        PROPOSED_MONEYNESS_NODES,
        PROPOSED_OPTION_TYPES,
        True,
        True,
        "Direct and fully active, but one expiry position loses term structure needed by the inverse objective.",
    ),
    CandidateSpec(
        "relative_near_mid_central3",
        "RELATIVE_EXPIRY",
        PROPOSED_EXPIRY_POSITIONS,
        (-0.05, 0.00, 0.05),
        PROPOSED_OPTION_TYPES,
        True,
        True,
        "Passes the rule but is strictly contained in the five-node candidate that also passes.",
    ),
    CandidateSpec(
        "relative_near_mid_central5",
        "RELATIVE_EXPIRY",
        PROPOSED_EXPIRY_POSITIONS,
        PROPOSED_MONEYNESS_NODES,
        PROPOSED_OPTION_TYPES,
        True,
        True,
        "Largest useful symmetric near/middle representation satisfying every declared rule.",
    ),
    CandidateSpec(
        "relative_near_mid_central7",
        "RELATIVE_EXPIRY",
        PROPOSED_EXPIRY_POSITIONS,
        (-0.15, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15),
        PROPOSED_OPTION_TYPES,
        True,
        True,
        "Adds unsupported wings and therefore fails common observed support.",
    ),
    CandidateSpec(
        "relative_near_mid_far_central5",
        "RELATIVE_EXPIRY",
        ("near", "mid", "far"),
        PROPOSED_MONEYNESS_NODES,
        PROPOSED_OPTION_TYPES,
        True,
        True,
        "Direct maturities, but the far-position activity floor is below the declared threshold.",
    ),
    CandidateSpec(
        "relative_near_mid_call_only",
        "RELATIVE_EXPIRY",
        PROPOSED_EXPIRY_POSITIONS,
        PROPOSED_MONEYNESS_NODES,
        ("CE",),
        True,
        True,
        "Smaller, but removing puts would impose an unfrozen carry/parity contract on market preprocessing.",
    ),
    CandidateSpec(
        "relative_near_mid_put_only",
        "RELATIVE_EXPIRY",
        PROPOSED_EXPIRY_POSITIONS,
        PROPOSED_MONEYNESS_NODES,
        ("PE",),
        True,
        True,
        "Smaller, but removing calls would discard independently observed market information without sufficient evidence.",
    ),
)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def snapshot_canonical_outputs(derived_root: Path) -> dict[str, str]:
    """Return hashes for the exact eight Stage A outputs, failing if any is absent."""
    result: dict[str, str] = {}
    for name in CANONICAL_STAGE_A_OUTPUTS:
        path = derived_root / name
        if not path.is_file():
            raise ArchiveIntegrityError(f"Canonical Stage A output is missing: {path}")
        result[name] = _sha256_file(path)
    return result


def assert_canonical_outputs_preserved(
    derived_root: Path, baseline: Mapping[str, str]
) -> None:
    observed = snapshot_canonical_outputs(derived_root)
    if observed != dict(baseline):
        changed = sorted(name for name in baseline if observed.get(name) != baseline[name])
        raise ArchiveIntegrityError(
            "Canonical Stage A outputs changed during G2 analysis: " + ", ".join(changed)
        )


def validate_canonical_stage_a_provenance(
    raw_root: Path, derived_root: Path
) -> tuple[dict[str, str], ...]:
    """Validate exact manifest identities, paths, archive/member bytes, and CSV hashes."""
    manifest_path = derived_root / "acquisition_manifest.csv"
    if not manifest_path.is_file():
        raise ArchiveIntegrityError(f"Canonical Stage A manifest is missing: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_identities = {
        (market, value.isoformat()) for value in G2_DATES for market in ("CM", "FO")
    }
    identities = [
        (str(row.get("market", "")).strip(), str(row.get("valuation_date", "")).strip())
        for row in rows
    ]
    if len(identities) != len(set(identities)):
        raise ArchiveIntegrityError("Canonical Stage A manifest contains duplicate identities.")
    if set(identities) != expected_identities or len(rows) != len(expected_identities):
        raise ArchiveIntegrityError(
            "Canonical Stage A manifest identities must be exactly CM/FO for the three G2 dates."
        )

    by_identity = {identity: row for identity, row in zip(identities, rows, strict=True)}
    validated: list[dict[str, str]] = []
    for value in G2_DATES:
        for market in ("CM", "FO"):
            identity = (market, value.isoformat())
            row = by_identity[identity]
            filename = nse_archive_filename(market, value)
            archive_path = raw_root / value.isoformat() / filename
            csv_path = archive_path.with_suffix("")
            expected = {
                "market": market,
                "valuation_date": value.isoformat(),
                "official_url": nse_archive_url(market, value),
                "original_filename": filename,
                "archive_member_name": filename.removesuffix(".zip"),
            }
            for field, required in expected.items():
                if str(row.get(field, "")).strip() != required:
                    raise ArchiveIntegrityError(
                        f"Canonical manifest {market}/{value.isoformat()} has unexpected {field}."
                    )
            for field, required_path in (("archive_path", archive_path), ("csv_path", csv_path)):
                recorded = str(row.get(field, "")).strip()
                if not recorded or Path(recorded).resolve() != required_path.resolve():
                    raise ArchiveIntegrityError(
                        f"Canonical manifest {market}/{value.isoformat()} has unexpected {field}."
                    )
            if str(row.get("zip_integrity", "")).strip().lower() != "true":
                raise ArchiveIntegrityError(
                    f"Canonical manifest {market}/{value.isoformat()} does not record ZIP integrity."
                )
            if not archive_path.is_file() or not csv_path.is_file():
                raise ArchiveIntegrityError(
                    f"Canonical manifest {market}/{value.isoformat()} points to missing raw evidence."
                )
            archive_bytes = archive_path.read_bytes()
            csv_bytes = csv_path.read_bytes()
            if str(len(archive_bytes)) != str(row.get("archive_size_bytes", "")).strip():
                raise ArchiveIntegrityError(
                    f"Canonical archive size mismatch for {market}/{value.isoformat()}."
                )
            if _sha256_bytes(archive_bytes) != str(row.get("archive_sha256", "")).strip().lower():
                raise ArchiveIntegrityError(
                    f"Canonical archive_sha256 mismatch for {market}/{value.isoformat()}."
                )
            if _sha256_bytes(csv_bytes) != str(row.get("csv_sha256", "")).strip().lower():
                raise ArchiveIntegrityError(
                    f"Canonical extracted CSV hash mismatch for {market}/{value.isoformat()}."
                )
            try:
                with zipfile.ZipFile(archive_path, "r") as archive:
                    if archive.testzip() is not None:
                        raise ArchiveIntegrityError(
                            f"Canonical ZIP integrity failure for {market}/{value.isoformat()}."
                        )
                    names = archive.namelist()
                    if names != [filename.removesuffix(".zip")]:
                        raise ArchiveIntegrityError(
                            f"Canonical ZIP member identity mismatch for {market}/{value.isoformat()}."
                        )
                    if archive.read(names[0]) != csv_bytes:
                        raise ArchiveIntegrityError(
                            f"Canonical ZIP member bytes differ from extracted CSV for {market}/{value.isoformat()}."
                        )
            except zipfile.BadZipFile as exc:
                raise ArchiveIntegrityError(
                    f"Canonical archive is not a valid ZIP for {market}/{value.isoformat()}."
                ) from exc
            validated.append(dict(row))
    return tuple(validated)


def load_balanced_panel(raw_root: Path) -> pd.DataFrame:
    """Read exactly the four primary stocks and three canonical valuation dates."""
    frames: list[pd.DataFrame] = []
    for value in G2_DATES:
        paths = {
            market: raw_root
            / value.isoformat()
            / nse_archive_filename(market, value).removesuffix(".zip")
            for market in ("CM", "FO")
        }
        cm = read_udiff_csv(paths["CM"], value, "CM")
        fo = read_udiff_csv(paths["FO"], value, "FO")
        scoped_fo = fo.loc[fo["TckrSymb"].astype(str).isin(PRIMARY_UNDERLYINGS)].copy()
        observations = derive_option_observations(scoped_fo, cm, value)
        frames.append(observations.loc[observations["underlying"].isin(PRIMARY_UNDERLYINGS)].copy())
    panel = pd.concat(frames, ignore_index=True)
    validate_balanced_panel(panel)
    panel["active_positive"] = (
        panel["traded_qty_positive"]
        | panel["open_interest_positive"]
        | panel["transactions_positive"]
    )
    panel["price_observed_positive"] = (
        panel["close_positive"] | panel["last_positive"] | panel["settlement_positive"]
    )
    return panel


def validate_balanced_panel(panel: pd.DataFrame) -> None:
    observed_stocks = set(panel["underlying"].astype(str))
    observed_dates = set(panel["valuation_date"].astype(str))
    expected_dates = {value.isoformat() for value in G2_DATES}
    if observed_stocks != set(PRIMARY_UNDERLYINGS):
        raise ValueError("G2 universe must be exactly NTPC, CIPLA, INFY, and HDFCBANK.")
    if observed_dates != expected_dates:
        raise ValueError("G2 panel must contain exactly the three canonical Stage A dates.")
    surfaces = panel[["underlying", "valuation_date"]].drop_duplicates()
    if len(surfaces) != len(PRIMARY_UNDERLYINGS) * len(G2_DATES):
        raise ValueError("G2 panel must contain exactly 12 stock/date surfaces.")
    if observed_stocks.intersection(BACKUP_UNDERLYINGS) or REFERENCE_UNDERLYING in observed_stocks:
        raise ValueError("Backups and NIFTY cannot enter G2 representation selection.")
    if observed_dates.intersection(value.isoformat() for value in POWER_TIEBREAK_DATES):
        raise ValueError("Power-only tie-break dates cannot enter the balanced G2 panel.")
    counts = panel.groupby(["underlying", "valuation_date"])["actual_expiry"].nunique()
    if not (counts == 3).all():
        raise ValueError("Every G2 stock/date surface must have exactly three listed expiries.")


def build_surface_support(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["underlying", "valuation_date", "actual_expiry", "DTE", "T", "expiry_slot", "OptnTp"]
    for key, group in panel.groupby(keys, sort=True):
        observed = pd.to_numeric(group["log_K_over_S"], errors="coerce").dropna()
        active = pd.to_numeric(
            group.loc[group["active_positive"], "log_K_over_S"], errors="coerce"
        ).dropna()
        positive = group.loc[group["price_observed_positive"]]
        row = dict(zip(keys, key, strict=True))
        row.update(
            {
                "option_type": row.pop("OptnTp"),
                "row_count": len(group),
                "unique_strike_count": int(group["strike"].nunique()),
                "price_observed_positive_count": len(positive),
                "price_observed_positive_pct": 100.0 * len(positive) / len(group),
                "active_positive_count": int(group["active_positive"].sum()),
                "active_positive_pct": 100.0 * float(group["active_positive"].mean()),
                "observed_min_log_moneyness": float(observed.min()),
                "observed_max_log_moneyness": float(observed.max()),
                "active_min_log_moneyness": float(active.min()) if len(active) else math.nan,
                "active_max_log_moneyness": float(active.max()) if len(active) else math.nan,
                "bid_available": False,
                "ask_available": False,
                "historical_bid_ask_inferred": False,
            }
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["underlying", "valuation_date", "DTE", "option_type"], kind="stable"
    ).reset_index(drop=True)


def _node_support(values: np.ndarray, node: float) -> dict[str, object]:
    clean = np.sort(np.unique(values[np.isfinite(values)]))
    if clean.size == 0:
        return {
            "classification": "UNSUPPORTED",
            "lower_log_moneyness": math.nan,
            "upper_log_moneyness": math.nan,
            "nearest_distance": math.nan,
            "bracket_width": math.nan,
            "inside_observed_bounds": False,
            "extrapolation_required": True,
        }
    distance = np.abs(clean - node)
    if float(distance.min()) <= DIRECT_MONEYNESS_TOLERANCE:
        exact = float(clean[int(distance.argmin())])
        return {
            "classification": "DIRECT",
            "lower_log_moneyness": exact,
            "upper_log_moneyness": exact,
            "nearest_distance": float(distance.min()),
            "bracket_width": 0.0,
            "inside_observed_bounds": True,
            "extrapolation_required": False,
        }
    lower = clean[clean < node]
    upper = clean[clean > node]
    if lower.size and upper.size:
        lo, hi = float(lower[-1]), float(upper[0])
        return {
            "classification": "INTERPOLATED",
            "lower_log_moneyness": lo,
            "upper_log_moneyness": hi,
            "nearest_distance": min(node - lo, hi - node),
            "bracket_width": hi - lo,
            "inside_observed_bounds": True,
            "extrapolation_required": False,
        }
    return {
        "classification": "UNSUPPORTED",
        "lower_log_moneyness": float(lower[-1]) if lower.size else math.nan,
        "upper_log_moneyness": float(upper[0]) if upper.size else math.nan,
        "nearest_distance": float(distance.min()),
        "bracket_width": math.nan,
        "inside_observed_bounds": False,
        "extrapolation_required": True,
    }


def build_moneyness_support(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["underlying", "valuation_date", "actual_expiry", "DTE", "T", "expiry_slot", "OptnTp"]
    for key, group in panel.groupby(keys, sort=True):
        close_positive = group["close_positive"]
        observed_values = pd.to_numeric(
            group.loc[close_positive, "log_K_over_S"], errors="coerce"
        ).to_numpy(float)
        active_values = pd.to_numeric(
            group.loc[close_positive & group["active_positive"], "log_K_over_S"],
            errors="coerce",
        ).to_numpy(float)
        for node in MONEYNESS_CANDIDATE_NODES:
            observed = _node_support(observed_values, node)
            active = _node_support(active_values, node)
            row = dict(zip(keys, key, strict=True))
            row["option_type"] = row.pop("OptnTp")
            row.update(
                {
                    "log_moneyness_node": node,
                    **observed,
                    "active_classification": active["classification"],
                    "active_lower_log_moneyness": active["lower_log_moneyness"],
                    "active_upper_log_moneyness": active["upper_log_moneyness"],
                    "active_nearest_distance": active["nearest_distance"],
                    "active_bracket_width": active["bracket_width"],
                    "active_inside_observed_bounds": active["inside_observed_bounds"],
                    "active_extrapolation_required": active["extrapolation_required"],
                    "strike_interpolation_rule": "adjacent listed strikes; linear normalized price in K/S",
                    "strike_extrapolation_authorized": False,
                    "market_price_source": "ClsPric",
                    "market_price_normalization": "ClsPric/cm_spot",
                }
            )
            rows.append(row)
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["log_moneyness_node", "underlying", "valuation_date", "DTE", "option_type"],
        kind="stable",
    ).reset_index(drop=True)


def _fixed_maturity_support(dtes: np.ndarray, target: int) -> dict[str, object]:
    clean = np.sort(np.unique(dtes.astype(int)))
    distance = np.abs(clean - target)
    if target in clean:
        return {
            "classification": "DIRECT",
            "lower_DTE": target,
            "upper_DTE": target,
            "nearest_distance_days": 0,
            "bracket_span_days": 0,
            "inside_observed_expiry_bounds": True,
            "extrapolation_required": False,
        }
    lower = clean[clean < target]
    upper = clean[clean > target]
    if lower.size and upper.size:
        lo, hi = int(lower[-1]), int(upper[0])
        return {
            "classification": "INTERPOLATED",
            "lower_DTE": lo,
            "upper_DTE": hi,
            "nearest_distance_days": int(min(target - lo, hi - target)),
            "bracket_span_days": hi - lo,
            "inside_observed_expiry_bounds": True,
            "extrapolation_required": False,
        }
    return {
        "classification": "UNSUPPORTED",
        "lower_DTE": int(lower[-1]) if lower.size else math.nan,
        "upper_DTE": int(upper[0]) if upper.size else math.nan,
        "nearest_distance_days": int(distance.min()),
        "bracket_span_days": math.nan,
        "inside_observed_expiry_bounds": False,
        "extrapolation_required": True,
    }


def build_maturity_support(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (underlying, valuation_date), group in panel.groupby(
        ["underlying", "valuation_date"], sort=True
    ):
        dtes = np.sort(group["DTE"].unique().astype(int))
        for target in FIXED_DTE_TARGETS:
            rows.append(
                {
                    "representation_type": "FIXED_DTE",
                    "underlying": underlying,
                    "valuation_date": valuation_date,
                    "target": str(target),
                    "actual_DTE_values": "|".join(str(value) for value in dtes),
                    **_fixed_maturity_support(dtes, target),
                    "maturity_interpolation_rule": "NONE_VALIDATED",
                    "maturity_extrapolation_authorized": False,
                }
            )
        slots = (
            group[["expiry_slot", "DTE", "actual_expiry"]]
            .drop_duplicates()
            .sort_values("DTE", kind="stable")
        )
        for item in slots.itertuples(index=False):
            rows.append(
                {
                    "representation_type": "RELATIVE_EXPIRY",
                    "underlying": underlying,
                    "valuation_date": valuation_date,
                    "target": str(item.expiry_slot),
                    "actual_DTE_values": "|".join(str(value) for value in dtes),
                    "classification": "DIRECT",
                    "lower_DTE": int(item.DTE),
                    "upper_DTE": int(item.DTE),
                    "nearest_distance_days": 0,
                    "bracket_span_days": 0,
                    "inside_observed_expiry_bounds": True,
                    "extrapolation_required": False,
                    "maturity_interpolation_rule": "NONE_REQUIRED",
                    "maturity_extrapolation_authorized": False,
                }
            )
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["representation_type", "target", "underlying", "valuation_date"], kind="stable"
    ).reset_index(drop=True)


def _minimum_group_support(
    frame: pd.DataFrame, available_column: str, grouping: Sequence[str]
) -> float:
    if frame.empty:
        return 0.0
    return 100.0 * float(frame.groupby(list(grouping))[available_column].mean().min())


def _candidate_expiry_slots(spec: CandidateSpec, maturity: pd.DataFrame) -> tuple[str, ...]:
    if spec.representation_type == "RELATIVE_EXPIRY":
        return tuple(str(value) for value in spec.maturity_nodes)
    slots: set[str] = set()
    for target in spec.maturity_nodes:
        fixed = maturity.loc[
            (maturity["representation_type"] == "FIXED_DTE")
            & (maturity["target"] == str(target))
        ]
        for row in fixed.itertuples(index=False):
            actual = [int(value) for value in str(row.actual_DTE_values).split("|")]
            names = dict(zip(actual, ("near", "mid", "far"), strict=True))
            if not pd.isna(row.lower_DTE) and int(row.lower_DTE) in names:
                slots.add(names[int(row.lower_DTE)])
            if not pd.isna(row.upper_DTE) and int(row.upper_DTE) in names:
                slots.add(names[int(row.upper_DTE)])
    return tuple(value for value in ("near", "mid", "far") if value in slots)


def select_maximal_passing_geometry(
    candidates: pd.DataFrame,
    specs: Sequence[CandidateSpec] = CANDIDATE_SPECS,
) -> str:
    """Select the unique maximal passing moneyness set without using feature count."""
    passing_ids = set(
        candidates.loc[candidates["decision_rule_pass"], "candidate_id"].astype(str)
    )
    by_id = {spec.candidate_id: spec for spec in specs}
    unknown = passing_ids.difference(by_id)
    if unknown:
        raise RuntimeError(f"Passing candidates lack specifications: {sorted(unknown)}")

    def strictly_contains(candidate: CandidateSpec, other: CandidateSpec) -> bool:
        same_family = (
            candidate.representation_type == other.representation_type
            and candidate.maturity_nodes == other.maturity_nodes
            and candidate.option_types == other.option_types
            and candidate.include_maturity_coordinates
            == other.include_maturity_coordinates
        )
        return same_family and set(candidate.moneyness_nodes) > set(other.moneyness_nodes)

    maximal = []
    for candidate_id in sorted(passing_ids):
        candidate = by_id[candidate_id]
        if not any(
            strictly_contains(by_id[other_id], candidate)
            for other_id in passing_ids
            if other_id != candidate_id
        ):
            maximal.append(candidate_id)
    if len(maximal) != 1:
        raise RuntimeError(
            "Geometry rule is unresolved; expected one maximal passing candidate but found "
            + ", ".join(maximal)
        )
    return maximal[0]


def build_representation_candidates(
    maturity: pd.DataFrame, moneyness: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in CANDIDATE_SPECS:
        if spec.representation_type == "FIXED_DTE":
            maturity_rows = maturity.loc[
                (maturity["representation_type"] == "FIXED_DTE")
                & maturity["target"].isin(str(value) for value in spec.maturity_nodes)
            ].copy()
        else:
            maturity_rows = maturity.loc[
                (maturity["representation_type"] == "RELATIVE_EXPIRY")
                & maturity["target"].isin(str(value) for value in spec.maturity_nodes)
            ].copy()
        total_maturity = len(maturity_rows)
        maturity_counts = maturity_rows["classification"].value_counts()
        direct_pct = 100.0 * int(maturity_counts.get("DIRECT", 0)) / total_maturity
        interpolated_pct = 100.0 * int(maturity_counts.get("INTERPOLATED", 0)) / total_maturity
        unsupported_pct = 100.0 * int(maturity_counts.get("UNSUPPORTED", 0)) / total_maturity
        slots = _candidate_expiry_slots(spec, maturity)
        option_rows = moneyness.loc[
            moneyness["expiry_slot"].isin(slots)
            & moneyness["log_moneyness_node"].isin(spec.moneyness_nodes)
            & moneyness["option_type"].isin(spec.option_types)
        ].copy()
        option_rows["observed_available"] = option_rows["classification"].isin(
            ["DIRECT", "INTERPOLATED"]
        )
        option_rows["active_available"] = option_rows["active_classification"].isin(
            ["DIRECT", "INTERPOLATED"]
        )
        observed_pct = 100.0 * float(option_rows["observed_available"].mean())
        active_pct = 100.0 * float(option_rows["active_available"].mean())
        worst_stock = _minimum_group_support(
            option_rows, "active_available", ["log_moneyness_node", "underlying"]
        )
        worst_date = _minimum_group_support(
            option_rows, "active_available", ["log_moneyness_node", "valuation_date"]
        )
        worst_position = _minimum_group_support(
            option_rows, "active_available", ["log_moneyness_node", "expiry_slot"]
        )
        worst_option_type = _minimum_group_support(
            option_rows, "active_available", ["log_moneyness_node", "option_type"]
        )
        bracket_width = pd.to_numeric(
            option_rows.loc[option_rows["classification"] == "INTERPOLATED", "bracket_width"],
            errors="coerce",
        )
        max_width = float(bracket_width.max()) if bracket_width.notna().any() else 0.0
        no_maturity_extrapolation = unsupported_pct == 0.0
        no_strike_extrapolation = observed_pct == 100.0
        activity_rule = min(
            active_pct, worst_stock, worst_date, worst_position, worst_option_type
        ) >= MIN_ACTIVE_SUPPORT_PCT
        bracket_rule = max_width <= MAX_LOG_MONEYNESS_BRACKET_WIDTH + 1e-12
        term_structure_rule = len(spec.maturity_nodes) >= MIN_TERM_STRUCTURE_POSITIONS
        maturity_rule = (
            no_maturity_extrapolation
            and (
                spec.representation_type == "RELATIVE_EXPIRY"
                or (
                    interpolated_pct == 0.0
                    and spec.maturity_interpolation_contract_available
                )
            )
        )
        both_types_rule = spec.option_types == PROPOSED_OPTION_TYPES
        rule_pass = all(
            (
                maturity_rule,
                no_strike_extrapolation,
                activity_rule,
                bracket_rule,
                term_structure_rule,
                both_types_rule,
            )
        )
        price_features = len(spec.maturity_nodes) * len(spec.moneyness_nodes) * len(spec.option_types)
        coordinate_features = (
            len(spec.maturity_nodes) if spec.include_maturity_coordinates else 0
        )
        nearest = pd.to_numeric(maturity_rows["nearest_distance_days"], errors="coerce")
        spans = pd.to_numeric(maturity_rows["bracket_span_days"], errors="coerce")
        rows.append(
            {
                "candidate_id": spec.candidate_id,
                "representation_type": spec.representation_type,
                "maturity_nodes": "|".join(str(value) for value in spec.maturity_nodes),
                "moneyness_nodes": "|".join(f"{value:+.2f}" for value in spec.moneyness_nodes),
                "option_types": "|".join(spec.option_types),
                "normalized_price_feature_count": price_features,
                "coordinate_feature_count": coordinate_features,
                "total_input_dimension": price_features + coordinate_features,
                "maturity_direct_pct": direct_pct,
                "maturity_interpolated_pct": interpolated_pct,
                "maturity_unsupported_pct": unsupported_pct,
                "max_maturity_nearest_distance_days": float(nearest.max()),
                "max_maturity_bracket_span_days": float(spans.max()) if spans.notna().any() else math.nan,
                "moneyness_observed_support_pct": observed_pct,
                "moneyness_direct_support_pct": 100.0
                * float((option_rows["classification"] == "DIRECT").mean()),
                "moneyness_interpolated_support_pct": 100.0
                * float((option_rows["classification"] == "INTERPOLATED").mean()),
                "moneyness_active_support_pct": active_pct,
                "worst_stock_active_support_pct": worst_stock,
                "worst_date_active_support_pct": worst_date,
                "worst_expiry_position_active_support_pct": worst_position,
                "worst_option_type_active_support_pct": worst_option_type,
                "max_log_moneyness_bracket_width": max_width,
                "no_maturity_extrapolation_pass": no_maturity_extrapolation,
                "no_strike_extrapolation_pass": no_strike_extrapolation,
                "maturity_policy_pass": maturity_rule,
                "activity_rule_pass": activity_rule,
                "bracket_width_rule_pass": bracket_rule,
                "term_structure_rule_pass": term_structure_rule,
                "both_option_types_rule_pass": both_types_rule,
                "decision_rule_pass": rule_pass,
                "carry_conditioning_rule_pass": False,
                "g2_gate_pass": False,
                "rationale": spec.rationale,
            }
        )
    result = pd.DataFrame(rows)
    proposed_id = select_maximal_passing_geometry(result)
    result["decision"] = "REJECTED"
    result.loc[result["decision_rule_pass"], "decision"] = "PASS_BUT_DOMINATED"
    result.loc[result["candidate_id"] == proposed_id, "decision"] = "PROPOSED_GEOMETRY"
    proposed_row = result.loc[result["candidate_id"] == proposed_id].iloc[0]
    if (
        int(proposed_row["normalized_price_feature_count"])
        != PROPOSED_PRICE_INPUT_DIMENSION
        or int(proposed_row["coordinate_feature_count"])
        != PROPOSED_MATURITY_COORDINATE_DIMENSION
        or int(proposed_row["total_input_dimension"])
        != PROPOSED_SURFACE_INPUT_DIMENSION
    ):
        raise RuntimeError("Proposed G2 surface subtotal differs from the declared geometry.")
    if bool(result["g2_gate_pass"].any()) or G2_VERDICT != "NOT_PASSED":
        raise RuntimeError("Carry conditioning is unresolved, so G2 must not pass.")
    return result


def _serialize_csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.12g").encode("utf-8")


def publish_evidence_atomically(
    frames: Mapping[str, pd.DataFrame], output_root: Path
) -> dict[str, Path]:
    """Publish the four G2 CSVs from a same-directory staging tree."""
    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".g2_outputs_", dir=output_root))
    paths: dict[str, Path] = {}
    try:
        for key, filename in CSV_OUTPUT_NAMES.items():
            if key not in frames:
                raise ValueError(f"Missing G2 evidence frame: {key}")
            staged = staging / filename
            staged.write_bytes(_serialize_csv(frames[key]))
            paths[key] = output_root / filename
        for key, destination in paths.items():
            os.replace(staging / CSV_OUTPUT_NAMES[key], destination)
    finally:
        for filename in CSV_OUTPUT_NAMES.values():
            (staging / filename).unlink(missing_ok=True)
        staging.rmdir()
    return paths


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_figures(
    maturity: pd.DataFrame,
    moneyness: pd.DataFrame,
    candidates: pd.DataFrame,
    figure_root: Path,
) -> tuple[Path, ...]:
    figure_root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    fixed = maturity.loc[maturity["representation_type"] == "FIXED_DTE"].copy()
    fixed["supported"] = fixed["classification"].map(
        {"DIRECT": 2.0, "INTERPOLATED": 1.0, "UNSUPPORTED": 0.0}
    )
    matrix = fixed.pivot_table(
        index="target", columns="valuation_date", values="supported", aggfunc="mean"
    ).sort_index(key=lambda values: values.astype(int))
    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="RdYlGn", vmin=0, vmax=2)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_xlabel("Valuation date")
    ax.set_ylabel("Target DTE")
    ax.set_title("G2 fixed-DTE support (0 unsupported, 1 interpolated, 2 direct)")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    path = figure_root / "g2_maturity_support_heatmap.png"
    fig.savefig(path, dpi=160, metadata={"Software": "matplotlib"})
    plt.close(fig)
    paths.append(path)

    central = moneyness.loc[
        moneyness["expiry_slot"].isin(PROPOSED_EXPIRY_POSITIONS)
    ].copy()
    central["active_available"] = central["active_classification"].isin(
        ["DIRECT", "INTERPOLATED"]
    )
    matrix = (
        central.groupby(["log_moneyness_node", "expiry_slot"])["active_available"]
        .mean()
        .mul(100.0)
        .unstack()
        .reindex(columns=PROPOSED_EXPIRY_POSITIONS)
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=100)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns)
    ax.set_yticks(range(len(matrix.index)), [f"{value:+.2f}" for value in matrix.index])
    ax.set_xlabel("Listed expiry position")
    ax.set_ylabel("log(K/S) node")
    ax.set_title("G2 active bounded-strike support (%)")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    path = figure_root / "g2_moneyness_support_heatmap.png"
    fig.savefig(path, dpi=160, metadata={"Software": "matplotlib"})
    plt.close(fig)
    paths.append(path)

    comparison = candidates.set_index("candidate_id")
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(comparison))
    ax.bar(x - 0.2, comparison["moneyness_observed_support_pct"], 0.4, label="observed")
    ax.bar(x + 0.2, comparison["moneyness_active_support_pct"], 0.4, label="active")
    ax.axhline(MIN_ACTIVE_SUPPORT_PCT, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x, comparison.index, rotation=45, ha="right")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Support (%)")
    ax.set_title("G2 representation candidates")
    ax.legend()
    fig.tight_layout()
    path = figure_root / "g2_candidate_comparison.png"
    fig.savefig(path, dpi=160, metadata={"Software": "matplotlib"})
    plt.close(fig)
    paths.append(path)
    return tuple(paths)


def _pct(value: object) -> str:
    return f"{float(value):.1f}%"


def render_report(
    panel: pd.DataFrame,
    maturity: pd.DataFrame,
    moneyness: pd.DataFrame,
    candidates: pd.DataFrame,
    canonical_hashes: Mapping[str, str],
) -> str:
    fixed = maturity.loc[maturity["representation_type"] == "FIXED_DTE"].copy()
    fixed_summary = []
    for target, group in fixed.groupby("target", sort=False):
        counts = group["classification"].value_counts()
        fixed_summary.append(
            (
                int(target),
                int(counts.get("DIRECT", 0)),
                int(counts.get("INTERPOLATED", 0)),
                int(counts.get("UNSUPPORTED", 0)),
                int(group["nearest_distance_days"].max()),
                int(pd.to_numeric(group["bracket_span_days"], errors="coerce").max())
                if pd.to_numeric(group["bracket_span_days"], errors="coerce").notna().any()
                else "—",
            )
        )
    fixed_summary.sort()

    relative = maturity.loc[maturity["representation_type"] == "RELATIVE_EXPIRY"]
    relative_summary = (
        relative.groupby("target")["lower_DTE"]
        .agg(["min", "max", lambda values: "|".join(str(int(value)) for value in sorted(set(values)))])
        .rename(columns={"<lambda_0>": "values"})
    )

    selected_scope = moneyness.loc[
        moneyness["expiry_slot"].isin(PROPOSED_EXPIRY_POSITIONS)
        & moneyness["log_moneyness_node"].isin(MONEYNESS_CANDIDATE_NODES)
    ].copy()
    selected_scope["observed"] = selected_scope["classification"].isin(
        ["DIRECT", "INTERPOLATED"]
    )
    selected_scope["active"] = selected_scope["active_classification"].isin(
        ["DIRECT", "INTERPOLATED"]
    )
    moneyness_rows = []
    for node, group in selected_scope.groupby("log_moneyness_node", sort=True):
        interpolated = pd.to_numeric(
            group.loc[group["classification"] == "INTERPOLATED", "bracket_width"],
            errors="coerce",
        )
        moneyness_rows.append(
            (
                node,
                100.0 * float(group["observed"].mean()),
                100.0 * float((group["classification"] == "DIRECT").mean()),
                100.0 * float(group["active"].mean()),
                _minimum_group_support(group, "active", ["underlying"]),
                _minimum_group_support(group, "active", ["valuation_date"]),
                _minimum_group_support(group, "active", ["expiry_slot"]),
                _minimum_group_support(group, "active", ["option_type"]),
                float(interpolated.max()) if interpolated.notna().any() else math.nan,
            )
        )

    proposed = candidates.loc[candidates["decision"] == "PROPOSED_GEOMETRY"].iloc[0]
    close_positive_pct = 100.0 * float(panel["close_positive"].mean())
    last_positive_pct = 100.0 * float(panel["last_positive"].mean())
    settlement_positive_pct = 100.0 * float(panel["settlement_positive"].mean())
    far = moneyness.loc[
        (moneyness["expiry_slot"] == "far")
        & moneyness["log_moneyness_node"].isin(PROPOSED_MONEYNESS_NODES)
    ].copy()
    far["active"] = far["active_classification"].isin(["DIRECT", "INTERPOLATED"])
    far_range = (
        100.0 * far.groupby("log_moneyness_node")["active"].mean().min(),
        100.0 * far.groupby("log_moneyness_node")["active"].mean().max(),
    )

    lines = [
        "# G2 Common-Support Analysis",
        "",
        "## Gate result",
        "",
        "**G2 = NOT_PASSED.** The market-support evidence uniquely proposes two direct listed expiry positions "
        "(`near`, `middle`), five symmetric log-moneyness nodes `[-0.10, -0.05, 0.00, +0.05, +0.10]`, "
        "and both calls and puts. That surface geometry contains 20 normalized-price features plus the two actual maturity "
        "coordinates `T_near` and `T_middle`, a **22-feature surface subtotal**. The final inverse input dimension is not frozen because "
        "the conditioning treatment for risk-free rate and dividend/carry remains unresolved.",
        "",
        "This is a representation decision only. No final 10,000-surface dataset, ANN/PINN training, or later gate was run.",
        "",
        "## Balanced market panel and provenance",
        "",
        f"The analysis used exactly `{len(PRIMARY_UNDERLYINGS)}` stocks × `{len(G2_DATES)}` dates = `12` stock/date surfaces, "
        f"`{panel.groupby(['underlying','valuation_date','actual_expiry']).ngroups}` expiry slices, and `{len(panel):,}` option rows:",
        "",
        "- stocks: `NTPC | CIPLA | INFY | HDFCBANK`;",
        "- dates: `2026-07-01 | 2026-07-15 | 2026-07-22`; and",
        "- source: only the six canonical official-NSE CM/FO manifest identities.",
        "",
        "The Power-only dates `2026-07-08` and `2026-07-29`, all backup stocks, NIFTY, and Bloomberg were excluded. "
        "Securities were never pooled or averaged. Before analysis, archive identity, official URL, archive/member bytes, "
        "archive SHA-256, extracted CSV SHA-256, schema, date, and market were checked fail-closed against the canonical manifest.",
        "",
        "## Market-support decision rule declared before proposing geometry",
        "",
        "A geometry candidate passes market support only when all of the following hold:",
        "",
        "1. all four stocks and all three dates are present with no excluded date/security;",
        "2. no maturity or strike extrapolation is required;",
        "3. maturity nodes are direct listed-expiry positions unless a separately validated maturity-price interpolation contract exists;",
        f"4. at least `{MIN_TERM_STRUCTURE_POSITIONS}` maturity positions remain, preserving term-structure information;",
        "5. every retained moneyness node is bracketed for every stock/date/expiry-position/option-type slice;",
        f"6. active bracketing is at least `{MIN_ACTIVE_SUPPORT_PCT:.0f}%` overall and in the worst stock, date, expiry position, and option type;",
        f"7. the adjacent-strike interpolation bracket is at most `{MAX_LOG_MONEYNESS_BRACKET_WIDTH:.2f}` log-moneyness;",
        "8. the moneyness grid is symmetric and zero-centered;",
        "9. both option types remain unless a parity/carry contract is independently justified; and",
        "10. among passing candidates in the same representation family, select the unique maximal symmetric contiguous moneyness set under strict set inclusion.",
        "",
        f"The implemented maximal-set rule derives `{proposed['candidate_id']}` and is invariant to candidate ordering; it does not optimize a feature count or use absolute NSE traded-quantity magnitude. "
        "Passing this geometry rule is necessary but not sufficient for G2: a carry-conditioned inverse contract must also be frozen and validated.",
        "",
        "## Maturity support",
        "",
        "All four stocks shared the same direct listed DTE schedules on each date: `27|55|90`, `13|41|76`, and `6|34|69`. "
        "Relative expiry positions are therefore direct on all 12 surfaces and remain implementable on future dates by sorting revised/actual expiries, while carrying the actual `T` coordinates into the model.",
        "",
        "| Relative position | Direct support | Observed DTE values | DTE range |",
        "|---|---:|---|---:|",
    ]
    for slot in ("near", "mid", "far"):
        row = relative_summary.loc[slot]
        label = "middle" if slot == "mid" else slot
        lines.append(
            f"| {label} | 12/12 (100.0%) | `{row['values']}` | {int(row['min'])}–{int(row['max'])} |"
        )
    lines.extend(
        [
            "",
            "Fixed-DTE alternatives were classified independently on every surface:",
            "",
            "| Target DTE | Direct | Bounded interpolation | Unsupported/extrapolated | Worst nearest distance | Largest bracket span |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for target, direct, interpolated, unsupported, nearest, span in fixed_summary:
        lines.append(
            f"| {target} | {direct}/12 | {interpolated}/12 | {unsupported}/12 | {nearest} d | {span if span == '—' else str(span) + ' d'} |"
        )
    lines.extend(
        [
            "",
            "`180` DTE is unsupported on 12/12 surfaces. The old `7`, `14`, and `90` nodes require extrapolation on part of the panel. "
            "Although `30` and `60` DTE are bounded on all surfaces, they are direct on 0/12 and have worst nearest-expiry distances of 11 and 16 days; "
            "no maturity-price or total-variance interpolation contract (including rate/dividend inputs) is frozen. They are not promoted merely to keep a fixed grid.",
            "",
            f"The far listed expiry is direct but fails central activity support: across the five proposed nodes its expiry-position active support is only `{far_range[0]:.1f}%–{far_range[1]:.1f}%`. "
            "Near is 100% active at every proposed node; near+middle retains 87.5%–97.9% overall active support and a 75.0% worst balanced view. "
            "Therefore near+middle is the largest maturity geometry that passes the market-support rule.",
            "",
            "## Moneyness support",
            "",
            f"The proposed market price field is official NSE `ClsPric`: it is positive on `{close_positive_pct:.1f}%` of the 2,446 selected rows, "
            f"versus `{last_positive_pct:.1f}%` for `LastPric` and `{settlement_positive_pct:.1f}%` for `SttlmPric`. "
            "Last and settlement remain diagnostics, not silent substitutes; historical bid/ask is unavailable and is not inferred.",
            "",
            "A fixed target `k = log(K/S)` almost never equals a listed strike exactly: direct exact-node support is 0% at every candidate node. "
            "The relevant distinction is therefore bounded adjacent-strike interpolation versus extrapolation. The policy is linear interpolation of normalized price in `K/S` "
            "between the two adjacent listed strikes for the same security, valuation date, revised/actual expiry, and option type. No cross-security, cross-expiry, or strike extrapolation is allowed.",
            "",
            "The table below uses only the proposed near+middle positions (48 option-type slices per node):",
            "",
            "| log(K/S) | Observed bracket | Exact direct | Active bracket | Worst stock | Worst date | Worst position | Worst call/put | Max bracket width |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in moneyness_rows:
        lines.append(
            f"| {row[0]:+.2f} | {_pct(row[1])} | {_pct(row[2])} | {_pct(row[3])} | {_pct(row[4])} | {_pct(row[5])} | {_pct(row[6])} | {_pct(row[7])} | {row[8]:.6f} |"
        )
    lines.extend(
        [
            "",
            "Only `[-0.10, -0.05, 0.00, +0.05, +0.10]` form the largest symmetric contiguous set that has 100% observed bracketing, "
            "meets every 75% active-support view, and keeps the largest adjacent-strike bracket below 0.05. The `±0.15` extension fails 100% observed support; `±0.20` and `±0.30` are materially weaker. "
            "The extreme `-0.30` node has no observed bracket anywhere in the panel.",
            "",
            "## Call and put finding",
            "",
            "Both calls and puts are retained. Calls and puts exist in every selected near/middle slice and each option type meets the support rule. "
            "The Double Heston engine can derive puts by put-call parity, so synthetic prices are theoretically redundant conditional on risk-free rate and dividend yield. "
            "The market preprocessing contract does not yet freeze those carry inputs, and observed calls/puts contain distinct activity and microstructure information. "
            "Removing either type would therefore impose an unsupported carry/parity assumption rather than a scientifically demonstrated simplification.",
            "",
            "## Candidate comparison",
            "",
            "| Candidate | Maturity direct/interpolated/unsupported | Observed/active moneyness | Worst active view | Price + coordinate inputs | Decision |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in candidates.itertuples(index=False):
        worst = min(
            row.worst_stock_active_support_pct,
            row.worst_date_active_support_pct,
            row.worst_expiry_position_active_support_pct,
            row.worst_option_type_active_support_pct,
        )
        lines.append(
            f"| `{row.candidate_id}` | {row.maturity_direct_pct:.1f}% / {row.maturity_interpolated_pct:.1f}% / {row.maturity_unsupported_pct:.1f}% | "
            f"{row.moneyness_observed_support_pct:.1f}% / {row.moneyness_active_support_pct:.1f}% | {worst:.1f}% | "
            f"{row.normalized_price_feature_count} + {row.coordinate_feature_count} = {row.total_input_dimension} | {row.decision} |"
        )
    lines.extend(
        [
            "",
            "## Proposed surface geometry (not a frozen inverse contract)",
            "",
            "- **Representation type:** coordinate-aware relative listed-expiry surface.",
            "- **Maturity representation:** first and second revised/actual listed expiries (`near`, `middle`), sorted by actual expiry; include `T_near` and `T_middle` using calendar DTE/365.",
            "- **Moneyness nodes:** `[-0.10, -0.05, 0.00, +0.05, +0.10]`, where `k = log(K/S)` and `S` is the independent same-date CM close.",
            "- **Option types:** calls and puts.",
            "- **Price features:** 2 expiry positions × 5 nodes × 2 option types = **20**.",
            "- **Coordinate features:** 2 actual maturity fractions.",
            "- **Surface subtotal:** **22** features before any carry-conditioning coordinates or transformation.",
            "- **Final inverse input dimension:** **UNRESOLVED / NOT FROZEN**.",
            "- **Ordering:** `[T_near, T_middle]`, then option-major (`call`, `put`), expiry-major (`near`, `middle`), moneyness ascending.",
            "- **Market price and normalization:** official NSE `ClsPric / S` (`normalized_close`), with `S` equal to the independent same-date CM close; synthetic theoretical prices later use the same `/ S` scale. `T=DTE/365` is already dimensionless and receives no cross-stock normalization at this gate. Do not normalize or rank by absolute NSE traded quantity.",
            "- **Strike interpolation:** bounded adjacent listed strikes only, linear normalized price in `K/S`, separately within security/date/expiry/type; maximum accepted log-moneyness bracket width 0.05.",
            "- **Maturity interpolation:** none.",
            "- **Extrapolation:** prohibited for both maturity and strike.",
            "- **Missing support:** mark the surface unavailable; do not impute, borrow another security, or silently reduce the mask.",
            "- **Market-support applicability:** the geometry passes for all four primaries on all three dates. Future dates must rerun the same support/preprocessing checks; `near`/`middle` are reproducible positions, not assumed fixed DTEs.",
            "",
            "Compared with the rejected provisional 108-price representation, the proposed geometry removes the unsupported 180-DTE node, all fixed-DTE semantics, the far listed expiry, and the weak `±0.20/±0.30` wings. "
            "Its supported subtotal is 20 price inputs plus 2 maturity coordinates, while retaining both option types; this is not yet the final ANN dimension.",
            "",
            "## Blocking ambiguity and minimum additional evidence",
            "",
            "The Double Heston engine prices conditionally on risk-free rate `r` and dividend yield/carry `q`. The current synthetic contract fixes `r=0.02` and `q=0.01`, "
            "but Stage A intentionally did not freeze a real-market rate source, dividend/carry source, tenor mapping, or a validated forward/discount normalization. "
            "Allowing market `r/q` to vary without conditioning can confound carry with the ten Heston targets, so the 22-feature surface subtotal cannot by itself close G2.",
            "",
            "Minimum evidence to reopen and pass G2:",
            "",
            "1. predeclare and provenance-validate either (a) rate/dividend-carry coordinates aligned to near/middle maturities, (b) discount-factor and forward coordinates, or (c) a forward/discount normalization that demonstrably removes carry;",
            "2. show availability and deterministic preprocessing for all 12 stock/date surfaces without maturity extrapolation;",
            "3. update the declared ordering and exact total dimension for the chosen conditioning contract; and",
            "4. run a local Jacobian-rank/conditioning and noisy multi-start recovery check on the proposed two-maturity × five-strike geometry, treating carry according to that contract. If conditioning is inadequate, reopen the geometry rather than force G2.",
            "",
            "## Downstream compatibility (identified, not implemented)",
            "",
            "The later implementation must update `src/constants.py`, `src/surface_grid.py`, `src/dataset.py`, `src/synthetic_dataset.py`, `src/models.py`, "
            "`configs/ann_dataset_FIRST_RESEARCH.yaml`, `configs/ann_baseline.yaml`, dataset/shape/model tests, and the real-market preprocessing entry point. "
            "The Double Heston pricing interface already accepts quote-aligned maturity arrays, both option types, `r`, and `q`; its ten-parameter target order remains "
            "`kappa_slow, theta_slow, sigma_slow, rho_slow, v0_slow, kappa_fast, theta_fast, sigma_fast, rho_fast, v0_fast`. "
            "No downstream file was changed in this task.",
            "",
            "## Reproducibility and preservation",
            "",
            "Run `python scripts/run_g2_common_support_analysis.py`. The four CSVs and three figures are generated under the ignored Stage A derived tree. "
            "The script validates provenance before writing and verifies the eight canonical Stage A output hashes after writing.",
            "",
            "| Canonical Stage A output | SHA-256 preserved in this run |",
            "|---|---|",
        ]
    )
    for name in CANONICAL_STAGE_A_OUTPUTS:
        lines.append(f"| `{name}` | `{canonical_hashes[name].upper()}` |")
    lines.extend(
        [
            "",
            "```text",
            "G2 = NOT_PASSED",
            "PROPOSED_SURFACE_INPUT_SUBTOTAL = 22",
            "FINAL_TOTAL_INPUT_DIMENSION = UNRESOLVED",
            "CARRY_CONDITIONING = UNRESOLVED",
            "FINAL_SYNTHETIC_RESEARCH_DATA = NOT_GENERATED",
            "ANN_RESEARCH_TRAINING = NOT_STARTED",
            "PINN = NOT_DERIVED_OR_TRAINED",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def run_analysis(
    raw_root: Path = CANONICAL_RAW_ROOT,
    derived_root: Path = CANONICAL_DERIVED_ROOT,
    report_path: Path = DEFAULT_REPORT_PATH,
    figure_root: Path = DEFAULT_FIGURE_ROOT,
    *,
    write_outputs: bool = True,
) -> dict[str, object]:
    baseline = snapshot_canonical_outputs(derived_root)
    validate_canonical_stage_a_provenance(raw_root, derived_root)
    panel = load_balanced_panel(raw_root)
    surface = build_surface_support(panel)
    moneyness = build_moneyness_support(panel)
    maturity = build_maturity_support(panel)
    candidates = build_representation_candidates(maturity, moneyness)
    report = render_report(panel, maturity, moneyness, candidates, baseline)
    evidence_paths: dict[str, Path] = {}
    figure_paths: tuple[Path, ...] = ()
    if write_outputs:
        evidence_paths = publish_evidence_atomically(
            {
                "maturity": maturity,
                "moneyness": moneyness,
                "surface": surface,
                "candidates": candidates,
            },
            derived_root,
        )
        figure_paths = write_figures(maturity, moneyness, candidates, figure_root)
        _atomic_write_bytes(report_path, report.encode("utf-8"))
    assert_canonical_outputs_preserved(derived_root, baseline)
    return {
        "panel": panel,
        "surface": surface,
        "moneyness": moneyness,
        "maturity": maturity,
        "candidates": candidates,
        "report": report,
        "evidence_paths": evidence_paths,
        "figure_paths": figure_paths,
        "canonical_hashes": baseline,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=CANONICAL_RAW_ROOT)
    parser.add_argument("--derived-root", type=Path, default=CANONICAL_DERIVED_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--figure-root", type=Path, default=DEFAULT_FIGURE_ROOT)
    parser.add_argument("--no-write", action="store_true", help="Analyze and validate without publishing outputs.")
    args = parser.parse_args()
    result = run_analysis(
        args.raw_root,
        args.derived_root,
        args.report_path,
        args.figure_root,
        write_outputs=not args.no_write,
    )
    proposed = result["candidates"].loc[
        result["candidates"]["decision"] == "PROPOSED_GEOMETRY"
    ].iloc[0]
    print(
        f"{ANALYSIS_ID} surfaces=12 expiry_slices=36 option_rows={len(result['panel'])} "
        f"proposed_geometry={proposed['candidate_id']} "
        f"surface_input_subtotal={int(proposed['total_input_dimension'])}"
    )
    print("G2=NOT_PASSED carry_conditioning=UNRESOLVED final_input_dimension=UNRESOLVED")
    for path in result["evidence_paths"].values():
        print(f"evidence={path}")
    for path in result["figure_paths"]:
        print(f"figure={path}")
    if not args.no_write:
        print(f"report={args.report_path}")


if __name__ == "__main__":
    main()
