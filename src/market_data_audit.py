"""Stage A real-market availability loading and derived audit calculations.

This module is deliberately separate from the frozen pricing and model code. It
validates canonical export columns and derives observational diagnostics only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "market_data_audit_stage_a.yaml"
RAW_FILE_NAMES: Mapping[str, str] = {
    "options": "options_raw.xlsx",
    "futures": "futures_raw.xlsx",
    "spot": "spot_raw.xlsx",
}
EXPIRY_BUCKETS = ("near", "mid", "far")


@dataclass(frozen=True)
class StageASurfaceRaw:
    """Loaded raw tables and collection metadata for one surface directory."""

    options: pd.DataFrame
    futures: pd.DataFrame
    spot: pd.DataFrame
    manifest: Mapping[str, Any]


def load_audit_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the Stage A configuration without requiring Bloomberg."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Stage A audit configuration must be a YAML mapping")
    return config


def required_columns(
    table_kind: str, config: Mapping[str, Any] | None = None
) -> tuple[str, ...]:
    """Return canonical required columns for an options, futures, or spot table."""
    active_config = load_audit_config() if config is None else config
    try:
        columns = active_config["raw_semantic_fields"][table_kind]["required"]
    except KeyError as exc:
        raise ValueError(f"Unknown raw table kind: {table_kind}") from exc
    return tuple(str(column) for column in columns)


def validate_required_columns(
    frame: pd.DataFrame,
    table_kind: str,
    config: Mapping[str, Any] | None = None,
) -> None:
    """Raise a useful error when a raw table lacks canonical semantic columns."""
    required = required_columns(table_kind, config)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{table_kind} raw table is missing required columns: {missing}"
        )


def read_raw_table(
    path: str | Path,
    table_kind: str,
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Read one `.xlsx` export and validate its canonical columns."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Missing Stage A raw file: {source}")
    frame = pd.read_excel(source, engine="openpyxl")
    validate_required_columns(frame, table_kind, config)
    return frame


def load_stage_a_surface(
    directory: str | Path,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> StageASurfaceRaw:
    """Load the three raw workbooks and manifest for one underlying-date surface."""
    surface_directory = Path(directory)
    config = load_audit_config(config_path)
    manifest_path = surface_directory / "collection_manifest.yaml"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing Stage A collection manifest: {manifest_path}")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Stage A collection manifest must be a YAML mapping")
    _validate_manifest(manifest)

    tables = {
        kind: read_raw_table(surface_directory / filename, kind, config)
        for kind, filename in RAW_FILE_NAMES.items()
    }
    _validate_loaded_identity(tables, manifest)
    return StageASurfaceRaw(
        options=tables["options"],
        futures=tables["futures"],
        spot=tables["spot"],
        manifest=manifest,
    )


def derive_option_metrics(
    options: pd.DataFrame,
    spot: float | pd.DataFrame,
) -> pd.DataFrame:
    """Return a copy with Stage A option diagnostics and independent flags."""
    validate_required_columns(options, "options")
    result = options.copy(deep=True)
    result["valuation_date"] = _parse_dates(result["valuation_date"], "valuation_date")
    result["expiry_date"] = _parse_dates(result["expiry_date"], "expiry_date")
    result = _attach_spot(result, spot)

    for column in ("strike", "bid", "ask", "volume", "open_interest", "spot"):
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result["surface_id"] = (
        result["underlying"].astype(str)
        + "|"
        + result["valuation_date"].dt.strftime("%Y-%m-%d")
    )
    result["dte"] = (result["expiry_date"] - result["valuation_date"]).dt.days
    result["T"] = result["dte"] / 365.0

    valid_ratio = (result["strike"] > 0.0) & (result["spot"] > 0.0)
    result["k_over_s"] = np.nan
    result.loc[valid_ratio, "k_over_s"] = (
        result.loc[valid_ratio, "strike"] / result.loc[valid_ratio, "spot"]
    )
    result["log_k_over_s"] = np.nan
    result.loc[valid_ratio, "log_k_over_s"] = np.log(
        result.loc[valid_ratio, "k_over_s"]
    )
    result["mid"] = (result["bid"] + result["ask"]) / 2.0
    result["normalized_mid"] = np.nan
    positive_spot = result["spot"] > 0.0
    result.loc[positive_spot, "normalized_mid"] = (
        result.loc[positive_spot, "mid"] / result.loc[positive_spot, "spot"]
    )
    result["relative_bid_ask_spread"] = np.nan
    positive_mid = result["mid"] > 0.0
    result.loc[positive_mid, "relative_bid_ask_spread"] = (
        (result.loc[positive_mid, "ask"] - result.loc[positive_mid, "bid"])
        / result.loc[positive_mid, "mid"]
    )

    recognized_option_type = (
        result["option_type"].astype(str).str.strip().str.lower().isin({"call", "put", "c", "p"})
    )
    result["price_usable"] = (
        np.isfinite(result["spot"])
        & (result["spot"] > 0.0)
        & np.isfinite(result["strike"])
        & (result["strike"] > 0.0)
        & (result["dte"] > 0)
        & recognized_option_type
        & np.isfinite(result["bid"])
        & (result["bid"] >= 0.0)
        & np.isfinite(result["ask"])
        & (result["ask"] > 0.0)
        & (result["ask"] >= result["bid"])
        & np.isfinite(result["mid"])
        & (result["mid"] > 0.0)
    )

    result["volume_reported"] = result["volume"].notna()
    result["volume_positive"] = result["volume"] > 0.0
    result["open_interest_reported"] = result["open_interest"].notna()
    result["open_interest_positive"] = result["open_interest"] > 0.0
    result["expiry_bucket"] = _expiry_buckets(result)
    return result


def summarize_option_coverage(derived_options: pd.DataFrame) -> pd.DataFrame:
    """Summarize observed option coverage at the whole-surface level."""
    required = {
        "underlying",
        "valuation_date",
        "expiry_date",
        "expiry_bucket",
        "price_usable",
        "volume_positive",
        "open_interest_positive",
        "dte",
        "k_over_s",
    }
    missing = sorted(required.difference(derived_options.columns))
    if missing:
        raise ValueError(f"Derived option table is missing columns: {missing}")

    rows: list[dict[str, Any]] = []
    group_columns = ["underlying", "valuation_date"]
    for (underlying, valuation_date), group in derived_options.groupby(
        group_columns, sort=True, dropna=False
    ):
        buckets = set(group["expiry_bucket"].dropna())
        usable_count = int(group["price_usable"].sum())
        rows.append(
            {
                "underlying": underlying,
                "valuation_date": valuation_date,
                "surface_count": 1,
                "quote_row_count": int(len(group)),
                "expiry_count": int(group["expiry_date"].nunique()),
                "has_near": "near" in buckets,
                "has_mid": "mid" in buckets,
                "has_far": "far" in buckets,
                "complete_near_mid_far": set(EXPIRY_BUCKETS).issubset(buckets),
                "price_usable_count": usable_count,
                "price_usable_fraction": usable_count / len(group) if len(group) else np.nan,
                "volume_positive_count": int(group["volume_positive"].sum()),
                "open_interest_positive_count": int(group["open_interest_positive"].sum()),
                "observed_min_dte": group["dte"].min(),
                "observed_max_dte": group["dte"].max(),
                "observed_min_k_over_s": group["k_over_s"].min(),
                "observed_max_k_over_s": group["k_over_s"].max(),
            }
        )
    return pd.DataFrame(rows)


def summarize_futures_coverage(futures: pd.DataFrame) -> pd.DataFrame:
    """Summarize observed futures availability without selecting a carry convention."""
    validate_required_columns(futures, "futures")
    frame = futures.copy(deep=True)
    frame["valuation_date"] = _parse_dates(frame["valuation_date"], "valuation_date")
    frame["expiry_date"] = _parse_dates(frame["expiry_date"], "expiry_date")
    frame["futures_price"] = pd.to_numeric(frame["futures_price"], errors="coerce")
    frame["dte"] = (frame["expiry_date"] - frame["valuation_date"]).dt.days
    frame["T"] = frame["dte"] / 365.0
    frame["futures_price_usable"] = (
        np.isfinite(frame["futures_price"])
        & (frame["futures_price"] > 0.0)
        & (frame["dte"] > 0)
    )
    return (
        frame.groupby(["underlying", "valuation_date"], sort=True, dropna=False)
        .agg(
            futures_contract_count=("expiry_date", "size"),
            futures_expiry_count=("expiry_date", "nunique"),
            usable_futures_count=("futures_price_usable", "sum"),
            observed_min_futures_dte=("dte", "min"),
            observed_max_futures_dte=("dte", "max"),
        )
        .reset_index()
    )


def futures_implied_carry(
    risk_free_rate: float | np.ndarray,
    futures_price: float | np.ndarray,
    spot: float | np.ndarray,
    maturity_years: float | np.ndarray,
) -> float | np.ndarray:
    """Calculate `q_impl = r - ln(F/S)/T` as an unselected audit diagnostic."""
    r = np.asarray(risk_free_rate, dtype=np.float64)
    futures = np.asarray(futures_price, dtype=np.float64)
    spot_value = np.asarray(spot, dtype=np.float64)
    maturity = np.asarray(maturity_years, dtype=np.float64)
    if not (
        np.isfinite(r).all()
        and np.isfinite(futures).all()
        and np.isfinite(spot_value).all()
        and np.isfinite(maturity).all()
    ):
        raise ValueError("Carry inputs must be finite")
    if np.any(futures <= 0.0) or np.any(spot_value <= 0.0) or np.any(maturity <= 0.0):
        raise ValueError("Futures price, spot, and maturity must be strictly positive")
    result = r - np.log(futures / spot_value) / maturity
    return float(result) if result.ndim == 0 else result


def _attach_spot(options: pd.DataFrame, spot: float | pd.DataFrame) -> pd.DataFrame:
    if np.isscalar(spot):
        result = options.copy(deep=True)
        result["spot"] = float(spot)
        return result

    validate_required_columns(spot, "spot")
    spot_table = spot.copy(deep=True)
    spot_table["valuation_date"] = _parse_dates(
        spot_table["valuation_date"], "spot.valuation_date"
    )
    if spot_table.duplicated(["underlying", "valuation_date"]).any():
        raise ValueError("Spot table contains duplicate underlying-date rows")
    return options.merge(
        spot_table[["underlying", "valuation_date", "spot"]],
        on=["underlying", "valuation_date"],
        how="left",
        validate="many_to_one",
    )


def _expiry_buckets(frame: pd.DataFrame) -> pd.Series:
    buckets = pd.Series(index=frame.index, dtype="object")
    for _, group in frame.groupby(["underlying", "valuation_date"], sort=False, dropna=False):
        expiries = sorted(group["expiry_date"].dropna().unique())
        if len(expiries) > len(EXPIRY_BUCKETS):
            raise ValueError(
                "A Stage A surface may contain at most three selected expiry slices "
                "(near, mid, far)"
            )
        mapping = dict(zip(expiries, EXPIRY_BUCKETS, strict=False))
        buckets.loc[group.index] = group["expiry_date"].map(mapping)
    return buckets


def _parse_dates(values: pd.Series, name: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce").dt.normalize()
    if parsed.isna().any():
        raise ValueError(f"{name} contains missing or invalid dates")
    return parsed


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    required = {
        "stage",
        "collection_status",
        "underlying",
        "valuation_date",
        "reference_only",
        "surface_definition",
        "expected_files",
    }
    missing = sorted(required.difference(manifest))
    if missing:
        raise ValueError(f"Stage A collection manifest is missing keys: {missing}")
    if str(manifest["stage"]) != "A":
        raise ValueError("Collection manifest stage must be A")
    manifest_files = set(manifest["expected_files"])
    missing_files = sorted(set(RAW_FILE_NAMES.values()).difference(manifest_files))
    if missing_files:
        raise ValueError(
            f"Collection manifest expected_files is missing raw files: {missing_files}"
        )


def _validate_loaded_identity(
    tables: Mapping[str, pd.DataFrame], manifest: Mapping[str, Any]
) -> None:
    identities: set[tuple[str, pd.Timestamp]] = set()
    for kind, frame in tables.items():
        if frame.empty:
            continue
        dates = _parse_dates(frame["valuation_date"], f"{kind}.valuation_date")
        table_identities = set(zip(frame["underlying"].astype(str), dates, strict=True))
        if len(table_identities) != 1:
            raise ValueError(f"{kind} raw table must contain one underlying-date identity")
        identities.update(table_identities)
    if len(identities) > 1:
        raise ValueError("Stage A raw tables do not share one underlying-date identity")
    manifest_date = pd.to_datetime(manifest["valuation_date"], errors="coerce")
    if pd.isna(manifest_date):
        raise ValueError("Collection manifest valuation_date is invalid")
    manifest_identity = (
        str(manifest["underlying"]),
        pd.Timestamp(manifest_date).normalize(),
    )
    if identities and identities != {manifest_identity}:
        raise ValueError("Stage A raw-table identity does not match collection manifest")


__all__ = [
    "StageASurfaceRaw",
    "derive_option_metrics",
    "futures_implied_carry",
    "load_audit_config",
    "load_stage_a_surface",
    "read_raw_table",
    "required_columns",
    "summarize_futures_coverage",
    "summarize_option_coverage",
    "validate_required_columns",
]
