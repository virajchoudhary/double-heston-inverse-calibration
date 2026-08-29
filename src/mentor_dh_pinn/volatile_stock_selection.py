"""Official-NSE-only acquisition and mentor-aligned volatile-stock audit."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import urllib.error
import urllib.request

import yaml

from src.nse_stage_a import read_udiff_csv


REPORT_COLUMNS = (
    "SYMBOL", "SERIES", "DATE1", "PREV_CLOSE", "OPEN_PRICE", "HIGH_PRICE",
    "LOW_PRICE", "LAST_PRICE", "CLOSE_PRICE", "AVG_PRICE", "TTL_TRD_QNTY",
    "TURNOVER_LACS", "NO_OF_TRADES", "DELIV_QTY", "DELIV_PER",
)
SPEC_KEYS = {
    "schema_version", "history_start", "history_end", "option_dates", "symbols", "series",
    "calendar_months", "minimum_returns", "annualization_days", "official_report_url_template",
    "raw_root", "derived_root", "stage_a_raw_root", "rate_observations", "domain",
    "surface_gate", "parameter_source",
}
DOMAIN_KEYS = {
    "spot_min", "spot_max", "moneyness_min", "moneyness_max", "tau_min", "tau_max",
    "rate_min", "rate_max", "carry_min", "carry_max", "variance_floor", "variance_ceiling",
    "variance_theta_min_multiplier", "variance_theta_max_multiplier",
}
SURFACE_KEYS = {
    "minimum_active_calls", "minimum_distinct_strikes", "minimum_eligible_expiries",
    "split_quantiles",
}
PARAMETER_KEYS = {"theta_slow", "theta_fast"}
RATE_KEYS = {
    "source_authority", "source_url", "source_identifier", "page_title", "release_date",
    "instrument", "measure", "cutoff_price", "simple_annual_yield_percent",
    "simple_annual_yield_decimal", "retrieved_on", "observation_method",
    "verbatim_field_extract", "preserved_html",
}
PARSER_ID = "mentor_dh_pinn.volatile_stock_selection/security_bhavdata_v1"


class VolatileStockAuditError(ValueError):
    """Fail-closed audit error."""


@dataclass(frozen=True)
class AuditSpec:
    history_start: date
    history_end: date
    option_dates: tuple[date, ...]
    symbols: tuple[str, ...]
    series: str
    calendar_months: int
    minimum_returns: int
    annualization_days: int
    official_report_url_template: str
    raw_root: Path
    derived_root: Path
    stage_a_raw_root: Path
    rate_observations: Mapping[date, Path]
    domain: Mapping[str, float]
    surface_gate: Mapping[str, object]
    parameter_source: Mapping[str, float]


def load_spec(path: str | Path) -> AuditSpec:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != SPEC_KEYS:
        raise VolatileStockAuditError(f"Config fields drifted: {sorted(set(raw or {}) ^ SPEC_KEYS)}")
    _exact_keys(raw["domain"], DOMAIN_KEYS, "domain")
    _exact_keys(raw["surface_gate"], SURFACE_KEYS, "surface_gate")
    _exact_keys(raw["parameter_source"], PARAMETER_KEYS, "parameter_source")
    if raw["schema_version"] != "mentor_dh_pinn_volatile_stock_selection_v1":
        raise VolatileStockAuditError("Unknown schema_version")
    symbols = tuple(raw["symbols"])
    if symbols != ("ADANIENT", "ADANIPOWER", "NTPC", "POWERGRID", "SUNPHARMA", "CIPLA", "INFY", "TCS", "ICICIBANK", "HDFCBANK"):
        raise VolatileStockAuditError("Candidate universe drifted")
    if int(raw["calendar_months"]) != 3 or int(raw["minimum_returns"]) != 50 or int(raw["annualization_days"]) != 252:
        raise VolatileStockAuditError("Volatility methodology drifted")
    rates: dict[date, Path] = {}
    for key, value in raw["rate_observations"].items():
        rates[_as_date(key)] = Path(value)
    return AuditSpec(
        history_start=_as_date(raw["history_start"]), history_end=_as_date(raw["history_end"]),
        option_dates=tuple(_as_date(value) for value in raw["option_dates"]), symbols=symbols,
        series=str(raw["series"]), calendar_months=3, minimum_returns=50, annualization_days=252,
        official_report_url_template=str(raw["official_report_url_template"]),
        raw_root=Path(raw["raw_root"]), derived_root=Path(raw["derived_root"]),
        stage_a_raw_root=Path(raw["stage_a_raw_root"]), rate_observations=rates,
        domain={key: float(value) for key, value in raw["domain"].items()},
        surface_gate=dict(raw["surface_gate"]),
        parameter_source={key: float(value) for key, value in raw["parameter_source"].items()},
    )


def parse_security_report(content: bytes, expected_date: date) -> pd.DataFrame:
    if content.startswith(bytes.fromhex("504b0304")):
        try:
            frame = pd.read_excel(io.BytesIO(content), engine="openpyxl")
        except Exception as exc:
            raise VolatileStockAuditError("Official OOXML report is unreadable") from exc
    else:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise VolatileStockAuditError("Official report is neither UTF-8 nor OOXML") from exc
        frame = pd.read_csv(io.StringIO(text), skipinitialspace=True)
    frame.columns = [str(value).strip() for value in frame.columns]
    if tuple(frame.columns) != REPORT_COLUMNS:
        raise VolatileStockAuditError(f"Unexpected official security-report schema: {tuple(frame.columns)}")
    parsed_dates = pd.to_datetime(frame["DATE1"].astype(str).str.strip(), format="mixed", dayfirst=True, errors="raise").dt.date
    if frame.empty or set(parsed_dates) != {expected_date}:
        raise VolatileStockAuditError("Official report date identity mismatch")
    frame["DATE1"] = parsed_dates
    for column in REPORT_COLUMNS:
        if column not in {"SYMBOL", "SERIES", "DATE1"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["SYMBOL"] = frame["SYMBOL"].astype(str).str.strip()
    frame["SERIES"] = frame["SERIES"].astype(str).str.strip()
    return frame


def acquire_official_histories(spec: AuditSpec, *, offline: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Acquire immutable daily official reports and return history plus raw manifest."""
    spec.raw_root.mkdir(parents=True, exist_ok=True)
    spec.derived_root.mkdir(parents=True, exist_ok=True)
    prior = _read_manifest(spec.derived_root / "underlying_raw_report_manifest.csv")
    records: list[dict[str, object]] = []
    selected: list[pd.DataFrame] = []
    missing_weekdays: list[str] = []
    for index, value in enumerate(_weekdays(spec.history_start, spec.history_end), start=1):
        filename = f"sec_bhavdata_full_{value:%d%m%Y}.csv"
        raw_path = spec.raw_root / f"{value:%Y-%m-%d}" / filename
        url = spec.official_report_url_template.format(date_ddmmyyyy=f"{value:%d%m%Y}")
        action = "REUSED"
        retrieved = ""
        if raw_path.is_file():
            content = raw_path.read_bytes()
        else:
            if offline:
                if value.isoformat() in prior:
                    missing_weekdays.append(value.isoformat())
                continue
            status, content = _get_with_retry(url)
            if status == 404:
                continue
            if status != 200:
                raise VolatileStockAuditError(f"Official NSE acquisition failed for {value}: HTTP {status}")
            try:
                parse_security_report(content, value)
            except VolatileStockAuditError as exc:
                if str(exc) == "Official report date identity mismatch":
                    continue
                raise VolatileStockAuditError(f"{value}: {exc}") from exc
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            _write_new(raw_path, content)
            action = "DOWNLOADED"
            retrieved = datetime.now(timezone.utc).isoformat()
        frame = parse_security_report(content, value)
        digest = _sha256(content)
        identity = value.isoformat()
        if identity in prior:
            if prior[identity]["raw_sha256"] != digest:
                raise VolatileStockAuditError(f"Raw hash conflict for {identity}; existing evidence was not overwritten")
            first_timestamp = prior[identity]["acquisition_timestamp_utc"]
        else:
            first_timestamp = retrieved or datetime.fromtimestamp(raw_path.stat().st_mtime, timezone.utc).isoformat()
        records.append({
            "source": "National Stock Exchange of India", "report_identity": filename,
            "trade_date": identity, "official_url": url, "raw_path": str(raw_path),
            "raw_sha256": digest, "raw_size_bytes": len(content), "acquisition_timestamp_utc": first_timestamp,
            "current_run_action": action, "parser_identity": PARSER_ID, "tool_git_sha": _git_sha(),
        })
        subset = frame.loc[(frame["SYMBOL"].isin(spec.symbols)) & (frame["SERIES"] == spec.series)].copy()
        selected.append(subset)
        if index % 100 == 0:
            print(f"official-history progress: {value} ({index} weekdays checked)", flush=True)
    if offline and missing_weekdays:
        raise VolatileStockAuditError(f"Offline raw corpus incomplete; first missing weekday: {missing_weekdays[0]}")
    if not selected:
        raise VolatileStockAuditError("No official NSE history reports were available")
    history = validate_history(pd.concat(selected, ignore_index=True), spec)
    manifest = pd.DataFrame(records).sort_values("trade_date", kind="stable").reset_index(drop=True)
    _atomic_csv(manifest, spec.derived_root / "underlying_raw_report_manifest.csv")
    _write_symbol_evidence(history, manifest, spec)
    return history, manifest


def validate_history(history: pd.DataFrame, spec: AuditSpec) -> pd.DataFrame:
    frame = history.copy()
    keys = ["SYMBOL", "SERIES", "DATE1"]
    duplicates = frame.loc[frame.duplicated(keys, keep=False)]
    for _, group in duplicates.groupby(keys, sort=False):
        if len(group.drop_duplicates()) != 1:
            raise VolatileStockAuditError(f"Conflicting duplicate symbol/date observation: {tuple(group.iloc[0][keys])}")
    frame = frame.drop_duplicates(keys, keep="first")
    for column in ("PREV_CLOSE", "OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE", "CLOSE_PRICE", "AVG_PRICE", "TTL_TRD_QNTY"):
        if frame[column].isna().any() or (frame[column] <= 0).any():
            raise VolatileStockAuditError(f"Invalid required underlying field: {column}")
    return frame.sort_values(["SYMBOL", "DATE1"], kind="stable").reset_index(drop=True)


def volatility_windows(history: pd.DataFrame, symbol: str, spec: AuditSpec) -> pd.DataFrame:
    frame = history.loc[history["SYMBOL"] == symbol].sort_values("DATE1").reset_index(drop=True)
    dates = list(frame["DATE1"])
    closes = frame["CLOSE_PRICE"].to_numpy(float)
    previous = frame["PREV_CLOSE"].to_numpy(float)
    linked = np.ones(len(frame), dtype=bool)
    linked[1:] = np.isclose(previous[1:], closes[:-1], rtol=0.0, atol=0.011)
    returns = np.full(len(frame), np.nan)
    returns[1:][linked[1:]] = np.log(closes[1:][linked[1:]] / closes[:-1][linked[1:]])
    rows: list[dict[str, object]] = []
    for endpoint in dates:
        start = (pd.Timestamp(endpoint) - pd.DateOffset(months=spec.calendar_months)).date()
        indices = [i for i, day in enumerate(dates) if start <= day < endpoint]
        if len(indices) < spec.minimum_returns + 1:
            continue
        window_returns = returns[indices[1:]]
        if len(window_returns) < spec.minimum_returns or not np.isfinite(window_returns).all():
            continue
        rows.append({
            "symbol": symbol, "endpoint": endpoint, "window_start": start,
            "window_end": dates[indices[-1]], "closes": len(indices), "returns": len(window_returns),
            "rv_3m": float(np.std(window_returns, ddof=1) * math.sqrt(spec.annualization_days)),
        })
    return pd.DataFrame(rows)


def candidate_volatility(history: pd.DataFrame, symbol: str, valuation_date: date, spec: AuditSpec) -> dict[str, object]:
    frame = history.loc[history["SYMBOL"] == symbol].sort_values("DATE1").reset_index(drop=True)
    if (spec.history_end - spec.history_start).days >= 5 * 365:
        if frame.empty or frame["DATE1"].min() > spec.history_start + timedelta(days=14):
            raise VolatileStockAuditError(f"{symbol}: official EQ history does not begin near the frozen start")
        if frame["DATE1"].max() < spec.history_end or len(frame) < 1000:
            raise VolatileStockAuditError(f"{symbol}: fewer than approximately five years of usable official EQ history")
    start = (pd.Timestamp(valuation_date) - pd.DateOffset(months=spec.calendar_months)).date()
    window = frame.loc[(frame["DATE1"] >= start) & (frame["DATE1"] < valuation_date)].copy()
    if len(window) < spec.minimum_returns + 1:
        raise VolatileStockAuditError(f"{symbol} {valuation_date}: insufficient 3M closes")
    prior_close = window["CLOSE_PRICE"].shift(1)
    links = np.isclose(window["PREV_CLOSE"].iloc[1:].to_numpy(float), prior_close.iloc[1:].to_numpy(float), rtol=0.0, atol=0.011)
    if not links.all():
        raise VolatileStockAuditError(f"{symbol} {valuation_date}: discontinuous previous-close linkage")
    returns = np.log(window["CLOSE_PRICE"].to_numpy(float)[1:] / window["CLOSE_PRICE"].to_numpy(float)[:-1])
    rolling = volatility_windows(history, symbol, spec)
    rv = float(np.std(returns, ddof=1) * math.sqrt(spec.annualization_days))
    if rolling.empty:
        raise VolatileStockAuditError(f"{symbol}: no valid rolling 3M distribution")
    values = rolling["rv_3m"].to_numpy(float)
    return {
        "history_start": frame["DATE1"].min(), "history_end": frame["DATE1"].max(),
        "valid_history_observations": len(frame), "window_start": start,
        "window_end": window["DATE1"].iloc[-1], "closes": len(window), "returns": len(returns),
        "rv_3m": rv, "rolling_median": float(np.median(values)),
        "rolling_p75": float(np.percentile(values, 75)), "rolling_p90": float(np.percentile(values, 90)),
        "rolling_max": float(np.max(values)), "rolling_count": len(values),
        "percentile_rank": float(100.0 * np.count_nonzero(values <= rv) / len(values)),
    }


def option_surface(spec: AuditSpec, symbol: str, valuation_date: date) -> tuple[dict[str, object], pd.DataFrame]:
    root = spec.stage_a_raw_root / valuation_date.isoformat()
    cm_name = f"BhavCopy_NSE_CM_0_0_0_{valuation_date:%Y%m%d}_F_0000.csv"
    fo_name = f"BhavCopy_NSE_FO_0_0_0_{valuation_date:%Y%m%d}_F_0000.csv"
    cm = read_udiff_csv(root / cm_name, valuation_date, "CM")
    fo = read_udiff_csv(root / fo_name, valuation_date, "FO")
    spot_rows = cm.loc[(cm["TckrSymb"].astype(str) == symbol) & (cm["SctySrs"].astype(str) == spec.series)]
    if len(spot_rows) != 1:
        return {"status": "FAIL", "reason": "missing or duplicate official EQ spot", "call_count": 0}, pd.DataFrame()
    spot = float(pd.to_numeric(spot_rows["ClsPric"], errors="raise").iloc[0])
    instrument = fo["FinInstrmTp"].astype(str).str.upper()
    calls = fo.loc[(fo["TckrSymb"].astype(str) == symbol) & (instrument == "STO") & (fo["OptnTp"].astype(str) == "CE")].copy()
    if calls.empty:
        return {"status": "FAIL", "reason": "no trustworthy genuine CALL observations", "call_count": 0, "spot": spot}, calls
    for column in ("ClsPric", "TtlTradgVol", "OpnIntrst", "TtlNbOfTxsExctd", "StrkPric"):
        calls[column] = pd.to_numeric(calls[column], errors="coerce")
    calls["expiry"] = pd.to_datetime(calls["FininstrmActlXpryDt"].replace("", np.nan).fillna(calls["XpryDt"]), format="mixed", dayfirst=True, errors="raise").dt.date
    calls["tau"] = calls["expiry"].map(lambda value: (value - valuation_date).days / 365.0)
    calls["K_over_S"] = calls["StrkPric"] / spot
    calls["active"] = (calls[["ClsPric", "TtlTradgVol", "OpnIntrst", "TtlNbOfTxsExctd"]] > 0).all(axis=1)
    active = calls.loc[calls["active"]].copy()
    result: dict[str, object] = {
        "spot": spot, "call_count": len(active), "put_count": int(((fo["TckrSymb"].astype(str) == symbol) & (instrument == "STO") & (fo["OptnTp"].astype(str) == "PE")).sum()),
        "call_strikes": int(active["StrkPric"].nunique()), "call_expiries": int(active["expiry"].nunique()),
        "k_s_min": float(active["K_over_S"].min()) if not active.empty else math.nan,
        "k_s_max": float(active["K_over_S"].max()) if not active.empty else math.nan,
        "tau_min": float(active["tau"].min()) if not active.empty else math.nan,
        "tau_max": float(active["tau"].max()) if not active.empty else math.nan,
    }
    if valuation_date not in spec.rate_observations:
        result.update(status="FAIL", reason="no preserved official risk-free-rate observation; r/q domain unverified", r_min=math.nan, r_max=math.nan, q_min=math.nan, q_max=math.nan)
        return result, active
    rate_raw = json.loads(spec.rate_observations[valuation_date].read_text(encoding="utf-8"))
    if set(rate_raw) != RATE_KEYS or _as_date(rate_raw["release_date"]) > valuation_date or rate_raw["source_authority"] != "Reserve Bank of India":
        raise VolatileStockAuditError("Official rate evidence contract drifted")
    simple_rate = float(rate_raw["simple_annual_yield_decimal"])
    futures = fo.loc[(fo["TckrSymb"].astype(str) == symbol) & (instrument == "STF")].copy()
    futures["expiry"] = pd.to_datetime(futures["FininstrmActlXpryDt"].replace("", np.nan).fillna(futures["XpryDt"]), format="mixed", dayfirst=True, errors="raise").dt.date
    futures["future"] = pd.to_numeric(futures["ClsPric"], errors="coerce")
    future_map = futures.groupby("expiry")["future"].apply(lambda values: sorted(set(values.dropna()))).to_dict()
    qs: list[float] = []
    rates: list[float] = []
    for expiry, tau in active[["expiry", "tau"]].drop_duplicates().itertuples(index=False):
        matches = future_map.get(expiry, [])
        if len(matches) != 1 or matches[0] <= 0:
            result.update(status="FAIL", reason=f"missing or ambiguous matched futures for {expiry}", r_min=math.nan, r_max=math.nan, q_min=math.nan, q_max=math.nan)
            return result, active
        rate = -math.log(1.0 / (1.0 + simple_rate * tau)) / tau
        q = rate - math.log(matches[0] / spot) / tau
        rates.append(rate); qs.append(q)
    result.update(r_min=min(rates), r_max=max(rates), q_min=min(qs), q_max=max(qs))
    d = spec.domain
    failures: list[str] = []
    if active.empty or len(active) < int(spec.surface_gate["minimum_active_calls"]): failures.append("insufficient active CALL count")
    if active["StrkPric"].nunique() < int(spec.surface_gate["minimum_distinct_strikes"]): failures.append("insufficient distinct CALL strikes")
    if active["expiry"].nunique() < int(spec.surface_gate["minimum_eligible_expiries"]): failures.append("insufficient CALL expiry support")
    if not ((active["K_over_S"] >= d["moneyness_min"]) & (active["K_over_S"] <= d["moneyness_max"])).all(): failures.append("active CALL K/S outside frozen domain")
    if not ((active["tau"] >= d["tau_min"]) & (active["tau"] <= d["tau_max"])).all(): failures.append("active CALL tau outside frozen domain")
    if not (d["rate_min"] <= min(rates) <= max(rates) <= d["rate_max"]): failures.append("r outside frozen domain")
    if not (d["carry_min"] <= min(qs) <= max(qs) <= d["carry_max"]): failures.append("q outside frozen domain")
    if not (d["spot_min"] <= 1.0 <= d["spot_max"]): failures.append("normalized spot outside frozen domain")
    for name, theta in spec.parameter_source.items():
        low = max(d["variance_floor"], d["variance_theta_min_multiplier"] * theta)
        high = min(d["variance_ceiling"], d["variance_theta_max_multiplier"] * theta)
        if not low < high: failures.append(f"invalid {name} variance-state calibration bounds")
    result.update(status="PASS" if not failures else "FAIL", reason="eligible" if not failures else "; ".join(failures))
    return result, active


def propose_split(active: pd.DataFrame, spec: AuditSpec) -> pd.DataFrame:
    quantiles = [float(value) for value in spec.surface_gate["split_quantiles"]]
    if quantiles != [0.0, 0.25, 0.5, 0.75, 1.0]:
        raise VolatileStockAuditError("Split quantiles drifted")
    rows: list[pd.Series] = []
    expiry_order = active.groupby("expiry").size().sort_values(ascending=False, kind="stable").index[:2]
    for expiry in sorted(expiry_order):
        group = (
            active.loc[active["expiry"] == expiry]
            .sort_values("StrkPric", kind="stable")
            .drop_duplicates("StrkPric")
            .reset_index(drop=True)
        )
        if len(group) < len(quantiles):
            raise VolatileStockAuditError(f"Fewer than five distinct strikes for expiry {expiry}")
        indices = [round(value * (len(group) - 1)) for value in quantiles]
        if len(set(indices)) != len(indices):
            raise VolatileStockAuditError(f"Split quantiles duplicate a contract for expiry {expiry}")
        for quantile, index in zip(quantiles, indices, strict=True):
            chosen = group.iloc[index].copy()
            chosen["split_quantile"] = quantile
            chosen["role"] = "holdout" if quantile in {0.0, 1.0} else "calibration"
            rows.append(chosen)
    result = pd.DataFrame(rows).sort_values(["expiry", "split_quantile"], kind="stable")
    if result.duplicated(["expiry", "StrkPric"]).any():
        raise VolatileStockAuditError("Deterministic split produced duplicate contracts")
    return result.reset_index(drop=True)
def audit_candidates(history: pd.DataFrame, spec: AuditSpec) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows: list[dict[str, object]] = []
    active_by_key: dict[str, pd.DataFrame] = {}
    for symbol in spec.symbols:
        for valuation_date in spec.option_dates:
            try:
                vol = candidate_volatility(history, symbol, valuation_date, spec)
                vol_reason = "valid"
            except VolatileStockAuditError as exc:
                vol = {}; vol_reason = str(exc)
            option, active = option_surface(spec, symbol, valuation_date)
            if option.get("status") == "PASS":
                try:
                    propose_split(active, spec)
                except VolatileStockAuditError as exc:
                    option["status"] = "FAIL"
                    option["reason"] = f"deterministic split infeasible: {exc}"
            key = f"{symbol}|{valuation_date.isoformat()}"; active_by_key[key] = active
            status = "ELIGIBLE" if vol and option.get("status") == "PASS" else "REJECTED"
            reason = "eligible" if status == "ELIGIBLE" else "; ".join(value for value in (None if vol else vol_reason, option.get("reason")) if value)
            rows.append({"symbol": symbol, "option_valuation_date": valuation_date, **vol, **option, "classification": status, "rejection_reason": reason})
    return pd.DataFrame(rows).sort_values(["symbol", "option_valuation_date"], kind="stable").reset_index(drop=True), active_by_key


def _write_symbol_evidence(history: pd.DataFrame, manifest: pd.DataFrame, spec: AuditSpec) -> None:
    rows: list[dict[str, object]] = []
    for symbol in spec.symbols:
        subset = history.loc[history["SYMBOL"] == symbol].copy()
        path = spec.derived_root / "underlying" / f"{symbol}_EQ_20210101_20260721.csv"
        _atomic_csv(subset, path)
        rows.append({
            "source": "National Stock Exchange of India", "symbol": symbol, "series": spec.series,
            "requested_start": spec.history_start, "requested_end": spec.history_end,
            "received_start": subset["DATE1"].min(), "received_end": subset["DATE1"].max(),
            "row_count": len(subset), "raw_sha256": _sha256(path.read_bytes()),
            "acquisition_timestamp_utc": manifest["acquisition_timestamp_utc"].max(),
            "source_report_manifest_sha256": _sha256((spec.derived_root / "underlying_raw_report_manifest.csv").read_bytes()),
            "parser_identity": PARSER_ID, "tool_git_sha": _git_sha(),
        })
    _atomic_csv(pd.DataFrame(rows), spec.derived_root / "acquisition_manifest.csv")


def _get_with_retry(url: str) -> tuple[int, bytes]:
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.nseindia.com/all-reports"}
    last_status = 0
    for attempt in range(5):
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return int(response.status), response.read()
        except urllib.error.HTTPError as exc:
            last_status = int(exc.code)
            if last_status == 404:
                return last_status, b""
            if last_status not in {429, 500, 502, 503, 504}:
                return last_status, exc.read()
        except urllib.error.URLError as exc:
            if attempt == 4:
                raise VolatileStockAuditError(f"Official NSE request failed: {url}: {exc}") from exc
        time.sleep(1.0 + attempt)
    return last_status, b""
def _read_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file(): return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["trade_date"]: row for row in csv.DictReader(handle)}


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle: handle.write(content); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def _write_new(path: Path, content: bytes) -> None:
    try:
        with path.open("xb") as handle: handle.write(content); handle.flush(); os.fsync(handle.fileno())
    except FileExistsError:
        if path.read_bytes() != content: raise VolatileStockAuditError(f"Concurrent raw conflict: {path}")


def _weekdays(start: date, end: date) -> Iterable[date]:
    value = start
    while value <= end:
        if value.weekday() < 5: yield value
        value += timedelta(days=1)


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def _sha256(content: bytes) -> str: return hashlib.sha256(content).hexdigest()


def _as_date(value: object) -> date:
    if isinstance(value, datetime): return value.date()
    if isinstance(value, date): return value
    return date.fromisoformat(str(value))


def _exact_keys(value: object, expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise VolatileStockAuditError(f"{label} fields drifted: {sorted(set(value or {}) ^ expected)}")


__all__ = [
    "AuditSpec", "VolatileStockAuditError", "acquire_official_histories", "audit_candidates",
    "candidate_volatility", "load_spec", "option_surface", "parse_security_report", "propose_split",
    "validate_history", "volatility_windows",
]
