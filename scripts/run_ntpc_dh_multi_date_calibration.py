"""Calibrate canonical Double Heston jointly to three real NTPC dates.

Eight structural parameters are shared. Each valuation date has its own
``v0_slow`` and ``v0_fast`` state. This is not a new 14-parameter model; each
date is priced by the canonical ten-parameter Double Heston vector.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import platform
import re
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.special import expit
from scipy.spatial.distance import pdist

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import run_ntpc_dh_stability_reparameterization as geometry
from scripts import run_ntpc_single_stock_pilot as pilot
from src.calibrate_double_heston import boundary_diagnostics, load_hard_safety_bounds, unconstrained_to_parameters
from src.constants import PARAMETER_NAMES
from src.nse_stage_a import read_udiff_csv


VALUATION_DATES = ("2026-07-01", "2026-07-15", "2026-07-22")
TARGETS = (-0.10, -0.05, 0.0, 0.05, 0.10)
CALIBRATION_TARGETS = (-0.05, 0.0, 0.05)
HOLDOUT_TARGETS = (-0.10, 0.10)
START_COUNT = 12
MAX_NFEV = 320
NODE_COUNT = 64
MATERIAL_DISTANCE = 0.05
CLUSTER_DISTANCE = 0.05
BOUNDS_PATH = pilot.BOUNDS_PATH
RAW_ROOT = pilot.RAW_ROOT
OUTPUT_ROOT = pilot.OUTPUT_ROOT.parent / "ntpc_dh_multi_date_calibration"
RATE_PROVENANCE_ROOT = OUTPUT_ROOT / "rate_provenance"
FIGURE_ROOT = OUTPUT_ROOT / "figures"
REPORT_PATH = REPOSITORY_ROOT / "docs" / "NTPC_DH_MULTI_DATE_CALIBRATION.md"
MANIFEST_PATH = REPOSITORY_ROOT / "docs" / "evidence" / "NTPC_DH_MULTI_DATE_CALIBRATION_MANIFEST.json"
SHARED_NAMES = ("kappa_slow", "theta_slow", "sigma_slow", "rho_slow",
                "kappa_fast", "theta_fast", "sigma_fast", "rho_fast")
SHARED_CANONICAL_INDEX = (0, 1, 2, 3, 5, 6, 7, 8)
SINGLE_DATE_HOLDOUT = 0.9268247197137796
HESTON_HOLDOUT = 0.910569
RATE_SOURCES = {
    "2026-07-01": {"yield": 0.052521, "cutoff_price": 98.7075, "release_date": "2026-07-01",
        "source_identifier": "RBI Press Release 2026-2027/584",
        "url": "https://www.rbi.org.in/scripts/BS_PressReleaseDisplay.aspx?prid=63062",
        "html": "rbi_tbill_full_auction_result_20260701.html",
        "sha256": "E9CCF5D501E4488F18F185490A119C3B184F341C6C42F45861B898DCB6285B00"},
    "2026-07-15": {"yield": 0.053324, "cutoff_price": 98.6880, "release_date": "2026-07-15",
        "source_identifier": "RBI Press Release 2026-2027/672",
        "url": pilot.RISK_FREE_SOURCE_URL,
        "html": "rbi_tbill_full_auction_result_20260715.html",
        "sha256": "98A4DE63E7C8D427ECBBA9156EDE47F7BED9B802A5508DBE4B73C98801F499D6"},
}
RATE_SOURCE_BY_VALUATION = {"2026-07-01": "2026-07-01", "2026-07-15": "2026-07-15", "2026-07-22": "2026-07-15"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def validated_rate_contract() -> dict[str, dict[str, Any]]:
    validated: dict[str, dict[str, Any]] = {}
    for valuation_date, source_date in RATE_SOURCE_BY_VALUATION.items():
        source = dict(RATE_SOURCES[source_date]); path = RATE_PROVENANCE_ROOT / source["html"]
        if not path.is_file() or sha256(path) != source["sha256"]:
            raise RuntimeError(f"RBI rate provenance hash mismatch: {path}")
        plain = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", path.read_text(encoding="utf-8", errors="replace"))))
        expected = (f"Date : {date.fromisoformat(source['release_date']).strftime('%b %d, %Y')}",
                    "Treasury Bills: Full Auction Result", f"{source['cutoff_price']:.4f}",
                    f"(YTM: {source['yield'] * 100:.4f}%)", source["source_identifier"].replace("RBI Press Release ", ""))
        if any(field not in plain for field in expected):
            raise RuntimeError(f"RBI rate provenance fields missing: {path}")
        if date.fromisoformat(source["release_date"]) > date.fromisoformat(valuation_date):
            raise RuntimeError(f"future RBI rate observation for {valuation_date}")
        source["observed"] = source["release_date"]
        source["preserved_path"] = path.relative_to(REPOSITORY_ROOT).as_posix()
        validated[valuation_date] = source
    return validated


def _raw_paths(value: str) -> tuple[Path, Path]:
    stamp = value.replace("-", "")
    root = RAW_ROOT / value
    return (root / f"BhavCopy_NSE_CM_0_0_0_{stamp}_F_0000.csv",
            root / f"BhavCopy_NSE_FO_0_0_0_{stamp}_F_0000.csv")


def _assign_targets(group: pd.DataFrame) -> dict[int, float]:
    eligible = group.loc[
        group["activity_eligible"] & group["within_moneyness"] & group["iv_eligible"]
    ].copy()
    eligible = eligible.sort_values(["strike", "FinInstrmId"]).drop_duplicates("strike", keep="first")
    candidates = eligible["log_moneyness"].to_numpy(float)
    if not len(candidates):
        return {}
    costs = np.abs(np.asarray(TARGETS)[:, None] - candidates[None, :])
    costs += 1e-12 * np.arange(len(candidates))[None, :]
    costs[costs > 0.05 + 1e-12] = 1e6
    dummy = np.full((len(TARGETS), len(TARGETS)), 1e6)
    np.fill_diagonal(dummy, 0.0500000001)
    left, right = linear_sum_assignment(np.concatenate([costs, dummy], axis=1))
    return {int(eligible.index[j]): float(TARGETS[i]) for i, j in zip(left, right, strict=True) if j < len(eligible)}


def _date_panel(value: str) -> pd.DataFrame:
    valuation = date.fromisoformat(value)
    cm_path, fo_path = _raw_paths(value)
    cm = read_udiff_csv(cm_path, valuation, "CM")
    fo = read_udiff_csv(fo_path, valuation, "FO")
    spot_rows = cm.loc[(cm["TckrSymb"] == "NTPC") & (cm["SctySrs"] == "EQ")]
    if len(spot_rows) != 1:
        raise RuntimeError(f"expected one NTPC EQ spot row for {value}")
    spot = float(spot_rows.iloc[0]["ClsPric"])
    futures = fo.loc[(fo["TckrSymb"] == "NTPC") & (fo["FinInstrmTp"] == "STF")].copy()
    futures["expiry"] = pd.to_datetime(futures["FininstrmActlXpryDt"]).dt.date
    expiries = tuple(sorted(futures["expiry"].unique())[:2])
    rate_contract = validated_rate_contract()[value]
    rate_yield = float(rate_contract["yield"])
    future_contract: dict[date, dict[str, float]] = {}
    for expiry in expiries:
        row = futures.loc[futures["expiry"] == expiry]
        if len(row) != 1:
            raise RuntimeError(f"expected one NTPC future for {value}/{expiry}")
        row = row.iloc[0]
        if min(float(row["ClsPric"]), float(row["TtlTradgVol"]), float(row["OpnIntrst"]), float(row["TtlNbOfTxsExctd"])) <= 0:
            raise RuntimeError(f"inactive matched future for {value}/{expiry}")
        dte = (expiry - valuation).days
        maturity = dte / 365.0
        discount = 1.0 / (1.0 + rate_yield * maturity)
        rate = -math.log(discount) / maturity
        forward = float(row["ClsPric"])
        carry = rate - math.log(forward / spot) / maturity
        future_contract[expiry] = {"forward": forward, "discount": discount, "rate": rate, "carry": carry}
    options = fo.loc[(fo["TckrSymb"] == "NTPC") & (fo["FinInstrmTp"] == "STO")].copy()
    options["expiry_obj"] = pd.to_datetime(options["FininstrmActlXpryDt"]).dt.date
    options = options.loc[options["expiry_obj"].isin(expiries)].copy()
    options["valuation_date"] = value
    options["expiry_date"] = options["expiry_obj"].map(date.isoformat)
    options["DTE"] = options["expiry_obj"].map(lambda expiry: (expiry - valuation).days)
    options["T"] = options["DTE"].to_numpy(float) / 365.0
    options["option_type"] = options["OptnTp"].map({"CE": "call", "PE": "put"})
    options["strike"] = pd.to_numeric(options["StrkPric"], errors="raise")
    options["spot"] = spot
    options["log_moneyness"] = np.log(options["strike"] / spot)
    options["observed_price"] = pd.to_numeric(options["ClsPric"], errors="coerce")
    options["traded_volume"] = pd.to_numeric(options["TtlTradgVol"], errors="coerce")
    options["open_interest"] = pd.to_numeric(options["OpnIntrst"], errors="coerce")
    options["trade_count"] = pd.to_numeric(options["TtlNbOfTxsExctd"], errors="coerce")
    options["activity_eligible"] = (options["observed_price"].gt(0) & options["traded_volume"].gt(0)
                                    & options["open_interest"].gt(0) & options["trade_count"].gt(0))
    options["within_moneyness"] = options["log_moneyness"].abs() <= 0.10 + 1e-12
    options["matched_futures_price"] = options["expiry_obj"].map(lambda x: future_contract[x]["forward"])
    options["risk_free_simple_yield"] = rate_yield
    options["discount_factor"] = options["expiry_obj"].map(lambda x: future_contract[x]["discount"])
    options["continuous_rate"] = options["expiry_obj"].map(lambda x: future_contract[x]["rate"])
    options["futures_implied_carry"] = options["expiry_obj"].map(lambda x: future_contract[x]["carry"])
    market_iv: list[float] = []
    for row in options.itertuples():
        try:
            market_iv.append(pilot.implied_volatility(
                float(row.observed_price), float(row.matched_futures_price), float(row.strike),
                float(row.T), float(row.discount_factor), str(row.option_type)
            ))
        except ValueError:
            market_iv.append(np.nan)
    options["market_implied_volatility"] = market_iv
    options["iv_eligible"] = np.isfinite(options["market_implied_volatility"])
    options["target_log_moneyness"] = np.nan
    for (_, _), group in options.groupby(["expiry_date", "option_type"], sort=True):
        for index, target in _assign_targets(group).items():
            options.loc[index, "target_log_moneyness"] = target
    options["sample_role"] = "EXCLUDED"
    options.loc[options["target_log_moneyness"].isin(CALIBRATION_TARGETS), "sample_role"] = "CALIBRATION"
    options.loc[options["target_log_moneyness"].isin(HOLDOUT_TARGETS), "sample_role"] = "HOLDOUT"
    options = options.loc[options["sample_role"] != "EXCLUDED"].copy()
    options["cm_source_sha256"] = sha256(cm_path)
    options["fo_source_sha256"] = sha256(fo_path)
    options["rate_observation_date"] = rate_contract["observed"]
    if any(date.fromisoformat(item) > valuation for item in options["rate_observation_date"]):
        raise RuntimeError("future rate observation leaked into a valuation date")
    return options[["valuation_date", "expiry_date", "DTE", "T", "option_type", "strike", "spot",
                    "log_moneyness", "observed_price", "traded_volume", "open_interest", "trade_count",
                    "activity_eligible", "matched_futures_price", "risk_free_simple_yield", "discount_factor",
                    "continuous_rate", "futures_implied_carry", "market_implied_volatility",
                    "target_log_moneyness", "sample_role", "cm_source_sha256", "fo_source_sha256",
                    "rate_observation_date"]].sort_values(
        ["sample_role", "expiry_date", "option_type", "target_log_moneyness"]
    ).reset_index(drop=True)


def build_three_date_panel() -> pd.DataFrame:
    return pd.concat([_date_panel(value) for value in VALUATION_DATES], ignore_index=True)


def support_inventory(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for valuation_date in VALUATION_DATES:
        expiries = sorted(panel.loc[panel["valuation_date"] == valuation_date, "expiry_date"].unique())
        for expiry_position, expiry_date in zip(("near", "middle"), expiries, strict=True):
            for option_type in ("call", "put"):
                for target in TARGETS:
                    selected = panel.loc[
                        (panel["valuation_date"] == valuation_date)
                        & (panel["expiry_date"] == expiry_date)
                        & (panel["option_type"] == option_type)
                        & (panel["target_log_moneyness"] == target)
                    ]
                    rows.append({"valuation_date": valuation_date, "expiry_position": expiry_position,
                                 "expiry_date": expiry_date, "option_type": option_type,
                                 "target_log_moneyness": target, "selected": len(selected) == 1,
                                 "sample_role": selected.iloc[0]["sample_role"] if len(selected) else "MISSING",
                                 "strike": float(selected.iloc[0]["strike"]) if len(selected) else None})
    return pd.DataFrame(rows)


def map_joint_coordinate(coordinate: Sequence[float], bounds: dict[str, tuple[float, float]]) -> dict[str, Any]:
    value = np.asarray(coordinate, dtype=float)
    if value.shape != (14,):
        raise ValueError("joint coordinate must have 14 quantities")
    base = unconstrained_to_parameters(value[:10], bounds)
    structural = {name: float(base[index]) for name, index in zip(SHARED_NAMES, SHARED_CANONICAL_INDEX, strict=True)}
    states = {VALUATION_DATES[0]: {"v0_slow": float(base[4]), "v0_fast": float(base[9])}}
    for offset, valuation_date in enumerate(VALUATION_DATES[1:]):
        slow_unit, fast_unit = expit(value[10 + 2 * offset:12 + 2 * offset])
        slow_bounds, fast_bounds = bounds["v0_slow"], bounds["v0_fast"]
        states[valuation_date] = {
            "v0_slow": float(slow_bounds[0] + slow_unit * (slow_bounds[1] - slow_bounds[0])),
            "v0_fast": float(fast_bounds[0] + fast_unit * (fast_bounds[1] - fast_bounds[0])),
        }
    return {"structural": structural, "states": states}


def canonical_vector(mapped: dict[str, Any], valuation_date: str) -> np.ndarray:
    s, v = mapped["structural"], mapped["states"][valuation_date]
    return np.asarray([s["kappa_slow"], s["theta_slow"], s["sigma_slow"], s["rho_slow"], v["v0_slow"],
                       s["kappa_fast"], s["theta_fast"], s["sigma_fast"], s["rho_fast"], v["v0_fast"]])


def date_balanced_residual(errors: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(errors[value], dtype=float) / math.sqrt(len(errors[value])) for value in VALUATION_DATES])


def reported_date_balanced_objective(errors: dict[str, np.ndarray]) -> float:
    return float(np.sqrt(np.mean([np.mean(np.asarray(errors[value], dtype=float) ** 2) for value in VALUATION_DATES])))


def joint_start_population() -> tuple[list[np.ndarray], pd.DataFrame]:
    bounds = load_hard_safety_bounds(BOUNDS_PATH)
    _, _, reviewed = geometry.paired_start_population(bounds)
    rng = np.random.default_rng(pilot.ANALYSIS_SEED + 300)
    extras = [np.zeros(4)] + [rng.normal(0.0, 1.25, 4) for _ in range(START_COUNT - 1)]
    starts, rows = [], []
    for start_id in range(START_COUNT):
        base = reviewed.loc[reviewed["start_id"] == start_id, [f"baseline_z_{i}" for i in range(10)]].iloc[0].to_numpy(float)
        start = np.concatenate([base, extras[start_id]])
        starts.append(start)
        rows.append({"start_id": start_id, "joint_start_sha256": hashlib.sha256(start.tobytes()).hexdigest().upper(),
                     "canonical_start_sha256": reviewed.loc[reviewed["start_id"] == start_id, "canonical_start_sha256"].item()})
    return starts, pd.DataFrame(rows)


def classify_multi_date(baseline: dict[str, float], result: dict[str, float]) -> str:
    improve = lambda key: 1.0 - result[key] / baseline[key]
    holdout_ok = result["holdout"] <= baseline["holdout"] * 1.05
    if (improve("median") >= 0.25 and improve("maximum") >= 0.25 and improve("clusters") >= 0.40
            and improve("displaced") >= 0.50 and holdout_ok):
        return "STRONG_MULTI_DATE_STABILITY_IMPROVEMENT"
    if ((improve("median") >= 0.10 or improve("maximum") >= 0.10 or improve("clusters") >= 0.10
         or improve("displaced") >= 0.10) and result["clusters"] <= baseline["clusters"] and holdout_ok):
        return "PARTIAL_MULTI_DATE_STABILITY_IMPROVEMENT"
    return "MULTI_DATE_INSUFFICIENT"


def _price(frame: pd.DataFrame, vector: np.ndarray) -> np.ndarray:
    return np.asarray([pilot._double_heston_row_price(row, vector, NODE_COUNT) for _, row in frame.iterrows()])


def _metrics(frame: pd.DataFrame, prices: np.ndarray) -> dict[str, float]:
    return pilot._metrics(frame, prices)


def _fit(panel: pd.DataFrame) -> pd.DataFrame:
    bounds = load_hard_safety_bounds(BOUNDS_PATH)
    calibration = panel.loc[panel["sample_role"] == "CALIBRATION"]
    holdout = panel.loc[panel["sample_role"] == "HOLDOUT"]
    starts, _ = joint_start_population()
    rows = []
    for start_id, start in enumerate(starts):
        begun = time.perf_counter()
        def errors(z: np.ndarray, source: pd.DataFrame = calibration) -> dict[str, np.ndarray]:
            mapped = map_joint_coordinate(z, bounds)
            return {value: _price(source.loc[source["valuation_date"] == value], canonical_vector(mapped, value))
                    - source.loc[source["valuation_date"] == value, "observed_price"].to_numpy(float)
                    for value in VALUATION_DATES}
        result = least_squares(lambda z: date_balanced_residual(errors(z)), start, method="trf", max_nfev=MAX_NFEV,
                               ftol=1e-9, xtol=1e-9, gtol=1e-9, diff_step=2e-5)
        mapped = map_joint_coordinate(result.x, bounds)
        item: dict[str, Any] = {"start_id": start_id, "optimizer_success": bool(result.success),
            "optimizer_status": int(result.status), "optimizer_message": str(result.message), "nfev": int(result.nfev),
            "reached_cap": int(result.nfev) >= MAX_NFEV, "runtime_seconds": time.perf_counter() - begun,
            "date_balanced_objective": reported_date_balanced_objective(errors(result.x))}
        item.update(mapped["structural"])
        for value in VALUATION_DATES:
            state = mapped["states"][value]
            item[f"v0_slow_{value}"] = state["v0_slow"]; item[f"v0_fast_{value}"] = state["v0_fast"]
            item[f"v0_total_{value}"] = state["v0_slow"] + state["v0_fast"]
            vector = canonical_vector(mapped, value)
            item[f"boundary_reasons_{value}"] = ";".join(boundary_diagnostics(vector, bounds))
            for role, source in (("calibration", calibration), ("holdout", holdout)):
                subset = source.loc[source["valuation_date"] == value]
                for name, metric in _metrics(subset, _price(subset, vector)).items():
                    item[f"{role}_{name}_{value}"] = metric
        rows.append(item)
    return pd.DataFrame(rows)


def _stability(frame: pd.DataFrame, names: Sequence[str], bounds: dict[str, tuple[float, float]]) -> tuple[dict[str, Any], pd.DataFrame]:
    if frame.empty:
        raise RuntimeError("stability analysis has no fitted solutions")
    best = frame.sort_values(["date_balanced_objective", "start_id"]).iloc[0]
    threshold = max(float(best["date_balanced_objective"]) * 1.05, float(best["date_balanced_objective"]) + 0.01)
    near = frame.loc[frame["date_balanced_objective"] <= threshold].copy()
    if near.empty:
        raise RuntimeError("stability analysis has no near-equivalent solutions; best-solution invariant failed")
    widths = np.asarray([bounds[name][1] - bounds[name][0] for name in names])
    scaled = near[list(names)].to_numpy(float) / widths / math.sqrt(len(names))
    distances = pdist(scaled) if len(near) > 1 else np.array([])
    best_scaled = best[list(names)].to_numpy(float) / widths / math.sqrt(len(names))
    from_best = np.linalg.norm(scaled - best_scaled, axis=1)
    labels = (
        fcluster(linkage(scaled, method="complete"), CLUSTER_DISTANCE, criterion="distance")
        if len(near) > 1
        else np.ones(1, dtype=int)
    )
    near["distance_from_best"] = from_best; near["cluster_id"] = labels
    stats = {name: {"minimum": float(near[name].min()), "maximum": float(near[name].max()),
                    "range": float(near[name].max() - near[name].min()),
                    "coefficient_of_variation": float(near[name].std(ddof=0) / abs(near[name].mean())) if abs(near[name].mean()) > 1e-12 else None}
             for name in names}
    boundary_columns = [name for name in near if name == "boundary_reasons" or name.startswith("boundary_reasons_")]
    boundary_hits = (near[boundary_columns].fillna("").astype(str).apply(lambda x: x.str.len().gt(0)).any(axis=1)
                     if boundary_columns else pd.Series(False, index=near.index))
    return {"near_equivalent_count": len(near), "materially_displaced_count": int(np.sum(from_best >= MATERIAL_DISTANCE)),
            "cluster_count": len(set(labels)),
            "median_pairwise_distance": float(np.median(distances)) if len(distances) else 0.0,
            "maximum_pairwise_distance": float(np.max(distances)) if len(distances) else 0.0,
            "maximum_distance_from_best": float(np.max(from_best)) if len(from_best) else 0.0,
            "parameter_statistics": stats, "boundary_hit_rate": float(boundary_hits.mean()),
            "cap_rate": float(frame["reached_cap"].mean()), "optimizer_success_rate": float(frame["optimizer_success"].mean())}, near


def single_date_shared_baseline(bounds: dict[str, tuple[float, float]]) -> dict[str, Any]:
    baseline = pd.read_csv(pilot.OUTPUT_ROOT / "double_heston_multistart.csv").copy()
    baseline["date_balanced_objective"] = baseline["calibration_price_rmse"]
    baseline["reached_cap"] = baseline["nfev"].astype(int) >= pilot.MAX_NFEV
    result, _ = _stability(baseline, SHARED_NAMES, bounds)
    return result


def _figures(panel: pd.DataFrame, near: pd.DataFrame, summary: dict[str, Any]) -> list[Path]:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True); paths=[]
    def save(name: str, fig: Any) -> None:
        path=FIGURE_ROOT/name; fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig); paths.append(path)
    counts=panel.groupby(["valuation_date","sample_role"]).size().unstack(fill_value=0)
    fig,ax=plt.subplots(); counts.plot.bar(ax=ax); ax.set_ylabel("selected rows"); save("01_three_date_option_support.png",fig)
    best=near.sort_values(["date_balanced_objective","start_id"]).iloc[0]
    fig,ax=plt.subplots(); ax.bar(["single 15-Jul","multi 01-Jul","multi 15-Jul","multi 22-Jul"],[SINGLE_DATE_HOLDOUT]+[best[f"holdout_price_rmse_{v}"] for v in VALUATION_DATES]); ax.tick_params(axis="x",rotation=20); save("02_holdout_rmse.png",fig)
    fig,ax=plt.subplots(figsize=(10,4)); ax.boxplot([near[n] for n in SHARED_NAMES],tick_labels=SHARED_NAMES); ax.tick_params(axis="x",rotation=35); save("03_shared_parameter_dispersion.png",fig)
    fig,ax=plt.subplots(); ax.bar(["single clusters","multi clusters","single median","multi median"],[summary["baseline_shared"]["cluster_count"],summary["multi_shared"]["cluster_count"],summary["baseline_shared"]["median_pairwise_distance"],summary["multi_shared"]["median_pairwise_distance"]]); ax.tick_params(axis="x",rotation=25); save("04_cluster_separation.png",fig)
    baseline_frame=pd.read_csv(pilot.OUTPUT_ROOT/"double_heston_multistart.csv"); baseline_frame["date_balanced_objective"]=baseline_frame["calibration_price_rmse"]; baseline_frame["reached_cap"]=baseline_frame["nfev"].astype(int)>=pilot.MAX_NFEV
    _,baseline_near=_stability(baseline_frame,SHARED_NAMES,load_hard_safety_bounds(BOUNDS_PATH))
    fig,ax=plt.subplots(); ax.boxplot([np.log(2)/baseline_near["kappa_slow"]*365,np.log(2)/near["kappa_slow"]*365,np.log(2)/baseline_near["kappa_fast"]*365,np.log(2)/near["kappa_fast"]*365],tick_labels=["single slow","multi slow","single fast","multi fast"]); ax.set_ylabel("half-life days"); ax.tick_params(axis="x",rotation=20); save("05_kappa_half_life.png",fig)
    fig,ax=plt.subplots(); x=np.arange(3); slow=[best[f"v0_slow_{v}"] for v in VALUATION_DATES]; fast=[best[f"v0_fast_{v}"] for v in VALUATION_DATES]; ax.bar(x,slow,label="slow"); ax.bar(x,fast,bottom=slow,label="fast"); ax.set_xticks(x,VALUATION_DATES,rotation=20); ax.legend(); save("06_date_specific_variance.png",fig)
    bounds=load_hard_safety_bounds(BOUNDS_PATH); mapped={"structural":{n:float(best[n]) for n in SHARED_NAMES},"states":{v:{"v0_slow":float(best[f'v0_slow_{v}']),"v0_fast":float(best[f'v0_fast_{v}'])} for v in VALUATION_DATES}}
    fig,ax=plt.subplots(); fig2,ax2=plt.subplots()
    for v in VALUATION_DATES:
        source=panel.loc[panel["valuation_date"]==v]; prices=_price(source,canonical_vector(mapped,v)); iv=[pilot.implied_volatility(float(p),float(r.matched_futures_price),float(r.strike),float(r.T),float(r.discount_factor),str(r.option_type)) for p,r in zip(prices,source.itertuples(),strict=True)]
        ax.scatter(source["market_implied_volatility"],iv,label=v); ax2.scatter(source["target_log_moneyness"],prices-source["observed_price"],label=v)
    ax.set(xlabel="market IV",ylabel="model IV"); ax.legend(); save("07_market_vs_model_iv.png",fig)
    ax2.axhline(0,color="black",lw=.8); ax2.set(xlabel="target log-moneyness",ylabel="price residual"); ax2.legend(); save("08_pricing_residuals.png",fig2)
    return paths


def _overall_rmse(best: pd.Series, row_counts: dict[str, dict[str, int]], role: str) -> float:
    squared_sum = sum(row_counts[value][role.upper()] * float(best[f"{role}_price_rmse_{value}"]) ** 2
                      for value in VALUATION_DATES)
    count = sum(row_counts[value][role.upper()] for value in VALUATION_DATES)
    return math.sqrt(squared_sum / count)


def _publish(panel: pd.DataFrame, fits: pd.DataFrame) -> dict[str, Any]:
    bounds=load_hard_safety_bounds(BOUNDS_PATH); rate_contract=validated_rate_contract()
    multi,near=_stability(fits,SHARED_NAMES,bounds); near.to_csv(OUTPUT_ROOT/"near_equivalent.csv",index=False,lineterminator="\n")
    baseline=single_date_shared_baseline(bounds); best=near.sort_values(["date_balanced_objective","start_id"]).iloc[0]
    classification=classify_multi_date({"median":baseline["median_pairwise_distance"],"maximum":baseline["maximum_pairwise_distance"],"clusters":baseline["cluster_count"],"displaced":baseline["materially_displaced_count"],"holdout":SINGLE_DATE_HOLDOUT},
        {"median":multi["median_pairwise_distance"],"maximum":multi["maximum_pairwise_distance"],"clusters":multi["cluster_count"],"displaced":multi["materially_displaced_count"],"holdout":float(best["holdout_price_rmse_2026-07-15"])})
    row_counts=panel.groupby(["valuation_date","sample_role"]).size().unstack(fill_value=0).to_dict(orient="index")
    support=support_inventory(panel)
    summary={"classification":classification,"baseline_shared":baseline,"multi_shared":multi,"row_counts":row_counts,"support_counts":{value:{"selected":int(support.loc[support["valuation_date"]==value,"selected"].sum()),"missing":int((~support.loc[support["valuation_date"]==value,"selected"]).sum())} for value in VALUATION_DATES},"best_start_id":int(best["start_id"]),"best":{key:(bool(value) if isinstance(value,(np.bool_,bool)) else float(value) if isinstance(value,(np.floating,float)) else int(value) if isinstance(value,(np.integer,int)) else str(value)) for key,value in best.items()},"runtime":float(fits["runtime_seconds"].sum()),"optimizer":{"max_nfev":MAX_NFEV,"reason":"highest completed fixed budget; no further budget study"},"rate_contract":rate_contract}
    summary["overall_price_rmse"]={"calibration":_overall_rmse(best,row_counts,"calibration"),"holdout":_overall_rmse(best,row_counts,"holdout")}
    total_count=sum(sum(row_counts[value].values()) for value in VALUATION_DATES)
    summary["overall_price_rmse"]["all_selected"] = math.sqrt(sum(row_counts[value][role] * float(best[f"{role.lower()}_price_rmse_{value}"]) ** 2 for value in VALUATION_DATES for role in ("CALIBRATION","HOLDOUT")) / total_count)
    summary["stability_improvements"]={key:1.0-multi[result]/baseline[result] for key,result in (("median","median_pairwise_distance"),("maximum","maximum_pairwise_distance"),("clusters","cluster_count"),("displaced","materially_displaced_count"))}
    baseline_frame=pd.read_csv(pilot.OUTPUT_ROOT/"double_heston_multistart.csv"); baseline_frame["date_balanced_objective"]=baseline_frame["calibration_price_rmse"]; baseline_frame["reached_cap"]=baseline_frame["nfev"].astype(int)>=pilot.MAX_NFEV
    _,baseline_near=_stability(baseline_frame,SHARED_NAMES,bounds); single_best=baseline_near.sort_values(["date_balanced_objective","start_id"]).iloc[0]
    def timescales(source: pd.DataFrame) -> dict[str, dict[str, float]]:
        return {speed:{"kappa_minimum":float(source[f"kappa_{speed}"].min()),"kappa_maximum":float(source[f"kappa_{speed}"].max()),"half_life_minimum_days":float(math.log(2)/source[f"kappa_{speed}"].max()*365),"half_life_maximum_days":float(math.log(2)/source[f"kappa_{speed}"].min()*365)} for speed in ("slow","fast")}
    summary["timescale_comparison"]={"single_date_best":{"kappa_slow":float(single_best["kappa_slow"]),"kappa_fast":float(single_best["kappa_fast"]),"slow_half_life_days":float(math.log(2)/single_best["kappa_slow"]*365),"fast_half_life_days":float(math.log(2)/single_best["kappa_fast"]*365)},"multi_date_best":{"kappa_slow":float(best["kappa_slow"]),"kappa_fast":float(best["kappa_fast"]),"slow_half_life_days":float(math.log(2)/best["kappa_slow"]*365),"fast_half_life_days":float(math.log(2)/best["kappa_fast"]*365)},"single_date_near_equivalent":timescales(baseline_near),"multi_date_near_equivalent":timescales(near),"conclusion":"MIXED_TIMESCALE_STABILITY_INSUFFICIENT"}
    figures=_figures(panel,near,summary); summary["figures"]=[p.relative_to(REPOSITORY_ROOT).as_posix() for p in figures]
    write_json(OUTPUT_ROOT/"summary.json",summary)
    render_report(summary)
    artifacts={p.relative_to(REPOSITORY_ROOT).as_posix():sha256(p) for p in OUTPUT_ROOT.rglob("*") if p.is_file() and RATE_PROVENANCE_ROOT not in p.parents}
    source_paths=[p for v in VALUATION_DATES for p in _raw_paths(v)]+[RATE_PROVENANCE_ROOT/RATE_SOURCES[v]["html"] for v in RATE_SOURCES]
    manifest={"base_main":"e1e5b0fba1795c84b8c7a8cb534a8962dc1d22e0","analysis_id":"NTPC_DH_MULTI_DATE_CALIBRATION","valuation_dates":VALUATION_DATES,"parameter_contract":"8 shared structural + 2 states per date; canonical 10-vector per date","date_balanced_loss":"raw price residuals divided by sqrt(calibration row count per date)","classification":classification,"rate_contract":rate_contract,"source_hashes":{str(p.relative_to(REPOSITORY_ROOT)).replace('\\','/'):sha256(p) for p in source_paths},"generated_artifact_hashes":artifacts,"report_sha256":sha256(REPORT_PATH),"runtime":{"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"scipy":scipy.__version__,"pandas":pd.__version__}}
    write_json(MANIFEST_PATH,manifest); return summary


def run() -> dict[str, Any]:
    panel=build_three_date_panel(); _,start_hashes=joint_start_population()
    OUTPUT_ROOT.mkdir(parents=True,exist_ok=True); panel.to_csv(OUTPUT_ROOT/"selected_options.csv",index=False,lineterminator="\n"); support_inventory(panel).to_csv(OUTPUT_ROOT/"target_support.csv",index=False,lineterminator="\n"); start_hashes.to_csv(OUTPUT_ROOT/"joint_starts.csv",index=False,lineterminator="\n")
    fits=_fit(panel); fits.to_csv(OUTPUT_ROOT/"multistart.csv",index=False,lineterminator="\n")
    return _publish(panel,fits)


def render_existing_outputs() -> dict[str, Any]:
    return _publish(pd.read_csv(OUTPUT_ROOT/"selected_options.csv"),pd.read_csv(OUTPUT_ROOT/"multistart.csv"))


def verify_manifest_artifacts() -> list[str]:
    manifest=json.loads(MANIFEST_PATH.read_text(encoding="utf-8")); failures=[]
    for section in ("source_hashes","generated_artifact_hashes"):
        for relative,expected in manifest[section].items():
            path=REPOSITORY_ROOT/relative
            if not path.is_file() or sha256(path) != expected:
                failures.append(relative)
    if sha256(REPORT_PATH) != manifest["report_sha256"]:
        failures.append(REPORT_PATH.relative_to(REPOSITORY_ROOT).as_posix())
    return failures


def render_report(summary: dict[str, Any]) -> None:
    best=summary["best"]; rows=[]
    for v in VALUATION_DATES:
        rows.append(f"| {v} | {summary['row_counts'][v].get('CALIBRATION',0)} | {summary['row_counts'][v].get('HOLDOUT',0)} | {best[f'calibration_price_rmse_{v}']:.9g} | {best[f'holdout_price_rmse_{v}']:.9g} | {best[f'calibration_iv_rmse_{v}']:.9g} | {best[f'holdout_iv_rmse_{v}']:.9g} |")
    shared="\n".join(f"| {n} | {best[n]:.9g} |" for n in SHARED_NAMES)
    states="\n".join(f"| {v} | {best[f'v0_slow_{v}']:.9g} | {best[f'v0_fast_{v}']:.9g} | {best[f'v0_total_{v}']:.9g} |" for v in VALUATION_DATES)
    slow_hl=math.log(2)/best["kappa_slow"]*365; fast_hl=math.log(2)/best["kappa_fast"]*365
    b,m=summary["baseline_shared"],summary["multi_shared"]
    improvements=summary["stability_improvements"]
    ranges="\n".join(f"| {name} | {m['parameter_statistics'][name]['minimum']:.9g} | {m['parameter_statistics'][name]['maximum']:.9g} | {m['parameter_statistics'][name]['range']:.9g} | {m['parameter_statistics'][name]['coefficient_of_variation']:.9g} |" for name in SHARED_NAMES)
    support="\n".join(f"| {value} | {summary['support_counts'][value]['selected']} | {summary['support_counts'][value]['missing']} |" for value in VALUATION_DATES)
    times=summary["timescale_comparison"]; single_times=times["single_date_best"]; multi_times=times["multi_date_best"]; single_ranges=times["single_date_near_equivalent"]; multi_ranges=times["multi_date_near_equivalent"]
    REPORT_PATH.write_text(f"""# NTPC Double Heston multi-date calibration

## Why this was attempted

The corrected 160-vs-320 comparison was valid but returned `OPTIMIZER_CAP_UNRESOLVED`: cap incidence stayed 10/12 under both charts and dispersion persisted. Optimizer-only work is closed, so this predeclared three-date real-market test asks whether additional NTPC surfaces stabilize shared dynamics.

## Data and formulation

Official NSE UDiFF CM/F&O rows are used only for `2026-07-01`, `2026-07-15`, and `2026-07-22`, with the first two listed expiries, active actual strikes nearest `[-0.10,-0.05,0,+0.05,+0.10]`, inner targets for calibration and outer targets for holdout. `T=DTE/365` is reconstructed. The official RBI 91-day cut-offs are 5.2521% from Press Release 2026-2027/584 dated 1 July and 5.3324% from Press Release 2026-2027/672 dated 15 July; the latter is the latest available observation carried into 22 July. Both dated RBI HTML responses are locally preserved, hash-sealed, and field-validated before panel construction, so no future observation is used.

Eight structural parameters are shared; each date has its own `v0_slow,v0_fast`. Every date is priced by the canonical ten-vector, so this remains canonical Double Heston—not a new 14-parameter model. The joint loss uses raw price residuals divided by `sqrt(n_date)`; unweighted metrics are reported below. Twelve deterministic starts use the fixed 320 budget selected before results.

The 60-cell target inventory is preserved separately and missing cells are explicit rather than interpolated:

| date | selected cells | missing cells |
|---|---:|---:|
{support}

| date | calibration rows | holdout rows | calibration RMSE | holdout RMSE | calibration IV RMSE | holdout IV RMSE |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Best shared parameters

| parameter | value |
|---|---:|
{shared}

| date | v0_slow | v0_fast | v0_total |
|---|---:|---:|---:|
{states}

Slow half-life is `{slow_hl:.6g}` days; fast half-life is `{fast_hl:.6g}` days. The primary direct-comparison canonical vector is the `2026-07-15` vector.

The reviewed single-date best had `kappa_slow={single_times['kappa_slow']:.9g}` and `kappa_fast={single_times['kappa_fast']:.9g}`, implying `{single_times['slow_half_life_days']:.6g}`- and `{single_times['fast_half_life_days']:.6g}`-day half-lives. The multi-date best shifts those to `kappa_slow={multi_times['kappa_slow']:.9g}` and `kappa_fast={multi_times['kappa_fast']:.9g}`, or `{multi_times['slow_half_life_days']:.6g}` and `{multi_times['fast_half_life_days']:.6g}` days.

Across near-equivalent starts, slow half-lives span `{single_ranges['slow']['half_life_minimum_days']:.6g}`–`{single_ranges['slow']['half_life_maximum_days']:.6g}` days for single-date and `{multi_ranges['slow']['half_life_minimum_days']:.6g}`–`{multi_ranges['slow']['half_life_maximum_days']:.6g}` days for multi-date; fast half-lives span `{single_ranges['fast']['half_life_minimum_days']:.6g}`–`{single_ranges['fast']['half_life_maximum_days']:.6g}` and `{multi_ranges['fast']['half_life_minimum_days']:.6g}`–`{multi_ranges['fast']['half_life_maximum_days']:.6g}` days. Slow-kappa range/CV improve, but fast-kappa CV worsens and both multi-date half-life ranges remain broad. Timescale stability is therefore **mixed and insufficient**, not resolved.

Overall raw-price RMSE is `{summary['overall_price_rmse']['calibration']:.9g}` on calibration rows, `{summary['overall_price_rmse']['holdout']:.9g}` on holdout rows, and `{summary['overall_price_rmse']['all_selected']:.9g}` over all selected rows. The best reported date-balanced objective is `{best['date_balanced_objective']:.9g}`.

## Stability and model comparison

| metric | reviewed single-date shared-8 | multi-date shared-8 |
|---|---:|---:|
| materially displaced | {b['materially_displaced_count']} | {m['materially_displaced_count']} |
| clusters | {b['cluster_count']} | {m['cluster_count']} |
| median separation | {b['median_pairwise_distance']:.9g} | {m['median_pairwise_distance']:.9g} |
| maximum separation | {b['maximum_pairwise_distance']:.9g} | {m['maximum_pairwise_distance']:.9g} |
| maximum distance from best | {b['maximum_distance_from_best']:.9g} | {m['maximum_distance_from_best']:.9g} |
| boundary-hit rate | {b['boundary_hit_rate']:.6g} | {m['boundary_hit_rate']:.6g} |
| cap rate | {b['cap_rate']:.6g} | {m['cap_rate']:.6g} |
| optimizer success rate | {b['optimizer_success_rate']:.6g} | {m['optimizer_success_rate']:.6g} |

Relative to the reviewed single-date shared-eight comparator, median separation improved `{improvements['median']:.3%}`, maximum separation `{improvements['maximum']:.3%}`, cluster count `{improvements['clusters']:.3%}`, and materially displaced count `{improvements['displaced']:.3%}`.

| shared parameter | minimum | maximum | range | coefficient of variation |
|---|---:|---:|---:|---:|
{ranges}

For 15 July, reviewed single-date DH holdout RMSE was `{SINGLE_DATE_HOLDOUT:.9g}` and Standard Heston was `{HESTON_HOLDOUT:.6g}`. The multi-date 15 July holdout is `{best['holdout_price_rmse_2026-07-15']:.9g}`, a `{best['holdout_price_rmse_2026-07-15']/SINGLE_DATE_HOLDOUT-1:.3%}` worsening that exceeds the predeclared 5% ceiling. Double Heston is not declared superior unless this evidence supports it.

The 12-start joint optimization consumed `{summary['runtime']:.6g}` seconds in total.

Final classification: **{summary['classification']}**.

The principal remaining limitation is that additional dates do not automatically turn local fit quality into unique recovery; shared timescales and allocations must be judged from the multi-start dispersion, cluster, boundary, and holdout evidence above.
""",encoding="utf-8",newline="\n")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--render-only",action="store_true"); args=parser.parse_args()
    print(json.dumps(render_existing_outputs() if args.render_only else run(),indent=2))
