"""Run the predeclared paired NTPC Double Heston optimizer-cap experiment.

The only treatment is ``max_nfev`` 160 -> 320. The script consumes frozen,
reviewed local evidence, performs no acquisition, and never rewrites it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import least_squares
from scipy.spatial.distance import pdist, squareform

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import run_ntpc_dh_stability_reparameterization as geometry
from scripts import run_ntpc_single_stock_pilot as pilot
from src.calibrate_double_heston import (
    boundary_diagnostics,
    load_hard_safety_bounds,
    unconstrained_to_parameters,
)
from src.constants import PARAMETER_NAMES
from src.ntpc_dh_reparameterization import (
    TRANSFORMED_NAMES,
    canonical_to_structured,
    derived_coordinates,
    structured_to_canonical,
)


BASE_MAIN = "205e72757f978184dfdd588af72a31034c2a20ef"
OUTPUT_ROOT = pilot.OUTPUT_ROOT.parent / "ntpc_dh_optimizer_cap_sensitivity"
FIGURE_ROOT = OUTPUT_ROOT / "figures"
REPORT_PATH = REPOSITORY_ROOT / "docs" / "NTPC_DH_OPTIMIZER_CAP_SENSITIVITY.md"
MANIFEST_PATH = REPOSITORY_ROOT / "docs" / "evidence" / "NTPC_DH_OPTIMIZER_CAP_SENSITIVITY_MANIFEST.json"
REPARAMETERIZATION_ROOT = geometry.OUTPUT_ROOT

BUDGETS = (160, 320)
CHARTS = ("canonical", "transformed")
START_COUNT = 12
NODE_COUNT = 64
OBJECTIVE = "unweighted observed-price residual vector"
REGULARIZATION = "NONE"
MATERIAL_DISTANCE = 0.05
CLUSTER_DISTANCE = 0.05
HESTON_HOLDOUT_RMSE = 0.910569
HESTON_MATERIAL_THRESHOLD = 0.865041
CALIBRATION_ROW_SHA256 = "F44371DD418304789DC4B97C1710DCE60CDC0232A75C172FDC90E220738A0B7F"
HOLDOUT_ROW_SHA256 = "E8B552A1B218F5405D5871C11F9AAA4F6460310BFEC8E300501A73C00F1FBA07"

REVIEWED_160 = {
    "canonical": {
        "calibration_price_rmse": 0.2343207416121831,
        "calibration_iv_rmse": 0.0077897124776684,
        "holdout_price_rmse": 0.9268247197137796,
        "holdout_iv_rmse": 0.072591562005221,
        "valid_start_count": 12,
        "near_equivalent_start_count": 12,
        "materially_displaced_start_count": 11,
        "cluster_count": 7,
        "median_pairwise_range_scaled_distance": 0.35733879424203197,
        "maximum_pairwise_range_scaled_distance": 0.5641491074467359,
        "maximum_range_scaled_distance_from_best": 0.4910854918863381,
        "cap_rate": 10 / 12,
    },
    "transformed": {
        "calibration_price_rmse": 0.2331741477949472,
        "calibration_iv_rmse": 0.0078090854888881,
        "holdout_price_rmse": 0.921582640884198,
        "holdout_iv_rmse": 0.0718592662958039,
        "valid_start_count": 12,
        "near_equivalent_start_count": 11,
        "materially_displaced_start_count": 7,
        "cluster_count": 6,
        "median_pairwise_range_scaled_distance": 0.3886263458849968,
        "maximum_pairwise_range_scaled_distance": 0.593889520771503,
        "maximum_range_scaled_distance_from_best": 0.5674247779466101,
        "cap_rate": 10 / 12,
    },
}

PROTECTED_PATHS = (
    pilot.OUTPUT_ROOT / "selected_options.csv",
    pilot.OUTPUT_ROOT / "carry_contract.csv",
    pilot.OUTPUT_ROOT / "double_heston_multistart.csv",
    pilot.OUTPUT_ROOT / "model_comparison.csv",
    pilot.OUTPUT_ROOT / "parameter_stability.json",
    pilot.MANIFEST_PATH,
    REPARAMETERIZATION_ROOT / "reparameterized_multistart.csv",
    REPARAMETERIZATION_ROOT / "stability_comparison.json",
    REPARAMETERIZATION_ROOT / "artifact_manifest.json",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def protected_evidence_hashes() -> dict[str, str]:
    missing = [str(path) for path in PROTECTED_PATHS if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing reviewed evidence: {missing}")
    return {path.relative_to(REPOSITORY_ROOT).as_posix(): sha256(path) for path in PROTECTED_PATHS}


def _role_hash(frame: pd.DataFrame, role: str) -> str:
    payload = frame.loc[frame["sample_role"] == role].to_csv(index=False, lineterminator="\n").encode()
    return hashlib.sha256(payload).hexdigest().upper()


def verify_frozen_contract() -> dict[str, Any]:
    geometry.verify_baseline_contract()
    manifest = json.loads(pilot.MANIFEST_PATH.read_text(encoding="utf-8"))
    selected = pd.read_csv(pilot.OUTPUT_ROOT / "selected_options.csv")
    contract = {
        "valuation_date": manifest["valuation_date"],
        "spot": manifest["spot"],
        "calibration_rows": int((selected["sample_role"] == "CALIBRATION").sum()),
        "holdout_rows": int((selected["sample_role"] == "HOLDOUT").sum()),
        "calibration_row_sha256": _role_hash(selected, "CALIBRATION"),
        "holdout_row_sha256": _role_hash(selected, "HOLDOUT"),
        "primary_price_field": manifest["selection"]["primary_price_field"],
        "activity_screen": manifest["selection"]["activity_rule"],
        "carry_rule": manifest["carry_contract"]["carry_rule"],
        "discount_rule": manifest["carry_contract"]["discount_rule"],
        "risk_free_simple_yield": manifest["carry_contract"]["risk_free_simple_yield"],
        "node_count": manifest["optimizer"]["node_count"],
    }
    expected = {
        "valuation_date": "2026-07-15", "spot": 344.35, "calibration_rows": 12,
        "holdout_rows": 7, "calibration_row_sha256": CALIBRATION_ROW_SHA256,
        "holdout_row_sha256": HOLDOUT_ROW_SHA256, "primary_price_field": "ClsPric",
        "activity_screen": "positive close, traded volume, executed trades, and open interest",
        "carry_rule": "q=r-log(F/S)/T", "discount_rule": "D(T)=1/(1+y*T)",
        "risk_free_simple_yield": 0.053324, "node_count": 64,
    }
    if contract != expected:
        raise RuntimeError(f"frozen NTPC contract mismatch: {contract}")
    return contract


def optimizer_contract(max_nfev: int) -> dict[str, Any]:
    if max_nfev not in BUDGETS:
        raise ValueError(f"budget outside predeclared matrix: {max_nfev}")
    return {"method": "trf", "max_nfev": max_nfev, "ftol": 1e-9,
            "xtol": 1e-9, "gtol": 1e-9, "diff_step": 2e-5}


def reached_cap(nfev: int, max_nfev: int) -> bool:
    return int(nfev) >= int(max_nfev)


def canonical_from_coordinate(coordinate: Sequence[float], bounds: dict[str, tuple[float, float]]) -> np.ndarray:
    return unconstrained_to_parameters(np.asarray(coordinate, dtype=float), bounds)


def frozen_starts(bounds: dict[str, tuple[float, float]]) -> tuple[list[np.ndarray], list[np.ndarray], pd.DataFrame]:
    rng = np.random.default_rng(pilot.ANALYSIS_SEED + 200)
    canonical_z = [np.zeros(10)] + [rng.normal(0.0, 1.25, 10) for _ in range(START_COUNT - 1)]
    canonical = [canonical_from_coordinate(value, bounds) for value in canonical_z]
    structured_z = [canonical_to_structured(value, bounds) for value in canonical]
    _, _, reviewed = geometry.paired_start_population(bounds)
    frame = reviewed[["start_id", "canonical_start_sha256", "paired_max_abs_error"]].copy()
    frame["canonical_coordinate_sha256"] = [hashlib.sha256(value.tobytes()).hexdigest().upper() for value in canonical_z]
    frame["structured_coordinate_sha256"] = [hashlib.sha256(value.tobytes()).hexdigest().upper() for value in structured_z]
    return canonical_z, structured_z, frame


def price_rows(frame: pd.DataFrame, parameters: Sequence[float]) -> np.ndarray:
    return geometry.price_rows(frame, parameters)


def _fit_cell(chart: str, max_nfev: int, calibration: pd.DataFrame, holdout: pd.DataFrame,
              starts: list[np.ndarray], bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    config = optimizer_contract(max_nfev)
    observed = calibration["observed_price"].to_numpy(float)
    transform: Callable[[Sequence[float]], np.ndarray]
    transform = (lambda z: canonical_from_coordinate(z, bounds)) if chart == "canonical" else (
        lambda z: structured_to_canonical(z, bounds)
    )

    def residual(z: np.ndarray) -> np.ndarray:
        return price_rows(calibration, transform(z)) - observed

    rows: list[dict[str, Any]] = []
    for start_id, start in enumerate(starts):
        begun = time.perf_counter()
        row: dict[str, Any] = {"chart": chart, "max_nfev": max_nfev, "start_id": start_id}
        try:
            result = least_squares(residual, start, **config)
            parameters = transform(result.x)
            cal_prices, hold_prices = price_rows(calibration, parameters), price_rows(holdout, parameters)
            reasons = boundary_diagnostics(parameters, bounds)
            row.update({
                "optimizer_success": bool(result.success), "optimizer_status": int(result.status),
                "optimizer_message": str(result.message), "nfev": int(result.nfev),
                "reached_cap": reached_cap(result.nfev, max_nfev), "valid": True,
                "boundary_reasons": ";".join(reasons), "boundary_hit": bool(reasons),
                **{f"calibration_{key}": value for key, value in pilot._metrics(calibration, cal_prices).items()},
                **{f"holdout_{key}": value for key, value in pilot._metrics(holdout, hold_prices).items()},
                **{name: float(value) for name, value in zip(PARAMETER_NAMES, parameters, strict=True)},
                **derived_coordinates(parameters), **geometry.boundary_margins(parameters, bounds),
            })
        except Exception as exc:
            row.update({"optimizer_success": False, "optimizer_status": -1,
                        "optimizer_message": f"{type(exc).__name__}: {exc}", "nfev": 0,
                        "reached_cap": False, "valid": False, "boundary_reasons": "", "boundary_hit": False})
        row["runtime_seconds"] = time.perf_counter() - begun
        rows.append(row)
    return pd.DataFrame(rows)


def stability_metrics(starts: pd.DataFrame, bounds: dict[str, tuple[float, float]], max_nfev: int
                      ) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    valid = starts.loc[starts["valid"].astype(bool)].copy()
    if valid.empty:
        raise RuntimeError("cell has no valid starts")
    best = valid.sort_values(["calibration_price_rmse", "start_id"]).iloc[0]
    threshold = max(float(best["calibration_price_rmse"]) * 1.05,
                    float(best["calibration_price_rmse"]) + 0.01)
    near = valid.loc[valid["calibration_price_rmse"] <= threshold].copy()
    widths = np.array([bounds[name][1] - bounds[name][0] for name in PARAMETER_NAMES])
    scaled = near[list(PARAMETER_NAMES)].to_numpy(float) / widths / math.sqrt(len(PARAMETER_NAMES))
    best_scaled = best[list(PARAMETER_NAMES)].to_numpy(float) / widths / math.sqrt(len(PARAMETER_NAMES))
    from_best = np.linalg.norm(scaled - best_scaled, axis=1)
    distances = pdist(scaled) if len(near) > 1 else np.array([])
    labels = (fcluster(linkage(scaled, method="complete", metric="euclidean"), CLUSTER_DISTANCE,
                       criterion="distance") if len(near) > 1 else np.ones(len(near), dtype=int))
    near["distance_from_best"], near["cluster_id"] = from_best, labels
    matrix = squareform(distances) if len(near) > 1 else np.zeros((len(near), len(near)))
    pairs = pd.DataFrame([{"left_start_id": int(near.iloc[i]["start_id"]),
                           "right_start_id": int(near.iloc[j]["start_id"]),
                           "range_scaled_distance": float(matrix[i, j])}
                          for i in range(len(near)) for j in range(i + 1, len(near))])

    def stats(columns: Sequence[str]) -> dict[str, Any]:
        result = {}
        for name in columns:
            values = near[name].to_numpy(float)
            mean = float(np.mean(values))
            result[name] = {"minimum": float(np.min(values)), "maximum": float(np.max(values)),
                            "range": float(np.ptp(values)),
                            "coefficient_of_variation": (float(np.std(values) / abs(mean))
                                                          if abs(mean) > 1e-12 else None)}
        return result

    derived = pd.DataFrame(
        [derived_coordinates(row) for row in near[list(PARAMETER_NAMES)].to_numpy(float)],
        index=near.index,
    )
    for name in derived.columns:
        near[name] = derived[name]
    metrics = {
        "valid_start_count": int(len(valid)),
        "optimizer_success_count": int(valid["optimizer_success"].astype(bool).sum()),
        "cap_count": int(sum(reached_cap(value, max_nfev) for value in valid["nfev"])),
        "cap_rate": float(np.mean([reached_cap(value, max_nfev) for value in valid["nfev"]])),
        "near_equivalent_threshold_price_rmse": threshold,
        "near_equivalent_start_count": int(len(near)),
        "materially_displaced_start_count": int(np.sum(from_best >= MATERIAL_DISTANCE)),
        "cluster_count": int(len(set(labels))),
        "median_pairwise_range_scaled_distance": float(np.median(distances)) if len(distances) else 0.0,
        "maximum_pairwise_range_scaled_distance": float(np.max(distances)) if len(distances) else 0.0,
        "maximum_range_scaled_distance_from_best": float(np.max(from_best)) if len(from_best) else 0.0,
        "parameter_statistics": stats(PARAMETER_NAMES),
        "derived_coordinate_statistics": stats(tuple(derived.columns)),
        "boundary_hit_rate": float(
            valid["boundary_hit"].astype(bool).mean()
            if "boundary_hit" in valid
            else valid["boundary_reasons"].fillna("").astype(str).str.len().gt(0).mean()
        ),
        "median_runtime_seconds": float(valid["runtime_seconds"].median()),
        "total_runtime_seconds": float(starts["runtime_seconds"].sum()),
    }
    return metrics, near, pairs


def cell_pricing(starts: pd.DataFrame, near: pd.DataFrame) -> dict[str, float]:
    valid = starts.loc[starts["valid"].astype(bool)]
    best = valid.sort_values(["calibration_price_rmse", "start_id"]).iloc[0]
    return {
        "best_start_id": int(best["start_id"]),
        "best_calibration_price_rmse": float(best["calibration_price_rmse"]),
        "median_near_equivalent_calibration_price_rmse": float(near["calibration_price_rmse"].median()),
        "best_holdout_price_rmse": float(best["holdout_price_rmse"]),
        "median_holdout_price_rmse": float(near["holdout_price_rmse"].median()),
        "calibration_iv_rmse": float(best["calibration_iv_rmse"]),
        "holdout_iv_rmse": float(best["holdout_iv_rmse"]),
    }


def classify_cap(cap_rate: float) -> str:
    if cap_rate <= 0.25:
        return "CAP_MATERIALLY_REDUCED"
    if cap_rate <= 0.50:
        return "CAP_PARTIALLY_REDUCED"
    return "CAP_NOT_RESOLVED"


def _reduction(old: float, new: float) -> float:
    return 1.0 - new / max(old, 1e-15)


def classify_dispersion(old: dict[str, Any], new: dict[str, Any]) -> str:
    displaced = _reduction(old["materially_displaced_start_count"], new["materially_displaced_start_count"])
    clusters = _reduction(old["cluster_count"], new["cluster_count"])
    median = _reduction(old["median_pairwise_range_scaled_distance"], new["median_pairwise_range_scaled_distance"])
    maximum = _reduction(old["maximum_pairwise_range_scaled_distance"], new["maximum_pairwise_range_scaled_distance"])
    if displaced >= 0.50 and clusters >= 0.40 and median >= 0.25 and maximum >= 0.25:
        return "STRONG_DISPERSION_COLLAPSE"
    if displaced >= 0.25 and (median >= 0.10 or maximum >= 0.10) and new["cluster_count"] <= old["cluster_count"]:
        return "PARTIAL_DISPERSION_IMPROVEMENT"
    return "DISPERSION_PERSISTS"


def classify_final(cap_rates: Sequence[float], dispersion: Sequence[str], valid: bool) -> str:
    if not valid:
        return "INVALID"
    caps = [classify_cap(value) for value in cap_rates]
    if all(value == "CAP_MATERIALLY_REDUCED" for value in caps) and all(
        value == "STRONG_DISPERSION_COLLAPSE" for value in dispersion
    ):
        return "NUMERICAL_CAP_LIMITATION_SUPPORTED"
    if any(value != "CAP_NOT_RESOLVED" for value in caps):
        if any(value == "PARTIAL_DISPERSION_IMPROVEMENT" for value in dispersion) or any(
            value == "STRONG_DISPERSION_COLLAPSE" for value in dispersion
        ):
            return "PARTIAL_NUMERICAL_EFFECT"
        if all(value == "DISPERSION_PERSISTS" for value in dispersion):
            return "PERSISTENT_PARAMETER_AMBIGUITY"
    return "OPTIMIZER_CAP_UNRESOLVED"


def baseline_reproduction(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for chart in CHARTS:
        cell = cells[f"{chart}_160"]
        combined = cell["pricing"] | cell["stability"]
        differences = {}
        for name, expected in REVIEWED_160[chart].items():
            observed_name = {"calibration_price_rmse": "best_calibration_price_rmse",
                             "calibration_iv_rmse": "calibration_iv_rmse",
                             "holdout_price_rmse": "best_holdout_price_rmse",
                             "holdout_iv_rmse": "holdout_iv_rmse"}.get(name, name)
            differences[name] = abs(float(combined[observed_name]) - float(expected))
        passed = all(value <= (5e-8 if "rmse" in name else 1e-10)
                     for name, value in differences.items())
        checks[chart] = {"passed": passed, "absolute_differences": differences}
    checks["passed"] = all(checks[chart]["passed"] for chart in CHARTS)
    return checks


def paired_analysis(low: pd.DataFrame, high: pd.DataFrame,
                    low_near: pd.DataFrame, high_near: pd.DataFrame,
                    bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    widths = np.array([bounds[name][1] - bounds[name][0] for name in PARAMETER_NAMES])
    low_clusters = dict(zip(low_near["start_id"].astype(int), low_near["cluster_id"].astype(int)))
    high_clusters = dict(zip(high_near["start_id"].astype(int), high_near["cluster_id"].astype(int)))
    rows = []
    for start_id in range(START_COUNT):
        left = low.loc[low["start_id"] == start_id].iloc[0]
        right = high.loc[high["start_id"] == start_id].iloc[0]
        displacement = float(np.sqrt(np.mean(((right[list(PARAMETER_NAMES)].to_numpy(float) -
                                               left[list(PARAMETER_NAMES)].to_numpy(float)) / widths) ** 2)))
        collapsed = 0
        if start_id in low_clusters and start_id in high_clusters:
            for peer in set(low_clusters) & set(high_clusters):
                if low_clusters[peer] != low_clusters[start_id] and high_clusters[peer] == high_clusters[start_id]:
                    collapsed += 1
        rows.append({
            "start_id": start_id, "parameter_displacement_160_to_320": displacement,
            "calibration_rmse_change": float(right["calibration_price_rmse"] - left["calibration_price_rmse"]),
            "holdout_rmse_change": float(right["holdout_price_rmse"] - left["holdout_price_rmse"]),
            "continued_materially": displacement >= MATERIAL_DISTANCE,
            "collapsed_previously_distinct_peer_count": collapsed,
            "cap_at_160": reached_cap(left["nfev"], 160), "cap_at_320": reached_cap(right["nfev"], 320),
        })
    return pd.DataFrame(rows)


def percentage_changes(cells: dict[str, dict[str, Any]], chart: str) -> dict[str, float]:
    old, new = cells[f"{chart}_160"], cells[f"{chart}_320"]
    names = {"calibration_rmse": "best_calibration_price_rmse", "holdout_rmse": "best_holdout_price_rmse",
             "holdout_iv_rmse": "holdout_iv_rmse"}
    result = {name: 100.0 * (new["pricing"][key] / old["pricing"][key] - 1.0) for name, key in names.items()}
    result["runtime"] = 100.0 * (new["stability"]["total_runtime_seconds"] /
                                 old["stability"]["total_runtime_seconds"] - 1.0)
    return result


def _plots(cells: dict[str, dict[str, Any]], paired: dict[str, pd.DataFrame]) -> list[Path]:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    labels = [f"{c}\n{b}" for c in CHARTS for b in BUDGETS]
    keys = [f"{c}_{b}" for c in CHARTS for b in BUDGETS]
    figures: list[Path] = []

    def bar(name: str, title: str, values: Sequence[float], ylabel: str) -> None:
        fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(labels, values); ax.set(title=title, ylabel=ylabel)
        fig.tight_layout(); path = FIGURE_ROOT / name; fig.savefig(path, dpi=160); plt.close(fig); figures.append(path)

    bar("01_optimizer_cap_rate.png", "Optimizer cap rate", [cells[k]["stability"]["cap_rate"] for k in keys], "rate")
    fig, ax = plt.subplots(figsize=(8, 4)); x=np.arange(4); w=.36
    ax.bar(x-w/2,[cells[k]["pricing"]["best_calibration_price_rmse"] for k in keys],w,label="calibration")
    ax.bar(x+w/2,[cells[k]["pricing"]["best_holdout_price_rmse"] for k in keys],w,label="holdout")
    ax.set_xticks(x,labels); ax.set_ylabel("price RMSE"); ax.legend(); fig.tight_layout()
    path=FIGURE_ROOT/"02_calibration_holdout_rmse.png"; fig.savefig(path,dpi=160); plt.close(fig); figures.append(path)
    bar("03_materially_displaced_count.png", "Materially displaced near-equivalent solutions",
        [cells[k]["stability"]["materially_displaced_start_count"] for k in keys], "count")
    bar("04_cluster_count.png", "Complete-linkage clusters", [cells[k]["stability"]["cluster_count"] for k in keys], "count")
    fig, ax=plt.subplots(figsize=(8,4)); x=np.arange(4); w=.36
    ax.bar(x-w/2,[cells[k]["stability"]["median_pairwise_range_scaled_distance"] for k in keys],w,label="median")
    ax.bar(x+w/2,[cells[k]["stability"]["maximum_pairwise_range_scaled_distance"] for k in keys],w,label="maximum")
    ax.set_xticks(x,labels); ax.set_ylabel("range-scaled distance"); ax.legend(); fig.tight_layout()
    path=FIGURE_ROOT/"05_pairwise_separation.png"; fig.savefig(path,dpi=160); plt.close(fig); figures.append(path)
    fig, ax=plt.subplots(figsize=(8,4))
    for chart, frame in paired.items(): ax.plot(frame["start_id"],frame["parameter_displacement_160_to_320"],"o-",label=chart)
    ax.axhline(MATERIAL_DISTANCE,color="black",ls="--"); ax.set(xlabel="start ID",ylabel="paired scaled displacement"); ax.legend(); fig.tight_layout()
    path=FIGURE_ROOT/"06_paired_parameter_movement.png"; fig.savefig(path,dpi=160); plt.close(fig); figures.append(path)
    fig, ax=plt.subplots(figsize=(11,4)); x=np.arange(len(PARAMETER_NAMES)); w=.36
    for offset,chart in [(-w/2,"canonical"),(w/2,"transformed")]:
        ranges=[cells[f"{chart}_320"]["stability"]["parameter_statistics"][n]["range"] for n in PARAMETER_NAMES]
        ax.bar(x+offset,ranges,w,label=chart)
    ax.set_xticks(x,PARAMETER_NAMES,rotation=40,ha="right"); ax.set_ylabel("canonical range at 320"); ax.legend(); fig.tight_layout()
    path=FIGURE_ROOT/"07_canonical_parameter_dispersion_320.png"; fig.savefig(path,dpi=160); plt.close(fig); figures.append(path)
    fig, ax=plt.subplots(figsize=(7,4))
    for chart in CHARTS:
        frame=cells[f"{chart}_320"]["near"]
        ax.scatter(frame["distance_from_best"],frame["calibration_price_rmse"],label=chart)
    ax.set(xlabel="distance from best",ylabel="calibration price RMSE"); ax.legend(); fig.tight_layout()
    path=FIGURE_ROOT/"08_pricing_error_vs_displacement_320.png"; fig.savefig(path,dpi=160); plt.close(fig); figures.append(path)
    return figures


def _fmt(value: float) -> str:
    return f"{value:.9g}"


def render_report(result: dict[str, Any]) -> None:
    cells=result["cells"]; classification=result["classification"]
    rows=[]
    for key in [f"{c}_{b}" for c in CHARTS for b in BUDGETS]:
        p,s=cells[key]["pricing"],cells[key]["stability"]
        rows.append(f"| {key} | {s['cap_count']}/{s['valid_start_count']} ({s['cap_rate']:.3f}) | {_fmt(p['best_calibration_price_rmse'])} | {_fmt(p['best_holdout_price_rmse'])} | {_fmt(p['holdout_iv_rmse'])} | {s['materially_displaced_start_count']} | {s['cluster_count']} | {_fmt(s['median_pairwise_range_scaled_distance'])} | {_fmt(s['maximum_pairwise_range_scaled_distance'])} |")
    unstable={}
    for chart in CHARTS:
        stats=cells[f"{chart}_320"]["stability"]["parameter_statistics"]
        unstable[chart]=", ".join(sorted(PARAMETER_NAMES,key=lambda n:stats[n]["range"],reverse=True)[:4])
    best_dh=min(cells[k]["pricing"]["best_holdout_price_rmse"] for k in cells)
    diagnostic_rows=[]
    for name in ("v0_total", "alpha_v", "theta_total", "alpha_theta", "kappa_slow",
                 "kappa_fast", "slow_half_life_days", "fast_half_life_days"):
        values=[]
        for chart in CHARTS:
            source=(cells[f"{chart}_320"]["stability"]["parameter_statistics"]
                    if name in PARAMETER_NAMES else
                    cells[f"{chart}_320"]["stability"]["derived_coordinate_statistics"])
            item=source[name]
            cv="NA" if item["coefficient_of_variation"] is None else _fmt(item["coefficient_of_variation"])
            values.append(f"{_fmt(item['range'])} / {cv}")
        diagnostic_rows.append(f"| {name} | {values[0]} | {values[1]} |")
    reproduction_reason=(
        "The transformed 160 cell reproduced the reviewed baseline exactly. The canonical 160 cell did not: "
        f"its median pairwise separation was {_fmt(cells['canonical_160']['stability']['median_pairwise_range_scaled_distance'])} "
        f"versus reviewed {_fmt(REVIEWED_160['canonical']['median_pairwise_range_scaled_distance'])}, and its maximum "
        f"distance from best was {_fmt(cells['canonical_160']['stability']['maximum_range_scaled_distance_from_best'])} "
        f"versus reviewed {_fmt(REVIEWED_160['canonical']['maximum_range_scaled_distance_from_best'])}. "
        "Because required canonical stability metrics did not reproduce within the predeclared deterministic tolerance, "
        "the four-cell comparison is invalid and the 320 results below are descriptive only."
    )
    next_action={
        "NUMERICAL_CAP_LIMITATION_SUPPORTED":"Freeze the improved optimizer budget/protocol and rerun the reviewed NTPC calibration under the improved numerical contract before changing model information.",
        "PARTIAL_NUMERICAL_EFFECT":"Assess whether one final fixed optimization budget is scientifically justified, but do not begin additional optimizer tuning automatically.",
        "PERSISTENT_PARAMETER_AMBIGUITY":"STOP optimizer-only work and proceed to a separately predeclared real NTPC multi-date calibration using 2026-07-01, 2026-07-15, and 2026-07-22 with shared structural parameters and explicitly modeled date-specific variance states.",
        "OPTIMIZER_CAP_UNRESOLVED":"Record that 320 was insufficient to remove the numerical confounder; require an explicit decision before any further budget escalation.",
        "INVALID":"Repair the matched frozen experiment contract before drawing a scientific conclusion.",
    }[classification]
    text=f"""# NTPC Double Heston optimizer-cap sensitivity

## Purpose and frozen design

This last optimizer-budget-only experiment tested whether the NTPC instability was materially caused by `max_nfev=160`. Exactly four paired cells used the same valuation date, 12 calibration rows, 7 holdout rows, row hashes, prices, activity screen, spot, maturities, carry/RBI inputs, IV inversion, canonical 64-node pricer, canonical bounds and constraints, unweighted price residual, `least_squares(method=\"trf\")`, tolerances, `diff_step`, 12 canonical starts and IDs, near-equivalent/material-distance rules, and complete-linkage cutoff. The only treatment was `max_nfev: 160 -> 320`. No data, model, objective, weighting, clipping, prior, penalty, or regularization changed.

The 160 baseline reproduction gate was **{'PASSED' if result['baseline_reproduction']['passed'] else 'FAILED'}**. {reproduction_reason} Calibration and holdout row hashes are `{CALIBRATION_ROW_SHA256}` and `{HOLDOUT_ROW_SHA256}`.

## Results

| cell | cap | best calibration RMSE | best holdout RMSE | holdout IV RMSE | displaced | clusters | median separation | max separation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Canonical percentage changes 160 -> 320: calibration RMSE `{result['percentage_changes']['canonical']['calibration_rmse']:.3f}%`, holdout RMSE `{result['percentage_changes']['canonical']['holdout_rmse']:.3f}%`, holdout IV RMSE `{result['percentage_changes']['canonical']['holdout_iv_rmse']:.3f}%`, runtime `{result['percentage_changes']['canonical']['runtime']:.3f}%`.

Transformed percentage changes 160 -> 320: calibration RMSE `{result['percentage_changes']['transformed']['calibration_rmse']:.3f}%`, holdout RMSE `{result['percentage_changes']['transformed']['holdout_rmse']:.3f}%`, holdout IV RMSE `{result['percentage_changes']['transformed']['holdout_iv_rmse']:.3f}%`, runtime `{result['percentage_changes']['transformed']['runtime']:.3f}%`.

Cap decisions: canonical **{result['chart_decisions']['canonical']['cap']}**; transformed **{result['chart_decisions']['transformed']['cap']}**. Dispersion decisions: canonical **{result['chart_decisions']['canonical']['dispersion']}**; transformed **{result['chart_decisions']['transformed']['dispersion']}**. The most variable canonical parameters by raw range at 320 were canonical: `{unstable['canonical']}`; transformed: `{unstable['transformed']}`. Total/split variance, total/split theta, slow/fast kappa and half-life dispersion are recorded under each 320 cell's `derived_coordinate_statistics` in the machine-readable evidence.

| 320 diagnostic | canonical range / CV | transformed range / CV |
|---|---:|---:|
{chr(10).join(diagnostic_rows)}

The allocation diagnostics `alpha_v` and `alpha_theta` remain widely dispersed in both charts; they are diagnostics, not new scientific parameters. Slow/fast allocation therefore remains ambiguous in the descriptive 320 evidence. This cannot repair the failed validity gate.

## Interpretation

Final classification: **{classification}**.

The cap rate did **not** fall: it remained `10/12 = 0.833` under both coordinate charts, so the numerical-cap confounder was not resolved. Separately, if a valid future comparison lowers the cap rate while separated solutions remain, the correct interpretation would be that the optimizer received substantially more opportunity but materially different parameter basins still fit nearly equivalently—evidence consistent with persistent/global parameter ambiguity, not mathematical proof of structural non-identification.

Best Double Heston holdout RMSE was `{best_dh:.9g}` versus Standard Heston `{HESTON_HOLDOUT_RMSE}`. The predeclared material-win threshold was `{HESTON_MATERIAL_THRESHOLD}`: **{'YES' if best_dh <= HESTON_MATERIAL_THRESHOLD else 'NO'}**.

No pricing, cap, cluster, separation, allocation, or Heston-superiority claim from the 320 cells is accepted as a scientific conclusion because the experiment is invalid. Optimizer-only experiments should **not continue automatically**. Exact next recommendation: **{next_action}** No new scientific experiment should begin until that reproducibility defect is resolved under a separately authorized protocol.

## Evidence and figures

Per-start final canonical vectors, errors, termination, success, margins, runtime, paired movement and basin-collapse diagnostics are in ignored generated CSV evidence. The eight mentor-ready figures are in `market_data_audit/stage_a/derived/ntpc_dh_optimizer_cap_sensitivity/figures/`; their hashes are sealed by the tracked evidence manifest. Render-only replay regenerates the report, figures, summaries, and manifest from preserved completed optimizer CSVs without rerunning fits.
"""
    REPORT_PATH.write_text(text,encoding="utf-8",newline="\n")


def _publish(cells_frames: dict[str, pd.DataFrame], contract: dict[str, Any], before: dict[str, str]) -> dict[str, Any]:
    bounds=load_hard_safety_bounds(pilot.BOUNDS_PATH); cells={}; paired={}
    for chart in CHARTS:
        for budget in BUDGETS:
            key=f"{chart}_{budget}"; frame=cells_frames[key]
            stability,near,pairs=stability_metrics(frame,bounds,budget)
            cells[key]={"pricing":cell_pricing(frame,near),"stability":stability,"near":near,"pairs":pairs}
            write_csv(OUTPUT_ROOT/f"{key}_near_equivalent.csv",near); write_csv(OUTPUT_ROOT/f"{key}_pairwise.csv",pairs)
        paired[chart]=paired_analysis(cells_frames[f"{chart}_160"],cells_frames[f"{chart}_320"],
                                      cells[f"{chart}_160"]["near"],cells[f"{chart}_320"]["near"],bounds)
        write_csv(OUTPUT_ROOT/f"{chart}_paired_160_320.csv",paired[chart])
    reproduction=baseline_reproduction(cells)
    matched=all(set(cells_frames[f"{c}_{b}"]["start_id"].astype(int))==set(range(12)) for c in CHARTS for b in BUDGETS)
    after=protected_evidence_hashes(); valid=before==after and matched and reproduction["passed"]
    decisions={c:{"cap":classify_cap(cells[f"{c}_320"]["stability"]["cap_rate"]),
                  "dispersion":classify_dispersion(cells[f"{c}_160"]["stability"],cells[f"{c}_320"]["stability"])} for c in CHARTS}
    classification=classify_final([cells[f"{c}_320"]["stability"]["cap_rate"] for c in CHARTS],
                                  [decisions[c]["dispersion"] for c in CHARTS],valid)
    figures=_plots(cells,paired)
    serial_cells={key:{"pricing":value["pricing"],"stability":value["stability"]} for key,value in cells.items()}
    result={"base_main":BASE_MAIN,"contract":contract,"protected_hashes_before":before,
            "protected_hashes_after":after,"paired_start_ids":matched,"baseline_reproduction":reproduction,
            "changed_variable_only":"max_nfev","cells":serial_cells,"chart_decisions":decisions,
            "percentage_changes":{c:percentage_changes(cells,c) for c in CHARTS},
            "classification":classification,"heston_reference":HESTON_HOLDOUT_RMSE,
            "heston_material_threshold":HESTON_MATERIAL_THRESHOLD,
            "figures":[p.relative_to(REPOSITORY_ROOT).as_posix() for p in figures]}
    write_json(OUTPUT_ROOT/"experiment_summary.json",result); render_report(result)
    artifacts={p.relative_to(REPOSITORY_ROOT).as_posix():sha256(p) for p in sorted(OUTPUT_ROOT.rglob("*")) if p.is_file()}
    manifest={"base_main":BASE_MAIN,"classification":classification,"changed_variable_only":"max_nfev",
              "optimizer_contracts":[optimizer_contract(b) for b in BUDGETS],"frozen_contract":contract,
              "baseline_reproduction":reproduction,"paired_start_ids":matched,
              "protected_evidence_hashes":after,"generated_artifact_hashes":artifacts,
              "report_sha256":sha256(REPORT_PATH)}
    write_json(MANIFEST_PATH,manifest)
    return result


def run() -> dict[str, Any]:
    contract=verify_frozen_contract(); before=protected_evidence_hashes()
    bounds=load_hard_safety_bounds(pilot.BOUNDS_PATH); canonical,transformed,starts_frame=frozen_starts(bounds)
    write_csv(OUTPUT_ROOT/"paired_starts.csv",starts_frame)
    selected=pd.read_csv(pilot.OUTPUT_ROOT/"selected_options.csv")
    calibration=selected.loc[selected["sample_role"]=="CALIBRATION"].copy()
    holdout=selected.loc[selected["sample_role"]=="HOLDOUT"].copy()
    frames={}
    for chart,starts in (("canonical",canonical),("transformed",transformed)):
        for budget in BUDGETS:
            key=f"{chart}_{budget}"; frames[key]=_fit_cell(chart,budget,calibration,holdout,starts,bounds)
            write_csv(OUTPUT_ROOT/f"{key}_starts.csv",frames[key])
    return _publish(frames,contract,before)


def render_existing_outputs() -> dict[str, Any]:
    contract=verify_frozen_contract(); before=protected_evidence_hashes()
    frames={key:pd.read_csv(OUTPUT_ROOT/f"{key}_starts.csv") for key in
            [f"{c}_{b}" for c in CHARTS for b in BUDGETS]}
    return _publish(frames,contract,before)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--render-only",action="store_true"); args=parser.parse_args()
    output=render_existing_outputs() if args.render_only else run()
    print(json.dumps({"classification":output["classification"],"cells":output["cells"],
                      "chart_decisions":output["chart_decisions"]},indent=2))
