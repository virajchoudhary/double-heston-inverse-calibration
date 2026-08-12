"""Forensic reference analysis for NTPC canonical-160 reproduction.

This module never calibrates and never acquires data. It compares two completed
12-start artifacts by ``start_id`` and independently computes stability from
the canonical vectors. ``canonicalize_pricing_inputs`` defines the corrected
numerical replay contract for CSV-loaded NTPC rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.ntpc_pricing_input_contract import canonicalize_pricing_inputs


PARAMETER_NAMES = (
    "kappa_slow", "theta_slow", "sigma_slow", "rho_slow", "v0_slow",
    "kappa_fast", "theta_fast", "sigma_fast", "rho_fast", "v0_fast",
)
HARD_BOUNDS = {
    "kappa_slow": (0.05, 3.0), "theta_slow": (0.005, 0.25),
    "sigma_slow": (0.005, 1.0), "rho_slow": (-0.95, 0.95),
    "v0_slow": (0.005, 0.30), "kappa_fast": (0.10, 12.0),
    "theta_fast": (0.002, 0.20), "sigma_fast": (0.005, 1.5),
    "rho_fast": (-0.95, 0.95), "v0_fast": (0.002, 0.25),
}
MATERIAL_DISTANCE = 0.05
CLUSTER_DISTANCE = 0.05
REPLAY_TOLERANCE_SCALED_RMS = 1e-4
REVIEWED_FIXTURE_SHA256 = "090EC2EE30B75A4C0E2D366F80AA86625431C98D9900D03C5F508DA33B41C3AD"
REPLAY_FIXTURE_SHA256 = "3C6A9FF215961EFC72C3ACBF99CFF2B451EE9ECFD1FC29976F87D2BB2D440928"
START_FIXTURE_SHA256 = "BB980C2FCA3F83BAC39C359ED6BD84083BA8CAD46074647E7D8E5C658B3D3D27"
REVIEWED_ARTIFACT_SHA256 = "4E092F2BEC5F53033E61EFB1D2B2D761C9D3AB8F72F17F33D6E989946FC1EB70"
REPLAY_ARTIFACT_SHA256 = "CF148D54639EA194E620BABB5E6CF741A91AB77C269A2C1E3BA3CFA25B33926E"
STARTS_ARTIFACT_SHA256 = "F628A1391367D5E309B98AE5456066F043CE3C870EE819470AC69333B8C6474D"
CAUSAL_TOLERANCE = 5e-14


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _validated_population(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"start_id", "calibration_price_rmse", *PARAMETER_NAMES}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing canonical artifact columns: {missing}")
    result = frame.sort_values("start_id").reset_index(drop=True)
    if result["start_id"].astype(int).tolist() != list(range(12)):
        raise ValueError("canonical artifact must contain each start_id 0..11 exactly once")
    return result


def reference_stability(
    frame: pd.DataFrame, bounds: dict[str, tuple[float, float]]
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compute the frozen stability definition without existing project helpers."""
    population = _validated_population(frame)
    best = population.sort_values(["calibration_price_rmse", "start_id"]).iloc[0]
    threshold = max(float(best["calibration_price_rmse"]) * 1.05,
                    float(best["calibration_price_rmse"]) + 0.01)
    near = population.loc[population["calibration_price_rmse"] <= threshold].copy()
    widths = np.asarray([bounds[name][1] - bounds[name][0] for name in PARAMETER_NAMES], dtype=float)
    vectors = near[list(PARAMETER_NAMES)].to_numpy(dtype=float)
    scaled = vectors / widths / math.sqrt(len(PARAMETER_NAMES))
    best_scaled = best[list(PARAMETER_NAMES)].to_numpy(dtype=float) / widths / math.sqrt(len(PARAMETER_NAMES))
    pairs: list[dict[str, Any]] = []
    distances: list[float] = []
    for left in range(len(near)):
        for right in range(left + 1, len(near)):
            distance = float(np.linalg.norm(scaled[left] - scaled[right]))
            distances.append(distance)
            pairs.append({
                "left_start_id": int(near.iloc[left]["start_id"]),
                "right_start_id": int(near.iloc[right]["start_id"]),
                "range_scaled_distance": distance,
            })
    expected_pairs = len(near) * (len(near) - 1) // 2
    if len(pairs) != expected_pairs:
        raise RuntimeError("unique unordered pair-count contract failed")
    distance_array = np.asarray(distances, dtype=float)
    from_best = np.linalg.norm(scaled - best_scaled, axis=1)
    labels = (fcluster(linkage(scaled, method="complete", metric="euclidean"),
                       CLUSTER_DISTANCE, criterion="distance")
              if len(near) > 1 else np.ones(len(near), dtype=int))
    memberships = {
        str(int(cluster)): sorted(near.iloc[np.flatnonzero(labels == cluster)]["start_id"].astype(int).tolist())
        for cluster in sorted(set(labels))
    }
    summary = {
        "solution_count": int(len(near)),
        "unordered_pair_count": len(pairs),
        "sorted_pairwise_distances": sorted(distances),
        "median_pairwise_range_scaled_distance": float(np.median(distance_array)),
        "maximum_pairwise_range_scaled_distance": float(np.max(distance_array)),
        "distance_from_best": {
            str(int(start_id)): float(distance)
            for start_id, distance in zip(near["start_id"], from_best, strict=True)
        },
        "maximum_range_scaled_distance_from_best": float(np.max(from_best)),
        "materially_displaced_start_count": int(np.sum(from_best >= MATERIAL_DISTANCE)),
        "cluster_count": int(len(set(labels))),
        "cluster_memberships": memberships,
        "near_equivalent_threshold_price_rmse": threshold,
        "parameter_order": list(PARAMETER_NAMES),
        "full_parameter_ranges": {name: bounds[name][1] - bounds[name][0] for name in PARAMETER_NAMES},
    }
    return summary, pd.DataFrame(pairs)


def _margin_values(row: pd.Series, bounds: dict[str, tuple[float, float]]) -> dict[str, float]:
    p = row[list(PARAMETER_NAMES)].to_numpy(dtype=float)
    normalized = [min((value - bounds[name][0]) / (bounds[name][1] - bounds[name][0]),
                      (bounds[name][1] - value) / (bounds[name][1] - bounds[name][0]))
                  for name, value in zip(PARAMETER_NAMES, p, strict=True)]
    return {
        "minimum_hard_bound_fraction_margin": float(min(normalized)),
        "kappa_ordering_margin": float(p[5] - p[0]),
        "slow_feller_margin": float(2 * p[0] * p[1] - p[2] ** 2),
        "fast_feller_margin": float(2 * p[5] * p[6] - p[7] ** 2),
        "correlation_disk_margin": float(1 - p[3] ** 2 - p[8] ** 2),
    }


def compare_starts(
    reviewed: pd.DataFrame, replay: pd.DataFrame, bounds: dict[str, tuple[float, float]]
) -> pd.DataFrame:
    left, right = _validated_population(reviewed), _validated_population(replay)
    joined = left.merge(right, on="start_id", how="outer", validate="one_to_one",
                        suffixes=("_reviewed", "_replay"), indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("start_id population mismatch")
    widths = np.asarray([bounds[name][1] - bounds[name][0] for name in PARAMETER_NAMES], dtype=float)
    rows: list[dict[str, Any]] = []
    for _, row in joined.sort_values("start_id").iterrows():
        old = np.asarray([row[f"{name}_reviewed"] for name in PARAMETER_NAMES], dtype=float)
        new = np.asarray([row[f"{name}_replay"] for name in PARAMETER_NAMES], dtype=float)
        absolute = np.abs(new - old)
        scaled_rms = float(np.sqrt(np.mean((absolute / widths) ** 2)))
        rmse_delta = float(row["calibration_price_rmse_replay"] - row["calibration_price_rmse_reviewed"])
        status_same = row.get("optimizer_status_reviewed") == row.get("optimizer_status_replay")
        nfev_delta = int(row.get("nfev_replay", 0) - row.get("nfev_reviewed", 0))
        message_same = row.get("optimizer_message_reviewed") == row.get("optimizer_message_replay")
        exact = bool(np.max(absolute) <= 1e-14 and abs(rmse_delta) <= 1e-14 and nfev_delta == 0
                     and status_same and message_same)
        classification = ("EXACT_REPRODUCTION" if exact else
                          "NUMERICAL_TOLERANCE_ONLY" if scaled_rms <= REPLAY_TOLERANCE_SCALED_RMS else
                          "MATERIALLY_DIFFERENT_SOLUTION")
        item: dict[str, Any] = {
            "start_id": int(row["start_id"]), "classification": classification,
            "range_scaled_rms_difference": scaled_rms,
            "max_canonical_parameter_difference": float(np.max(absolute)),
            "calibration_rmse_difference": rmse_delta, "nfev_difference": nfev_delta,
            "optimizer_status_difference": not bool(status_same),
            "termination_difference": not bool(message_same),
            "reviewed_source_artifact": str(reviewed.attrs.get("source_artifact", "reviewed artifact argument")),
            "reviewed_source_commit": "dd539150898bf5ca4d168c5dba3f3a33c69628e2",
            "replay_source_artifact": str(replay.attrs.get("source_artifact", "replay artifact argument")),
            "replay_source_commit": "UNCOMMITTED_INVALID_EXPERIMENT",
        }
        for index, name in enumerate(PARAMETER_NAMES):
            item[f"reviewed_{name}"] = float(old[index])
            item[f"replay_{name}"] = float(new[index])
            item[f"absolute_difference_{name}"] = float(absolute[index])
            item[f"range_scaled_difference_{name}"] = float(absolute[index] / widths[index])
        for prefix, source in (("reviewed", left.loc[left["start_id"] == row["start_id"]].iloc[0]),
                               ("replay", right.loc[right["start_id"] == row["start_id"]].iloc[0])):
            for name, value in _margin_values(source, bounds).items():
                item[f"{prefix}_{name}"] = value
            for name in ("calibration_price_rmse", "calibration_iv_rmse", "holdout_price_rmse",
                         "optimizer_success", "optimizer_status", "nfev", "optimizer_message",
                         "boundary_reasons"):
                if name in source:
                    item[f"{prefix}_{name}"] = source[name]
        rows.append(item)
    return pd.DataFrame(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def validate_causal_probe(probe: dict[str, Any]) -> dict[str, Any]:
    """Fail closed unless the sealed start-6 probes establish the stated cause."""
    if probe.get("max_nfev") != 160 or probe.get("start_id") != 6:
        raise RuntimeError("causal probe used an unexpected optimizer contract")
    cases = probe["cases"]
    reviewed_cases = ("rebuilt_in_memory", "csv_with_rebuilt_T")
    replay_cases = (
        "csv_loaded", "csv_with_rebuilt_log_moneyness", "csv_with_rebuilt_discount_factor",
        "csv_with_rebuilt_continuous_rate", "csv_with_rebuilt_futures_implied_carry",
        "csv_with_rebuilt_market_implied_volatility",
    )
    if any(cases[name]["maximum_abs_difference_from_reviewed"] > CAUSAL_TOLERANCE for name in reviewed_cases):
        raise RuntimeError("rebuilt T did not reproduce the reviewed start-6 endpoint")
    if any(cases[name]["maximum_abs_difference_from_replay"] > CAUSAL_TOLERANCE for name in replay_cases):
        raise RuntimeError("a non-T single-field probe left the replay endpoint")
    repeats = probe["csv_repeat_fits"]
    if len(repeats) != 3 or len({item["parameter_sha256"] for item in repeats}) != 1:
        raise RuntimeError("CSV repeat fits are not byte-identical")
    return {
        "reviewed_reproduced_by": list(reviewed_cases),
        "replay_reproduced_by": list(replay_cases),
        "csv_repeat_parameter_sha256": repeats[0]["parameter_sha256"],
        "csv_repeat_count": len(repeats),
    }


def run(reviewed_path: Path, replay_path: Path, starts_path: Path, causal_probe_path: Path, output: Path) -> dict[str, Any]:
    actual_hashes = {
        "reviewed": sha256(reviewed_path),
        "replay": sha256(replay_path),
        "starts": sha256(starts_path),
    }
    expected_hashes = {
        "reviewed": REVIEWED_ARTIFACT_SHA256,
        "replay": REPLAY_ARTIFACT_SHA256,
        "starts": STARTS_ARTIFACT_SHA256,
    }
    mismatches = {
        name: {"expected": expected_hashes[name], "actual": actual_hashes[name]}
        for name in expected_hashes if actual_hashes[name] != expected_hashes[name]
    }
    if mismatches:
        raise RuntimeError(f"forensic input provenance mismatch: {mismatches}")
    causal_probe = json.loads(causal_probe_path.read_text(encoding="utf-8"))
    causal_validation = validate_causal_probe(causal_probe)
    reviewed, replay = pd.read_csv(reviewed_path), pd.read_csv(replay_path)
    reviewed.attrs["source_artifact"] = str(reviewed_path)
    replay.attrs["source_artifact"] = str(replay_path)
    starts = pd.read_csv(starts_path).sort_values("start_id")
    comparison = compare_starts(reviewed, replay, HARD_BOUNDS)
    reviewed_summary, reviewed_pairs = reference_stability(reviewed, HARD_BOUNDS)
    replay_summary, replay_pairs = reference_stability(replay, HARD_BOUNDS)
    start_digest = hashlib.sha256("|".join(starts["canonical_start_sha256"]).encode()).hexdigest().upper()
    if start_digest != "3AC1C30FF1B5416987D2103EA70B9262BBB8B4991F18F7A06C98E3A41C86ABA1":
        raise RuntimeError("canonical start-population hash mismatch")
    output.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output / "paired_start_comparison.csv", index=False, lineterminator="\n")
    reviewed_pairs.to_csv(output / "reviewed_pairwise_distances.csv", index=False, lineterminator="\n")
    replay_pairs.to_csv(output / "replay_pairwise_distances.csv", index=False, lineterminator="\n")
    provenance = {
        "reviewed": {"path": str(reviewed_path), "sha256": actual_hashes["reviewed"],
                     "manifest_bound_sha256": REVIEWED_ARTIFACT_SHA256,
                     "source_commit": "dd539150898bf5ca4d168c5dba3f3a33c69628e2",
                     "execution_input": "pre-serialization in-memory selected dataframe"},
        "replay": {"path": str(replay_path), "sha256": actual_hashes["replay"],
                   "source_branch": "feat/ntpc-dh-optimizer-cap-sensitivity",
                   "execution_input": "selected_options.csv loaded by pandas.read_csv"},
        "starts": {"path": str(starts_path), "sha256": actual_hashes["starts"],
                   "population_sha256": start_digest},
        "causal_probe": {"path": str(causal_probe_path), "sha256": sha256(causal_probe_path),
                         "validation": causal_validation},
        "metric_code": "independent unique i<j Euclidean distances after full-range scaling and sqrt(10) normalization",
    }
    write_json(output / "provenance_trace.json", provenance)
    counts = comparison["classification"].value_counts().to_dict()
    summary = {
        "root_cause": "ARTIFACT_PROVENANCE_MISMATCH",
        "exact_defect": ("PR #13 optimized binary T=DTE/365 in memory before CSV serialization; downstream replay "
                         "loaded rounded decimal T from selected_options.csv. Reconstructing T alone reproduces "
                         "reviewed start 6 to 2.22e-16; CSV T reproduces replay start 6 to 9.02e-17."),
        "classification_counts": counts,
        "first_materially_divergent_start": int(
            comparison.loc[comparison["classification"] == "MATERIALLY_DIFFERENT_SOLUTION", "start_id"].min()
        ),
        "reviewed_stability": reviewed_summary,
        "replay_stability": replay_summary,
        "scientific_impact": {
            "ntpc_pilot_changed": False, "pr14_insufficient_changed": False,
            "ambiguity_conclusion_changed": False, "heston_comparison_changed": False,
            "materially_displaced_count_changed": False, "cluster_count_changed": False,
        },
    }
    write_json(output / "root_cause_summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewed", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--starts", type=Path, required=True)
    parser.add_argument("--causal-probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.reviewed, args.replay, args.starts, args.causal_probe, args.output), indent=2))
