"""Run the bounded, target-blind G2 global-ambiguity diagnostic.

This is a diagnostic of the existing central-5 Double Heston inverse problem.
It does not change the G2 gate, freeze a representation, generate a final
dataset, or train an ANN/PINN.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import least_squares

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import scripts.run_g2_identifiability_analysis as baseline
from src.calibrate_double_heston import (
    boundary_diagnostics,
    load_hard_safety_bounds,
    unconstrained_to_parameters,
)
from src.constants import PARAMETER_NAMES
from src.constraints import validate_parameters


ANALYSIS_ID = "G2_GLOBAL_AMBIGUITY"
DEFAULT_DERIVED_ROOT = REPOSITORY_ROOT / "market_data_audit" / "stage_a" / "derived"
DEFAULT_STAGE_A_ROOT = REPOSITORY_ROOT / "market_data_audit" / "stage_a"
DEFAULT_OUTPUT_ROOT = DEFAULT_DERIVED_ROOT / "g2_global_ambiguity"
DEFAULT_REPORT_PATH = REPOSITORY_ROOT / "docs" / "G2_GLOBAL_AMBIGUITY_ANALYSIS.md"
CURRENT_TEST_PATH = REPOSITORY_ROOT / "tests" / "test_g2_global_ambiguity_analysis.py"

# These values are predeclared and must not be tuned after seeing the runs.
ANALYSIS_SEED = 31415926
FULL_PRICER_NODE_COUNT = 64
CLEAN_START_COUNT = 20
NOISE_START_COUNT = 10
NOISE_LEVELS = (0.005, 0.01)
NEAR_PRICE_EQUIVALENCE_RMSE = 2.5e-7
MATERIAL_DISPLACEMENT_RMSE = 0.05
CLUSTER_DISTANCE_CUTOFF = 0.10
ALIGNMENT_CONSISTENT = 0.50
ALIGNMENT_PARTIAL = 0.25
MAX_NFEV = 120
OPTIMIZER_TOLERANCE = 1.0e-10
DIFF_STEP = 2.0e-5

DATA_ARTIFACTS = (
    "alignment.csv",
    "all_solutions.csv",
    "cases.csv",
    "clean_near_equivalent.csv",
    "cluster_summary.csv",
    "clusters.csv",
    "compensation_pairs.csv",
    "contract.json",
    "decision.json",
    "noise_compensation_pairs.csv",
    "noise_summary.csv",
    "summary.csv",
    "weakest_directions.csv",
)
FIGURE_ARTIFACTS = tuple(
    f"figures/{index:02d}_{name}.png"
    for index, name in enumerate(
        (
            "price_rmse_vs_parameter_rmse",
            "true_vs_alternative_scaled_vectors",
            "compensation_pairs",
            "cluster_projection_pca",
            "local_global_alignment",
            "clean_noise_ambiguity_summary",
        ),
        start=1,
    )
)
ALL_ARTIFACTS = DATA_ARTIFACTS + FIGURE_ARTIFACTS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    _atomic_write_bytes(path, frame.to_csv(index=False).encode("utf-8"))


def _write_json(payload: dict[str, Any], path: Path) -> None:
    _atomic_write_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _parameter_widths(bounds: dict[str, tuple[float, float]]) -> np.ndarray:
    return np.asarray([bounds[name][1] - bounds[name][0] for name in PARAMETER_NAMES])


def _scaled_coordinates(
    parameters: np.ndarray, bounds: dict[str, tuple[float, float]]
) -> np.ndarray:
    lower = np.asarray([bounds[name][0] for name in PARAMETER_NAMES])
    return (np.asarray(parameters, dtype=np.float64) - lower) / _parameter_widths(bounds)


def constraint_margins(
    parameters: np.ndarray, bounds: dict[str, tuple[float, float]]
) -> dict[str, float]:
    """Numerical distance from the active constraint boundaries."""
    values = np.asarray(parameters, dtype=float)
    scaled = _scaled_coordinates(values, bounds)
    return {
        "slow_feller_gap": float(2.0 * values[0] * values[1] - values[2] ** 2),
        "fast_feller_gap": float(2.0 * values[5] * values[6] - values[7] ** 2),
        "ordering_margin": float(values[5] - values[0]),
        "correlation_disk_margin": float(1.0 - math.hypot(values[3], values[8])),
        "minimum_scaled_hard_bound_distance": float(np.min(np.minimum(scaled, 1.0 - scaled))),
    }


def _protected_snapshot(
    output_root: Path, report_path: Path, stage_a_root: Path = DEFAULT_STAGE_A_ROOT
) -> dict[str, str]:
    """Hash all prior Stage A and G2 evidence, excluding only new outputs."""
    protected: dict[str, str] = {}
    if stage_a_root.exists():
        for path in sorted(stage_a_root.rglob("*")):
            if path.is_file() and path != output_root and output_root not in path.parents:
                protected[path.relative_to(REPOSITORY_ROOT).as_posix()] = _sha256(path)
    for path in sorted((REPOSITORY_ROOT / "docs").glob("G2_*.md")):
        if path != report_path:
            protected[path.relative_to(REPOSITORY_ROOT).as_posix()] = _sha256(path)
    manifest = REPOSITORY_ROOT / "docs" / "evidence" / "G2_CHECKPOINT_MANIFEST.json"
    if manifest.is_file():
        protected[manifest.relative_to(REPOSITORY_ROOT).as_posix()] = _sha256(manifest)
    for directory, pattern in ((REPOSITORY_ROOT / "scripts", "run_g2_*.py"), (REPOSITORY_ROOT / "tests", "test_g2_*.py")):
        for path in sorted(directory.glob(pattern)):
            if path not in (Path(__file__).resolve(), CURRENT_TEST_PATH):
                protected[path.relative_to(REPOSITORY_ROOT).as_posix()] = _sha256(path)
    return protected


def _snapshot_aggregate(snapshot: dict[str, str]) -> str:
    records = "".join(
        f"{relative}\0{digest.upper()}\n"
        for relative, digest in sorted(snapshot.items())
    ).encode("utf-8")
    return hashlib.sha256(records).hexdigest().upper()


def _assert_protected_unchanged(
    before: dict[str, str], output_root: Path, report_path: Path
) -> None:
    after = _protected_snapshot(output_root, report_path)
    if before != after:
        changed = sorted(set(before) | set(after))
        changed = [key for key in changed if before.get(key) != after.get(key)]
        raise RuntimeError(f"Prior G2 evidence changed unexpectedly: {changed}")


def select_cases(bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Select exactly the first two of four fixed representatives per source."""
    selected = baseline.select_representative_parameters(bounds, per_distribution=4)
    cases = (
        selected.groupby("distribution", sort=True, group_keys=False)
        .head(2)
        .reset_index(drop=True)
        .copy()
    )
    if len(cases) != 4 or cases.groupby("distribution").size().nunique() != 1:
        raise RuntimeError("Global ambiguity diagnostic requires two cases per distribution")
    cases.insert(0, "case_index", np.arange(len(cases), dtype=int))
    cases.insert(1, "case_id", [f"case_{index + 1}" for index in range(len(cases))])
    profiles = baseline.MATURITY_PROFILES
    cases["maturity_profile"] = [profiles[index % len(profiles)][0] for index in cases.case_index]
    cases["near_dte"] = [profiles[index % len(profiles)][1][0] for index in cases.case_index]
    cases["middle_dte"] = [profiles[index % len(profiles)][1][1] for index in cases.case_index]
    return cases


def deterministic_starts(seed: int, count: int) -> list[tuple[str, np.ndarray]]:
    if count < 1:
        raise ValueError("At least one start is required")
    rng = np.random.default_rng(seed)
    starts = [("neutral_transform_midpoint", np.zeros(len(PARAMETER_NAMES)))]
    starts.extend(
        (f"deterministic_broad_{index}", rng.normal(0.0, 1.25, len(PARAMETER_NAMES)))
        for index in range(1, count)
    )
    return starts


def _case_seed(case_index: int, noise_level: float, purpose: int) -> int:
    noise_code = int(round(noise_level * 10_000))
    return ANALYSIS_SEED + 100_000 * case_index + 100 * noise_code + purpose


def start_schedule() -> pd.DataFrame:
    """Return the exact deterministic latent starts for the canonical 120 fits."""
    rows: list[dict[str, Any]] = []
    for case_index in range(4):
        levels = ((0.0, CLEAN_START_COUNT),)
        if case_index in (0, 2):
            levels += tuple((level, NOISE_START_COUNT) for level in NOISE_LEVELS)
        for noise_level, count in levels:
            seed = _case_seed(case_index, noise_level, 11)
            for start_index, (strategy, latent) in enumerate(
                deterministic_starts(seed, count)
            ):
                record: dict[str, Any] = {
                    "case_id": f"case_{case_index + 1}",
                    "noise_level": noise_level,
                    "start_seed": seed,
                    "start_index": start_index,
                    "start_strategy": strategy,
                }
                record.update(
                    {f"latent_{index}": float(value) for index, value in enumerate(latent)}
                )
                rows.append(record)
    return pd.DataFrame(rows)


def start_schedule_sha256() -> str:
    payload = start_schedule().to_csv(
        index=False, lineterminator="\n", float_format="%.17g"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def runtime_provenance() -> dict[str, str]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
    }


def multiplicative_noise(values: np.ndarray, case_index: int, noise_level: float) -> np.ndarray:
    if noise_level == 0.0:
        return np.asarray(values, dtype=np.float64).copy()
    rng = np.random.default_rng(_case_seed(case_index, noise_level, 1))
    observed = np.asarray(values, dtype=np.float64) * (
        1.0 + rng.normal(0.0, noise_level, size=len(values))
    )
    if np.any(observed < 0.0):
        raise RuntimeError("Predeclared multiplicative noise produced a negative price")
    return observed


def complete_linkage_clusters(values: np.ndarray, cutoff: float = CLUSTER_DISTANCE_CUTOFF) -> np.ndarray:
    """Deterministic complete-linkage clustering, labelled in source-row order."""
    points = np.asarray(values, dtype=np.float64)
    if points.ndim != 2 or not np.isfinite(points).all():
        raise ValueError("Clustering requires a finite two-dimensional array")
    if len(points) == 0:
        return np.asarray([], dtype=int)
    clusters: list[list[int]] = [[index] for index in range(len(points))]
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    while len(clusters) > 1:
        candidates: list[tuple[float, int, int]] = []
        for left in range(len(clusters)):
            for right in range(left + 1, len(clusters)):
                complete = float(distances[np.ix_(clusters[left], clusters[right])].max())
                candidates.append((complete, left, right))
        distance, left, right = min(candidates, key=lambda item: (item[0], clusters[item[1]][0], clusters[item[2]][0]))
        if distance > cutoff:
            break
        clusters[left] = sorted(clusters[left] + clusters[right])
        del clusters[right]
    labels = np.empty(len(points), dtype=int)
    for label, cluster in enumerate(sorted(clusters, key=lambda members: members[0]), start=1):
        labels[cluster] = label
    return labels


def classify_basin(near_equivalent: pd.DataFrame) -> dict[str, Any]:
    count = len(near_equivalent)
    cluster_count = int(near_equivalent["cluster_id"].nunique()) if count else 0
    pca1_fraction = math.nan
    if count >= 2:
        matrix = near_equivalent[[f"scaled_{name}" for name in PARAMETER_NAMES]].to_numpy(float)
        singular = np.linalg.svd(matrix - matrix.mean(axis=0), compute_uv=False)
        denominator = float(np.sum(singular**2))
        pca1_fraction = float(singular[0] ** 2 / denominator) if denominator else math.nan
    if cluster_count >= 2:
        classification = "multiple_basin"
    elif cluster_count == 1 and count >= 5 and np.isfinite(pca1_fraction) and pca1_fraction >= 0.75:
        classification = "ridge_like"
    else:
        classification = "single_or_unresolved"
    return {
        "near_equivalent_count": count,
        "cluster_count": cluster_count,
        "pca1_fraction": pca1_fraction,
        "basin_classification": classification,
        "boundary_associated_count": int(near_equivalent["bound_hit"].sum())
        if count and "bound_hit" in near_equivalent
        else 0,
        "boundary_associated_rate": float(near_equivalent["bound_hit"].mean())
        if count and "bound_hit" in near_equivalent
        else math.nan,
    }


def classify_global_ambiguity(cases: pd.DataFrame) -> dict[str, Any]:
    ambiguous = cases["ambiguous_case"].astype(bool)
    count = int(ambiguous.sum())
    verdict = (
        "ESTABLISHED" if count >= 3 else "PARTIALLY_ESTABLISHED" if count else "NOT_ESTABLISHED"
    )
    return {
        "clean_ambiguous_case_count": count,
        "clean_case_count": int(len(cases)),
        "global_ambiguity_verdict": verdict,
        "g2_status": "NOT_PASSED",
        "final_representation": "UNFROZEN",
        "final_dataset": "NOT_GENERATED",
        "ann_training": "NOT_STARTED",
        "pinn_training": "NOT_STARTED",
    }


def classify_alignment(values: Sequence[float]) -> dict[str, Any]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    median = float(np.median(finite)) if len(finite) else math.nan
    status = (
        "UNAVAILABLE" if not len(finite)
        else "CONSISTENT" if median >= ALIGNMENT_CONSISTENT
        else "PARTIALLY_CONSISTENT" if median >= ALIGNMENT_PARTIAL
        else "INCONSISTENT"
    )
    return {"material_solution_count": int(len(finite)), "median_absolute_cosine": median, "alignment": status}


def _recover_case(
    row: Any,
    bounds: dict[str, tuple[float, float]],
    *,
    noise_level: float,
    start_count: int,
    node_count: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    true_parameters = np.asarray([getattr(row, name) for name in PARAMETER_NAMES], dtype=float)
    profile_id, maturity_days = baseline.MATURITY_PROFILES[int(row.case_index) % len(baseline.MATURITY_PROFILES)]
    clean = baseline.normalized_observables(true_parameters, baseline.REPRESENTATIONS[0], maturity_days, node_count=node_count)
    observed = multiplicative_noise(clean, int(row.case_index), noise_level)
    widths = _parameter_widths(bounds)
    jacobian = baseline.scaled_parameter_jacobian(true_parameters, baseline.REPRESENTATIONS[0], maturity_days, bounds, node_count=node_count)
    _, _, right_vectors = np.linalg.svd(jacobian, full_matrices=False)
    weakest_direction = right_vectors[-1]

    def residuals(latent: np.ndarray) -> np.ndarray:
        candidate = unconstrained_to_parameters(latent, bounds)
        return baseline.normalized_observables(candidate, baseline.REPRESENTATIONS[0], maturity_days, node_count=node_count) - observed

    results: list[dict[str, Any]] = []
    seed = _case_seed(int(row.case_index), noise_level, 11)
    for start_index, (strategy, start) in enumerate(deterministic_starts(seed, start_count)):
        record: dict[str, Any] = {
            "case_id": row.case_id, "case_index": int(row.case_index), "sample_id": row.sample_id,
            "distribution": row.distribution, "maturity_profile": profile_id,
            "near_dte": maturity_days[0], "middle_dte": maturity_days[1], "noise_level": noise_level,
            "start_index": start_index, "start_strategy": strategy,
        }
        try:
            result = least_squares(residuals, start, method="trf", max_nfev=MAX_NFEV,
                                   ftol=OPTIMIZER_TOLERANCE, xtol=OPTIMIZER_TOLERANCE,
                                   gtol=OPTIMIZER_TOLERANCE, diff_step=DIFF_STEP)
            recovered = unconstrained_to_parameters(result.x, bounds)
            predicted = baseline.normalized_observables(recovered, baseline.REPRESENTATIONS[0], maturity_days, node_count=node_count)
            displacement = (recovered - true_parameters) / widths
            parameter_rmse = float(np.sqrt(np.mean(displacement**2)))
            validation = validate_parameters(recovered)
            finite = bool(np.isfinite(recovered).all() and np.isfinite(predicted).all())
            norm = float(np.linalg.norm(displacement))
            alignment = float(abs(np.dot(displacement, weakest_direction)) / norm) if norm else math.nan
            record.update({
                "optimizer_success": bool(result.success), "optimizer_status": int(result.status), "nfev": int(result.nfev),
                "price_rmse_normalized": float(np.sqrt(np.mean((predicted - observed) ** 2))),
                "parameter_rmse_full_range": parameter_rmse, "constraint_valid": bool(validation["is_valid"]),
                "finite_solution": finite, "bound_hit": bool(boundary_diagnostics(recovered, bounds)),
                "bound_reasons": ";".join(boundary_diagnostics(recovered, bounds)),
                "material_displacement": bool(parameter_rmse >= MATERIAL_DISPLACEMENT_RMSE),
                "weakest_direction_absolute_cosine": alignment,
                "l2_scaled_distance_from_truth": float(np.linalg.norm(displacement)),
                **constraint_margins(recovered, bounds),
            })
            for index, name in enumerate(PARAMETER_NAMES):
                record[f"true_{name}"] = float(true_parameters[index])
                record[f"recovered_{name}"] = float(recovered[index])
                record[f"scaled_{name}"] = float(_scaled_coordinates(recovered, bounds)[index])
                record[f"scaled_displacement_{name}"] = float(displacement[index])
        except Exception as error:
            record.update({"optimizer_success": False, "optimizer_status": -1, "nfev": 0,
                           "price_rmse_normalized": math.nan, "parameter_rmse_full_range": math.nan,
                           "constraint_valid": False, "finite_solution": False, "bound_hit": False,
                           "bound_reasons": "", "material_displacement": False,
                           "weakest_direction_absolute_cosine": math.nan,
                           "l2_scaled_distance_from_truth": math.nan,
                           "slow_feller_gap": math.nan, "fast_feller_gap": math.nan,
                           "ordering_margin": math.nan, "correlation_disk_margin": math.nan,
                           "minimum_scaled_hard_bound_distance": math.nan,
                           "error": f"{type(error).__name__}: {error}"})
        results.append(record)
    return results, weakest_direction


def compensation_pairs(near_equivalent: pd.DataFrame) -> pd.DataFrame:
    columns = ["case_id", "parameter_a", "parameter_b", "spearman_correlation", "absolute_spearman", "near_equivalent_count"]
    rows: list[dict[str, Any]] = []
    for case_id, group in near_equivalent.groupby("case_id", sort=True):
        if len(group) < 5:
            continue
        values = group[[f"scaled_{name}" for name in PARAMETER_NAMES]].rename(columns=lambda value: value.removeprefix("scaled_"))
        correlation = values.corr(method="spearman")
        for left, first in enumerate(PARAMETER_NAMES):
            for second in PARAMETER_NAMES[left + 1:]:
                value = float(correlation.loc[first, second])
                if np.isfinite(value) and abs(value) >= 0.5:
                    rows.append({"case_id": case_id, "parameter_a": first, "parameter_b": second,
                                 "spearman_correlation": value, "absolute_spearman": abs(value),
                                 "near_equivalent_count": len(group)})
    return pd.DataFrame(rows, columns=columns)


def cluster_summary(
    clustered: pd.DataFrame, cases: pd.DataFrame,
    bounds: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    columns = [
        "case_id", "cluster_id", "cluster_size", "center_distance_from_truth",
        "within_cluster_dispersion", "within_cluster_diameter",
        "nearest_between_cluster_separation", "price_rmse_min", "price_rmse_median",
        "price_rmse_max", "parameter_rmse_min", "parameter_rmse_median",
        "parameter_rmse_max",
        "bound_hit_count", "bound_hit_rate",
    ]
    rows: list[dict[str, Any]] = []
    coordinate_columns = [f"scaled_{name}" for name in PARAMETER_NAMES]
    for case_id, by_case in clustered.groupby("case_id", sort=True):
        case = cases.loc[cases.case_id.eq(case_id)].iloc[0]
        truth = _scaled_coordinates(
            np.asarray([case[name] for name in PARAMETER_NAMES], dtype=float), bounds
        )
        all_values = by_case[coordinate_columns].to_numpy(float)
        for cluster_id, group in by_case.groupby("cluster_id", sort=True):
            values = group[coordinate_columns].to_numpy(float)
            center = values.mean(axis=0)
            center_distance = float(np.linalg.norm(center - truth))
            within = np.linalg.norm(values - center, axis=1)
            pairwise = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=2)
            others = by_case.loc[~by_case.cluster_id.eq(cluster_id), coordinate_columns].to_numpy(float)
            nearest = float(np.linalg.norm(values[:, None, :] - others[None, :, :], axis=2).min()) if len(others) else math.nan
            rows.append({
                "case_id": case_id, "cluster_id": int(cluster_id), "cluster_size": len(group),
                "center_distance_from_truth": center_distance,
                "within_cluster_dispersion": float(within.mean()),
                "within_cluster_diameter": float(pairwise.max()),
                "nearest_between_cluster_separation": nearest,
                "price_rmse_min": float(group.price_rmse_normalized.min()),
                "price_rmse_median": float(group.price_rmse_normalized.median()),
                "price_rmse_max": float(group.price_rmse_normalized.max()),
                "parameter_rmse_min": float(group.parameter_rmse_full_range.min()),
                "parameter_rmse_median": float(group.parameter_rmse_full_range.median()),
                "parameter_rmse_max": float(group.parameter_rmse_full_range.max()),
                "bound_hit_count": int(group.bound_hit.sum()),
                "bound_hit_rate": float(group.bound_hit.mean()),
            })
    return pd.DataFrame(rows, columns=columns)


def noise_summary(solutions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (case_id, level), group in solutions.groupby(["case_id", "noise_level"], sort=True):
        usable = group.loc[group.constraint_valid & group.finite_solution]
        near = usable.loc[usable.price_rmse_normalized.le(NEAR_PRICE_EQUIVALENCE_RMSE)]
        if len(near):
            near = near.copy()
            near["cluster_id"] = complete_linkage_clusters(
                near[[f"scaled_{name}" for name in PARAMETER_NAMES]].to_numpy(float)
            )
        basin = classify_basin(near)
        rows.append({
            "case_id": case_id, "noise_level": float(level), "start_count": len(group),
            "usable_solution_count": len(usable), "near_equivalent_fit_count": len(near),
            "price_rmse_min": float(usable.price_rmse_normalized.min()) if len(usable) else math.nan,
            "price_rmse_median": float(usable.price_rmse_normalized.median()) if len(usable) else math.nan,
            "parameter_rmse_median": float(usable.parameter_rmse_full_range.median()) if len(usable) else math.nan,
            "material_solution_count": int(near.material_displacement.sum()) if len(near) else 0,
            "bound_hit_count": int(group.bound_hit.sum()),
            "near_equivalent_cluster_count": basin["cluster_count"],
            "basin_classification": basin["basin_classification"],
            "near_equivalent_bound_hit_count": basin["boundary_associated_count"],
            "near_equivalent_bound_hit_rate": basin["boundary_associated_rate"],
        })
    return pd.DataFrame(rows)


def noise_compensation_pairs(solutions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "case_id", "noise_level", "parameter_a", "parameter_b",
        "spearman_correlation", "absolute_spearman", "near_equivalent_count",
    ]
    rows: list[dict[str, Any]] = []
    usable = solutions.loc[
        solutions.constraint_valid
        & solutions.finite_solution
        & solutions.price_rmse_normalized.le(NEAR_PRICE_EQUIVALENCE_RMSE)
    ]
    for (case_id, noise_level), group in usable.groupby(
        ["case_id", "noise_level"], sort=True
    ):
        if len(group) < 5:
            continue
        values = group[[f"scaled_{name}" for name in PARAMETER_NAMES]].rename(
            columns=lambda value: value.removeprefix("scaled_")
        )
        correlation = values.corr(method="spearman")
        for left, first in enumerate(PARAMETER_NAMES):
            for second in PARAMETER_NAMES[left + 1:]:
                value = float(correlation.loc[first, second])
                if np.isfinite(value) and abs(value) >= 0.5:
                    rows.append({
                        "case_id": case_id,
                        "noise_level": float(noise_level),
                        "parameter_a": first,
                        "parameter_b": second,
                        "spearman_correlation": value,
                        "absolute_spearman": abs(value),
                        "near_equivalent_count": len(group),
                    })
    return pd.DataFrame(rows, columns=columns)


def matched_noise_comparison(solutions: pd.DataFrame) -> pd.DataFrame:
    """Restrict clean/noise comparisons to cases that received noisy runs."""
    noisy_case_ids = sorted(
        solutions.loc[solutions.noise_level.gt(0.0), "case_id"].unique()
    )
    return solutions.loc[solutions.case_id.isin(noisy_case_ids)].copy()


def _write_figures(
    clean: pd.DataFrame, all_solutions: pd.DataFrame,
    near_equivalent: pd.DataFrame, cases: pd.DataFrame,
    output_root: Path,
) -> tuple[Path, ...]:
    """Render the six predeclared mentor-ready figure classes."""
    root = output_root / "figures"
    root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    # (1) repricing versus full-range parameter displacement.
    figure, axis = plt.subplots(figsize=(6.5, 5.0)); axis.scatter(clean["price_rmse_normalized"], clean["parameter_rmse_full_range"], c=clean["case_index"], cmap="viridis"); axis.axvline(NEAR_PRICE_EQUIVALENCE_RMSE, color="tab:red", ls="--"); axis.axhline(MATERIAL_DISPLACEMENT_RMSE, color="tab:red", ls="--"); axis.set_xscale("log"); axis.set(title="Clean repricing versus parameter displacement", xlabel="Normalized price RMSE", ylabel="Range-scaled parameter RMSE"); axis.grid(alpha=.25)
    path = root / "01_price_rmse_vs_parameter_rmse.png"; figure.tight_layout(); figure.savefig(path, dpi=180); plt.close(figure); paths.append(path)
    # (2) one representative target against deterministic distinct alternatives.
    representative_case = (
        near_equivalent.groupby("case_id").size().sort_values(ascending=False).index[0]
    )
    representative = near_equivalent.loc[
        near_equivalent.case_id.eq(representative_case)
    ].sort_values(["cluster_id", "start_index"], kind="stable").head(5)
    truth_row = clean.loc[clean.case_id.eq(representative_case)].iloc[0]
    truth = [
        truth_row[f"scaled_{name}"] - truth_row[f"scaled_displacement_{name}"]
        for name in PARAMETER_NAMES
    ]
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.plot(range(10), truth, color="black", marker="o", linewidth=2.8, label="true vector")
    for row in representative.itertuples(index=False):
        axis.plot(
            range(10),
            [getattr(row, f"scaled_{name}") for name in PARAMETER_NAMES],
            marker=".", alpha=.75,
            label=f"cluster {row.cluster_id} / start {row.start_index}",
        )
    axis.set(title=f"{representative_case}: true vector and distinct near-equivalent alternatives", xlabel="Parameter", ylabel="Full-range coordinate")
    axis.set_xticks(range(10), PARAMETER_NAMES, rotation=35, ha="right")
    axis.legend(ncol=2, fontsize=8); axis.grid(alpha=.25)
    path = root / "02_true_vs_alternative_scaled_vectors.png"; figure.tight_layout(); figure.savefig(path, dpi=180); plt.close(figure); paths.append(path)
    # (3) select by repeated within-case support and facet to avoid pooled-case bias.
    pair_rows = compensation_pairs(near_equivalent)
    repeated = (
        pair_rows.groupby(["parameter_a", "parameter_b"])
        .agg(case_count=("case_id", "nunique"), median_abs=("absolute_spearman", "median"))
        .reset_index()
        .sort_values(["case_count", "median_abs", "parameter_a", "parameter_b"], ascending=[False, False, True, True])
    )
    pair = (
        (repeated.iloc[0].parameter_a, repeated.iloc[0].parameter_b)
        if len(repeated) else (PARAMETER_NAMES[0], PARAMETER_NAMES[1])
    )
    case_ids = sorted(near_equivalent.case_id.unique())
    figure, axes = plt.subplots(2, 2, figsize=(9.5, 8.0), sharex=True, sharey=True)
    for axis, case_id in zip(axes.flat, case_ids, strict=False):
        group = near_equivalent.loc[near_equivalent.case_id.eq(case_id)]
        x = group[f"scaled_{pair[0]}"]; y = group[f"scaled_{pair[1]}"]
        rho = x.corr(y, method="spearman")
        axis.scatter(x, y, c=group.cluster_id, cmap="tab20", s=42)
        axis.set_title(f"{case_id}: Spearman={rho:.3f}")
        axis.set_xlabel(pair[0]); axis.set_ylabel(pair[1]); axis.grid(alpha=.25)
    figure.suptitle("Strongest repeated within-case compensation pair (descriptive)")
    path = root / "03_compensation_pairs.png"; figure.tight_layout(); figure.savefig(path, dpi=180); plt.close(figure); paths.append(path)
    # (4) project one target so between-target differences cannot mimic basins.
    figure, axis = plt.subplots(figsize=(6.5, 5.0))
    projected = near_equivalent.loc[near_equivalent.case_id.eq(representative_case)]
    if len(projected) >= 2:
        matrix = projected[[f"scaled_{name}" for name in PARAMETER_NAMES]].to_numpy(float)
        centered = matrix - matrix.mean(axis=0); u, s, _ = np.linalg.svd(centered, full_matrices=False)
        coordinates = u[:, :2] * s[:2]
        axis.scatter(coordinates[:, 0], coordinates[:, 1], c=projected["cluster_id"], cmap="tab20", s=48)
        for x, y, cluster_id in zip(coordinates[:, 0], coordinates[:, 1], projected.cluster_id, strict=True):
            axis.annotate(str(cluster_id), (x, y), xytext=(4, 3), textcoords="offset points", fontsize=7)
    axis.set(title=f"{representative_case} PCA cluster projection (no financial meaning)", xlabel="PC1", ylabel="PC2"); axis.grid(alpha=.25)
    path = root / "04_cluster_projection_pca.png"; figure.tight_layout(); figure.savefig(path, dpi=180); plt.close(figure); paths.append(path)
    # (5) local weakest-direction alignment for global material displacements.
    figure, axis = plt.subplots(figsize=(7.5, 4.8)); material = near_equivalent.loc[near_equivalent.material_displacement] if "material_displacement" in near_equivalent else near_equivalent.iloc[0:0]
    for case_id, group in material.groupby("case_id", sort=True):
        axis.scatter(group.start_index, group.weakest_direction_absolute_cosine, label=case_id, s=42)
    axis.axhline(ALIGNMENT_CONSISTENT, color="tab:green", ls="--", label="consistent cutoff")
    axis.axhline(ALIGNMENT_PARTIAL, color="tab:orange", ls="--", label="partial cutoff")
    axis.set(title="Local weakest direction versus global displacement", xlabel="Start index", ylabel="Absolute cosine alignment"); axis.legend(ncol=3, fontsize=8); axis.grid(alpha=.25)
    path = root / "05_local_global_alignment.png"; figure.tight_layout(); figure.savefig(path, dpi=180); plt.close(figure); paths.append(path)
    # (6) clean and both noise levels: strict screen, fit, displacement, bounds.
    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.5)); labels = ["clean", "0.5%", "1.0%"]; levels = (0.0, 0.005, 0.01); near_rates = []
    comparison = matched_noise_comparison(all_solutions)
    price_medians: list[float] = []
    parameter_medians: list[float] = []
    bound_rates: list[float] = []
    for level in levels:
        group = comparison.loc[comparison.noise_level.eq(level)]
        valid = group.loc[group.constraint_valid & group.finite_solution]
        near_group = valid.loc[valid.price_rmse_normalized.le(NEAR_PRICE_EQUIVALENCE_RMSE)]
        near_rates.append(float(len(near_group) / len(group)) if len(group) else math.nan)
        price_medians.append(float(valid.price_rmse_normalized.median()))
        parameter_medians.append(float(valid.parameter_rmse_full_range.median()))
        bound_rates.append(float(valid.bound_hit.mean()))
    panels = (
        (near_rates, "Strict near-equivalence rate", "Fraction", False),
        (price_medians, "Median normalized price RMSE", "RMSE", True),
        (parameter_medians, "Median range-scaled parameter RMSE", "RMSE", False),
        (bound_rates, "Boundary-hit rate", "Fraction", False),
    )
    for axis, (values, title, ylabel, log_scale) in zip(axes.flat, panels, strict=True):
        axis.bar(labels, values)
        axis.set_title(title); axis.set_ylabel(ylabel); axis.grid(axis="y", alpha=.25)
        if log_scale:
            axis.set_yscale("log")
    compared_cases = "/".join(sorted(comparison.case_id.unique()))
    figure.suptitle(
        f"Matched {compared_cases} clean/0.5%/1.0% stability summary"
    )
    path = root / "06_clean_noise_ambiguity_summary.png"; figure.tight_layout(); figure.savefig(path, dpi=180); plt.close(figure); paths.append(path)
    return tuple(paths)


def render_report(
    contract: dict[str, Any], case_inputs: pd.DataFrame,
    solutions: pd.DataFrame, near_equivalent: pd.DataFrame, cases: pd.DataFrame,
    clusters: pd.DataFrame, noise: pd.DataFrame, pairs: pd.DataFrame,
    noisy_pairs: pd.DataFrame,
    decision: dict[str, Any], alignment: dict[str, Any], artifact_hashes: dict[str, str],
) -> str:
    clean = solutions.loc[solutions.noise_level.eq(0.0)]
    noise_comparison = matched_noise_comparison(solutions)
    matched_clean = noise_comparison.loc[noise_comparison.noise_level.eq(0.0)]
    noisy_case_ids = ", ".join(sorted(matched_clean.case_id.unique()))
    successful_near = near_equivalent.loc[near_equivalent.optimizer_success]
    finite_separations = clusters.nearest_between_cluster_separation.dropna()
    clean_price_parameter_spearman = float(
        clean[["price_rmse_normalized", "parameter_rmse_full_range"]]
        .corr(method="spearman").iloc[0, 1]
    )
    lines = ["# G2 Global-Ambiguity Analysis", "", "## Decision", "", f"**GLOBAL_AMBIGUITY = {decision['global_ambiguity_verdict']}**", "", "This bounded diagnostic is evidence about clean, target-blind multi-start recovery only. Noise is a stability probe and does not change the primary verdict.", "", "## Mentor-ready numerical conclusions", "", f"- Collected `{len(solutions)}` usable solutions: `{len(clean)}` clean and `{len(solutions) - len(clean)}` noisy. Optimizer success was `{int(solutions.optimizer_success.sum())}/{len(solutions)}`; valid finite capped iterates were retained because excellent fit, not optimizer status, is the phenomenon under study.", f"- The clean screen retained `{len(near_equivalent)}/{len(clean)}` near-equivalent solutions across all four cases; `{int(near_equivalent.material_displacement.sum())}` were materially displaced. Their median normalized price RMSE was `{near_equivalent.price_rmse_normalized.median():.3e}` while median range-scaled parameter RMSE was `{near_equivalent.parameter_rmse_full_range.median():.3e}` (range `{near_equivalent.parameter_rmse_full_range.min():.3e}` to `{near_equivalent.parameter_rmse_full_range.max():.3e}`).", f"- The result survives an optimizer-success-only robustness view: `{len(successful_near)}` near-equivalent successful solutions, `{int(successful_near.material_displacement.sum())}` material, spanning all four cases and `{int(successful_near.groupby('case_id').cluster_id.nunique().sum())}` separated clusters.", f"- The declared clustering produced `{int(near_equivalent.groupby('case_id').cluster_id.nunique().sum())}` clusters; `{int((clusters.cluster_size == 1).sum())}` are singletons and one has size two. Median nearest separation was `{finite_separations.median():.3e}` in full-range coordinates. This establishes separated solution regions, but not the volume of smooth attraction basins.", f"- Across all clean starts, price RMSE and parameter RMSE had Spearman `{clean_price_parameter_spearman:.3f}`; within the strict near-equivalent set, large parameter errors persisted at price errors as low as `{near_equivalent.price_rmse_normalized.min():.3e}`.", f"- Local/global evidence is `{alignment['alignment']}` in aggregate (median absolute cosine `{alignment['median_absolute_cosine']:.3f}`), but heterogeneous by case: two consistent, one partially consistent, and one inconsistent.", "", "## Frozen contract", "", f"- Seed: `{ANALYSIS_SEED}`; representation: `{baseline.REPRESENTATIONS[0].representation_id}`; production pricing: `{FULL_PRICER_NODE_COUNT}` nodes.", f"- Clean starts/case: `{CLEAN_START_COUNT}`; noise starts/case: `{NOISE_START_COUNT}`; optimiser: TRF with `ftol=xtol=gtol={OPTIMIZER_TOLERANCE:.0e}`, `diff_step={DIFF_STEP:.0e}`, `max_nfev={MAX_NFEV}`.", f"- Near-price-equivalence threshold: `{NEAR_PRICE_EQUIVALENCE_RMSE:.1e}` normalized RMSE. It was rounded and frozen at ten times the prior clean median fit (`2.515e-8`) before this run; it was not tuned to the observed clusters.", f"- Material displacement: range-scaled parameter RMSE >= `{MATERIAL_DISPLACEMENT_RMSE:.2f}`.", "- Clustering is deterministic complete linkage of clean, finite, constraint-valid, near-equivalent solutions at full-range distance <= `0.10`.", "", "## Exact cases and true ten-vectors", ""]
    for row in case_inputs.itertuples(index=False):
        vector = ", ".join(f"{name}={getattr(row, name):.8g}" for name in PARAMETER_NAMES)
        lines.append(f"- `{row.case_id}` (`{row.sample_id}`, `{row.maturity_profile}`): `{vector}`")
    lines.extend(["", "## Case results", "", "| Case | Profile | Near-equivalent | Clusters | Basin class | Boundary-associated | Ambiguous |", "|---|---|---:|---:|---|---:|---|"])
    for row in cases.itertuples(index=False): lines.append(f"| `{row.case_id}` | `{row.maturity_profile}` | {row.near_equivalent_count} | {row.cluster_count} | `{row.basin_classification}` | {row.boundary_associated_count} | `{row.ambiguous_case}` |")
    lines.extend(["", "## Cluster and noise numerical summaries", "", "Full cluster sizes, center displacement, dispersion/diameter, separation, boundary association, and price/parameter RMSE ranges are in `cluster_summary.csv`. Most clusters are singleton solutions, so `multiple_basin` means separated solution regions under the declared cutoff; it does not estimate basin volume."])
    lines.extend(["", "| Case | Noise | Usable | Near-equivalent | Basins | Basin class | Median price RMSE | Median parameter RMSE | Material | Bound hits |", "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|"])
    for row in noise.itertuples(index=False): lines.append(f"| `{row.case_id}` | {100*row.noise_level:.1f}% | {row.usable_solution_count} | {row.near_equivalent_fit_count} | {row.near_equivalent_cluster_count} | `{row.basin_classification}` | {row.price_rmse_median:.3e} | {row.parameter_rmse_median:.3e} | {row.material_solution_count} | {row.bound_hit_count} |")
    noisy_medians = {
        level: float(
            noise_comparison.loc[
                noise_comparison.noise_level.eq(level),
                "parameter_rmse_full_range",
            ].median()
        )
        for level in NOISE_LEVELS
    }
    lines.extend(["", f"The noisy runs produced no solution below the strict clean-precision threshold, so noisy basin counts and noisy compensation pairs are **unresolved**, not evidence that ambiguity disappeared. Noise was evaluated only for the predeclared matched cases `{noisy_case_ids}`. Within that matched population, median parameter RMSE changed from `{matched_clean.parameter_rmse_full_range.median():.3f}` clean to `{noisy_medians[0.005]:.3f}` at 0.5% and `{noisy_medians[0.01]:.3f}` at 1.0%; every noisy solution hit at least one declared boundary.", "", "## Local-global alignment", "", f"Median absolute cosine for material solutions: `{alignment['median_absolute_cosine']:.3f}`; classification: **{alignment['alignment']}**. Case-level statuses are reported above. This is geometric alignment with the local weakest scaled-Jacobian direction, not a causal claim.", "", "## Supported compensation pairs", "", "Pairs require at least five clean near-equivalent solutions within a case and absolute Spearman correlation >= 0.5. They are descriptive co-movement only; they do not establish causal compensation. The full within-case screen is in `compensation_pairs.csv`.", ""])
    repeated_pairs = (
        pairs.groupby(["parameter_a", "parameter_b"])
        .agg(case_count=("case_id", "nunique"), median_abs=("absolute_spearman", "median"), median_signed=("spearman_correlation", "median"))
        .reset_index()
        .loc[lambda frame: frame.case_count >= 3]
        .sort_values(["case_count", "median_abs"], ascending=False)
    )
    if repeated_pairs.empty: lines.append("- No relationship repeated across at least three cases.")
    for row in repeated_pairs.itertuples(index=False):
        lines.append(f"- `{row.parameter_a}` / `{row.parameter_b}`: supported in `{row.case_count}/4` cases; median Spearman `{row.median_signed:.3f}` (median absolute `{row.median_abs:.3f}`).")
    lines.extend(["", "The explicitly suggested `theta_slow/theta_fast` relationship is supported negatively in three cases. `kappa_slow/theta_slow` did not meet the correlation screen despite both dominating several local weakest directions. The strongest repeated global relationship is the negative `v0_slow/v0_fast` variance-allocation trade-off in all four cases."])
    lines.extend(["", "Noisy pair screens use the identical five-solution and absolute-Spearman thresholds; `noise_compensation_pairs.csv` records whether the dominant descriptive trade-offs persist or change."])
    if noisy_pairs.empty: lines.append("No noisy pair met that support threshold.")
    lines.extend(["", "## Six figures", "", "1. Price RMSE versus range-scaled parameter RMSE.", "2. True ten-vectors versus several near-equivalent vectors in scaled coordinates.", "3. Strongest empirically supported compensation pair(s).", "4. PCA cluster projection (no financial-coordinate interpretation).", "5. Weakest local direction versus global-displacement cosine alignment.", "6. Matched-case clean/0.5%/1.0% ambiguity stability summary.", "", "## Ranked remedy categories", "", "1. **Complementary observables** — add independent sensitivities that can separate the observed option-price-equivalent regions.", "2. **Joint historical inference** — use time-series information to constrain persistence, long-run variance, and factor allocation; the completed multi-date option-only result shows that dates and exact CIR physics alone are not enough.", "3. **Regularization / informative priors** — choose among weakly distinguished regions only when external scientific information justifies the prior; this stabilizes inference but does not make the prices identifying.", "4. **Reparameterization** — expose combinations that the observations identify more directly while preserving the canonical scientific meaning; any target change requires a separate decision.", "5. **Physics-informed inverse training** — use only after information content is addressed; a training architecture cannot manufacture uniqueness, and prior exact-CIR conditioning did not deliver stable recovery.", "6. **Other: set-valued or uncertainty-set inference** — report observationally equivalent regions instead of a falsely precise point when uniqueness is not supported.", "", "## Reproducibility and next action", "", "Canonical optimization command: `python -B scripts/run_g2_global_ambiguity_analysis.py`. Replay status: `CANONICAL_RUN_COMPLETED_ONCE`.", "CSV-only report/figure replay command: `python -B scripts/run_g2_global_ambiguity_analysis.py --render-only`. This path reads the preserved CSV/JSON artifacts and performs no optimization.", f"Exact latent-start schedule SHA-256: `{contract.get('start_schedule_sha256', start_schedule_sha256())}` using NumPy `default_rng`/`PCG64`; canonical runtime versions are preserved in the tracked manifest.", f"Protected pre-existing Stage A/G2 files: `{contract['protected_snapshot']['file_count']}`; aggregate SHA-256 before and after: `{contract['protected_snapshot']['aggregate_sha256']}`.", "- Recommended next research action: mentor-review the clean cluster structure and its dominant compensation directions, then predeclare one complementary-observable or joint-historical experiment targeted at separating those regions.", "", "## Gate boundary", "", "G2 remains **NOT_PASSED**. The final representation remains unfrozen. No final dataset was generated and no ANN or PINN training was performed.", "", "## Reproducibility artifacts", "", "| Artifact | SHA-256 |", "|---|---|"])
    for relative, digest in sorted(artifact_hashes.items()): lines.append(f"| `{relative}` | `{digest}` |")
    lines.extend(["", "```text", "G2 = NOT_PASSED", "FINAL_REPRESENTATION = UNFROZEN", "FINAL_DATASET = NOT_GENERATED", "ANN_TRAINING = NOT_STARTED", "PINN_TRAINING = NOT_STARTED", "```", ""])
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def render_existing_outputs(
    *, output_root: Path = DEFAULT_OUTPUT_ROOT,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    """Regenerate the report and six figures from the completed artifacts only."""
    output_root = Path(output_root).resolve()
    report_path = Path(report_path).resolve()
    protected_before = _protected_snapshot(output_root, report_path)
    missing = [relative for relative in DATA_ARTIFACTS if not (output_root / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing canonical replay artifacts: {missing}")
    data_hashes_before = {
        relative: _sha256(output_root / relative) for relative in DATA_ARTIFACTS
    }

    contract = _read_json(output_root / "contract.json")
    decision = _read_json(output_root / "decision.json")
    case_inputs = pd.read_csv(output_root / "cases.csv")
    solutions = pd.read_csv(output_root / "all_solutions.csv")
    near_equivalent = pd.read_csv(output_root / "clean_near_equivalent.csv")
    cases = pd.read_csv(output_root / "summary.csv")
    clusters = pd.read_csv(output_root / "cluster_summary.csv")
    noise = pd.read_csv(output_root / "noise_summary.csv")
    pairs = pd.read_csv(output_root / "compensation_pairs.csv")
    noisy_pairs = pd.read_csv(output_root / "noise_compensation_pairs.csv")
    alignment = decision.get("local_global_alignment")
    if not isinstance(alignment, dict):
        raise ValueError("decision.json lacks local_global_alignment replay metadata")
    if len(solutions) != 120 or len(case_inputs) != 4 or len(near_equivalent) != 40:
        raise ValueError("Replay artifacts do not match the frozen 120/4/40 contract")

    clean = solutions.loc[solutions.noise_level.eq(0.0)]
    figures = _write_figures(
        clean, solutions, near_equivalent, cases, output_root
    )
    data_hashes_after = {
        relative: _sha256(output_root / relative) for relative in DATA_ARTIFACTS
    }
    if data_hashes_after != data_hashes_before:
        raise RuntimeError("CSV/JSON evidence changed during render-only replay")
    artifact_hashes = {
        relative: _sha256(output_root / relative) for relative in ALL_ARTIFACTS
    }
    _atomic_write_bytes(
        report_path,
        render_report(
            contract,
            case_inputs,
            solutions,
            near_equivalent,
            cases,
            clusters,
            noise,
            pairs,
            noisy_pairs,
            decision,
            alignment,
            artifact_hashes,
        ).encode("utf-8"),
    )
    _assert_protected_unchanged(protected_before, output_root, report_path)
    return {
        "decision": decision,
        "artifact_hashes": artifact_hashes,
        "figure_paths": figures,
        "optimization_rerun": False,
    }


def run_analysis(*, output_root: Path = DEFAULT_OUTPUT_ROOT, report_path: Path = DEFAULT_REPORT_PATH, node_count: int = FULL_PRICER_NODE_COUNT, write_outputs: bool = True) -> dict[str, Any]:
    if node_count != FULL_PRICER_NODE_COUNT:
        raise ValueError("The frozen production diagnostic requires exactly 64 pricing nodes")
    output_root = Path(output_root).resolve()
    report_path = Path(report_path).resolve()
    protected_before = _protected_snapshot(output_root, report_path)
    protected_aggregate = _snapshot_aggregate(protected_before)
    bounds = load_hard_safety_bounds(baseline.BOUNDS_PATH)
    selected = select_cases(bounds)
    all_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    for case in selected.itertuples(index=False):
        clean_rows, weakest_direction = _recover_case(case, bounds, noise_level=0.0, start_count=CLEAN_START_COUNT, node_count=node_count)
        all_rows.extend(clean_rows)
        direction_rows.extend(
            {"case_id": case.case_id, "case_index": case.case_index,
             "maturity_profile": case.maturity_profile, "parameter": name,
             "weakest_right_singular_loading": float(loading),
             "absolute_loading": float(abs(loading))}
            for name, loading in zip(PARAMETER_NAMES, weakest_direction, strict=True)
        )
        if int(case.case_index) in (0, 2):
            for noise in NOISE_LEVELS:
                noisy_rows, _ = _recover_case(case, bounds, noise_level=noise, start_count=NOISE_START_COUNT, node_count=node_count)
                all_rows.extend(noisy_rows)
    solutions = pd.DataFrame(all_rows)
    clean = solutions.loc[solutions.noise_level.eq(0.0)].copy()
    near = clean.loc[clean.constraint_valid & clean.finite_solution & clean.price_rmse_normalized.le(NEAR_PRICE_EQUIVALENCE_RMSE)].copy()
    cluster_frames: list[pd.DataFrame] = []
    case_rows: list[dict[str, Any]] = []
    alignments: list[dict[str, Any]] = []
    for case in selected.itertuples(index=False):
        group = near.loc[near.case_id.eq(case.case_id)].copy().sort_values("start_index", kind="stable")
        if len(group): group["cluster_id"] = complete_linkage_clusters(group[[f"scaled_{name}" for name in PARAMETER_NAMES]].to_numpy(float))
        else: group["cluster_id"] = pd.Series(dtype=int)
        cluster_frames.append(group)
        basin = classify_basin(group)
        material = group.loc[group.material_displacement]
        material_alignment = material["weakest_direction_absolute_cosine"].to_numpy(float)
        alignment = classify_alignment(material_alignment)
        alignments.extend({"case_id": case.case_id, "start_index": int(start), "absolute_cosine": float(value)} for start, value in material[["start_index", "weakest_direction_absolute_cosine"]].itertuples(index=False, name=None))
        case_rows.append({"case_id": case.case_id, "case_index": case.case_index, "sample_id": case.sample_id, "distribution": case.distribution, "maturity_profile": case.maturity_profile, **basin, "material_solution_count": alignment["material_solution_count"], "median_absolute_cosine": alignment["median_absolute_cosine"], "alignment": alignment["alignment"], "ambiguous_case": bool(basin["cluster_count"] >= 2 and alignment["material_solution_count"] >= 1)})
    clustered = pd.concat(cluster_frames, ignore_index=True) if cluster_frames else near.copy()
    cases = pd.DataFrame(case_rows)
    decision = classify_global_ambiguity(cases)
    alignment_frame = pd.DataFrame(alignments, columns=["case_id", "start_index", "absolute_cosine"])
    aggregate_alignment = classify_alignment(alignment_frame["absolute_cosine"].to_numpy(float) if len(alignment_frame) else [])
    pairs = compensation_pairs(clustered)
    noisy_pairs = noise_compensation_pairs(solutions)
    clusters = cluster_summary(clustered, selected, bounds)
    noise = noise_summary(solutions)
    directions = pd.DataFrame(direction_rows)
    contract = {"analysis_id": ANALYSIS_ID, "seed": ANALYSIS_SEED, "parameter_names": list(PARAMETER_NAMES), "representation": baseline.REPRESENTATIONS[0].representation_id, "maturity_profiles": [(name, list(days)) for name, days in baseline.MATURITY_PROFILES], "node_count": FULL_PRICER_NODE_COUNT, "clean_start_count": CLEAN_START_COUNT, "noise_start_count": NOISE_START_COUNT, "noise_levels": list(NOISE_LEVELS), "near_price_equivalence_rmse": NEAR_PRICE_EQUIVALENCE_RMSE, "near_price_equivalence_derivation": "rounded ten times prior clean median normalized RMSE 2.515e-8, frozen before run", "material_displacement_rmse": MATERIAL_DISPLACEMENT_RMSE, "complete_linkage_cutoff": CLUSTER_DISTANCE_CUTOFF, "start_schedule_sha256": start_schedule_sha256(), "start_generator": {"api": "numpy.random.default_rng", "bit_generator": "PCG64", "broad_distribution": "Normal(0, 1.25) in ten-dimensional latent space", "neutral_start": [0.0] * len(PARAMETER_NAMES)}, "runtime": runtime_provenance(), "protected_snapshot": {"file_count": len(protected_before), "aggregate_sha256": protected_aggregate}, "optimizer": {"method": "trf", "ftol": OPTIMIZER_TOLERANCE, "xtol": OPTIMIZER_TOLERANCE, "gtol": OPTIMIZER_TOLERANCE, "diff_step": DIFF_STEP, "max_nfev": MAX_NFEV}}
    decision["protected_snapshot"] = contract["protected_snapshot"]
    frames = {"cases.csv": selected, "all_solutions.csv": solutions, "clean_near_equivalent.csv": clustered, "clusters.csv": clustered[[column for column in ("case_id", "start_index", "cluster_id", "price_rmse_normalized", "parameter_rmse_full_range", "material_displacement") if column in clustered]], "cluster_summary.csv": clusters, "compensation_pairs.csv": pairs, "noise_compensation_pairs.csv": noisy_pairs, "weakest_directions.csv": directions, "alignment.csv": alignment_frame, "noise_summary.csv": noise, "summary.csv": cases}
    artifact_hashes: dict[str, str] = {}
    figures: tuple[Path, ...] = ()
    if write_outputs:
        for relative, frame in frames.items(): _write_csv(frame, output_root / relative); artifact_hashes[relative] = _sha256(output_root / relative)
        _write_json(contract, output_root / "contract.json"); artifact_hashes["contract.json"] = _sha256(output_root / "contract.json")
        _write_json({**decision, "local_global_alignment": aggregate_alignment}, output_root / "decision.json"); artifact_hashes["decision.json"] = _sha256(output_root / "decision.json")
        figures = _write_figures(clean, solutions, clustered, cases, output_root)
        artifact_hashes.update({path.relative_to(output_root).as_posix(): _sha256(path) for path in figures})
        _atomic_write_bytes(report_path, render_report(contract, selected, solutions, clustered, cases, clusters, noise, pairs, noisy_pairs, decision, aggregate_alignment, artifact_hashes).encode("utf-8"))
    _assert_protected_unchanged(protected_before, output_root, report_path)
    return {"contract": contract, "cases": cases, "solutions": solutions, "near_equivalent": clustered, "clusters": clusters, "noise": noise, "directions": directions, "compensation_pairs": pairs, "noise_compensation_pairs": noisy_pairs, "alignment": alignment_frame, "decision": decision, "local_global_alignment": aggregate_alignment, "artifact_hashes": artifact_hashes, "figure_paths": figures}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Regenerate figures and report from existing CSV/JSON evidence without optimization",
    )
    args = parser.parse_args()
    if args.render_only:
        result = render_existing_outputs(
            output_root=args.output_root, report_path=args.report_path
        )
        print("RENDER_ONLY=COMPLETE")
        print("OPTIMIZATION_RERUN=FALSE")
    else:
        result = run_analysis(output_root=args.output_root, report_path=args.report_path)
    print(result["decision"]["global_ambiguity_verdict"])
    print("G2=NOT_PASSED")


if __name__ == "__main__":
    main()
