"""Global ambiguity / cluster diagnostics for the G2 R2-vs-R3 study.

Deterministic complete-linkage clustering on full-range-scaled parameter
coordinates at the frozen cutoff 0.10 (committed G2 convention, labels in
source-row order), pairwise dispersion statistics over the near-equivalent
solution set, and basin classification.  The near-equivalent set is
predeclared: solutions whose repricing RMSE is at most
``max(1.05 x best, 2.5e-7)`` — the absolute clean-data convention and the
relative noisy-data convention unified without any post-hoc tuning.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..constants import PARAMETER_NAMES
from . import frozen


def complete_linkage_clusters(
    values: np.ndarray, cutoff: float = frozen.CLUSTER_DISTANCE_CUTOFF
) -> np.ndarray:
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
                complete = float(
                    distances[np.ix_(clusters[left], clusters[right])].max()
                )
                candidates.append((complete, left, right))
        distance, left, right = min(
            candidates,
            key=lambda item: (item[0], clusters[item[1]][0], clusters[item[2]][0]),
        )
        if distance > cutoff:
            break
        clusters[left] = sorted(clusters[left] + clusters[right])
        del clusters[right]
    labels = np.empty(len(points), dtype=int)
    for label, cluster in enumerate(
        sorted(clusters, key=lambda members: members[0]), start=1
    ):
        labels[cluster] = label
    return labels


def _scaled_matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame[[f"scaled_{name}" for name in PARAMETER_NAMES]].to_numpy(float)


def classify_basin(near: pd.DataFrame) -> dict[str, Any]:
    count = len(near)
    cluster_count = int(near["cluster_id"].nunique()) if count else 0
    pca1_fraction = math.nan
    if count >= 2:
        matrix = _scaled_matrix(near)
        singular = np.linalg.svd(matrix - matrix.mean(axis=0), compute_uv=False)
        denominator = float(np.sum(singular**2))
        pca1_fraction = (
            float(singular[0] ** 2 / denominator) if denominator else math.nan
        )
    if cluster_count >= 2:
        classification = "multiple_basin"
    elif (
        cluster_count == 1
        and count >= 5
        and np.isfinite(pca1_fraction)
        and pca1_fraction >= 0.75
    ):
        classification = "ridge_like"
    else:
        classification = "single_or_unresolved"
    return {
        "near_equivalent_count": count,
        "cluster_count": cluster_count,
        "pca1_fraction": None if math.isnan(pca1_fraction) else pca1_fraction,
        "basin_classification": classification,
    }


def dispersion_record(cell_runs: pd.DataFrame) -> dict[str, Any]:
    """Cluster + dispersion statistics for the twelve runs of one cell."""
    frame = cell_runs.reset_index(drop=True)
    finite = frame.loc[np.isfinite(frame["repricing_rmse"].to_numpy(float))]
    if finite.empty:
        return {
            "near_equivalent_count": 0,
            "cluster_count": 0,
            "median_pairwise_distance": float("nan"),
            "maximum_pairwise_distance": float("nan"),
            "maximum_distance_from_best": float("nan"),
            "materially_displaced_count": 0,
            "boundary_hit_rate": float("nan"),
            "optimizer_success_rate": float("nan"),
            "best_parameter_rmse_scaled": float("nan"),
            "best_repricing_rmse": float("nan"),
            "best_repricing_rmse_relative": float("nan"),
            "basin_classification": "no_finite_solution",
            "pca1_fraction": None,
        }
    best_index = finite["repricing_rmse"].idxmin()
    best_rmse = float(finite.loc[best_index, "repricing_rmse"])
    threshold = max(
        frozen.NEAR_EQUIVALENCE_RELATIVE_MARGIN * best_rmse,
        frozen.NEAR_PRICE_EQUIVALENCE_RMSE,
    )
    near = finite.loc[finite["repricing_rmse"] <= threshold].copy()
    if near.empty:  # defensive; best itself always qualifies
        near = finite.loc[[best_index]].copy()
    coordinates = _scaled_matrix(near)
    labels = complete_linkage_clusters(coordinates)
    near["cluster_id"] = labels
    if len(near) > 1:
        differences = coordinates[:, None, :] - coordinates[None, :, :]
        pairwise = np.linalg.norm(differences, axis=2)
        upper = pairwise[np.triu_indices(len(near), k=1)]
        median_pairwise = float(np.median(upper))
        maximum_pairwise = float(np.max(upper))
    else:
        median_pairwise = 0.0
        maximum_pairwise = 0.0
    best_row = near.loc[near["repricing_rmse"].idxmin()]
    best_scaled = np.asarray(
        [best_row[f"scaled_{name}"] for name in PARAMETER_NAMES], dtype=np.float64
    )
    from_best = np.linalg.norm(coordinates - best_scaled[None, :], axis=1)
    boundary_hits = near["boundary_reasons"].fillna("").astype(str).str.len().gt(0)
    record = {
        "near_equivalent_count": int(len(near)),
        "cluster_count": int(labels.max()),
        "median_pairwise_distance": median_pairwise,
        "maximum_pairwise_distance": maximum_pairwise,
        "maximum_distance_from_best": float(np.max(from_best)) if len(from_best) else 0.0,
        "materially_displaced_count": int(
            np.sum(from_best >= frozen.MATERIAL_DISPLACEMENT_RMSE)
        ),
        "boundary_hit_rate": float(boundary_hits.mean()),
        "optimizer_success_rate": float(frame["success"].mean()),
        "best_parameter_rmse_scaled": float(
            finite.loc[best_index, "parameter_rmse_scaled"]
        ),
        "best_repricing_rmse": best_rmse,
        "best_repricing_rmse_relative": float(
            finite.loc[best_index, "repricing_rmse_relative"]
        ),
    }
    record.update(classify_basin(near))
    return record


def aggregate_dispersion(
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-cell dispersion records into repr-level summary metrics."""
    frame = pd.DataFrame(records)
    if frame.empty:
        raise RuntimeError("no dispersion records to aggregate")
    return {
        "median_of_median_pairwise": float(frame["median_pairwise_distance"].median()),
        "median_of_maximum_pairwise": float(frame["maximum_pairwise_distance"].median()),
        "maximum_of_maximum_pairwise": float(frame["maximum_pairwise_distance"].max()),
        "median_cluster_count": float(frame["cluster_count"].median()),
        "mean_cluster_count": float(frame["cluster_count"].mean()),
        "median_boundary_hit_rate": float(frame["boundary_hit_rate"].median()),
        "median_best_parameter_rmse_scaled": float(
            frame["best_parameter_rmse_scaled"].median()
        ),
        "median_best_repricing_rmse_relative": float(
            frame["best_repricing_rmse_relative"].median()
        ),
        "cells_with_multiple_basins": int(
            (frame["basin_classification"] == "multiple_basin").sum()
        ),
        "cell_count": int(len(frame)),
    }
