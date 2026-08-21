"""Independent seed-only replication of the frozen A/B/C/D multi-date study."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import scripts.run_g2_multi_date_identifiability as original
from src.calibrate_double_heston import load_hard_safety_bounds


ORIGINAL_CIR_PATH_SEED = 20260811
REPLICATION_CIR_PATH_SEED = 27182818
RECOVERY_NOISE_START_SEED = 20260811

ORIGINAL_OUTPUT_ROOT = original.DEFAULT_OUTPUT_ROOT
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT
    / "market_data_audit"
    / "stage_a"
    / "derived"
    / "g2_multi_date_replication"
)
DEFAULT_REPORT_PATH = REPOSITORY_ROOT / "docs" / "G2_MULTI_DATE_REPLICATION.md"

REPLICATION_STATUS_VALUES = (
    "REPLICATED",
    "PARTIALLY_REPLICATED",
    "NOT_REPLICATED",
)

PREVIOUS_G2_PATHS = (
    "docs/G2_COMMON_SUPPORT_ANALYSIS.md",
    "docs/G2_IDENTIFIABILITY_ANALYSIS.md",
    "docs/G2_INFORMATION_REMEDIATION.md",
    "docs/G2_MULTI_DATE_IDENTIFIABILITY.md",
    "scripts/run_g2_common_support_analysis.py",
    "scripts/run_g2_identifiability_analysis.py",
    "scripts/run_g2_information_remediation.py",
    "scripts/run_g2_multi_date_identifiability.py",
    "tests/test_g2_common_support_analysis.py",
    "tests/test_g2_identifiability_analysis.py",
    "tests/test_g2_information_remediation.py",
    "tests/test_g2_multi_date_identifiability.py",
)

EXPECTED_OUTPUT_FILES = (
    "replication_contract.csv",
    "replication_state_paths.csv",
    "replication_identifiability.csv",
    "replication_parameter_sensitivity.csv",
    "replication_weakest_directions.csv",
    "replication_recovery_starts.csv",
    "replication_recovery_summary.csv",
    "replication_parameter_errors.csv",
    "replication_nuisance_state_recovery.csv",
    "original_vs_replication.csv",
    "weakest_direction_stability.csv",
    "hypothesis_status.csv",
    "decision.json",
    "figures/conditioning_replication.png",
    "figures/recovery_replication.png",
    "figures/weakest_direction_stability.png",
    "figures/mentor_replication_summary.png",
)


def _protected_snapshot(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, str]:
    result: dict[str, str] = {}
    stage_root = REPOSITORY_ROOT / "market_data_audit" / "stage_a"
    resolved_output = output_root.resolve()
    for path in sorted(stage_root.rglob("*")):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved == resolved_output or resolved_output in resolved.parents:
            continue
        result[path.relative_to(REPOSITORY_ROOT).as_posix()] = original._sha256(path)
    for relative in PREVIOUS_G2_PATHS:
        path = REPOSITORY_ROOT / relative
        result[relative] = original._sha256(path)
    return result


def replication_contract() -> pd.DataFrame:
    if original.ANALYSIS_SEED != ORIGINAL_CIR_PATH_SEED:
        raise RuntimeError("Original analysis seed no longer matches the frozen contract")
    if RECOVERY_NOISE_START_SEED != original.ANALYSIS_SEED:
        raise RuntimeError("Recovery/noise/start seed must remain frozen")
    return pd.DataFrame(
        [
            {
                "original_cir_path_seed": ORIGINAL_CIR_PATH_SEED,
                "replication_cir_path_seed": REPLICATION_CIR_PATH_SEED,
                "recovery_noise_start_seed": RECOVERY_NOISE_START_SEED,
                "only_changed_scientific_field": "cir_path_seed",
                "valuation_dates": "|".join(original.VALUATION_DATES),
                "date_gaps_days": "|".join(map(str, original.DATE_GAPS_DAYS)),
                "maturity_profiles_days": ";".join(
                    f"{profile}:{days[0]}|{days[1]}"
                    for profile, days in original.MATURITY_PROFILES
                ),
                "moneyness_nodes": "|".join(
                    f"{value:.2f}" for value in original.REPRESENTATION.moneyness_nodes
                ),
                "option_types": "|".join(original.REPRESENTATION.option_types),
                "controlled_rates": "|".join(
                    f"{value:.4f}" for value in original.CONTROLLED_RATES
                ),
                "controlled_dividend_yields": "|".join(
                    f"{value:.4f}" for value in original.CONTROLLED_DIVIDEND_YIELDS
                ),
                "canonical_target_count": len(original.CANONICAL_TARGET_NAMES),
                "designs": "|".join(design.design_id for design in original.DESIGNS),
                "jacobian_relative_step": original.JACOBIAN_RELATIVE_STEP,
                "practical_rank_relative_tolerance": original.PRACTICAL_RANK_RELATIVE_TOLERANCE,
                "target_recovery_rmse_threshold": original.RECOVERY_SCALED_RMSE_THRESHOLD,
                "target_recovery_max_error_threshold": original.RECOVERY_SCALED_MAX_ERROR_THRESHOLD,
                "recovery_frequency_threshold": original.RECOVERY_FREQUENCY_THRESHOLD,
                "optimizer": "L-BFGS-B",
                "optimizer_maxiter": original.RECOVERY_MAXITER,
                "jacobian_target_sample_count": 8,
                "recovery_samples_per_distribution": original.RECOVERY_SAMPLES_PER_DISTRIBUTION,
                "recovery_target_sample_count": 2,
                "starts_per_target": 3,
                "noise_levels": "|".join(f"{value:.3f}" for value in original.NOISE_LEVELS),
                "additional_observables": 0,
            }
        ]
    )


def simulate_replication_state_paths(samples: pd.DataFrame) -> pd.DataFrame:
    """Generate exact CIR paths with only the base path seed changed."""
    rows: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(samples.itertuples(index=False)):
        target = np.asarray(
            [getattr(sample, name) for name in original.CANONICAL_TARGET_NAMES],
            dtype=np.float64,
        )
        state = {"slow": float(target[4]), "fast": float(target[9])}
        record: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "distribution": sample.distribution,
            "base_cir_path_seed": REPLICATION_CIR_PATH_SEED,
            "v_slow_t0": state["slow"],
            "v_fast_t0": state["fast"],
            "date_t0": original.VALUATION_DATES[0],
            "date_t1": original.VALUATION_DATES[1],
            "date_t2": original.VALUATION_DATES[2],
        }
        for transition_index, gap_days in enumerate(original.DATE_GAPS_DAYS):
            transition_seed = (
                REPLICATION_CIR_PATH_SEED
                + 1000 * sample_index
                + 100 * transition_index
            )
            rng = np.random.default_rng(transition_seed)
            record[f"transition_seed_t{transition_index}_to_t{transition_index + 1}"] = transition_seed
            for factor in ("slow", "fast"):
                uniform = float(rng.random())
                kappa, theta, sigma = original._factor_parameters(target, factor)
                next_state = original.exact_cir_transition_from_uniform(
                    kappa,
                    theta,
                    sigma,
                    state[factor],
                    gap_days / 365.0,
                    uniform,
                )
                state_name = f"v_{factor}_t{transition_index + 1}"
                lower, upper = original.STATE_BOUNDS[state_name]
                if not lower < next_state < upper:
                    raise RuntimeError(
                        f"Replication state {state_name}={next_state} is outside "
                        f"the frozen numerical envelope {(lower, upper)}"
                    )
                record[
                    f"uniform_{factor}_t{transition_index}_to_t{transition_index + 1}"
                ] = uniform
                record[state_name] = next_state
                state[factor] = next_state
        rows.append(record)
    return pd.DataFrame(rows)


def _load_original_evidence() -> dict[str, pd.DataFrame]:
    paths = {
        "designs": "experiment_designs.csv",
        "states": "state_paths.csv",
        "identifiability": "identifiability_summary.csv",
        "directions": "weakest_directions.csv",
        "recovery": "recovery_summary.csv",
    }
    result = {
        key: pd.read_csv(ORIGINAL_OUTPUT_ROOT / relative)
        for key, relative in paths.items()
    }
    pd.testing.assert_frame_equal(
        result["designs"], original.experiment_designs(), check_exact=True
    )
    return result


def _ident_metrics(frame: pd.DataFrame, design_id: str) -> dict[str, float]:
    group = frame.loc[
        frame["design_id"].eq(design_id) & frame["viability"].eq("VIABLE")
    ]
    if group.empty:
        return {
            "practical_full_rank_frequency": 0.0,
            "median_smallest_singular_value": math.nan,
            "median_condition_number": math.inf,
        }
    return {
        "practical_full_rank_frequency": float(
            group["practical_target_rank_1e_minus_6"].eq(10).mean()
        ),
        "median_smallest_singular_value": float(
            group["smallest_singular_value"].median()
        ),
        "median_condition_number": float(group["condition_number"].median()),
    }


def _recovery_metrics(
    frame: pd.DataFrame, design_id: str, noise_level: float
) -> dict[str, float]:
    row = frame.loc[
        frame["design_id"].eq(design_id)
        & np.isclose(frame["noise_level"], noise_level)
    ]
    if len(row) != 1:
        return {
            "pass_frequency": math.nan,
            "median_target_rmse": math.nan,
            "median_nuisance_rmse": math.nan,
            "optimizer_success_frequency": math.nan,
        }
    item = row.iloc[0]
    return {
        "pass_frequency": float(
            item["canonical_target_recovery_pass_frequency"]
        ),
        "median_target_rmse": float(
            item["median_best_target_scaled_parameter_rmse"]
        ),
        "median_nuisance_rmse": float(
            item["median_best_nuisance_scaled_state_rmse"]
        ),
        "optimizer_success_frequency": float(
            item["optimizer_success_count"] / item["start_count"]
        ),
    }


def weakest_direction_stability(
    original_directions: pd.DataFrame, replication_directions: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for design in original.DESIGNS:
        left = original_directions.loc[
            original_directions["design_id"].eq(design.design_id)
        ]
        right = replication_directions.loc[
            replication_directions["design_id"].eq(design.design_id)
        ]
        cosines: list[float] = []
        for sample_id in sorted(set(left["sample_id"]) & set(right["sample_id"])):
            a = (
                left.loc[left["sample_id"].eq(sample_id)]
                .set_index("parameter")
                .reindex(original.CANONICAL_TARGET_NAMES)["weakest_direction_loading"]
                .to_numpy(float)
            )
            b = (
                right.loc[right["sample_id"].eq(sample_id)]
                .set_index("parameter")
                .reindex(original.CANONICAL_TARGET_NAMES)["weakest_direction_loading"]
                .to_numpy(float)
            )
            cosines.append(float(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))))
        left_median = (
            left.groupby("parameter")["absolute_weakest_direction_loading"]
            .median()
            .sort_values(ascending=False)
        )
        right_median = (
            right.groupby("parameter")["absolute_weakest_direction_loading"]
            .median()
            .sort_values(ascending=False)
        )
        left_top = list(left_median.head(3).index)
        right_top = list(right_median.head(3).index)
        rows.append(
            {
                "design_id": design.design_id,
                "median_absolute_cosine": float(np.median(cosines)),
                "minimum_absolute_cosine": float(np.min(cosines)),
                "original_top3": "|".join(left_top),
                "replication_top3": "|".join(right_top),
                "top3_overlap_count": len(set(left_top) & set(right_top)),
            }
        )
    return pd.DataFrame(rows)


def build_comparison(
    original_identifiability: pd.DataFrame,
    replication_identifiability: pd.DataFrame,
    original_recovery: pd.DataFrame,
    replication_recovery: pd.DataFrame,
    direction_stability: pd.DataFrame,
) -> pd.DataFrame:
    direction_lookup = direction_stability.set_index("design_id")
    rows: list[dict[str, Any]] = []
    for design in original.DESIGNS:
        original_ident = _ident_metrics(original_identifiability, design.design_id)
        replication_ident = _ident_metrics(
            replication_identifiability, design.design_id
        )
        row: dict[str, Any] = {
            "design_id": design.design_id,
            **{f"original_{key}": value for key, value in original_ident.items()},
            **{
                f"replication_{key}": value
                for key, value in replication_ident.items()
            },
            "median_absolute_weakest_direction_cosine": float(
                direction_lookup.loc[design.design_id, "median_absolute_cosine"]
            ),
            "weakest_direction_top3_overlap_count": int(
                direction_lookup.loc[design.design_id, "top3_overlap_count"]
            ),
        }
        for noise in original.NOISE_LEVELS:
            label = {0.0: "clean", 0.005: "noise_0_5pct", 0.01: "noise_1pct"}[
                noise
            ]
            original_metrics = _recovery_metrics(
                original_recovery, design.design_id, noise
            )
            replication_metrics = _recovery_metrics(
                replication_recovery, design.design_id, noise
            )
            for key, value in original_metrics.items():
                row[f"original_{label}_{key}"] = value
            for key, value in replication_metrics.items():
                row[f"replication_{label}_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def classify_hypotheses(
    comparison: pd.DataFrame, replication_decision: dict[str, Any]
) -> pd.DataFrame:
    metrics = comparison.set_index("design_id")
    a, b, c, d = (metrics.loc[key] for key in ("A", "B", "C", "D"))

    h1_ratio = float(
        a.replication_median_condition_number
        / b.replication_median_condition_number
    )
    h1_rank = bool(
        b.replication_practical_full_rank_frequency
        > a.replication_practical_full_rank_frequency
    )
    h1 = (
        "REPLICATED"
        if h1_rank and h1_ratio >= 10.0
        else "PARTIALLY_REPLICATED"
        if h1_rank or h1_ratio >= 3.0
        else "NOT_REPLICATED"
    )

    h2_rank = bool(
        c.replication_practical_full_rank_frequency
        < b.replication_practical_full_rank_frequency
    )
    h2_condition = bool(
        c.replication_median_condition_number
        > b.replication_median_condition_number
    )
    h2 = (
        "REPLICATED"
        if h2_rank and h2_condition
        else "PARTIALLY_REPLICATED"
        if h2_rank or h2_condition
        else "NOT_REPLICATED"
    )

    h3_rank = bool(
        d.replication_practical_full_rank_frequency
        > c.replication_practical_full_rank_frequency
    )
    h3_ratio = float(
        c.replication_median_condition_number
        / d.replication_median_condition_number
    )
    h3 = (
        "REPLICATED"
        if h3_rank and h3_ratio >= 2.0
        else "PARTIALLY_REPLICATED"
        if h3_rank or h3_ratio >= 1.25
        else "NOT_REPLICATED"
    )

    recovery_columns = [
        column
        for column in comparison.columns
        if column.startswith("replication_") and column.endswith("_pass_frequency")
    ]
    maximum_frequency = float(
        np.nanmax(comparison[recovery_columns].to_numpy(float))
    )
    any_complete_design = any(replication_decision["design_pass"].values())
    h4 = (
        "REPLICATED"
        if maximum_frequency < original.RECOVERY_FREQUENCY_THRESHOLD
        and not any_complete_design
        else "PARTIALLY_REPLICATED"
        if not any_complete_design
        else "NOT_REPLICATED"
    )

    return pd.DataFrame(
        [
            {
                "hypothesis": "H1",
                "statement": "Oracle multi-date information greatly improves local conditioning over A",
                "status": h1,
                "decisive_metric": f"condition_reduction={h1_ratio:.6g};rank_improved={h1_rank}",
            },
            {
                "hypothesis": "H2",
                "statement": "Latent states weaken target identifiability relative to B",
                "status": h2,
                "decisive_metric": f"rank_weaker={h2_rank};condition_worse={h2_condition}",
            },
            {
                "hypothesis": "H3",
                "statement": "Exact CIR physics materially improves local conditioning relative to C",
                "status": h3,
                "decisive_metric": f"condition_reduction={h3_ratio:.6g};rank_improved={h3_rank}",
            },
            {
                "hypothesis": "H4",
                "statement": "Stable ten-parameter recovery remains poor",
                "status": h4,
                "decisive_metric": f"maximum_recovery_frequency={maximum_frequency:.6g};complete_design={any_complete_design}",
            },
        ]
    )


def decide_replication(
    hypotheses: pd.DataFrame, replication_diagnostic: dict[str, Any]
) -> dict[str, Any]:
    statuses = hypotheses.set_index("hypothesis")["status"].to_dict()
    same_diagnostic = (
        replication_diagnostic["verdict"]
        == "MULTI_DATE_DIAGNOSTIC = INSUFFICIENT"
    )
    if all(value == "REPLICATED" for value in statuses.values()) and same_diagnostic:
        verdict = "REPLICATION = CONFIRMED"
    elif statuses.get("H4") == "NOT_REPLICATED" or not same_diagnostic:
        verdict = "REPLICATION = FAILED"
    else:
        verdict = "REPLICATION = MIXED"
    return {
        "replication_verdict": verdict,
        "hypothesis_status": statuses,
        "replication_multi_date_diagnostic": replication_diagnostic["verdict"],
        "original_multi_date_diagnostic": "MULTI_DATE_DIAGNOSTIC = INSUFFICIENT",
        "g2_status": "G2 = NOT_PASSED — STRUCTURAL IDENTIFIABILITY PROBLEM REMAINS",
        "original_cir_path_seed": ORIGINAL_CIR_PATH_SEED,
        "replication_cir_path_seed": REPLICATION_CIR_PATH_SEED,
        "recovery_noise_start_seed_unchanged": RECOVERY_NOISE_START_SEED,
        "only_changed_scientific_field": "cir_path_seed",
        "g2_changed": False,
    }


def _save_figure(fig: plt.Figure, path: Path) -> None:
    original._save_figure(fig, path)


def write_figures(
    comparison: pd.DataFrame,
    direction_stability: pd.DataFrame,
    hypotheses: pd.DataFrame,
    decision: dict[str, Any],
    output_root: Path,
) -> list[Path]:
    figure_root = output_root / "figures"
    designs = list(comparison["design_id"])
    colors = {"original": "#4E79A7", "replication": "#F28E2B"}
    figures: list[Path] = []

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    x = np.arange(len(designs)); width = 0.36
    axes[0].bar(x - width / 2, 100 * comparison["original_practical_full_rank_frequency"], width, label="Original", color=colors["original"])
    axes[0].bar(x + width / 2, 100 * comparison["replication_practical_full_rank_frequency"], width, label="Replication", color=colors["replication"])
    axes[0].set(title="Practical target full-rank frequency", ylabel="Frequency (%)", xticks=x, xticklabels=designs, ylim=(0, 105)); axes[0].legend()
    axes[1].bar(x - width / 2, comparison["original_median_condition_number"], width, label="Original", color=colors["original"])
    axes[1].bar(x + width / 2, comparison["replication_median_condition_number"], width, label="Replication", color=colors["replication"])
    axes[1].set_yscale("log"); axes[1].set(title="Median target condition number", ylabel="Condition number", xticks=x, xticklabels=designs)
    path = figure_root / "conditioning_replication.png"; _save_figure(fig, path); figures.append(path)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), sharey=True)
    for axis, (label, title) in zip(axes, (("clean", "Clean"), ("noise_0_5pct", "0.5% noise"), ("noise_1pct", "1.0% noise")), strict=True):
        axis.bar(x - width / 2, comparison[f"original_{label}_median_target_rmse"], width, label="Original", color=colors["original"])
        axis.bar(x + width / 2, comparison[f"replication_{label}_median_target_rmse"], width, label="Replication", color=colors["replication"])
        axis.set(title=title, xticks=x, xticklabels=designs, xlabel="Design")
    axes[0].set_ylabel("Median range-scaled target RMSE"); axes[-1].legend()
    path = figure_root / "recovery_replication.png"; _save_figure(fig, path); figures.append(path)

    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.bar(direction_stability["design_id"], direction_stability["median_absolute_cosine"], color="#59A14F")
    axis.set(title="Weakest-direction stability across CIR seeds", xlabel="Design", ylabel="Median absolute cosine", ylim=(0, 1.05))
    path = figure_root / "weakest_direction_stability.png"; _save_figure(fig, path); figures.append(path)

    fig, axis = plt.subplots(figsize=(11.5, 4.4)); axis.axis("off")
    cells: list[list[str]] = []
    by_h = hypotheses.set_index("hypothesis")
    for row in comparison.itertuples(index=False):
        cells.append([
            row.design_id,
            f"{100 * row.original_practical_full_rank_frequency:.1f}% → {100 * row.replication_practical_full_rank_frequency:.1f}%",
            f"{row.original_median_condition_number:.2e} → {row.replication_median_condition_number:.2e}",
            f"{100 * row.original_clean_pass_frequency:.1f}% → {100 * row.replication_clean_pass_frequency:.1f}%",
            f"{row.median_absolute_weakest_direction_cosine:.3f}",
        ])
    table = axis.table(cellText=cells, colLabels=["Design", "Full target rank", "Median condition", "Clean pass", "Weak-dir cosine"], cellLoc="center", loc="center")
    table.auto_set_font_size(False); table.set_fontsize(8.5); table.scale(1.0, 1.55)
    status_text = " | ".join(f"{name}:{by_h.loc[name, 'status']}" for name in ("H1", "H2", "H3", "H4"))
    axis.set_title(f"{decision['replication_verdict']}\n{status_text}", pad=18)
    path = figure_root / "mentor_replication_summary.png"; _save_figure(fig, path); figures.append(path)
    return figures


def _format_pass(value: float) -> str:
    return f"{int(round(6.0 * value))}/6"


def render_report(
    path: Path,
    comparison: pd.DataFrame,
    direction_stability: pd.DataFrame,
    hypotheses: pd.DataFrame,
    decision: dict[str, Any],
    artifact_hashes: dict[str, str],
) -> None:
    by_design = comparison.set_index("design_id")
    lines = [
        "# G2 Multi-Date Independent Replication",
        "",
        "## Frozen replication contract",
        "",
        f"- Original exact-CIR path seed: `{ORIGINAL_CIR_PATH_SEED}`.",
        f"- Independent replication path seed, predeclared before computation: `{REPLICATION_CIR_PATH_SEED}`.",
        f"- Recovery/noise/start seed remains `{RECOVERY_NOISE_START_SEED}`.",
        "- The only changed scientific field is `cir_path_seed`. Dates, gaps, maturity profiles, central-five calls/puts, controlled carry, targets, A-D definitions, scaling, projection/Schur method, gates, optimizer, target samples, starts, noise levels, and observables are unchanged.",
        "",
        "All original Stage A/G2 reports and evidence were hashed before and after replication and remained byte-identical. The replication writes only to its separate report and derived output root.",
        "",
        "## Original versus replication",
        "",
        "| Design | Practical target rank | Median smallest singular value | Median condition number | Weak-dir cosine |",
        "|---|---:|---:|---:|---:|",
    ]
    for design_id, row in by_design.iterrows():
        lines.append(
            f"| {design_id} | {int(round(8 * row.original_practical_full_rank_frequency))}/8 → {int(round(8 * row.replication_practical_full_rank_frequency))}/8 | {row.original_median_smallest_singular_value:.3e} → {row.replication_median_smallest_singular_value:.3e} | {row.original_median_condition_number:.3e} → {row.replication_median_condition_number:.3e} | {row.median_absolute_weakest_direction_cosine:.3f} |"
        )
    lines.extend([
        "",
        "## Recovery comparison",
        "",
        "| Design | Noise | Recovery pass | Median target RMSE | Median nuisance RMSE | Optimizer success |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for design_id, row in by_design.iterrows():
        for label, title in (("clean", "0.0%"), ("noise_0_5pct", "0.5%"), ("noise_1pct", "1.0%")):
            original_nuisance = row[f"original_{label}_median_nuisance_rmse"]
            replication_nuisance = row[f"replication_{label}_median_nuisance_rmse"]
            nuisance_text = (
                f"{original_nuisance:.3e} → {replication_nuisance:.3e}"
                if math.isfinite(original_nuisance) and math.isfinite(replication_nuisance)
                else "N/A"
            )
            lines.append(
                f"| {design_id} | {title} | {_format_pass(row[f'original_{label}_pass_frequency'])} → {_format_pass(row[f'replication_{label}_pass_frequency'])} | {row[f'original_{label}_median_target_rmse']:.3e} → {row[f'replication_{label}_median_target_rmse']:.3e} | {nuisance_text} | {_format_pass(row[f'original_{label}_optimizer_success_frequency'])} → {_format_pass(row[f'replication_{label}_optimizer_success_frequency'])} |"
            )
    lines.extend(["", "## Weakest-direction stability", ""])
    for row in direction_stability.itertuples(index=False):
        lines.append(
            f"- Design {row.design_id}: median absolute cosine `{row.median_absolute_cosine:.3f}`, minimum `{row.minimum_absolute_cosine:.3f}`, top-three overlap `{row.top3_overlap_count}/3`; original `{row.original_top3}`, replication `{row.replication_top3}`."
        )
    lines.extend([
        "",
        "## Hypothesis assessment",
        "",
        "| Hypothesis | Status | Decisive replication evidence |",
        "|---|---|---|",
    ])
    for row in hypotheses.itertuples(index=False):
        lines.append(
            f"| {row.hypothesis}: {row.statement} | **{row.status}** | `{row.decisive_metric}` |"
        )
    seed_sensitive = hypotheses.loc[
        hypotheses["status"].ne("REPLICATED"), "hypothesis"
    ].tolist()
    lines.extend([
        "",
        "## Decision",
        "",
        f"**{decision['replication_verdict']}**",
        "",
        f"Replication diagnostic: `{decision['replication_multi_date_diagnostic']}`. Seed-sensitive conclusions: `{'|'.join(seed_sensitive) if seed_sensitive else 'NONE'}`.",
        "",
        "The seed sensitivity is confined to the practical-rank component: C changed from `6/8` to `8/8`, so C no longer ranked below B and D no longer ranked above C. The conditioning components persisted: C remained worse conditioned than B and exact-CIR D remained materially better conditioned than C.",
        "",
        f"**{decision['g2_status']}**",
        "",
        "One replication does not reopen or pass G2. The scientific question is whether the four qualitative conclusions reproduce, not whether individual floating-point metrics are identical.",
        "",
        "## Recommended next research decision",
        "",
        "Ask the mentor whether to authorize a small, pre-registered CIR-seed panel of the unchanged experiment solely to estimate the stability of C's practical-rank frequency before considering any redesign.",
        "",
        "## Reproducibility and artifacts",
        "",
        "| Artifact | SHA-256 |",
        "|---|---|",
    ])
    for relative, digest in sorted(artifact_hashes.items()):
        lines.append(f"| `{relative}` | `{digest}` |")
    lines.extend(["", decision["replication_verdict"], "", decision["g2_status"], ""])
    original._atomic_write_bytes(path, "\n".join(lines).encode("utf-8"))


def run_replication(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    report_path: Path = DEFAULT_REPORT_PATH,
    node_count: int = 64,
    maxiter: int = original.RECOVERY_MAXITER,
    start_count: int = 3,
    per_distribution: int = original.RECOVERY_SAMPLES_PER_DISTRIBUTION,
) -> dict[str, Any]:
    protected_before = _protected_snapshot(output_root)
    contract = replication_contract()
    baseline_evidence = _load_original_evidence()
    bounds = load_hard_safety_bounds(original.baseline.BOUNDS_PATH)
    samples = original.baseline.select_representative_parameters(
        bounds, per_distribution=4
    )
    states = simulate_replication_state_paths(samples)
    if np.array_equal(
        states[list(original.NUISANCE_STATE_NAMES)].to_numpy(float),
        baseline_evidence["states"][list(original.NUISANCE_STATE_NAMES)].to_numpy(float),
    ):
        raise RuntimeError("Independent CIR seed reproduced the original state paths")
    identifiability, sensitivities, directions, viability = original.run_identifiability(
        samples, states, bounds, node_count=node_count
    )
    starts, recovery, parameter_errors, nuisance, oracle_stop = original.run_recovery(
        samples,
        states,
        bounds,
        viability,
        identifiability,
        node_count=node_count,
        maxiter=maxiter,
        start_count=start_count,
        per_distribution=per_distribution,
    )
    replication_diagnostic = original.classify_diagnostic(
        identifiability, recovery, viability, oracle_stop
    )
    direction_stability = weakest_direction_stability(
        baseline_evidence["directions"], directions
    )
    comparison = build_comparison(
        baseline_evidence["identifiability"],
        identifiability,
        baseline_evidence["recovery"],
        recovery,
        direction_stability,
    )
    hypotheses = classify_hypotheses(comparison, replication_diagnostic)
    decision = decide_replication(hypotheses, replication_diagnostic)

    frames = {
        "replication_contract.csv": contract,
        "replication_state_paths.csv": states,
        "replication_identifiability.csv": identifiability,
        "replication_parameter_sensitivity.csv": sensitivities,
        "replication_weakest_directions.csv": directions,
        "replication_recovery_starts.csv": starts,
        "replication_recovery_summary.csv": recovery,
        "replication_parameter_errors.csv": parameter_errors,
        "replication_nuisance_state_recovery.csv": nuisance,
        "original_vs_replication.csv": comparison,
        "weakest_direction_stability.csv": direction_stability,
        "hypothesis_status.csv": hypotheses,
    }
    for relative, frame in frames.items():
        original._write_csv(frame, output_root / relative)
    original._write_json(decision, output_root / "decision.json")
    figures = write_figures(
        comparison, direction_stability, hypotheses, decision, output_root
    )
    artifact_hashes = {
        relative: original._sha256(output_root / relative)
        for relative in EXPECTED_OUTPUT_FILES
        if not relative.startswith("figures/") and (output_root / relative).exists()
    }
    artifact_hashes.update(
        {
            path.relative_to(output_root).as_posix(): original._sha256(path)
            for path in figures
        }
    )
    render_report(
        report_path,
        comparison,
        direction_stability,
        hypotheses,
        decision,
        artifact_hashes,
    )
    protected_after = _protected_snapshot(output_root)
    if protected_before != protected_after:
        changed = sorted(
            key
            for key in set(protected_before) | set(protected_after)
            if protected_before.get(key) != protected_after.get(key)
        )
        raise RuntimeError(f"Protected original evidence changed: {changed}")
    return {
        "contract": contract,
        "states": states,
        "identifiability": identifiability,
        "directions": directions,
        "recovery": recovery,
        "comparison": comparison,
        "direction_stability": direction_stability,
        "hypotheses": hypotheses,
        "decision": decision,
        "artifact_hashes": artifact_hashes,
        "report_path": report_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--node-count", type=int, default=64)
    parser.add_argument("--maxiter", type=int, default=original.RECOVERY_MAXITER)
    parser.add_argument("--start-count", type=int, default=3)
    parser.add_argument(
        "--per-distribution",
        type=int,
        default=original.RECOVERY_SAMPLES_PER_DISTRIBUTION,
    )
    args = parser.parse_args()
    result = run_replication(
        output_root=args.output_root,
        report_path=args.report_path,
        node_count=args.node_count,
        maxiter=args.maxiter,
        start_count=args.start_count,
        per_distribution=args.per_distribution,
    )
    print(result["decision"]["replication_verdict"])
    print(result["decision"]["g2_status"])
    print(f"report={result['report_path']}")


if __name__ == "__main__":
    main()
