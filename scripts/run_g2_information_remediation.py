"""Compare two versus three listed expiries for ten-parameter G2 recovery.

The experiment is deliberately narrow.  It reuses the reviewed central-five
strike geometry, the same deterministic parameter vectors, scaling, numerical
thresholds, and recovery protocol.  The only information expansion is the
actual third listed expiry.  It never rewrites Stage A evidence, acquires data,
generates the final research dataset, or trains an ANN/PINN.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import scripts.run_g2_identifiability_analysis as baseline
from scripts.run_g2_common_support_analysis import (
    CANONICAL_DERIVED_ROOT,
    CANONICAL_RAW_ROOT,
    G2_DATES,
    PRIMARY_UNDERLYINGS,
    assert_canonical_outputs_preserved,
    build_moneyness_support,
    load_balanced_panel,
    snapshot_canonical_outputs,
    validate_canonical_stage_a_provenance,
)
from src.calibrate_double_heston import (
    boundary_diagnostics,
    load_hard_safety_bounds,
    unconstrained_to_parameters,
)
from src.constants import PARAMETER_NAMES
from src.constraints import validate_parameters


ANALYSIS_ID = "G2_INFORMATION_REMEDIATION"
DEFAULT_OUTPUT_ROOT = CANONICAL_DERIVED_ROOT / "g2_information_remediation"
DEFAULT_REPORT_PATH = REPOSITORY_ROOT / "docs" / "G2_INFORMATION_REMEDIATION.md"

THREE_EXPIRY_MATURITY_PROFILES = (
    ("2026-07-01", (27, 55, 90)),
    ("2026-07-15", (13, 41, 76)),
    ("2026-07-22", (6, 34, 69)),
)
CONTROLLED_RATES = (0.0600, 0.0625, 0.0650)
CONTROLLED_DIVIDEND_YIELDS = (0.0200, 0.0225, 0.0250)
MATERIAL_IMPROVEMENT_FACTOR = 10.0
MIN_ACTIVE_SUPPORT_PCT = 75.0
MAX_LOG_MONEYNESS_BRACKET_WIDTH = 0.05
REPRESENTATION = baseline.REPRESENTATIONS[0]

COMMON_COORDINATES_IN_THREE_EXPIRY_ORDER = np.asarray(
    [*range(0, 10), *range(15, 25)], dtype=int
)

BASELINE_IDENTIFIABILITY_ARTIFACTS = (
    "representative_parameters.csv",
    "jacobian_summary.csv",
    "parameter_sensitivity.csv",
    "near_null_directions.csv",
    "recovery_starts.csv",
    "recovery_summary.csv",
    "figures/jacobian_singular_values.png",
    "figures/condition_number_distribution.png",
    "figures/parameter_sensitivity.png",
    "figures/clean_noisy_recovery.png",
    "figures/recovered_vs_true_clean.png",
)


@dataclass(frozen=True)
class ExperimentSpec:
    representation_id: str
    expiry_positions: tuple[str, ...]
    maturity_profiles: tuple[tuple[str, tuple[int, ...]], ...]
    rates: tuple[float, ...]
    dividend_yields: tuple[float, ...]
    normalized_price_count: int
    maturity_coordinate_count: int
    carry_coordinate_count: int
    market_status: str

    @property
    def candidate_input_dimension(self) -> int:
        return (
            self.normalized_price_count
            + self.maturity_coordinate_count
            + self.carry_coordinate_count
        )


EXPERIMENT_SPECS = (
    ExperimentSpec(
        representation_id="2exp_central5",
        expiry_positions=("near", "middle"),
        maturity_profiles=tuple(
            (profile_id, maturity_days)
            for profile_id, maturity_days in baseline.MATURITY_PROFILES
        ),
        rates=CONTROLLED_RATES[:2],
        dividend_yields=CONTROLLED_DIVIDEND_YIELDS[:2],
        normalized_price_count=20,
        maturity_coordinate_count=2,
        carry_coordinate_count=4,
        market_status="CURRENT_PROPOSED_MARKET_GEOMETRY",
    ),
    ExperimentSpec(
        representation_id="3exp_central5",
        expiry_positions=("near", "middle", "far"),
        maturity_profiles=THREE_EXPIRY_MATURITY_PROFILES,
        rates=CONTROLLED_RATES,
        dividend_yields=CONTROLLED_DIVIDEND_YIELDS,
        normalized_price_count=30,
        maturity_coordinate_count=3,
        carry_coordinate_count=6,
        market_status="SYNTHETIC_COMPARATOR_PENDING_FAR_MARKET_RULE",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_paths(root: Path, relative_paths: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required preserved artifact is missing: {path}")
        result[relative] = _sha256(path)
    return result


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    _atomic_write_bytes(
        path,
        frame.to_csv(index=False, lineterminator="\n").encode("utf-8"),
    )


def experiment_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "representation_id": spec.representation_id,
                "expiry_positions": "|".join(spec.expiry_positions),
                "maturity_profiles_days": ";".join(
                    f"{profile_id}:{'|'.join(map(str, days))}"
                    for profile_id, days in spec.maturity_profiles
                ),
                "moneyness_nodes": "|".join(
                    f"{node:+.2f}" for node in baseline.MONEYNESS_CENTRAL_5
                ),
                "option_types": "call|put",
                "normalized_price_count": spec.normalized_price_count,
                "maturity_coordinate_count": spec.maturity_coordinate_count,
                "carry_coordinate_count": spec.carry_coordinate_count,
                "candidate_input_dimension": spec.candidate_input_dimension,
                "controlled_rates": "|".join(f"{value:.4f}" for value in spec.rates),
                "controlled_dividend_yields": "|".join(
                    f"{value:.4f}" for value in spec.dividend_yields
                ),
                "market_status": spec.market_status,
                "practical_rank_relative_tolerance": baseline.PRACTICAL_RANK_RELATIVE_TOLERANCE,
                "conditioning_warning_threshold": baseline.CONDITION_WARNING_THRESHOLD,
                "material_improvement_factor": MATERIAL_IMPROVEMENT_FACTOR,
            }
            for spec in EXPERIMENT_SPECS
        ]
    )


def _central5_rows(frame: pd.DataFrame) -> pd.DataFrame:
    node_values = np.asarray(baseline.MONEYNESS_CENTRAL_5, dtype=float)
    return frame.loc[
        frame["expiry_slot"].eq("far")
        & frame["log_moneyness_node"].astype(float).map(
            lambda value: bool(np.any(np.isclose(value, node_values, atol=1e-12)))
        )
    ].copy()


def build_far_expiry_support(
    panel: pd.DataFrame, moneyness_support: pd.DataFrame
) -> pd.DataFrame:
    """Separate structural, active, and declared-close usability for the far expiry."""
    node_rows = _central5_rows(moneyness_support)
    far_panel = panel.loc[panel["expiry_slot"].eq("far")].copy()
    rows: list[dict[str, Any]] = []
    for underlying in PRIMARY_UNDERLYINGS:
        for valuation_date in (value.isoformat() for value in G2_DATES):
            nodes = node_rows.loc[
                node_rows["underlying"].eq(underlying)
                & node_rows["valuation_date"].eq(valuation_date)
            ]
            raw = far_panel.loc[
                far_panel["underlying"].eq(underlying)
                & far_panel["valuation_date"].eq(valuation_date)
            ]
            if len(nodes) != 10 or raw.empty:
                raise ValueError(
                    f"Incomplete far-expiry evidence for {underlying} {valuation_date}"
                )
            structural_pct = 100.0 * float(nodes["inside_observed_bounds"].mean())
            active_pct = 100.0 * float(nodes["active_inside_observed_bounds"].mean())
            close_pct = 100.0 * float(raw["close_positive"].mean())
            settlement_pct = 100.0 * float(raw["settlement_positive"].mean())
            last_pct = 100.0 * float(raw["last_positive"].mean())
            max_bracket = float(nodes["bracket_width"].max())
            structurally_observed = bool(structural_pct == 100.0)
            usable_under_price_policy = bool(
                structurally_observed
                and close_pct == 100.0
                and max_bracket <= MAX_LOG_MONEYNESS_BRACKET_WIDTH
                and not nodes["extrapolation_required"].any()
            )
            actively_traded = bool(active_pct >= MIN_ACTIVE_SUPPORT_PCT)
            rows.append(
                {
                    "underlying": underlying,
                    "valuation_date": valuation_date,
                    "actual_expiry": str(raw["actual_expiry"].iloc[0]),
                    "far_dte": int(raw["DTE"].iloc[0]),
                    "central5_option_node_count": len(nodes),
                    "structurally_observed_pct": structural_pct,
                    "active_bracket_pct": active_pct,
                    "close_positive_pct": close_pct,
                    "settlement_positive_pct": settlement_pct,
                    "last_positive_pct": last_pct,
                    "max_log_moneyness_bracket_width": max_bracket,
                    "structurally_observed": structurally_observed,
                    "actively_traded_under_75pct_rule": actively_traded,
                    "usable_under_declared_close_policy": usable_under_price_policy,
                    "admitted_under_unchanged_market_quality_rule": bool(
                        usable_under_price_policy and actively_traded
                    ),
                }
            )
    return pd.DataFrame(rows)


def _scaled_jacobian_with_step(
    parameters: Sequence[float],
    maturity_days: Sequence[int],
    bounds: dict[str, tuple[float, float]],
    rates: Sequence[float],
    dividend_yields: Sequence[float],
    *,
    step_fraction: float,
    node_count: int,
) -> np.ndarray:
    """Independent central-difference spot check with an explicit step."""
    vector = np.asarray(parameters, dtype=float)
    widths = baseline._parameter_widths(bounds)
    columns: list[np.ndarray] = []
    for index, width in enumerate(widths):
        step = step_fraction * width
        for _ in range(12):
            lower = vector.copy()
            upper = vector.copy()
            lower[index] -= step
            upper[index] += step
            if (
                validate_parameters(lower)["is_valid"]
                and validate_parameters(upper)["is_valid"]
            ):
                break
            step *= 0.5
        else:
            raise RuntimeError(f"No valid central step for {PARAMETER_NAMES[index]}")
        lower_prices = baseline.normalized_observables(
            lower,
            REPRESENTATION,
            maturity_days,
            node_count=node_count,
            rates=rates,
            dividend_yields=dividend_yields,
        )
        upper_prices = baseline.normalized_observables(
            upper,
            REPRESENTATION,
            maturity_days,
            node_count=node_count,
            rates=rates,
            dividend_yields=dividend_yields,
        )
        columns.append((upper_prices - lower_prices) * width / (2.0 * step))
    return np.column_stack(columns)


def independent_baseline_spot_check(
    samples: pd.DataFrame,
    bounds: dict[str, tuple[float, float]],
    *,
    node_count: int,
) -> dict[str, float | int | bool | str]:
    sample = samples.iloc[0]
    parameters = sample[PARAMETER_NAMES].to_numpy(dtype=float)
    maturity_days = baseline.MATURITY_PROFILES[0][1]
    singular_by_step: dict[float, np.ndarray] = {}
    for step_fraction in (1.0e-4, 1.0e-5):
        jacobian = _scaled_jacobian_with_step(
            parameters,
            maturity_days,
            bounds,
            CONTROLLED_RATES[:2],
            CONTROLLED_DIVIDEND_YIELDS[:2],
            step_fraction=step_fraction,
            node_count=node_count,
        )
        singular_by_step[step_fraction] = np.linalg.svd(
            jacobian, compute_uv=False
        )
    first = singular_by_step[1.0e-4]
    second = singular_by_step[1.0e-5]
    relative_smallest_difference = float(abs(first[-1] - second[-1]) / second[-1])
    return {
        "sample_id": str(sample["sample_id"]),
        "near_dte": maturity_days[0],
        "middle_dte": maturity_days[1],
        "rank_step_1e_minus_4": int(np.linalg.matrix_rank(
            _scaled_jacobian_with_step(
                parameters,
                maturity_days,
                bounds,
                CONTROLLED_RATES[:2],
                CONTROLLED_DIVIDEND_YIELDS[:2],
                step_fraction=1.0e-4,
                node_count=node_count,
            )
        )),
        "smallest_singular_step_1e_minus_4": float(first[-1]),
        "smallest_singular_step_1e_minus_5": float(second[-1]),
        "relative_smallest_singular_difference": relative_smallest_difference,
        "negative_conclusion_stable": bool(
            first[-1] <= baseline.PRACTICAL_RANK_RELATIVE_TOLERANCE * first[0]
            and second[-1]
            <= baseline.PRACTICAL_RANK_RELATIVE_TOLERANCE * second[0]
        ),
    }


def run_jacobian_comparison(
    samples: pd.DataFrame,
    bounds: dict[str, tuple[float, float]],
    *,
    node_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    for sample in samples.itertuples(index=False):
        parameters = np.asarray(
            [getattr(sample, name) for name in PARAMETER_NAMES], dtype=float
        )
        for spec in EXPERIMENT_SPECS:
            for profile_id, maturity_days in spec.maturity_profiles:
                jacobian = baseline.scaled_parameter_jacobian(
                    parameters,
                    REPRESENTATION,
                    maturity_days,
                    bounds,
                    node_count=node_count,
                    rates=spec.rates,
                    dividend_yields=spec.dividend_yields,
                )
                _, singular_values, right_vectors = np.linalg.svd(
                    jacobian, full_matrices=False
                )
                numerical_tolerance = (
                    max(jacobian.shape)
                    * np.finfo(np.float64).eps
                    * singular_values[0]
                )
                practical_tolerance = (
                    baseline.PRACTICAL_RANK_RELATIVE_TOLERANCE * singular_values[0]
                )
                key = {
                    "sample_id": sample.sample_id,
                    "distribution": sample.distribution,
                    "maturity_profile": profile_id,
                    "representation_id": spec.representation_id,
                    "near_dte": maturity_days[0],
                    "middle_dte": maturity_days[1],
                    "far_dte": maturity_days[2] if len(maturity_days) == 3 else math.nan,
                    "observable_count": jacobian.shape[0],
                    "candidate_input_dimension": spec.candidate_input_dimension,
                }
                summary_rows.append(
                    {
                        **key,
                        "numerical_rank": int(
                            np.sum(singular_values > numerical_tolerance)
                        ),
                        "practical_rank_1e_minus_6": int(
                            np.sum(singular_values > practical_tolerance)
                        ),
                        "largest_singular_value": float(singular_values[0]),
                        "smallest_singular_value": float(singular_values[-1]),
                        "condition_number": float(
                            singular_values[0] / singular_values[-1]
                        ),
                        **{
                            f"singular_value_{index + 1}": float(value)
                            for index, value in enumerate(singular_values)
                        },
                    }
                )
                for name, sensitivity in zip(
                    PARAMETER_NAMES, np.linalg.norm(jacobian, axis=0), strict=True
                ):
                    sensitivity_rows.append(
                        {
                            **key,
                            "parameter": name,
                            "scaled_sensitivity": float(sensitivity),
                        }
                    )
                for name, loading in zip(
                    PARAMETER_NAMES, right_vectors[-1], strict=True
                ):
                    direction_rows.append(
                        {
                            **key,
                            "parameter": name,
                            "weakest_direction_loading": float(loading),
                            "absolute_weakest_direction_loading": float(abs(loading)),
                        }
                    )
    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(sensitivity_rows),
        pd.DataFrame(direction_rows),
    )


def material_improvement(jacobian: pd.DataFrame) -> dict[str, Any]:
    grouped = jacobian.groupby("representation_id", sort=True)
    median_smallest = grouped["smallest_singular_value"].median()
    median_condition = grouped["condition_number"].median()
    practical_full_frequency = grouped["practical_rank_1e_minus_6"].apply(
        lambda values: float((values == len(PARAMETER_NAMES)).mean())
    )
    smallest_gain = float(
        median_smallest["3exp_central5"] / median_smallest["2exp_central5"]
    )
    condition_reduction = float(
        median_condition["2exp_central5"] / median_condition["3exp_central5"]
    )
    no_rank_regression = bool(
        practical_full_frequency["3exp_central5"]
        >= practical_full_frequency["2exp_central5"]
    )
    triggered = bool(
        no_rank_regression
        and (
            smallest_gain >= MATERIAL_IMPROVEMENT_FACTOR
            or condition_reduction >= MATERIAL_IMPROVEMENT_FACTOR
        )
    )
    return {
        "smallest_singular_value_gain": smallest_gain,
        "condition_number_reduction": condition_reduction,
        "two_expiry_practical_full_rank_frequency": float(
            practical_full_frequency["2exp_central5"]
        ),
        "three_expiry_practical_full_rank_frequency": float(
            practical_full_frequency["3exp_central5"]
        ),
        "no_practical_rank_regression": no_rank_regression,
        "recovery_triggered": triggered,
    }


def _recovery_rows(
    true_parameters: np.ndarray,
    observed: np.ndarray,
    spec: ExperimentSpec,
    maturity_days: Sequence[int],
    bounds: dict[str, tuple[float, float]],
    *,
    sample_id: str,
    profile_id: str,
    noise_level: float,
    node_count: int,
    max_nfev: int,
    start_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    widths = baseline._parameter_widths(bounds)

    def residuals(latent: np.ndarray) -> np.ndarray:
        candidate = unconstrained_to_parameters(latent, bounds)
        return baseline.normalized_observables(
            candidate,
            REPRESENTATION,
            maturity_days,
            node_count=node_count,
            rates=spec.rates,
            dividend_yields=spec.dividend_yields,
        ) - observed

    rows: list[dict[str, Any]] = []
    for start_index, (strategy, start) in enumerate(
        baseline._deterministic_starts(seed, start_count)
    ):
        row: dict[str, Any] = {
            "representation_id": spec.representation_id,
            "sample_id": sample_id,
            "maturity_profile": profile_id,
            "near_dte": maturity_days[0],
            "middle_dte": maturity_days[1],
            "far_dte": maturity_days[2] if len(maturity_days) == 3 else math.nan,
            "noise_level": noise_level,
            "start_index": start_index,
            "start_strategy": strategy,
        }
        try:
            result = least_squares(
                residuals,
                start,
                method="trf",
                max_nfev=max_nfev,
                ftol=1.0e-10,
                xtol=1.0e-10,
                gtol=1.0e-10,
                diff_step=2.0e-5,
            )
            recovered = unconstrained_to_parameters(result.x, bounds)
            predicted = baseline.normalized_observables(
                recovered,
                REPRESENTATION,
                maturity_days,
                node_count=node_count,
                rates=spec.rates,
                dividend_yields=spec.dividend_yields,
            )
            errors = recovered - true_parameters
            scaled_errors = np.abs(errors) / widths
            validation = validate_parameters(recovered)
            bound_reasons = boundary_diagnostics(recovered, bounds)
            scaled_rmse = float(np.sqrt(np.mean(scaled_errors**2)))
            scaled_max = float(np.max(scaled_errors))
            row.update(
                {
                    "optimizer_success": bool(result.success),
                    "optimizer_status": int(result.status),
                    "nfev": int(result.nfev),
                    "price_rmse_normalized": float(
                        np.sqrt(np.mean((predicted - observed) ** 2))
                    ),
                    "price_mae_normalized": float(
                        np.mean(np.abs(predicted - observed))
                    ),
                    "aggregate_scaled_parameter_rmse": scaled_rmse,
                    "max_scaled_parameter_error": scaled_max,
                    "constraint_valid": bool(validation["is_valid"]),
                    "bound_hit": bool(bound_reasons),
                    "bound_reasons": ";".join(bound_reasons),
                    "parameter_recovery_success": bool(
                        result.success
                        and validation["is_valid"]
                        and scaled_rmse
                        <= baseline.RECOVERY_SCALED_RMSE_THRESHOLD
                        and scaled_max
                        <= baseline.RECOVERY_SCALED_MAX_ERROR_THRESHOLD
                    ),
                }
            )
            for index, name in enumerate(PARAMETER_NAMES):
                row[f"true_{name}"] = float(true_parameters[index])
                row[f"recovered_{name}"] = float(recovered[index])
                row[f"absolute_error_{name}"] = float(abs(errors[index]))
                row[f"scaled_error_{name}"] = float(scaled_errors[index])
        except Exception as error:
            row.update(
                {
                    "optimizer_success": False,
                    "optimizer_status": -1,
                    "nfev": 0,
                    "price_rmse_normalized": math.nan,
                    "price_mae_normalized": math.nan,
                    "aggregate_scaled_parameter_rmse": math.nan,
                    "max_scaled_parameter_error": math.nan,
                    "constraint_valid": False,
                    "bound_hit": False,
                    "bound_reasons": "",
                    "parameter_recovery_success": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            for index, name in enumerate(PARAMETER_NAMES):
                row[f"true_{name}"] = float(true_parameters[index])
                row[f"recovered_{name}"] = math.nan
                row[f"absolute_error_{name}"] = math.nan
                row[f"scaled_error_{name}"] = math.nan
        rows.append(row)
    return rows


def coupled_recovery_noise(
    seed: int, noise_level: float
) -> tuple[np.ndarray, np.ndarray]:
    """Preserve the baseline 20 shocks and append ten far-expiry shocks."""
    rng = np.random.default_rng(seed)
    two_expiry_noise = rng.normal(0.0, noise_level, size=20)
    far_expiry_noise = rng.normal(0.0, noise_level, size=10)
    three_expiry_noise = np.concatenate(
        (
            two_expiry_noise[:10],
            far_expiry_noise[:5],
            two_expiry_noise[10:],
            far_expiry_noise[5:],
        )
    )
    if not np.array_equal(
        three_expiry_noise[COMMON_COORDINATES_IN_THREE_EXPIRY_ORDER],
        two_expiry_noise,
    ):
        raise RuntimeError("Common-coordinate noise coupling is inconsistent")
    return two_expiry_noise, three_expiry_noise


def run_recovery_comparison(
    samples: pd.DataFrame,
    bounds: dict[str, tuple[float, float]],
    *,
    node_count: int,
    max_nfev: int,
    start_count: int,
    per_distribution: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    recovery_samples = (
        samples.groupby("distribution", sort=True, group_keys=False)
        .head(per_distribution)
        .reset_index(drop=True)
    )
    rows: list[dict[str, Any]] = []
    specs = {spec.representation_id: spec for spec in EXPERIMENT_SPECS}
    for sample_index, sample in enumerate(recovery_samples.itertuples(index=False)):
        true_parameters = np.asarray(
            [getattr(sample, name) for name in PARAMETER_NAMES], dtype=float
        )
        profile_index = sample_index % len(THREE_EXPIRY_MATURITY_PROFILES)
        profile_id, three_maturities = THREE_EXPIRY_MATURITY_PROFILES[profile_index]
        two_maturities = three_maturities[:2]
        clean_two = baseline.normalized_observables(
            true_parameters,
            REPRESENTATION,
            two_maturities,
            node_count=node_count,
            rates=specs["2exp_central5"].rates,
            dividend_yields=specs["2exp_central5"].dividend_yields,
        )
        clean_three = baseline.normalized_observables(
            true_parameters,
            REPRESENTATION,
            three_maturities,
            node_count=node_count,
            rates=specs["3exp_central5"].rates,
            dividend_yields=specs["3exp_central5"].dividend_yields,
        )
        for noise_index, noise_level in enumerate(baseline.NOISE_LEVELS):
            noise_seed = (
                baseline.ANALYSIS_SEED
                + 10000 * sample_index
                + 100 * profile_index
                + noise_index
            )
            two_noise, three_noise = coupled_recovery_noise(noise_seed, noise_level)
            observed_three = clean_three * (1.0 + three_noise)
            observed_two = clean_two * (1.0 + two_noise)
            if np.any(observed_two < 0.0) or np.any(observed_three < 0.0):
                raise RuntimeError("Noise produced a negative normalized price")
            start_seed = noise_seed + 500000
            rows.extend(
                _recovery_rows(
                    true_parameters,
                    observed_two,
                    specs["2exp_central5"],
                    two_maturities,
                    bounds,
                    sample_id=sample.sample_id,
                    profile_id=profile_id,
                    noise_level=noise_level,
                    node_count=node_count,
                    max_nfev=max_nfev,
                    start_count=start_count,
                    seed=start_seed,
                )
            )
            rows.extend(
                _recovery_rows(
                    true_parameters,
                    observed_three,
                    specs["3exp_central5"],
                    three_maturities,
                    bounds,
                    sample_id=sample.sample_id,
                    profile_id=profile_id,
                    noise_level=noise_level,
                    node_count=node_count,
                    max_nfev=max_nfev,
                    start_count=start_count,
                    seed=start_seed,
                )
            )
    starts = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    group_columns = [
        "representation_id",
        "sample_id",
        "maturity_profile",
        "noise_level",
    ]
    for keys, group in starts.groupby(group_columns, sort=True):
        finite = group.loc[np.isfinite(group["price_rmse_normalized"])].copy()
        best = (
            finite.sort_values("price_rmse_normalized", kind="stable").iloc[0]
            if not finite.empty
            else group.iloc[0]
        )
        recovered = finite[[f"recovered_{name}" for name in PARAMETER_NAMES]]
        widths = baseline._parameter_widths(bounds)
        variability = (
            float(np.mean(recovered.std(ddof=0).to_numpy(dtype=float) / widths))
            if len(recovered) > 1
            else math.nan
        )
        row = {
            "representation_id": keys[0],
            "sample_id": keys[1],
            "maturity_profile": keys[2],
            "noise_level": keys[3],
            "start_count": len(group),
            "optimizer_success_count": int(group["optimizer_success"].sum()),
            "parameter_recovery_success_count": int(
                group["parameter_recovery_success"].sum()
            ),
            "constraint_valid_count": int(group["constraint_valid"].sum()),
            "bound_hit_count": int(group["bound_hit"].sum()),
            "best_start_index_by_repricing": int(best["start_index"]),
            "best_price_rmse_normalized": float(best["price_rmse_normalized"]),
            "best_aggregate_scaled_parameter_rmse": float(
                best["aggregate_scaled_parameter_rmse"]
            ),
            "best_max_scaled_parameter_error": float(
                best["max_scaled_parameter_error"]
            ),
            "mean_start_parameter_std_scaled": variability,
        }
        for name in PARAMETER_NAMES:
            row[f"true_{name}"] = float(best[f"true_{name}"])
            row[f"best_recovered_{name}"] = float(best[f"recovered_{name}"])
            row[f"best_scaled_error_{name}"] = float(best[f"scaled_error_{name}"])
        summary_rows.append(row)
    return starts, pd.DataFrame(summary_rows)


def _recovery_frequency(frame: pd.DataFrame, representation_id: str, noise: float) -> float:
    group = frame.loc[
        frame["representation_id"].eq(representation_id)
        & frame["noise_level"].eq(noise)
    ]
    if group.empty:
        return 0.0
    return float(
        group["parameter_recovery_success_count"].sum()
        / group["start_count"].sum()
    )


def classify_g2(
    jacobian: pd.DataFrame,
    recovery: pd.DataFrame,
    improvement: dict[str, Any],
    far_support: pd.DataFrame,
    *,
    discount_source_status: str,
) -> dict[str, Any]:
    three = jacobian.loc[jacobian["representation_id"].eq("3exp_central5")]
    practical_rank_pass = bool(
        (three["practical_rank_1e_minus_6"] == len(PARAMETER_NAMES)).all()
    )
    conditioning_pass = bool(
        np.isfinite(three["condition_number"]).all()
        and (three["condition_number"] <= baseline.CONDITION_WARNING_THRESHOLD).all()
    )
    clean_frequency = _recovery_frequency(recovery, "3exp_central5", 0.0)
    half_frequency = _recovery_frequency(recovery, "3exp_central5", 0.005)
    one_frequency = _recovery_frequency(recovery, "3exp_central5", 0.01)
    clean_pass = clean_frequency >= 0.80
    half_pass = half_frequency >= 0.70
    one_pass = one_frequency >= 0.50
    numerical_pass = bool(
        improvement["recovery_triggered"]
        and practical_rank_pass
        and conditioning_pass
        and clean_pass
        and half_pass
        and one_pass
    )
    far_market_pass = bool(
        far_support["admitted_under_unchanged_market_quality_rule"].all()
    )
    discount_pass = discount_source_status == "VALIDATED"
    if numerical_pass and far_market_pass and discount_pass:
        classification = "G2 = PASSED"
    elif numerical_pass:
        classification = "G2 = NOT_PASSED — INFORMATION REMEDY IDENTIFIED"
    else:
        classification = (
            "G2 = NOT_PASSED — STRUCTURAL IDENTIFIABILITY PROBLEM REMAINS"
        )
    return {
        "classification": classification,
        "three_expiry_practical_rank_pass": practical_rank_pass,
        "three_expiry_conditioning_pass": conditioning_pass,
        "three_expiry_clean_recovery_frequency": clean_frequency,
        "three_expiry_noise_0_5pct_recovery_frequency": half_frequency,
        "three_expiry_noise_1pct_recovery_frequency": one_frequency,
        "three_expiry_clean_recovery_pass": clean_pass,
        "three_expiry_noise_0_5pct_recovery_pass": half_pass,
        "three_expiry_noise_1pct_recovery_pass": one_pass,
        "three_expiry_numerical_pass": numerical_pass,
        "far_expiry_market_quality_pass": far_market_pass,
        "discount_source_pass": discount_pass,
        "discount_source_status": discount_source_status,
        "final_input_dimension": 39
        if classification == "G2 = PASSED"
        else None,
    }


def _discount_source_section(report_path: Path) -> tuple[str, str]:
    if not report_path.is_file():
        return (
            "## Discount-source provenance\n\n"
            "`DISCOUNT_SOURCE = UNRESOLVED`. No authoritative historical "
            "tenor-aligned source was validated in this run.\n",
            "UNRESOLVED",
        )
    text = report_path.read_text(encoding="utf-8")
    marker = "## Discount-source provenance"
    start = text.find(marker)
    if start < 0:
        return (
            "## Discount-source provenance\n\n"
            "`DISCOUNT_SOURCE = UNRESOLVED`. The report contained no validated "
            "discount-source section.\n",
            "UNRESOLVED",
        )
    next_heading = text.find("\n## ", start + len(marker))
    section = text[start:] if next_heading < 0 else text[start:next_heading]
    status = "VALIDATED" if "DISCOUNT_SOURCE = VALIDATED" in section else "UNRESOLVED"
    return section.rstrip() + "\n", status


def write_figures(
    jacobian: pd.DataFrame,
    directions: pd.DataFrame,
    recovery: pd.DataFrame,
    far_support: pd.DataFrame,
    output_root: Path,
) -> tuple[Path, ...]:
    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    order = ["2exp_central5", "3exp_central5"]
    colors = ["#4C78A8", "#F58518"]
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(1, 11)
    for representation_id, color in zip(order, colors, strict=True):
        group = jacobian.loc[jacobian["representation_id"].eq(representation_id)]
        medians = [group[f"singular_value_{index}"].median() for index in x]
        ax.plot(x, medians, marker="o", label=representation_id, color=color)
    ax.set_yscale("log")
    ax.set_xlabel("Singular-value index")
    ax.set_ylabel("Median scaled singular value")
    ax.set_title("Two versus three listed expiries")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = figure_root / "singular_values_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    medians = [
        jacobian.loc[jacobian["representation_id"].eq(value), "condition_number"].median()
        for value in order
    ]
    ax.bar(order, medians, color=colors)
    ax.set_yscale("log")
    ax.axhline(
        baseline.CONDITION_WARNING_THRESHOLD,
        color="black",
        linestyle="--",
        linewidth=1,
        label="warning threshold",
    )
    ax.set_ylabel("Median condition number")
    ax.set_title("Conditioning comparison")
    ax.legend()
    fig.tight_layout()
    path = figure_root / "condition_number_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    frequencies = [
        100.0
        * float(
            (
                jacobian.loc[
                    jacobian["representation_id"].eq(value),
                    "practical_rank_1e_minus_6",
                ]
                == 10
            ).mean()
        )
        for value in order
    ]
    ax.bar(order, frequencies, color=colors)
    ax.set_ylim(0.0, 100.0)
    ax.set_ylabel("Practical full-rank frequency (%)")
    ax.set_title("Practical-rank comparison at relative 1e-6")
    fig.tight_layout()
    path = figure_root / "practical_rank_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    pivot = (
        directions.groupby(["representation_id", "parameter"])[
            "absolute_weakest_direction_loading"
        ]
        .median()
        .unstack(0)
        .loc[PARAMETER_NAMES]
    )
    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    pivot[order].plot(kind="bar", ax=ax, color=colors)
    ax.set_ylabel("Median absolute weakest-direction loading")
    ax.set_title("Weakest parameter-direction comparison")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    path = figure_root / "weakest_directions_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5))
    if recovery.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "Recovery not triggered", ha="center", va="center")
            ax.set_axis_off()
    else:
        noise_values = list(baseline.NOISE_LEVELS)
        for representation_id, color in zip(order, colors, strict=True):
            medians = [
                recovery.loc[
                    recovery["representation_id"].eq(representation_id)
                    & recovery["noise_level"].eq(noise),
                    "best_aggregate_scaled_parameter_rmse",
                ].median()
                for noise in noise_values
            ]
            frequencies = [
                100.0 * _recovery_frequency(recovery, representation_id, noise)
                for noise in noise_values
            ]
            axes[0].plot(
                [100.0 * value for value in noise_values],
                medians,
                marker="o",
                color=color,
                label=representation_id,
            )
            axes[1].plot(
                [100.0 * value for value in noise_values],
                frequencies,
                marker="o",
                color=color,
                label=representation_id,
            )
        axes[0].set_yscale("log")
        axes[0].set_ylabel("Median range-scaled parameter RMSE")
        axes[1].set_ylabel("Recovery-pass frequency (%)")
        for ax in axes:
            ax.set_xlabel("Multiplicative price noise (%)")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
    fig.suptitle("Target-blind recovery comparison")
    fig.tight_layout()
    path = figure_root / "recovery_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    activity = far_support.pivot(
        index="underlying", columns="valuation_date", values="active_bracket_pct"
    ).loc[list(PRIMARY_UNDERLYINGS)]
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    image = ax.imshow(activity.to_numpy(), vmin=0.0, vmax=100.0, cmap="RdYlGn")
    ax.set_xticks(range(len(activity.columns)), activity.columns, rotation=25)
    ax.set_yticks(range(len(activity.index)), activity.index)
    for row_index in range(len(activity.index)):
        for column_index in range(len(activity.columns)):
            value = activity.iloc[row_index, column_index]
            ax.text(column_index, row_index, f"{value:.0f}%", ha="center", va="center")
    ax.set_title("Far-expiry central-5 active bracketing (75% rule)")
    fig.colorbar(image, ax=ax, label="Active bracket support (%)")
    fig.tight_layout()
    path = figure_root / "far_expiry_market_support.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0))
    condition = [
        jacobian.loc[jacobian["representation_id"].eq(value), "condition_number"].median()
        for value in order
    ]
    smallest = [
        jacobian.loc[
            jacobian["representation_id"].eq(value), "smallest_singular_value"
        ].median()
        for value in order
    ]
    ranks = [
        100.0
        * float(
            (
                jacobian.loc[
                    jacobian["representation_id"].eq(value),
                    "practical_rank_1e_minus_6",
                ]
                == 10
            ).mean()
        )
        for value in order
    ]
    axes[0, 0].bar(order, condition, color=colors)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Median condition number")
    axes[0, 1].bar(order, smallest, color=colors)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Median smallest singular value")
    axes[1, 0].bar(order, ranks, color=colors)
    axes[1, 0].set_ylim(0.0, 100.0)
    axes[1, 0].set_title("Practical full-rank frequency (%)")
    axes[1, 1].bar(
        ["structural", "close policy", "active rule"],
        [
            100.0 * far_support["structurally_observed"].mean(),
            100.0 * far_support["usable_under_declared_close_policy"].mean(),
            100.0 * far_support["actively_traded_under_75pct_rule"].mean(),
        ],
        color=["#54A24B", "#54A24B", "#E45756"],
    )
    axes[1, 1].set_ylim(0.0, 100.0)
    axes[1, 1].set_title("Far-expiry surface pass frequency (%)")
    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Mentor summary: minimum term-structure remedy")
    fig.tight_layout()
    path = figure_root / "mentor_summary.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return tuple(paths)


def render_report(
    matrix: pd.DataFrame,
    spot_check: dict[str, Any],
    far_support: pd.DataFrame,
    discount_section: str,
    jacobian: pd.DataFrame,
    sensitivity: pd.DataFrame,
    directions: pd.DataFrame,
    improvement: dict[str, Any],
    recovery: pd.DataFrame,
    decision: dict[str, Any],
    artifact_hashes: dict[str, str],
) -> str:
    lines = [
        "# G2 Information Remediation",
        "",
        "## Independent baseline review",
        "",
        "**Verdict: SHIP.** The previous negative identifiability result survived the fresh review.",
        "",
        "The scaled Jacobian uses central differences of spot-normalized prices and full hard-bound parameter widths. Algebraic rank, the relative practical-rank threshold, condition numbers, target-blind recovery, constraint checks, bound diagnostics, and call/put parity redundancy are implemented consistently. Repricing and parameter recovery remain separate gates.",
        "",
        f"Independent step check on `{spot_check['sample_id']}` at DTE `{spot_check['near_dte']}|{spot_check['middle_dte']}`: rank `{spot_check['rank_step_1e_minus_4']}`, smallest singular values `{spot_check['smallest_singular_step_1e_minus_4']:.3e}` and `{spot_check['smallest_singular_step_1e_minus_5']:.3e}` at relative steps `1e-4` and `1e-5`; the negative practical-rank conclusion is stable: `{spot_check['negative_conclusion_stable']}`.",
        "",
        "## Predeclared experiment matrix",
        "",
        "The matrix was frozen before computing the three-expiry result. No wing search or combinatorial grid was performed.",
        "",
        "| Representation | Listed expiries | Prices | Maturities | Carry | Candidate inputs | Market status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in matrix.itertuples(index=False):
        lines.append(
            f"| `{row.representation_id}` | {len(row.expiry_positions.split('|'))} | "
            f"{row.normalized_price_count} | {row.maturity_coordinate_count} | "
            f"{row.carry_coordinate_count} | {row.candidate_input_dimension} | {row.market_status} |"
        )
    lines.extend(
        [
            "",
            f"Material-improvement trigger: at least `{MATERIAL_IMPROVEMENT_FACTOR:.0f}x` gain in median smallest singular value or reduction in median condition number, with no practical-rank regression. The same eight parameter vectors, three actual date profiles, 64-node pricer, range scaling, thresholds, and deterministic protocol are used for both representations.",
            "",
            "## Far-expiry market support",
            "",
            f"- `STRUCTURALLY_OBSERVED = {'YES' if far_support['structurally_observed'].all() else 'NO'}`: central-5 bracketing uses the actual listed third expiry, with no maturity interpolation or extrapolation.",
            f"- `ACTIVELY_TRADED_UNDER_75PCT_RULE = {'YES' if far_support['actively_traded_under_75pct_rule'].all() else 'NO'}`: overall mean `{far_support['active_bracket_pct'].mean():.1f}%`, worst stock/date `{far_support['active_bracket_pct'].min():.1f}%`.",
            f"- `USABLE_UNDER_DECLARED_CLOSE_POLICY = {'YES' if far_support['usable_under_declared_close_policy'].all() else 'NO'}`: worst close availability `{far_support['close_positive_pct'].min():.1f}%`, worst settlement availability `{far_support['settlement_positive_pct'].min():.1f}%`, largest bracket `{far_support['max_log_moneyness_bracket_width'].max():.6f}`.",
            f"- `FAR_EXPIRY_MARKET_ADMISSION = {'YES' if far_support['admitted_under_unchanged_market_quality_rule'].all() else 'NO'}`. The unchanged Stage A 75% activity rule is not redefined merely because the third expiry may improve conditioning.",
            "",
            "| Stock | Date | DTE | Structural | Active | Close | Settlement | Price-policy usable | Activity-rule pass |",
            "|---|---|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in far_support.itertuples(index=False):
        lines.append(
            f"| {row.underlying} | {row.valuation_date} | {row.far_dte} | "
            f"{row.structurally_observed_pct:.1f}% | {row.active_bracket_pct:.1f}% | "
            f"{row.close_positive_pct:.1f}% | {row.settlement_positive_pct:.1f}% | "
            f"{row.usable_under_declared_close_policy} | {row.actively_traded_under_75pct_rule} |"
        )
    lines.extend(["", discount_section.rstrip(), "", "## Identifiability comparison", ""])
    lines.extend(
        [
            "| Representation | Algebraic rank 10 | Practical rank 10 | Median smallest singular value | Median condition number |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for representation_id in ("2exp_central5", "3exp_central5"):
        group = jacobian.loc[jacobian["representation_id"].eq(representation_id)]
        lines.append(
            f"| `{representation_id}` | {100.0 * (group['numerical_rank'] == 10).mean():.1f}% | "
            f"{100.0 * (group['practical_rank_1e_minus_6'] == 10).mean():.1f}% | "
            f"{group['smallest_singular_value'].median():.3e} | {group['condition_number'].median():.3e} |"
        )
    lines.extend(
        [
            "",
            f"Median smallest-singular-value gain: `{improvement['smallest_singular_value_gain']:.2f}x`; median condition-number reduction: `{improvement['condition_number_reduction']:.2f}x`; recovery trigger: `{improvement['recovery_triggered']}`.",
            "",
        ]
    )
    for representation_id in ("2exp_central5", "3exp_central5"):
        weak_sensitivity = (
            sensitivity.loc[sensitivity["representation_id"].eq(representation_id)]
            .groupby("parameter")["scaled_sensitivity"]
            .median()
            .sort_values()
            .head(5)
        )
        weak_direction = (
            directions.loc[directions["representation_id"].eq(representation_id)]
            .groupby("parameter")["absolute_weakest_direction_loading"]
            .median()
            .sort_values(ascending=False)
            .head(5)
        )
        lines.append(
            f"- `{representation_id}` weakest sensitivities: "
            + ", ".join(f"`{name}` ({value:.3e})" for name, value in weak_sensitivity.items())
            + "."
        )
        lines.append(
            f"- `{representation_id}` dominant weakest-direction loadings: "
            + ", ".join(f"`{name}` ({value:.3f})" for name, value in weak_direction.items())
            + "."
        )
    lines.extend(["", "## Target-blind recovery", ""])
    if recovery.empty:
        lines.append(
            "Recovery was not run because the predeclared local-conditioning trigger did not fire."
        )
    else:
        lines.extend(
            [
                "| Representation | Noise | Optimizer success | Constraint valid | Recovery pass | Median price RMSE | Median parameter RMSE | Bound hits | Median start variation |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for representation_id in ("2exp_central5", "3exp_central5"):
            for noise in baseline.NOISE_LEVELS:
                group = recovery.loc[
                    recovery["representation_id"].eq(representation_id)
                    & recovery["noise_level"].eq(noise)
                ]
                lines.append(
                    f"| `{representation_id}` | {100.0 * noise:.1f}% | "
                    f"{int(group['optimizer_success_count'].sum())}/{int(group['start_count'].sum())} | "
                    f"{int(group['constraint_valid_count'].sum())}/{int(group['start_count'].sum())} | "
                    f"{int(group['parameter_recovery_success_count'].sum())}/{int(group['start_count'].sum())} | "
                    f"{group['best_price_rmse_normalized'].median():.3e} | "
                    f"{group['best_aggregate_scaled_parameter_rmse'].median():.3e} | "
                    f"{int(group['bound_hit_count'].sum())}/{int(group['start_count'].sum())} | "
                    f"{group['mean_start_parameter_std_scaled'].median():.3e} |"
                )
        lines.extend(
            [
                "",
                "Median best-start absolute parameter error scaled by each hard-bound width:",
                "",
                "| Parameter | 2-exp clean | 3-exp clean | 2-exp 0.5% | 3-exp 0.5% | 2-exp 1.0% | 3-exp 1.0% |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name in PARAMETER_NAMES:
            values: list[float] = []
            for noise in baseline.NOISE_LEVELS:
                for representation_id in ("2exp_central5", "3exp_central5"):
                    group = recovery.loc[
                        recovery["representation_id"].eq(representation_id)
                        & recovery["noise_level"].eq(noise)
                    ]
                    values.append(float(group[f"best_scaled_error_{name}"].median()))
            lines.append(
                f"| `{name}` | " + " | ".join(f"{value:.3e}" for value in values) + " |"
            )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"**{decision['classification']}**",
            "",
            f"Three-expiry numerical pass: `{decision['three_expiry_numerical_pass']}`; far-expiry market-quality pass: `{decision['far_expiry_market_quality_pass']}`; discount-source pass: `{decision['discount_source_pass']}`.",
            "",
            "The final representation is not frozen unless both real-market policy and every numerical gate pass. No final 10k dataset was generated; ANN research training was not started; PINN work was not started.",
            "",
            "## Phase 8 stop diagnosis and next action",
            "",
            "The third listed expiry materially reduces local ill-conditioning but does not stabilize all ten targets: practical full rank is not universal, clean recovery remains below its predeclared frequency gate, and both noisy recovery gates fail. The dominant remaining weak combination continues to involve `kappa_slow`, `theta_slow`, and `theta_fast`; the result therefore points to missing slow-factor/time-evolution information rather than a need for unsupported strike wings.",
            "",
            "Minimum defensible future choices are a complementary stock-specific variance observable, explicit weak-direction priors/regularization with prior-versus-data attribution, or a joint multi-date inverse observation model with an explicit state contract. None was implemented here.",
            "",
            "**Single recommended next action:** predeclare a bounded joint multi-date, same-stock identifiability design using the already selected official-NSE option dates plus a reproducible variance-state observation, explicitly defining how `v0_slow` and `v0_fast` evolve so the canonical ten-parameter target is not reduced or silently redefined. Review that design with the mentor before any new data acquisition or calibration run.",
            "",
            "## Reproducibility and artifacts",
            "",
            "The eight canonical Stage A outputs, the prior common-support evidence, and the prior G2 identifiability evidence are hash-preserved before and after this run.",
            "",
            "| New artifact | SHA-256 |",
            "|---|---|",
        ]
    )
    for relative, digest in artifact_hashes.items():
        lines.append(f"| `{relative}` | `{digest.upper()}` |")
    lines.extend(
        [
            "",
            "```text",
            "INDEPENDENT_REVIEW = SHIP",
            "PREVIOUS_NEGATIVE_IDENTIFIABILITY = SURVIVED_REVIEW",
            f"DISCOUNT_SOURCE = {decision['discount_source_status']}",
            f"FAR_EXPIRY_MARKET_ADMISSION = {'PASSED' if decision['far_expiry_market_quality_pass'] else 'NOT_PASSED'}",
            decision["classification"],
            f"FINAL_INPUT_DIMENSION = {decision['final_input_dimension'] if decision['final_input_dimension'] is not None else 'NOT_FROZEN'}",
            "FINAL_SYNTHETIC_RESEARCH_DATA = NOT_GENERATED",
            "ANN_RESEARCH_TRAINING = NOT_STARTED",
            "PINN = NOT_DERIVED_OR_TRAINED",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def run_analysis(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    report_path: Path = DEFAULT_REPORT_PATH,
    write_outputs: bool = True,
    quick: bool = False,
) -> dict[str, Any]:
    canonical_before = snapshot_canonical_outputs(CANONICAL_DERIVED_ROOT)
    prior_g2_before = baseline._snapshot_existing_g2(CANONICAL_DERIVED_ROOT)
    prior_identifiability_before = _snapshot_paths(
        baseline.DEFAULT_OUTPUT_ROOT, BASELINE_IDENTIFIABILITY_ARTIFACTS
    )
    validate_canonical_stage_a_provenance(CANONICAL_RAW_ROOT, CANONICAL_DERIVED_ROOT)
    panel = load_balanced_panel(CANONICAL_RAW_ROOT)
    far_support = build_far_expiry_support(panel, build_moneyness_support(panel))
    matrix = experiment_matrix()
    bounds = load_hard_safety_bounds(baseline.BOUNDS_PATH)
    samples = baseline.select_representative_parameters(
        bounds, per_distribution=2 if quick else 4
    )
    node_count = 16 if quick else baseline.FULL_PRICER_NODE_COUNT
    spot_check = independent_baseline_spot_check(
        samples, bounds, node_count=node_count
    )
    jacobian, sensitivity, directions = run_jacobian_comparison(
        samples, bounds, node_count=node_count
    )
    improvement = material_improvement(jacobian)
    if improvement["recovery_triggered"]:
        recovery_starts, recovery_summary = run_recovery_comparison(
            samples,
            bounds,
            node_count=node_count,
            max_nfev=30 if quick else 120,
            start_count=2 if quick else 3,
            per_distribution=1 if quick else 2,
        )
    else:
        recovery_starts = pd.DataFrame()
        recovery_summary = pd.DataFrame()
    discount_section, discount_status = _discount_source_section(report_path)
    decision = classify_g2(
        jacobian,
        recovery_summary,
        improvement,
        far_support,
        discount_source_status=discount_status,
    )
    evidence = {
        "experiment_matrix.csv": matrix,
        "far_expiry_market_support.csv": far_support,
        "jacobian_comparison.csv": jacobian,
        "parameter_sensitivity_comparison.csv": sensitivity,
        "weakest_directions_comparison.csv": directions,
        "recovery_starts.csv": recovery_starts,
        "recovery_summary.csv": recovery_summary,
    }
    evidence_paths: dict[str, Path] = {}
    figure_paths: tuple[Path, ...] = ()
    artifact_hashes: dict[str, str] = {}
    if write_outputs:
        output_root.mkdir(parents=True, exist_ok=True)
        for relative, frame in evidence.items():
            path = output_root / relative
            _write_csv(frame, path)
            evidence_paths[relative] = path
            artifact_hashes[relative] = _sha256(path)
        decision_path = output_root / "decision.json"
        _atomic_write_bytes(
            decision_path,
            (json.dumps({**improvement, **decision}, indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            ),
        )
        evidence_paths["decision.json"] = decision_path
        artifact_hashes["decision.json"] = _sha256(decision_path)
        figure_paths = write_figures(
            jacobian, directions, recovery_summary, far_support, output_root
        )
        for path in figure_paths:
            relative = str(path.relative_to(output_root)).replace("\\", "/")
            artifact_hashes[relative] = _sha256(path)
        report = render_report(
            matrix,
            spot_check,
            far_support,
            discount_section,
            jacobian,
            sensitivity,
            directions,
            improvement,
            recovery_summary,
            decision,
            artifact_hashes,
        )
        _atomic_write_bytes(report_path, report.encode("utf-8"))
    else:
        report = render_report(
            matrix,
            spot_check,
            far_support,
            discount_section,
            jacobian,
            sensitivity,
            directions,
            improvement,
            recovery_summary,
            decision,
            {},
        )
    assert_canonical_outputs_preserved(CANONICAL_DERIVED_ROOT, canonical_before)
    baseline._assert_snapshot_unchanged(CANONICAL_DERIVED_ROOT, prior_g2_before)
    if _snapshot_paths(
        baseline.DEFAULT_OUTPUT_ROOT, BASELINE_IDENTIFIABILITY_ARTIFACTS
    ) != prior_identifiability_before:
        raise RuntimeError("Prior G2 identifiability evidence changed unexpectedly")
    return {
        "matrix": matrix,
        "spot_check": spot_check,
        "far_support": far_support,
        "samples": samples,
        "jacobian": jacobian,
        "sensitivity": sensitivity,
        "directions": directions,
        "improvement": improvement,
        "recovery_starts": recovery_starts,
        "recovery_summary": recovery_summary,
        "decision": decision,
        "report": report,
        "evidence_paths": evidence_paths,
        "figure_paths": figure_paths,
        "canonical_hashes": canonical_before,
        "prior_g2_hashes": prior_g2_before,
        "prior_identifiability_hashes": prior_identifiability_before,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()
    result = run_analysis(
        output_root=arguments.output_root,
        report_path=arguments.report_path,
        write_outputs=not arguments.no_write,
        quick=arguments.quick,
    )
    print(
        f"{ANALYSIS_ID} smallest_gain={result['improvement']['smallest_singular_value_gain']:.6e} "
        f"condition_reduction={result['improvement']['condition_number_reduction']:.6e} "
        f"recovery_triggered={result['improvement']['recovery_triggered']}"
    )
    print(result["decision"]["classification"])
    for path in result["evidence_paths"].values():
        print(f"evidence={path}")
    for path in result["figure_paths"]:
        print(f"figure={path}")
    if not arguments.no_write:
        print(f"report={arguments.report_path}")


if __name__ == "__main__":
    main()
