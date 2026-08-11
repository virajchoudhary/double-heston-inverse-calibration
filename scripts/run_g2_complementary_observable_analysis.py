"""Bounded complementary-observable information experiment for G2.

The canonical ten-parameter target and central-five option geometry are fixed.
Design A is replayed from the preserved global-ambiguity evidence; it is not
re-optimized.  Designs B-D reuse those target-blind solutions as deterministic
warm starts and add only the predeclared complementary observations below.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import least_squares
from scipy.stats import norm

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import scripts.run_g2_global_ambiguity_analysis as ambiguity
import scripts.run_g2_identifiability_analysis as baseline
import scripts.run_g2_multi_date_identifiability as multi_date
from src.calibrate_double_heston import (
    boundary_diagnostics,
    load_hard_safety_bounds,
    parameters_to_unconstrained,
    unconstrained_to_parameters,
)
from src.constants import PARAMETER_NAMES
from src.constraints import validate_parameters


ANALYSIS_ID = "G2_COMPLEMENTARY_OBSERVABLE"
ANALYSIS_SEED = 20260811
FULL_PRICER_NODE_COUNT = 64
TRADING_DAYS_PER_YEAR = 252
HISTORY_TRADING_DAYS = 252
SHORT_RV_WINDOW = 21
LONG_RV_WINDOW = 126
PERSISTENCE_BLOCK_DAYS = 5
PERSISTENCE_LAG_BLOCKS = 4
PERSISTENCE_LAG_DAYS = PERSISTENCE_BLOCK_DAYS * PERSISTENCE_LAG_BLOCKS
DT = 1.0 / TRADING_DAYS_PER_YEAR

# Frozen before calibration; they are not tuned from the resulting clusters.
OPTION_NEAR_EQUIVALENCE_RMSE = ambiguity.NEAR_PRICE_EQUIVALENCE_RMSE
MATERIAL_DISPLACEMENT_RMSE = ambiguity.MATERIAL_DISPLACEMENT_RMSE
CLUSTER_DISTANCE_CUTOFF = ambiguity.CLUSTER_DISTANCE_CUTOFF
PRACTICAL_RANK_RELATIVE_TOLERANCE = baseline.PRACTICAL_RANK_RELATIVE_TOLERANCE
ORACLE_TOTAL_VARIANCE_SCALE = 1.0e-4
LOG_REALIZED_VARIANCE_SCALE = 0.10
PERSISTENCE_SCALE = 0.10
COMPLEMENTARY_NEAR_EQUIVALENCE_RMSE = 1.0
DECISION_CLUSTER_RATIO_MAX = 0.50
DECISION_MATERIAL_RATIO_MAX = 0.50
DECISION_PARAMETER_ERROR_RATIO_MAX = 0.75
DECISION_MIN_CASE_COUNT = 3
MAX_NFEV = 80
NOISY_WARM_START_COUNT = 5
EXPECTED_CASE_IDS = ("case_1", "case_2", "case_3", "case_4")
NOISE_COMPARISON_CASE_IDS = ("case_1", "case_3")
MARKET_SECURITIES = ("NTPC", "CIPLA", "INFY", "HDFCBANK")
MARKET_VALUATION_DATES = tuple(profile[0] for profile in baseline.MATURITY_PROFILES)
OPTIMIZER_TOLERANCE = 1.0e-10
DIFF_STEP = 2.0e-5
NOISE_LEVELS = (0.0, 0.005, 0.01)
LOG_RV_NOISE_SD = {0.0: 0.0, 0.005: 0.05, 0.01: 0.10}
PERSISTENCE_NOISE_SD = {0.0: 0.0, 0.005: 0.05, 0.01: 0.10}

DEFAULT_STAGE_A_ROOT = REPOSITORY_ROOT / "market_data_audit" / "stage_a"
DEFAULT_OUTPUT_ROOT = DEFAULT_STAGE_A_ROOT / "derived" / "g2_complementary_observables"
DEFAULT_REPORT_PATH = REPOSITORY_ROOT / "docs" / "G2_COMPLEMENTARY_OBSERVABLE_ANALYSIS.md"
PRIOR_SOLUTIONS_PATH = (
    DEFAULT_STAGE_A_ROOT / "derived" / "g2_global_ambiguity" / "all_solutions.csv"
)
CURRENT_TEST_PATH = REPOSITORY_ROOT / "tests" / "test_g2_complementary_observable_analysis.py"


@dataclass(frozen=True)
class Design:
    design_id: str
    label: str
    feature_names: tuple[str, ...]
    market_status: str


DESIGNS = (
    Design("A", "Options only", (), "CANONICAL_AMBIGUITY_BASELINE"),
    Design(
        "B",
        "Options + oracle total variance",
        ("oracle_total_variance",),
        "ORACLE_UPPER_BOUND_NOT_MARKET_OBSERVABLE",
    ),
    Design(
        "C",
        "Options + 21D/126D realized variance",
        ("log_rv_21", "log_rv_126"),
        "MARKET_CONTRACT_PENDING",
    ),
    Design(
        "D",
        "Options + 21D/126D realized variance + persistence",
        ("log_rv_21", "log_rv_126", "rv_block_persistence"),
        "MARKET_CONTRACT_PENDING",
    ),
)
DESIGN_BY_ID = {item.design_id: item for item in DESIGNS}
FEATURE_SCALES = {
    "oracle_total_variance": ORACLE_TOTAL_VARIANCE_SCALE,
    "log_rv_21": LOG_REALIZED_VARIANCE_SCALE,
    "log_rv_126": LOG_REALIZED_VARIANCE_SCALE,
    "rv_block_persistence": PERSISTENCE_SCALE,
}

DATA_ARTIFACTS = (
    "predeclared_contract.json",
    "experiment_matrix.csv",
    "cases.csv",
    "synthetic_return_history.csv",
    "path_observables.csv",
    "truth_fit_diagnostics.csv",
    "jacobian_summary.csv",
    "singular_values.csv",
    "parameter_sensitivities.csv",
    "weakest_directions.csv",
    "recovery_solutions.csv",
    "recovery_summary.csv",
    "ambiguity_summary.csv",
    "parameter_errors.csv",
    "market_feasibility.csv",
    "contract.json",
    "decision.json",
)
FIGURE_ARTIFACTS = (
    "figures/01_conditioning_comparison.png",
    "figures/02_parameter_error_comparison.png",
    "figures/03_global_ambiguity_comparison.png",
    "figures/04_slow_fast_variance_allocation.png",
    "figures/05_theta_kappa_information.png",
    "figures/06_clean_vs_noisy.png",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


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
        path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )


def _parameter_widths(bounds: dict[str, tuple[float, float]]) -> np.ndarray:
    return np.asarray([bounds[name][1] - bounds[name][0] for name in PARAMETER_NAMES])


def _scaled_coordinates(
    parameters: Sequence[float], bounds: dict[str, tuple[float, float]]
) -> np.ndarray:
    lower = np.asarray([bounds[name][0] for name in PARAMETER_NAMES])
    return (np.asarray(parameters, dtype=float) - lower) / _parameter_widths(bounds)


def experiment_matrix() -> pd.DataFrame:
    rows = []
    for design in DESIGNS:
        rows.append(
            {
                "design_id": design.design_id,
                "label": design.label,
                "option_observation_count": 20,
                "complementary_observation_count": len(design.feature_names),
                "complementary_observables": ";".join(design.feature_names) or "NONE",
                "market_status": design.market_status,
                "broad_feature_search": False,
                "canonical_target_count": len(PARAMETER_NAMES),
            }
        )
    return pd.DataFrame(rows)


def predeclared_contract() -> dict[str, Any]:
    """Return the source-defined contract written before calibration starts."""
    return {
        "analysis_id": ANALYSIS_ID,
        "status": "SOURCE_DEFINED_BEFORE_CALIBRATION",
        "analysis_seed": ANALYSIS_SEED,
        "node_count": FULL_PRICER_NODE_COUNT,
        "parameter_order": PARAMETER_NAMES,
        "designs": experiment_matrix().to_dict(orient="records"),
        "windows": {
            "history_trading_days": HISTORY_TRADING_DAYS,
            "short_rv_trading_days": SHORT_RV_WINDOW,
            "long_rv_trading_days": LONG_RV_WINDOW,
            "persistence_block_days": PERSISTENCE_BLOCK_DAYS,
            "persistence_lag_blocks": PERSISTENCE_LAG_BLOCKS,
        },
        "scales": FEATURE_SCALES,
        "noise": {
            "option_levels": NOISE_LEVELS,
            "log_rv_sd": LOG_RV_NOISE_SD,
            "persistence_sd": PERSISTENCE_NOISE_SD,
        },
        "thresholds": {
            "practical_rank_relative": PRACTICAL_RANK_RELATIVE_TOLERANCE,
            "price_near_equivalence_rmse": OPTION_NEAR_EQUIVALENCE_RMSE,
            "complementary_near_equivalence_standardized_rmse": COMPLEMENTARY_NEAR_EQUIVALENCE_RMSE,
            "material_parameter_rmse": MATERIAL_DISPLACEMENT_RMSE,
            "cluster_distance": CLUSTER_DISTANCE_CUTOFF,
        },
        "decision_rule": {
            "cluster_ratio_max": DECISION_CLUSTER_RATIO_MAX,
            "material_solution_ratio_max": DECISION_MATERIAL_RATIO_MAX,
            "parameter_error_ratio_max": DECISION_PARAMETER_ERROR_RATIO_MAX,
            "minimum_distinct_case_count": DECISION_MIN_CASE_COUNT,
            "required_case_ids": EXPECTED_CASE_IDS,
            "required_market_securities": MARKET_SECURITIES,
            "required_market_valuation_dates": MARKET_VALUATION_DATES,
        },
        "reporting_contract": {
            "global_clean_case_ids": EXPECTED_CASE_IDS,
            "matched_clean_noisy_case_ids": NOISE_COMPARISON_CASE_IDS,
            "clean_noisy_comparison_requires_identical_case_population": True,
        },
        "optimizer": {
            "max_nfev": MAX_NFEV,
            "ftol_xtol_gtol": OPTIMIZER_TOLERANCE,
            "diff_step": DIFF_STEP,
        },
        "warm_starts": {
            "clean": "ALL_ESTABLISHED_CLEAN_NEAR_EQUIVALENT_A_SOLUTIONS",
            "noisy": "TOP_FIVE_PRIOR_A_PRICE_FITS_PER_CASE_AND_NOISE_LEVEL",
            "noisy_count": NOISY_WARM_START_COUNT,
        },
        "validity_rule": (
            "C_OR_D_REQUIRES_TRUE_PARAMETER_COMPLEMENTARY_RMSE_LE_1_IN_AT_LEAST_3_OF_4_CASES; "
            "OTHERWISE_ONLY_CURRENT_EXPERIMENT_EVIDENCE_IS_INSUFFICIENT"
        ),
    }


def _predeclared_contract_sha256() -> str:
    payload = (json.dumps(predeclared_contract(), indent=2, sort_keys=True) + "\n").encode()
    return hashlib.sha256(payload).hexdigest().upper()


def _case_seed(case_index: int, purpose: int, noise_level: float = 0.0) -> int:
    return ANALYSIS_SEED + 100_000 * case_index + 100 * int(round(noise_level * 10_000)) + purpose


def _factor_indices(prefix: str) -> tuple[int, int, int, int, int]:
    offset = 0 if prefix == "slow" else 5
    return offset, offset + 1, offset + 2, offset + 3, offset + 4


def simulate_causal_return_history(
    parameters: Sequence[float], case_index: int
) -> pd.DataFrame:
    """Simulate a causal one-year history ending at the declared valuation state.

    Each CIR factor is sampled backward from its valuation-time state using the
    exact transition quantile.  A stationary CIR diffusion is reversible, so
    reversing that chain gives the conditional pre-valuation variance history.
    Daily returns use a disclosed Gaussian-copula leverage approximation; the
    calibration mapping never receives these random draws or their seed.
    """
    vector = np.asarray(parameters, dtype=float)
    if not validate_parameters(vector)["is_valid"]:
        raise ValueError("history simulation requires a valid canonical ten-vector")
    rng = np.random.default_rng(_case_seed(case_index, 17))
    uniforms = {
        "slow": rng.uniform(1.0e-9, 1.0 - 1.0e-9, HISTORY_TRADING_DAYS),
        "fast": rng.uniform(1.0e-9, 1.0 - 1.0e-9, HISTORY_TRADING_DAYS),
    }
    independent = {
        "slow": rng.standard_normal(HISTORY_TRADING_DAYS),
        "fast": rng.standard_normal(HISTORY_TRADING_DAYS),
    }
    chronological: dict[str, np.ndarray] = {}
    variance_scores: dict[str, np.ndarray] = {}
    for factor in ("slow", "fast"):
        kappa_i, theta_i, sigma_i, _, v0_i = _factor_indices(factor)
        reverse = np.empty(HISTORY_TRADING_DAYS + 1, dtype=float)
        reverse[0] = vector[v0_i]
        for step, uniform in enumerate(uniforms[factor]):
            reverse[step + 1] = multi_date.exact_cir_transition_from_uniform(
                vector[kappa_i],
                vector[theta_i],
                vector[sigma_i],
                reverse[step],
                DT,
                float(uniform),
            )
        chronological[factor] = reverse[::-1]
        variance_scores[factor] = norm.ppf(uniforms[factor])[::-1]

    returns = np.zeros(HISTORY_TRADING_DAYS, dtype=float)
    factor_returns: dict[str, np.ndarray] = {}
    for factor in ("slow", "fast"):
        _, _, _, rho_i, _ = _factor_indices(factor)
        rho = vector[rho_i]
        z = rho * variance_scores[factor] + math.sqrt(1.0 - rho * rho) * independent[factor]
        interval_variance = 0.5 * (
            chronological[factor][:-1] + chronological[factor][1:]
        )
        factor_returns[factor] = np.sqrt(np.maximum(interval_variance, 0.0) * DT) * z
        returns += factor_returns[factor]
    total_interval_variance = 0.5 * (
        chronological["slow"][:-1]
        + chronological["slow"][1:]
        + chronological["fast"][:-1]
        + chronological["fast"][1:]
    )
    returns -= 0.5 * total_interval_variance * DT
    offsets = np.arange(-HISTORY_TRADING_DAYS + 1, 1, dtype=int)
    return pd.DataFrame(
        {
            "case_index": case_index,
            "trading_day_offset": offsets,
            "v_slow_start": chronological["slow"][:-1],
            "v_slow_end": chronological["slow"][1:],
            "v_fast_start": chronological["fast"][:-1],
            "v_fast_end": chronological["fast"][1:],
            "log_return": returns,
            "slow_return_component": factor_returns["slow"],
            "fast_return_component": factor_returns["fast"],
        }
    )


def path_observables(history: pd.DataFrame) -> dict[str, float]:
    values = history["log_return"].to_numpy(float)
    if len(values) != HISTORY_TRADING_DAYS:
        raise ValueError("path observables require the frozen 252-day history")
    rv_short = TRADING_DAYS_PER_YEAR * float(np.mean(values[-SHORT_RV_WINDOW:] ** 2))
    rv_long = TRADING_DAYS_PER_YEAR * float(np.mean(values[-LONG_RV_WINDOW:] ** 2))
    usable = values[-(HISTORY_TRADING_DAYS // PERSISTENCE_BLOCK_DAYS) * PERSISTENCE_BLOCK_DAYS :]
    blocks = np.sum(usable.reshape(-1, PERSISTENCE_BLOCK_DAYS) ** 2, axis=1)
    left = blocks[:-PERSISTENCE_LAG_BLOCKS]
    right = blocks[PERSISTENCE_LAG_BLOCKS:]
    persistence = float(np.corrcoef(left, right)[0, 1])
    if not np.isfinite(persistence):
        raise RuntimeError("synthetic persistence statistic is non-finite")
    return {
        "log_rv_21": float(math.log(rv_short)),
        "log_rv_126": float(math.log(rv_long)),
        "rv_block_persistence": persistence,
        "rv_21": rv_short,
        "rv_126": rv_long,
    }


def truth_fit_diagnostics(path_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in path_frame.itertuples(index=False):
        residuals = {
            "log_rv_21": (
                float(row.expected_log_rv_21) - float(row.log_rv_21)
            )
            / LOG_REALIZED_VARIANCE_SCALE,
            "log_rv_126": (
                float(row.expected_log_rv_126) - float(row.log_rv_126)
            )
            / LOG_REALIZED_VARIANCE_SCALE,
            "rv_block_persistence": (
                float(row.expected_rv_block_persistence)
                - float(row.rv_block_persistence)
            )
            / PERSISTENCE_SCALE,
        }
        for design_id, names in (
            ("C", ("log_rv_21", "log_rv_126")),
            ("D", ("log_rv_21", "log_rv_126", "rv_block_persistence")),
        ):
            rmse = float(np.sqrt(np.mean([residuals[name] ** 2 for name in names])))
            rows.append(
                {
                    "case_id": row.case_id,
                    "sample_id": row.sample_id,
                    "design_id": design_id,
                    "true_parameter_complementary_rmse_standardized": rmse,
                    "truth_passes_complementary_screen": bool(
                        rmse <= COMPLEMENTARY_NEAR_EQUIVALENCE_RMSE
                    ),
                }
            )
    return pd.DataFrame(rows)


def _expected_trailing_variance(
    kappa: float, theta: float, current_variance: float, window: int
) -> float:
    lags = np.arange(1, window + 1, dtype=float) * DT
    return float(theta + (current_variance - theta) * np.mean(np.exp(-kappa * lags)))


def expected_persistence(parameters: Sequence[float]) -> float:
    """Approximate ACF of non-overlapping 5-day squared-return blocks at lag 4."""
    vector = np.asarray(parameters, dtype=float)
    components = []
    for factor in ("slow", "fast"):
        kappa_i, theta_i, sigma_i, _, _ = _factor_indices(factor)
        kappa = vector[kappa_i]
        stationary_variance = vector[theta_i] * vector[sigma_i] ** 2 / (2.0 * kappa)
        components.append((kappa, stationary_variance))
    total_mean = vector[1] + vector[6]
    total_variance = sum(item[1] for item in components)
    diagonal = 2.0 * total_mean**2 + 3.0 * total_variance
    block_variance = PERSISTENCE_BLOCK_DAYS * diagonal
    for lag in range(1, PERSISTENCE_BLOCK_DAYS):
        covariance = sum(a * math.exp(-k * lag * DT) for k, a in components)
        block_variance += 2.0 * (PERSISTENCE_BLOCK_DAYS - lag) * covariance
    separation = PERSISTENCE_LAG_DAYS
    block_covariance = 0.0
    for left in range(PERSISTENCE_BLOCK_DAYS):
        for right in range(PERSISTENCE_BLOCK_DAYS):
            lag = separation + right - left
            block_covariance += sum(
                a * math.exp(-k * lag * DT) for k, a in components
            )
    return float(block_covariance / block_variance)


def model_features(parameters: Sequence[float]) -> dict[str, float]:
    vector = np.asarray(parameters, dtype=float)
    rv_short = _expected_trailing_variance(vector[0], vector[1], vector[4], SHORT_RV_WINDOW)
    rv_short += _expected_trailing_variance(vector[5], vector[6], vector[9], SHORT_RV_WINDOW)
    rv_long = _expected_trailing_variance(vector[0], vector[1], vector[4], LONG_RV_WINDOW)
    rv_long += _expected_trailing_variance(vector[5], vector[6], vector[9], LONG_RV_WINDOW)
    return {
        "oracle_total_variance": float(vector[4] + vector[9]),
        "log_rv_21": float(math.log(rv_short)),
        "log_rv_126": float(math.log(rv_long)),
        "rv_block_persistence": expected_persistence(vector),
    }


def observed_features(
    true_parameters: Sequence[float],
    path_values: dict[str, float],
    design: Design,
    case_index: int,
    noise_level: float,
) -> np.ndarray:
    values = model_features(true_parameters)
    values.update({name: path_values[name] for name in ("log_rv_21", "log_rv_126", "rv_block_persistence")})
    if noise_level:
        rng = np.random.default_rng(_case_seed(case_index, 41, noise_level))
        values["log_rv_21"] += float(rng.normal(0.0, LOG_RV_NOISE_SD[noise_level]))
        values["log_rv_126"] += float(rng.normal(0.0, LOG_RV_NOISE_SD[noise_level]))
        values["rv_block_persistence"] = float(
            np.clip(
                values["rv_block_persistence"]
                + rng.normal(0.0, PERSISTENCE_NOISE_SD[noise_level]),
                -0.99,
                0.99,
            )
        )
    return np.asarray([values[name] for name in design.feature_names], dtype=float)


def _model_feature_vector(parameters: Sequence[float], design: Design) -> np.ndarray:
    values = model_features(parameters)
    return np.asarray([values[name] for name in design.feature_names], dtype=float)


def _scaled_model_vector(
    parameters: Sequence[float],
    design: Design,
    maturity_days: Sequence[int],
    *,
    node_count: int,
) -> np.ndarray:
    prices = baseline.normalized_observables(
        parameters, baseline.REPRESENTATIONS[0], maturity_days, node_count=node_count
    )
    price_piece = prices / (OPTION_NEAR_EQUIVALENCE_RMSE * math.sqrt(len(prices)))
    if not design.feature_names:
        return price_piece
    features = _model_feature_vector(parameters, design)
    scales = np.asarray([FEATURE_SCALES[name] for name in design.feature_names])
    return np.concatenate((price_piece, features / (scales * math.sqrt(len(features)))))


def scaled_jacobian(
    parameters: Sequence[float],
    design: Design,
    maturity_days: Sequence[int],
    bounds: dict[str, tuple[float, float]],
    *,
    node_count: int,
) -> np.ndarray:
    vector = np.asarray(parameters, dtype=float)
    widths = _parameter_widths(bounds)
    columns = []
    for index, width in enumerate(widths):
        step = baseline.JACOBIAN_RELATIVE_STEP * width
        for _ in range(12):
            lower = vector.copy()
            upper = vector.copy()
            lower[index] -= step
            upper[index] += step
            if validate_parameters(lower)["is_valid"] and validate_parameters(upper)["is_valid"]:
                break
            step *= 0.5
        else:
            raise RuntimeError(f"Could not form valid difference for {PARAMETER_NAMES[index]}")
        lower_values = _scaled_model_vector(
            lower, design, maturity_days, node_count=node_count
        )
        upper_values = _scaled_model_vector(
            upper, design, maturity_days, node_count=node_count
        )
        columns.append((upper_values - lower_values) * width / (2.0 * step))
    return np.column_stack(columns)


def run_identifiability(
    cases: pd.DataFrame,
    bounds: dict[str, tuple[float, float]],
    *,
    node_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    singular_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    for row in cases.itertuples(index=False):
        parameters = np.asarray([getattr(row, name) for name in PARAMETER_NAMES])
        maturity_days = (int(row.near_dte), int(row.middle_dte))
        for design in DESIGNS:
            jacobian = scaled_jacobian(
                parameters, design, maturity_days, bounds, node_count=node_count
            )
            _, singular_values, right = np.linalg.svd(jacobian, full_matrices=False)
            cutoff = singular_values[0] * PRACTICAL_RANK_RELATIVE_TOLERANCE
            practical_rank = int(np.sum(singular_values >= cutoff))
            algebraic_rank = int(np.linalg.matrix_rank(jacobian))
            condition = float(singular_values[0] / singular_values[-1])
            summary_rows.append(
                {
                    "case_id": row.case_id,
                    "sample_id": row.sample_id,
                    "design_id": design.design_id,
                    "algebraic_rank": algebraic_rank,
                    "practical_rank": practical_rank,
                    "smallest_singular_value": float(singular_values[-1]),
                    "largest_singular_value": float(singular_values[0]),
                    "condition_number": condition,
                }
            )
            for index, value in enumerate(singular_values, start=1):
                singular_rows.append(
                    {"case_id": row.case_id, "design_id": design.design_id, "singular_index": index, "singular_value": float(value)}
                )
            sensitivities = np.linalg.norm(jacobian, axis=0)
            weakest = right[-1]
            for index, name in enumerate(PARAMETER_NAMES):
                sensitivity_rows.append(
                    {"case_id": row.case_id, "design_id": design.design_id, "parameter": name, "scaled_sensitivity": float(sensitivities[index])}
                )
                direction_rows.append(
                    {"case_id": row.case_id, "design_id": design.design_id, "parameter": name, "weakest_direction_loading": float(weakest[index]), "absolute_loading": float(abs(weakest[index]))}
                )
    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(singular_rows),
        pd.DataFrame(sensitivity_rows),
        pd.DataFrame(direction_rows),
    )


def _prior_solutions(cases: pd.DataFrame) -> pd.DataFrame:
    if not PRIOR_SOLUTIONS_PATH.is_file():
        raise FileNotFoundError(f"Missing preserved Design A evidence: {PRIOR_SOLUTIONS_PATH}")
    prior = pd.read_csv(PRIOR_SOLUTIONS_PATH)
    return prior.loc[prior["case_id"].isin(cases["case_id"])].copy()


def _case_truth(row: Any) -> np.ndarray:
    return np.asarray([getattr(row, f"true_{name}") for name in PARAMETER_NAMES], dtype=float)


def _noise_observed_prices(
    truth: np.ndarray, case_index: int, maturity_days: Sequence[int], noise_level: float, node_count: int
) -> np.ndarray:
    clean = baseline.normalized_observables(
        truth, baseline.REPRESENTATIONS[0], maturity_days, node_count=node_count
    )
    return ambiguity.multiplicative_noise(clean, case_index, noise_level)


def _record_solution(
    base: dict[str, Any],
    recovered: np.ndarray,
    truth: np.ndarray,
    predicted_prices: np.ndarray,
    observed_prices: np.ndarray,
    predicted_features: np.ndarray,
    observed_complement: np.ndarray,
    design: Design,
    bounds: dict[str, tuple[float, float]],
    optimizer_success: bool,
    optimizer_status: int,
    nfev: int,
) -> dict[str, Any]:
    widths = _parameter_widths(bounds)
    displacement = (recovered - truth) / widths
    price_rmse = float(np.sqrt(np.mean((predicted_prices - observed_prices) ** 2)))
    if design.feature_names:
        scales = np.asarray([FEATURE_SCALES[name] for name in design.feature_names])
        complement_residual = (predicted_features - observed_complement) / scales
        complement_rmse = float(np.sqrt(np.mean(complement_residual**2)))
    else:
        complement_residual = np.asarray([], dtype=float)
        complement_rmse = math.nan
    price_group_rmse = price_rmse / OPTION_NEAR_EQUIVALENCE_RMSE
    joint_objective = float(
        math.sqrt(
            (price_group_rmse**2 + (complement_rmse**2 if design.feature_names else 0.0))
            / (2.0 if design.feature_names else 1.0)
        )
    )
    validation = validate_parameters(recovered)
    bound_reasons = boundary_diagnostics(recovered, bounds)
    near = bool(
        validation["is_valid"]
        and np.isfinite(recovered).all()
        and price_rmse <= OPTION_NEAR_EQUIVALENCE_RMSE
        and (
            not design.feature_names
            or complement_rmse <= COMPLEMENTARY_NEAR_EQUIVALENCE_RMSE
        )
    )
    record = {
        **base,
        "optimizer_success": optimizer_success,
        "optimizer_status": optimizer_status,
        "nfev": nfev,
        "constraint_valid": bool(validation["is_valid"]),
        "finite_solution": bool(np.isfinite(recovered).all()),
        "bound_hit": bool(bound_reasons),
        "bound_reasons": ";".join(bound_reasons),
        "price_rmse_normalized": price_rmse,
        "complementary_rmse_standardized": complement_rmse,
        "joint_objective_rmse": joint_objective,
        "parameter_rmse_full_range": float(np.sqrt(np.mean(displacement**2))),
        "maximum_absolute_parameter_error_full_range": float(np.max(np.abs(displacement))),
        "material_displacement": bool(np.sqrt(np.mean(displacement**2)) >= MATERIAL_DISPLACEMENT_RMSE),
        "near_equivalent": near,
    }
    for index, name in enumerate(PARAMETER_NAMES):
        record[f"true_{name}"] = float(truth[index])
        record[f"recovered_{name}"] = float(recovered[index])
        record[f"scaled_{name}"] = float(_scaled_coordinates(recovered, bounds)[index])
        record[f"scaled_displacement_{name}"] = float(displacement[index])
    for index, name in enumerate(design.feature_names):
        record[f"observed_{name}"] = float(observed_complement[index])
        record[f"predicted_{name}"] = float(predicted_features[index])
        record[f"standardized_residual_{name}"] = float(complement_residual[index])
    return record


def run_recovery(
    cases: pd.DataFrame,
    paths: dict[str, dict[str, float]],
    bounds: dict[str, tuple[float, float]],
    *,
    node_count: int,
    skip_combined: bool = False,
) -> pd.DataFrame:
    prior = _prior_solutions(cases)
    noisy_best = (
        prior.loc[prior["noise_level"].gt(0.0)]
        .sort_values(
            ["case_id", "noise_level", "price_rmse_normalized", "start_index"],
            kind="stable",
        )
        .groupby(["case_id", "noise_level"], sort=True)
        .head(NOISY_WARM_START_COUNT)
    )
    noisy_schedule = set(
        zip(
            noisy_best["case_id"],
            noisy_best["noise_level"],
            noisy_best["start_index"],
            strict=True,
        )
    )
    records: list[dict[str, Any]] = []
    for row in prior.itertuples(index=False):
        truth = _case_truth(row)
        maturity_days = (int(row.near_dte), int(row.middle_dte))
        observed_prices = _noise_observed_prices(
            truth, int(row.case_index), maturity_days, float(row.noise_level), node_count
        )
        recovered_a = np.asarray([getattr(row, f"recovered_{name}") for name in PARAMETER_NAMES])
        predicted_a = baseline.normalized_observables(
            recovered_a, baseline.REPRESENTATIONS[0], maturity_days, node_count=node_count
        )
        base = {
            "case_id": row.case_id,
            "case_index": int(row.case_index),
            "sample_id": row.sample_id,
            "distribution": row.distribution,
            "maturity_profile": row.maturity_profile,
            "noise_level": float(row.noise_level),
            "start_index": int(row.start_index),
            "start_strategy": row.start_strategy,
            "design_id": "A",
            "warm_start_source": "PRESERVED_G2_GLOBAL_AMBIGUITY",
        }
        records.append(
            _record_solution(
                base,
                recovered_a,
                truth,
                predicted_a,
                observed_prices,
                np.asarray([]),
                np.asarray([]),
                DESIGN_BY_ID["A"],
                bounds,
                bool(row.optimizer_success),
                int(row.optimizer_status),
                int(row.nfev),
            )
        )
        clean_ambiguity_warm_start = bool(
            float(row.noise_level) == 0.0
            and bool(row.constraint_valid)
            and bool(row.finite_solution)
            and float(row.price_rmse_normalized) <= OPTION_NEAR_EQUIVALENCE_RMSE
        )
        noisy_warm_start = (
            row.case_id,
            float(row.noise_level),
            int(row.start_index),
        ) in noisy_schedule
        if skip_combined or not (clean_ambiguity_warm_start or noisy_warm_start):
            continue
        for design in DESIGNS[1:]:
            observed_complement = observed_features(
                truth,
                paths[row.case_id],
                design,
                int(row.case_index),
                float(row.noise_level),
            )
            scales = np.asarray([FEATURE_SCALES[name] for name in design.feature_names])

            def residuals(latent: np.ndarray) -> np.ndarray:
                candidate = unconstrained_to_parameters(latent, bounds)
                prices = baseline.normalized_observables(
                    candidate, baseline.REPRESENTATIONS[0], maturity_days, node_count=node_count
                )
                price_piece = (prices - observed_prices) / (
                    OPTION_NEAR_EQUIVALENCE_RMSE * math.sqrt(len(prices))
                )
                features = _model_feature_vector(candidate, design)
                complement_piece = (features - observed_complement) / (
                    scales * math.sqrt(len(features))
                )
                return np.concatenate((price_piece, complement_piece))

            combined_base = {
                **base,
                "design_id": design.design_id,
                "warm_start_source": (
                    "ESTABLISHED_CLEAN_NEAR_EQUIVALENT_SOLUTION"
                    if clean_ambiguity_warm_start
                    else "TOP_FIVE_NOISY_OPTION_FIT"
                ),
            }
            try:
                start = parameters_to_unconstrained(recovered_a, bounds)
                result = least_squares(
                    residuals,
                    start,
                    method="trf",
                    max_nfev=MAX_NFEV,
                    ftol=OPTIMIZER_TOLERANCE,
                    xtol=OPTIMIZER_TOLERANCE,
                    gtol=OPTIMIZER_TOLERANCE,
                    diff_step=DIFF_STEP,
                )
                recovered = unconstrained_to_parameters(result.x, bounds)
                predicted_prices = baseline.normalized_observables(
                    recovered, baseline.REPRESENTATIONS[0], maturity_days, node_count=node_count
                )
                predicted_features = _model_feature_vector(recovered, design)
                records.append(
                    _record_solution(
                        combined_base,
                        recovered,
                        truth,
                        predicted_prices,
                        observed_prices,
                        predicted_features,
                        observed_complement,
                        design,
                        bounds,
                        bool(result.success),
                        int(result.status),
                        int(result.nfev),
                    )
                )
            except Exception as error:
                records.append(
                    {
                        **combined_base,
                        "optimizer_success": False,
                        "optimizer_status": -1,
                        "nfev": 0,
                        "constraint_valid": False,
                        "finite_solution": False,
                        "bound_hit": False,
                        "bound_reasons": "",
                        "price_rmse_normalized": math.nan,
                        "complementary_rmse_standardized": math.nan,
                        "joint_objective_rmse": math.nan,
                        "parameter_rmse_full_range": math.nan,
                        "maximum_absolute_parameter_error_full_range": math.nan,
                        "material_displacement": False,
                        "near_equivalent": False,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
    return pd.DataFrame(records)


def summarize_recovery(
    solutions: pd.DataFrame, bounds: dict[str, tuple[float, float]]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    recovery_rows: list[dict[str, Any]] = []
    ambiguity_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    usable = solutions.loc[solutions["constraint_valid"] & solutions["finite_solution"]].copy()
    for (design_id, noise_level), group in usable.groupby(["design_id", "noise_level"], sort=True):
        best = group.sort_values(["case_id", "joint_objective_rmse", "start_index"], kind="stable").groupby("case_id", sort=True).head(1)
        recovery_rows.append(
            {
                "design_id": design_id,
                "noise_level": noise_level,
                "case_count": int(best["case_id"].nunique()),
                "optimizer_success_count": int(group["optimizer_success"].sum()),
                "usable_solution_count": int(len(group)),
                "bound_hit_count": int(group["bound_hit"].sum()),
                "best_solution_parameter_rmse_median": float(best["parameter_rmse_full_range"].median()),
                "best_solution_parameter_rmse_maximum": float(best["parameter_rmse_full_range"].max()),
                "best_solution_max_parameter_error_median": float(best["maximum_absolute_parameter_error_full_range"].median()),
                "best_solution_price_rmse_median": float(best["price_rmse_normalized"].median()),
                "best_solution_complementary_rmse_median": float(best["complementary_rmse_standardized"].median()) if design_id != "A" else math.nan,
            }
        )
        for row in best.itertuples(index=False):
            for name in PARAMETER_NAMES:
                parameter_rows.append(
                    {
                        "design_id": design_id,
                        "noise_level": noise_level,
                        "case_id": row.case_id,
                        "parameter": name,
                        "absolute_error_full_range": abs(float(getattr(row, f"scaled_displacement_{name}"))),
                    }
                )
        near = group.loc[group["near_equivalent"]].copy()
        cluster_count = 0
        materially_displaced = 0
        if not near.empty:
            for _, case_group in near.groupby("case_id", sort=True):
                matrix = case_group[[f"scaled_{name}" for name in PARAMETER_NAMES]].to_numpy(float)
                labels = ambiguity.complete_linkage_clusters(matrix, CLUSTER_DISTANCE_CUTOFF)
                cluster_count += int(len(np.unique(labels)))
            materially_displaced = int(near["material_displacement"].sum())
        ambiguity_rows.append(
            {
                "design_id": design_id,
                "noise_level": noise_level,
                "near_equivalent_solution_count": int(len(near)),
                "materially_displaced_solution_count": materially_displaced,
                "scaled_parameter_cluster_count": cluster_count,
                "near_equivalent_case_count": int(near["case_id"].nunique()) if len(near) else 0,
                "near_equivalent_parameter_rmse_median": float(near["parameter_rmse_full_range"].median()) if len(near) else math.nan,
                "near_equivalent_parameter_rmse_maximum": float(near["parameter_rmse_full_range"].max()) if len(near) else math.nan,
                "near_equivalent_bound_hit_count": int(near["bound_hit"].sum()) if len(near) else 0,
                "near_equivalent_price_rmse_median": float(near["price_rmse_normalized"].median()) if len(near) else math.nan,
                "near_equivalent_complementary_rmse_median": float(near["complementary_rmse_standardized"].median()) if len(near) and design_id != "A" else math.nan,
            }
        )
    return pd.DataFrame(recovery_rows), pd.DataFrame(ambiguity_rows), pd.DataFrame(parameter_rows)


def add_matched_noise_comparison(
    recovery: pd.DataFrame, solutions: pd.DataFrame
) -> pd.DataFrame:
    """Add clean/noisy metrics on the exact cases available at every noise level."""
    usable = solutions.loc[
        solutions["constraint_valid"] & solutions["finite_solution"]
    ].copy()
    best = (
        usable.sort_values(
            ["design_id", "noise_level", "case_id", "joint_objective_rmse", "start_index"],
            kind="stable",
        )
        .groupby(["design_id", "noise_level", "case_id"], sort=True)
        .head(1)
    )
    expected_groups = {
        (design.design_id, noise_level)
        for design in DESIGNS
        for noise_level in NOISE_LEVELS
    }
    case_sets = {
        (design_id, float(noise_level)): set(group["case_id"])
        for (design_id, noise_level), group in best.groupby(
            ["design_id", "noise_level"], sort=True
        )
    }
    if set(case_sets) != expected_groups:
        raise ValueError("Clean/noisy comparison is missing a design/noise group")
    shared_cases = set.intersection(*(case_sets[key] for key in sorted(case_sets)))
    if shared_cases != set(NOISE_COMPARISON_CASE_IDS):
        raise ValueError(
            "Clean/noisy shared case population differs from the frozen case_1/case_3 contract"
        )

    result = recovery.copy()
    comparison_ids = ";".join(NOISE_COMPARISON_CASE_IDS)
    for index, row in result.iterrows():
        mask = (
            best["design_id"].eq(row["design_id"])
            & best["noise_level"].eq(float(row["noise_level"]))
            & best["case_id"].isin(NOISE_COMPARISON_CASE_IDS)
        )
        matched_best = best.loc[mask]
        solution_mask = (
            usable["design_id"].eq(row["design_id"])
            & usable["noise_level"].eq(float(row["noise_level"]))
            & usable["case_id"].isin(NOISE_COMPARISON_CASE_IDS)
        )
        matched_solutions = usable.loc[solution_mask]
        if set(matched_best["case_id"]) != set(NOISE_COMPARISON_CASE_IDS):
            raise ValueError("Clean/noisy comparison does not contain both frozen cases")
        result.loc[index, "noise_comparison_case_ids"] = comparison_ids
        result.loc[index, "noise_comparison_case_count"] = len(NOISE_COMPARISON_CASE_IDS)
        result.loc[index, "noise_comparison_parameter_rmse_median"] = float(
            matched_best["parameter_rmse_full_range"].median()
        )
        result.loc[index, "noise_comparison_parameter_rmse_maximum"] = float(
            matched_best["parameter_rmse_full_range"].max()
        )
        result.loc[index, "noise_comparison_bound_hit_count"] = int(
            matched_solutions["bound_hit"].sum()
        )
        result.loc[index, "noise_comparison_usable_solution_count"] = int(
            len(matched_solutions)
        )
    result["noise_comparison_case_count"] = result[
        "noise_comparison_case_count"
    ].astype(int)
    result["noise_comparison_bound_hit_count"] = result[
        "noise_comparison_bound_hit_count"
    ].astype(int)
    result["noise_comparison_usable_solution_count"] = result[
        "noise_comparison_usable_solution_count"
    ].astype(int)
    return result


def market_feasibility() -> pd.DataFrame:
    """Audit the existing official-NSE framework without acquiring new data."""
    raw_root = DEFAULT_STAGE_A_ROOT / "raw" / "nse"
    available_dates = sorted(
        path.parent.name
        for path in raw_root.glob("*/BhavCopy_NSE_CM_0_0_0_*_F_0000.csv.zip")
    )
    rows = []
    for stock in MARKET_SECURITIES:
        for valuation_date in MARKET_VALUATION_DATES:
            start = (pd.Timestamp(valuation_date) - pd.offsets.BDay(HISTORY_TRADING_DAYS)).date().isoformat()
            eligible = [item for item in available_dates if start <= item <= valuation_date]
            rows.append(
                {
                    "security": stock,
                    "valuation_date": valuation_date,
                    "required_trading_day_lookback": HISTORY_TRADING_DAYS,
                    "earliest_required_weekday_proxy": start,
                    "available_canonical_cm_dates_in_window": len(eligible),
                    "required_close_field": "official NSE CM EQ ClsPric",
                    "corporate_action_handling": "UNIMPLEMENTED_IN_STAGE_A",
                    "missing_day_policy": "UNRESOLVED_FAIL_CLOSED",
                    "return_construction": "log(adjusted_close_t/adjusted_close_t_minus_1)",
                    "provenance": "OFFICIAL_NSE_UDIFF_ARCHIVE_PLUS_HASH_MANIFEST_CAPABLE",
                    "causal_by_valuation_date": True,
                    "new_data_acquired": False,
                    "observable_contract": "UNRESOLVED",
                }
            )
    return pd.DataFrame(rows)


def classify_decision(
    recovery: pd.DataFrame,
    ambiguity_summary: pd.DataFrame,
    market: pd.DataFrame,
    truth_diagnostics: pd.DataFrame,
) -> dict[str, Any]:
    clean_recovery = recovery.loc[recovery["noise_level"].eq(0.0)].set_index("design_id")
    clean_ambiguity = ambiguity_summary.loc[ambiguity_summary["noise_level"].eq(0.0)].set_index("design_id")
    baseline_clusters = float(clean_ambiguity.loc["A", "scaled_parameter_cluster_count"])
    baseline_material = float(clean_ambiguity.loc["A", "materially_displaced_solution_count"])
    baseline_error = float(clean_recovery.loc["A", "best_solution_parameter_rmse_median"])
    rule = predeclared_contract()["decision_rule"]
    candidate_flags: dict[str, bool] = {}
    truth_pass_counts: dict[str, int] = {}
    truth_panel_validity: dict[str, bool] = {}
    for design_id in ("C", "D"):
        truth_group = truth_diagnostics.loc[
            truth_diagnostics["design_id"].eq(design_id)
        ]
        truth_panel_validity[design_id] = bool(
            len(truth_group) == len(EXPECTED_CASE_IDS)
            and not truth_group["case_id"].duplicated().any()
            and set(truth_group["case_id"]) == set(EXPECTED_CASE_IDS)
        )
        truth_pass_counts[design_id] = int(
            truth_group.loc[
                truth_group["truth_passes_complementary_screen"].astype(bool),
                "case_id",
            ].nunique()
        )
        clusters = float(clean_ambiguity.loc[design_id, "scaled_parameter_cluster_count"])
        material = float(clean_ambiguity.loc[design_id, "materially_displaced_solution_count"])
        error = float(clean_recovery.loc[design_id, "best_solution_parameter_rmse_median"])
        candidate_flags[design_id] = bool(
            truth_panel_validity[design_id]
            and truth_pass_counts[design_id] >= rule["minimum_distinct_case_count"]
            and clean_ambiguity.loc[design_id, "near_equivalent_case_count"]
            >= rule["minimum_distinct_case_count"]
            and clusters <= rule["cluster_ratio_max"] * baseline_clusters
            and material <= rule["material_solution_ratio_max"] * baseline_material
            and error <= rule["parameter_error_ratio_max"] * baseline_error
        )
    truth_panel_canonical = all(truth_panel_validity.values())
    if not truth_panel_canonical:
        candidate_flags = {design_id: False for design_id in candidate_flags}
    information_value = any(candidate_flags.values())
    experiment_valid = truth_panel_canonical and any(
        truth_panel_validity[design_id]
        and truth_pass_counts[design_id] >= rule["minimum_distinct_case_count"]
        for design_id in ("C", "D")
    )
    expected_market_panel = {
        (security, valuation_date)
        for security in MARKET_SECURITIES
        for valuation_date in MARKET_VALUATION_DATES
    }
    market_columns_present = {
        "security",
        "valuation_date",
        "observable_contract",
    }.issubset(market.columns)
    market_panel_valid = bool(
        market_columns_present
        and len(market) == len(expected_market_panel)
        and not market[["security", "valuation_date"]].duplicated().any()
        and set(zip(market["security"], market["valuation_date"], strict=True))
        == expected_market_panel
    )
    market_ready = bool(
        market_panel_valid and market["observable_contract"].eq("RESOLVED").all()
    )
    if information_value and market_ready:
        verdict = "PROMISING"
    elif information_value:
        verdict = "INFORMATION_ONLY"
    else:
        verdict = "INSUFFICIENT"
    if not truth_panel_canonical:
        experiment_validity = "NOT_PASSED_NONCANONICAL_TRUTH_PANEL"
    elif experiment_valid:
        experiment_validity = "PASSED"
    else:
        experiment_validity = "NOT_PASSED_TRUTH_OUTSIDE_COMPLEMENTARY_SCREEN"
    return {
        "complementary_observable": verdict,
        "verdict_interpretation": (
            "CURRENT_EXPERIMENT_EVIDENCE_INSUFFICIENT_NOT_INTRINSIC_OBSERVABLE_IMPOSSIBILITY"
            if not experiment_valid
            else "PREDECLARED_INFORMATION_VALUE_RULE_APPLIED"
        ),
        "experiment_validity": experiment_validity,
        "truth_screen_pass_count_by_design": {
            key: int(value) for key, value in truth_pass_counts.items()
        },
        "truth_panel_validity_by_design": truth_panel_validity,
        "market_panel_valid": market_panel_valid,
        "information_value_trigger_by_design": candidate_flags,
        "market_observable_contract": "RESOLVED" if market_ready else "UNRESOLVED",
        "global_ambiguity": "ESTABLISHED",
        "g2": "NOT_PASSED",
        "g2_market_supported_geometry": "ESTABLISHED",
        "g2_final_representation": "NOT_FROZEN",
        "final_10k_dataset": "NOT_GENERATED",
        "ann_training": "NOT_STARTED",
        "pinn_training": "NOT_STARTED",
        "decision_rule": rule,
    }


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _validate_output_paths(output_root: Path, report_path: Path) -> None:
    repository = REPOSITORY_ROOT.resolve()
    output = output_root.resolve()
    report = report_path.resolve()
    default_output = DEFAULT_OUTPUT_ROOT.resolve()
    default_report = DEFAULT_REPORT_PATH.resolve()
    if output != default_output and (output == repository or repository in output.parents):
        raise ValueError("Custom output_root must be outside the repository")
    if report != default_report and (report == repository or repository in report.parents):
        raise ValueError("Custom report_path must be outside the repository")
    protected_roots = (
        (REPOSITORY_ROOT / "docs" / "evidence").resolve(),
        (REPOSITORY_ROOT / "scripts").resolve(),
        (REPOSITORY_ROOT / "tests").resolve(),
    )
    if any(output == root or root in output.parents for root in protected_roots):
        raise ValueError("output_root overlaps protected G2 source/evidence paths")
    if report != default_report and any(
        report == root or root in report.parents for root in protected_roots
    ):
        raise ValueError("report_path overlaps protected G2 source/evidence paths")


def _validate_run_mode(
    output_root: Path,
    report_path: Path,
    *,
    node_count: int,
    sample_limit: int | None,
    skip_recovery: bool,
) -> None:
    canonical = (
        node_count == FULL_PRICER_NODE_COUNT
        and sample_limit is None
        and not skip_recovery
    )
    if not canonical and (
        output_root.resolve() == DEFAULT_OUTPUT_ROOT.resolve()
        or report_path.resolve() == DEFAULT_REPORT_PATH.resolve()
    ):
        raise ValueError(
            "Noncanonical node/sample/recovery modes require external output_root and report_path"
        )


def _validate_replay_evidence(
    frames: dict[str, pd.DataFrame],
    contract: dict[str, Any],
    decision: dict[str, Any],
    predeclared_path: Path,
) -> None:
    if contract.get("node_count") != FULL_PRICER_NODE_COUNT:
        raise ValueError("Replay evidence is not the frozen 64-node run")
    if _sha256(predeclared_path) != contract.get("predeclared_contract_sha256"):
        raise ValueError("Predeclared contract hash does not match canonical contract")
    source_contract = json.loads(json.dumps(predeclared_contract()))
    if json.loads(predeclared_path.read_text(encoding="utf-8")) != source_contract:
        raise ValueError("Predeclared contract contents do not match source contract")
    matrix = frames["experiment_matrix.csv"]
    if matrix["design_id"].tolist() != ["A", "B", "C", "D"]:
        raise ValueError("Replay experiment matrix is not canonical A/B/C/D")
    cases = frames["cases.csv"]
    if (
        len(cases) != len(EXPECTED_CASE_IDS)
        or cases["case_id"].duplicated().any()
        or set(cases["case_id"]) != set(EXPECTED_CASE_IDS)
    ):
        raise ValueError("Replay cases are not the exact four-case panel")
    jacobian = frames["jacobian_summary.csv"]
    expected_jacobian = {
        (case_id, design_id)
        for case_id in EXPECTED_CASE_IDS
        for design_id in ("A", "B", "C", "D")
    }
    if (
        len(jacobian) != len(expected_jacobian)
        or jacobian[["case_id", "design_id"]].duplicated().any()
        or set(zip(jacobian["case_id"], jacobian["design_id"], strict=True))
        != expected_jacobian
    ):
        raise ValueError("Replay Jacobian evidence has noncanonical cardinality")
    truth = frames["truth_fit_diagnostics.csv"]
    expected_truth = {
        (case_id, design_id)
        for case_id in EXPECTED_CASE_IDS
        for design_id in ("C", "D")
    }
    if (
        len(truth) != len(expected_truth)
        or truth[["case_id", "design_id"]].duplicated().any()
        or set(zip(truth["case_id"], truth["design_id"], strict=True))
        != expected_truth
    ):
        raise ValueError("Replay truth-fit evidence has noncanonical cardinality")
    recovery = frames["recovery_summary.csv"]
    comparison_columns = (
        "noise_comparison_case_ids",
        "noise_comparison_case_count",
        "noise_comparison_parameter_rmse_median",
        "noise_comparison_parameter_rmse_maximum",
        "noise_comparison_bound_hit_count",
        "noise_comparison_usable_solution_count",
    )
    if not set(comparison_columns).issubset(recovery.columns):
        raise ValueError("Replay recovery evidence lacks matched clean/noisy metrics")
    expected_recovery = add_matched_noise_comparison(
        recovery, frames["recovery_solutions.csv"]
    )
    for column in comparison_columns:
        stored = recovery[column].reset_index(drop=True)
        expected = expected_recovery[column].reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(expected):
            matches = np.allclose(stored.to_numpy(float), expected.to_numpy(float), equal_nan=True)
        else:
            matches = stored.astype(str).equals(expected.astype(str))
        if not matches:
            raise ValueError(f"Replay matched clean/noisy metric mismatch for {column}")
    recomputed = classify_decision(
        frames["recovery_summary.csv"],
        frames["ambiguity_summary.csv"],
        frames["market_feasibility.csv"],
        truth,
    )
    for key in (
        "complementary_observable",
        "experiment_validity",
        "verdict_interpretation",
        "truth_screen_pass_count_by_design",
        "truth_panel_validity_by_design",
        "market_panel_valid",
        "information_value_trigger_by_design",
        "decision_rule",
    ):
        stored_value = json.loads(json.dumps(decision.get(key), sort_keys=True))
        recomputed_value = json.loads(json.dumps(recomputed.get(key), sort_keys=True))
        if stored_value != recomputed_value:
            raise ValueError(f"Replay decision mismatch for {key}")


def write_figures(
    output_root: Path,
    jacobian: pd.DataFrame,
    recovery: pd.DataFrame,
    ambiguity_summary: pd.DataFrame,
    parameter_errors: pd.DataFrame,
    solutions: pd.DataFrame,
) -> None:
    clean_j = jacobian.groupby("design_id", sort=False).agg(
        condition_number=("condition_number", "median"), practical_rank=("practical_rank", "median")
    ).reindex([item.design_id for item in DESIGNS])
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.bar(clean_j.index, clean_j["condition_number"], color="#3C78D8")
    ax.set_yscale("log")
    ax.set_ylabel("Median scaled-Jacobian condition number (log scale)")
    ax.set_title("A/B/C/D conditioning comparison")
    for index, value in enumerate(clean_j["practical_rank"]):
        ax.text(
            index,
            clean_j["condition_number"].iloc[index] * 1.18,
            f"rank {value:.1f}/10",
            ha="center",
        )
    ax.set_ylim(top=float(clean_j["condition_number"].max()) * 1.65)
    _save_figure(fig, output_root / FIGURE_ARTIFACTS[0])

    clean_errors = parameter_errors.loc[parameter_errors["noise_level"].eq(0.0)]
    pivot = clean_errors.groupby(["design_id", "parameter"])["absolute_error_full_range"].median().unstack().reindex([item.design_id for item in DESIGNS])[PARAMETER_NAMES]
    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis", vmin=0.0)
    ax.set_xticks(range(len(PARAMETER_NAMES)), PARAMETER_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title("Median absolute ten-parameter recovery error (full-range scale)")
    fig.colorbar(image, ax=ax, label="Absolute scaled error")
    _save_figure(fig, output_root / FIGURE_ARTIFACTS[1])

    clean_a = ambiguity_summary.loc[ambiguity_summary["noise_level"].eq(0.0)].set_index("design_id").reindex([item.design_id for item in DESIGNS])
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    x = np.arange(len(clean_a))
    ax.bar(x - 0.18, clean_a["scaled_parameter_cluster_count"], 0.36, label="Clusters")
    ax.bar(x + 0.18, clean_a["materially_displaced_solution_count"], 0.36, label="Material near-equivalent solutions")
    ax.set_xticks(x, clean_a.index)
    ax.set_ylabel("Count across qualifying solutions/clusters")
    ax.set_title("Global ambiguity under the frozen joint-fit screen")
    ax.legend()
    for index, design_id in enumerate(clean_a.index):
        if clean_a.loc[design_id, "near_equivalent_solution_count"] == 0:
            ax.text(index, 1.0, "0 qualifying fits\nnot ambiguity resolution", ha="center", va="bottom")
    _save_figure(fig, output_root / FIGURE_ARTIFACTS[2])

    near = solutions.loc[solutions["noise_level"].eq(0.0) & solutions["near_equivalent"]].copy()
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.8), sharex=True, sharey=True)
    for ax, design in zip(axes, DESIGNS, strict=True):
        group = near.loc[near["design_id"].eq(design.design_id)]
        ax.scatter(group["recovered_v0_slow"], group["recovered_v0_fast"], s=18, alpha=0.7)
        truths = solutions.loc[solutions["design_id"].eq(design.design_id), ["case_id", "true_v0_slow", "true_v0_fast"]].drop_duplicates()
        ax.scatter(truths["true_v0_slow"], truths["true_v0_fast"], marker="x", s=55, color="black")
        if group.empty:
            ax.text(
                0.5,
                0.48,
                "No jointly\nnear-equivalent fits",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#9C0006",
            )
        ax.set_title(f"Design {design.design_id}")
        ax.set_xlabel("v0_slow")
    axes[0].set_ylabel("v0_fast")
    fig.suptitle("Slow/fast current-variance allocation (x = truth)")
    _save_figure(fig, output_root / FIGURE_ARTIFACTS[3])

    subset = clean_errors.loc[clean_errors["parameter"].isin(("theta_slow", "theta_fast", "kappa_slow", "kappa_fast"))]
    values = subset.groupby(["design_id", "parameter"])["absolute_error_full_range"].median().unstack().reindex([item.design_id for item in DESIGNS])
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    values.plot(kind="bar", ax=ax)
    ax.set_ylabel("Median absolute error (full-range scale)")
    ax.set_title("Theta / kappa information by design")
    ax.tick_params(axis="x", rotation=0)
    _save_figure(fig, output_root / FIGURE_ARTIFACTS[4])

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    for design in [item.design_id for item in DESIGNS]:
        group = recovery.loc[recovery["design_id"].eq(design)].sort_values("noise_level")
        ax.plot(
            100.0 * group["noise_level"],
            group["noise_comparison_parameter_rmse_median"],
            marker="o",
            label=design,
        )
    ax.set_xlabel("Option-price noise (%)")
    ax.set_ylabel("Median best-solution parameter RMSE")
    ax.set_title("Matched-case clean vs noisy recovery (case_1 and case_3; n=2)")
    ax.legend(title="Design")
    _save_figure(fig, output_root / FIGURE_ARTIFACTS[5])


def _protected_snapshot(output_root: Path, report_path: Path) -> dict[str, str]:
    protected: dict[str, str] = {}
    for path in sorted(DEFAULT_STAGE_A_ROOT.rglob("*")):
        if path.is_file() and path != output_root and output_root not in path.parents:
            protected[path.relative_to(REPOSITORY_ROOT).as_posix()] = _sha256(path)
    for path in sorted((REPOSITORY_ROOT / "docs").glob("G2_*.md")):
        if path != report_path:
            protected[path.relative_to(REPOSITORY_ROOT).as_posix()] = _sha256(path)
    for directory, pattern in (
        (REPOSITORY_ROOT / "docs" / "evidence", "G2_*.json"),
        (REPOSITORY_ROOT / "scripts", "run_g2_*.py"),
        (REPOSITORY_ROOT / "tests", "test_g2_*.py"),
    ):
        for path in sorted(directory.glob(pattern)):
            if path not in (Path(__file__).resolve(), CURRENT_TEST_PATH):
                protected[path.relative_to(REPOSITORY_ROOT).as_posix()] = _sha256(path)
    return protected


def _snapshot_aggregate(snapshot: dict[str, str]) -> str:
    payload = "".join(f"{name}\0{value}\n" for name, value in sorted(snapshot.items())).encode()
    return hashlib.sha256(payload).hexdigest().upper()


def _assert_protected_unchanged(before: dict[str, str], output_root: Path, report_path: Path) -> None:
    after = _protected_snapshot(output_root, report_path)
    if before != after:
        changed = sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))
        raise RuntimeError(f"Prior Stage A/G2 evidence changed unexpectedly: {changed}")


def render_report(
    contract: dict[str, Any],
    jacobian: pd.DataFrame,
    recovery: pd.DataFrame,
    ambiguity_summary: pd.DataFrame,
    parameter_errors: pd.DataFrame,
    market: pd.DataFrame,
    decision: dict[str, Any],
    truth_diagnostics: pd.DataFrame,
    artifact_hashes: dict[str, str],
) -> str:
    j = jacobian.groupby("design_id", sort=False).agg(
        practical_rank=("practical_rank", "median"),
        smallest=("smallest_singular_value", "median"),
        condition=("condition_number", "median"),
    ).reindex([item.design_id for item in DESIGNS])
    clean_r = recovery.loc[recovery["noise_level"].eq(0.0)].set_index("design_id").reindex(j.index)
    clean_a = ambiguity_summary.loc[ambiguity_summary["noise_level"].eq(0.0)].set_index("design_id").reindex(j.index)
    lines = [
        "# G2 Complementary-Observable Analysis",
        "",
        "## Decision",
        "",
        f"**COMPLEMENTARY_OBSERVABLE = {decision['complementary_observable']}**",
        "",
        f"**EXPERIMENT_VALIDITY = {decision['experiment_validity']}**",
        "",
        "`INSUFFICIENT` means this current experiment did not establish usable information value. It does **not** prove that properly sampling-aware realized-variance observables are intrinsically incapable of helping.",
        "",
        "**G2 = NOT_PASSED — STRUCTURAL IDENTIFIABILITY PROBLEM REMAINS.** The final representation is not frozen.",
        "",
        "## Scientific observable contract",
        "",
        f"- `RV_SHORT`: annualized squared close-to-close log returns over the last `{SHORT_RV_WINDOW}` trading days.",
        f"- `RV_LONG`: the same statistic over `{LONG_RV_WINDOW}` trading days.",
        f"- `RV_PERSISTENCE`: correlation of non-overlapping `{PERSISTENCE_BLOCK_DAYS}`-day realized-variance blocks separated by `{PERSISTENCE_LAG_BLOCKS}` blocks (`{PERSISTENCE_LAG_DAYS}` trading days), estimated from a `{HISTORY_TRADING_DAYS}`-day causal history.",
        "- `ORACLE_TOTAL_VARIANCE`: `v0_slow + v0_fast`; this is an information upper bound, not a claimed market observation.",
        "",
        "The 21/126-day pair was frozen because the four representative fast-factor e-folding times are about 34-68 trading days: 21 days is shorter, while 126 days is longer and begins to expose slow reversion. No alternative windows, lags, indicators, or feature combinations were searched.",
        "",
        "For a CIR factor, the conditional trailing population variance under stationary reversibility is `theta + (v0-theta) * mean_j exp(-kappa*j/252)`. The persistence moment uses `Cov(v_t,v_(t+h)) = theta*sigma^2/(2*kappa) * exp(-kappa*h)` aggregated over the declared blocks, with the Gaussian squared-return variance in the denominator.",
        "",
        "Primary references: [Heston (1993)](https://doi.org/10.1093/rfs/6.2.327), [Cox, Ingersoll and Ross (1985)](https://doi.org/10.2307/1911242), and [Andersen, Bollerslev, Diebold and Labys (2003)](https://doi.org/10.1111/1468-0262.00418).",
        "",
        "One persistence scalar is only a weighted mixture of the two decay modes; it cannot identify both kappas by itself. The bounded D design follows the requested one-statistic limit and tests incremental information only. Bollerslev and Zhou's two-factor integrated-variance result motivates the decay mixture but does not make one sample autocorrelation an exact two-root estimator ([primary paper](https://doi.org/10.1016/S0304-4076(01)00141-5)).",
        "",
        "## Synthetic construction and leakage boundary",
        "",
        f"Each case has an exact-CIR-marginal `{HISTORY_TRADING_DAYS}`-day variance history ending at the same `v0_slow/v0_fast` used by its option surface. Frozen seed `{ANALYSIS_SEED}` and case-derived seeds are recorded in `contract.json`. Returns use a disclosed daily Gaussian-copula leverage approximation. All windows end at valuation time; no future return enters an observable.",
        "",
        "The historical path seed and shocks are never supplied to calibration. Calibration maps candidate parameters to population moments; therefore it does not replay the truth path or infer its innovations. The clean path retains intrinsic finite-history sampling variation. Added robustness noise is separate: option prices 0.5%/1.0% multiplicative; log-RV 0.05/0.10 standard deviation; persistence 0.05/0.10 absolute standard deviation. The oracle remains exact because it is explicitly an upper-bound diagnostic.",
        "",
        "The synthetic experiment deliberately uses the same structural vector for return and option dynamics. In market language this is a `P=Q`/zero variance-risk-premium diagnostic assumption. Historical kappa/theta are physical-measure evidence, whereas the option engine is risk-neutral; no physical-to-risk-neutral bridge is established here, so this assumption cannot freeze a market input contract.",
        "",
        "## A/B/C/D local identifiability",
        "",
        "| Design | Median practical rank | Median smallest singular value | Median condition number |",
        "|---|---:|---:|---:|",
    ]
    for design_id, row in j.iterrows():
        lines.append(f"| {design_id} | {row.practical_rank:.1f}/10 | {row.smallest:.3e} | {row.condition:.3e} |")
    lines.extend([
        "",
        "Rows are group-balanced and statistically scaled before SVD: the unchanged normalized-price equivalence scale, 0.10 log-RV, 0.10 persistence, and 1e-4 oracle total variance. Parameters remain scaled by full hard-bound widths exactly as in prior G2 work.",
        "",
        "## Attempted clean global recovery and ambiguity",
        "",
        "| Design | Median best parameter RMSE | Maximum best parameter RMSE | Near-equivalent | Materially displaced | Clusters | Median price RMSE | Median complementary RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for design_id in j.index:
        r = clean_r.loc[design_id]
        a = clean_a.loc[design_id]
        price = (
            "no qualifying fit"
            if not np.isfinite(a.near_equivalent_price_rmse_median)
            else f"{a.near_equivalent_price_rmse_median:.3e}"
        )
        comp = (
            "n/a"
            if design_id == "A"
            else "no qualifying fit"
            if not np.isfinite(a.near_equivalent_complementary_rmse_median)
            else f"{a.near_equivalent_complementary_rmse_median:.3e}"
        )
        lines.append(
            f"| {design_id} | {r.best_solution_parameter_rmse_median:.3e} | {r.best_solution_parameter_rmse_maximum:.3e} | {int(a.near_equivalent_solution_count)} | {int(a.materially_displaced_solution_count)} | {int(a.scaled_parameter_cluster_count)} | {price} | {comp} |"
        )
    lines.extend(
        [
            "",
            "Zero qualifying C/D fits do **not** mean the ambiguity disappeared. They mean none of the reoptimized established option-equivalent regions simultaneously met the unchanged price threshold and the declared complementary-observable tolerance. The clean best complementary RMSE medians were 3.098 for C and 2.558 for D, both above the 1.0 screen.",
            "",
            "The oracle B diagnostic lowered median best-fit parameter RMSE from 0.1214 to 0.0518, but retained 33 clusters and 29 materially displaced near-equivalent solutions; its cluster and material-solution reductions (15% and 26%) missed the 50% global thresholds. D lowered the median condition number from 6.556e8 to 1.641e6 (about 400x) and raised median practical rank from 7.5 to 9.5, yet median best-fit parameter RMSE worsened to 0.1998. This is local information gain without global recovery.",
        ]
    )
    lines.extend(
        [
            "",
            "### Truth-fit validity diagnostic",
            "",
            "| Case | Design | True-parameter complementary RMSE | Truth passes <=1 screen |",
            "|---|---|---:|---|",
        ]
    )
    for row in truth_diagnostics.itertuples(index=False):
        lines.append(
            f"| {row.case_id} | {row.design_id} | {row.true_parameter_complementary_rmse_standardized:.3f} | {bool(row.truth_passes_complementary_screen)} |"
        )
    lines.extend(
        [
            "",
            "The true vector fails the complementary screen in every C/D case because one finite 252-day path is compared with population moments using fixed 0.10 scales. Therefore C/D zero-fit and clustering rows are **invalid for global ambiguity inference**. They are retained as evidence that this observation-model/scaling contract is not fit for the decision, not as evidence that the underlying observables cannot help.",
        ]
    )
    clean_errors = parameter_errors.loc[parameter_errors["noise_level"].eq(0.0)]
    focused = clean_errors.loc[clean_errors["parameter"].isin(("v0_slow", "v0_fast", "theta_slow", "theta_fast", "kappa_slow", "kappa_fast"))]
    pivot = focused.groupby(["design_id", "parameter"])["absolute_error_full_range"].median().unstack().reindex(j.index)
    lines.extend([
        "",
        "### Slow/fast, theta, and kappa errors",
        "",
        "Median absolute full-range-scaled errors of the target-blind best-fit solution:",
        "",
        "| Design | v0_slow | v0_fast | theta_slow | theta_fast | kappa_slow | kappa_fast |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for design_id, row in pivot.iterrows():
        lines.append("| " + design_id + " | " + " | ".join(f"{row[name]:.3e}" for name in ("v0_slow", "v0_fast", "theta_slow", "theta_fast", "kappa_slow", "kappa_fast")) + " |")
    lines.extend([
        "",
        "## Noise robustness",
        "",
        "The comparison below uses the identical `case_1;case_3` population (`n=2`) at every clean/noisy point. The four-case clean recovery and ambiguity evidence above remains the decision basis and is not replaced by this matched sensitivity view.",
        "",
        "| Design | Option noise | Matched cases (n) | RV log-noise SD | Persistence noise SD | Median best parameter RMSE | Max best parameter RMSE | Bound hits |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ])
    for row in recovery.sort_values(["design_id", "noise_level"]).itertuples(index=False):
        lines.append(
            f"| {row.design_id} | {100*row.noise_level:.1f}% | {row.noise_comparison_case_ids} ({int(row.noise_comparison_case_count)}) | {LOG_RV_NOISE_SD[float(row.noise_level)]:.2f} | {PERSISTENCE_NOISE_SD[float(row.noise_level)]:.2f} | {row.noise_comparison_parameter_rmse_median:.3e} | {row.noise_comparison_parameter_rmse_maximum:.3e} | {int(row.noise_comparison_bound_hit_count)}/{int(row.noise_comparison_usable_solution_count)} |"
        )
    lines.extend(
        [
            "",
            "On the matched two-case population, at 0.5% and 1.0% option noise median best-fit parameter RMSE was about 0.36-0.37 and 0.40-0.41 across all designs; every usable B/C/D 1.0% solution hit a declared bound. The strict clean price-equivalence threshold retained no noisy solutions, so noisy cluster counts remain unresolved rather than zero ambiguity.",
            "",
            "Across the 180 scheduled B/C/D fits, 174 produced valid finite capped iterates and six failed with the pricer's declared degenerate-denominator guard. No clean B/C/D fit satisfied SciPy's convergence termination within `max_nfev=80`; valid capped iterates were retained under the same evidence rule as the established global analysis. This limits any uniqueness claim and reinforces the fail-closed verdict.",
        ]
    )
    lines.extend([
        "",
        "## Real-market feasibility",
        "",
        f"`MARKET_OBSERVABLE_CONTRACT = {decision['market_observable_contract']}`.",
        "",
        f"The predeclared maximum lookback is `{HISTORY_TRADING_DAYS}` trading days. The checkout contains only the three canonical valuation-date CM archives, not a continuous history. The official-NSE UDiFF acquisition/hash framework is reusable, and `ClsPric` is the declared close, but corporate-action adjustment, exchange-holiday completeness, missing-day rules, and a replayable adjusted-close contract are not implemented. No new data was acquired because the bounded feasibility decision is already unresolved and bulk acquisition would not cure those missing policies.",
        "",
        "All four securities remain separate in `market_feasibility.csv`: NTPC, CIPLA, INFY, and HDFCBANK. Every requested return window is causal by construction, but it is not market-admissible until adjustment and completeness rules pass.",
        "",
        "## Decision rule and boundary",
        "",
        "A market design is materially informative only if, versus A, it reduces clean clusters and materially displaced near-equivalent solutions by at least 50%, lowers median best-fit parameter RMSE by at least 25%, and retains a near-equivalent fit in at least three of four cases. This rule was frozen before calibration.",
        "",
        f"Design triggers: `{decision['information_value_trigger_by_design']}`. Therefore `COMPLEMENTARY_OBSERVABLE = {decision['complementary_observable']}`.",
        "",
        "**Mentor-ready numerical conclusion:** the exact total-variance oracle improved point recovery but left most separated option-equivalent regions intact. The sampled 21/126-day RV plus one persistence design cannot be judged globally because its fixed scales reject the truth in 4/4 cases; its apparent local conditioning gain does not rescue that invalid observation contract. The current experiment is insufficient evidence, not a proof of intrinsic observable insufficiency.",
        "",
        "This experiment does not change the ten-parameter target, impose priors, reparameterize the model, generate the final 10k dataset, or train ANN/PINN. It does not modify prior G2 or checkpoint artifacts. G2 remains `NOT_PASSED` regardless of local conditioning gains.",
        "",
        "**Single recommended next action:** predeclare and run a new sampling-aware synthetic design using empirically justified finite-window likelihood/scales and multiple fixed path seeds before any market-data acquisition.",
        "",
        "## Reproducibility",
        "",
        "- Canonical command: `python -B scripts/run_g2_complementary_observable_analysis.py`.",
        f"- Source-defined pre-run contract SHA-256: `{contract['predeclared_contract_sha256']}`; it is written before calibration as `predeclared_contract.json`.",
        f"- Seed: `{ANALYSIS_SEED}`; node count: `{contract['node_count']}`; optimizer `max_nfev={MAX_NFEV}`.",
        f"- Protected prior files: `{contract['protected_snapshot']['file_count']}`; aggregate SHA-256 `{contract['protected_snapshot']['aggregate_sha256']}`.",
        "- Design A comes from the preserved global-ambiguity CSV; B-D use those recovered vectors as target-blind warm starts.",
        "- B-D reoptimize all 40 established clean near-equivalent A solutions (39 clusters); noisy runs use the five lowest-price-RMSE prior A solutions per matched case/noise level. This schedule was frozen after a no-output runtime timeout and before any B-D result existed.",
        "",
        "## Mentor-ready figures",
        "",
        "1. Conditioning comparison.",
        "2. Ten-parameter error comparison.",
        "3. Global ambiguity/clustering comparison.",
        "4. Slow/fast variance allocation.",
        "5. Theta/kappa information.",
        "6. Matched-case clean versus noisy recovery (`case_1;case_3`, `n=2`).",
        "",
        "## Artifact hashes",
        "",
        "| Artifact | SHA-256 |",
        "|---|---|",
    ])
    lines.extend(f"| `{name}` | `{digest}` |" for name, digest in sorted(artifact_hashes.items()))
    lines.extend([
        "",
        "```text",
        f"COMPLEMENTARY_OBSERVABLE = {decision['complementary_observable']}",
        f"EXPERIMENT_VALIDITY = {decision['experiment_validity']}",
        f"MARKET_OBSERVABLE_CONTRACT = {decision['market_observable_contract']}",
        "G2 = NOT_PASSED",
        "FINAL_REPRESENTATION = NOT_FROZEN",
        "FINAL_10K_DATASET = NOT_GENERATED",
        "ANN_TRAINING = NOT_STARTED",
        "PINN_TRAINING = NOT_STARTED",
        "```",
        "",
    ])
    return "\n".join(lines)


def run_analysis(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    report_path: Path = DEFAULT_REPORT_PATH,
    node_count: int = FULL_PRICER_NODE_COUNT,
    sample_limit: int | None = None,
    skip_recovery: bool = False,
) -> dict[str, Any]:
    _validate_run_mode(
        output_root,
        report_path,
        node_count=node_count,
        sample_limit=sample_limit,
        skip_recovery=skip_recovery,
    )
    _validate_output_paths(output_root, report_path)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(predeclared_contract(), output_root / "predeclared_contract.json")
    before = _protected_snapshot(output_root, report_path)
    bounds = load_hard_safety_bounds(baseline.BOUNDS_PATH)
    cases = ambiguity.select_cases(bounds)
    if sample_limit is not None:
        cases = cases.head(sample_limit).copy()
    histories = []
    path_rows = []
    path_values: dict[str, dict[str, float]] = {}
    for row in cases.itertuples(index=False):
        parameters = np.asarray([getattr(row, name) for name in PARAMETER_NAMES])
        history = simulate_causal_return_history(parameters, int(row.case_index))
        history.insert(0, "case_id", row.case_id)
        histories.append(history)
        values = path_observables(history)
        expected = model_features(parameters)
        values.update(
            {
                "case_id": row.case_id,
                "sample_id": row.sample_id,
                **{f"expected_{name}": value for name, value in expected.items()},
            }
        )
        path_rows.append(values)
        path_values[row.case_id] = values
    history_frame = pd.concat(histories, ignore_index=True)
    path_frame = pd.DataFrame(path_rows)
    truth_diagnostics = truth_fit_diagnostics(path_frame)
    jacobian, singular_values, sensitivities, directions = run_identifiability(
        cases, bounds, node_count=node_count
    )
    solutions = run_recovery(
        cases, path_values, bounds, node_count=node_count, skip_combined=skip_recovery
    )
    recovery, ambiguity_summary, parameter_errors = summarize_recovery(solutions, bounds)
    if not skip_recovery:
        recovery = add_matched_noise_comparison(recovery, solutions)
    market = market_feasibility()
    if skip_recovery:
        decision = {
            "complementary_observable": "NOT_CLASSIFIED_QUICK_RUN",
            "market_observable_contract": "UNRESOLVED",
            "global_ambiguity": "ESTABLISHED",
            "g2": "NOT_PASSED",
            "information_value_trigger_by_design": {},
        }
    else:
        decision = classify_decision(
            recovery, ambiguity_summary, market, truth_diagnostics
        )
    contract = {
        "analysis_id": ANALYSIS_ID,
        "analysis_seed": ANALYSIS_SEED,
        "predeclared_contract_sha256": _predeclared_contract_sha256(),
        "node_count": node_count,
        "canonical_parameter_order": PARAMETER_NAMES,
        "windows": {
            "short_rv_trading_days": SHORT_RV_WINDOW,
            "long_rv_trading_days": LONG_RV_WINDOW,
            "history_trading_days": HISTORY_TRADING_DAYS,
            "persistence_block_days": PERSISTENCE_BLOCK_DAYS,
            "persistence_lag_blocks": PERSISTENCE_LAG_BLOCKS,
        },
        "noise": {
            "option_levels": NOISE_LEVELS,
            "log_rv_sd": LOG_RV_NOISE_SD,
            "persistence_sd": PERSISTENCE_NOISE_SD,
            "oracle_noise": "NONE_UPPER_BOUND_DIAGNOSTIC",
        },
        "thresholds": {
            "practical_rank_relative": PRACTICAL_RANK_RELATIVE_TOLERANCE,
            "price_near_equivalence_rmse": OPTION_NEAR_EQUIVALENCE_RMSE,
            "complementary_near_equivalence_standardized_rmse": COMPLEMENTARY_NEAR_EQUIVALENCE_RMSE,
            "material_parameter_rmse": MATERIAL_DISPLACEMENT_RMSE,
            "cluster_distance": CLUSTER_DISTANCE_CUTOFF,
        },
        "replay_state": {
            "design_a": "PRESERVED_NOT_RERUN",
            "design_b_c_d": "ALL_CLEAN_NEAR_EQUIVALENT_A_SOLUTIONS_PLUS_TOP_FIVE_NOISY_A_FITS",
            "clean_combined_warm_start_count": 40 if sample_limit is None else None,
            "noisy_warm_start_count_per_case_level": NOISY_WARM_START_COUNT,
            "sample_ids": cases["sample_id"].tolist(),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "protected_snapshot": {
            "file_count": len(before),
            "aggregate_sha256": _snapshot_aggregate(before),
        },
    }
    frames = {
        "experiment_matrix.csv": experiment_matrix(),
        "cases.csv": cases,
        "synthetic_return_history.csv": history_frame,
        "path_observables.csv": path_frame,
        "truth_fit_diagnostics.csv": truth_diagnostics,
        "jacobian_summary.csv": jacobian,
        "singular_values.csv": singular_values,
        "parameter_sensitivities.csv": sensitivities,
        "weakest_directions.csv": directions,
        "recovery_solutions.csv": solutions,
        "recovery_summary.csv": recovery,
        "ambiguity_summary.csv": ambiguity_summary,
        "parameter_errors.csv": parameter_errors,
        "market_feasibility.csv": market,
    }
    for relative, frame in frames.items():
        _write_csv(frame, output_root / relative)
    _write_json(contract, output_root / "contract.json")
    _write_json(decision, output_root / "decision.json")
    if not skip_recovery:
        write_figures(output_root, jacobian, recovery, ambiguity_summary, parameter_errors, solutions)
    artifact_names = DATA_ARTIFACTS + (() if skip_recovery else FIGURE_ARTIFACTS)
    artifact_hashes = {name: _sha256(output_root / name) for name in artifact_names}
    if skip_recovery:
        report = (
            "# G2 Complementary-Observable Quick Diagnostic\n\n"
            "This reduced replay checks deterministic construction only. It does not "
            "classify information value or change G2.\n"
        )
    else:
        report = render_report(
            contract,
            jacobian,
            recovery,
            ambiguity_summary,
            parameter_errors,
            market,
            decision,
            truth_diagnostics,
            artifact_hashes,
        )
    _atomic_write_bytes(report_path, report.encode("utf-8"))
    _assert_protected_unchanged(before, output_root, report_path)
    return {
        "decision": decision,
        "artifact_hashes": artifact_hashes,
        "report_sha256": _sha256(report_path),
        "output_root": str(output_root),
        "report_path": str(report_path),
    }


def render_existing_outputs(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    _validate_output_paths(output_root, report_path)
    required = DATA_ARTIFACTS
    missing = [name for name in required if not (output_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing canonical replay artifacts: {missing}")
    before = _protected_snapshot(output_root, report_path)
    frames = {
        name: pd.read_csv(output_root / name)
        for name in DATA_ARTIFACTS
        if name.endswith(".csv")
    }
    contract = json.loads((output_root / "contract.json").read_text(encoding="utf-8"))
    decision = json.loads((output_root / "decision.json").read_text(encoding="utf-8"))
    _validate_replay_evidence(
        frames,
        contract,
        decision,
        output_root / "predeclared_contract.json",
    )
    write_figures(
        output_root,
        frames["jacobian_summary.csv"],
        frames["recovery_summary.csv"],
        frames["ambiguity_summary.csv"],
        frames["parameter_errors.csv"],
        frames["recovery_solutions.csv"],
    )
    artifact_hashes = {
        name: _sha256(output_root / name) for name in DATA_ARTIFACTS + FIGURE_ARTIFACTS
    }
    report = render_report(
        contract,
        frames["jacobian_summary.csv"],
        frames["recovery_summary.csv"],
        frames["ambiguity_summary.csv"],
        frames["parameter_errors.csv"],
        frames["market_feasibility.csv"],
        decision,
        frames["truth_fit_diagnostics.csv"],
        artifact_hashes,
    )
    _atomic_write_bytes(report_path, report.encode("utf-8"))
    _assert_protected_unchanged(before, output_root, report_path)
    return {
        "decision": decision,
        "artifact_hashes": artifact_hashes,
        "report_sha256": _sha256(report_path),
        "output_root": str(output_root),
        "report_path": str(report_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--node-count", type=int, default=FULL_PRICER_NODE_COUNT)
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--skip-recovery", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if args.render_only:
        result = render_existing_outputs(
            output_root=args.output_root, report_path=args.report_path
        )
    else:
        result = run_analysis(
            output_root=args.output_root,
            report_path=args.report_path,
            node_count=args.node_count,
            sample_limit=args.sample_limit,
            skip_recovery=args.skip_recovery,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
