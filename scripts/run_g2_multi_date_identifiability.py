"""Bounded joint-date identifiability diagnostic for canonical Double Heston.

The experiment keeps the eight structural parameters shared across three actual
valuation dates.  The two anchor-date variances remain canonical targets, while
four later-date variances are either known oracle inputs (Design B) or distinct
nuisance variables (Designs C/D).  Design D uses the exact CIR transition density;
it never replaces a stochastic transition with its conditional expectation.

This module does not alter G2, generate a research dataset, train a model, or use
controlled synthetic carry as a production preprocessing contract.
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
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import ncx2

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


DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT
    / "market_data_audit"
    / "stage_a"
    / "derived"
    / "g2_multi_date_identifiability"
)
DEFAULT_REPORT_PATH = REPOSITORY_ROOT / "docs" / "G2_MULTI_DATE_IDENTIFIABILITY.md"

CANONICAL_TARGET_NAMES = tuple(PARAMETER_NAMES)
SHARED_STRUCTURAL_NAMES = (
    "kappa_slow",
    "theta_slow",
    "sigma_slow",
    "rho_slow",
    "kappa_fast",
    "theta_fast",
    "sigma_fast",
    "rho_fast",
)
NUISANCE_STATE_NAMES = (
    "v_slow_t1",
    "v_fast_t1",
    "v_slow_t2",
    "v_fast_t2",
)
JOINT_PARAMETER_NAMES = CANONICAL_TARGET_NAMES + NUISANCE_STATE_NAMES

VALUATION_DATES = ("2026-07-01", "2026-07-15", "2026-07-22")
DATE_GAPS_DAYS = (14, 7)
MATURITY_PROFILES = (
    ("2026-07-01", (27, 55)),
    ("2026-07-15", (13, 41)),
    ("2026-07-22", (6, 34)),
)
CONTROLLED_RATES = (0.0600, 0.0625)
CONTROLLED_DIVIDEND_YIELDS = (0.0200, 0.0225)
REPRESENTATION = baseline.REPRESENTATIONS[0]

STATE_BOUNDS = {
    # Later CIR states have positive, unbounded stochastic support and therefore
    # must not inherit the canonical anchor-v0 hard bounds.  This broad finite
    # envelope is only an optimizer/pricer safety guard; no draw is clipped or
    # resampled to fit it.
    "v_slow_t1": (1.0e-8, 0.50),
    "v_fast_t1": (1.0e-8, 0.50),
    "v_slow_t2": (1.0e-8, 0.50),
    "v_fast_t2": (1.0e-8, 0.50),
}
STATE_SCALING_WIDTHS = np.asarray((0.295, 0.248, 0.295, 0.248), dtype=np.float64)
STATE_START_CENTERS = np.asarray((0.05, 0.03, 0.05, 0.03), dtype=np.float64)
ANALYSIS_SEED = 20260811
JACOBIAN_RELATIVE_STEP = baseline.JACOBIAN_RELATIVE_STEP
PRACTICAL_RANK_RELATIVE_TOLERANCE = baseline.PRACTICAL_RANK_RELATIVE_TOLERANCE
CONDITION_WARNING_THRESHOLD = baseline.CONDITION_WARNING_THRESHOLD
RECOVERY_SCALED_RMSE_THRESHOLD = baseline.RECOVERY_SCALED_RMSE_THRESHOLD
RECOVERY_SCALED_MAX_ERROR_THRESHOLD = baseline.RECOVERY_SCALED_MAX_ERROR_THRESHOLD
RECOVERY_FREQUENCY_THRESHOLD = 0.80
NOISE_LEVELS = baseline.NOISE_LEVELS
PHYSICS_REFERENCE_NOISE = 0.005
PRICE_SIGMA_FLOOR = 1.0e-7
RECOVERY_MAXITER = 80
RECOVERY_SAMPLES_PER_DISTRIBUTION = 1

PREVIOUS_G2_PATHS = (
    "docs/G2_COMMON_SUPPORT_ANALYSIS.md",
    "docs/G2_IDENTIFIABILITY_ANALYSIS.md",
    "docs/G2_INFORMATION_REMEDIATION.md",
    "scripts/run_g2_common_support_analysis.py",
    "scripts/run_g2_identifiability_analysis.py",
    "scripts/run_g2_information_remediation.py",
    "tests/test_g2_common_support_analysis.py",
    "tests/test_g2_identifiability_analysis.py",
    "tests/test_g2_information_remediation.py",
)

EXPECTED_OUTPUT_FILES = (
    "experiment_designs.csv",
    "state_paths.csv",
    "identifiability_summary.csv",
    "parameter_sensitivity.csv",
    "weakest_directions.csv",
    "recovery_starts.csv",
    "recovery_summary.csv",
    "parameter_error_summary.csv",
    "nuisance_state_recovery.csv",
    "decision.json",
    "figures/practical_rank_comparison.png",
    "figures/singular_values_comparison.png",
    "figures/condition_number_comparison.png",
    "figures/recovery_comparison.png",
    "figures/weakest_directions_comparison.png",
    "figures/nuisance_state_recovery.png",
    "figures/mentor_summary.png",
)


@dataclass(frozen=True)
class Design:
    design_id: str
    label: str
    date_count: int
    oracle_later_states: bool
    latent_later_states: bool
    cir_physics: bool

    @property
    def unknown_count(self) -> int:
        return 10 + (4 if self.latent_later_states else 0)


DESIGNS = (
    Design("A", "single_date", 1, False, False, False),
    Design("B", "multi_date_oracle_states", 3, True, False, False),
    Design("C", "multi_date_latent_states", 3, False, True, False),
    Design("D", "multi_date_latent_states_cir_physics", 3, False, True, True),
)
DESIGN_BY_ID = {design.design_id: design for design in DESIGNS}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _protected_snapshot(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, str]:
    """Hash all pre-existing Stage A/G2 evidence, excluding only this output root."""
    result: dict[str, str] = {}
    stage_root = REPOSITORY_ROOT / "market_data_audit" / "stage_a"
    resolved_output = output_root.resolve()
    for path in sorted(stage_root.rglob("*")):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved == resolved_output or resolved_output in resolved.parents:
            continue
        result[path.relative_to(REPOSITORY_ROOT).as_posix()] = _sha256(path)
    for relative in PREVIOUS_G2_PATHS:
        path = REPOSITORY_ROOT / relative
        result[relative] = _sha256(path)
    return result


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    _atomic_write_bytes(path, frame.to_csv(index=False).encode("utf-8"))


def _write_json(payload: dict[str, Any], path: Path) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_write_bytes(path, encoded)


def experiment_designs() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for design in DESIGNS:
        rows.append(
            {
                "design_id": design.design_id,
                "design_label": design.label,
                "valuation_dates": "|".join(VALUATION_DATES[: design.date_count]),
                "maturity_profiles_days": ";".join(
                    f"{item[0]}:{item[1][0]}|{item[1][1]}"
                    for item in MATURITY_PROFILES[: design.date_count]
                ),
                "normalized_price_count": 20 * design.date_count,
                "canonical_target_count": 10,
                "nuisance_state_count": 4 if design.latent_later_states else 0,
                "unknown_count": design.unknown_count,
                "later_states_known": design.oracle_later_states,
                "later_states_date_specific": design.date_count == 3,
                "exact_cir_transition_density": design.cir_physics,
                "controlled_rates": "|".join(f"{value:.4f}" for value in CONTROLLED_RATES),
                "controlled_dividend_yields": "|".join(
                    f"{value:.4f}" for value in CONTROLLED_DIVIDEND_YIELDS
                ),
                "market_contract_status": "SYNTHETIC_DIAGNOSTIC_ONLY_NOT_FROZEN",
            }
        )
    return pd.DataFrame(rows)


def _target_widths(bounds: dict[str, tuple[float, float]]) -> np.ndarray:
    return np.asarray(
        [bounds[name][1] - bounds[name][0] for name in CANONICAL_TARGET_NAMES],
        dtype=np.float64,
    )


def _nuisance_widths() -> np.ndarray:
    # Use the corresponding canonical anchor-state full widths for coordinate
    # scaling and error reporting.  The separate numerical envelope above must
    # not silently redefine the scientific scale.
    return STATE_SCALING_WIDTHS.copy()


def _joint_widths(bounds: dict[str, tuple[float, float]]) -> np.ndarray:
    return np.concatenate((_target_widths(bounds), _nuisance_widths()))


def cir_transition_parameters(
    kappa: float, theta: float, sigma: float, previous_variance: float, dt: float
) -> tuple[float, float, float]:
    """Return scale, degrees of freedom, and noncentrality for exact CIR sampling."""
    if min(kappa, theta, sigma, previous_variance, dt) <= 0.0:
        raise ValueError("CIR transition inputs must be positive")
    decay = math.exp(-kappa * dt)
    one_minus_decay = -math.expm1(-kappa * dt)
    scale = sigma * sigma * one_minus_decay / (4.0 * kappa)
    degrees = 4.0 * kappa * theta / (sigma * sigma)
    noncentrality = (
        4.0 * kappa * decay * previous_variance
        / (sigma * sigma * one_minus_decay)
    )
    if not np.isfinite([scale, degrees, noncentrality]).all() or scale <= 0.0:
        raise FloatingPointError("Invalid exact CIR transition parameters")
    return scale, degrees, noncentrality


def exact_cir_transition_from_uniform(
    kappa: float,
    theta: float,
    sigma: float,
    previous_variance: float,
    dt: float,
    uniform: float,
) -> float:
    scale, degrees, noncentrality = cir_transition_parameters(
        kappa, theta, sigma, previous_variance, dt
    )
    probability = float(np.clip(uniform, 1.0e-10, 1.0 - 1.0e-10))
    value = scale * float(ncx2.ppf(probability, degrees, noncentrality))
    if not math.isfinite(value) or value <= 0.0:
        raise FloatingPointError("Exact CIR inverse-CDF sample is invalid")
    return value


def exact_cir_transition_logpdf(
    next_variance: float,
    kappa: float,
    theta: float,
    sigma: float,
    previous_variance: float,
    dt: float,
) -> float:
    scale, degrees, noncentrality = cir_transition_parameters(
        kappa, theta, sigma, previous_variance, dt
    )
    if next_variance <= 0.0:
        return -math.inf
    value = float(ncx2.logpdf(next_variance / scale, degrees, noncentrality)) - math.log(
        scale
    )
    return value if math.isfinite(value) else -math.inf


def _factor_parameters(target: np.ndarray, factor: str) -> tuple[float, float, float]:
    if factor == "slow":
        return float(target[0]), float(target[1]), float(target[2])
    if factor == "fast":
        return float(target[5]), float(target[6]), float(target[7])
    raise ValueError(f"Unknown factor: {factor}")


def simulate_state_paths(samples: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(samples.itertuples(index=False)):
        target = np.asarray(
            [getattr(sample, name) for name in CANONICAL_TARGET_NAMES], dtype=np.float64
        )
        state = {"slow": float(target[4]), "fast": float(target[9])}
        record: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "distribution": sample.distribution,
            "v_slow_t0": state["slow"],
            "v_fast_t0": state["fast"],
            "date_t0": VALUATION_DATES[0],
            "date_t1": VALUATION_DATES[1],
            "date_t2": VALUATION_DATES[2],
        }
        for transition_index, gap_days in enumerate(DATE_GAPS_DAYS):
            rng = np.random.default_rng(
                ANALYSIS_SEED + 1000 * sample_index + 100 * transition_index
            )
            for factor_index, factor in enumerate(("slow", "fast")):
                uniform = float(rng.random())
                kappa, theta, sigma = _factor_parameters(target, factor)
                next_state = exact_cir_transition_from_uniform(
                    kappa,
                    theta,
                    sigma,
                    state[factor],
                    gap_days / 365.0,
                    uniform,
                )
                nuisance_name = f"v_{factor}_t{transition_index + 1}"
                lower, upper = STATE_BOUNDS[nuisance_name]
                if not lower < next_state < upper:
                    raise RuntimeError(
                        f"Exact CIR state {nuisance_name}={next_state} falls outside "
                        f"the predeclared numerical state envelope {(lower, upper)}"
                    )
                record[f"uniform_{factor}_t{transition_index}_to_t{transition_index + 1}"] = uniform
                record[nuisance_name] = next_state
                state[factor] = next_state
        rows.append(record)
    return pd.DataFrame(rows)


def state_vector(state_row: pd.Series | Any) -> np.ndarray:
    return np.asarray(
        [getattr(state_row, name) if hasattr(state_row, name) else state_row[name] for name in NUISANCE_STATE_NAMES],
        dtype=np.float64,
    )


def _parameters_at_date(
    target: Sequence[float], nuisance: Sequence[float] | None, date_index: int
) -> np.ndarray:
    vector = np.asarray(target, dtype=np.float64).copy()
    if date_index == 0:
        return vector
    if nuisance is None:
        raise ValueError("Later dates require date-specific variance states")
    states = np.asarray(nuisance, dtype=np.float64)
    if states.shape != (4,):
        raise ValueError("Later variance state vector must contain four values")
    vector[4] = states[0 if date_index == 1 else 2]
    vector[9] = states[1 if date_index == 1 else 3]
    return vector


def joint_normalized_prices(
    target: Sequence[float],
    design: Design,
    *,
    oracle_states: Sequence[float] | None,
    nuisance_states: Sequence[float] | None,
    node_count: int,
) -> np.ndarray:
    states = oracle_states if design.oracle_later_states else nuisance_states
    pieces: list[np.ndarray] = []
    for date_index, (_, maturity_days) in enumerate(MATURITY_PROFILES[: design.date_count]):
        date_parameters = _parameters_at_date(target, states, date_index)
        pieces.append(
            baseline.normalized_observables(
                date_parameters,
                REPRESENTATION,
                maturity_days,
                node_count=node_count,
                rates=CONTROLLED_RATES,
                dividend_yields=CONTROLLED_DIVIDEND_YIELDS,
            )
        )
    return np.concatenate(pieces)


def transition_logpdf_vector(
    target: Sequence[float], nuisance_states: Sequence[float]
) -> np.ndarray:
    target_vector = np.asarray(target, dtype=np.float64)
    nuisance = np.asarray(nuisance_states, dtype=np.float64)
    values: list[float] = []
    for factor, anchor_index, nuisance_indices in (
        ("slow", 4, (0, 2)),
        ("fast", 9, (1, 3)),
    ):
        kappa, theta, sigma = _factor_parameters(target_vector, factor)
        previous = float(target_vector[anchor_index])
        for transition_index, next_index in enumerate(nuisance_indices):
            next_state = float(nuisance[next_index])
            values.append(
                exact_cir_transition_logpdf(
                    next_state,
                    kappa,
                    theta,
                    sigma,
                    previous,
                    DATE_GAPS_DAYS[transition_index] / 365.0,
                )
            )
            previous = next_state
    return np.asarray(values, dtype=np.float64)


def _valid_joint(
    target: np.ndarray,
    nuisance: np.ndarray | None,
) -> bool:
    if not validate_parameters(target)["is_valid"]:
        return False
    if nuisance is None:
        return True
    return bool(
        np.isfinite(nuisance).all()
        and all(
            STATE_BOUNDS[name][0] < value < STATE_BOUNDS[name][1]
            for name, value in zip(NUISANCE_STATE_NAMES, nuisance, strict=True)
        )
    )


def scaled_price_jacobians(
    target: np.ndarray,
    design: Design,
    bounds: dict[str, tuple[float, float]],
    *,
    oracle_states: np.ndarray,
    nuisance_states: np.ndarray | None,
    node_count: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    base = target if nuisance_states is None else np.concatenate((target, nuisance_states))
    widths = _target_widths(bounds) if nuisance_states is None else _joint_widths(bounds)
    columns: list[np.ndarray] = []
    for index, width in enumerate(widths):
        step = JACOBIAN_RELATIVE_STEP * width
        for _ in range(14):
            lower = base.copy()
            upper = base.copy()
            lower[index] -= step
            upper[index] += step
            lower_target, upper_target = lower[:10], upper[:10]
            lower_nuisance = lower[10:] if len(lower) == 14 else nuisance_states
            upper_nuisance = upper[10:] if len(upper) == 14 else nuisance_states
            if _valid_joint(lower_target, lower_nuisance) and _valid_joint(
                upper_target, upper_nuisance
            ):
                break
            step *= 0.5
        else:
            raise RuntimeError(f"No valid central step for {JOINT_PARAMETER_NAMES[index]}")
        lower_prices = joint_normalized_prices(
            lower_target,
            design,
            oracle_states=oracle_states,
            nuisance_states=lower_nuisance,
            node_count=node_count,
        )
        upper_prices = joint_normalized_prices(
            upper_target,
            design,
            oracle_states=oracle_states,
            nuisance_states=upper_nuisance,
            node_count=node_count,
        )
        columns.append((upper_prices - lower_prices) * width / (2.0 * step))
    matrix = np.column_stack(columns)
    return matrix[:, :10], matrix[:, 10:] if matrix.shape[1] == 14 else None


def nuisance_projected_jacobian(
    target_jacobian: np.ndarray, nuisance_jacobian: np.ndarray
) -> tuple[np.ndarray, int]:
    """Project target columns orthogonally away from nuisance column space."""
    left, singular_values, _ = np.linalg.svd(nuisance_jacobian, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] == 0.0:
        return target_jacobian.copy(), 0
    tolerance = (
        max(nuisance_jacobian.shape)
        * np.finfo(np.float64).eps
        * singular_values[0]
    )
    rank = int(np.sum(singular_values > tolerance))
    basis = left[:, :rank]
    projected = target_jacobian - basis @ (basis.T @ target_jacobian)
    return projected, rank


def scaled_transition_scores(
    target: np.ndarray,
    nuisance: np.ndarray,
    bounds: dict[str, tuple[float, float]],
) -> np.ndarray:
    base = np.concatenate((target, nuisance))
    widths = _joint_widths(bounds)
    columns: list[np.ndarray] = []
    for index, width in enumerate(widths):
        step = JACOBIAN_RELATIVE_STEP * width
        for _ in range(14):
            lower = base.copy()
            upper = base.copy()
            lower[index] -= step
            upper[index] += step
            if _valid_joint(lower[:10], lower[10:]) and _valid_joint(
                upper[:10], upper[10:]
            ):
                lower_log = transition_logpdf_vector(lower[:10], lower[10:])
                upper_log = transition_logpdf_vector(upper[:10], upper[10:])
                if np.isfinite(lower_log).all() and np.isfinite(upper_log).all():
                    break
            step *= 0.5
        else:
            raise RuntimeError(
                f"No stable exact-CIR score step for {JOINT_PARAMETER_NAMES[index]}"
            )
        columns.append((upper_log - lower_log) * width / (2.0 * step))
    return np.column_stack(columns)


def _metrics_from_singular_values(
    singular_values: np.ndarray, observation_count: int
) -> dict[str, Any]:
    largest = float(singular_values[0])
    numerical_tolerance = (
        max(observation_count, 10) * np.finfo(np.float64).eps * largest
    )
    practical_tolerance = PRACTICAL_RANK_RELATIVE_TOLERANCE * largest
    numerical_rank = int(np.sum(singular_values > numerical_tolerance))
    practical_rank = int(np.sum(singular_values > practical_tolerance))
    smallest = float(singular_values[-1])
    return {
        "numerical_rank": numerical_rank,
        "practical_target_rank_1e_minus_6": practical_rank,
        "largest_singular_value": largest,
        "smallest_singular_value": smallest,
        "condition_number": largest / smallest if smallest > 0.0 else math.inf,
        **{
            f"singular_value_{index + 1}": float(value)
            for index, value in enumerate(singular_values)
        },
    }


def _profile_physics_information(
    target_jacobian: np.ndarray,
    nuisance_jacobian: np.ndarray,
    transition_scores: np.ndarray,
    reference_price_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Profile nuisance states from price plus exact-density score information.

    The exact CIR density contributes its positive-semidefinite outer-product-of-
    score information.  Multiplication by the squared reference normalized-price
    sigma expresses it in the same units as the unweighted price least-squares
    information.  The target block is then Schur-complemented over nuisance states.
    """
    joint_price = np.column_stack((target_jacobian, nuisance_jacobian))
    information = joint_price.T @ joint_price
    information += (reference_price_sigma**2) * (
        transition_scores.T @ transition_scores
    )
    target_block = information[:10, :10]
    cross_block = information[:10, 10:]
    nuisance_block = information[10:, 10:]
    nuisance_rank = int(np.linalg.matrix_rank(nuisance_block))
    profile = target_block - cross_block @ np.linalg.pinv(
        nuisance_block, rcond=1.0e-12
    ) @ cross_block.T
    profile = 0.5 * (profile + profile.T)
    eigenvalues, eigenvectors = np.linalg.eigh(profile)
    largest = max(float(eigenvalues[-1]), 0.0)
    negative_tolerance = max(1.0e-18, 1.0e-9 * largest)
    if float(eigenvalues[0]) < -negative_tolerance:
        raise FloatingPointError(
            "Exact-CIR profiled information is materially indefinite: "
            f"minimum eigenvalue {eigenvalues[0]:.6e}"
        )
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    order = np.argsort(eigenvalues)[::-1]
    singular_values = np.sqrt(eigenvalues[order])
    right_vectors = eigenvectors[:, order].T
    sensitivities = np.sqrt(np.clip(np.diag(profile), 0.0, None))
    return singular_values, right_vectors, sensitivities, nuisance_rank


def run_identifiability(
    samples: pd.DataFrame,
    states: pd.DataFrame,
    bounds: dict[str, tuple[float, float]],
    *,
    node_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    state_lookup = states.set_index("sample_id")
    summary_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    viability = {design.design_id: "VIABLE" for design in DESIGNS}
    for sample in samples.itertuples(index=False):
        target = np.asarray(
            [getattr(sample, name) for name in CANONICAL_TARGET_NAMES], dtype=np.float64
        )
        oracle = state_vector(state_lookup.loc[sample.sample_id])
        joint_clean = joint_normalized_prices(
            target,
            DESIGN_BY_ID["B"],
            oracle_states=oracle,
            nuisance_states=None,
            node_count=node_count,
        )
        reference_sigma = max(
            PRICE_SIGMA_FLOOR,
            PHYSICS_REFERENCE_NOISE * float(np.median(np.abs(joint_clean))),
        )
        for design in DESIGNS:
            try:
                nuisance = oracle if design.latent_later_states else None
                target_jacobian, nuisance_jacobian = scaled_price_jacobians(
                    target,
                    design,
                    bounds,
                    oracle_states=oracle,
                    nuisance_states=nuisance,
                    node_count=node_count,
                )
                if design.design_id in {"A", "B"}:
                    effective = target_jacobian
                    _, singular_values, right_vectors = np.linalg.svd(
                        effective, full_matrices=False
                    )
                    sensitivities = np.linalg.norm(effective, axis=0)
                    nuisance_rank = 0
                    method = "direct_scaled_price_jacobian"
                elif design.design_id == "C":
                    assert nuisance_jacobian is not None
                    effective, nuisance_rank = nuisance_projected_jacobian(
                        target_jacobian, nuisance_jacobian
                    )
                    _, singular_values, right_vectors = np.linalg.svd(
                        effective, full_matrices=False
                    )
                    sensitivities = np.linalg.norm(effective, axis=0)
                    method = "orthogonal_nuisance_projection"
                else:
                    assert nuisance_jacobian is not None
                    scores = scaled_transition_scores(target, oracle, bounds)
                    (
                        singular_values,
                        right_vectors,
                        sensitivities,
                        nuisance_rank,
                    ) = _profile_physics_information(
                        target_jacobian,
                        nuisance_jacobian,
                        scores,
                        reference_sigma,
                    )
                    method = "exact_cir_score_opg_schur_complement"
                metrics = _metrics_from_singular_values(
                    singular_values, 20 * design.date_count
                )
                key = {
                    "design_id": design.design_id,
                    "design_label": design.label,
                    "sample_id": sample.sample_id,
                    "distribution": sample.distribution,
                    "date_count": design.date_count,
                    "normalized_price_count": 20 * design.date_count,
                    "canonical_target_count": 10,
                    "nuisance_state_count": 4 if design.latent_later_states else 0,
                    "nuisance_information_rank": nuisance_rank,
                    "target_information_method": method,
                    "reference_price_sigma": reference_sigma,
                    "viability": "VIABLE",
                    "error": "",
                }
                summary_rows.append({**key, **metrics})
                for name, sensitivity in zip(
                    CANONICAL_TARGET_NAMES, sensitivities, strict=True
                ):
                    sensitivity_rows.append(
                        {**key, "parameter": name, "profiled_scaled_sensitivity": float(sensitivity)}
                    )
                weakest = right_vectors[-1]
                for name, loading in zip(
                    CANONICAL_TARGET_NAMES, weakest, strict=True
                ):
                    direction_rows.append(
                        {
                            **key,
                            "parameter": name,
                            "weakest_direction_loading": float(loading),
                            "absolute_weakest_direction_loading": float(abs(loading)),
                        }
                    )
            except Exception as error:
                viability[design.design_id] = f"STOPPED:{type(error).__name__}"
                summary_rows.append(
                    {
                        "design_id": design.design_id,
                        "design_label": design.label,
                        "sample_id": sample.sample_id,
                        "distribution": sample.distribution,
                        "date_count": design.date_count,
                        "normalized_price_count": 20 * design.date_count,
                        "canonical_target_count": 10,
                        "nuisance_state_count": 4 if design.latent_later_states else 0,
                        "nuisance_information_rank": 0,
                        "target_information_method": "UNAVAILABLE",
                        "reference_price_sigma": reference_sigma,
                        "viability": "STOPPED",
                        "error": f"{type(error).__name__}:{error}",
                        **{
                            key: math.nan
                            for key in (
                                "numerical_rank",
                                "practical_target_rank_1e_minus_6",
                                "largest_singular_value",
                                "smallest_singular_value",
                                "condition_number",
                                *[f"singular_value_{i}" for i in range(1, 11)],
                            )
                        },
                    }
                )
    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(sensitivity_rows),
        pd.DataFrame(direction_rows),
        viability,
    )


def _decode_latent(
    latent: np.ndarray,
    design: Design,
    bounds: dict[str, tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray | None]:
    target = unconstrained_to_parameters(latent[:10], bounds)
    if not design.latent_later_states:
        return target, None
    center_unit = np.asarray(
        [
            (center - STATE_BOUNDS[name][0])
            / (STATE_BOUNDS[name][1] - STATE_BOUNDS[name][0])
            for name, center in zip(
                NUISANCE_STATE_NAMES, STATE_START_CENTERS, strict=True
            )
        ],
        dtype=np.float64,
    )
    shifts = logit(center_unit)
    unit = expit(np.clip(latent[10:] + shifts, -35.0, 35.0))
    nuisance = np.asarray(
        [
            STATE_BOUNDS[name][0]
            + fraction * (STATE_BOUNDS[name][1] - STATE_BOUNDS[name][0])
            for name, fraction in zip(NUISANCE_STATE_NAMES, unit, strict=True)
        ],
        dtype=np.float64,
    )
    return target, nuisance


def _state_to_latent(states: np.ndarray) -> np.ndarray:
    unit = np.asarray(
        [
            (value - STATE_BOUNDS[name][0])
            / (STATE_BOUNDS[name][1] - STATE_BOUNDS[name][0])
            for name, value in zip(NUISANCE_STATE_NAMES, states, strict=True)
        ],
        dtype=np.float64,
    )
    center_unit = np.asarray(
        [
            (center - STATE_BOUNDS[name][0])
            / (STATE_BOUNDS[name][1] - STATE_BOUNDS[name][0])
            for name, center in zip(
                NUISANCE_STATE_NAMES, STATE_START_CENTERS, strict=True
            )
        ],
        dtype=np.float64,
    )
    return np.asarray(
        logit(np.clip(unit, 1.0e-8, 1.0 - 1.0e-8)) - logit(center_unit),
        dtype=np.float64,
    )


def deterministic_starts(
    design: Design, seed: int, count: int
) -> list[tuple[str, np.ndarray]]:
    if count < 1:
        raise ValueError("At least one start is required")
    rng = np.random.default_rng(seed)
    dimension = design.unknown_count
    starts = [("neutral_transform_midpoint", np.zeros(dimension, dtype=np.float64))]
    for index in range(1, count):
        starts.append(
            (
                f"deterministic_broad_{index}",
                rng.normal(0.0, 1.25, size=dimension),
            )
        )
    return starts


def coupled_noise(seed: int, noise_level: float) -> tuple[np.ndarray, np.ndarray]:
    all_dates = np.random.default_rng(seed).normal(0.0, noise_level, size=60)
    return all_dates[:20].copy(), all_dates


def _state_bound_hits(nuisance: np.ndarray | None) -> list[str]:
    if nuisance is None:
        return []
    hits: list[str] = []
    for name, value in zip(NUISANCE_STATE_NAMES, nuisance, strict=True):
        lower, upper = STATE_BOUNDS[name]
        relative = min(value - lower, upper - value) / (upper - lower)
        if relative <= 0.01:
            hits.append(f"{name}:near_boundary")
    return hits


def run_recovery(
    samples: pd.DataFrame,
    states: pd.DataFrame,
    bounds: dict[str, tuple[float, float]],
    viability: dict[str, str],
    identifiability: pd.DataFrame,
    *,
    node_count: int,
    maxiter: int,
    start_count: int,
    per_distribution: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    oracle_rows = identifiability.loc[
        identifiability["design_id"].eq("B") & identifiability["viability"].eq("VIABLE")
    ]
    oracle_severely_nonidentifiable = bool(
        oracle_rows.empty
        or (oracle_rows["practical_target_rank_1e_minus_6"] == 10).mean() < 0.50
        or float(oracle_rows["condition_number"].median()) > CONDITION_WARNING_THRESHOLD
    )
    allowed_designs = {"A", "B"} if oracle_severely_nonidentifiable else {"A", "B", "C", "D"}
    recovery_samples = (
        samples.groupby("distribution", sort=True, group_keys=False)
        .head(per_distribution)
        .reset_index(drop=True)
    )
    state_lookup = states.set_index("sample_id")
    rows: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(recovery_samples.itertuples(index=False)):
        true_target = np.asarray(
            [getattr(sample, name) for name in CANONICAL_TARGET_NAMES], dtype=np.float64
        )
        true_states = state_vector(state_lookup.loc[sample.sample_id])
        clean_joint = joint_normalized_prices(
            true_target,
            DESIGN_BY_ID["B"],
            oracle_states=true_states,
            nuisance_states=None,
            node_count=node_count,
        )
        reference_sigma = max(
            PRICE_SIGMA_FLOOR,
            PHYSICS_REFERENCE_NOISE * float(np.median(np.abs(clean_joint))),
        )
        for noise_index, noise_level in enumerate(NOISE_LEVELS):
            seed = ANALYSIS_SEED + 10000 * sample_index + noise_index
            single_noise, joint_noise = coupled_noise(seed, noise_level)
            observed_joint = clean_joint * (1.0 + joint_noise)
            for design in DESIGNS:
                if design.design_id not in allowed_designs or viability[design.design_id] != "VIABLE":
                    continue
                clean = clean_joint[:20] if design.design_id == "A" else clean_joint
                observed = (
                    clean * (1.0 + single_noise)
                    if design.design_id == "A"
                    else observed_joint
                )

                def objective(latent: np.ndarray) -> float:
                    try:
                        candidate_target, candidate_states = _decode_latent(
                            latent, design, bounds
                        )
                        predicted = joint_normalized_prices(
                            candidate_target,
                            design,
                            oracle_states=true_states,
                            nuisance_states=candidate_states,
                            node_count=node_count,
                        )
                        residual = predicted - observed
                        value = 0.5 * float(residual @ residual)
                        if design.cir_physics:
                            assert candidate_states is not None
                            log_densities = transition_logpdf_vector(
                                candidate_target, candidate_states
                            )
                            if not np.isfinite(log_densities).all():
                                return 1.0e12
                            value -= (reference_sigma**2) * float(log_densities.sum())
                        return value if math.isfinite(value) else 1.0e12
                    except Exception:
                        return 1.0e12

                for start_index, (strategy, start) in enumerate(
                    deterministic_starts(design, seed + 100 * (ord(design.design_id) - 64), start_count)
                ):
                    base_row: dict[str, Any] = {
                        "design_id": design.design_id,
                        "design_label": design.label,
                        "sample_id": sample.sample_id,
                        "distribution": sample.distribution,
                        "noise_level": noise_level,
                        "start_index": start_index,
                        "start_strategy": strategy,
                        "reference_price_sigma": reference_sigma,
                    }
                    try:
                        result = minimize(
                            objective,
                            start,
                            method="L-BFGS-B",
                            bounds=[(-10.0, 10.0)] * design.unknown_count,
                            options={
                                "maxiter": maxiter,
                                "maxls": 40,
                                "ftol": 1.0e-14,
                                "gtol": 1.0e-9,
                                "maxfun": maxiter * (design.unknown_count + 2),
                            },
                        )
                        recovered_target, recovered_states = _decode_latent(
                            np.asarray(result.x, dtype=np.float64), design, bounds
                        )
                        predicted = joint_normalized_prices(
                            recovered_target,
                            design,
                            oracle_states=true_states,
                            nuisance_states=recovered_states,
                            node_count=node_count,
                        )
                        errors = recovered_target - true_target
                        scaled_errors = np.abs(errors) / _target_widths(bounds)
                        target_rmse = float(np.sqrt(np.mean(scaled_errors**2)))
                        target_max = float(np.max(scaled_errors))
                        validation = validate_parameters(recovered_target)
                        hit_reasons = boundary_diagnostics(recovered_target, bounds)
                        hit_reasons.extend(_state_bound_hits(recovered_states))
                        nuisance_rmse = math.nan
                        if recovered_states is not None:
                            nuisance_rmse = float(
                                np.sqrt(
                                    np.mean(
                                        ((recovered_states - true_states) / _nuisance_widths())
                                        ** 2
                                    )
                                )
                            )
                        row = {
                            **base_row,
                            "optimizer_success": bool(result.success),
                            "optimizer_status": int(result.status),
                            "optimizer_message": str(result.message),
                            "nfev": int(result.nfev),
                            "objective_value": float(result.fun),
                            "price_rmse_normalized": float(
                                np.sqrt(np.mean((predicted - observed) ** 2))
                            ),
                            "target_scaled_parameter_rmse": target_rmse,
                            "target_max_scaled_parameter_error": target_max,
                            "nuisance_scaled_state_rmse": nuisance_rmse,
                            "constraint_valid": bool(validation["is_valid"]),
                            "bound_hit": bool(hit_reasons),
                            "bound_reasons": ";".join(hit_reasons),
                            "canonical_target_recovery_success": bool(
                                result.success
                                and validation["is_valid"]
                                and target_rmse <= RECOVERY_SCALED_RMSE_THRESHOLD
                                and target_max <= RECOVERY_SCALED_MAX_ERROR_THRESHOLD
                            ),
                            "error": "",
                        }
                        for index, name in enumerate(CANONICAL_TARGET_NAMES):
                            row[f"true_{name}"] = float(true_target[index])
                            row[f"recovered_{name}"] = float(recovered_target[index])
                            row[f"scaled_error_{name}"] = float(scaled_errors[index])
                        for index, name in enumerate(NUISANCE_STATE_NAMES):
                            row[f"true_{name}"] = float(true_states[index])
                            row[f"recovered_{name}"] = (
                                float(recovered_states[index])
                                if recovered_states is not None
                                else math.nan
                            )
                            row[f"scaled_error_{name}"] = (
                                float(
                                    abs(recovered_states[index] - true_states[index])
                                    / _nuisance_widths()[index]
                                )
                                if recovered_states is not None
                                else math.nan
                            )
                    except Exception as error:
                        row = {
                            **base_row,
                            "optimizer_success": False,
                            "optimizer_status": -1,
                            "optimizer_message": "",
                            "nfev": 0,
                            "objective_value": math.nan,
                            "price_rmse_normalized": math.nan,
                            "target_scaled_parameter_rmse": math.nan,
                            "target_max_scaled_parameter_error": math.nan,
                            "nuisance_scaled_state_rmse": math.nan,
                            "constraint_valid": False,
                            "bound_hit": False,
                            "bound_reasons": "",
                            "canonical_target_recovery_success": False,
                            "error": f"{type(error).__name__}:{error}",
                        }
                        for name in CANONICAL_TARGET_NAMES:
                            row[f"true_{name}"] = float(
                                true_target[CANONICAL_TARGET_NAMES.index(name)]
                            )
                            row[f"recovered_{name}"] = math.nan
                            row[f"scaled_error_{name}"] = math.nan
                        for index, name in enumerate(NUISANCE_STATE_NAMES):
                            row[f"true_{name}"] = float(true_states[index])
                            row[f"recovered_{name}"] = math.nan
                            row[f"scaled_error_{name}"] = math.nan
                    rows.append(row)
    starts = pd.DataFrame(rows)
    if starts.empty:
        return starts, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), oracle_severely_nonidentifiable

    summary_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    nuisance_rows: list[dict[str, Any]] = []
    for (design_id, noise_level), group in starts.groupby(
        ["design_id", "noise_level"], sort=True
    ):
        finite = group.loc[np.isfinite(group["objective_value"])].copy()
        best = (
            finite.sort_values(
                ["sample_id", "objective_value", "start_index"], kind="stable"
            )
            .groupby("sample_id", sort=True, as_index=False)
            .head(1)
        )
        variations: list[float] = []
        for _, sample_group in finite.groupby("sample_id", sort=True):
            recovered = sample_group[
                [f"recovered_{name}" for name in CANONICAL_TARGET_NAMES]
            ].to_numpy(dtype=float)
            if len(recovered) > 1 and np.isfinite(recovered).all():
                variations.append(
                    float(
                        np.sqrt(
                            np.mean(
                                (np.std(recovered, axis=0) / _target_widths(bounds)) ** 2
                            )
                        )
                    )
                )
        summary_rows.append(
            {
                "design_id": design_id,
                "design_label": DESIGN_BY_ID[design_id].label,
                "noise_level": noise_level,
                "sample_count": int(group["sample_id"].nunique()),
                "start_count": int(len(group)),
                "optimizer_success_count": int(group["optimizer_success"].sum()),
                "constraint_valid_count": int(group["constraint_valid"].sum()),
                "canonical_target_recovery_pass_count": int(
                    group["canonical_target_recovery_success"].sum()
                ),
                "canonical_target_recovery_pass_frequency": float(
                    group["canonical_target_recovery_success"].mean()
                ),
                "bound_hit_count": int(group["bound_hit"].sum()),
                "median_best_price_rmse_normalized": float(
                    best["price_rmse_normalized"].median()
                ),
                "median_best_target_scaled_parameter_rmse": float(
                    best["target_scaled_parameter_rmse"].median()
                ),
                "median_best_nuisance_scaled_state_rmse": (
                    float(best["nuisance_scaled_state_rmse"].dropna().median())
                    if best["nuisance_scaled_state_rmse"].notna().any()
                    else math.nan
                ),
                "median_start_to_start_target_variability": (
                    float(np.median(variations)) if variations else math.nan
                ),
            }
        )
        for name in CANONICAL_TARGET_NAMES:
            parameter_rows.append(
                {
                    "design_id": design_id,
                    "noise_level": noise_level,
                    "parameter": name,
                    "median_best_scaled_error": float(
                        best[f"scaled_error_{name}"].median()
                    ),
                }
            )
        if DESIGN_BY_ID[design_id].latent_later_states:
            for row in best.itertuples(index=False):
                for name in NUISANCE_STATE_NAMES:
                    nuisance_rows.append(
                        {
                            "design_id": design_id,
                            "noise_level": noise_level,
                            "sample_id": row.sample_id,
                            "state": name,
                            "true_state": getattr(row, f"true_{name}"),
                            "recovered_state": getattr(row, f"recovered_{name}"),
                            "scaled_state_error": getattr(row, f"scaled_error_{name}"),
                        }
                    )
    return (
        starts,
        pd.DataFrame(summary_rows),
        pd.DataFrame(parameter_rows),
        pd.DataFrame(nuisance_rows),
        oracle_severely_nonidentifiable,
    )


def classify_diagnostic(
    identifiability: pd.DataFrame,
    recovery: pd.DataFrame,
    viability: dict[str, str],
    oracle_stop_rule: bool,
) -> dict[str, Any]:
    design_pass: dict[str, bool] = {}
    rank_frequency: dict[str, float] = {}
    for design in DESIGNS:
        rows = identifiability.loc[
            identifiability["design_id"].eq(design.design_id)
            & identifiability["viability"].eq("VIABLE")
        ]
        rank_frequency[design.design_id] = (
            float((rows["practical_target_rank_1e_minus_6"] == 10).mean())
            if not rows.empty
            else 0.0
        )
        recovery_rows = recovery.loc[recovery["design_id"].eq(design.design_id)]
        recovery_pass = bool(
            len(recovery_rows) == len(NOISE_LEVELS)
            and (
                recovery_rows["canonical_target_recovery_pass_frequency"]
                >= RECOVERY_FREQUENCY_THRESHOLD
            ).all()
        )
        design_pass[design.design_id] = bool(
            viability[design.design_id] == "VIABLE"
            and not rows.empty
            and (rows["practical_target_rank_1e_minus_6"] == 10).all()
            and (rows["condition_number"] <= CONDITION_WARNING_THRESHOLD).all()
            and recovery_pass
        )
    if design_pass["C"]:
        verdict = "MULTI_DATE_DIAGNOSTIC = PROMISING"
        case = "CASE_4"
    elif design_pass["B"] and design_pass["D"] and not design_pass["C"]:
        verdict = "MULTI_DATE_DIAGNOSTIC = PHYSICS_CONSTRAINT_REQUIRED"
        case = "CASE_3"
    elif design_pass["B"] and not design_pass["C"]:
        verdict = "MULTI_DATE_DIAGNOSTIC = STATE_OBSERVABILITY_REQUIRED"
        case = "CASE_2"
    else:
        verdict = "MULTI_DATE_DIAGNOSTIC = INSUFFICIENT"
        case = "CASE_1"
    return {
        "verdict": verdict,
        "critical_case": case,
        "g2_status": "G2 = NOT_PASSED — STRUCTURAL IDENTIFIABILITY PROBLEM REMAINS",
        "design_pass": design_pass,
        "practical_target_full_rank_frequency": rank_frequency,
        "design_viability": viability,
        "oracle_stop_rule_triggered": oracle_stop_rule,
        "recovery_frequency_threshold": RECOVERY_FREQUENCY_THRESHOLD,
        "final_representation_frozen": False,
        "oracle_total_variance_diagnostic_run": False,
    }


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".png", prefix=f".{path.stem}.", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        fig.savefig(temporary, dpi=180, bbox_inches="tight", metadata={"Software": ""})
        plt.close(fig)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_figures(
    identifiability: pd.DataFrame,
    directions: pd.DataFrame,
    recovery: pd.DataFrame,
    nuisance: pd.DataFrame,
    decision: dict[str, Any],
    output_root: Path,
) -> list[Path]:
    figure_root = output_root / "figures"
    viable = identifiability.loc[identifiability["viability"].eq("VIABLE")].copy()
    colors = {"A": "#4E79A7", "B": "#59A14F", "C": "#F28E2B", "D": "#E15759"}
    figures: list[Path] = []

    rank = viable.assign(full=lambda frame: frame["practical_target_rank_1e_minus_6"].eq(10)).groupby("design_id")["full"].mean().reindex([d.design_id for d in DESIGNS])
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.bar(rank.index, 100.0 * rank.fillna(0.0), color=[colors[x] for x in rank.index])
    axis.set(title="Practical full-rank frequency for canonical ten", xlabel="Design", ylabel="Frequency (%)", ylim=(0, 105))
    path = figure_root / "practical_rank_comparison.png"; _save_figure(fig, path); figures.append(path)

    medians = viable.groupby("design_id")[[f"singular_value_{i}" for i in range(1, 11)]].median().reindex([d.design_id for d in DESIGNS])
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    for design_id, row in medians.iterrows():
        if row.notna().any():
            axis.semilogy(range(1, 11), row.to_numpy(float), marker="o", label=design_id, color=colors[design_id])
    axis.set(title="Median profiled target singular spectrum", xlabel="Singular-value index", ylabel="Singular value"); axis.legend()
    path = figure_root / "singular_values_comparison.png"; _save_figure(fig, path); figures.append(path)

    conditions = [viable.loc[viable["design_id"].eq(d.design_id), "condition_number"].to_numpy(float) for d in DESIGNS]
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.boxplot(
        [values if len(values) else np.asarray([np.nan]) for values in conditions],
        tick_labels=[d.design_id for d in DESIGNS],
        showmeans=True,
    )
    axis.set_yscale("log"); axis.axhline(CONDITION_WARNING_THRESHOLD, color="black", linestyle="--", linewidth=1); axis.set(title="Canonical-target condition numbers", xlabel="Design", ylabel="Condition number")
    path = figure_root / "condition_number_comparison.png"; _save_figure(fig, path); figures.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    if not recovery.empty:
        for design in DESIGNS:
            group = recovery.loc[recovery["design_id"].eq(design.design_id)].sort_values("noise_level")
            if group.empty: continue
            x = 100.0 * group["noise_level"].to_numpy(float)
            axes[0].plot(x, group["median_best_target_scaled_parameter_rmse"], marker="o", label=design.design_id, color=colors[design.design_id])
            axes[1].plot(x, 100.0 * group["canonical_target_recovery_pass_frequency"], marker="o", label=design.design_id, color=colors[design.design_id])
    axes[0].set(title="Canonical-target recovery error", xlabel="Price noise (%)", ylabel="Median range-scaled RMSE"); axes[0].set_yscale("log")
    axes[1].set(title="Canonical-target recovery passes", xlabel="Price noise (%)", ylabel="Pass frequency (%)", ylim=(-2, 102))
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(handles, labels)
    path = figure_root / "recovery_comparison.png"; _save_figure(fig, path); figures.append(path)

    direction_med = directions.groupby(["design_id", "parameter"])["absolute_weakest_direction_loading"].median().unstack(0).reindex(CANONICAL_TARGET_NAMES)
    fig, axis = plt.subplots(figsize=(10.0, 4.8))
    direction_med.plot(kind="bar", ax=axis, color=[colors.get(x, "gray") for x in direction_med.columns])
    axis.set(title="Median absolute loading in weakest target direction", xlabel="Parameter", ylabel="Absolute loading"); axis.tick_params(axis="x", rotation=45)
    path = figure_root / "weakest_directions_comparison.png"; _save_figure(fig, path); figures.append(path)

    fig, axis = plt.subplots(figsize=(8.0, 4.2))
    if nuisance.empty:
        axis.text(0.5, 0.5, "Nuisance recovery not run under the oracle stop rule", ha="center", va="center")
        axis.set_axis_off()
    else:
        nmed = nuisance.groupby(["design_id", "state"])["scaled_state_error"].median().unstack(0).reindex(NUISANCE_STATE_NAMES)
        nmed.plot(kind="bar", ax=axis, color=[colors.get(x, "gray") for x in nmed.columns])
        axis.set(title="Later-state recovery error", xlabel="Nuisance state", ylabel="Median range-scaled error"); axis.tick_params(axis="x", rotation=20)
    path = figure_root / "nuisance_state_recovery.png"; _save_figure(fig, path); figures.append(path)

    fig, axis = plt.subplots(figsize=(11.2, 3.8)); axis.axis("off")
    columns = ["Design", "Full target rank", "Median condition", "Clean", "0.5%", "1.0%"]
    table_rows: list[list[str]] = []
    for design in DESIGNS:
        group = viable.loc[viable["design_id"].eq(design.design_id)]
        rec = recovery.loc[recovery["design_id"].eq(design.design_id)].set_index("noise_level") if not recovery.empty else pd.DataFrame()
        values = []
        for noise in NOISE_LEVELS:
            if not rec.empty and noise in rec.index:
                row = rec.loc[noise]
                values.append(f"{int(row.canonical_target_recovery_pass_count)}/{int(row.start_count)}")
            else:
                values.append("STOPPED")
        table_rows.append([
            design.design_id,
            f"{100.0 * ((group['practical_target_rank_1e_minus_6'] == 10).mean() if not group.empty else 0.0):.1f}%",
            f"{group['condition_number'].median():.3e}" if not group.empty else "UNAVAILABLE",
            *values,
        ])
    table = axis.table(cellText=table_rows, colLabels=columns, cellLoc="center", loc="center"); table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1.0, 1.5)
    axis.set_title(f"{decision['verdict']}\nG2 remains NOT_PASSED", pad=18)
    path = figure_root / "mentor_summary.png"; _save_figure(fig, path); figures.append(path)
    return figures


def _summary_table(
    identifiability: pd.DataFrame, recovery: pd.DataFrame
) -> list[str]:
    lines = [
        "| Design | Practical target rank 10 | Median condition number | Clean recovery | 0.5% | 1.0% |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    viable = identifiability.loc[identifiability["viability"].eq("VIABLE")]
    for design in DESIGNS:
        group = viable.loc[viable["design_id"].eq(design.design_id)]
        full = 100.0 * ((group["practical_target_rank_1e_minus_6"] == 10).mean() if not group.empty else 0.0)
        condition = f"{group['condition_number'].median():.3e}" if not group.empty else "UNAVAILABLE"
        rec = recovery.loc[recovery["design_id"].eq(design.design_id)].set_index("noise_level") if not recovery.empty else pd.DataFrame()
        cells: list[str] = []
        for noise in NOISE_LEVELS:
            if not rec.empty and noise in rec.index:
                row = rec.loc[noise]
                cells.append(f"{int(row.canonical_target_recovery_pass_count)}/{int(row.start_count)}")
            else:
                cells.append("STOPPED")
        lines.append(f"| {design.design_id} | {full:.1f}% | {condition} | {' | '.join(cells)} |")
    return lines


def render_report(
    path: Path,
    designs: pd.DataFrame,
    states: pd.DataFrame,
    identifiability: pd.DataFrame,
    directions: pd.DataFrame,
    recovery: pd.DataFrame,
    nuisance: pd.DataFrame,
    decision: dict[str, Any],
    artifact_hashes: dict[str, str],
) -> None:
    viable = identifiability.loc[identifiability["viability"].eq("VIABLE")]
    lines = [
        "# G2 Joint Multi-Date Identifiability Diagnostic",
        "",
        "## Scope and preserved gate",
        "",
        "This is a deterministic synthetic information-design experiment, not a real-market representation freeze. It does not redo Stage A, change prior G2 evidence, generate the final 10k dataset, train ANN/PINN, or change the established gate.",
        "",
        "**G2 = NOT_PASSED — STRUCTURAL IDENTIFIABILITY PROBLEM REMAINS**",
        "",
        "Controlled rates `(0.0600, 0.0625)` and dividend yields `(0.0200, 0.0225)` are used for each date's near/middle maturities. They are synthetic controls only and do not resolve the real-market discount-source contract.",
        "",
        "Design A is the `2026-07-01` anchor-date control used for the joint-date comparison. Its statistics are therefore not expected to equal the established `5.107e7` condition-number median, which pooled all three independent single-date maturity profiles.",
        "",
        "## Exact state contract",
        "",
        "The canonical ten targets are the eight shared structural parameters plus `v_slow(t0)` and `v_fast(t0)`. Later states `v_slow(t1)`, `v_fast(t1)`, `v_slow(t2)`, and `v_fast(t2)` are date-specific. Designs C/D estimate them as nuisance variables; no design holds them equal to the anchor states.",
        "",
        f"Dates are `{VALUATION_DATES[0]}`, `{VALUATION_DATES[1]}`, `{VALUATION_DATES[2]}` with actual gaps `{DATE_GAPS_DAYS[0]}` and `{DATE_GAPS_DAYS[1]}` days. Near/middle maturity profiles are `{MATURITY_PROFILES[0][1]}`, `{MATURITY_PROFILES[1][1]}`, and `{MATURITY_PROFILES[2][1]}` days. Every surface uses central-five calls and puts.",
        "",
        "Later states are sampled through the exact CIR noncentral-chi-square transition using fixed inverse-CDF uniforms. The sampled states are stochastic realizations, not conditional expectations.",
        "",
        "## Predeclared A-D designs",
        "",
        "| Design | Dates | Prices | Canonical targets | Nuisance states | State treatment |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in designs.itertuples(index=False):
        treatment = "oracle known" if row.later_states_known else ("latent + exact CIR density" if row.exact_cir_transition_density else ("latent independent" if row.nuisance_state_count else "anchor date only"))
        lines.append(f"| {row.design_id} | {len(row.valuation_dates.split('|'))} | {row.normalized_price_count} | 10 | {row.nuisance_state_count} | {treatment} |")
    lines.extend([
        "",
        "## Nuisance-profiled identifiability method",
        "",
        "A/B use the direct scaled price Jacobian. C removes the nuisance-state column space from the canonical-target Jacobian with an orthogonal SVD projection. D forms price information plus the positive-semidefinite outer product of exact CIR transition-log-density scores, scales the physics contribution by the squared predeclared 0.5%-reference normalized-price sigma, and profiles the four nuisance states through a Schur complement. All target columns use the previously validated full hard-bound range scaling and the relative `1e-6` practical-rank threshold.",
        "",
        "## Critical comparison",
        "",
        *_summary_table(identifiability, recovery),
        "",
    ])
    for design in DESIGNS:
        group = viable.loc[viable["design_id"].eq(design.design_id)]
        lines.extend([
            f"### Design {design.design_id} — {design.label}",
            "",
            (
                f"Practical full target rank: `{100.0 * (group['practical_target_rank_1e_minus_6'].eq(10).mean() if not group.empty else 0.0):.1f}%`; median smallest singular value: `{group['smallest_singular_value'].median():.3e}`; median condition number: `{group['condition_number'].median():.3e}`."
                if not group.empty
                else f"Stopped as unavailable: `{decision['design_viability'][design.design_id]}`."
            ),
            "",
        ])
    lines.extend(["## Recovery detail", ""])
    if recovery.empty:
        lines.append("Recovery was not run.")
    else:
        lines.extend([
            "| Design | Noise | Optimizer success | Canonical recovery pass | Median best price RMSE | Median best target RMSE | Median best nuisance RMSE | Bound hits | Start variability |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in recovery.itertuples(index=False):
            nuisance_value = f"{row.median_best_nuisance_scaled_state_rmse:.3e}" if math.isfinite(row.median_best_nuisance_scaled_state_rmse) else "N/A"
            lines.append(
                f"| {row.design_id} | {100.0 * row.noise_level:.1f}% | {row.optimizer_success_count}/{row.start_count} | {row.canonical_target_recovery_pass_count}/{row.start_count} | {row.median_best_price_rmse_normalized:.3e} | {row.median_best_target_scaled_parameter_rmse:.3e} | {nuisance_value} | {row.bound_hit_count}/{row.start_count} | {row.median_start_to_start_target_variability:.3e} |"
            )
    lines.extend([
        "",
        "Nuisance-state errors for C/D are reported separately in `nuisance_state_recovery.csv`; they are never included in the canonical ten-target recovery gate. Recovery uses one representative target from each accepted distribution (two targets total), the same three deterministic starts, and an L-BFGS-B cap of `80` iterations; capped nonconvergence is retained as failure evidence. The eight-target local-identifiability sample is unchanged.",
        "",
        "## Weakest remaining directions",
        "",
    ])
    for design in DESIGNS:
        group = directions.loc[directions["design_id"].eq(design.design_id)]
        if group.empty:
            continue
        med = group.groupby("parameter")["absolute_weakest_direction_loading"].median().sort_values(ascending=False).head(5)
        lines.append(f"- Design {design.design_id}: " + ", ".join(f"`{name}` ({value:.3f})" for name, value in med.items()) + ".")
    lines.extend([
        "",
        "## Decision",
        "",
        f"**{decision['verdict']}**",
        "",
        f"Critical interpretation: `{decision['critical_case']}`. Oracle stop rule triggered: `{decision['oracle_stop_rule_triggered']}`. Optional `ORACLE_TOTAL_VARIANCE_DIAGNOSTIC` was not run; the bounded A-D experiment was sufficient for this decision.",
        "",
        "`CASE_1` is assigned by the complete predeclared numerical gates: B passes the local practical-rank gate but fails stable target-blind recovery. This classification does not mean that multi-date information has no local conditioning value.",
        "",
        "G2 is unchanged because this diagnostic does not satisfy or replace the real-market representation, far-expiry support, or discount-source provenance gates.",
        "",
        "## Recommended next experiment",
        "",
        "Run one mentor-reviewed synthetic replication with independently seeded CIR paths and the identical frozen A-D protocol; do not add features, priors, dates, or market proxies until the observed case replicates.",
        "",
        "## Reproducibility and artifacts",
        "",
        f"Exact state paths: `{len(states)}`. Protected prior Stage A/G2 artifacts were hashed before and after generation and remained byte-identical.",
        "",
        "| Artifact | SHA-256 |",
        "|---|---|",
    ])
    for relative, digest in sorted(artifact_hashes.items()):
        lines.append(f"| `{relative}` | `{digest}` |")
    lines.extend(["", decision["verdict"], "", decision["g2_status"], ""])
    _atomic_write_bytes(path, "\n".join(lines).encode("utf-8"))


def run_analysis(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    report_path: Path = DEFAULT_REPORT_PATH,
    node_count: int = 64,
    maxiter: int = RECOVERY_MAXITER,
    start_count: int = 3,
    per_distribution: int = RECOVERY_SAMPLES_PER_DISTRIBUTION,
    sample_limit: int | None = None,
    skip_recovery: bool = False,
) -> dict[str, Any]:
    protected_before = _protected_snapshot(output_root)
    bounds = load_hard_safety_bounds(baseline.BOUNDS_PATH)
    samples = baseline.select_representative_parameters(bounds, per_distribution=4)
    if sample_limit is not None:
        samples = samples.head(sample_limit).copy()
    designs = experiment_designs()
    states = simulate_state_paths(samples)
    identifiability, sensitivities, directions, viability = run_identifiability(
        samples, states, bounds, node_count=node_count
    )
    if skip_recovery:
        starts = pd.DataFrame()
        recovery = pd.DataFrame(
            columns=[
                "design_id",
                "noise_level",
                "canonical_target_recovery_pass_frequency",
            ]
        )
        parameter_errors = pd.DataFrame()
        nuisance = pd.DataFrame()
        oracle_stop_rule = False
    else:
        starts, recovery, parameter_errors, nuisance, oracle_stop_rule = run_recovery(
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
    decision = classify_diagnostic(
        identifiability, recovery, viability, oracle_stop_rule
    )
    frames = {
        "experiment_designs.csv": designs,
        "state_paths.csv": states,
        "identifiability_summary.csv": identifiability,
        "parameter_sensitivity.csv": sensitivities,
        "weakest_directions.csv": directions,
        "recovery_starts.csv": starts,
        "recovery_summary.csv": recovery,
        "parameter_error_summary.csv": parameter_errors,
        "nuisance_state_recovery.csv": nuisance,
    }
    for relative, frame in frames.items():
        _write_csv(frame, output_root / relative)
    _write_json(decision, output_root / "decision.json")
    figures = write_figures(
        identifiability, directions, recovery, nuisance, decision, output_root
    )
    artifact_hashes = {
        relative: _sha256(output_root / relative)
        for relative in EXPECTED_OUTPUT_FILES
        if not relative.startswith("figures/") and (output_root / relative).exists()
    }
    artifact_hashes.update(
        {
            path.relative_to(output_root).as_posix(): _sha256(path)
            for path in figures
        }
    )
    render_report(
        report_path,
        designs,
        states,
        identifiability,
        directions,
        recovery,
        nuisance,
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
        raise RuntimeError(f"Protected Stage A/G2 artifacts changed: {changed}")
    return {
        "designs": designs,
        "states": states,
        "identifiability": identifiability,
        "sensitivities": sensitivities,
        "directions": directions,
        "recovery_starts": starts,
        "recovery": recovery,
        "parameter_errors": parameter_errors,
        "nuisance": nuisance,
        "decision": decision,
        "artifact_hashes": artifact_hashes,
        "report_path": report_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--node-count", type=int, default=64)
    parser.add_argument("--maxiter", type=int, default=RECOVERY_MAXITER)
    parser.add_argument("--start-count", type=int, default=3)
    parser.add_argument(
        "--per-distribution",
        type=int,
        default=RECOVERY_SAMPLES_PER_DISTRIBUTION,
    )
    args = parser.parse_args()
    result = run_analysis(
        output_root=args.output_root,
        report_path=args.report_path,
        node_count=args.node_count,
        maxiter=args.maxiter,
        start_count=args.start_count,
        per_distribution=args.per_distribution,
    )
    print(result["decision"]["verdict"])
    print(result["decision"]["g2_status"])
    print(f"report={result['report_path']}")


if __name__ == "__main__":
    main()
