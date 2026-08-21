"""Resolve the G2 carry and reduced-grid identifiability blockers.

This analysis deliberately does not generate the final research dataset or train
an ANN/PINN.  It conditions the canonical Double Heston inverse problem on a
predeclared carry contract, measures scaled local identifiability, and performs
deterministic clean/noisy multi-start recovery experiments.
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
import yaml
from scipy.optimize import least_squares

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.run_g2_common_support_analysis import run_analysis as run_market_support
from src.calibrate_double_heston import (
    boundary_diagnostics,
    load_hard_safety_bounds,
    unconstrained_to_parameters,
)
from src.constants import CALL_OPTION, PARAMETER_NAMES, PUT_OPTION
from src.constraints import validate_parameters
from src.double_heston import price_double_heston_surface


BOUNDS_PATH = REPOSITORY_ROOT / "configs" / "parameter_bounds_PROVISIONAL.yaml"
INTERIOR_SAMPLE_PATH = (
    REPOSITORY_ROOT / "outputs" / "reviewed_sampling_audit" / "interior_accepted.csv"
)
WIDE_SAMPLE_PATH = (
    REPOSITORY_ROOT / "outputs" / "reviewed_sampling_audit" / "wide_valid_candidates.csv"
)
STAGE_A_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "market_data_audit_stage_a.yaml"
MARKET_AUDIT_SOURCE_PATH = REPOSITORY_ROOT / "src" / "market_data_audit.py"
SYNTHETIC_SOURCE_PATH = REPOSITORY_ROOT / "src" / "synthetic_dataset.py"
DEFAULT_DERIVED_ROOT = REPOSITORY_ROOT / "market_data_audit" / "stage_a" / "derived"
DEFAULT_OUTPUT_ROOT = DEFAULT_DERIVED_ROOT / "g2_identifiability"
DEFAULT_REPORT_PATH = REPOSITORY_ROOT / "docs" / "G2_IDENTIFIABILITY_ANALYSIS.md"

# Predeclared before any numerical experiment. Each retained expiry contributes
# its discount factor D(T) and normalized forward F(T)/S. The production pricer
# is called separately per expiry after recovering the equivalent scalar r(T)
# and q(T); carry is never a calibration target.
CARRY_CONTRACT_ID = "discount_forward_per_maturity_v1"
CARRY_INPUT_ORDER = (
    "discount_factor_near",
    "forward_over_spot_near",
    "discount_factor_middle",
    "forward_over_spot_middle",
)
CARRY_INPUT_COUNT = 4
SURFACE_SUBTOTAL = 22
CANDIDATE_INPUT_DIMENSION = CARRY_INPUT_COUNT + SURFACE_SUBTOTAL
EXPERIMENT_RATES = (0.0600, 0.0625)
EXPERIMENT_DIVIDEND_YIELDS = (0.0200, 0.0225)

MONEYNESS_CENTRAL_3 = (-0.05, 0.0, 0.05)
MONEYNESS_CENTRAL_5 = (-0.10, -0.05, 0.0, 0.05, 0.10)
MONEYNESS_CENTRAL_7 = (-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15)
MATURITY_PROFILES = (
    ("2026-07-01", (27, 55)),
    ("2026-07-15", (13, 41)),
    ("2026-07-22", (6, 34)),
)

JACOBIAN_RELATIVE_STEP = 1.0e-4
PRACTICAL_RANK_RELATIVE_TOLERANCE = 1.0e-6
CONDITION_WARNING_THRESHOLD = 1.0e8
RECOVERY_SCALED_RMSE_THRESHOLD = 0.05
RECOVERY_SCALED_MAX_ERROR_THRESHOLD = 0.15
NOISE_LEVELS = (0.0, 0.005, 0.01)
ANALYSIS_SEED = 20260810
FULL_PRICER_NODE_COUNT = 64

PREVIOUS_G2_ARTIFACTS = (
    "g2_maturity_support.csv",
    "g2_moneyness_support.csv",
    "g2_surface_support.csv",
    "g2_representation_candidates.csv",
    "g2_figures/g2_maturity_support_heatmap.png",
    "g2_figures/g2_moneyness_support_heatmap.png",
    "g2_figures/g2_candidate_comparison.png",
)


@dataclass(frozen=True)
class Representation:
    representation_id: str
    moneyness_nodes: tuple[float, ...]
    option_types: tuple[str, ...]
    market_support_status: str


REPRESENTATIONS = (
    Representation(
        "central5_calls_puts",
        MONEYNESS_CENTRAL_5,
        (CALL_OPTION, PUT_OPTION),
        "PROPOSED_MARKET_GEOMETRY",
    ),
    Representation(
        "central3_calls_puts",
        MONEYNESS_CENTRAL_3,
        (CALL_OPTION, PUT_OPTION),
        "PASS_BUT_DOMINATED",
    ),
    Representation(
        "central7_calls_puts",
        MONEYNESS_CENTRAL_7,
        (CALL_OPTION, PUT_OPTION),
        "EVIDENCE_COMPARATOR_ONLY_STAGE_A_SUPPORT_FAIL",
    ),
    Representation(
        "central5_call_only",
        MONEYNESS_CENTRAL_5,
        (CALL_OPTION,),
        "DIAGNOSTIC_PARITY_COMPARATOR_ONLY",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_existing_g2(derived_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in PREVIOUS_G2_ARTIFACTS:
        path = derived_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required prior G2 artifact is missing: {path}")
        result[relative] = _sha256(path)
    return result


def _assert_snapshot_unchanged(derived_root: Path, baseline: dict[str, str]) -> None:
    current = _snapshot_existing_g2(derived_root)
    if current != baseline:
        changed = sorted(name for name in baseline if baseline[name] != current.get(name))
        raise RuntimeError(f"Prior G2 evidence changed unexpectedly: {changed}")


def inspect_carry_contract_sources() -> dict[str, Any]:
    """Derive market readiness from the checked-in source/config contracts."""
    stage_a = yaml.safe_load(STAGE_A_CONFIG_PATH.read_text(encoding="utf-8"))
    market_source = MARKET_AUDIT_SOURCE_PATH.read_text(encoding="utf-8")
    synthetic_source = SYNTHETIC_SOURCE_PATH.read_text(encoding="utf-8")
    carry_audit = stage_a.get("futures_implied_carry_audit", {})
    explicit_rate_source = bool(
        carry_audit.get("risk_free_rate_source_selected", False)
        or carry_audit.get("discount_factor_source_selected", False)
        or carry_audit.get("yield_curve_source_selected", False)
    )
    combined_carry_helper = "futures_implied_carry" in market_source
    generic_synthetic_has_hidden_r_q = (
        "risk_free_rate" in synthetic_source
        and "dividend_yield" in synthetic_source
        and "normalized_price" in synthetic_source
    )
    return {
        "carry_contract_id": CARRY_CONTRACT_ID,
        "pricing_compatible_via_expirywise_calls": True,
        "reviewed_synthetic_contract_implemented": False,
        "generic_synthetic_has_hidden_r_q_confound": generic_synthetic_has_hidden_r_q,
        "futures_combined_carry_available": combined_carry_helper,
        "verified_external_rate_or_discount_source": explicit_rate_source,
        "market_discount_forward_ready": combined_carry_helper and explicit_rate_source,
        "candidate_input_dimension": CANDIDATE_INPUT_DIMENSION,
        "reason": (
            "NSE spot/futures can supply F/S but not D=exp(-rT). The checked-in "
            "Stage A contract has no verified external rate/discount source, so the "
            "four maturity-aligned discount/forward coordinates are not yet available."
        ),
    }


def build_grid(
    representation: Representation,
    maturity_days: Sequence[int],
    *,
    spot: float = 100.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    strikes: list[float] = []
    maturities: list[float] = []
    option_types: list[str] = []
    for option_type in representation.option_types:
        for days in maturity_days:
            for node in representation.moneyness_nodes:
                strikes.append(spot * math.exp(node))
                maturities.append(float(days) / 365.0)
                option_types.append(option_type)
    return (
        np.asarray(strikes, dtype=np.float64),
        np.asarray(maturities, dtype=np.float64),
        np.asarray(option_types, dtype=str),
    )


def normalized_observables(
    parameters: Sequence[float],
    representation: Representation,
    maturity_days: Sequence[int],
    *,
    node_count: int,
    spot: float = 100.0,
    rates: Sequence[float] = EXPERIMENT_RATES,
    dividend_yields: Sequence[float] = EXPERIMENT_DIVIDEND_YIELDS,
) -> np.ndarray:
    expiry_count = len(maturity_days)
    if expiry_count < 1 or len(rates) != expiry_count or len(dividend_yields) != expiry_count:
        raise ValueError(
            "Maturity days, rates, and dividend yields must contain the same "
            "positive number of maturity-aligned terms"
        )
    call_pieces: list[np.ndarray] = []
    put_pieces: list[np.ndarray] = []
    strikes = spot * np.exp(np.asarray(representation.moneyness_nodes))
    for index, days in enumerate(maturity_days):
        maturity = float(days) / 365.0
        discount = math.exp(-float(rates[index]) * maturity)
        forward_over_spot = math.exp(
            (float(rates[index]) - float(dividend_yields[index])) * maturity
        )
        recovered_rate = -math.log(discount) / maturity
        recovered_q = recovered_rate - math.log(forward_over_spot) / maturity
        calls = price_double_heston_surface(
            spot,
            strikes,
            np.full(len(strikes), maturity),
            recovered_rate,
            recovered_q,
            [CALL_OPTION] * len(strikes),
            parameters,
            node_count=node_count,
        )
        call_pieces.append(calls / spot)
        if PUT_OPTION in representation.option_types:
            puts = calls - spot * math.exp(-recovered_q * maturity) + strikes * discount
            put_pieces.append(puts / spot)
    pieces: list[np.ndarray] = []
    for option_type in representation.option_types:
        pieces.extend(call_pieces if option_type == CALL_OPTION else put_pieces)
    return np.concatenate(pieces)


def _parameter_widths(
    bounds: dict[str, tuple[float, float]],
) -> np.ndarray:
    return np.asarray(
        [bounds[name][1] - bounds[name][0] for name in PARAMETER_NAMES],
        dtype=np.float64,
    )


def scaled_parameter_jacobian(
    parameters: Sequence[float],
    representation: Representation,
    maturity_days: Sequence[int],
    bounds: dict[str, tuple[float, float]],
    *,
    node_count: int = 32,
    rates: Sequence[float] = EXPERIMENT_RATES,
    dividend_yields: Sequence[float] = EXPERIMENT_DIVIDEND_YIELDS,
) -> np.ndarray:
    """Differentiate spot-normalized prices by full-range-scaled parameters."""
    vector = np.asarray(parameters, dtype=np.float64)
    widths = _parameter_widths(bounds)
    columns: list[np.ndarray] = []
    for index, width in enumerate(widths):
        step = JACOBIAN_RELATIVE_STEP * width
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
            raise RuntimeError(
                f"Could not form a valid central difference for {PARAMETER_NAMES[index]}"
            )
        lower_prices = normalized_observables(
            lower,
            representation,
            maturity_days,
            node_count=node_count,
            rates=rates,
            dividend_yields=dividend_yields,
        )
        upper_prices = normalized_observables(
            upper,
            representation,
            maturity_days,
            node_count=node_count,
            rates=rates,
            dividend_yields=dividend_yields,
        )
        # Multiplying by the declared width yields a derivative with respect to
        # a dimensionless full-range coordinate and removes parameter-unit bias.
        columns.append((upper_prices - lower_prices) * width / (2.0 * step))
    return np.column_stack(columns)


def _read_accepted(path: Path, distribution: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "accepted" in frame:
        accepted = frame["accepted"].astype(str).str.lower().eq("true")
        frame = frame.loc[accepted].copy()
    frame["distribution"] = distribution
    required = set(PARAMETER_NAMES) | {"candidate_id", "distribution"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError(f"Invalid accepted-parameter evidence: {path}")
    return frame.sort_values("candidate_id", kind="stable").reset_index(drop=True)


def _maximin_select(
    frame: pd.DataFrame,
    count: int,
    bounds: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    if len(frame) < count:
        raise ValueError(f"Need {count} parameter rows, found {len(frame)}")
    values = frame[PARAMETER_NAMES].to_numpy(dtype=np.float64)
    lower = np.asarray([bounds[name][0] for name in PARAMETER_NAMES])
    scaled = (values - lower) / _parameter_widths(bounds)
    center = np.median(scaled, axis=0)
    selected = [int(np.argmin(np.linalg.norm(scaled - center, axis=1)))]
    while len(selected) < count:
        distances = np.min(
            np.linalg.norm(scaled[:, None, :] - scaled[selected][None, :, :], axis=2),
            axis=1,
        )
        distances[selected] = -np.inf
        selected.append(int(np.argmax(distances)))
    result = frame.iloc[selected].copy().reset_index(drop=True)
    result.insert(0, "sample_index", np.arange(len(result), dtype=int))
    return result


def select_representative_parameters(
    bounds: dict[str, tuple[float, float]], *, per_distribution: int = 4
) -> pd.DataFrame:
    interior = _maximin_select(
        _read_accepted(INTERIOR_SAMPLE_PATH, "interior_train"),
        per_distribution,
        bounds,
    )
    wide = _maximin_select(
        _read_accepted(WIDE_SAMPLE_PATH, "wide_valid_train"),
        per_distribution,
        bounds,
    )
    result = pd.concat([interior, wide], ignore_index=True)
    result["sample_id"] = [
        f"{row.distribution}_{int(row.candidate_id)}" for row in result.itertuples()
    ]
    result["sample_index"] = np.arange(len(result), dtype=int)
    return result


def run_jacobian_experiment(
    samples: pd.DataFrame,
    bounds: dict[str, tuple[float, float]],
    *,
    node_count: int = 32,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    for sample in samples.itertuples(index=False):
        parameters = np.asarray([getattr(sample, name) for name in PARAMETER_NAMES])
        for profile_id, maturity_days in MATURITY_PROFILES:
            for representation in REPRESENTATIONS:
                jacobian = scaled_parameter_jacobian(
                    parameters,
                    representation,
                    maturity_days,
                    bounds,
                    node_count=node_count,
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
                    PRACTICAL_RANK_RELATIVE_TOLERANCE * singular_values[0]
                )
                numerical_rank = int(np.sum(singular_values > numerical_tolerance))
                practical_rank = int(np.sum(singular_values > practical_tolerance))
                smallest = float(singular_values[-1])
                condition = (
                    float(singular_values[0] / smallest)
                    if smallest > 0.0
                    else math.inf
                )
                key = {
                    "sample_id": sample.sample_id,
                    "distribution": sample.distribution,
                    "maturity_profile": profile_id,
                    "near_dte": maturity_days[0],
                    "middle_dte": maturity_days[1],
                    "representation_id": representation.representation_id,
                    "market_support_status": representation.market_support_status,
                    "observable_count": jacobian.shape[0],
                }
                summary_rows.append(
                    {
                        **key,
                        "numerical_rank": numerical_rank,
                        "practical_rank_1e_minus_6": practical_rank,
                        "largest_singular_value": float(singular_values[0]),
                        "smallest_singular_value": smallest,
                        "condition_number": condition,
                        **{
                            f"singular_value_{index + 1}": float(value)
                            for index, value in enumerate(singular_values)
                        },
                    }
                )
                column_norms = np.linalg.norm(jacobian, axis=0)
                for name, sensitivity in zip(PARAMETER_NAMES, column_norms, strict=True):
                    sensitivity_rows.append(
                        {**key, "parameter": name, "scaled_sensitivity": float(sensitivity)}
                    )
                near_null = right_vectors[-1]
                for name, loading in zip(PARAMETER_NAMES, near_null, strict=True):
                    null_rows.append(
                        {
                            **key,
                            "parameter": name,
                            "near_null_loading": float(loading),
                            "absolute_near_null_loading": float(abs(loading)),
                        }
                    )
    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(sensitivity_rows),
        pd.DataFrame(null_rows),
    )


def _deterministic_starts(seed: int, count: int = 3) -> list[tuple[str, np.ndarray]]:
    if count < 1:
        raise ValueError("At least one deterministic start is required")
    rng = np.random.default_rng(seed)
    starts = [("neutral_transform_midpoint", np.zeros(10, dtype=np.float64))]
    for index in range(1, count):
        starts.append(
            (f"deterministic_broad_{index}", rng.normal(0.0, 1.25, size=10))
        )
    return starts


def _one_recovery(
    true_parameters: np.ndarray,
    observed: np.ndarray,
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
    representation = REPRESENTATIONS[0]
    widths = _parameter_widths(bounds)

    def residuals(latent: np.ndarray) -> np.ndarray:
        candidate = unconstrained_to_parameters(latent, bounds)
        return normalized_observables(
            candidate, representation, maturity_days, node_count=node_count
        ) - observed

    rows: list[dict[str, Any]] = []
    for start_index, (strategy, start) in enumerate(
        _deterministic_starts(seed, start_count)
    ):
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "maturity_profile": profile_id,
            "near_dte": maturity_days[0],
            "middle_dte": maturity_days[1],
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
            predicted = normalized_observables(
                recovered, representation, maturity_days, node_count=node_count
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
                    "price_mae_normalized": float(np.mean(np.abs(predicted - observed))),
                    "aggregate_scaled_parameter_rmse": scaled_rmse,
                    "max_scaled_parameter_error": scaled_max,
                    "constraint_valid": bool(validation["is_valid"]),
                    "bound_hit": bool(bound_reasons),
                    "bound_reasons": ";".join(bound_reasons),
                    "parameter_recovery_success": bool(
                        result.success
                        and validation["is_valid"]
                        and scaled_rmse <= RECOVERY_SCALED_RMSE_THRESHOLD
                        and scaled_max <= RECOVERY_SCALED_MAX_ERROR_THRESHOLD
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


def run_recovery_experiment(
    samples: pd.DataFrame,
    bounds: dict[str, tuple[float, float]],
    *,
    node_count: int = 32,
    max_nfev: int = 120,
    start_count: int = 3,
    per_distribution: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Deterministic representatives from each reviewed distribution. Maturity
    # profiles rotate across samples; all profiles are covered without turning
    # this diagnostic into large-scale dataset generation.
    recovery_samples = (
        samples.groupby("distribution", sort=True, group_keys=False)
        .head(per_distribution)
        .reset_index(drop=True)
    )
    rows: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(recovery_samples.itertuples(index=False)):
        true_parameters = np.asarray(
            [getattr(sample, name) for name in PARAMETER_NAMES], dtype=np.float64
        )
        profile_index = sample_index % len(MATURITY_PROFILES)
        profile_id, maturity_days = MATURITY_PROFILES[profile_index]
        clean = normalized_observables(
            true_parameters,
            REPRESENTATIONS[0],
            maturity_days,
            node_count=node_count,
        )
        for noise_index, noise_level in enumerate(NOISE_LEVELS):
            noise_seed = ANALYSIS_SEED + 10000 * sample_index + 100 * profile_index + noise_index
            rng = np.random.default_rng(noise_seed)
            observed = clean * (
                1.0 + rng.normal(0.0, noise_level, size=clean.shape)
            )
            if np.any(observed < 0.0):
                raise RuntimeError("Noise produced a negative normalized price")
            rows.extend(
                _one_recovery(
                    true_parameters,
                    observed,
                    maturity_days,
                    bounds,
                    sample_id=sample.sample_id,
                    profile_id=profile_id,
                    noise_level=noise_level,
                    node_count=node_count,
                    max_nfev=max_nfev,
                    start_count=start_count,
                    seed=noise_seed + 500000,
                )
            )
    starts = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    group_columns = ["sample_id", "maturity_profile", "noise_level"]
    for keys, group in starts.groupby(group_columns, sort=True):
        finite = group.loc[np.isfinite(group["price_rmse_normalized"])].copy()
        best = (
            finite.sort_values("price_rmse_normalized", kind="stable").iloc[0]
            if not finite.empty
            else group.iloc[0]
        )
        recovered = finite[[f"recovered_{name}" for name in PARAMETER_NAMES]]
        widths = _parameter_widths(bounds)
        variability = (
            float(np.mean(recovered.std(ddof=0).to_numpy(dtype=float) / widths))
            if len(recovered) > 1
            else math.nan
        )
        row = {
            "sample_id": keys[0],
            "maturity_profile": keys[1],
            "noise_level": keys[2],
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


def decide_gate(
    carry: dict[str, Any], jacobian: pd.DataFrame, recovery: pd.DataFrame
) -> dict[str, Any]:
    proposed = jacobian.loc[
        jacobian["representation_id"] == "central5_calls_puts"
    ]
    jacobian_rank_pass = bool(
        (proposed["practical_rank_1e_minus_6"] == len(PARAMETER_NAMES)).all()
    )
    conditioning_pass = bool(
        np.isfinite(proposed["condition_number"]).all()
        and (proposed["condition_number"] <= CONDITION_WARNING_THRESHOLD).all()
    )
    clean = recovery.loc[recovery["noise_level"] == 0.0]
    noise_half = recovery.loc[recovery["noise_level"] == 0.005]
    noise_one = recovery.loc[recovery["noise_level"] == 0.01]

    def frequency(frame: pd.DataFrame) -> float:
        return float(frame["parameter_recovery_success_count"].sum() / frame["start_count"].sum())

    clean_pass = frequency(clean) >= 0.80
    half_pass = frequency(noise_half) >= 0.70
    one_pass = frequency(noise_one) >= 0.50
    market_carry_pass = bool(carry["market_discount_forward_ready"])
    passed = all(
        (market_carry_pass, jacobian_rank_pass, conditioning_pass, clean_pass, half_pass, one_pass)
    )
    return {
        "g2_verdict": "PASSED" if passed else "NOT_PASSED",
        "market_carry_pass": market_carry_pass,
        "jacobian_rank_pass": jacobian_rank_pass,
        "conditioning_pass": conditioning_pass,
        "clean_recovery_pass": clean_pass,
        "noise_0_5pct_recovery_pass": half_pass,
        "noise_1pct_recovery_pass": one_pass,
        "clean_recovery_frequency": frequency(clean),
        "noise_0_5pct_recovery_frequency": frequency(noise_half),
        "noise_1pct_recovery_frequency": frequency(noise_one),
    }


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


def write_figures(
    jacobian: pd.DataFrame,
    sensitivity: pd.DataFrame,
    recovery: pd.DataFrame,
    output_root: Path,
) -> tuple[Path, ...]:
    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    proposed = jacobian.loc[
        jacobian["representation_id"] == "central5_calls_puts"
    ]
    singular_columns = [f"singular_value_{index}" for index in range(1, 11)]
    figure, axis = plt.subplots(figsize=(8.5, 5.0))
    for row in proposed.itertuples(index=False):
        values = [getattr(row, name) for name in singular_columns]
        axis.semilogy(range(1, 11), values, color="#4C78A8", alpha=0.22)
    medians = proposed[singular_columns].median().to_numpy(dtype=float)
    axis.semilogy(range(1, 11), medians, color="#D62728", marker="o", linewidth=2.2, label="median")
    axis.set(title="Central-5 scaled Jacobian singular values", xlabel="Singular-value index", ylabel="Singular value (log scale)")
    axis.set_xticks(range(1, 11))
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path = figure_root / "jacobian_singular_values.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)

    figure, axis = plt.subplots(figsize=(8.5, 5.0))
    order = [item.representation_id for item in REPRESENTATIONS]
    values = [
        jacobian.loc[jacobian["representation_id"] == item, "condition_number"].to_numpy()
        for item in order
    ]
    axis.boxplot(values, tick_labels=order, showfliers=True)
    axis.set_yscale("log")
    axis.axhline(CONDITION_WARNING_THRESHOLD, color="#D62728", linestyle="--", label="1e8 warning threshold")
    axis.set(title="Scaled Jacobian condition-number distribution", ylabel="Condition number (log scale)")
    axis.tick_params(axis="x", rotation=18)
    axis.grid(True, which="both", axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path = figure_root / "condition_number_distribution.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)

    proposed_sensitivity = sensitivity.loc[
        sensitivity["representation_id"] == "central5_calls_puts"
    ]
    med = proposed_sensitivity.groupby("parameter")["scaled_sensitivity"].median().reindex(PARAMETER_NAMES)
    figure, axis = plt.subplots(figsize=(9.0, 5.0))
    axis.bar(PARAMETER_NAMES, med.to_numpy(), color="#59A14F")
    axis.set_yscale("log")
    axis.set(title="Median scaled parameter-direction sensitivity", ylabel="Jacobian column norm (log scale)")
    axis.tick_params(axis="x", rotation=35)
    axis.grid(True, which="both", axis="y", alpha=0.25)
    figure.tight_layout()
    path = figure_root / "parameter_sensitivity.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)

    figure, axis = plt.subplots(figsize=(8.5, 5.0))
    groups = [
        recovery.loc[recovery["noise_level"] == level, "best_aggregate_scaled_parameter_rmse"].to_numpy()
        for level in NOISE_LEVELS
    ]
    axis.boxplot(groups, tick_labels=["clean", "0.5%", "1.0%"], showfliers=True)
    axis.axhline(RECOVERY_SCALED_RMSE_THRESHOLD, color="#D62728", linestyle="--", label="5% range-RMSE criterion")
    axis.set(title="Best-repricing start: parameter recovery error", ylabel="Aggregate parameter RMSE / hard-bound range")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path = figure_root / "clean_noisy_recovery.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)

    clean = recovery.loc[recovery["noise_level"] == 0.0]
    true_values: list[float] = []
    recovered_values: list[float] = []
    labels: list[str] = []
    for name in PARAMETER_NAMES:
        true_values.extend(clean[f"true_{name}"].tolist())
        recovered_values.extend(clean[f"best_recovered_{name}"].tolist())
        labels.extend([name] * len(clean))
    figure, axis = plt.subplots(figsize=(7.0, 6.0))
    for name in PARAMETER_NAMES:
        mask = np.asarray([label == name for label in labels])
        axis.scatter(np.asarray(true_values)[mask], np.asarray(recovered_values)[mask], s=18, alpha=0.7, label=name)
    finite_values = np.asarray(true_values + recovered_values, dtype=float)
    low, high = float(np.nanmin(finite_values)), float(np.nanmax(finite_values))
    axis.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1)
    axis.set(title="Clean recovery: recovered versus true", xlabel="True parameter", ylabel="Recovered parameter")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    path = figure_root / "recovered_vs_true_clean.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)
    return tuple(paths)


def render_report(
    carry: dict[str, Any],
    samples: pd.DataFrame,
    jacobian: pd.DataFrame,
    sensitivity: pd.DataFrame,
    null_vectors: pd.DataFrame,
    recovery: pd.DataFrame,
    decision: dict[str, Any],
    artifact_hashes: dict[str, str],
) -> str:
    proposed = jacobian.loc[jacobian["representation_id"] == "central5_calls_puts"]
    weakest_sensitivity = (
        sensitivity.loc[sensitivity["representation_id"] == "central5_calls_puts"]
        .groupby("parameter")["scaled_sensitivity"]
        .median()
        .sort_values()
    )
    weakest_null = (
        null_vectors.loc[null_vectors["representation_id"] == "central5_calls_puts"]
        .groupby("parameter")["absolute_near_null_loading"]
        .median()
        .sort_values(ascending=False)
    )
    lines = [
        "# G2 Carry and Reduced-Grid Identifiability Analysis",
        "",
        "## Decision",
        "",
        f"**G2 = {decision['g2_verdict']}.**",
        "",
        "The market-supported central-5 geometry was not changed. This analysis conditions the inverse problem on maturity-aligned discount factors and normalized forwards, then tests only the canonical ten Double Heston targets. A low repricing error is not treated as parameter recovery.",
        "",
        "## Predeclared carry contract",
        "",
        f"- Contract: `{CARRY_CONTRACT_ID}`.",
        "- Input order: `[D_near, F_near/S, D_middle, F_middle/S, T_near, T_middle]`, then option-major (`call`, `put`), expiry-major (`near`, `middle`), moneyness ascending.",
        f"- Candidate dimension: **{CANDIDATE_INPUT_DIMENSION}** = 4 carry + 2 maturity + 20 normalized-price coordinates.",
        "- Synthetic experiment term structure: `r=(0.0600, 0.0625)`, `q=(0.0200, 0.0225)`; the resulting `D` and `F/S` coordinates are known inputs, not fitted targets.",
        "- Compatibility: the canonical scalar-carry engine is called separately per expiry after exact conversion from each `(D,F/S)` pair. The reviewed synthetic generator for this contract does not yet exist.",
        "- Market limitation: official NSE spot/futures can provide `F/S`, but not `D=exp(-rT)`. The checked-in Stage A contract has no verified external short-rate/discount source or selected futures price field.",
        "",
        "Alternatives were not silently adopted: explicit `(r_i,q_i)` contains equivalent information but encourages an unsupported interpretation that futures identify `q_i`; a carry-removing forward normalization changes `K/S` to `K/F` and therefore requires a new market-support audit. The current scalar `(r,q)` surface API was rejected because it would impose flat carry across two expiries.",
        "",
        "## Experiment design",
        "",
        f"- Deterministic maximin sample: {len(samples)} valid vectors, balanced across reviewed interior and wide-valid evidence; all three observed near/middle DTE profiles were tested.",
        f"- Canonical production-pricer quadrature: `{FULL_PRICER_NODE_COUNT}` Gauss-Laguerre nodes for the full experiment.",
        "- Jacobian: central finite differences of spot-normalized prices; each parameter column is scaled by its full hard-bound width.",
        f"- Practical rank threshold: singular value greater than `{PRACTICAL_RANK_RELATIVE_TOLERANCE:.0e}` times the largest singular value.",
        f"- Conditioning warning threshold: `{CONDITION_WARNING_THRESHOLD:.0e}`.",
        "- Recovery: three target-blind deterministic starts, constrained latent transformation, and clean/0.5%/1.0% independent multiplicative price noise.",
        f"- Parameter recovery requires aggregate range-scaled RMSE <= {RECOVERY_SCALED_RMSE_THRESHOLD:.2f} and maximum range-scaled error <= {RECOVERY_SCALED_MAX_ERROR_THRESHOLD:.2f}; optimizer convergence alone is insufficient.",
        "",
        "## Jacobian results",
        "",
        f"Central-5 numerical rank 10 frequency: `{100.0 * (proposed['numerical_rank'] == 10).mean():.1f}%`; practical rank 10 frequency: `{100.0 * (proposed['practical_rank_1e_minus_6'] == 10).mean():.1f}%`.",
        f"Smallest singular value: median `{proposed['smallest_singular_value'].median():.3e}`, range `{proposed['smallest_singular_value'].min():.3e}` to `{proposed['smallest_singular_value'].max():.3e}`.",
        f"Condition number: median `{proposed['condition_number'].median():.3e}`, 90th percentile `{proposed['condition_number'].quantile(0.9):.3e}`, maximum `{proposed['condition_number'].max():.3e}`.",
        "",
        "Weakest median scaled-sensitivity parameters: "
        + ", ".join(f"`{name}` ({value:.3e})" for name, value in weakest_sensitivity.head(5).items())
        + ".",
        "Dominant median absolute loadings in the weakest right-singular direction: "
        + ", ".join(f"`{name}` ({value:.3f})" for name, value in weakest_null.head(5).items())
        + ".",
        "",
        "## Representation comparison",
        "",
        "| Representation | Observables | Practical rank 10 | Median condition number | Market status |",
        "|---|---:|---:|---:|---|",
    ]
    for representation in REPRESENTATIONS:
        group = jacobian.loc[jacobian["representation_id"] == representation.representation_id]
        lines.append(
            f"| `{representation.representation_id}` | {int(group['observable_count'].iloc[0])} | "
            f"{100.0 * (group['practical_rank_1e_minus_6'] == 10).mean():.1f}% | "
            f"{group['condition_number'].median():.3e} | {representation.market_support_status} |"
        )
    lines.extend(
        [
            "",
            "Calls and puts are parity-related conditional on known carry, so duplicating them does not create twenty independent stochastic-volatility observations. The call-only comparator quantifies this directly; both option types remain in the market representation for observed-data robustness, not as a claim of twenty independent equations.",
            "",
            "## Deterministic multi-start recovery",
            "",
            "| Noise | Cases | Optimizer success | Parameter-recovery success | Median best price RMSE | Median best parameter RMSE | Bound-hit starts | Median start variability |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for noise_level in NOISE_LEVELS:
        group = recovery.loc[recovery["noise_level"] == noise_level]
        total_starts = int(group["start_count"].sum())
        lines.append(
            f"| {100.0 * noise_level:.1f}% | {len(group)} | "
            f"{int(group['optimizer_success_count'].sum())}/{total_starts} | "
            f"{int(group['parameter_recovery_success_count'].sum())}/{total_starts} | "
            f"{group['best_price_rmse_normalized'].median():.3e} | "
            f"{group['best_aggregate_scaled_parameter_rmse'].median():.3e} | "
            f"{int(group['bound_hit_count'].sum())}/{total_starts} | "
            f"{group['mean_start_parameter_std_scaled'].median():.3e} |"
        )
    lines.extend(
        [
            "",
            "Median best-start absolute parameter error scaled by each hard-bound width:",
            "",
            "| Parameter | Clean | 0.5% noise | 1.0% noise |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in PARAMETER_NAMES:
        values = []
        for noise_level in NOISE_LEVELS:
            group = recovery.loc[recovery["noise_level"] == noise_level]
            values.append(float(group[f"best_scaled_error_{name}"].median()))
        lines.append(
            f"| `{name}` | {values[0]:.3e} | {values[1]:.3e} | {values[2]:.3e} |"
        )
    lines.extend(
        [
            "",
            "Per-parameter errors, every optimizer start, constraint validity, bound diagnostics, and true/recovered vectors are retained in the CSV evidence. Best-start rows are selected by repricing error, not by knowledge of the true parameters.",
            "",
            "## Gate components",
            "",
            "| Component | Pass |",
            "|---|---|",
        ]
    )
    for key in (
        "market_carry_pass",
        "jacobian_rank_pass",
        "conditioning_pass",
        "clean_recovery_pass",
        "noise_0_5pct_recovery_pass",
        "noise_1pct_recovery_pass",
    ):
        lines.append(f"| `{key}` | `{decision[key]}` |")
    lines.extend(
        [
            "",
            "## Minimum remedy",
            "",
            "1. Add and provenance-validate a tenor-aligned external short-rate/discount source and select the official NSE futures price field, producing `(D_i,F_i/S)` without pretending futures identify `r_i` and `q_i` separately.",
            "2. Treat the weak right-singular combinations reported above as the actual identifiability failure. Do not repair them by using the unsupported central-7 wings.",
            "3. Reopen the market-supported information design: add independently supported maturities or complementary observables, reduce/reparameterize the ten targets, or introduce scientifically justified priors. Re-run the same rank and target-blind noisy-recovery gates afterward.",
            "",
            "No final 10k dataset was generated. No ANN or PINN was trained.",
            "",
            "## Reproducibility and preservation",
            "",
            "The prior G2 analysis is replayed without writes before this experiment. Its four CSVs, three plots, and the eight canonical Stage A hashes are checked for preservation.",
            "",
            "| New evidence artifact | SHA-256 |",
            "|---|---|",
        ]
    )
    for name, digest in artifact_hashes.items():
        lines.append(f"| `{name}` | `{digest.upper()}` |")
    lines.extend(
        [
            "",
            "```text",
            f"CARRY_CONTRACT = {CARRY_CONTRACT_ID}",
            f"CANDIDATE_INPUT_DIMENSION = {CANDIDATE_INPUT_DIMENSION}",
            f"G2 = {decision['g2_verdict']}",
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
    # Revalidate the prior gate and official-NSE provenance without rewriting it.
    market_result = run_market_support(write_outputs=False)
    previous_snapshot = _snapshot_existing_g2(DEFAULT_DERIVED_ROOT)
    carry = inspect_carry_contract_sources()
    bounds = load_hard_safety_bounds(BOUNDS_PATH)
    samples = select_representative_parameters(
        bounds, per_distribution=2 if quick else 4
    )
    jacobian, sensitivity, null_vectors = run_jacobian_experiment(
        samples, bounds, node_count=16 if quick else FULL_PRICER_NODE_COUNT
    )
    recovery_starts, recovery_summary = run_recovery_experiment(
        samples,
        bounds,
        node_count=16 if quick else FULL_PRICER_NODE_COUNT,
        max_nfev=30 if quick else 120,
        start_count=2 if quick else 3,
        per_distribution=1 if quick else 2,
    )
    decision = decide_gate(carry, jacobian, recovery_summary)
    evidence = {
        "representative_parameters.csv": samples,
        "jacobian_summary.csv": jacobian,
        "parameter_sensitivity.csv": sensitivity,
        "near_null_directions.csv": null_vectors,
        "recovery_starts.csv": recovery_starts,
        "recovery_summary.csv": recovery_summary,
    }
    evidence_paths: dict[str, Path] = {}
    figure_paths: tuple[Path, ...] = ()
    artifact_hashes: dict[str, str] = {}
    if write_outputs:
        output_root.mkdir(parents=True, exist_ok=True)
        for name, frame in evidence.items():
            path = output_root / name
            _write_csv(frame, path)
            evidence_paths[name] = path
            artifact_hashes[name] = _sha256(path)
        figure_paths = write_figures(
            jacobian, sensitivity, recovery_summary, output_root
        )
        for path in figure_paths:
            artifact_hashes[str(path.relative_to(output_root)).replace("\\", "/")] = _sha256(path)
        report = render_report(
            carry,
            samples,
            jacobian,
            sensitivity,
            null_vectors,
            recovery_summary,
            decision,
            artifact_hashes,
        )
        _atomic_write_bytes(report_path, report.encode("utf-8"))
    else:
        report = render_report(
            carry,
            samples,
            jacobian,
            sensitivity,
            null_vectors,
            recovery_summary,
            decision,
            {},
        )
    _assert_snapshot_unchanged(DEFAULT_DERIVED_ROOT, previous_snapshot)
    return {
        "market_result": market_result,
        "carry": carry,
        "samples": samples,
        "jacobian": jacobian,
        "sensitivity": sensitivity,
        "null_vectors": null_vectors,
        "recovery_starts": recovery_starts,
        "recovery_summary": recovery_summary,
        "decision": decision,
        "report": report,
        "evidence_paths": evidence_paths,
        "figure_paths": figure_paths,
        "previous_g2_hashes": previous_snapshot,
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
    proposed = result["jacobian"].loc[
        result["jacobian"]["representation_id"] == "central5_calls_puts"
    ]
    print(
        f"carry_contract={CARRY_CONTRACT_ID} candidate_input_dimension={CANDIDATE_INPUT_DIMENSION} "
        f"central5_practical_rank10_frequency={(proposed['practical_rank_1e_minus_6'] == 10).mean():.6f} "
        f"central5_median_condition={proposed['condition_number'].median():.6e}"
    )
    print(f"G2={result['decision']['g2_verdict']}")
    for path in result["evidence_paths"].values():
        print(f"evidence={path}")
    for path in result["figure_paths"]:
        print(f"figure={path}")
    if not arguments.no_write:
        print(f"report={arguments.report_path}")


if __name__ == "__main__":
    main()
