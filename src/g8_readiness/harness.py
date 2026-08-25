"""Frozen pricing-family and inverse-method preparation interfaces."""

from __future__ import annotations

import math
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.optimize import least_squares, minimize_scalar

from ..calibrate_double_heston import load_hard_safety_bounds, unconstrained_to_parameters
from ..constants import PARAMETER_NAMES
from ..double_heston import price_double_heston_option
from ..r2_representation.serialization import payload_to_surface, surface_to_payload
from ..r2_representation.surface import R2Surface
from .contracts import (
    CALIBRATION_MONEYNESS,
    HOLDOUT_MONEYNESS,
    canonical_slot_roles,
    forward_black_price,
    implied_volatility,
)
from models.parameter_transform import TargetStandardizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BOUNDS_PATH = REPOSITORY_ROOT / "configs" / "parameter_bounds_PROVISIONAL.yaml"
NODE_COUNT = 64
MAX_NFEV = 300


def pricing_rows_from_surface(surface: R2Surface) -> pd.DataFrame:
    selected = {(int(row["rank"]), float(row["target"]), str(row["option_type"])): row for row in surface.metadata["selected_contracts"]}
    roles = canonical_slot_roles()
    rows: list[dict[str, Any]] = []
    for index, key in enumerate(surface.slot_keys):
        row = selected[(key.expiry_rank, key.target_log_moneyness, key.option_type)]
        role = "CALIBRATION" if key.target_log_moneyness in CALIBRATION_MONEYNESS else "HOLDOUT"
        maturity = float(surface.maturities[index])
        rows.append(
            {
                "slot_index": index,
                "expiry_rank": key.expiry_rank,
                "target_log_moneyness": key.target_log_moneyness,
                "option_type": key.option_type,
                "strike": float(row["strike"]),
                "observed_price": float(row["price"]) * surface.spot,
                "forward": float(row["forward"]),
                "discount": float(row["discount"]),
                "T": maturity,
                "continuous_rate": float(surface.rates[index]),
                "futures_implied_carry": float(surface.carries[index]),
                "spot": surface.spot,
        "sample_role": role,
                "market_iv": float(row["iv"]),
            }
        )
    return pd.DataFrame(rows)


def _price_metrics(frame: pd.DataFrame, predicted: np.ndarray) -> dict[str, float]:
    errors = predicted - frame["observed_price"].to_numpy(float)
    normalized_errors = errors / np.maximum(frame["observed_price"].to_numpy(float), 1.0)
    return {
        "price_rmse_dollar": float(np.sqrt(np.mean(errors ** 2))),
        "price_mae_dollar": float(np.mean(np.abs(errors))),
        "price_rmse_normalized": float(np.sqrt(np.mean(normalized_errors ** 2))),
    }


def _iv_metrics(frame: pd.DataFrame, predicted_prices: np.ndarray) -> dict[str, Any]:
    ivs: list[float] = []
    markets: list[float] = []
    failures = 0
    reasons: set[str] = set()
    for row, price in zip(frame.to_dict(orient="records"), predicted_prices, strict=True):
        try:
            ivs.append(implied_volatility(float(price), row["forward"], row["strike"], row["T"], row["discount"], row["option_type"]))
            markets.append(float(row["market_iv"]))
        except Exception as exc:
            failures += 1
            reasons.add(type(exc).__name__)
    valid = np.asarray(ivs, dtype=float)
    market = np.asarray(markets, dtype=float)
    finite = valid[np.isfinite(valid)]
    if len(finite):
        rmse = float(np.sqrt(np.mean((finite - market) ** 2)))
    else:
        rmse = float("nan")
    return {"iv_rmse_annualized": rmse, "iv_failure_count": failures, "failure_reasons": sorted(reasons)}


def _standard_heston_parameters(raw: Sequence[float]) -> np.ndarray:
    values = np.asarray(raw, dtype=float)
    unit = 1.0 / (1.0 + np.exp(-np.clip(values, -35.0, 35.0)))
    kappa = 0.05 + unit[0] * (12.0 - 0.05)
    theta = 0.002 + unit[1] * (0.30 - 0.002)
    v0 = 0.002 + unit[2] * (0.35 - 0.002)
    sigma_upper = min(1.5, math.sqrt(2.0 * kappa * theta) * (1.0 - 1e-7))
    sigma = 0.005 + unit[3] * (sigma_upper - 0.005)
    rho = 0.95 * math.tanh(float(values[4]))
    return np.asarray([kappa, theta, sigma, rho, v0])


PricingFunction = Callable[[pd.DataFrame, np.ndarray], np.ndarray]


def fit_pricing_family_surface(
    rows: pd.DataFrame,
    *,
    pricing_functions: Mapping[str, PricingFunction] | None = None,
    node_count: int = NODE_COUNT,
    max_nfev: int = MAX_NFEV,
) -> dict[str, Any]:
    """Fit BS/Heston/DH on inner slots only; wings are held out identically."""
    calibration_mask = rows["target_log_moneyness"].isin(CALIBRATION_MONEYNESS)
    holdout_mask = rows["target_log_moneyness"].isin(HOLDOUT_MONEYNESS)
    calibration = rows.loc[calibration_mask].copy()
    holdout = rows.loc[holdout_mask].copy()
    if len(calibration) < 6 or len(holdout) < 3:
        raise ValueError("pricing-family role support is insufficient")
    functions = pricing_functions
    started = time.perf_counter()
    summaries: dict[str, Any] = {}

    def bs_prices(frame: pd.DataFrame, parameters: np.ndarray) -> np.ndarray:
        return np.asarray([
            forward_black_price(row["forward"], row["strike"], row["T"], row["discount"], float(parameters[0]), row["option_type"])
            for row in frame.to_dict(orient="records")
        ])

    def fit_bs() -> None:
        observed = calibration["observed_price"].to_numpy(float)
        result = minimize_scalar(
            lambda sigma: float(np.mean((bs_prices(calibration, np.asarray([sigma])) - observed) ** 2)),
            bounds=(0.01, 2.0),
            method="bounded",
            options={"xatol": 1e-12},
        )
        parameter = np.asarray([float(result.x)])
        cal_prices, hold_prices = bs_prices(calibration, parameter), bs_prices(holdout, parameter)
        summary = {"parameters": {"sigma": float(parameter[0])}, "optimizer_success": bool(result.success)}
        summary.update({"calibration_" + key: value for key, value in _price_metrics(calibration, cal_prices).items()})
        summary.update({"holdout_" + key: value for key, value in _price_metrics(holdout, hold_prices).items()})
        summary.update({"holdout_" + key: value for key, value in _iv_metrics(holdout, hold_prices).items()})
        summary["objective"] = float(summary["calibration_price_rmse_dollar"] ** 2)
        summaries["BLACK_SCHOLES"] = summary

    def fit_stochastic(model: str) -> None:
        nonlocal functions
        dimension, count, seed = (5, 8, 20260912) if model == "STANDARD_HESTON" else (10, 12, 20260922)
        rng = np.random.default_rng(seed)
        starts = [np.zeros(dimension)] + [rng.normal(0.0, 1.25, dimension) for _ in range(count - 1)]
        hard_bounds = load_hard_safety_bounds(BOUNDS_PATH)
        if functions is None:
            if model == "STANDARD_HESTON":
                from scripts.run_ntpc_single_stock_pilot import price_heston_option

                def price_function(frame: pd.DataFrame, parameters: np.ndarray) -> np.ndarray:
                    return np.asarray([price_heston_option(pd.Series(row), parameters, node_count) for row in frame.to_dict(orient="records")])
            else:
                def price_function(frame: pd.DataFrame, parameters: np.ndarray) -> np.ndarray:
                    from scripts.run_ntpc_single_stock_pilot import _double_heston_row_price
                    return np.asarray([_double_heston_row_price(pd.Series(row), parameters, node_count) for row in frame.to_dict(orient="records")])
        else:
            price_function = functions[model]
        observed = calibration["observed_price"].to_numpy(float)

        def transform(raw: np.ndarray) -> np.ndarray:
            return _standard_heston_parameters(raw) if model == "STANDARD_HESTON" else unconstrained_to_parameters(raw, hard_bounds)

        start_rows: list[dict[str, Any]] = []
        for index, start in enumerate(starts):
            try:
                result = least_squares(
                    lambda raw: (price_function(calibration, transform(raw)) - observed) / np.maximum(observed, 1.0),
                    start,
                    method="trf",
                    ftol=1e-10,
                    xtol=1e-10,
                    gtol=1e-10,
                    diff_step=2e-5,
                    max_nfev=max_nfev,
                )
                parameters = transform(result.x)
                predicted_calibration = price_function(calibration, parameters)
                predicted_holdout = price_function(holdout, parameters)
                objective = float(np.mean(((predicted_calibration - observed) / np.maximum(observed, 1.0)) ** 2))
                entry = {
                    "start_index": index,
                    "valid": True,
                    "optimizer_success": bool(result.success),
                    "nfev": int(result.nfev),
                    "reached_cap": int(result.nfev) >= max_nfev,
                    "objective": objective,
                    "parameters": parameters.tolist(),
                }
            except Exception as exc:
                entry = {"start_index": index, "valid": False, "failure": f"{type(exc).__name__}: {exc}"}
            start_rows.append(entry)
        valid = [row for row in start_rows if row["valid"]]
        if not valid:
            summaries[model] = {"representative_fit_failed": True, "starts": start_rows}
            return
        representative = min(valid, key=lambda row: (row["objective"], row["start_index"]))
        parameters = np.asarray(representative["parameters"])
        cal_prices, hold_prices = price_function(calibration, parameters), price_function(holdout, parameters)
        names = ["kappa", "theta", "sigma", "rho", "v0"] if model == "STANDARD_HESTON" else PARAMETER_NAMES
        summary = {
            "parameters": dict(zip(names, map(float, parameters), strict=True)),
            "representative_start_index": representative["start_index"],
            "starts": start_rows,
        }
        summary.update({"calibration_" + key: value for key, value in _price_metrics(calibration, cal_prices).items()})
        summary.update({"holdout_" + key: value for key, value in _price_metrics(holdout, hold_prices).items()})
        summary.update({"holdout_" + key: value for key, value in _iv_metrics(holdout, hold_prices).items()})
        summary["objective"] = representative["objective"]
        summaries[model] = summary

    fit_bs()
    fit_stochastic("STANDARD_HESTON")
    fit_stochastic("DOUBLE_HESTON")
    runtime = time.perf_counter() - started
    return {
        "schema_version": "g8.pricing_family_surface_run/1",
        "data_classification": "SYNTHETIC_G8_PIPELINE_FIXTURE",
        "models": summaries,
        "runtime_seconds": runtime,
        "node_count": node_count,
        "max_nfev": max_nfev,
    }


def aggregate_pricing_family(runs: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the frozen unweighted surface-level winner rule."""
    model_names = ("BLACK_SCHOLES", "STANDARD_HESTON", "DOUBLE_HESTON")
    aggregates: dict[str, dict[str, Any]] = {}
    for model in model_names:
        price_values: list[float] = []
        normalized_values: list[float] = []
        iv_values: list[float] = []
        failures = 0
        total = 0
        for run in runs:
            total += 1
            result = run["models"][model]
            failed = bool(result.get("representative_fit_failed"))
            failures += int(failed)
            if not failed:
                price_values.append(float(result["holdout_price_rmse_dollar"]))
                normalized_values.append(float(result["holdout_price_rmse_normalized"]))
                if math.isfinite(float(result["holdout_iv_rmse_annualized"])):
                    iv_values.append(float(result["holdout_iv_rmse_annualized"]))
        aggregates[model] = {
            "eligible_surfaces": total,
            "failure_rate": failures / total if total else 1.0,
            "median_holdout_price_rmse_dollar": float(np.median(price_values)) if price_values else None,
            "median_holdout_price_rmse_normalized": float(np.median(normalized_values)) if normalized_values else None,
            "median_holdout_iv_rmse": float(np.median(iv_values)) if iv_values else None,
        }
    winner = "NO_CLEAR_PRICING_FAMILY_WINNER"
    complete = all(
        aggregates[model]["median_holdout_price_rmse_normalized"] is not None
        and aggregates[model]["median_holdout_iv_rmse"] is not None
        for model in model_names
    )
    if complete:
        best_failure_rate = min(item["failure_rate"] for item in aggregates.values())
        candidates = []
        for candidate_name, candidate in aggregates.items():
            price_clear = all(
                candidate["median_holdout_price_rmse_normalized"] * 0.95
                <= aggregates[other]["median_holdout_price_rmse_normalized"]
                for other in model_names if other != candidate_name
            )
            iv_clear = all(
                candidate["median_holdout_iv_rmse"] * 0.95
                <= aggregates[other]["median_holdout_iv_rmse"]
                for other in model_names if other != candidate_name
            )
            failure_ok = candidate["failure_rate"] <= best_failure_rate + 0.10 + 1e-15
            if price_clear and iv_clear and failure_ok:
                candidates.append(candidate_name)
        if len(candidates) == 1:
            winner = f"CLEAR_PRICING_FAMILY_PREFERENCE_{candidates[0]}"
    return {
        "schema_version": "g8.pricing_family_aggregation/1",
        "aggregates": aggregates,
        "winner_label": winner,
        "parameter_truth_claim": "NOT_APPLICABLE_PRICING_FAMILY",
    }


def real_g8_traditional_starts(
    hard_bounds: Mapping[str, tuple[float, float]],
    *,
    seed: int = 42,
) -> list[tuple[str, np.ndarray]]:
    """Only neutral midpoint and seeded broad start; never accepts truth labels."""
    rng = np.random.default_rng(seed)
    starts = [
        ("neutral_transform_midpoint", np.zeros(len(PARAMETER_NAMES))),
        ("deterministic_broad_start", rng.normal(0.0, 1.25, len(PARAMETER_NAMES))),
    ]
    if any(strategy == "disclosed_target_perturbation" for strategy, _ in starts):
        raise AssertionError("truth-informed traditional start leaked into G8 adapter")
    del hard_bounds
    return starts


@dataclass(frozen=True)
class TraditionalRunResult:
    representative: np.ndarray
    starts: list[dict[str, Any]]
    wall_seconds_all_starts: float


def calibrate_real_g8_traditional(
    surface: R2Surface,
    *,
    bounds_path: Path | str = BOUNDS_PATH,
    max_nfev: int = MAX_NFEV,
    node_count: int = NODE_COUNT,
    pricer: Callable[..., np.ndarray] | None = None,
) -> TraditionalRunResult:
    """Two-start real-market adapter; this path never receives a truth vector."""
    mask = np.asarray(surface.mask, dtype=bool)
    metadata_contracts = surface.metadata["selected_contracts"]
    contract_by_slot: dict[int, Mapping[str, Any]] = {}
    for index, key in enumerate(surface.slot_keys):
        matches = [row for row in metadata_contracts if int(row["rank"]) == key.expiry_rank and float(row["target"]) == key.target_log_moneyness and str(row["option_type"]) == key.option_type]
        if mask[index]:
            if len(matches) != 1:
                raise ValueError("valid slot lacks one selected contract identity")
            contract_by_slot[index] = matches[0]
    strikes = np.asarray([contract_by_slot[index]["strike"] for index in range(20) if mask[index]])
    maturities = np.asarray(surface.maturities)[mask]
    rates = np.asarray(surface.rates)[mask]
    carries = np.asarray(surface.carries)[mask]
    option_types = [key.option_type for key, valid in zip(surface.slot_keys, surface.mask, strict=True) if valid]
    observed = np.asarray(surface.prices)[mask] * surface.spot
    hard_bounds = load_hard_safety_bounds(bounds_path)
    def residual(raw: np.ndarray) -> np.ndarray:
        parameters = unconstrained_to_parameters(raw, hard_bounds)
        if pricer is None:
            predicted = np.asarray([
                price_double_heston_option(
                    surface.spot,
                    float(strikes[index]),
                    float(maturities[index]),
                    float(rates[index]),
                    float(carries[index]),
                    option_types[index],
                    parameters,
                    node_count=node_count,
                )
                for index in range(len(observed))
            ])
        else:
            predicted = np.asarray(
                pricer(
                    surface.spot,
                    strikes,
                    maturities,
                    rates,
                    carries,
                    option_types,
                    parameters,
                    node_count=node_count,
                )
            )
        return (predicted - observed) / np.maximum(observed, 1.0)

    starts = real_g8_traditional_starts(hard_bounds, seed=42)
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, (_, initial_raw) in enumerate(starts):
        try:
            result = least_squares(
                residual,
                initial_raw,
                method="trf",
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
                diff_step=2e-5,
                max_nfev=max_nfev,
            )
            parameters = unconstrained_to_parameters(result.x, hard_bounds)
            objective = float(np.mean(residual(result.x) ** 2))
            records.append(
                {
                    "start_index": index,
                    "start_strategy": starts[index][0],
                    "success": bool(result.success),
                    "nfev": int(result.nfev),
                    "reached_cap": int(result.nfev) >= max_nfev,
                    "objective": objective,
                    "parameters": parameters.tolist(),
                }
            )
        except Exception as exc:
            records.append(
                {
                    "start_index": index,
                    "start_strategy": starts[index][0],
                    "success": False,
                    "failure": f"{type(exc).__name__}: {exc}",
                }
            )
    wall = time.perf_counter() - started
    valid = [record for record in records if "objective" in record and math.isfinite(record["objective"])]
    if not valid:
        raise RuntimeError("all real-G8 traditional starts failed; failures are retained")
    chosen = min(valid, key=lambda record: (record["objective"], record["start_index"]))
    return TraditionalRunResult(
        representative=np.asarray(chosen["parameters"], dtype=float),
        starts=records,
        wall_seconds_all_starts=wall,
    )


class NeuralCheckpointAdapter:
    """Identity-first inference-only adapter for Model1/Model2 frozen seeds."""

    def __init__(
        self,
        *,
        method: str,
        seed: int,
        checkpoint_path: Path | str,
        expected_sha256: str,
        recorded_git_sha: str,
        loader: Callable[[Path], Mapping[str, Any]] | None = None,
        model_factory: Callable[[str], torch.nn.Module] | None = None,
    ) -> None:
        self.method = method
        self.seed = seed
        path = Path(checkpoint_path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_sha256.lower():
            raise ValueError(f"checkpoint SHA mismatch for {path}")
        payload = dict(loader(path) if loader is not None else torch.load(path, map_location="cpu", weights_only=False))
        if payload.get("git_sha") != recorded_git_sha or int(payload.get("seed", -1)) != seed:
            raise ValueError("checkpoint registry identity mismatch")
        if payload.get("run_kind") != "RESEARCH":
            raise ValueError("only RESEARCH checkpoints may enter G8 inference")
        if method not in {"MODEL1_ANN", "MODEL2_CONSTRAINT_REPRICING_INFORMED"}:
            raise ValueError("unsupported neural method")
        if model_factory is not None:
            model = model_factory(method)
        elif method == "MODEL1_ANN":
            from ..r2_primary.training import build_model1
            model = build_model1()
        else:
            from ..r2_primary.training import build_model2
            model = build_model2()
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        self.model = model
        self.standardizer = None
        if method == "MODEL1_ANN":
            standardizer_state = payload["target_standardizer"]
            standardizer = TargetStandardizer()
            standardizer.mean = standardizer_state["mean"]
            standardizer.scale = standardizer_state["scale"]
            self.standardizer = standardizer

    def predict(self, surface: R2Surface) -> np.ndarray:
        feature = build_g8_neural_features(surface)
        with torch.no_grad():
            output = self.model(torch.as_tensor(feature).unsqueeze(0))
            if self.method == "MODEL1_ANN":
                assert self.standardizer is not None
                output = self.standardizer.inverse_transform(output)
        return output.detach().cpu().numpy()[0]


def build_g8_neural_features(surface: R2Surface) -> np.ndarray:
    """Build the frozen shared 100-feature vector without synthetic-only gates."""
    from ..r2_primary.dataset import build_r2_features

    return build_r2_features(payload_to_surface(surface_to_payload(surface)))
