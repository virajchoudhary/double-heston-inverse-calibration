"""Run the bounded, auditable NTPC single-stock calibration pilot.

The pilot uses only the 15 July 2026 official NSE option/futures/spot panel,
official NSE CM history through the completed near expiry, and an official RBI
91-day Treasury-bill observation.  Future realized returns are never used in
calibration and unavailable future expiry horizons fail closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import brentq, least_squares, linear_sum_assignment, minimize_scalar
from scipy.special import expit, ndtr

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.calibrate_double_heston import (
    boundary_diagnostics,
    load_hard_safety_bounds,
    unconstrained_to_parameters,
)
from src.constants import PARAMETER_NAMES
from src.constraints import validate_parameters
from src.double_heston import (
    _gauss_laguerre_rule,
    heston_log_characteristic_exponent,
    price_double_heston_option,
)
from src.nse_stage_a import acquire_udiff_archive, read_udiff_csv


RAW_ROOT = REPOSITORY_ROOT / "market_data_audit" / "stage_a" / "raw" / "nse"
OUTPUT_ROOT = (
    REPOSITORY_ROOT
    / "market_data_audit"
    / "stage_a"
    / "derived"
    / "ntpc_single_stock_pilot"
)
FIGURE_ROOT = OUTPUT_ROOT / "figures"
REPORT_PATH = REPOSITORY_ROOT / "docs" / "NTPC_SINGLE_STOCK_CALIBRATION.md"
MENTOR_PATH = REPOSITORY_ROOT / "docs" / "NTPC_MENTOR_CHECKPOINT.md"
MANIFEST_PATH = REPOSITORY_ROOT / "docs" / "evidence" / "NTPC_SINGLE_STOCK_PILOT_MANIFEST.json"
BOUNDS_PATH = REPOSITORY_ROOT / "configs" / "parameter_bounds_PROVISIONAL.yaml"

VALUATION_DATE = date(2026, 7, 15)
AS_OF_DATE = date(2026, 8, 12)
SYMBOL = "NTPC"
SPOT = 344.35
EXPIRIES = (date(2026, 7, 28), date(2026, 8, 25), date(2026, 9, 29))
PRIMARY_EXPIRIES = EXPIRIES[:2]
TARGET_MONEYNESS = (-0.10, -0.05, 0.00, 0.05, 0.10)
CALIBRATION_TARGETS = (-0.05, 0.00, 0.05)
HOLDOUT_TARGETS = (-0.10, 0.10)
MONEYNESS_LIMIT = 0.10
YEAR_BASIS = 365.0
TRADING_DAYS_PER_YEAR = 252.0
PRIMARY_PRICE_FIELD = "ClsPric"
RISK_FREE_SIMPLE_YIELD = 0.053324
RISK_FREE_SOURCE_URL = "https://www.rbi.org.in/scripts/BS_PressReleaseDisplay.aspx?prid=63150"
RISK_FREE_SOURCE_DESCRIPTION = (
    "Reserve Bank of India Press Release 2026-2027/672 dated 15 July 2026: "
    "91-day T-bill auction cut-off price 98.6880 and YTM 5.3324%"
)
RISK_FREE_OBSERVATION_PATH = OUTPUT_ROOT / "raw" / "rbi_91_day_tbill_observation_20260715.json"
RISK_FREE_HTTP_RESPONSE_PATH = OUTPUT_ROOT / "raw" / "rbi_tbill_full_auction_result_20260715.html"
CORPORATE_ACTIONS_URL = (
    "https://www.nseindia.com/api/corporates-corporateActions?index=equities"
    "&from_date=15-07-2026&to_date=28-07-2026&symbol=NTPC"
)
HISTORY_DATES = tuple(
    date(2026, 7, day) for day in (15, 16, 17, 20, 21, 22, 23, 24, 27, 28)
)
ANALYSIS_SEED = 20260812
HESTON_STARTS = 8
DOUBLE_HESTON_STARTS = 12
NODE_COUNT = 64
MAX_NFEV = 160
WINNER_MARGIN = 0.05

OPTION_COLUMNS = [
    "valuation_date",
    "expiry_date",
    "DTE",
    "T",
    "option_type",
    "strike",
    "spot",
    "log_moneyness",
    "observed_price",
    "settlement_price",
    "last_price",
    "traded_volume",
    "volume_active",
    "open_interest",
    "open_interest_active",
    "trade_count",
    "trade_count_active",
    "activity_eligible",
    "matched_futures_price",
    "risk_free_simple_yield",
    "discount_factor",
    "continuous_rate",
    "futures_implied_carry",
    "market_implied_volatility",
    "target_log_moneyness",
    "sample_role",
    "included",
    "inclusion_exclusion_reason",
    "source_filename",
    "source_sha256",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def write_json(path: Path, payload: Any) -> None:
    atomic_write(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_write(path, frame.to_csv(index=False, lineterminator="\n").encode("utf-8"))


def dte_and_time(valuation_date: date, expiry_date: date) -> tuple[int, float]:
    dte = (expiry_date - valuation_date).days
    if dte <= 0:
        raise ValueError("expiry must be after valuation date")
    return dte, dte / YEAR_BASIS


def log_moneyness(strike: float, spot: float) -> float:
    if not np.isfinite([strike, spot]).all() or strike <= 0.0 or spot <= 0.0:
        raise ValueError("strike and spot must be finite and positive")
    return float(math.log(strike / spot))


def discount_from_simple_yield(simple_yield: float, maturity: float) -> float:
    if simple_yield < 0.0 or maturity <= 0.0:
        raise ValueError("yield must be non-negative and maturity positive")
    return float(1.0 / (1.0 + simple_yield * maturity))


def carry_from_forward(spot: float, forward: float, discount: float, maturity: float) -> tuple[float, float]:
    if min(spot, forward, discount, maturity) <= 0.0 or discount > 1.0:
        raise ValueError("invalid forward/discount inputs")
    rate = -math.log(discount) / maturity
    carry = rate - math.log(forward / spot) / maturity
    return float(rate), float(carry)


def forward_black_price(
    forward: float,
    strike: float,
    maturity: float,
    discount: float,
    sigma: float,
    option_type: str,
) -> float:
    if min(forward, strike, maturity, discount, sigma) <= 0.0:
        raise ValueError("forward Black inputs must be positive")
    root_t = math.sqrt(maturity)
    d1 = (math.log(forward / strike) + 0.5 * sigma * sigma * maturity) / (sigma * root_t)
    d2 = d1 - sigma * root_t
    if option_type == "call":
        return float(discount * (forward * ndtr(d1) - strike * ndtr(d2)))
    if option_type == "put":
        return float(discount * (strike * ndtr(-d2) - forward * ndtr(-d1)))
    raise ValueError("option_type must be call or put")


def no_arbitrage_bounds(forward: float, strike: float, discount: float, option_type: str) -> tuple[float, float]:
    if option_type == "call":
        return discount * max(forward - strike, 0.0), discount * forward
    if option_type == "put":
        return discount * max(strike - forward, 0.0), discount * strike
    raise ValueError("option_type must be call or put")


def implied_volatility(
    price: float,
    forward: float,
    strike: float,
    maturity: float,
    discount: float,
    option_type: str,
) -> float:
    lower, upper = no_arbitrage_bounds(forward, strike, discount, option_type)
    tolerance = 1e-10 * max(1.0, upper)
    if not np.isfinite(price) or price < lower - tolerance or price > upper + tolerance:
        raise ValueError(f"price outside no-arbitrage bounds [{lower}, {upper}]")
    intrinsic_limit = forward_black_price(forward, strike, maturity, discount, 1e-8, option_type)
    if abs(price - intrinsic_limit) <= tolerance:
        return 1e-8

    def objective(sigma: float) -> float:
        return forward_black_price(forward, strike, maturity, discount, sigma, option_type) - price

    low, high = 1e-8, 5.0
    if objective(low) * objective(high) > 0.0:
        raise ValueError("implied-volatility root is not bracketed on [1e-8, 5]")
    return float(brentq(objective, low, high, xtol=1e-12, rtol=1e-12, maxiter=200))


def realized_volatility(closes: Sequence[float]) -> tuple[float, int, float]:
    values = np.asarray(closes, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("at least two finite positive closes are required")
    returns = np.diff(np.log(values))
    n = len(returns)
    rv = float(np.sum(returns * returns))
    return float(math.sqrt(TRADING_DAYS_PER_YEAR / n * rv)), n, rv


def expected_average_variance(kappa: float, theta: float, v0: float, maturity: float) -> float:
    if min(kappa, theta, v0, maturity) <= 0.0:
        raise ValueError("variance inputs must be positive")
    return float(theta + (v0 - theta) * (-math.expm1(-kappa * maturity)) / (kappa * maturity))


def half_life(kappa: float) -> tuple[float, float]:
    if kappa <= 0.0:
        raise ValueError("kappa must be positive")
    years = math.log(2.0) / kappa
    return float(years), float(365.0 * years)


def _option_type(raw: str) -> str:
    return {"CE": "call", "PE": "put"}[raw]


def _fo_paths() -> tuple[Path, Path]:
    directory = RAW_ROOT / VALUATION_DATE.isoformat()
    base = "BhavCopy_NSE_FO_0_0_0_20260715_F_0000.csv"
    return directory / base, directory / f"{base}.zip"


def _acquire_corporate_actions() -> tuple[list[dict[str, Any]], Path]:
    path = OUTPUT_ROOT / "raw" / "ntpc_corporate_actions_20260715_20260728.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("corporate-action evidence must be a JSON list")
        return payload, path
    request = urllib.request.Request(
        CORPORATE_ACTIONS_URL,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        raw = response.read()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, list):
        raise ValueError("unexpected corporate-action response")
    atomic_write(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return payload, path


def _futures_contract(fo: pd.DataFrame) -> pd.DataFrame:
    futures = fo.loc[(fo["TckrSymb"] == SYMBOL) & (fo["FinInstrmTp"] == "STF")].copy()
    futures["expiry_date"] = pd.to_datetime(futures["FininstrmActlXpryDt"]).dt.date
    rows: list[dict[str, Any]] = []
    for expiry in EXPIRIES:
        selected = futures.loc[futures["expiry_date"] == expiry]
        if len(selected) != 1:
            raise ValueError(f"expected one NTPC future for {expiry}")
        row = selected.iloc[0]
        dte, maturity = dte_and_time(VALUATION_DATE, expiry)
        forward = float(row["ClsPric"])
        if not (
            forward > 0.0
            and float(row["TtlTradgVol"]) > 0.0
            and float(row["OpnIntrst"]) > 0.0
            and float(row["TtlNbOfTxsExctd"]) > 0.0
        ):
            raise ValueError(f"inactive matched future for {expiry}")
        discount = discount_from_simple_yield(RISK_FREE_SIMPLE_YIELD, maturity)
        rate, carry = carry_from_forward(SPOT, forward, discount, maturity)
        rows.append(
            {
                "expiry_date": expiry.isoformat(),
                "DTE": dte,
                "T": maturity,
                "futures_close": forward,
                "futures_last": float(row["LastPric"]),
                "futures_settlement": float(row["SttlmPric"]),
                "futures_volume": float(row["TtlTradgVol"]),
                "futures_open_interest": float(row["OpnIntrst"]),
                "futures_trade_count": float(row["TtlNbOfTxsExctd"]),
                "risk_free_simple_yield": RISK_FREE_SIMPLE_YIELD,
                "discount_factor": discount,
                "continuous_rate": rate,
                "futures_implied_carry": carry,
            }
        )
    return pd.DataFrame(rows)


def _unique_target_assignment(group: pd.DataFrame) -> dict[int, float]:
    eligible = group.loc[group["activity_eligible"] & group["primary_expiry"] & group["within_moneyness"]].copy()
    eligible = eligible.sort_values(["strike", "FinInstrmId"]).drop_duplicates("strike", keep="first")
    candidates = eligible["log_moneyness"].to_numpy(float)
    real_costs = np.abs(np.asarray(TARGET_MONEYNESS)[:, None] - candidates[None, :])
    real_costs += 1e-12 * np.arange(len(candidates))[None, :]
    real_costs[real_costs > 0.05 + 1e-12] = 1e6
    dummy_costs = np.full((len(TARGET_MONEYNESS), len(TARGET_MONEYNESS)), 1e6)
    np.fill_diagonal(dummy_costs, 0.0500000001)
    costs = np.concatenate([real_costs, dummy_costs], axis=1)
    target_index, candidate_index = linear_sum_assignment(costs)
    assignment: dict[int, float] = {}
    for target_i, candidate_i in zip(target_index, candidate_index, strict=True):
        if candidate_i >= len(eligible):
            continue
        if costs[target_i, candidate_i] > 0.05 + 1e-12:
            raise RuntimeError("invalid target assignment escaped the fail-closed gate")
        matched_index = int(eligible.index[candidate_i])
        assignment[matched_index] = float(TARGET_MONEYNESS[target_i])
    return assignment


def build_option_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fo_csv, _ = _fo_paths()
    fo = read_udiff_csv(fo_csv, VALUATION_DATE, "FO")
    futures = _futures_contract(fo)
    futures_lookup = futures.set_index("expiry_date")
    options = fo.loc[(fo["TckrSymb"] == SYMBOL) & (fo["FinInstrmTp"] == "STO")].copy()
    if len(options) != 146:
        raise ValueError(f"expected 146 NTPC option rows, got {len(options)}")
    options["expiry_date_obj"] = pd.to_datetime(options["FininstrmActlXpryDt"]).dt.date
    options["expiry_date"] = options["expiry_date_obj"].map(date.isoformat)
    options["DTE"] = options["expiry_date_obj"].map(lambda expiry: (expiry - VALUATION_DATE).days)
    options["T"] = options["DTE"] / YEAR_BASIS
    options["option_type"] = options["OptnTp"].map(_option_type)
    options["strike"] = pd.to_numeric(options["StrkPric"], errors="raise")
    options["spot"] = SPOT
    options["log_moneyness"] = np.log(options["strike"] / SPOT)
    for target, source in (
        ("observed_price", PRIMARY_PRICE_FIELD),
        ("settlement_price", "SttlmPric"),
        ("last_price", "LastPric"),
        ("traded_volume", "TtlTradgVol"),
        ("open_interest", "OpnIntrst"),
        ("trade_count", "TtlNbOfTxsExctd"),
    ):
        options[target] = pd.to_numeric(options[source], errors="coerce")
    options["volume_active"] = options["traded_volume"] > 0.0
    options["open_interest_active"] = options["open_interest"] > 0.0
    options["trade_count_active"] = options["trade_count"] > 0.0
    options["activity_eligible"] = (
        options["observed_price"].gt(0.0)
        & options["volume_active"]
        & options["open_interest_active"]
        & options["trade_count_active"]
    )
    options["primary_expiry"] = options["expiry_date_obj"].isin(PRIMARY_EXPIRIES)
    options["within_moneyness"] = options["log_moneyness"].abs() <= MONEYNESS_LIMIT + 1e-12
    options["target_log_moneyness"] = np.nan
    for (_, _), group in options.groupby(["expiry_date", "option_type"], sort=True):
        if bool(group["primary_expiry"].iloc[0]):
            for index, target in _unique_target_assignment(group).items():
                options.loc[index, "target_log_moneyness"] = target
    options["sample_role"] = "EXCLUDED"
    options.loc[options["target_log_moneyness"].isin(CALIBRATION_TARGETS), "sample_role"] = "CALIBRATION"
    options.loc[options["target_log_moneyness"].isin(HOLDOUT_TARGETS), "sample_role"] = "HOLDOUT"
    options["included"] = options["sample_role"].isin(["CALIBRATION", "HOLDOUT"])

    def reason(row: pd.Series) -> str:
        if row["included"]:
            return f"INCLUDED_{row['sample_role']}_NEAREST_UNIQUE_TARGET"
        if not row["primary_expiry"]:
            return "EXCLUDED_FAR_EXPIRY_NO_TRADED_OPTION_ROWS_DIAGNOSTIC_ONLY"
        if not row["within_moneyness"]:
            return "EXCLUDED_OUTSIDE_ABS_LOG_MONEYNESS_0_10"
        if not row["activity_eligible"]:
            failures = []
            if not row["volume_active"]:
                failures.append("ZERO_VOLUME")
            if not row["trade_count_active"]:
                failures.append("ZERO_TRADES")
            if not row["open_interest_active"]:
                failures.append("ZERO_OPEN_INTEREST")
            if not (np.isfinite(row["observed_price"]) and row["observed_price"] > 0.0):
                failures.append("INVALID_CLOSE")
            return "EXCLUDED_ACTIVITY_" + "_".join(failures)
        return "EXCLUDED_NOT_NEAREST_UNIQUE_PREDECLARED_TARGET"

    options["inclusion_exclusion_reason"] = options.apply(reason, axis=1)
    options["matched_futures_price"] = options["expiry_date"].map(futures_lookup["futures_close"])
    options["risk_free_simple_yield"] = RISK_FREE_SIMPLE_YIELD
    options["discount_factor"] = options["expiry_date"].map(futures_lookup["discount_factor"])
    options["continuous_rate"] = options["expiry_date"].map(futures_lookup["continuous_rate"])
    options["futures_implied_carry"] = options["expiry_date"].map(futures_lookup["futures_implied_carry"])
    options["market_implied_volatility"] = np.nan
    for index, row in options.loc[options["included"]].iterrows():
        options.loc[index, "market_implied_volatility"] = implied_volatility(
            float(row["observed_price"]),
            float(row["matched_futures_price"]),
            float(row["strike"]),
            float(row["T"]),
            float(row["discount_factor"]),
            str(row["option_type"]),
        )
    options["valuation_date"] = VALUATION_DATE.isoformat()
    options["source_filename"] = fo_csv.name
    options["source_sha256"] = sha256(fo_csv)
    selected = options.loc[options["included"]].copy().sort_values(
        ["sample_role", "expiry_date", "option_type", "target_log_moneyness"]
    )
    if len(selected) != 19 or (selected["sample_role"] == "CALIBRATION").sum() != 12 or (selected["sample_role"] == "HOLDOUT").sum() != 7:
        raise ValueError("fail-closed 12-calibration/7-holdout alternative was not produced")
    if set(selected.loc[selected["sample_role"] == "CALIBRATION", "target_log_moneyness"]) != set(CALIBRATION_TARGETS):
        raise ValueError("calibration targets differ from predeclared inner targets")
    return options[OPTION_COLUMNS].copy(), selected[OPTION_COLUMNS].copy(), futures


def _rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.to_dict(orient="records")


def _price_rows(
    frame: pd.DataFrame,
    price_function: Callable[[pd.Series], float],
) -> np.ndarray:
    return np.asarray([price_function(row) for _, row in frame.iterrows()], dtype=np.float64)


def _metrics(frame: pd.DataFrame, predicted: np.ndarray) -> dict[str, float]:
    observed = frame["observed_price"].to_numpy(float)
    error = predicted - observed
    relative = np.abs(error) / np.maximum(observed, 1.0)
    predicted_iv: list[float] = []
    for (_, row), price in zip(frame.iterrows(), predicted, strict=True):
        predicted_iv.append(
            implied_volatility(
                float(price),
                float(row["matched_futures_price"]),
                float(row["strike"]),
                float(row["T"]),
                float(row["discount_factor"]),
                str(row["option_type"]),
            )
        )
    iv_error = np.asarray(predicted_iv) - frame["market_implied_volatility"].to_numpy(float)
    return {
        "price_rmse": float(np.sqrt(np.mean(error * error))),
        "price_mae": float(np.mean(np.abs(error))),
        "relative_price_error_mean": float(np.mean(relative)),
        "iv_rmse": float(np.sqrt(np.mean(iv_error * iv_error))),
    }


def fit_black_scholes(calibration: pd.DataFrame, holdout: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    started = time.perf_counter()

    def prices(frame: pd.DataFrame, sigma: float) -> np.ndarray:
        return _price_rows(
            frame,
            lambda row: forward_black_price(
                float(row["matched_futures_price"]),
                float(row["strike"]),
                float(row["T"]),
                float(row["discount_factor"]),
                sigma,
                str(row["option_type"]),
            ),
        )

    observed = calibration["observed_price"].to_numpy(float)
    result = minimize_scalar(
        lambda sigma: float(np.mean((prices(calibration, sigma) - observed) ** 2)),
        bounds=(0.01, 2.0),
        method="bounded",
        options={"xatol": 1e-12},
    )
    sigma = float(result.x)
    runtime = time.perf_counter() - started
    cal_prices, hold_prices = prices(calibration, sigma), prices(holdout, sigma)
    summary: dict[str, Any] = {
        "model": "BLACK_SCHOLES",
        "parameter_count": 1,
        "sigma": sigma,
        "optimizer_success": bool(result.success),
        "runtime_seconds": runtime,
    }
    summary.update({f"calibration_{key}": value for key, value in _metrics(calibration, cal_prices).items()})
    summary.update({f"holdout_{key}": value for key, value in _metrics(holdout, hold_prices).items()})
    predictions = pd.concat(
        [
            _prediction_frame(calibration, "BLACK_SCHOLES", cal_prices),
            _prediction_frame(holdout, "BLACK_SCHOLES", hold_prices),
        ],
        ignore_index=True,
    )
    return summary, predictions


def _heston_parameters(x: Sequence[float]) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    unit = expit(np.clip(values, -35.0, 35.0))
    kappa = 0.05 + unit[0] * (12.0 - 0.05)
    theta = 0.002 + unit[1] * (0.30 - 0.002)
    v0 = 0.002 + unit[2] * (0.35 - 0.002)
    sigma_upper = min(1.5, math.sqrt(2.0 * kappa * theta) * (1.0 - 1e-7))
    sigma = 0.005 + unit[3] * (sigma_upper - 0.005)
    rho = 0.95 * math.tanh(float(values[4]))
    return np.asarray([kappa, theta, sigma, rho, v0], dtype=float)


def price_heston_option(row: pd.Series, parameters: Sequence[float], node_count: int = NODE_COUNT) -> float:
    kappa, theta, sigma, rho, v0 = np.asarray(parameters, dtype=float)
    spot, strike, maturity = float(row["spot"]), float(row["strike"]), float(row["T"])
    rate, carry = float(row["continuous_rate"]), float(row["futures_implied_carry"])
    nodes, weights = _gauss_laguerre_rule(node_count)

    def characteristic(u: complex | np.ndarray) -> complex | np.ndarray:
        exponent = 1j * np.asarray(u) * (math.log(spot) + (rate - carry) * maturity)
        exponent += heston_log_characteristic_exponent(u, maturity, kappa, theta, sigma, rho, v0)
        return np.exp(exponent)

    phi_u = characteristic(nodes)
    phi_shifted = characteristic(nodes - 1j)
    phi_minus_i = characteristic(-1j)
    oscillation = np.exp(-1j * nodes * math.log(strike))
    inverse_iu = 1.0 / (1j * nodes)
    compensation = np.exp(nodes)
    p1 = 0.5 + np.sum(weights * compensation * np.real(oscillation * phi_shifted * inverse_iu / phi_minus_i)) / math.pi
    p2 = 0.5 + np.sum(weights * compensation * np.real(oscillation * phi_u * inverse_iu)) / math.pi
    call = spot * math.exp(-carry * maturity) * p1 - strike * math.exp(-rate * maturity) * p2
    if row["option_type"] == "call":
        result = call
    else:
        result = call - spot * math.exp(-carry * maturity) + strike * math.exp(-rate * maturity)
    if not np.isfinite(result):
        raise FloatingPointError("non-finite Heston price")
    return float(result)


def _double_heston_row_price(row: pd.Series, parameters: Sequence[float], node_count: int = NODE_COUNT) -> float:
    return price_double_heston_option(
        float(row["spot"]),
        float(row["strike"]),
        float(row["T"]),
        float(row["continuous_rate"]),
        float(row["futures_implied_carry"]),
        str(row["option_type"]),
        parameters,
        node_count=node_count,
    )


def _prediction_frame(frame: pd.DataFrame, model: str, predicted: np.ndarray) -> pd.DataFrame:
    result = frame[
        ["sample_role", "expiry_date", "option_type", "strike", "log_moneyness", "observed_price", "market_implied_volatility", "matched_futures_price", "T", "discount_factor"]
    ].copy()
    result["model"] = model
    result["predicted_price"] = predicted
    result["price_residual"] = predicted - result["observed_price"]
    result["predicted_implied_volatility"] = [
        implied_volatility(float(price), float(row.matched_futures_price), float(row.strike), float(row.T), float(row.discount_factor), str(row.option_type))
        for row, price in zip(result.itertuples(index=False), predicted, strict=True)
    ]
    return result


def fit_stochastic_model(
    calibration: pd.DataFrame,
    holdout: pd.DataFrame,
    model: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, np.ndarray]:
    if model not in {"HESTON", "DOUBLE_HESTON"}:
        raise ValueError("unknown stochastic model")
    rng = np.random.default_rng(ANALYSIS_SEED + (100 if model == "HESTON" else 200))
    start_count = HESTON_STARTS if model == "HESTON" else DOUBLE_HESTON_STARTS
    dimension = 5 if model == "HESTON" else 10
    starts = [np.zeros(dimension)] + [rng.normal(0.0, 1.25, dimension) for _ in range(start_count - 1)]
    hard_bounds = load_hard_safety_bounds(BOUNDS_PATH) if model == "DOUBLE_HESTON" else None
    observed = calibration["observed_price"].to_numpy(float)
    rows: list[dict[str, Any]] = []

    def transform(x: np.ndarray) -> np.ndarray:
        return _heston_parameters(x) if model == "HESTON" else unconstrained_to_parameters(x, hard_bounds)

    def price(frame: pd.DataFrame, parameters: np.ndarray) -> np.ndarray:
        if model == "HESTON":
            return _price_rows(frame, lambda row: price_heston_option(row, parameters))
        return _price_rows(frame, lambda row: _double_heston_row_price(row, parameters))

    def residual(x: np.ndarray) -> np.ndarray:
        return price(calibration, transform(x)) - observed

    for start_id, start in enumerate(starts):
        started = time.perf_counter()
        row: dict[str, Any] = {"model": model, "start_id": start_id}
        try:
            result = least_squares(
                residual,
                start,
                method="trf",
                max_nfev=MAX_NFEV,
                ftol=1e-9,
                xtol=1e-9,
                gtol=1e-9,
                diff_step=2e-5,
            )
            parameters = transform(result.x)
            cal_prices = price(calibration, parameters)
            hold_prices = price(holdout, parameters)
            row.update(
                {
                    "optimizer_success": bool(result.success),
                    "optimizer_status": int(result.status),
                    "optimizer_message": str(result.message),
                    "nfev": int(result.nfev),
                    "valid": True,
                    "boundary_reasons": ";".join(boundary_diagnostics(parameters, hard_bounds)) if model == "DOUBLE_HESTON" else _heston_boundary_reasons(parameters),
                }
            )
            row.update({f"calibration_{key}": value for key, value in _metrics(calibration, cal_prices).items()})
            row.update({f"holdout_{key}": value for key, value in _metrics(holdout, hold_prices).items()})
            names = ["kappa", "theta", "sigma", "rho", "v0"] if model == "HESTON" else list(PARAMETER_NAMES)
            row.update({name: float(value) for name, value in zip(names, parameters, strict=True)})
        except Exception as error:
            row.update(
                {
                    "optimizer_success": False,
                    "optimizer_status": -1,
                    "optimizer_message": f"{type(error).__name__}: {error}",
                    "nfev": 0,
                    "valid": False,
                    "boundary_reasons": "",
                    "calibration_price_rmse": np.nan,
                    "holdout_price_rmse": np.nan,
                }
            )
        row["runtime_seconds"] = time.perf_counter() - started
        rows.append(row)
    starts_frame = pd.DataFrame(rows)
    valid = starts_frame.loc[starts_frame["valid"]].copy()
    if valid.empty:
        raise RuntimeError(f"all {model} starts failed")
    best_row = valid.sort_values(["calibration_price_rmse", "start_id"]).iloc[0]
    names = ["kappa", "theta", "sigma", "rho", "v0"] if model == "HESTON" else list(PARAMETER_NAMES)
    best_parameters = best_row[names].to_numpy(float)
    cal_prices = price(calibration, best_parameters)
    hold_prices = price(holdout, best_parameters)
    summary = {
        "model": model,
        "parameter_count": len(names),
        "best_start_id": int(best_row["start_id"]),
        "optimizer_success": bool(best_row["optimizer_success"]),
        "runtime_seconds": float(starts_frame["runtime_seconds"].sum()),
        **{name: float(value) for name, value in zip(names, best_parameters, strict=True)},
        **{f"calibration_{key}": value for key, value in _metrics(calibration, cal_prices).items()},
        **{f"holdout_{key}": value for key, value in _metrics(holdout, hold_prices).items()},
    }
    predictions = pd.concat(
        [
            _prediction_frame(calibration, model, cal_prices),
            _prediction_frame(holdout, model, hold_prices),
        ],
        ignore_index=True,
    )
    return summary, predictions, starts_frame, best_parameters


def _heston_boundary_reasons(parameters: Sequence[float]) -> str:
    kappa, theta, sigma, rho, v0 = np.asarray(parameters, dtype=float)
    reasons = []
    if sigma * sigma / (2.0 * kappa * theta) >= 0.98:
        reasons.append("feller:near_boundary")
    if abs(rho) >= 0.93:
        reasons.append("rho:near_boundary")
    for name, value, lower, upper in (
        ("kappa", kappa, 0.05, 12.0),
        ("theta", theta, 0.002, 0.30),
        ("sigma", sigma, 0.005, 1.5),
        ("v0", v0, 0.002, 0.35),
    ):
        fraction = (value - lower) / (upper - lower)
        if fraction <= 0.02 or fraction >= 0.98:
            reasons.append(f"{name}:near_hard_bound")
    return ";".join(reasons)


def model_winner(comparison: pd.DataFrame) -> str:
    ordered = comparison.sort_values(["holdout_price_rmse", "holdout_iv_rmse", "parameter_count"])
    best, second = ordered.iloc[0], ordered.iloc[1]
    price_clear = best["holdout_price_rmse"] <= (1.0 - WINNER_MARGIN) * second["holdout_price_rmse"]
    iv_not_worse = best["holdout_iv_rmse"] <= second["holdout_iv_rmse"] * (1.0 + 1e-12)
    if price_clear and iv_not_worse:
        return {"BLACK_SCHOLES": "BS_BEST", "HESTON": "HESTON_BEST", "DOUBLE_HESTON": "DOUBLE_HESTON_BEST"}[best["model"]]
    return "NO_CLEAR_WINNER"


def parameter_stability(starts: pd.DataFrame, best: np.ndarray) -> dict[str, Any]:
    valid = starts.loc[starts["valid"]].copy()
    best_rmse = float(valid["calibration_price_rmse"].min())
    threshold = max(best_rmse * 1.05, best_rmse + 0.01)
    near = valid.loc[valid["calibration_price_rmse"] <= threshold].copy()
    bounds = load_hard_safety_bounds(BOUNDS_PATH)
    widths = np.asarray([bounds[name][1] - bounds[name][0] for name in PARAMETER_NAMES])
    distances = []
    for _, row in near.iterrows():
        candidate = row[list(PARAMETER_NAMES)].to_numpy(float)
        distances.append(float(np.sqrt(np.mean(((candidate - best) / widths) ** 2))))
    materially_displaced = sum(value >= 0.05 for value in distances)
    if len(near) < 2:
        classification = "UNRESOLVED_SINGLE_NEAR_EQUIVALENT_START"
    elif materially_displaced:
        classification = "UNSTABLE_MATERIAL_MULTI_START_DISPLACEMENT"
    else:
        classification = "LOCALLY_STABLE_WITHIN_DECLARED_STARTS"
    return {
        "classification": classification,
        "near_equivalent_threshold_price_rmse": threshold,
        "near_equivalent_start_count": int(len(near)),
        "materially_displaced_start_count": int(materially_displaced),
        "maximum_range_scaled_distance_from_best": max(distances) if distances else 0.0,
    }


def build_realized_history() -> tuple[pd.DataFrame, dict[str, Any], Path]:
    records = [acquire_udiff_archive("CM", value) for value in HISTORY_DATES]
    rows = []
    for value, record in zip(HISTORY_DATES, records, strict=True):
        frame = read_udiff_csv(record.csv_path, value, "CM")
        selected = frame.loc[(frame["TckrSymb"] == SYMBOL) & (frame["SctySrs"] == "EQ")]
        if len(selected) != 1:
            raise ValueError(f"expected one NTPC EQ row on {value}")
        row = selected.iloc[0]
        close = float(row["ClsPric"])
        if close <= 0.0:
            raise ValueError(f"invalid NTPC close on {value}")
        rows.append(
            {
                "trading_date": value.isoformat(),
                "close": close,
                "source_filename": record.csv_path.name,
                "archive_sha256": record.archive_sha256.upper(),
                "csv_sha256": record.csv_sha256.upper(),
                "official_url": record.official_url,
            }
        )
    history = pd.DataFrame(rows)
    if tuple(pd.to_datetime(history["trading_date"]).dt.date) != HISTORY_DATES:
        raise ValueError("trading-day completeness contract failed")
    corporate_actions, action_path = _acquire_corporate_actions()
    if corporate_actions:
        raise ValueError("price-adjusting corporate action requires an explicit adjustment contract")
    annualized, count, rv = realized_volatility(history["close"])
    status = {
        "corporate_action_count": 0,
        "corporate_action_status": "NO_NTPC_ACTIONS_RETURNED_BY_OFFICIAL_NSE_API_2026-07-15_TO_2026-07-28",
        "adjustment_rule": "UNADJUSTED_OFFICIAL_EQ_CLOSE_ALLOWED_BECAUSE_NO_ACTIONS_IN_COMPLETED_NEAR_WINDOW",
        "near_expiry": EXPIRIES[0].isoformat(),
        "near_return_count": count,
        "near_realized_variance": rv,
        "near_annualized_realized_volatility": annualized,
        "middle_status": "UNAVAILABLE_EXPIRY_AFTER_AS_OF_DATE",
        "far_status": "UNAVAILABLE_EXPIRY_AFTER_AS_OF_DATE",
        "as_of_date": AS_OF_DATE.isoformat(),
    }
    return history, status, action_path


def volatility_comparison(
    selected: pd.DataFrame,
    bs: dict[str, Any],
    heston: np.ndarray,
    double_heston: np.ndarray,
    realized_status: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    for expiry in EXPIRIES:
        expiry_text = expiry.isoformat()
        dte, maturity = dte_and_time(VALUATION_DATE, expiry)
        expiry_rows = selected.loc[selected["expiry_date"] == expiry_text]
        market_atm_iv = float(
            expiry_rows.loc[expiry_rows["target_log_moneyness"] == 0.0, "market_implied_volatility"].mean()
        ) if not expiry_rows.empty else np.nan
        heston_vol = math.sqrt(expected_average_variance(heston[0], heston[1], heston[4], maturity))
        dh_vol = math.sqrt(
            expected_average_variance(double_heston[0], double_heston[1], double_heston[4], maturity)
            + expected_average_variance(double_heston[5], double_heston[6], double_heston[9], maturity)
        )
        actual = realized_status["near_annualized_realized_volatility"] if expiry == EXPIRIES[0] else np.nan
        rows.append(
            {
                "expiry_date": expiry_text,
                "DTE": dte,
                "T": maturity,
                "market_atm_iv": market_atm_iv,
                "bs_predicted_volatility": bs["sigma"],
                "heston_predicted_average_volatility": heston_vol,
                "double_heston_predicted_average_volatility": dh_vol,
                "actual_ex_post_realized_volatility": actual,
                "actual_status": "COMPLETE" if expiry == EXPIRIES[0] else "UNAVAILABLE_FUTURE_EXPIRY_AS_OF_2026-08-12",
            }
        )
    frame = pd.DataFrame(rows)
    for model in ("bs", "heston", "double_heston"):
        source = "bs_predicted_volatility" if model == "bs" else f"{model}_predicted_average_volatility"
        frame[f"{model}_absolute_realized_volatility_error"] = (
            frame[source] - frame["actual_ex_post_realized_volatility"]
        ).abs()
    return frame


def _figures(
    all_options: pd.DataFrame,
    selected: pd.DataFrame,
    predictions: pd.DataFrame,
    comparison: pd.DataFrame,
    heston: np.ndarray,
    double_heston: np.ndarray,
    starts: pd.DataFrame,
    volatility: pd.DataFrame,
) -> list[Path]:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    def save(name: str) -> None:
        path = FIGURE_ROOT / name
        plt.tight_layout()
        plt.savefig(path, dpi=160, bbox_inches="tight")
        plt.close()
        paths.append(path)

    plt.figure(figsize=(9, 5))
    role_style = {
        "EXCLUDED": ("lightgray", "x"),
        "CALIBRATION": ("tab:blue", "o"),
        "HOLDOUT": ("tab:orange", "^"),
    }
    for role, (color, marker) in role_style.items():
        group = all_options.loc[all_options["sample_role"] == role]
        plt.scatter(group["strike"], group["expiry_date"], c=color, marker=marker, s=24, label=role)
    plt.xlabel("Strike (INR)"); plt.ylabel("Actual expiry date"); plt.title("NTPC strike support: excluded, calibration, and holdout — valuation 2026-07-15"); plt.legend()
    save("01_strike_support.png")

    plt.figure(figsize=(9, 5))
    for (expiry, role), group in selected.groupby(["expiry_date", "sample_role"]):
        marker = "o" if role == "CALIBRATION" else "^"
        plt.scatter(group["log_moneyness"], group["market_implied_volatility"], marker=marker, label=f"{expiry} {role}")
    plt.xlabel("log(K/S)"); plt.ylabel("Market implied volatility (annualized)"); plt.title("NTPC market IV smile — valuation 2026-07-15"); plt.legend()
    save("02_market_iv_smile.png")

    plt.figure(figsize=(8, 6))
    for (model, role), group in predictions.groupby(["model", "sample_role"]):
        marker = "o" if role == "CALIBRATION" else "^"
        plt.scatter(group["observed_price"], group["predicted_price"], marker=marker, label=f"{model} {role}", alpha=0.75)
    limit = float(max(predictions["observed_price"].max(), predictions["predicted_price"].max()))
    plt.plot([0, limit], [0, limit], "k--"); plt.xlabel("Observed close price (INR)"); plt.ylabel("Predicted price (INR)"); plt.title("NTPC observed vs predicted option prices — 2026-07-15"); plt.legend()
    save("03_market_vs_predicted_price.png")

    plt.figure(figsize=(9, 5))
    for (model, role), group in predictions.groupby(["model", "sample_role"]):
        marker = "o" if role == "CALIBRATION" else "^"
        plt.scatter(group["log_moneyness"], group["price_residual"], marker=marker, label=f"{model} {role}", alpha=0.75)
    plt.axhline(0, color="black", linewidth=1); plt.xlabel("log(K/S)"); plt.ylabel("Price residual (INR)"); plt.title("NTPC pricing residuals — valuation 2026-07-15"); plt.legend()
    save("04_residual_vs_moneyness.png")

    plt.figure(figsize=(9, 5))
    for (model, role), group in predictions.groupby(["model", "sample_role"]):
        marker = "o" if role == "CALIBRATION" else "^"
        plt.scatter(group["market_implied_volatility"], group["predicted_implied_volatility"], marker=marker, label=f"{model} {role}", alpha=0.75)
    low = float(predictions["market_implied_volatility"].min()); high = float(predictions["market_implied_volatility"].max())
    plt.plot([low, high], [low, high], "k--"); plt.xlabel("Market IV (annualized)"); plt.ylabel("Model-implied IV (annualized)"); plt.title("NTPC market IV vs model-implied IV — 2026-07-15"); plt.legend()
    save("05_market_iv_vs_model_iv.png")

    middle_t = dte_and_time(VALUATION_DATE, PRIMARY_EXPIRIES[1])[1]
    times = np.linspace(0.0, middle_t, 120)
    slow = double_heston[1] + (double_heston[4] - double_heston[1]) * np.exp(-double_heston[0] * times)
    fast = double_heston[6] + (double_heston[9] - double_heston[6]) * np.exp(-double_heston[5] * times)
    plt.figure(figsize=(9, 5)); plt.plot(times * 365, slow, label="v_slow"); plt.plot(times * 365, fast, label="v_fast"); plt.plot(times * 365, slow + fast, label="v_total")
    plt.xlabel("Calendar days from 2026-07-15"); plt.ylabel("Expected annualized variance"); plt.title("NTPC Double Heston variance decomposition — valuation 2026-07-15"); plt.legend()
    save("06_variance_decomposition.png")

    kappas = [heston[0], double_heston[0], double_heston[5]]; labels = ["Heston", "DH slow", "DH fast"]
    days = [half_life(value)[1] for value in kappas]
    plt.figure(figsize=(8, 5)); plt.bar(labels, days); plt.ylabel("Half-life (calendar days)"); plt.title("NTPC model-implied mean-reversion half-lives — 2026-07-15")
    save("07_kappa_half_life.png")

    plt.figure(figsize=(9, 5))
    usable = volatility.loc[volatility["actual_ex_post_realized_volatility"].notna()]
    x = np.arange(len(usable))
    width = 0.18
    columns = ["market_atm_iv", "bs_predicted_volatility", "heston_predicted_average_volatility", "double_heston_predicted_average_volatility", "actual_ex_post_realized_volatility"]
    for offset, column in enumerate(columns):
        plt.bar(x + (offset - 2) * width, usable[column], width, label=column)
    plt.xticks(x, usable["expiry_date"]); plt.ylabel("Annualized volatility"); plt.title("NTPC option-implied Q diagnostics vs ex-post physical P realized volatility — 2026-07-15"); plt.legend(fontsize=7)
    save("08_predicted_vs_realized_volatility.png")

    plt.figure(figsize=(11, 6))
    valid = starts.loc[starts["valid"]]
    normalized = valid[list(PARAMETER_NAMES)].copy()
    bounds = load_hard_safety_bounds(BOUNDS_PATH)
    for name in PARAMETER_NAMES:
        lower, upper = bounds[name]; normalized[name] = (normalized[name] - lower) / (upper - lower)
    plt.imshow(normalized.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1); plt.colorbar(label="Hard-bound normalized parameter"); plt.xticks(range(10), PARAMETER_NAMES, rotation=45, ha="right"); plt.yticks(range(len(valid)), valid["start_id"]); plt.ylabel("Start ID"); plt.title("NTPC Double Heston multi-start parameter stability — valuation 2026-07-15")
    save("09_double_heston_multistart_stability.png")

    plt.figure(figsize=(8, 5)); plt.bar(comparison["model"], comparison["holdout_price_rmse"]); plt.ylabel("Holdout price RMSE (INR)"); plt.title("NTPC model comparison — outer-strike holdout, valuation 2026-07-15")
    save("10_model_comparison.png")
    return paths


def _markdown_table(frame: pd.DataFrame, digits: int = 6) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(lambda value: "n/a" if pd.isna(value) else f"{value:.{digits}g}")
    headers = [str(column) for column in display.columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in display.itertuples(index=False, name=None))
    return "\n".join(lines)


def render_reports(
    all_options: pd.DataFrame,
    selected: pd.DataFrame,
    futures: pd.DataFrame,
    comparison: pd.DataFrame,
    heston: np.ndarray,
    double_heston: np.ndarray,
    stability: dict[str, Any],
    volatility: pd.DataFrame,
    realized_status: dict[str, Any],
    figures: list[Path],
) -> None:
    parameters = pd.DataFrame(
        {
            "parameter": ["kappa", "theta", "sigma", "rho", "v0"] + list(PARAMETER_NAMES),
            "model": ["HESTON"] * 5 + ["DOUBLE_HESTON"] * 10,
            "value": list(heston) + list(double_heston),
        }
    )
    half_lives = pd.DataFrame(
        [
            {"component": "Heston", "kappa": heston[0], "half_life_years": half_life(heston[0])[0], "half_life_calendar_days": half_life(heston[0])[1]},
            {"component": "Double Heston slow", "kappa": double_heston[0], "half_life_years": half_life(double_heston[0])[0], "half_life_calendar_days": half_life(double_heston[0])[1]},
            {"component": "Double Heston fast", "kappa": double_heston[5], "half_life_years": half_life(double_heston[5])[0], "half_life_calendar_days": half_life(double_heston[5])[1]},
        ]
    )
    winner = str(comparison["winner_classification"].iloc[0])
    near_errors = volatility.iloc[0][["bs_absolute_realized_volatility_error", "heston_absolute_realized_volatility_error", "double_heston_absolute_realized_volatility_error"]]
    realized_best = str(near_errors.astype(float).idxmin()).replace("_absolute_realized_volatility_error", "").upper()
    figure_lines = "\n".join(f"- `{path.relative_to(REPOSITORY_ROOT).as_posix()}`" for path in figures)
    report = f"""# NTPC Single-Stock Calibration Pilot

## Decision boundary

This is a bounded real-market calibration pilot for **NTPC only**, valued on **2026-07-15**. It compares Black-Scholes, Standard Heston, and the canonical ten-parameter Double Heston engine. The Double Heston vector is a **BEST-FIT NTPC CALIBRATION UNDER THE DECLARED CONTRACT**, not true NTPC parameters.

The model-comparison classification is **{winner}**. In the completed near-expiry arithmetic diagnostic, **{realized_best}** has the smallest absolute difference from realized volatility. The Heston quantities are risk-neutral (`Q`) expected-average variances inferred from option prices, whereas realized volatility is a physical-measure (`P`) outcome. No variance-risk-premium mapping was estimated, so this is not a validated physical-volatility forecast or a general forecasting-winner claim. The middle and far expiries were after the fixed 2026-08-12 as-of date and have no fabricated ex-post result.

## Official market and carry contract

- Official NSE F&O UDiFF source: `{_fo_paths()[0].name}` (SHA-256 `{sha256(_fo_paths()[0])}`).
- Spot: official NTPC EQ close and matching F&O `UndrlygPric`, **INR {SPOT:.2f}**.
- Actual expiries: 2026-07-28, 2026-08-25, 2026-09-29; DTE `13/41/76`; `T=DTE/365`.
- Primary price: official `ClsPric`, admitted only when `TtlTradgVol>0`, `TtlNbOfTxsExctd>0`, and `OpnIntrst>0`.
- Primary domain: near and middle expiries and exact `abs(log(K/S)) <= 0.10`. Far option volume was zero in all 26 rows and the far expiry is diagnostic-only.
- Deterministic unique strike matching minimizes total absolute distance to targets `[-0.10,-0.05,0,+0.05,+0.10]` separately by expiry and option type, with a strict maximum target distance of 0.05. Inner three targets are calibration; outer two are holdout. The near-call negative-outer target had no qualifying active strike and was left unmatched, giving the smallest fail-closed alternative: 12 calibration and 7 holdout rows.
- Risk-free source: [official RBI Treasury-bill auction result]({RISK_FREE_SOURCE_URL}), Press Release 2026-2027/672 dated 2026-07-15, showing the 91-day T-bill cut-off price **98.6880** and YTM **5.3324%**. The declared contract holds that simple yield flat across the three expiries: this is a short-end extrapolation for 13 and 41 days and a 91-day proxy for 76 days, not an acquired daily zero curve. It gives `D(T)=1/(1+yT)` and the maturity-equivalent continuous rate `-log(D)/T`.
- Rate provenance: the successful dated official HTML response and normalized field extract `raw/rbi_91_day_tbill_observation_20260715.json` are both hash-sealed in the manifest.
- The separately preserved `raw/rbi_current_rates_archive_20260715.html` is an earlier RBI perimeter-challenge response. It is retained and hash-listed for audit disclosure only, is not numerical evidence, and is not used by the carry contract.
- Carry: matched active NTPC futures close by actual expiry. `q=r-log(F/S)/T`; this is labelled futures-implied carry, not an observed dividend yield.

{_markdown_table(futures[["expiry_date", "DTE", "T", "futures_close", "discount_factor", "continuous_rate", "futures_implied_carry"]])}

## Data selection and market IV

Raw NTPC option rows: **{len(all_options)}**. Retained rows: **{len(selected)}**. Every excluded row and reason is in the ignored detailed CSV. Market IV is the robust bracketed solution of the forward-Black equation; impossible prices are rejected, never clipped.

{_markdown_table(selected[["sample_role", "expiry_date", "option_type", "target_log_moneyness", "strike", "log_moneyness", "observed_price", "market_implied_volatility"]])}

## Model comparison

The winner rule requires at least a 5% holdout-price-RMSE improvement over the runner-up and no worse holdout IV RMSE. Otherwise the result is `NO_CLEAR_WINNER`.

{_markdown_table(comparison[["model", "parameter_count", "calibration_price_rmse", "holdout_price_rmse", "calibration_price_mae", "holdout_price_mae", "calibration_relative_price_error_mean", "holdout_relative_price_error_mean", "calibration_iv_rmse", "holdout_iv_rmse", "runtime_seconds"]])}

## Best-fit parameters and mean reversion

{_markdown_table(parameters)}

At calibration time, `v0_total = v0_slow + v0_fast = {double_heston[4] + double_heston[9]:.8f}`. Expected total variance is `v_total(t)=v_slow(t)+v_fast(t)`. Prices are produced by one joint characteristic function; Heston prices are not added.

{_markdown_table(half_lives)}

Double Heston multi-start stability: **{stability['classification']}**; near-equivalent starts `{stability['near_equivalent_start_count']}`, materially displaced `{stability['materially_displaced_start_count']}`, maximum full-range-scaled distance `{stability['maximum_range_scaled_distance_from_best']:.6g}`. Optimizer convergence is not treated as parameter-identification evidence.

The selected Heston and Double Heston best iterates both reached the declared `max_nfev={MAX_NFEV}` cap without a SciPy convergence termination. They remain valid finite capped iterates and are reported as such; this further limits parameter interpretation.

## Predicted versus actual volatility

Here “predicted” follows the experiment's mechanical formula, not a claim of a physical-measure forecast: BS is the fitted option-implied constant and Heston/Double Heston are `Q`-measure expected-average volatilities from option-calibrated parameters. Comparing them with ex-post `P`-measure realized volatility is descriptive because no variance-risk-premium or `Q`-to-`P` mapping was fitted.

{_markdown_table(volatility)}

Official NSE CM closes from 2026-07-15 through the completed 2026-07-28 near expiry supplied **{realized_status['near_return_count']}** log returns. The official NSE corporate-actions API returned zero NTPC actions for that window, so the official EQ close series was used without adjustment. Middle/far realized volatility is unavailable because those expiries were future dates at the fixed as-of date.

## Figures

{figure_lines}

## Remaining limitations

- Historical bid/ask and quote sizes are unavailable in the free NSE UDiFF files; close-price activity filtering does not recreate them.
- Calls and puts are not independent after carry is fixed; retaining both exposes observed parity/microstructure differences but does not double structural information.
- The far expiry is not calibrated because every far NTPC option row had zero volume and zero executed trades.
- Only the near realized-volatility horizon is complete as of 2026-08-12.
- Real-market parameter truth is unknown; price fit and optimizer success do not validate the ten NTPC parameters.
"""
    mentor = f"""# NTPC Mentor Checkpoint

## What was done

NTPC was selected as the Power-sector primary at moderate confidence in Stage A. The pilot used only official NSE observations on **2026-07-15** from `{_fo_paths()[0].name}` (SHA-256 `{sha256(_fo_paths()[0])}`): spot **INR {SPOT:.2f}**, actual expiries **2026-07-28 / 2026-08-25 / 2026-09-29**, DTE **13 / 41 / 76**, and `T=DTE/365`.

The primary price is active-row `ClsPric`. Near/middle rows were restricted to `abs(log(K/S))<=0.10`; unique nearest listed strikes to `[-0.10,-0.05,0,+0.05,+0.10]` were selected separately by expiry/type under a strict 0.05 target-distance gate. Inner targets formed 12 calibration rows. The unsupported near-call negative-outer target was left unmatched, so the untouched holdout has 7 rows. The far expiry had no traded option rows and was excluded.

Market IV solves the forward-Black equation from observed prices. Carry uses active matched NTPC futures and a declared flat proxy based on the official RBI 91-day T-bill auction YTM 5.3324% from Press Release 2026-2027/672 dated 2026-07-15. The 13- and 41-day rates are short-end extrapolations; no exact daily zero curve was acquired. The successful official HTML and dated field extract are hash-sealed in the manifest. Futures-implied `q` is not called an observed dividend yield.

## Results

{_markdown_table(comparison[["model", "calibration_price_rmse", "holdout_price_rmse", "calibration_relative_price_error_mean", "holdout_relative_price_error_mean", "calibration_iv_rmse", "holdout_iv_rmse", "runtime_seconds"]])}

Classification: **{winner}**.

Black-Scholes fitted one common volatility, `sigma_BS={float(comparison.loc[comparison['model'] == 'BLACK_SCHOLES', 'sigma'].iloc[0]):.8f}`. Double Heston has the smallest calibration RMSE; Heston has the smallest holdout price RMSE, but its advantage over Double Heston is below the predeclared 5% margin and its holdout IV RMSE is slightly worse. Therefore no clear repricing winner is declared.

Best-fit Heston: `{dict(zip(['kappa','theta','sigma','rho','v0'], map(float, heston), strict=True))}`.

Best-fit canonical Double Heston: `{dict(zip(PARAMETER_NAMES, map(float, double_heston), strict=True))}`.

`v0_slow={double_heston[4]:.8f}`, `v0_fast={double_heston[9]:.8f}`, `v0_total={double_heston[4] + double_heston[9]:.8f}`. At each horizon, expected `v_total(t)=v_slow(t)+v_fast(t)` inside one joint characteristic function; Heston option prices are not added. Multi-start stability: **{stability['classification']}**.

The selected Heston and Double Heston iterates both reached the declared `max_nfev={MAX_NFEV}` cap without a SciPy convergence termination. They are finite capped best fits, not convergence or parameter-identification evidence.

{_markdown_table(half_lives)}

## Volatility checkpoint

{_markdown_table(volatility)}

The near expiry has {realized_status['near_return_count']} official close-to-close returns and a completed annualized realized volatility of {realized_status['near_annualized_realized_volatility']:.6f}. **{realized_best}** has the smallest near-horizon numerical absolute difference. However, Heston/Double Heston values are option-implied risk-neutral (`Q`) expected-average volatilities and realized volatility is a physical-measure (`P`) outcome. No variance-risk-premium mapping was estimated, so no physical forecasting winner is claimed. Middle/far actual volatility remains unavailable because the expiries had not occurred as of 2026-08-12.

## Unresolved

No bid/ask history, no far-expiry active option sample, only one completed realized horizon, and no known real NTPC Double Heston parameter truth. The best-fit vector is a calibration result, not truth.
"""
    atomic_write(REPORT_PATH, report.encode("utf-8"))
    atomic_write(MENTOR_PATH, mentor.encode("utf-8"))


def run() -> dict[str, Any]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    all_options, selected, futures = build_option_dataset()
    calibration = selected.loc[selected["sample_role"] == "CALIBRATION"].copy()
    holdout = selected.loc[selected["sample_role"] == "HOLDOUT"].copy()
    bs_summary, bs_predictions = fit_black_scholes(calibration, holdout)
    h_summary, h_predictions, h_starts, h_parameters = fit_stochastic_model(calibration, holdout, "HESTON")
    dh_summary, dh_predictions, dh_starts, dh_parameters = fit_stochastic_model(calibration, holdout, "DOUBLE_HESTON")
    comparison = pd.DataFrame([bs_summary, h_summary, dh_summary])
    winner = model_winner(comparison)
    comparison["winner_classification"] = winner
    stability = parameter_stability(dh_starts, dh_parameters)
    history, realized_status, action_path = build_realized_history()
    volatility = volatility_comparison(selected, bs_summary, h_parameters, dh_parameters, realized_status)
    predictions = pd.concat([bs_predictions, h_predictions, dh_predictions], ignore_index=True)
    figures = _figures(all_options, selected, predictions, comparison, h_parameters, dh_parameters, dh_starts, volatility)

    artifacts = {
        "all_options.csv": all_options,
        "selected_options.csv": selected,
        "carry_contract.csv": futures,
        "model_comparison.csv": comparison,
        "model_predictions.csv": predictions,
        "heston_multistart.csv": h_starts,
        "double_heston_multistart.csv": dh_starts,
        "realized_history.csv": history,
        "volatility_comparison.csv": volatility,
    }
    for name, frame in artifacts.items():
        write_csv(OUTPUT_ROOT / name, frame)
    write_json(OUTPUT_ROOT / "parameter_stability.json", stability)
    write_json(
        OUTPUT_ROOT / "risk_free_contract.json",
        {
            "source_url": RISK_FREE_SOURCE_URL,
            "source_description": RISK_FREE_SOURCE_DESCRIPTION,
            "valuation_date": VALUATION_DATE.isoformat(),
            "market_observation_date": "2026-07-15",
            "simple_annual_yield": RISK_FREE_SIMPLE_YIELD,
            "discount_rule": "D(T)=1/(1+y*T)",
            "continuous_rate_rule": "r(T)=-log(D(T))/T",
            "tenor_rule": "hold official 91-day simple yield flat; short-end extrapolation at 13 and 41 days; 91-day proxy at 76 days",
            "exact_daily_zero_curve_acquired": False,
            "preserved_observation_path": RISK_FREE_OBSERVATION_PATH.relative_to(OUTPUT_ROOT).as_posix(),
            "preserved_observation_sha256": sha256(RISK_FREE_OBSERVATION_PATH),
        },
    )
    render_reports(all_options, selected, futures, comparison, h_parameters, dh_parameters, stability, volatility, realized_status, figures)

    fo_csv, fo_zip = _fo_paths()
    if not RISK_FREE_OBSERVATION_PATH.is_file() or not RISK_FREE_HTTP_RESPONSE_PATH.is_file():
        raise FileNotFoundError("preserved RBI observation extract and disclosed HTTP response are required")
    source_files = {
        fo_csv.name: sha256(fo_csv),
        fo_zip.name: sha256(fo_zip),
        action_path.name: sha256(action_path),
        RISK_FREE_OBSERVATION_PATH.relative_to(OUTPUT_ROOT).as_posix(): sha256(RISK_FREE_OBSERVATION_PATH),
        RISK_FREE_HTTP_RESPONSE_PATH.relative_to(OUTPUT_ROOT).as_posix(): sha256(RISK_FREE_HTTP_RESPONSE_PATH),
    }
    for value in HISTORY_DATES:
        directory = RAW_ROOT / value.isoformat()
        base = f"BhavCopy_NSE_CM_0_0_0_{value:%Y%m%d}_F_0000.csv"
        source_files[f"{value.isoformat()}/{base}"] = sha256(directory / base)
        source_files[f"{value.isoformat()}/{base}.zip"] = sha256(directory / f"{base}.zip")
    ignored_artifacts = sorted(
        [path for path in OUTPUT_ROOT.rglob("*") if path.is_file() and path != MANIFEST_PATH],
        key=lambda path: path.as_posix(),
    )
    manifest = {
        "analysis_id": "NTPC_SINGLE_STOCK_PILOT",
        "status": {
            "experiment": "COMPLETE_WITH_FUTURE_REALIZED_HORIZONS_UNAVAILABLE",
            "carry_contract": "RESOLVED_WITH_DECLARED_OFFICIAL_91D_PROXY",
            "realized_vol_contract_near": "RESOLVED",
            "realized_vol_contract_middle_far": "UNAVAILABLE_FUTURE_EXPIRIES",
            "winner": winner,
            "double_heston_parameter_truth": "UNKNOWN_NOT_CLAIMED",
        },
        "valuation_date": VALUATION_DATE.isoformat(),
        "as_of_date": AS_OF_DATE.isoformat(),
        "underlying": SYMBOL,
        "spot": SPOT,
        "expiries": [value.isoformat() for value in EXPIRIES],
        "dte": [(value - VALUATION_DATE).days for value in EXPIRIES],
        "T": [(value - VALUATION_DATE).days / YEAR_BASIS for value in EXPIRIES],
        "selection": {
            "primary_price_field": PRIMARY_PRICE_FIELD,
            "activity_rule": "positive close, traded volume, executed trades, and open interest",
            "moneyness_rule": "exact abs(log(K/S)) <= 0.10",
            "target_assignment": "minimum-total-distance unique matching by expiry and option type",
            "maximum_target_distance": 0.05,
            "geometry_alternative": "near-call negative-outer target unmatched; no gate widening",
            "calibration_targets": list(CALIBRATION_TARGETS),
            "holdout_targets": list(HOLDOUT_TARGETS),
            "calibration_rows": len(calibration),
            "holdout_rows": len(holdout),
            "raw_option_rows": len(all_options),
        },
        "carry_contract": {
            "risk_free_source_url": RISK_FREE_SOURCE_URL,
            "risk_free_simple_yield": RISK_FREE_SIMPLE_YIELD,
            "discount_rule": "D(T)=1/(1+y*T)",
            "tenor_rule": "hold official 91-day simple yield flat; short-end extrapolation at 13 and 41 days; 91-day proxy at 76 days",
            "exact_daily_zero_curve_acquired": False,
            "forward_source": "matched active NTPC futures ClsPric",
            "carry_rule": "q=r-log(F/S)/T",
            "dividend_truth_claimed": False,
        },
        "seeds": {"analysis": ANALYSIS_SEED, "heston_starts": HESTON_STARTS, "double_heston_starts": DOUBLE_HESTON_STARTS},
        "optimizer": {"method": "scipy.optimize.least_squares trf", "max_nfev": MAX_NFEV, "node_count": NODE_COUNT},
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "platform": platform.platform(),
        },
        "source_identifiers": {
            "fo_archive_url": "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_20260715_F_0000.csv.zip",
            "cm_archive_url_pattern": "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_YYYYMMDD_F_0000.csv.zip",
            "risk_free_url": RISK_FREE_SOURCE_URL,
            "risk_free_observation_path": RISK_FREE_OBSERVATION_PATH.relative_to(OUTPUT_ROOT).as_posix(),
            "corporate_actions_url": CORPORATE_ACTIONS_URL,
        },
        "winner_rule": {"holdout_price_rmse_margin": WINNER_MARGIN, "requires_no_worse_holdout_iv_rmse": True},
        "source_hashes": source_files,
        "artifact_hashes": {path.relative_to(OUTPUT_ROOT).as_posix(): sha256(path) for path in ignored_artifacts},
        "tracked_artifact_hashes": {
            REPORT_PATH.relative_to(REPOSITORY_ROOT).as_posix(): sha256(REPORT_PATH),
            MENTOR_PATH.relative_to(REPOSITORY_ROOT).as_posix(): sha256(MENTOR_PATH),
            Path(__file__).resolve().relative_to(REPOSITORY_ROOT).as_posix(): sha256(Path(__file__).resolve()),
        },
        "commands": {
            "run": "C:\\Python313\\python.exe -B scripts/run_ntpc_single_stock_pilot.py",
            "render_replay": "C:\\Python313\\python.exe -B scripts/run_ntpc_single_stock_pilot.py --render-only",
            "focused_tests": "C:\\Python313\\python.exe -m pytest -q -p no:cacheprovider tests/test_ntpc_single_stock_pilot.py",
            "full_tests": "C:\\Python313\\python.exe -m pytest -q -p no:cacheprovider",
            "diff_check": "git diff --check",
        },
        "verification": {"focused_tests": "PENDING", "full_tests": "PENDING", "deterministic_replay": "PENDING", "independent_review": "PENDING"},
    }
    write_json(MANIFEST_PATH, manifest)
    return {
        "winner": winner,
        "comparison": _rows(comparison),
        "heston_parameters": dict(zip(["kappa", "theta", "sigma", "rho", "v0"], map(float, h_parameters), strict=True)),
        "double_heston_parameters": dict(zip(PARAMETER_NAMES, map(float, dh_parameters), strict=True)),
        "stability": stability,
        "realized_status": realized_status,
        "manifest": str(MANIFEST_PATH),
    }


def render_existing_outputs() -> dict[str, str]:
    required = {
        "all_options": OUTPUT_ROOT / "all_options.csv",
        "selected": OUTPUT_ROOT / "selected_options.csv",
        "futures": OUTPUT_ROOT / "carry_contract.csv",
        "comparison": OUTPUT_ROOT / "model_comparison.csv",
        "predictions": OUTPUT_ROOT / "model_predictions.csv",
        "heston_starts": OUTPUT_ROOT / "heston_multistart.csv",
        "double_heston_starts": OUTPUT_ROOT / "double_heston_multistart.csv",
        "history": OUTPUT_ROOT / "realized_history.csv",
        "volatility": OUTPUT_ROOT / "volatility_comparison.csv",
        "stability": OUTPUT_ROOT / "parameter_stability.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing canonical replay artifacts: {missing}")
    frames = {name: pd.read_csv(path) for name, path in required.items() if name != "stability"}
    stability = json.loads(required["stability"].read_text(encoding="utf-8"))
    comparison = frames["comparison"]
    heston_row = comparison.loc[comparison["model"] == "HESTON"].iloc[0]
    double_row = comparison.loc[comparison["model"] == "DOUBLE_HESTON"].iloc[0]
    heston = heston_row[["kappa", "theta", "sigma", "rho", "v0"]].to_numpy(float)
    double_heston = double_row[list(PARAMETER_NAMES)].to_numpy(float)
    near_volatility, near_count, near_rv = realized_volatility(frames["history"]["close"])
    realized_status = {
        "near_return_count": near_count,
        "near_realized_variance": near_rv,
        "near_annualized_realized_volatility": near_volatility,
    }
    replay_targets = [REPORT_PATH, MENTOR_PATH] + sorted(FIGURE_ROOT.glob("*.png"))
    before = {str(path): sha256(path) for path in replay_targets}
    figures = _figures(
        frames["all_options"],
        frames["selected"],
        frames["predictions"],
        comparison,
        heston,
        double_heston,
        frames["double_heston_starts"],
        frames["volatility"],
    )
    render_reports(
        frames["all_options"],
        frames["selected"],
        frames["futures"],
        comparison,
        heston,
        double_heston,
        stability,
        frames["volatility"],
        realized_status,
        figures,
    )
    after = {str(path): sha256(path) for path in replay_targets}
    changed = [path for path in before if before[path] != after[path]]
    if changed:
        raise RuntimeError(f"deterministic render replay changed artifacts: {changed}")
    if MANIFEST_PATH.is_file():
        write_json(
            OUTPUT_ROOT / "risk_free_contract.json",
            {
                "source_url": RISK_FREE_SOURCE_URL,
                "source_description": RISK_FREE_SOURCE_DESCRIPTION,
                "valuation_date": VALUATION_DATE.isoformat(),
                "market_observation_date": "2026-07-15",
                "simple_annual_yield": RISK_FREE_SIMPLE_YIELD,
                "discount_rule": "D(T)=1/(1+y*T)",
                "continuous_rate_rule": "r(T)=-log(D(T))/T",
                "tenor_rule": "hold official 91-day simple yield flat; short-end extrapolation at 13 and 41 days; 91-day proxy at 76 days",
                "exact_daily_zero_curve_acquired": False,
                "preserved_observation_path": RISK_FREE_OBSERVATION_PATH.relative_to(OUTPUT_ROOT).as_posix(),
                "preserved_observation_sha256": sha256(RISK_FREE_OBSERVATION_PATH),
            },
        )
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        manifest["status"]["carry_contract"] = "RESOLVED_WITH_DECLARED_OFFICIAL_91D_PROXY"
        manifest["selection"]["moneyness_rule"] = "exact abs(log(K/S)) <= 0.10"
        manifest["carry_contract"].update(
            {
                "risk_free_source_url": RISK_FREE_SOURCE_URL,
                "risk_free_simple_yield": RISK_FREE_SIMPLE_YIELD,
                "tenor_rule": "hold official 91-day simple yield flat; short-end extrapolation at 13 and 41 days; 91-day proxy at 76 days",
                "exact_daily_zero_curve_acquired": False,
            }
        )
        manifest["runtime"] = {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "platform": platform.platform(),
        }
        manifest["source_identifiers"] = {
            "fo_archive_url": "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_20260715_F_0000.csv.zip",
            "cm_archive_url_pattern": "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_YYYYMMDD_F_0000.csv.zip",
            "risk_free_url": RISK_FREE_SOURCE_URL,
            "risk_free_observation_path": RISK_FREE_OBSERVATION_PATH.relative_to(OUTPUT_ROOT).as_posix(),
            "corporate_actions_url": CORPORATE_ACTIONS_URL,
        }
        manifest["source_hashes"][RISK_FREE_OBSERVATION_PATH.relative_to(OUTPUT_ROOT).as_posix()] = sha256(
            RISK_FREE_OBSERVATION_PATH
        )
        manifest["source_hashes"][RISK_FREE_HTTP_RESPONSE_PATH.relative_to(OUTPUT_ROOT).as_posix()] = sha256(
            RISK_FREE_HTTP_RESPONSE_PATH
        )
        manifest["commands"]["render_replay"] = (
            "C:\\Python313\\python.exe -B scripts\\run_ntpc_single_stock_pilot.py --render-only"
        )
        ignored_artifacts = sorted(
            [path for path in OUTPUT_ROOT.rglob("*") if path.is_file()],
            key=lambda path: path.as_posix(),
        )
        manifest["artifact_hashes"] = {
            path.relative_to(OUTPUT_ROOT).as_posix(): sha256(path) for path in ignored_artifacts
        }
        manifest["tracked_artifact_hashes"] = {
            REPORT_PATH.relative_to(REPOSITORY_ROOT).as_posix(): sha256(REPORT_PATH),
            MENTOR_PATH.relative_to(REPOSITORY_ROOT).as_posix(): sha256(MENTOR_PATH),
            Path(__file__).resolve().relative_to(REPOSITORY_ROOT).as_posix(): sha256(Path(__file__).resolve()),
        }
        write_json(MANIFEST_PATH, manifest)
    return {str(Path(path).relative_to(REPOSITORY_ROOT)): digest for path, digest in after.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-only", action="store_true")
    arguments = parser.parse_args()
    result = render_existing_outputs() if arguments.render_only else run()
    print(json.dumps(result, indent=2, sort_keys=True))
