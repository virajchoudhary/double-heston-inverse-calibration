#!/usr/bin/env python3
"""Leakage-safe single-Heston calibration on the authentic NSE model-input CSV."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.optimize import least_squares
from scipy.special import expit, roots_laguerre
from scipy.stats import norm


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "outputs" / "019fc8a0" / "model_input_option_prices.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "single_heston"
GL_X, GL_W = roots_laguerre(64)
PARAMETER_STARTS = (
    (1.5, 0.06, 0.30, -0.60, 0.06),
    (3.0, 0.04, 0.25, -0.30, 0.04),
    (0.7, 0.12, 0.35, -0.75, 0.12),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def heston_characteristic(u, maturity, spot, rate, dividend, params):
    kappa, theta, sigma, rho, v0 = params
    iu = 1j * u
    b = kappa - rho * sigma * iu
    d = np.sqrt(b * b + sigma * sigma * (u * u + iu))
    g = (b - d) / (b + d)
    exp_dt = np.exp(-d * maturity)
    c = (
        (rate - dividend) * iu * maturity
        + (kappa * theta / sigma**2)
        * ((b - d) * maturity - 2 * np.log((1 - g * exp_dt) / (1 - g)))
    )
    big_d = ((b - d) / sigma**2) * ((1 - exp_dt) / (1 - g * exp_dt))
    return np.exp(iu * np.log(spot) + c + big_d * v0)


def heston_call_prices(spot, strikes, maturity, rate, dividend, params):
    """Little-Heston-trap probabilities with 64-point Gauss-Laguerre integration."""
    strikes = np.asarray(strikes, dtype=float)
    u = GL_X
    phase = np.exp(-1j * np.log(strikes)[:, None] * u[None, :])
    phi = heston_characteristic(u, maturity, spot, rate, dividend, params)
    phi_minus_i = heston_characteristic(
        np.array([-1j]), maturity, spot, rate, dividend, params
    )[0]
    phi_shift = heston_characteristic(u - 1j, maturity, spot, rate, dividend, params)
    quadrature = GL_W * np.exp(GL_X)
    p2 = 0.5 + np.sum(
        quadrature[None, :] * np.real(phase * phi[None, :] / (1j * u)[None, :]),
        axis=1,
    ) / np.pi
    p1 = 0.5 + np.sum(
        quadrature[None, :]
        * np.real(phase * phi_shift[None, :] / ((1j * u) * phi_minus_i)[None, :]),
        axis=1,
    ) / np.pi
    return spot * np.exp(-dividend * maturity) * p1 - strikes * np.exp(
        -rate * maturity
    ) * p2


def heston_prices(quotes: pd.DataFrame, params) -> np.ndarray:
    prices = np.empty(len(quotes), dtype=float)
    for _, idx in quotes.groupby("expiry_date", sort=False).groups.items():
        positions = quotes.index.get_indexer(idx)
        group = quotes.loc[idx]
        first = group.iloc[0]
        calls = heston_call_prices(
            float(first.spot),
            group.strike.to_numpy(float),
            float(first.maturity),
            float(first.rate),
            float(first.dividend),
            params,
        )
        puts = calls - float(first.spot) * math.exp(
            -float(first.dividend) * float(first.maturity)
        ) + group.strike.to_numpy(float) * math.exp(
            -float(first.rate) * float(first.maturity)
        )
        prices[positions] = np.where(group.option_type.eq("CE"), calls, puts)
    return prices


def black_scholes_price(spot, strike, maturity, rate, dividend, volatility, is_call):
    spot = np.asarray(spot, float)
    strike = np.asarray(strike, float)
    maturity = np.asarray(maturity, float)
    volatility = np.asarray(volatility, float)
    root_t = np.sqrt(maturity)
    d1 = (
        np.log(spot / strike)
        + (rate - dividend + 0.5 * volatility**2) * maturity
    ) / (volatility * root_t)
    d2 = d1 - volatility * root_t
    call = spot * np.exp(-dividend * maturity) * norm.cdf(d1) - strike * np.exp(
        -rate * maturity
    ) * norm.cdf(d2)
    put = call - spot * np.exp(-dividend * maturity) + strike * np.exp(
        -rate * maturity
    )
    return np.where(is_call, call, put)


def black_scholes_vega(spot, strike, maturity, rate, dividend, volatility):
    root_t = np.sqrt(maturity)
    d1 = (
        np.log(spot / strike)
        + (rate - dividend + 0.5 * volatility**2) * maturity
    ) / (volatility * root_t)
    return spot * np.exp(-dividend * maturity) * norm.pdf(d1) * root_t


def implied_volatility(price, spot, strike, maturity, rate, dividend, is_call):
    arrays = np.broadcast_arrays(price, spot, strike, maturity, rate, dividend, is_call)
    price, spot, strike, maturity, rate, dividend, is_call = arrays
    lower = np.where(
        is_call,
        np.maximum(spot * np.exp(-dividend * maturity) - strike * np.exp(-rate * maturity), 0),
        np.maximum(strike * np.exp(-rate * maturity) - spot * np.exp(-dividend * maturity), 0),
    )
    upper = np.where(
        is_call,
        spot * np.exp(-dividend * maturity),
        strike * np.exp(-rate * maturity),
    )
    valid = (
        np.isfinite(price)
        & (maturity > 0)
        & (price > lower + 1e-9)
        & (price < upper - 1e-9)
    )
    lo = np.full(price.shape, 1e-4)
    hi = np.full(price.shape, 5.0)
    for _ in range(64):
        mid = (lo + hi) / 2
        estimate = black_scholes_price(
            spot, strike, maturity, rate, dividend, mid, is_call
        )
        lo = np.where(estimate < price, mid, lo)
        hi = np.where(estimate >= price, mid, hi)
    result = (lo + hi) / 2
    return np.where(valid, result, np.nan)


def encode_params(params):
    kappa, theta, sigma, rho, v0 = params
    logistic = lambda value, low, high: np.log((value - low) / (high - value))
    sigma_cap = 0.995 * math.sqrt(2 * kappa * theta)
    return np.array(
        [
            logistic(kappa, 0.05, 10.0),
            logistic(theta, 0.005, 1.0),
            logistic(sigma, 0.01, sigma_cap),
            np.arctanh(rho / 0.98),
            logistic(v0, 0.005, 1.0),
        ]
    )


def decode_params(values):
    kappa = 0.05 + 9.95 * expit(values[0])
    theta = 0.005 + 0.995 * expit(values[1])
    sigma_cap = 0.995 * math.sqrt(2 * kappa * theta)
    sigma = 0.01 + (sigma_cap - 0.01) * expit(values[2])
    rho = 0.98 * np.tanh(values[3])
    v0 = 0.005 + 0.995 * expit(values[4])
    return np.array([kappa, theta, sigma, rho, v0])


def residuals(values, quotes):
    params = decode_params(values)
    model = heston_prices(quotes, params)
    scale = np.maximum(quotes.vega.to_numpy(float), 0.002 * quotes.spot.to_numpy(float))
    return (model - quotes.market_price_adjusted.to_numpy(float)) / scale * quotes.weight.to_numpy(float)


def fit_all_params(quotes):
    fits = []
    for start in PARAMETER_STARTS:
        result = least_squares(
            residuals,
            encode_params(start),
            args=(quotes,),
            bounds=(-8, 8),
            loss="soft_l1",
            f_scale=0.05,
            max_nfev=90,
        )
        params = decode_params(result.x)
        fits.append((float(np.mean(residuals(result.x, quotes) ** 2)), params, result.nfev))
    return min(fits, key=lambda item: item[0]), fits


def encode_joint(structural, initial_variances):
    first = encode_params((*structural, initial_variances[0]))
    remaining = [encode_params((*structural, value))[4] for value in initial_variances[1:]]
    return np.r_[first[:4], first[4], remaining]


def decode_joint(values):
    first = decode_params(np.r_[values[:4], values[4]])
    variances = [first[4]]
    for value in values[5:]:
        variances.append(0.005 + 0.995 * expit(value))
    return first[:4], np.asarray(variances)


def joint_residuals(values, surfaces):
    structural, variances = decode_joint(values)
    output = []
    for quotes, v0 in zip(surfaces, variances):
        model = heston_prices(quotes, np.r_[structural, v0])
        scale = np.maximum(quotes.vega.to_numpy(float), 0.002 * quotes.spot.to_numpy(float))
        # Equal date influence prevents a liquid date from becoming the whole model.
        output.append(
            (model - quotes.market_price_adjusted.to_numpy(float))
            / scale
            * quotes.weight.to_numpy(float)
            / math.sqrt(len(quotes))
        )
    return np.concatenate(output)


def fit_joint_structural(surfaces):
    initial_variances = [
        float(np.clip(np.nanmedian(quotes.market_iv) ** 2, 0.006, 0.9))
        for quotes in surfaces
    ]
    fits = []
    for start in PARAMETER_STARTS:
        result = least_squares(
            joint_residuals,
            encode_joint(start[:4], initial_variances),
            args=(surfaces,),
            bounds=(-8, 8),
            loss="soft_l1",
            f_scale=0.02,
            max_nfev=140,
        )
        structural, variances = decode_joint(result.x)
        objective = float(np.mean(joint_residuals(result.x, surfaces) ** 2))
        fits.append((objective, structural, variances, result.nfev))
    return min(fits, key=lambda item: item[0]), fits


def fit_v0(quotes, structural):
    start_v0 = float(np.clip(np.nanmedian(quotes.market_iv) ** 2, 0.006, 0.9))
    start = (*structural, start_v0)

    def one_state(value):
        params = np.array([*structural, 0.005 + 0.995 * expit(value[0])])
        model = heston_prices(quotes, params)
        scale = np.maximum(quotes.vega.to_numpy(float), 0.002 * quotes.spot.to_numpy(float))
        return (model - quotes.market_price_adjusted.to_numpy(float)) / scale * quotes.weight.to_numpy(float)

    encoded = encode_params(start)[4]
    result = least_squares(
        one_state,
        [encoded],
        bounds=(-8, 8),
        loss="soft_l1",
        f_scale=0.05,
        max_nfev=50,
    )
    return 0.005 + 0.995 * expit(result.x[0]), result.nfev


def robust_carry(strike, call_minus_put, spot, maturity):
    design = np.column_stack([np.ones(len(strike)), strike])
    beta = np.linalg.lstsq(design, call_minus_put, rcond=None)[0]
    for _ in range(6):
        error = call_minus_put - design @ beta
        scale = max(np.median(np.abs(error)) * 1.4826, 0.05)
        weights = np.minimum(1.0, 1.345 * scale / np.maximum(np.abs(error), 1e-12))
        beta = np.linalg.lstsq(
            design * weights[:, None], call_minus_put * weights, rcond=None
        )[0]
    discount = -float(beta[1])
    forward = float(beta[0] / discount) if discount > 0 else np.nan
    error = call_minus_put - discount * (forward - strike)
    nrmse = float(np.sqrt(np.mean(error**2)) / spot)
    valid = (
        np.isfinite(forward)
        and math.exp(-0.25 * maturity) <= discount <= math.exp(0.10 * maturity)
        and 0.75 <= forward / spot <= 1.25
        and nrmse <= 0.025
        and (strike.max() - strike.min()) / spot >= 0.04
    )
    if not valid:
        return None
    rate = -math.log(discount) / maturity
    dividend = rate - math.log(forward / spot) / maturity
    if not (-0.50 <= dividend <= 0.50):
        return None
    return forward, discount, rate, dividend, nrmse


def prepare_date(day: pd.DataFrame, audit: Counter, holdout_fold: int = 1) -> pd.DataFrame | None:
    result = []
    audit["authentic_ready_rows_seen"] += len(day)
    spot = float(day.adjusted_underlying_value.median())
    for expiry, group in day.groupby("expiry_date", sort=True):
        maturity = float(group.days_to_expiry.iloc[0]) / 365.0
        if not 7 / 365 <= maturity <= 180 / 365:
            audit["expiry_rejected_maturity"] += 1
            continue
        calls = (
            group[group.option_type.eq("CE")]
            .sort_values("number_of_contracts", ascending=False)
            .drop_duplicates("adjusted_strike_price")
            .set_index("adjusted_strike_price")
        )
        puts = (
            group[group.option_type.eq("PE")]
            .sort_values("number_of_contracts", ascending=False)
            .drop_duplicates("adjusted_strike_price")
            .set_index("adjusted_strike_price")
        )
        paired = calls.join(puts, lsuffix="_ce", rsuffix="_pe", how="inner").sort_index()
        audit["unpaired_or_duplicate_rows_excluded"] += len(group) - 2 * len(paired)
        if len(paired) < 6:
            audit["expiry_rejected_fewer_than_six_pairs"] += 1
            continue
        sequence = np.arange(len(paired))
        if holdout_fold not in (0, 1, 2):
            raise ValueError("holdout_fold must be 0, 1, or 2")
        holdout = sequence % 3 == holdout_fold
        anchors = paired.iloc[~holdout]
        carry = robust_carry(
            anchors.index.to_numpy(float),
            (
                anchors.adjusted_observed_option_price_ce
                - anchors.adjusted_observed_option_price_pe
            ).to_numpy(float),
            spot,
            maturity,
        )
        if carry is None:
            audit["expiry_rejected_unreliable_put_call_parity"] += 1
            continue
        forward, discount, rate, dividend, parity_nrmse = carry
        for position, (strike, row) in enumerate(paired.iterrows()):
            option = "CE" if strike >= forward else "PE"
            suffix = "ce" if option == "CE" else "pe"
            price_adjusted = float(row[f"adjusted_observed_option_price_{suffix}"])
            price_raw = float(row[f"observed_option_price_{suffix}"])
            contracts = float(row[f"number_of_contracts_{suffix}"])
            factor = float(row[f"price_adjustment_factor_{suffix}"])
            result.append(
                {
                    "symbol": row[f"symbol_{suffix}"],
                    "trade_date": row[f"trade_date_{suffix}"],
                    "expiry_date": expiry,
                    "option_type": option,
                    "strike": float(strike),
                    "spot": spot,
                    "maturity": maturity,
                    "days_to_expiry": int(row[f"days_to_expiry_{suffix}"]),
                    "forward": forward,
                    "discount_factor": discount,
                    "rate": rate,
                    "dividend": dividend,
                    "parity_nrmse": parity_nrmse,
                    "fold": "holdout" if holdout[position] else "calibration",
                    "market_price_adjusted": price_adjusted,
                    "market_price_raw": price_raw,
                    "price_adjustment_factor": factor,
                    "contracts": contracts,
                    "open_interest": float(row[f"open_interest_{suffix}"]),
                    "row_key": row[f"row_key_{suffix}"],
                    "source_file": row[f"source_file_{suffix}"],
                    "anchor_ce_row_key": row.row_key_ce if not holdout[position] else "",
                    "anchor_pe_row_key": row.row_key_pe if not holdout[position] else "",
                }
            )
    if not result:
        return None
    quotes = pd.DataFrame(result)
    calls = quotes.option_type.eq("CE").to_numpy()
    quotes["log_forward_moneyness"] = np.log(quotes.strike / quotes.forward)
    quotes["market_iv"] = implied_volatility(
        quotes.market_price_adjusted.to_numpy(float),
        quotes.spot.to_numpy(float),
        quotes.strike.to_numpy(float),
        quotes.maturity.to_numpy(float),
        quotes.rate.to_numpy(float),
        quotes.dividend.to_numpy(float),
        calls,
    )
    quotes["vega"] = black_scholes_vega(
        quotes.spot.to_numpy(float),
        quotes.strike.to_numpy(float),
        quotes.maturity.to_numpy(float),
        quotes.rate.to_numpy(float),
        quotes.dividend.to_numpy(float),
        quotes.market_iv.to_numpy(float),
    )
    valid = (
        quotes.market_iv.between(0.03, 2.5)
        & quotes.log_forward_moneyness.abs().le(0.35)
        & quotes.vega.ge(0.002 * quotes.spot)
    )
    audit["quote_rejected_iv_moneyness_or_vega"] += int((~valid).sum())
    quotes = quotes[valid].copy()
    counts = quotes.groupby(["expiry_date", "fold"]).size().unstack(fill_value=0)
    valid_expiry = counts.index[
        (counts.get("calibration", 0) >= 3) & (counts.get("holdout", 0) >= 1)
    ]
    audit["quote_rejected_after_fold_minimums"] += int(
        (~quotes.expiry_date.isin(valid_expiry)).sum()
    )
    quotes = quotes[quotes.expiry_date.isin(valid_expiry)].copy()
    if quotes.fold.eq("calibration").sum() < 6 or quotes.fold.eq("holdout").sum() < 2:
        audit["date_rejected_insufficient_clean_quotes"] += 1
        return None
    liquidity = np.log1p(quotes.contracts.clip(lower=1)) ** 0.25
    expiry_count = quotes.groupby("expiry_date").expiry_date.transform("size")
    quotes["weight"] = liquidity / np.sqrt(expiry_count)
    quotes["weight"] /= quotes.weight.median()
    # ponytail: fold-by-strike is the smallest auditable barrier against quote leakage.
    assert not quotes.groupby(["expiry_date", "strike"]).fold.nunique().gt(1).any()
    audit["clean_quotes_retained"] += len(quotes)
    return quotes.reset_index(drop=True)


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    clean = frame.dropna(subset=["market_iv", "heston_iv"])
    error = clean.heston_iv - clean.market_iv
    price_error = clean.heston_price_raw - clean.market_price_raw
    denominator = np.sum((clean.market_iv - clean.market_iv.mean()) ** 2)
    return {
        "rows": len(clean),
        "iv_rmse": float(np.sqrt(np.mean(error**2))) if len(clean) else np.nan,
        "iv_mae": float(np.mean(np.abs(error))) if len(clean) else np.nan,
        "iv_bias": float(np.mean(error)) if len(clean) else np.nan,
        "iv_r2": float(1 - np.sum(error**2) / denominator)
        if len(clean) > 1 and denominator > 0
        else np.nan,
        "raw_price_rmse": float(np.sqrt(np.mean(price_error**2))) if len(clean) else np.nan,
        "raw_price_mae": float(np.mean(np.abs(price_error))) if len(clean) else np.nan,
    }


def predict(quotes, params, split, fitted_rows):
    output = quotes.copy()
    output["heston_price_adjusted"] = heston_prices(output, params)
    output["heston_price_raw"] = (
        output.heston_price_adjusted / output.price_adjustment_factor
    )
    output["heston_iv"] = implied_volatility(
        output.heston_price_adjusted.to_numpy(float),
        output.spot.to_numpy(float),
        output.strike.to_numpy(float),
        output.maturity.to_numpy(float),
        output.rate.to_numpy(float),
        output.dividend.to_numpy(float),
        output.option_type.eq("CE").to_numpy(),
    )
    output["iv_error"] = output.heston_iv - output.market_iv
    output["absolute_iv_error"] = output.iv_error.abs()
    output["split"] = split
    output["used_in_v0_fit"] = output.fold.eq("calibration")
    output["structural_parameters_from"] = "chronological_train_only"
    output["v0_fitted_rows"] = fitted_rows
    return output


def choose_training_dates(dates, maximum=12):
    if len(dates) <= maximum:
        return list(dates)
    positions = np.linspace(0, len(dates) - 1, maximum).round().astype(int)
    return [dates[position] for position in np.unique(positions)]


def make_split_map(data):
    mapping = {}
    for symbol, group in data.groupby("symbol"):
        dates = np.array(sorted(group.trade_date.unique()))
        train_end = max(1, int(len(dates) * 0.70))
        validation_end = max(train_end + 1, int(len(dates) * 0.85))
        for position, day in enumerate(dates):
            mapping[(symbol, pd.Timestamp(day))] = (
                "train"
                if position < train_end
                else "validation"
                if position < validation_end
                else "test"
            )
    return mapping


def add_realized_volatility(data):
    spots = (
        data.groupby(["symbol", "trade_date"], as_index=False)
        .adjusted_underlying_value.median()
        .sort_values(["symbol", "trade_date"])
    )
    spots["log_return"] = spots.groupby("symbol").adjusted_underlying_value.transform(
        lambda values: np.log(values).diff()
    )
    spots["realized_vol_20d"] = spots.groupby("symbol").log_return.transform(
        lambda values: values.rolling(20, min_periods=15).std(ddof=1) * math.sqrt(252)
    )
    return spots.set_index(["symbol", "trade_date"]).realized_vol_20d.to_dict()


def plot_actual_vs_model(predictions, output):
    test = predictions[(predictions.split.eq("test")) & (predictions.fold.eq("holdout"))]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    axes[0].hexbin(
        test.market_iv,
        test.heston_iv,
        gridsize=45,
        mincnt=1,
        bins="log",
        cmap="viridis",
    )
    limit = float(np.nanpercentile(np.r_[test.market_iv, test.heston_iv], 99))
    limit = min(max(limit, 0.5), 2.5)
    axes[0].plot([0, limit], [0, limit], "r--", linewidth=1, label="perfect agreement")
    axes[0].set(xlim=(0, limit), ylim=(0, limit), xlabel="Market-implied volatility", ylabel="Heston-implied volatility", title="Held-out NSE quotes")
    axes[0].legend(frameon=False)
    symbol_metrics = test.groupby("symbol").apply(lambda frame: pd.Series(metrics(frame)), include_groups=False)
    symbol_metrics = symbol_metrics.sort_values("iv_rmse")
    axes[1].barh(symbol_metrics.index, symbol_metrics.iv_rmse, color="#3b4cc0")
    axes[1].set(xlabel="Held-out IV RMSE", title="Error by power-sector stock")
    fig.suptitle("Authentic NSE market IV versus leakage-safe single-Heston IV")
    fig.savefig(output / "actual_vs_heston_iv.png", dpi=180)
    plt.close(fig)


def plot_realized_vs_state(states, output):
    test = states[(states.split.eq("test")) & states.realized_vol_20d.notna()]
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    for symbol, group in test.groupby("symbol"):
        ax.scatter(group.realized_vol_20d, group.heston_spot_vol, s=15, alpha=0.55, label=symbol)
    limit = float(np.nanpercentile(np.r_[test.realized_vol_20d, test.heston_spot_vol], 99))
    limit = min(max(limit, 0.5), 2.5)
    ax.plot([0, limit], [0, limit], "k--", linewidth=1)
    ax.set(
        xlim=(0, limit),
        ylim=(0, limit),
        xlabel="Trailing 20-session realized volatility (causal)",
        ylabel="Heston instantaneous volatility sqrt(v0)",
        title="Historical realized volatility versus Heston state",
    )
    ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.savefig(output / "realized_vs_heston_spot_vol.png", dpi=180)
    plt.close(fig)


def make_surface_artifacts(predictions, states, parameters, prepared, output):
    grids = []
    surface_dir = output / "surfaces"
    surface_dir.mkdir(exist_ok=True)
    test_states = states[states.split.eq("test")].copy()
    for symbol, candidates in test_states.groupby("symbol"):
        candidates = candidates.sort_values(["expiry_count", "holdout_rows"], ascending=False)
        state = candidates.iloc[0]
        day = pd.Timestamp(state.trade_date)
        quotes = prepared[(symbol, day)]
        expiries = (
            quotes[["expiry_date", "maturity", "forward", "rate", "dividend", "spot"]]
            .drop_duplicates("expiry_date")
            .sort_values("maturity")
        )
        structural = parameters.loc[symbol, ["kappa", "theta", "sigma", "rho"]].to_numpy(float)
        params = np.r_[structural, float(state.v0)]
        money = np.linspace(-0.25, 0.25, 41)
        rows = []
        for expiry in expiries.itertuples(index=False):
            strike = expiry.forward * np.exp(money)
            option_type = np.where(strike >= expiry.forward, "CE", "PE")
            frame = pd.DataFrame(
                {
                    "expiry_date": expiry.expiry_date,
                    "spot": expiry.spot,
                    "strike": strike,
                    "maturity": expiry.maturity,
                    "rate": expiry.rate,
                    "dividend": expiry.dividend,
                    "option_type": option_type,
                }
            )
            price = heston_prices(frame, params)
            iv = implied_volatility(
                price,
                frame.spot,
                frame.strike,
                frame.maturity,
                frame.rate,
                frame.dividend,
                frame.option_type.eq("CE"),
            )
            for moneyness, strike_value, price_value, iv_value in zip(money, strike, price, iv):
                rows.append(
                    {
                        "symbol": symbol,
                        "trade_date": day.date().isoformat(),
                        "expiry_date": pd.Timestamp(expiry.expiry_date).date().isoformat(),
                        "maturity": expiry.maturity,
                        "log_forward_moneyness": moneyness,
                        "strike": strike_value,
                        "heston_price_adjusted": price_value,
                        "heston_iv": iv_value,
                        "value_origin": "HESTON_MODEL_GENERATED_NOT_MARKET_DATA",
                    }
                )
        grid = pd.DataFrame(rows)
        grids.append(grid)
        fig = plt.figure(figsize=(8, 6), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        pivot = grid.pivot(index="maturity", columns="log_forward_moneyness", values="heston_iv")
        xx, yy = np.meshgrid(pivot.columns.to_numpy(float), pivot.index.to_numpy(float))
        if len(pivot.index) >= 2:
            ax.plot_surface(xx, yy, pivot.to_numpy(), cmap=cm.viridis, alpha=0.72)
        else:
            ax.plot(pivot.columns, np.repeat(pivot.index[0], len(pivot.columns)), pivot.iloc[0], color="#3b4cc0")
        market = predictions[
            predictions.symbol.eq(symbol)
            & predictions.trade_date.eq(day)
            & predictions.fold.eq("holdout")
        ]
        ax.scatter(
            market.log_forward_moneyness,
            market.maturity,
            market.market_iv,
            color="crimson",
            s=24,
            label="held-out market IV",
        )
        ax.set(
            xlabel="log(K/F)",
            ylabel="maturity (years)",
            zlabel="implied volatility",
            title=f"{symbol} single-Heston surface — {day.date()}",
        )
        ax.legend(frameon=False)
        fig.savefig(surface_dir / f"{symbol}_heston_surface.png", dpi=180)
        plt.close(fig)
    pd.concat(grids, ignore_index=True).to_csv(output / "single_heston_surface_grid.csv", index=False)


def load_ready_data(path):
    columns = [
        "symbol", "trade_date", "expiry_date", "option_type", "days_to_expiry",
        "adjusted_strike_price", "adjusted_observed_option_price",
        "adjusted_underlying_value", "observed_option_price", "price_adjustment_factor",
        "number_of_contracts", "open_interest", "row_key", "source_file", "is_model_ready",
    ]
    data = pd.read_csv(path, usecols=columns, parse_dates=["trade_date", "expiry_date"], low_memory=False)
    ready = data.is_model_ready.astype(str).str.lower().isin(["t", "true", "1"])
    data = data[ready].copy()
    numeric = [column for column in columns if column not in {"symbol", "trade_date", "expiry_date", "option_type", "row_key", "source_file", "is_model_ready"}]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    required = ["adjusted_strike_price", "adjusted_observed_option_price", "adjusted_underlying_value", "observed_option_price", "number_of_contracts"]
    assert data[required].notna().all().all()
    assert data.row_key.is_unique
    return data


def build_checks(data, parameters, predictions, states, split_map, input_hash):
    test = predictions[(predictions.split.eq("test")) & predictions.fold.eq("holdout")]
    calibration = predictions[predictions.fold.eq("calibration")]
    keys_overlap = len(set(test.row_key) & set(calibration.row_key))
    structural_future_rows = int((parameters.training_last_date >= parameters.test_first_date).sum())
    feller_gap = 2 * parameters.kappa * parameters.theta - parameters.sigma**2
    calls = test.option_type.eq("CE")
    lower = np.where(
        calls,
        np.maximum(test.spot * np.exp(-test.dividend * test.maturity) - test.strike * np.exp(-test.rate * test.maturity), 0),
        np.maximum(test.strike * np.exp(-test.rate * test.maturity) - test.spot * np.exp(-test.dividend * test.maturity), 0),
    )
    upper = np.where(
        calls,
        test.spot * np.exp(-test.dividend * test.maturity),
        test.strike * np.exp(-test.rate * test.maturity),
    )
    checks = [
        ("input_sha256_recorded", bool(input_hash), input_hash),
        ("model_ready_source_rows_only", len(data) == 215636, len(data)),
        ("chronological_split_per_symbol", structural_future_rows == 0, structural_future_rows),
        ("test_row_keys_absent_from_calibration", keys_overlap == 0, keys_overlap),
        ("all_structural_parameters_feller_valid", bool((feller_gap > 0).all()), float(feller_gap.min())),
        ("all_correlations_inside_unit_interval", bool(parameters.rho.abs().lt(1).all()), float(parameters.rho.abs().max())),
        ("all_variance_parameters_positive", bool((parameters[["theta", "v0_training_median"]] > 0).all().all()), float(parameters[["theta", "v0_training_median"]].min().min())),
        ("test_prices_inside_no_arbitrage_bounds", bool(((test.heston_price_adjusted >= lower - 1e-6) & (test.heston_price_adjusted <= upper + 1e-6)).all()), int((~((test.heston_price_adjusted >= lower - 1e-6) & (test.heston_price_adjusted <= upper + 1e-6))).sum())),
        ("test_implied_volatility_finite", bool(np.isfinite(test.heston_iv).all()), int((~np.isfinite(test.heston_iv)).sum())),
        ("every_test_prediction_authentic_source_file", bool(test.source_file.str.startswith("raw/nse_fo_bhavcopies/").all()), int((~test.source_file.str.startswith("raw/nse_fo_bhavcopies/")).sum())),
        ("daily_state_only_uses_anchor_rows", bool((states.calibration_rows < states.total_clean_rows).all()), int((states.calibration_rows >= states.total_clean_rows).sum())),
    ]
    return pd.DataFrame(checks, columns=["check", "passed", "observed"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-training-dates", type=int, default=12)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    input_hash = sha256(args.input)
    data = load_ready_data(args.input)
    split_map = make_split_map(data)
    realized = add_realized_volatility(data)
    audit = Counter()
    prepared = {}

    def get_prepared(symbol, day):
        key = (symbol, pd.Timestamp(day))
        if key not in prepared:
            frame = data[data.symbol.eq(symbol) & data.trade_date.eq(day)]
            prepared[key] = prepare_date(frame, audit)
        return prepared[key]

    parameter_rows = []
    training_rows = []
    structural_by_symbol = {}
    for symbol, symbol_data in data.groupby("symbol"):
        train_dates = sorted(
            day for day in symbol_data.trade_date.unique() if split_map[(symbol, pd.Timestamp(day))] == "train"
        )
        candidates = choose_training_dates(train_dates, args.max_training_dates * 5)
        candidates += [day for day in train_dates if day not in set(candidates)]
        surfaces = []
        surface_dates = []
        for day in candidates:
            quotes = get_prepared(symbol, day)
            if quotes is None:
                continue
            calibration = quotes[quotes.fold.eq("calibration")].reset_index(drop=True)
            surfaces.append(calibration)
            surface_dates.append(pd.Timestamp(day))
            if len(surfaces) >= args.max_training_dates:
                break
        if len(surfaces) < 3:
            raise RuntimeError(f"{symbol}: fewer than three valid training dates")
        best, all_starts = fit_joint_structural(surfaces)
        loss, structural, variances, nfev = best
        structural_by_symbol[symbol] = structural
        start_loss_spread = float(np.ptp([item[0] for item in all_starts]))
        per_date_rows = []
        for day, calibration, v0 in zip(surface_dates, surfaces, variances):
            quotes = get_prepared(symbol, day)
            holdout = quotes[quotes.fold.eq("holdout")].reset_index(drop=True)
            params = np.r_[structural, v0]
            calibration_prediction = predict(calibration, params, "train", len(calibration))
            holdout_prediction = predict(holdout, params, "train", len(calibration))
            calibration_metrics = metrics(calibration_prediction)
            holdout_metrics = metrics(holdout_prediction)
            row = {
                "symbol": symbol,
                "trade_date": pd.Timestamp(day),
                "kappa": params[0], "theta": params[1], "sigma": params[2], "rho": params[3], "v0": params[4],
                "objective": loss, "optimizer_evaluations": nfev, "starting_value_loss_spread": start_loss_spread,
                "calibration_rows": len(calibration), "holdout_rows": len(holdout),
                "calibration_iv_rmse": calibration_metrics["iv_rmse"],
                "holdout_iv_rmse": holdout_metrics["iv_rmse"],
                "feller_gap": 2 * params[0] * params[1] - params[2] ** 2,
            }
            per_date_rows.append(row)
            training_rows.append(row)
        fit_frame = pd.DataFrame(per_date_rows)
        all_dates = sorted(symbol_data.trade_date.unique())
        first_test = next(day for day in all_dates if split_map[(symbol, pd.Timestamp(day))] == "test")
        last_train = max(day for day in all_dates if split_map[(symbol, pd.Timestamp(day))] == "train")
        parameter_rows.append(
            {
                "symbol": symbol,
                "kappa": structural[0], "theta": structural[1], "sigma": structural[2], "rho": structural[3],
                "v0_training_median": float(np.median(variances)),
                "feller_gap": 2 * structural[0] * structural[1] - structural[2] ** 2,
                "training_surface_fits": len(fit_frame),
                "stable_training_fits": len(fit_frame),
                "selected_training_date": "joint_train_fit",
                "training_last_date": pd.Timestamp(last_train),
                "test_first_date": pd.Timestamp(first_test),
                "selection_rule": "joint train-only fit with one v0 state per date and equal date weights",
            }
        )

    parameters = pd.DataFrame(parameter_rows).set_index("symbol").sort_index()
    prediction_frames = []
    state_rows = []
    for symbol, symbol_data in data.groupby("symbol"):
        structural = structural_by_symbol[symbol]
        for day in sorted(symbol_data.trade_date.unique()):
            split = split_map[(symbol, pd.Timestamp(day))]
            if split == "train":
                continue
            quotes = get_prepared(symbol, day)
            if quotes is None:
                continue
            calibration = quotes[quotes.fold.eq("calibration")].reset_index(drop=True)
            if len(calibration) < 6:
                continue
            v0, nfev = fit_v0(calibration, structural)
            params = np.r_[structural, v0]
            prediction = predict(quotes, params, split, len(calibration))
            prediction_frames.append(prediction)
            calibration_metrics = metrics(prediction[prediction.fold.eq("calibration")])
            holdout_metrics = metrics(prediction[prediction.fold.eq("holdout")])
            state_rows.append(
                {
                    "symbol": symbol,
                    "trade_date": pd.Timestamp(day),
                    "split": split,
                    "v0": v0,
                    "heston_spot_vol": math.sqrt(v0),
                    "realized_vol_20d": realized.get((symbol, pd.Timestamp(day)), np.nan),
                    "calibration_rows": int(prediction.fold.eq("calibration").sum()),
                    "holdout_rows": int(prediction.fold.eq("holdout").sum()),
                    "total_clean_rows": len(prediction),
                    "expiry_count": prediction.expiry_date.nunique(),
                    "optimizer_evaluations": nfev,
                    "calibration_iv_rmse": calibration_metrics["iv_rmse"],
                    "holdout_iv_rmse": holdout_metrics["iv_rmse"],
                    "overfit_gap_iv_rmse": holdout_metrics["iv_rmse"] - calibration_metrics["iv_rmse"],
                }
            )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    states = pd.DataFrame(state_rows)
    training = pd.DataFrame(training_rows)

    metric_rows = []
    for (split, fold), frame in predictions.groupby(["split", "fold"]):
        metric_rows.append({"scope": "all_symbols", "split": split, "fold": fold, **metrics(frame)})
    for (symbol, split, fold), frame in predictions.groupby(["symbol", "split", "fold"]):
        metric_rows.append({"scope": symbol, "split": split, "fold": fold, **metrics(frame)})
    metric_table = pd.DataFrame(metric_rows)

    parameter_export = parameters.reset_index()
    checks = build_checks(data, parameter_export, predictions, states, split_map, input_hash)
    if not checks.passed.all():
        raise RuntimeError("one or more strict single-Heston checks failed\n" + checks.to_string(index=False))

    parameter_export.to_csv(args.output / "single_heston_parameters.csv", index=False)
    training.to_csv(args.output / "single_heston_training_fits.csv", index=False)
    states.to_csv(args.output / "single_heston_daily_state.csv", index=False)
    predictions.to_csv(args.output / "single_heston_predictions.csv", index=False)
    metric_table.to_csv(args.output / "single_heston_metrics.csv", index=False)
    pd.DataFrame(sorted(audit.items()), columns=["event", "count"]).to_csv(
        args.output / "single_heston_noise_audit.csv", index=False
    )
    checks.to_csv(args.output / "single_heston_checks.csv", index=False)
    plot_actual_vs_model(predictions, args.output)
    plot_realized_vs_state(states, args.output)
    make_surface_artifacts(predictions, states, parameters, prepared, args.output)

    test_metrics = metrics(
        predictions[(predictions.split.eq("test")) & predictions.fold.eq("holdout")]
    )
    summary = {
        "input_file": str(args.input),
        "input_sha256": input_hash,
        "authentic_model_ready_rows": len(data),
        "symbols": data.symbol.nunique(),
        "model_ready_period": [str(data.trade_date.min().date()), str(data.trade_date.max().date())],
        "split": "chronological 70/15/15 independently within each symbol",
        "model": "canonical one-factor Heston; four train-only structural parameters plus date-state v0",
        "leakage_control": "whole strikes assigned to calibration or holdout; holdout call and put excluded from carry and fitting",
        "noise_control": "invalid expiries/quotes excluded and counted; no observed price altered or imputed",
        "test_holdout_metrics": test_metrics,
        "strict_checks_passed": int(checks.passed.sum()),
        "strict_checks_total": len(checks),
    }
    (args.output / "single_heston_run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
