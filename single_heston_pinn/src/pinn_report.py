#!/usr/bin/env python3
"""Figures, surface reconstructions and the strict-check table for the PINN study."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pinn_heston_core as C
import pinn_calibrate as K
import pinn_run_training as R

ROOT = Path(__file__).resolve().parent
DEFAULT_DIR = ROOT / "outputs" / "pinn_single_heston"
MONTHS = {"1 month": 30, "2 month": 60, "3 month": 90}


# ------------------------------------------------------------------ figures

def plot_training(report, path):
    history = pd.DataFrame(report["history"])
    offset, offsets = 0, {}
    for r, group in history.groupby("round"):
        offsets[r] = offset
        offset += int(group.step.max()) + 1
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for r, group in history.groupby("round"):
        ax.semilogy(group.step + offsets[r], group.loss, label="round %d" % r, linewidth=1.3)
    ax.set_xlabel("training step")
    ax.set_ylabel("total loss (log scale)")
    ax.set_title("PINN training: PDE residual + smile anchor + no-arbitrage penalties")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_accuracy_vs_exact(model, box, path, n=30000, seed=4242):
    norm = C.Normaliser.from_box(box)
    pts = C.sample_anchor(n, box, seed=seed)
    tgt = C.anchor_targets(pts, g_limit=C.G_LIMIT)
    keep = tgt["usable"]
    import mlx.core as mx
    f = lambda a: mx.array(np.asarray(a, dtype=np.float32))
    g = np.asarray(model(C.build_features(
        f(pts["x"]), f(pts["variance"]), f(pts["maturity"]), f(pts["kappa"]),
        f(pts["theta"]), f(pts["sigma"]), f(pts["rho"]), norm)), dtype=np.float64)
    w_hat = pts["maturity"] * tgt["vbar"] * np.exp(2 * g)
    iv_hat = np.sqrt(w_hat / pts["maturity"])
    iv_true = np.sqrt(tgt["total_variance"] / pts["maturity"])
    traded = keep & (np.abs(pts["z"]) <= 3.0) & (pts["maturity"] * 365 >= 7)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    axes[0].scatter(iv_true[traded], iv_hat[traded], s=2, alpha=0.25, color="#1f4e79")
    lim = [0, float(np.nanpercentile(iv_true[traded], 99.5))]
    axes[0].plot(lim, lim, color="crimson", linewidth=1)
    axes[0].set(xlim=lim, ylim=lim, xlabel="exact Heston implied volatility",
                ylabel="PINN implied volatility", title="Traded region (|z| <= 3, T >= 7d)")
    axes[0].grid(alpha=0.3)

    err = (iv_hat - iv_true)
    axes[1].hist(err[traded], bins=90, color="#1f4e79", alpha=0.85)
    axes[1].set(xlabel="PINN minus exact implied volatility", ylabel="count",
                title="Error, traded region\nRMSE %.5f" % np.sqrt(np.mean(err[traded] ** 2)))
    axes[1].grid(alpha=0.3)

    sel = keep
    sc = axes[2].scatter(pts["z"][sel], pts["maturity"][sel] * 365,
                         c=np.abs(err[sel]), s=3, cmap="viridis",
                         vmin=0, vmax=float(np.nanpercentile(np.abs(err[sel]), 98)))
    axes[2].set(xlabel="standardised moneyness z", ylabel="days to expiry",
                title="Absolute implied-volatility error")
    plt.colorbar(sc, ax=axes[2])
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return {"traded_iv_rmse": float(np.sqrt(np.mean(err[traded] ** 2))),
            "traded_points": int(traded.sum())}


def reconstruct_surface(engine, params, spot_grid, strike, maturity, rate, dividend):
    n = spot_grid.size
    return engine.price_rowwise(
        spot_grid, np.full(n, strike), np.full(n, maturity), np.full(n, rate),
        np.full(n, dividend), np.ones(n, dtype=bool),
        *(np.full(n, float(p)) for p in params))


def plot_requested_domain(pinn, fourier, params, symbol, ceiling, strike, path):
    """The study's own domain: S from 0 to 1.5 x the ten-year maximum traded price."""
    spot = np.linspace(0.0, ceiling, 420)
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8), sharex=True)
    rows = []
    for j, (label, days) in enumerate(MONTHS.items()):
        maturity = days / 365.0
        a = reconstruct_surface(pinn, params, spot, strike, maturity, 0.07, 0.02)
        b = reconstruct_surface(fourier, params, spot, strike, maturity, 0.07, 0.02)
        axes[0, j].plot(spot, b, color="#111111", linewidth=2.0, label="exact Heston")
        axes[0, j].plot(spot, a, color="#e07b39", linewidth=1.3, linestyle="--", label="PINN")
        axes[0, j].plot(spot, np.maximum(spot - strike, 0), color="#999999",
                        linewidth=0.9, label="intrinsic")
        axes[0, j].set_title("%s  (T = %d/365 yr)" % (label, days))
        axes[0, j].grid(alpha=0.3)
        if j == 0:
            axes[0, j].set_ylabel("call price (Rs)")
            axes[0, j].legend(fontsize=8)
        axes[1, j].plot(spot, a - b, color="#1f4e79", linewidth=1.1)
        axes[1, j].axhline(0, color="#999999", linewidth=0.8)
        axes[1, j].set_xlabel("spot S (Rs);  0 to 1.5 x ten-year maximum = %.0f" % ceiling)
        axes[1, j].grid(alpha=0.3)
        if j == 0:
            axes[1, j].set_ylabel("PINN minus exact (Rs)")
        rows.append({"symbol": symbol, "slice": label, "days_to_expiry": days,
                     "maturity_years": maturity, "strike": strike,
                     "spot_low": 0.0, "spot_high": ceiling,
                     "max_abs_price_gap": float(np.max(np.abs(a - b))),
                     "max_abs_price_gap_pct_of_strike": float(100 * np.max(np.abs(a - b)) / strike),
                     "rmse_price_gap": float(np.sqrt(np.mean((a - b) ** 2)))})
    fig.suptitle("%s: reconstructed call surface on the specified domain "
                 "(kappa=%.2f theta=%.3f sigma=%.2f rho=%+.2f v0=%.3f)"
                 % (symbol, *params), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(path, dpi=150); plt.close(fig)
    return pd.DataFrame(rows)


def plot_market_fit(predictions, path):
    fig, axes = plt.subplots(1, len(predictions), figsize=(6.2 * len(predictions), 5), squeeze=False)
    for i, (name, frame) in enumerate(predictions.items()):
        test = frame[(frame.split == "test") & (frame.fold == "holdout")].dropna(
            subset=["market_iv", "model_iv"])
        ax = axes[0, i]
        ax.scatter(test.market_iv, test.model_iv, s=6, alpha=0.35, color="#1f4e79")
        lim = [0, float(np.nanpercentile(test.market_iv, 99.5)) * 1.05]
        ax.plot(lim, lim, color="crimson", linewidth=1)
        m = K.metrics(test)
        ax.set(xlim=lim, ylim=lim, xlabel="market implied volatility (NSE close)",
               ylabel="model implied volatility",
               title="%s engine, test holdout\nrows %d  IV RMSE %.4f  R2 %.3f"
                     % (name, m["rows"], m["iv_rmse"], m["iv_r2"]))
        ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_collocation(collocation, path):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    frac = collocation["spot"] / collocation["spot_ceiling"]
    sc = axes[0].scatter(frac, collocation["maturity_days"], c=np.sqrt(collocation["variance"]),
                         s=3, cmap="viridis", alpha=0.6)
    axes[0].set(xlabel="spot as a fraction of 1.5 x ten-year maximum",
                ylabel="days to expiry",
                title="%d collocation points on the specified domain" % collocation["x"].size)
    plt.colorbar(sc, ax=axes[0], label="instantaneous volatility sqrt(v)")
    axes[1].hist(collocation["maturity_days"], bins=92, color="#1f4e79")
    for d in C.MONTH_SLICE_DAYS:
        axes[1].axvline(d, color="crimson", linewidth=0.9, linestyle="--")
    axes[1].set(xlabel="days to expiry", ylabel="collocation points",
                title="Maturity axis: continuous fill plus 1M / 2M / 3M slices")
    axes[2].hist(np.sqrt(collocation["variance"]), bins=70, color="#1f4e79")
    axes[2].set(xlabel="sqrt(v) = volatility state", ylabel="collocation points",
                title="Variance axis, taken from inverse-BSM implied volatility")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_state_paths(states, path):
    symbols = sorted(states.symbol.unique())[:6]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 7.5), sharey=False)
    for ax, symbol in zip(axes.ravel(), symbols):
        sub = states[states.symbol == symbol].sort_values("trade_date")
        for engine, colour in (("pinn", "#e07b39"), ("fourier", "#111111")):
            part = sub[sub.engine == engine]
            if len(part):
                ax.plot(pd.to_datetime(part.trade_date), part.spot_vol, label=engine,
                        linewidth=1.2, color=colour)
        ax.set_title(symbol, fontsize=10)
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Calibrated Heston spot volatility sqrt(v0) per trade date", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(path, dpi=150); plt.close(fig)
