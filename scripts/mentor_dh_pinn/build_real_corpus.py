#!/usr/bin/env python3
"""Assemble a corpus of REAL option surfaces for self-supervised projection fine-tuning.

No labels are produced or needed. The projection objective is fit error against observed
quotes, which the differentiable exact pricer computes without knowing any p*.

Leakage control, which matters more than usual because the evaluation sets are small:
  * the stock panel carries its own chronological split; only `split == "train"` is used;
  * every ADANIPOWER date used for testing lives in `split == "test"` and is therefore
    excluded automatically;
  * NIFTY training dates exclude the 20 dates reserved as the NIFTY test and validation
    windows in outputs/nifty_selection.json and the ladder scan.

Everything is stored in the forward measure (spot = F, r = q = 0), which is exact to
2.4e-15 against the direct form and removes the panel's parity-implied rates -- which range
from -9.8% to +20.7% -- from the objective.
"""
from __future__ import annotations

import argparse, json, math, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.mentor_dh_pinn.nifty_panel import black76, implied_vol, surface as nifty_surface

PANEL = Path("/Users/dhruvaambhaikar/Documents/Options pricing/outputs/"
             "pinn_single_heston/pinn_quote_panel.parquet")
SCAN = Path("/Users/dhruvaambhaikar/Documents/Options pricing/outputs/nifty_ladder_scan.csv")
MAX_QUOTES = 100
MIN_QUOTES = 6


def _vega(fwd, strike, tau, iv):
    tau = np.maximum(tau, 1e-6); root = np.maximum(iv, 1e-4) * np.sqrt(tau)
    d1 = np.log(np.maximum(fwd, 1e-12) / np.maximum(strike, 1e-12)) / root + 0.5 * root
    return fwd * np.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi) * np.sqrt(tau)


def _pack(fwd, strike, tau, price, iv, label):
    """One surface, spot-normalised so scales are comparable across underlyings."""
    n = len(tau)
    if n < MIN_QUOTES:
        return None
    if n > MAX_QUOTES:                       # keep the most liquid centre of the surface
        keep = np.argsort(np.abs(np.log(fwd / strike)))[:MAX_QUOTES]; keep.sort()
        fwd, strike, tau, price, iv = (a[keep] for a in (fwd, strike, tau, price, iv))
        n = MAX_QUOTES
    s = float(np.median(fwd))
    ok = np.isfinite(iv) & (price > 1e-8 * s) & (price < fwd)
    if ok.sum() < MIN_QUOTES:
        return None
    fwd, strike, tau, price, iv = (a[ok] for a in (fwd, strike, tau, price, iv))
    veg = _vega(fwd, strike, tau, iv)
    # quote-noise scale: dispersion of implied vol about a smooth quadratic smile, mapped
    # back to price units through vega. NSE publishes no bid-ask, so this is the honest
    # available estimate rather than a spread.
    x = np.log(fwd / strike)
    sig = np.full(len(x), np.nan)
    for t in np.unique(tau):
        m = tau == t
        if m.sum() >= 4:
            c = np.polyfit(x[m], iv[m], min(2, m.sum() - 1))
            sig[m] = iv[m] - np.polyval(c, x[m])
        else:
            sig[m] = iv[m] - iv[m].mean()
    iv_noise = float(np.clip(1.4826 * np.median(np.abs(sig[np.isfinite(sig)])), 2e-3, 0.15))
    return {"label": label, "n": len(tau), "spot": fwd / s, "strike": strike / s,
            "tau": tau, "price": price / s, "iv": iv, "vega": np.maximum(veg / s, 1e-8),
            "quote_sigma": np.maximum(iv_noise * veg / s, 1e-8),
            "rate": np.zeros(len(tau)), "carry": np.zeros(len(tau)),
            "iv_noise": iv_noise, "n_expiries": int(len(np.unique(tau)))}


def stock_surfaces(split: str) -> list[dict]:
    p = pd.read_parquet(PANEL)
    p = p[p.split == split]
    out = []
    for (sym, date), g in p.groupby(["symbol", "trade_date"]):
        g = g.copy()
        fwd_price = g.market_price_adjusted / g.discount_factor
        put = ~g.is_call
        fwd_price = fwd_price + np.where(put, g.forward - g.strike, 0.0)   # forward parity
        rec = _pack(g.forward.to_numpy(float), g.strike.to_numpy(float),
                    g.maturity.to_numpy(float), fwd_price.to_numpy(float),
                    g.market_iv.to_numpy(float), f"{sym}|{date.date()}")
        if rec: out.append(rec)
    return out


def nifty_surfaces(dates: list[str]) -> list[dict]:
    out = []
    for d in dates:
        try:
            s = nifty_surface(d)
        except Exception:
            continue
        if s.empty: continue
        rec = _pack(s.forward.to_numpy(float), s.strike.to_numpy(float),
                    s.tau.to_numpy(float), s.fwd_call.to_numpy(float),
                    s.iv.to_numpy(float), f"NIFTY|{d}")
        if rec: out.append(rec)
    return out


def save(recs: list[dict], path: Path) -> None:
    n = len(recs); m = max(r["n"] for r in recs)
    pad = lambda k: np.stack([np.pad(r[k], (0, m - r["n"]), constant_values=r[k][-1])
                              for r in recs]).astype(np.float64)
    d = {k: pad(k) for k in ("spot", "strike", "tau", "rate", "carry", "price",
                             "iv", "vega", "quote_sigma")}
    d["mask"] = np.stack([np.pad(np.ones(r["n"]), (0, m - r["n"])) for r in recs])
    d["n_quotes"] = np.array([r["n"] for r in recs])
    d["iv_noise"] = np.array([r["iv_noise"] for r in recs])
    d["n_expiries"] = np.array([r["n_expiries"] for r in recs])
    d["label"] = np.array([r["label"] for r in recs])
    np.savez_compressed(path, **d)
    print(f"  {path.name}: {n} surfaces, {int(d['n_quotes'].sum()):,} quotes, "
          f"{d['n_quotes'].min()}-{d['n_quotes'].max()} per surface, "
          f"median {int(np.median(d['n_quotes']))}, "
          f"expiries {d['n_expiries'].min()}-{d['n_expiries'].max()}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "real_corpus")
    ap.add_argument("--nifty-train", type=int, default=60)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)

    held = set(json.loads(Path("/Users/dhruvaambhaikar/Documents/Options pricing/"
                               "outputs/nifty_selection.json").read_text())["dates"])
    scan = pd.read_csv(SCAN, parse_dates=["date"])
    scan = scan[scan.covers_ladder & scan.rv21.notna()]
    held |= {d.strftime("%Y-%m-%d") for d in scan.nlargest(20, "rv21").date}   # test + val
    pool = [d.strftime("%Y-%m-%d") for d in sorted(scan.date) if d.strftime("%Y-%m-%d") not in held]
    step = max(1, len(pool) // a.nifty_train)
    nifty_train = pool[::step][:a.nifty_train]
    print(f"NIFTY: {len(pool)} eligible dates, {len(nifty_train)} sampled for training; "
          f"{len(held)} dates withheld", flush=True)

    print("building training corpus...", flush=True)
    train = stock_surfaces("train") + nifty_surfaces(nifty_train)
    save(train, a.out / "real_train.npz")
    print("building validation corpus...", flush=True)
    val = stock_surfaces("validation")
    save(val, a.out / "real_validation.npz")


if __name__ == "__main__":
    main()
