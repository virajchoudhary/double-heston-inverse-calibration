"""NIFTY option surfaces from raw NSE F&O bhavcopies, on the grid the dual PINN needs.

Why NIFTY. The trained dual PINN consumes a fixed 45-vector: five expiries at
30/60/90/180/365 days by nine strikes at 0.85-1.15 moneyness, spot 1, r = 0.05,
q = 0.01. Every single-stock name in the NSE segment carries one or two expiries
inside 60 days, so that vector cannot be formed for them at all. NIFTY carries
11-12 expiries with material open interest out past 1300 days, which puts all five
targets strictly *inside* the traded ladder on every date used here.

Two data facts drive the implementation:

* `SttlmPric`, not `ClsPric`. Closing prices for the thinner strikes are stale, and
  put-call parity on them yields five-day discount factors near 0.86. Settlement
  prices give a coherent term structure (r about 6.3%, q about 0) and parity NRMSE
  of a few basis points.
* Open interest, not volume, decides whether a series is real. Some long-dated
  series carry an exchange-computed settlement price with exactly zero parity
  residual, zero volume and zero open interest; those are model output, not market
  data, and are dropped.

Forward and discount come per expiry from `single_heston.robust_carry` -- the
repository's own Huber-reweighted parity fit with its validity gates -- so this
module assumes nothing about the repo rate or dividend yield.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

PROJECT = Path("/Users/dhruvaambhaikar/Documents/Options pricing")
BHAV = PROJECT / "raw" / "nse_fo_bhavcopies"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from single_heston import robust_carry            # noqa: E402

SIG_LO, SIG_HI = 0.01, 4.0
MIN_OPEN_INTEREST = 10_000
GRID_DAYS = (30.0, 60.0, 90.0, 180.0, 365.0)
GRID_STRIKES = np.linspace(0.85, 1.15, 9)
CANON_RATE, CANON_CARRY = 0.05, 0.01


def black76(F, K, tau, sigma):
    if sigma <= 0.0 or tau <= 0.0:
        return max(F - K, 0.0)
    r = sigma * math.sqrt(tau)
    d1 = math.log(F / K) / r + 0.5 * r
    return F * norm.cdf(d1) - K * norm.cdf(d1 - r)


def implied_vol(fwd_price, F, K, tau):
    if not (max(F - K, 0.0) < fwd_price < F):
        return np.nan
    try:
        return brentq(lambda s: black76(F, K, tau, s) - fwd_price, SIG_LO, SIG_HI, xtol=1e-12)
    except (ValueError, RuntimeError):
        return np.nan


def read_quotes(date: str) -> pd.DataFrame:
    p = BHAV / date[:4] / f"BhavCopy_NSE_FO_0_0_0_{date.replace('-','')}_F_0000.csv.zip"
    d = pd.read_csv(p, usecols=["TradDt", "XpryDt", "TckrSymb", "OptnTp", "StrkPric",
                                "SttlmPric", "UndrlygPric", "TtlTradgVol", "OpnIntrst"])
    d = d[(d.TckrSymb == "NIFTY") & d.OptnTp.notna()].copy()
    d["dte"] = (pd.to_datetime(d.XpryDt) - pd.to_datetime(d.TradDt)).dt.days
    return d[(d.dte > 0) & (d.SttlmPric > 0)]


def surface(date: str, *, moneyness=(0.78, 1.22)) -> pd.DataFrame:
    """Every live quote on one date, with its expiry's forward, discount and implied vol."""
    d = read_quotes(date)
    spot = float(d.UndrlygPric.iloc[0])
    out = []
    for dte, g in d.groupby("dte"):
        if g.TtlTradgVol.fillna(0).sum() <= 0 or g.OpnIntrst.fillna(0).sum() < MIN_OPEN_INTEREST:
            continue                                   # exchange-computed series, not market
        c = g[g.OptnTp == "CE"].set_index("StrkPric").SttlmPric
        p = g[g.OptnTp == "PE"].set_index("StrkPric").SttlmPric
        k = c.index.intersection(p.index)
        if len(k) < 6:
            continue
        K = k.to_numpy(float)
        carry = robust_carry(K, (c[k] - p[k]).to_numpy(float), spot, dte / 365.0)
        if carry is None:
            continue
        F, df, rate, div, nrmse = carry
        tau = dte / 365.0
        q = g[(g.StrkPric >= moneyness[0] * F) & (g.StrkPric <= moneyness[1] * F)].copy()
        q = q[np.where(q.StrkPric >= F, q.OptnTp == "CE", q.OptnTp == "PE")]   # OTM side
        if len(q) < 4:
            continue
        Kq = q.StrkPric.to_numpy(float)
        fwd_call = np.where(q.OptnTp.to_numpy() == "CE", q.SttlmPric.to_numpy(float) / df,
                            q.SttlmPric.to_numpy(float) / df + F - Kq)
        iv = np.array([implied_vol(fwd_call[i], F, Kq[i], tau) for i in range(len(q))])
        out.append(pd.DataFrame({
            "trade_date": pd.Timestamp(date), "dte": dte, "tau": tau, "strike": Kq,
            "opt": q.OptnTp.to_numpy(), "forward": F, "discount": df, "rate": rate,
            "dividend": div, "parity_nrmse": nrmse, "spot": spot, "fwd_call": fwd_call,
            "x": np.log(F / Kq), "iv": iv,
            "volume": q.TtlTradgVol.fillna(0).to_numpy(float),
            "open_interest": q.OpnIntrst.fillna(0).to_numpy(float)}))
    if not out:
        return pd.DataFrame()
    s = pd.concat(out, ignore_index=True)
    return s[s.iv.notna() & (s.iv > SIG_LO * 1.01) & (s.iv < SIG_HI * 0.99)]


def fit_smile(g: pd.DataFrame):
    """Total variance w = iv^2 tau as a quadratic in log-forward-moneyness.

    Weighted by open interest, which is what says a quote is real. Returns a callable
    and the weighted RMS relative price residual, used as the network's noise input.
    """
    x, w = g.x.to_numpy(float), (g.iv.to_numpy(float) ** 2) * g.tau.to_numpy(float)
    wt = np.sqrt(np.maximum(g.open_interest.to_numpy(float), 1.0))
    deg = 2 if len(g) >= 6 else 1
    A = np.vander(x, deg + 1)
    coef, *_ = np.linalg.lstsq(A * wt[:, None], w * wt, rcond=None)
    f = lambda xx: np.maximum(np.polyval(coef, xx), 1e-8)
    tau = float(g.tau.iloc[0]); F = float(g.forward.iloc[0])
    model = np.array([black76(F, float(k), tau, math.sqrt(float(f(xx)) / tau))
                      for k, xx in zip(g.strike, x)])
    rel = (model - g.fwd_call.to_numpy(float)) / np.maximum(g.fwd_call.to_numpy(float), 1e-8)
    return f, float(np.sqrt(np.average(rel ** 2, weights=wt)))


def canonical_grid(s: pd.DataFrame, fit_mask: np.ndarray):
    """Build the dual PINN's 45-vector from the fit quotes only.

    Smile in x per expiry, then total variance interpolated linearly in tau between the
    two bracketing traded expiries -- the standard construction, and every target day is
    interior to the traded ladder, so nothing is extrapolated.
    """
    fit = s[fit_mask]
    smiles, noises = {}, []
    for dte, g in fit.groupby("dte"):
        if len(g) < 4:
            continue
        fn, nz = fit_smile(g)
        smiles[float(dte)] = fn
        noises.append(nz)
    days = np.array(sorted(smiles))
    prices, detail = [], []
    for d in GRID_DAYS:
        tau = d / 365.0
        Fc = math.exp((CANON_RATE - CANON_CARRY) * tau)
        below, above = days[days <= d], days[days >= d]
        if not len(below) or not len(above):
            return None, None, None
        lo, hi = float(below.max()), float(above.min())
        for K in GRID_STRIKES:
            xc = math.log(Fc / K)
            if lo == hi:
                w = float(smiles[lo](xc))
            else:                                    # linear in total variance vs tau
                wl, wh = float(smiles[lo](xc)), float(smiles[hi](xc))
                w = wl + (wh - wl) * (d - lo) / (hi - lo)
            iv = math.sqrt(max(w, 1e-10) / tau)
            prices.append(math.exp(-CANON_RATE * tau) * black76(Fc, float(K), tau, iv))
            detail.append({"days": d, "strike": float(K), "iv": iv,
                           "bracket_lo": lo, "bracket_hi": hi})
    return np.array(prices), pd.DataFrame(detail), float(np.median(noises))
