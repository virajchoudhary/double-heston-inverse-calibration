"""Training data for the parameter-conditioned V4 pricer.

Parameters come from the sealed 10,000-vector panel -- the project's own frozen
parameter population -- plus a fresh component drawn inside the padded box so the
network is not restricted to the panel's exact points.  For every drawn vector
the variance states are taken from that vector's own CIR stationary law, and the
same reference-engine admissibility rule applies as in V2/V3.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.double_heston import price_double_heston_call
from .model_v4 import PARAM_BOX, PARAM_NAMES

PANEL = Path("evidence/final_r2_candidate_pool_readiness_20260822/final_parameter_panel.csv")
MONTH_SLICE_DAYS = (30.0, 60.0, 90.0)
STATIONARY_MASS = 0.998
MIN_TOTAL_SD = 0.04
MAX_ABS_Z = 6.0


def load_panel(path: Path = PANEL) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def _valid(p: dict) -> np.ndarray:
    """Canonical structural constraints, applied elementwise."""
    ok = np.ones(len(p["kappa_slow"]), dtype=bool)
    ok &= p["kappa_slow"] < p["kappa_fast"]
    ok &= 2 * p["kappa_slow"] * p["theta_slow"] - p["sigma_slow"] ** 2 > 0
    ok &= 2 * p["kappa_fast"] * p["theta_fast"] - p["sigma_fast"] ** 2 > 0
    ok &= np.abs(p["rho_slow"]) < 1
    ok &= np.abs(p["rho_fast"]) < 1
    ok &= p["rho_slow"] ** 2 + p["rho_fast"] ** 2 < 1
    return ok


def draw_parameters(rng, n: int, panel: pd.DataFrame, panel_fraction: float = 0.70) -> dict:
    n_panel = int(round(panel_fraction * n))
    rows = panel.iloc[rng.integers(0, len(panel), n_panel)]
    out = {k: [rows[k].to_numpy(float)] for k in PARAM_NAMES}
    need = n - n_panel
    while need > 0:
        m = int(max(512, 1.6 * need))
        cand = {}
        for k in PARAM_NAMES:
            lo, hi = PARAM_BOX[k]
            cand[k] = (rng.uniform(lo, hi, m) if k.startswith("rho")
                       else np.exp(rng.uniform(math.log(lo), math.log(hi), m)))
        ok = _valid(cand)
        take = min(need, int(ok.sum()))
        for k in PARAM_NAMES:
            out[k].append(cand[k][ok][:take])
        need -= take
    return {k: np.concatenate(v)[:n] for k, v in out.items()}


def stationary_bounds(kappa, theta, sigma, mass: float = STATIONARY_MASS):
    shape = 2.0 * kappa * theta / sigma ** 2
    scale = sigma ** 2 / (2.0 * kappa)
    tail = 0.5 * (1.0 - mass)
    return (stats.gamma.ppf(tail, shape, scale=scale),
            stats.gamma.ppf(1.0 - tail, shape, scale=scale), shape, scale)


def draw_states(rng, p: dict, stationary_fraction: float = 0.65):
    """Variance states from each drawn vector's own stationary law."""
    n = len(p["kappa_slow"])
    states = {}
    for tag in ("slow", "fast"):
        lo, hi, shape, scale = stationary_bounds(p[f"kappa_{tag}"], p[f"theta_{tag}"], p[f"sigma_{tag}"])
        clo = stats.gamma.cdf(lo, shape, scale=scale)
        chi = stats.gamma.cdf(hi, shape, scale=scale)
        u = rng.uniform(clo, chi, n)
        stat = stats.gamma.ppf(u, shape, scale=scale)
        cover = np.exp(rng.uniform(np.log(lo), np.log(hi)))
        use_stat = rng.random(n) < stationary_fraction
        states[tag] = np.where(use_stat, stat, cover)
        states[f"{tag}_low"], states[f"{tag}_high"] = lo, hi
    return states


def mean_reversion_ratio(u):
    return np.where(u < 0.05,
                    1 - u / 2 + u * u / 6 - u ** 3 / 24 + u ** 4 / 120,
                    (1 - np.exp(-np.where(u < 0.05, 1.0, u))) / np.where(u < 0.05, 1.0, u))


def build(n: int, seed: int, panel: pd.DataFrame, *, tau_low=7 / 365, tau_high=92 / 365,
          strike_low=0.70, strike_high=1.30, rate_low=0.01, rate_high=0.08,
          carry_low=0.0, carry_high=0.03) -> dict:
    rng = np.random.default_rng(seed)
    keep = {k: [] for k in list(PARAM_NAMES) + ["v_slow", "v_fast", "tau", "spot", "strike",
                                                 "rate", "carry", "z"]}
    have = 0
    while have < n:
        m = int(max(4096, 1.6 * (n - have)))
        p = draw_parameters(rng, m, panel)
        st = draw_states(rng, p)
        vs, vf = st["slow"], st["fast"]
        tau = np.exp(rng.uniform(math.log(tau_low), math.log(tau_high), m))
        on = rng.random(m) < 0.33
        tau = np.where(on, np.array(MONTH_SLICE_DAYS)[rng.integers(0, 3, m)] / 365.0, tau)
        vbar = (p["theta_slow"] + (vs - p["theta_slow"]) * mean_reversion_ratio(p["kappa_slow"] * tau)
                + p["theta_fast"] + (vf - p["theta_fast"]) * mean_reversion_ratio(p["kappa_fast"] * tau))
        sd = np.sqrt(vbar * tau)
        z = np.where(rng.random(m) < 0.72, rng.normal(0, 1.8, m), rng.uniform(-MAX_ABS_Z, MAX_ABS_Z, m))
        z = np.clip(z, -MAX_ABS_Z, MAX_ABS_Z)
        x = z * sd
        strike = np.exp(rng.uniform(math.log(strike_low), math.log(strike_high), m))
        rate = rng.uniform(rate_low, rate_high, m)
        carry = rng.uniform(carry_low, carry_high, m)
        spot = strike * np.exp(x - (rate - carry) * tau)
        ok = (sd >= MIN_TOTAL_SD) & (np.abs(z) <= MAX_ABS_Z) & (spot > 0) & np.isfinite(spot)
        for k in PARAM_NAMES:
            keep[k].append(p[k][ok])
        for k, v in (("v_slow", vs), ("v_fast", vf), ("tau", tau), ("spot", spot),
                     ("strike", strike), ("rate", rate), ("carry", carry), ("z", z)):
            keep[k].append(v[ok])
        have += int(ok.sum())
    cols = {k: np.concatenate(v)[:n] for k, v in keep.items()}
    price = np.empty(n)
    for i in range(n):
        price[i] = price_double_heston_call(
            float(cols["spot"][i]), float(cols["strike"][i]), float(cols["tau"][i]),
            float(cols["rate"][i]), float(cols["carry"][i]),
            (cols["kappa_slow"][i], cols["theta_slow"][i], cols["sigma_slow"][i],
             cols["rho_slow"][i], cols["v_slow"][i],
             cols["kappa_fast"][i], cols["theta_fast"][i], cols["sigma_fast"][i],
             cols["rho_fast"][i], cols["v_fast"][i]), node_count=64)
    cols["price"] = price
    return cols
