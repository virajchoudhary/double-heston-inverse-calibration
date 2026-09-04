"""Noise-augmented surface dataset for the two specialist inverse networks.

Why noise augmentation is the point, not a detail. Classical least-squares
calibration has no shrinkage: at 0.5% quote noise the ten-parameter fit absorbs
the noise and generalises worse than a five-parameter one -- measured, at every
geometry tested. A network trained on *noisy* surfaces learns the conditional
mean of the parameters given a noisy surface, which is shrunk toward the prior
by construction. That is the mechanism by which an inverse network can beat
classical calibration under noise, and it is the reason each sample here carries
its own randomly drawn noise level rather than a single fixed one.

Scale follows the deep-calibration literature: Bayer & Stemper (2018) drew about
one million parameter combinations for the rough-Bergomi pricing map, and Liu,
Borovykh, Grzelak & Oosterlee (2019) applied the same scale to Heston and Bates.
Those counts are per *price*; the sample unit here is a whole surface of 45
quotes, so the surface count is correspondingly smaller for the same pricing
budget.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.double_heston import price_double_heston_call
from .model_v4 import PARAM_BOX
from .data_v4 import load_panel, _valid

# Expiry ladder: three NSE stock-cycle expiries plus two index-style long dates,
# because the sensitivity study shows theta and kappa are only visible past 180 days.
SHORT_DAYS = (30.0, 60.0, 90.0)
LONG_DAYS = (180.0, 365.0)
ALL_DAYS = SHORT_DAYS + LONG_DAYS
STRIKES = np.linspace(0.85, 1.15, 9)
RATE, CARRY, SPOT = 0.05, 0.01, 1.0

SHORT_TARGETS = ("v0_slow", "v0_fast", "sigma_slow", "sigma_fast", "rho_slow", "rho_fast")
LONG_TARGETS = ("theta_slow", "theta_fast", "kappa_slow", "kappa_fast")
CANONICAL = ("kappa_slow", "theta_slow", "sigma_slow", "rho_slow", "v0_slow",
             "kappa_fast", "theta_fast", "sigma_fast", "rho_fast", "v0_fast")
V0_BOX = (0.01, 0.60)
NOISE_MAX = 0.01          # up to 1% multiplicative quote noise


def draw_truths(rng, n: int, panel: pd.DataFrame, panel_fraction: float = 0.6) -> dict:
    """Structural parameters from the sealed panel plus a fresh in-box component."""
    n_panel = int(round(panel_fraction * n))
    rows = panel.iloc[rng.integers(0, len(panel), n_panel)]
    keys = ("kappa_slow", "theta_slow", "sigma_slow", "rho_slow",
            "kappa_fast", "theta_fast", "sigma_fast", "rho_fast")
    out = {k: [rows[k].to_numpy(float)] for k in keys}
    need = n - n_panel
    while need > 0:
        m = int(max(1024, 1.6 * need))
        cand = {}
        for k in keys:
            lo, hi = PARAM_BOX[k]
            cand[k] = (rng.uniform(lo, hi, m) if k.startswith("rho")
                       else np.exp(rng.uniform(math.log(lo), math.log(hi), m)))
        ok = _valid(cand)
        take = min(need, int(ok.sum()))
        for k in keys:
            out[k].append(cand[k][ok][:take])
        need -= take
    truths = {k: np.concatenate(v)[:n] for k, v in out.items()}
    for tag in ("slow", "fast"):
        truths[f"v0_{tag}"] = np.exp(rng.uniform(math.log(V0_BOX[0]), math.log(V0_BOX[1]), n))
    return truths


def price_surface(truth_row) -> np.ndarray:
    """45 calls: 5 expiries x 9 strikes, spot normalised to 1."""
    out = np.empty(len(ALL_DAYS) * len(STRIKES))
    i = 0
    p = [float(truth_row[k]) for k in CANONICAL]
    for d in ALL_DAYS:
        tau = d / 365.0
        for k in STRIKES:
            out[i] = price_double_heston_call(SPOT, float(k), tau, RATE, CARRY, p, node_count=64)
            i += 1
    return out


def build(n: int, seed: int, panel: pd.DataFrame, *, noise_max: float = NOISE_MAX) -> dict:
    """Clean surfaces, per-sample noise level, and the noisy surface the nets see."""
    rng = np.random.default_rng(seed)
    truths = draw_truths(rng, n, panel)
    q = len(ALL_DAYS) * len(STRIKES)
    clean = np.empty((n, q))
    for i in range(n):
        clean[i] = price_surface({k: truths[k][i] for k in CANONICAL})
    # each surface gets its own noise level, so the network learns a map that is
    # correct across the whole realistic noise range rather than at one point
    level = rng.uniform(0.0, noise_max, n)
    shock = rng.normal(0.0, 1.0, (n, q)) * level[:, None]
    noisy = clean * np.exp(shock - 0.5 * level[:, None] ** 2)
    return {"clean": clean, "noisy": noisy, "noise_level": level,
            **{k: truths[k] for k in CANONICAL}}


def save(path: Path, data: dict) -> None:
    np.savez_compressed(path, **data)
