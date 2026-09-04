"""Variable-geometry, regime-balanced, richly-noised synthetic surfaces.

Three defects of the previous training distribution are addressed here.

Geometry.  The old data was always 5 expiries x 9 strikes at spot 1, r = 0.05, q = 0.01.
That made the 45-vector an architectural assumption rather than a property of one market,
and it is why a single-expiry ADANIPOWER date could not be ingested at all. Here every
surface has its own randomly drawn geometry: number of expiries, the maturities themselves,
the number and placement of strikes, the spot, the rate and the carry. The historical
5 x 9 layout is retained as *one* sampled geometry among many, for backwards comparison.

Prior.  The old prior had median total instantaneous volatility near 48% -- built for
volatile power-sector stocks. NIFTY at 17.4% sat at the 1.8th percentile of its v0 marginal
and the 0.23rd of its theta marginal. Here the prior is an explicit mixture over four
regimes in *transformed* coordinates (total volatility, factor split, mean-reversion
timescale, Feller ratio, correlation), so index-like volatility is a first-class region
rather than a tail.

Noise.  The old noise was iid multiplicative lognormal with scale U(0, 1%). Real quote
error is correlated along maturity and strike, heteroskedastic, heavy-tailed, and larger
than 1%: the NIFTY panel's own estimate exceeded 1% on every date. Here noise is a sum of
a surface-wide level, a maturity-correlated term, a smooth strike-correlated term and a
heavy-tailed idiosyncratic term, with occasional outliers and random quote deletion.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .params_v2 import decode, encode, to_array
from .torch_pricer import price_call

MAX_QUOTES = 100
HISTORICAL_DAYS = (30.0, 60.0, 90.0, 180.0, 365.0)
HISTORICAL_STRIKES = np.linspace(0.85, 1.15, 9)

# Four volatility regimes, as total instantaneous volatility sqrt(v0_slow + v0_fast).
# The index regime exists because NIFTY sits at 17.4%; the old prior effectively excluded it.
REGIMES = (
    ("index",    0.13, 0.24, 0.30),      # low-volatility equity index
    ("ordinary", 0.20, 0.40, 0.30),      # ordinary single stock
    ("high",     0.35, 0.75, 0.28),      # high-volatility stock
    ("stress",   0.60, 1.30, 0.12),      # stress regime
)


def sample_parameters(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Regime-balanced draw in transformed coordinates. Returns (params, regime_index)."""
    names = [r[0] for r in REGIMES]
    probs = np.array([r[3] for r in REGIMES], dtype=float); probs /= probs.sum()
    which = rng.choice(len(REGIMES), size=n, p=probs)
    z = np.zeros((n, 10))
    for i in range(n):
        _, lo, hi, _ = REGIMES[which[i]]
        vol_now = math.exp(rng.uniform(math.log(lo), math.log(hi)))          # sqrt(v0_tot)
        # long-run level is anchored to the current level but free to sit either side
        vol_run = vol_now * math.exp(rng.normal(0.0, 0.35))
        vol_run = min(max(vol_run, 0.05), 2.0)
        z[i, 4] = math.log(vol_now ** 2)                                     # log v0_total
        z[i, 2] = math.log(vol_run ** 2)                                     # log theta_total
        z[i, 5] = rng.normal(0.0, 1.2)                                       # v0 split
        z[i, 3] = rng.normal(0.0, 1.2)                                       # theta split
        # mean-reversion timescales in log space: slow 0.5-8 years, fast 0.02-0.8 years
        ks = 1.0 / math.exp(rng.uniform(math.log(0.5), math.log(8.0)))
        kf = 1.0 / math.exp(rng.uniform(math.log(0.02), math.log(0.8)))
        kf = max(kf, ks * (1.0 + 1e-3))
        z[i, 0] = math.log(ks)
        z[i, 1] = math.log(max(kf / ks - 1.0, 1e-9))
        z[i, 6] = rng.normal(0.0, 1.3)                                       # Feller ratio slow
        z[i, 7] = rng.normal(0.0, 1.3)                                       # Feller ratio fast
        # equity correlation: skewed negative, occasionally positive
        ang = rng.normal(math.pi, 0.75) if rng.random() < 0.85 else rng.uniform(0, 2 * math.pi)
        rad = math.atanh(min(rng.beta(2.0, 2.2), 1 - 1e-9))
        z[i, 8], z[i, 9] = rad * math.cos(ang), rad * math.sin(ang)
    params = np.stack([to_array(decode(z[i])) for i in range(n)])
    return params, which


def sample_geometry(rng: np.random.Generator) -> dict:
    """One market geometry. Returns spot/strike/tau/rate/carry arrays of common length."""
    kind = rng.random()
    spot = 1.0
    rate = rng.uniform(0.0, 0.10)
    carry = rng.uniform(0.0, 0.06)
    if kind < 0.10:                                   # the historical layout, kept for comparison
        days = np.array(HISTORICAL_DAYS)
        strikes = [HISTORICAL_STRIKES] * len(days)
        tag = "historical_5x9"
    else:
        if kind < 0.35:                               # single expiry, deliberately oversampled
            n_exp, tag = 1, "single_expiry"
        elif kind < 0.50:
            n_exp, tag = 2, "two_expiry"
        elif kind < 0.80:
            n_exp, tag = int(rng.integers(3, 6)), "multi_expiry"
        else:
            n_exp, tag = int(rng.integers(6, 9)), "dense_expiry"
        days = np.exp(rng.uniform(math.log(7.0), math.log(1095.0), n_exp))
        days = np.sort(days)
        strikes = []
        for _ in range(n_exp):
            n_k = int(rng.integers(3, 16))
            width = rng.uniform(0.06, 0.45)           # half-width in log-forward-moneyness
            xs = np.sort(rng.uniform(-width, width, n_k))
            strikes.append(np.exp(-xs))               # K/F; converted to K below
    tau = np.concatenate([np.full(len(k), d / 365.0) for d, k in zip(days, strikes)])
    fwd = spot * np.exp((rate - carry) * tau)
    kk = np.concatenate([np.asarray(k, dtype=float) for k in strikes])
    strike = kk * fwd if kind >= 0.10 else kk * spot   # historical grid is spot-normalised
    n = len(tau)
    if n > MAX_QUOTES:
        keep = rng.choice(n, MAX_QUOTES, replace=False); keep.sort()
        tau, strike = tau[keep], strike[keep]; n = MAX_QUOTES
    return {"spot": np.full(n, spot), "strike": strike, "tau": tau,
            "rate": np.full(n, rate), "carry": np.full(n, carry), "tag": tag}


def expected_total_variance(params: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """E[(1/tau) int_0^tau v_s ds] summed over both factors: theta + (v-theta)(1-e^{-k t})/(k t)."""
    out = np.zeros_like(tau, dtype=float)
    for k, th, v in ((params[0], params[1], params[4]), (params[5], params[6], params[9])):
        kt = np.maximum(k * tau, 1e-12)
        out = out + th + (v - th) * (1.0 - np.exp(-kt)) / kt
    return out


def admissible(params: np.ndarray, geo: dict, clean: np.ndarray) -> np.ndarray:
    """Where the 64-node Fourier engine can be trusted to supervise.

    Two conditions, both measured earlier in this study: enough total standard deviation for
    the quadrature to resolve time value, and a standardised moneyness within a few standard
    deviations so the price is not numerically its own no-arbitrage bound. Without this the
    generator emits quotes worth ~1e-16 whose Fourier error is larger than the price itself,
    and which no market would quote either.
    """
    tau = geo["tau"]
    vbar = expected_total_variance(params, tau)
    total_sd = np.sqrt(np.maximum(vbar * tau, 0.0))
    fwd = geo["spot"] * np.exp((geo["rate"] - geo["carry"]) * tau)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.log(fwd / geo["strike"]) / np.maximum(total_sd, 1e-300)
    intrinsic = np.maximum(fwd - geo["strike"], 0.0) * np.exp(-geo["rate"] * tau)
    return (np.isfinite(clean) & (total_sd >= 0.02) & (np.abs(z) <= 6.0)
            & (clean >= intrinsic - 1e-10) & (clean > 1e-8 * geo["spot"]))


def apply_noise(rng: np.random.Generator, clean: np.ndarray, geo: dict) -> tuple[np.ndarray, float]:
    """Correlated, heteroskedastic, heavy-tailed multiplicative noise with outliers."""
    tau, strike, spot = geo["tau"], geo["strike"], geo["spot"]
    level = float(np.exp(rng.uniform(math.log(0.0015), math.log(0.06))))   # 0.15% .. 6%
    x = np.log(strike / (spot * np.exp((geo["rate"] - geo["carry"]) * tau)))
    e_surface = rng.normal(0.0, 1.0)
    uniq = np.unique(tau)
    per_tau = {t: rng.normal(0.0, 1.0) for t in uniq}
    e_tau = np.array([per_tau[t] for t in tau])
    a, b = rng.normal(0.0, 1.0, 2)                                          # smooth in strike
    e_strike = a * x / (np.abs(x).max() + 1e-9) + b * (x / (np.abs(x).max() + 1e-9)) ** 2
    e_idio = rng.standard_t(df=3.5, size=len(tau)) / math.sqrt(3.5 / 1.5)   # heavy tailed
    hetero = 1.0 + 1.5 * np.abs(x) / (np.abs(x).max() + 1e-9)               # wings noisier
    shock = level * hetero * (0.45 * e_surface + 0.45 * e_tau + 0.45 * e_strike + 0.8 * e_idio)
    if rng.random() < 0.25:                                                 # occasional outliers
        k = int(rng.integers(1, max(2, len(tau) // 10)))
        idx = rng.choice(len(tau), k, replace=False)
        shock[idx] += level * rng.normal(0.0, 6.0, k)
    noisy = clean * np.exp(shock - 0.5 * (level ** 2))
    return np.maximum(noisy, 1e-12), level


def price_batch(params: np.ndarray, geos: list[dict], node_count: int = 64) -> list[np.ndarray]:
    """Exact prices for a list of variable-length geometries, batched by padding."""
    n = len(geos)
    m = max(len(g["tau"]) for g in geos)
    pad = lambda key: np.stack([np.pad(g[key], (0, m - len(g[key])),
                                       constant_values=g[key][-1]) for g in geos])
    with torch.no_grad():
        out = price_call(torch.tensor(params), torch.tensor(pad("spot")),
                         torch.tensor(pad("strike")), torch.tensor(pad("tau")),
                         torch.tensor(pad("rate")), torch.tensor(pad("carry")),
                         node_count=node_count).numpy()
    return [out[i, :len(geos[i]["tau"])] for i in range(n)]


def build(n: int, seed: int, *, node_count: int = 64, chunk: int = 256) -> dict:
    """Generate `n` surfaces. Stored padded, with an explicit length and mask."""
    rng = np.random.default_rng(seed)
    params, regime = sample_parameters(rng, n)
    P = np.zeros((n, MAX_QUOTES)); C = np.zeros((n, MAX_QUOTES)); Y = np.zeros((n, MAX_QUOTES))
    S = np.zeros((n, MAX_QUOTES)); K = np.zeros((n, MAX_QUOTES)); T = np.zeros((n, MAX_QUOTES))
    R = np.zeros((n, MAX_QUOTES)); Q = np.zeros((n, MAX_QUOTES))
    L = np.zeros(n); N = np.zeros(n, dtype=np.int64); TAGS = []
    ok = np.ones(n, dtype=bool)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        geos = [sample_geometry(rng) for _ in range(e - s)]
        cleans = price_batch(params[s:e], geos, node_count)
        for i, (g, c) in enumerate(zip(geos, cleans)):
            j = s + i
            keep = admissible(params[j], g, c)      # drop untrustworthy quotes, not the surface
            if keep.sum() < 3:
                ok[j] = False; TAGS.append(g["tag"]); continue
            idx = np.where(keep)[0]
            c = c[idx]
            for key in ("spot", "strike", "tau", "rate", "carry"):
                g[key] = g[key][idx]
            noisy, lvl = apply_noise(rng, c, g)
            if rng.random() < 0.30:                      # random quote deletion
                keep = rng.random(len(c)) > rng.uniform(0.05, 0.35)
                if keep.sum() >= 3:
                    idx = np.where(keep)[0]
                    c, noisy = c[idx], noisy[idx]
                    for key in ("spot", "strike", "tau", "rate", "carry"):
                        g[key] = g[key][idx]
            q = len(c); N[j] = q; L[j] = lvl; TAGS.append(g["tag"])
            C[j, :q] = c; Y[j, :q] = noisy
            S[j, :q] = g["spot"]; K[j, :q] = g["strike"]; T[j, :q] = g["tau"]
            R[j, :q] = g["rate"]; Q[j, :q] = g["carry"]
    f32 = lambda a: a.astype(np.float32)      # geometry and prices; params stay float64
    return {"params": params, "regime": regime, "clean": f32(C), "noisy": f32(Y),
            "n_quotes": N, "spot": f32(S), "strike": f32(K), "tau": f32(T),
            "rate": f32(R), "carry": f32(Q), "noise_level": L, "ok": ok,
            "tag": np.array(TAGS)}
