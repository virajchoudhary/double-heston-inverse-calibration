"""Out-of-distribution reporting: the model must fail visibly, never silently.

The audited `calibrate.decode` hid its failures. A market vector with theta_slow = 0.40 and
kappa_fast = 14.0 came back as 0.24 and 11.0, with sigma dragged along too, and nothing said
so. `params_v2.decode` removes the clipping -- it is a bijection onto the engine's model
class -- which means the model can now *express* an unusual answer. This module is what makes
it *report* one.

Two situations are deliberately distinguished, because they have different remedies:

* outside the TRAINING PRIOR   -- expressible and priceable, but the network is extrapolating;
                                 the number may be poor and the uncertainty should be wide.
* outside the MODEL CLASS      -- the exact engine refuses to price it at all. This cannot be
                                 returned as a calibration under any circumstances.

Status is one of: in_distribution, mild_extrapolation, severe_extrapolation, numerically_unsafe.
These are ordinal labels backed by measured percentiles, not calibrated probabilities, and are
reported as such.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

STATUSES = ("in_distribution", "mild_extrapolation", "severe_extrapolation", "numerically_unsafe")
_QUANTILES = (0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999)


def build_reference(train: dict, path: Path) -> dict:
    """Summarise the training support once, from the training set only."""
    ok = train["ok"]
    p = train["params"][ok]
    n = train["n_quotes"][ok]
    tau = train["tau"][ok]; mask = np.arange(tau.shape[1])[None, :] < n[:, None]
    strike, spot = train["strike"][ok], train["spot"][ok]
    rate, carry = train["rate"][ok], train["carry"][ok]
    fwd = spot * np.exp((rate - carry) * tau)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log(np.where(mask, strike, np.nan) / np.where(mask, fwd, np.nan))
    ref = {
        "param_quantiles": {k: np.quantile(p[:, i], _QUANTILES).tolist()
                            for i, k in enumerate(
                                ("kappa_slow", "theta_slow", "sigma_slow", "rho_slow", "v0_slow",
                                 "kappa_fast", "theta_fast", "sigma_fast", "rho_fast", "v0_fast"))},
        "vol_now": np.quantile(np.sqrt(p[:, 4] + p[:, 9]), _QUANTILES).tolist(),
        "vol_run": np.quantile(np.sqrt(p[:, 1] + p[:, 6]), _QUANTILES).tolist(),
        "n_quotes": np.quantile(n, _QUANTILES).tolist(),
        "tau_days": np.quantile(tau[mask] * 365.0, _QUANTILES).tolist(),
        "log_moneyness": np.nanquantile(x, _QUANTILES).tolist(),
        "quantile_levels": list(_QUANTILES),
    }
    path.write_text(json.dumps(ref, indent=2))
    return ref


def _pct(value: float, q: list[float], levels=_QUANTILES) -> float:
    """Approximate percentile of `value` within a stored quantile ladder."""
    return float(np.interp(value, q, np.array(levels) * 100.0, left=0.0, right=100.0))


def assess(params: np.ndarray, geo: dict, ref: dict, *, sigma_diag=None,
           priced_ok: bool = True) -> dict:
    """Locate one calibration inside the training support and report a status."""
    names = list(ref["param_quantiles"])
    pct = {k: _pct(float(params[i]), ref["param_quantiles"][k]) for i, k in enumerate(names)}
    vol_now = float(np.sqrt(params[4] + params[9]))
    vol_run = float(np.sqrt(params[1] + params[6]))
    tau_days = np.asarray(geo["tau"]) * 365.0
    fwd = np.asarray(geo["spot"]) * np.exp((np.asarray(geo["rate"]) - np.asarray(geo["carry"]))
                                           * np.asarray(geo["tau"]))
    x = np.log(np.asarray(geo["strike"]) / fwd)
    marks = {
        "vol_now_pct": _pct(vol_now, ref["vol_now"]),
        "vol_run_pct": _pct(vol_run, ref["vol_run"]),
        "n_quotes_pct": _pct(float(len(tau_days)), ref["n_quotes"]),
        "tau_min_pct": _pct(float(tau_days.min()), ref["tau_days"]),
        "tau_max_pct": _pct(float(tau_days.max()), ref["tau_days"]),
        "moneyness_min_pct": _pct(float(x.min()), ref["log_moneyness"]),
        "moneyness_max_pct": _pct(float(x.max()), ref["log_moneyness"]),
    }
    extremes = [v for v in list(pct.values()) + list(marks.values())]
    edge = min(min(extremes), 100.0 - max(extremes))       # distance to the nearest tail, in %
    if not priced_ok:
        status = "numerically_unsafe"
    elif edge >= 1.0:
        status = "in_distribution"
    elif edge >= 0.1:
        status = "mild_extrapolation"
    else:
        status = "severe_extrapolation"
    out = {"status": status, "tail_distance_pct": edge,
           "parameter_percentiles": pct, "geometry_percentiles": marks,
           "note": "ordinal label from measured training quantiles; not a calibrated probability"}
    if sigma_diag is not None:
        out["latent_sd"] = [float(v) for v in np.sqrt(np.maximum(sigma_diag, 0.0))]
    return out
