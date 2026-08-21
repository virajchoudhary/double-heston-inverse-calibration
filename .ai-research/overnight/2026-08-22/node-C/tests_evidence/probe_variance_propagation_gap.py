"""Node C diagnostic: deterministic variance-mean propagation vs true state.

Question: U(S, v0_s, v0_f, tau) is the price with CURRENT variance state v0
(time-homogeneity contract). If instead one advances delta days and prices
with the propagated CONDITIONIONAL MEAN states E[v_delta] = theta + (v0 -
theta) e^{-kappa delta} for maturity tau - delta (the shortcut suggested by
`propagate_variance_state`), how large is the price gap?

The true relation is U(S, v0, tau) = E_delta[U(S_delta, v_delta, tau-delta)]
(tower property over the JOINT (S_delta, v_delta)); substituting means for
random states is a Jensen-type approximation. This quantifies why the
multi-date NTPC calibration fits per-date v0 states instead of propagating.

`propagate_variance_state` itself is currently unused by calibration scripts
(verified); this probe documents the magnitude that ANY future use as a
pricing shortcut would carry.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
SRC_ROOT = PROJECT_ROOT / "src"
for entry in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import math

import numpy as np

_source = (SRC_ROOT / "double_heston.py").read_text()
for _old, _new in (
    ("from typing import TypeAlias", "TypeAlias = object"),
    ("ParameterInput: TypeAlias = Sequence[float] | Mapping[str, float]", "ParameterInput = None"),
    ("ComplexResult: TypeAlias = complex | np.ndarray", "ComplexResult = None"),
):
    assert _source.count(_old) == 1, _old
    _source = _source.replace(_old, _new)
import src as _src_pkg

_mod = importlib.util.module_from_spec(
    importlib.util.spec_from_file_location("src.double_heston", SRC_ROOT / "double_heston.py")
)
_mod.__dict__["__package__"] = "src"
sys.modules["src.double_heston"] = _mod
exec(compile(_source, "double_heston.py", "exec"), _mod.__dict__)

VEC_A = [0.60, 0.040, 0.180, -0.55, 0.040, 2.80, 0.030, 0.350, -0.35, 0.020]
S, K, r, q = 100.0, 100.0, 0.05, 0.0

rows = []
for tau in (0.5, 1.0):
    base = _mod.price_double_heston_call(S, K, tau, r, q, VEC_A)
    for delta_days in (1.0, 7.0, 30.0, 90.0):
        delta = delta_days / 365.0
        if delta >= tau:
            continue
        v_s = VEC_A[1] + (VEC_A[4] - VEC_A[1]) * math.exp(-VEC_A[0] * delta)
        v_f = VEC_A[6] + (VEC_A[9] - VEC_A[6]) * math.exp(-VEC_A[5] * delta)
        shifted = VEC_A.copy()
        shifted[4], shifted[9] = v_s, v_f
        propagated = _mod.price_double_heston_call(S, K, tau - delta, r, q, shifted)
        rows.append({
            "tau": tau, "delta_days": delta_days, "base_price": float(base),
            "mean_propagated_price": float(propagated),
            "abs_gap": float(propagated) - float(base),
            "rel_gap": (float(propagated) - float(base)) / float(base),
        })

results = {
    "rows": rows,
    "note": "Gap = U(S, E[v_delta], tau-delta) - U(S, v0, tau). The true bridge averages over "
            "the JOINT (S_delta, v_delta) law; deterministic mean propagation is a Jensen-type "
            "approximation. Documents the cost of any future propagate-then-price shortcut.",
}
print(json.dumps(results, indent=2))
Path(__file__).with_name("variance_propagation_gap.json").write_text(json.dumps(results, indent=2))
