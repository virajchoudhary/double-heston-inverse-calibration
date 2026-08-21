"""Node A diagnostic: constraint-map output range vs reviewed sampling bounds (Phase D).

Question: the canonical DoubleHestonConstraintMap enforces structural validity by
construction, but nothing restricts predictions to the reviewed synthetic sampling ranges.
How far outside those ranges can unconstrained raw outputs land?

Method: draw raw unconstrained vectors from several increasingly wild distributions
(untrained-head surrogates), map through DoubleHestonConstraintMap, and measure
inclusion in (a) hard numerical safety bounds and (b) PILOT empirical sampling ranges
from configs/parameter_bounds_PROVISIONAL.yaml.

CPU-only, seeded, read-only. Diagnostic evidence only.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))

import torch

from models.pinn_model import DoubleHestonConstraintMap
from src.constants import PARAMETER_NAMES

N = 200_000
HARD = {
    "kappa_slow": (0.05, 3.0), "theta_slow": (0.005, 0.25), "sigma_slow": (0.005, 1.0),
    "rho_slow": (-0.95, 0.95), "v0_slow": (0.005, 0.30),
    "kappa_fast": (0.10, 12.0), "theta_fast": (0.002, 0.20), "sigma_fast": (0.005, 1.5),
    "rho_fast": (-0.95, 0.95), "v0_fast": (0.002, 0.25),
}
PILOT = {
    "kappa_slow": (0.20, 1.50), "theta_slow": (0.020, 0.120), "sigma_slow": (0.030, 0.45),
    "rho_slow": (-0.75, 0.20), "v0_slow": (0.015, 0.150),
    "kappa_fast": (1.60, 6.00), "theta_fast": (0.010, 0.080), "sigma_fast": (0.030, 0.65),
    "rho_fast": (-0.60, 0.35), "v0_fast": (0.008, 0.100),
}

constraint_map = DoubleHestonConstraintMap()
print(f"{'raw dist':<14} {'in-hard%':>9} {'in-pilot%':>10}  worst-violations (median ratio to nearest bound edge)")
for label, scale in [("N(0,1)", 1.0), ("N(0,3)", 3.0), ("N(0,10)", 10.0), ("uniform[-50,50]", None)]:
    g = torch.Generator().manual_seed(42)
    if scale is not None:
        raw = torch.randn(N, 10, generator=g) * scale
    else:
        raw = (torch.rand(N, 10, generator=g) - 0.5) * 100.0
    with torch.no_grad():
        params = constraint_map(raw)

    in_hard = torch.ones(N, dtype=torch.bool)
    in_pilot = torch.ones(N, dtype=torch.bool)
    worst = []
    for i, name in enumerate(PARAMETER_NAMES):
        col = params[:, i]
        lo_h, hi_h = HARD[name]
        lo_p, hi_p = PILOT[name]
        in_hard &= (col >= lo_h) & (col <= hi_h)
        in_pilot &= (col >= lo_p) & (col <= hi_p)
        # how many times past the nearer violated pilot edge, per violating element
        viol_lo = (col < lo_p) & (col > 0)
        viol_hi = col > hi_p
        ratio = torch.where(
            viol_lo, lo_p / col.clamp_min(1e-12),
            torch.where(viol_hi, col / hi_p, torch.ones_like(col)),
        )
        frac_v = (viol_lo | viol_hi).float().mean().item()
        med_ratio = ratio[(viol_lo | viol_hi)].median().item() if frac_v > 0 else 1.0
        worst.append((name, frac_v, med_ratio))
    worst.sort(key=lambda t: -t[1])
    top = ", ".join(f"{n}:{f:.0%}@x{r:.1f}" for n, f, r in worst[:3])
    print(f"{label:<14} {100*in_hard.float().mean().item():8.2f}% {100*in_pilot.float().mean().item():9.2f}%  {top}")

# Structural validity check on the wildest set (should be 100% by construction)
g = torch.Generator().manual_seed(7)
raw = (torch.rand(200_000, 10, generator=g) - 0.5) * 1000.0
with torch.no_grad():
    p = constraint_map(raw)
k_s, t_s, s_s, k_f, t_f, s_f = p[:, 0], p[:, 1], p[:, 2], p[:, 5], p[:, 6], p[:, 7]
feller_ok = ((2*k_s*t_s > s_s**2) & (2*k_f*t_f > s_f**2)).float().mean().item()
disk_ok = (p[:, 3]**2 + p[:, 8]**2 < 1.0).float().mean().item()
order_ok = (k_f > k_s).float().mean().item()
pos_ok = (p[:, [0, 1, 2, 4, 5, 6, 7, 9]] > 0).all(dim=1).float().mean().item()
print(f"\nStructural validity on uniform[-1000,1000] raws (n=200k):")
print(f"  Feller both factors: {feller_ok:.4f} | disk: {disk_ok:.4f} | kappa order: {order_ok:.4f} | positivity: {pos_ok:.4f}")
print(f"  rho_slow range observed: [{p[:,3].min():.3f}, {p[:,3].max():.3f}] (pilot: [-0.75, 0.20])")
print(f"  kappa_fast max observed: {k_f.max():.0f} (hard cap 12, pilot cap 6)")
