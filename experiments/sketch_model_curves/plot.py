"""Reproduce the requested spot/expiry sketches without fitting or retraining.

Run: python -m experiments.sketch_model_curves.plot
Uses the branch-selected DH PINN, numerical SH and analytical Black-Scholes.
"""
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.pinn_architecture_v6.evaluate import nets
from experiments.nifty_multifactor_v4 import run as old
from experiments.nifty_multifactor_v4.literature_exact import exact, black, adaptive, admissible

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "artifacts"
V6 = ROOT / "experiments/pinn_architecture_v6/artifacts"
LABELS = ["Black–Scholes (analytical)", "Single Heston (numerical)",
          "Double Heston (improved PINN)"]
COLORS = ["#57738b", "#db8831", "#087e8b"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    selected = json.loads((V6 / "selection.json").read_text())
    assert selected["chosen"] == "ARCH_A_MODIFIED_MLP"
    # Check the complete saved selection before loading; no new model selection.
    for name, digest in selected["files"].items():
        assert sha(ROOT / name) == digest, name
    models = nets(selected["chosen"])
    for net in models:
        net.requires_grad_(False)
    cfg = json.loads((old.HERE / "config.json").read_text())
    dh = np.array(cfg["published_double_slow_first"])
    sh = np.array(cfg["published_single"])
    admissible(dh); admissible(sh)
    sigma = float(np.sqrt(dh[4] + dh[9]))
    K = 100.0  # Illustrative units; not a market quote or a forecast.
    audit = []

    def prices(spot, days):
        spot, days = np.broadcast_arrays(np.asarray(spot, float), np.asarray(days, float))
        spot, days = spot.ravel(), days.ravel()
        tau = days / 365
        x = np.log(spot / K)  # r=q=0 => F=S, discount=1.
        assert np.all(np.abs(x) <= .36 + 1e-12)
        bs = np.maximum(spot - K, 0)
        positive = tau > 0
        bs[positive] = spot[positive] * black(x[positive], tau[positive], sigma)
        single = spot * exact(sh, x, tau, audit)
        reference = spot * exact(dh, x, tau, audit)
        pinn = spot * old.neural(models, dh, x, tau)[0]
        values = np.column_stack([bs, single, pinn, reference])
        assert np.isfinite(values).all()
        return {"spot": spot, "days": days, "tau_years": tau, "values": values}

    spot_curve = prices(np.linspace(K*np.exp(-.36), K*np.exp(.36), 301), 90.)
    expiry_curve = prices(K, np.linspace(0, 90, 181))
    checks = {}
    for name, curve in [("spot", spot_curve), ("expiry", expiry_curve)]:
        y = curve["values"]
        lo = np.maximum(curve["spot"] - K, 0)[:, None]
        hi = curve["spot"][:, None]
        domain = (curve["days"] >= 7) | (curve["days"] == 0)
        err = y[:, 2] - y[:, 3]
        checks[name] = {
            "finite": bool(np.isfinite(y).all()),
            "bound_violations_by_model": ((y < lo-1e-7) | (y > hi+1e-7)).sum(0).tolist(),
            "negative_increments_by_model": (np.diff(y, axis=0) < -1e-7).sum(0).tolist(),
            "PINN_reference_RMSE_in_domain": float(np.sqrt(np.mean(err[domain]**2))),
            "PINN_reference_max_error_in_domain": float(np.max(abs(err[domain]))),
            "PINN_reference_max_error_extrapolated": float(np.max(abs(err[~domain]))) if (~domain).any() else None,
        }
        if name == "spot":
            checks[name]["negative_second_differences_by_model"] = (np.diff(y, n=2, axis=0) < -1e-7).sum(0).tolist()
        else:
            checks[name]["terminal_prices"] = y[0].tolist()
            assert np.allclose(y[0], 0, atol=1e-10)
        assert not any(checks[name]["bound_violations_by_model"])
        assert not any(checks[name]["negative_increments_by_model"])
    assert not any(checks["spot"]["negative_second_differences_by_model"])

    # Independent CF sanity check: vanishing vol-of-vol at constant variance
    # must approach Black-Scholes. This does not force the main DH curve to BS.
    xcheck = np.array([-.2, 0., .2])
    tcheck = np.array([.25, .5, 1.])
    bs_limit = 100*np.exp(xcheck)*black(xcheck, tcheck, .2)
    for label, p in [("single", [2., .04, 1e-4, -.5, .04]),
                     ("double", [1., .015, 1e-4, -.5, .015, 5., .025, 1e-4, -.5, .025])]:
        got = 100*np.exp(xcheck)*exact(np.array(p), xcheck, tcheck)
        checks[label+"_BS_limit_max_error"] = float(max(abs(got-bs_limit)))
        assert checks[label+"_BS_limit_max_error"] < .002
    # Independently integrated points verify the main numerical reference.
    checks["adaptive_crosscheck_max_error"] = max(
        abs(100*np.exp(x)*adaptive(p, x, .25)[0] - 100*np.exp(x)*exact(p, np.array([x]), np.array([.25]))[0])
        for p in [sh, dh] for x in [-.2, 0., .2])
    assert checks["adaptive_crosscheck_max_error"] < 2e-5

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    def draw(ax, curve, variable):
        x = curve[variable]
        for j, (label, color) in enumerate(zip(LABELS, COLORS)):
            ax.plot(x, curve["values"][:, j], color=color, lw=2.4,
                    ls=[":", "--", "-"][j], label=label)
        ax.grid(alpha=.18)
        ax.set_ylabel("European call price C (illustrative units)")
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=9)
        if variable == "spot":
            ax.set(xlabel="Underlying price S", title="1. Rising underlying price", xlim=(x.min(), x.max()))
            ax.axvline(K, c="#777777", ls=":", lw=.8)
        else:
            ax.set(xlabel="Time remaining to expiry (days): 90 → 0", title="2. Approaching expiry", xlim=(90, 0))
            ax.axvspan(0, 7, color="#e7cba3", alpha=.35)
            ax.text(.985, .91, "Shaded: <7 days\nPINN extrapolation", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8, color="#6d4b1a")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.subplots_adjust(left=.07, right=.985, top=.80, bottom=.25, wspace=.24)
    draw(axes[0], spot_curve, "spot"); draw(axes[1], expiry_curve, "days")
    fig.suptitle("Single Heston, Double Heston PINN and Black–Scholes", fontsize=17, y=.97)
    fig.text(.5, .88, f"Fixed published anchors · K=100 · r=q=0 · BS volatility={100*sigma:.3f}% · no fitting or retraining",
             ha="center", fontsize=10)
    fig.text(.27, .045, "Fixed maturity: 90 days. Call prices rise with S.\nCurves may cross; their ranking is not imposed.", ha="center", fontsize=10)
    fig.text(.76, .045, "Fixed S=K=100. Less time remaining lowers value.\nAt expiry every curve reaches the payoff: zero here.", ha="center", fontsize=10)
    fig.savefig(OUT / "model_curves.png", dpi=220)
    plt.close(fig)
    for name, curve, variable in [("stock_price", spot_curve, "spot"), ("toward_expiry", expiry_curve, "days")]:
        fig, ax = plt.subplots(figsize=(9, 5.6))
        fig.subplots_adjust(bottom=.20, top=.85, left=.12, right=.97)
        draw(ax, curve, variable)
        fig.text(.5, .06, "Synthetic fixed-parameter comparison; not observed market prices. K=100, r=q=0.\n"
                 + ("Maturity held at 90 days." if variable == "spot" else "S held at 100; time remaining decreases left to right."), ha="center", fontsize=10)
        fig.savefig(OUT / (name+".png"), dpi=220); plt.close(fig)
    metadata = {
        "branch": "codex/improved-double-heston-pinn",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "candidate": selected["chosen"], "seeds": [17, 43], "K": K, "r": 0, "q": 0,
        "parameters_order": ["kappa", "theta", "sigma", "rho", "v0"],
        "single": sh.tolist(), "double_slow_then_fast": dh.tolist(), "BS_sigma": sigma,
        "parameter_source": "https://doi.org/10.1155/2021/6634779, Table 1 ordinary Heston/DH comparator rows",
        "model_order": LABELS+["Double Heston numerical reference (validation only)"],
        "no_training_or_market_data": True, "no_price_clipping_or_smoothing": True,
        "selection_sha256": sha(V6/"selection.json"), "generator_sha256": sha(Path(__file__)),
        "checks": checks, "adaptive_fallback_count": len(audit),
        "curves": {name: {k: v.tolist() for k, v in curve.items()} for name, curve in [("spot", spot_curve), ("expiry", expiry_curve)]},
    }
    for name, digest in selected["files"].items():
        assert sha(ROOT / name) == digest, name
    (OUT / "results.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
    (OUT / "REPORT.md").write_text(
        "# Requested model curves\n\n"
        f"Branch `{metadata['branch']}`, commit `{metadata['commit']}`. No retraining or market data used.\n\n"
        "![Curves](model_curves.png)\n\n"
        "## Interpretation\n\n"
        "1. The horizontal axis is underlying price, not calendar time or an observed market path. "
        "All three call-price curves increase and are convex on this grid at fixed 90-day maturity.\n"
        "2. The second horizontal axis deliberately runs from 90 days remaining to zero. "
        "With S=K=100 and r=q=0 held fixed, prices fall toward the zero terminal payoff. "
        "If tau instead increased left to right, these same maturity curves would rise. "
        "The 0<tau<7-day PINN region is explicitly extrapolation; tau=0 is its exact payoff construction.\n\n"
        "## What Black–Scholes consistency means\n\n"
        "Shared call bounds, spot monotonicity, convexity, terminal payoff and the appropriate constant-variance limit—"
        "not identical prices or a predetermined DH>SH>BS ranking. Spot, rates and variance inputs are fixed "
        "on the time slice; a real market path need not decline smoothly.\n\n"
        "SH uses the branch's Fourier pricer, DH uses its selected ARCH_A_MODIFIED_MLP two-seed PINN, "
        "and BS uses the analytical formula. The DH numerical price is separately computed for error checks. "
        "These are not three PINNs or a new out-of-sample accuracy test.\n\n"
        "## Parameters and provenance\n\n"
        "[Chang, Wang & Zhang (2021), Table 1](https://doi.org/10.1155/2021/6634779) supplies the unchanged "
        "ordinary Heston/DH anchors, originally DJIA ETF values—not Bitcoin parameters. "
        f"Order per factor is kappa,theta,sigma,rho,v0. SH={sh.tolist()}; DH slow/fast={dh.tolist()}. "
        f"BS sigma=sqrt(DH initial total variance)={sigma:.10f}. K=100; r=q=0. "
        "SH and DH retain their own published variances; differences are scenario-specific, not a ranking of accuracy.\n\n"
        "Numerical check details and every plotted value are in [results.json](results.json). "
        "No line offsets, target-based parameter selection, clipping, or smoothing was applied.\n\n"
        "Reproduce from this checkout with `../.venv/bin/python -m experiments.sketch_model_curves.plot`.\n")
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == "__main__":
    main()
