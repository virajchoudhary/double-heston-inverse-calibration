#!/usr/bin/env python3
"""NIFTY and ADANIPOWER, with the unified model fed ACTUAL quote locations.

The previous architecture required every real surface to be interpolated onto a fixed
5 x 9 lattice, which smoothed the data before the model saw it and made ADANIPOWER's
single-expiry dates impossible to ingest at all. Nothing is interpolated here.

Protocol, unchanged from the earlier real-market work so the numbers are comparable:
every third strike is held out, all arms fit the same remaining quotes, and all are scored
on the held-out quotes in implied-volatility RMSE at their own forward and maturity.

Ingestion is not identification. A single-expiry ADANIPOWER date becoming processable does
not mean theta and kappa are identified from it; the uncertainty and OOD outputs are
reported precisely so that distinction is visible.
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
torch.set_default_dtype(torch.float64)
from src.mentor_dh_pinn.baselines import (bs_surface, dh_surface, fit_black_scholes,
                                          fit_double_heston, fit_single_heston, sh_surface)
from src.mentor_dh_pinn.nifty_panel import black76, implied_vol, surface as nifty_surface
from src.mentor_dh_pinn.ood import assess
from src.mentor_dh_pinn.params_v2 import CANONICAL
from src.mentor_dh_pinn.unified import UnifiedCalibrator

PANEL = Path("/Users/dhruvaambhaikar/Documents/Options pricing/outputs/"
             "pinn_single_heston/pinn_quote_panel.parquet")


def load_model(ckpt):
    ck = torch.load(ckpt, weights_only=False); c = ck.get("config", {})
    m = UnifiedCalibrator(d_model=c.get("d_model", 128), rounds=c.get("rounds", 3),
                          node_count=c.get("nodes", 48))
    m.load_state_dict(ck["state_dict"]); m.eval(); return m, ck


def as_batch(geo, price, noise):
    t = lambda a: torch.tensor(np.asarray(a, dtype=float)).unsqueeze(0)
    n = len(geo["tau"])
    return {"spot": t(geo["spot"]), "strike": t(geo["strike"]), "tau": t(geo["tau"]),
            "rate": t(geo["rate"]), "carry": t(geo["carry"]), "price": t(price),
            "clean": t(price), "mask": t(np.ones(n)),
            "noise_level": torch.tensor([float(noise)]), "n_quotes": torch.tensor([n])}


def iv_rmse(params, geo_h, iv_h, kind):
    fn = {"dh": dh_surface, "sh": sh_surface}[kind] if kind in ("dh", "sh") else None
    m = fn(params, geo_h) if fn else bs_surface(params, geo_h)
    errs = []
    for i in range(len(iv_h)):
        # geo is already in the forward measure: spot = F, r = q = 0
        v = implied_vol(float(m[i]), float(geo_h["spot"][i]), float(geo_h["strike"][i]),
                        float(geo_h["tau"][i]))
        if np.isfinite(v): errs.append(v - iv_h[i])
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else np.nan


def nifty_cases(dates):
    out = []
    for dt in dates:
        s = nifty_surface(dt)
        if s.empty: continue
        s = s.reset_index(drop=True)
        fit = np.ones(len(s), bool)
        for _, g in s.groupby("dte"):
            fit[g.sort_values("strike").index.to_numpy()[2::3]] = False
        hold = (~fit) & s.dte.between(20, 800).to_numpy()
        if hold.sum() < 4 or fit.sum() < 8: continue
        mk = lambda m: {"spot": s.forward.to_numpy()[m], "strike": s.strike.to_numpy()[m],
                        "tau": s.tau.to_numpy()[m], "rate": np.zeros(int(m.sum())),
                        "carry": np.zeros(int(m.sum()))}
        out.append({"label": dt, "geo": mk(fit), "price": s.fwd_call.to_numpy()[fit],
                    "geo_h": mk(hold), "iv_h": s.iv.to_numpy()[hold],
                    "spot": float(s.spot.iloc[0]),
                    "n_expiries": int(s[fit].dte.nunique())})
    return out


def adanipower_cases():
    p = pd.read_parquet(PANEL)
    p = p[(p.symbol == "ADANIPOWER") & (p.split == "test")]
    out = []
    for date, g in p.groupby("trade_date"):
        g = g.copy()
        g["fwd_price"] = g.market_price_adjusted / g.discount_factor
        g.loc[~g.is_call, "fwd_price"] += g.loc[~g.is_call, "forward"] - g.loc[~g.is_call, "strike"]
        cal, hol = g[g.fold == "calibration"], g[g.fold == "holdout"]
        if len(cal) < 5 or len(hol) < 2: continue
        mk = lambda x: {"spot": x.forward.to_numpy(float), "strike": x.strike.to_numpy(float),
                        "tau": x.maturity.to_numpy(float), "rate": np.zeros(len(x)),
                        "carry": np.zeros(len(x))}
        iv_h = np.array([implied_vol(float(r.fwd_price), float(r.forward), float(r.strike),
                                     float(r.maturity)) for r in hol.itertuples()])
        ok = np.isfinite(iv_h)
        if ok.sum() < 2: continue
        gh = mk(hol); gh = {k: v[ok] for k, v in gh.items()}
        out.append({"label": str(date.date()), "geo": mk(cal),
                    "price": cal.fwd_price.to_numpy(float), "geo_h": gh, "iv_h": iv_h[ok],
                    "spot": float(g.spot.iloc[0]), "n_expiries": int(cal.expiry_date.nunique())})
    return out


def run(cases, model, ref, refine, tag):
    rows = []
    for c in cases:
        geo, obs = c["geo"], c["price"]
        # quote-noise estimate: dispersion of the fit residual of a smooth 1-sigma fit,
        # which is all that is available -- NSE publishes no bid/ask in the bhavcopy
        sig = fit_black_scholes(geo, obs)
        resid = (bs_surface(sig, geo) - obs) / np.maximum(obs, 1e-9)
        noise = float(np.clip(np.median(np.abs(resid)) * 1.4826, 0.001, 0.08))
        b = as_batch(geo, obs, noise)
        t0 = time.perf_counter()
        with torch.no_grad(): o0 = model(b, refine_steps=0)
        t_enc = time.perf_counter() - t0
        t0 = time.perf_counter()
        with torch.no_grad(): o = model(b, refine_steps=refine)
        t_tot = time.perf_counter() - t0
        pre = o["params_pre"].numpy()[0]; post = o["params"].numpy()[0]
        Sig = (o["L"] @ o["L"].transpose(-1, -2)).numpy()[0]
        r = {"label": c["label"], "n_expiries": c["n_expiries"], "n_fit": len(obs),
             "n_hold": len(c["iv_h"]), "noise_est": noise,
             "unified_pre_iv": iv_rmse(pre, c["geo_h"], c["iv_h"], "dh"),
             "unified_post_iv": iv_rmse(post, c["geo_h"], c["iv_h"], "dh"),
             "t_encoder": t_enc, "t_total": t_tot}
        r.update({f"post_{k}": float(post[j]) for j, k in enumerate(CANONICAL)})
        od = assess(post, geo, ref, sigma_diag=np.diag(Sig))
        r["ood_status"] = od["status"]; r["ood_tail_pct"] = od["tail_distance_pct"]
        sd = np.sqrt(np.diag(Sig))
        r["sd_theta"] = float(np.mean(sd[[2, 3]])); r["sd_kappa"] = float(np.mean(sd[[0, 1]]))
        r["sd_v0"] = float(np.mean(sd[[4, 5]]))
        t0 = time.perf_counter(); r["bs_iv"] = iv_rmse(sig, c["geo_h"], c["iv_h"], "bs")
        r["bs_s"] = time.perf_counter() - t0
        t0 = time.perf_counter(); sh = fit_single_heston(geo, obs)
        r["sh_iv"] = iv_rmse(sh["params"], c["geo_h"], c["iv_h"], "sh") if sh else np.nan
        r["sh_s"] = time.perf_counter() - t0
        t0 = time.perf_counter(); dh = fit_double_heston(geo, obs)
        r["dh_iv"] = iv_rmse(dh["params"], c["geo_h"], c["iv_h"], "dh") if dh else np.nan
        r["dh_s"] = time.perf_counter() - t0
        rows.append(r)
        print(f"  {tag} {c['label']}  exp {c['n_expiries']}  BS {r['bs_iv']:.4f}  "
              f"SH {r['sh_iv']:.4f}  DHcold {r['dh_iv']:.4f}  "
              f"UNIpre {r['unified_pre_iv']:.4f}  UNIpost {r['unified_post_iv']:.4f}  "
              f"[{r['ood_status']}]", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=ROOT / "outputs" / "unified_v6" / "unified.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "unified_v6")
    ap.add_argument("--refine", type=int, default=3)
    a = ap.parse_args()
    model, ck = load_model(a.ckpt)
    ref = json.loads((a.out / "ood_reference.json").read_text())
    sel = json.loads(Path("/Users/dhruvaambhaikar/Documents/Options pricing/outputs/"
                          "nifty_selection.json").read_text())["dates"]
    print("=== NIFTY (actual quote locations, no interpolation) ===", flush=True)
    nf = run(nifty_cases(sel), model, ref, a.refine, "NIFTY")
    print("\n=== ADANIPOWER (single-expiry dates the old architecture could not ingest) ===",
          flush=True)
    ad = run(adanipower_cases(), model, ref, a.refine, "ADANI")
    nf.to_csv(a.out / "real_nifty_detail.csv", index=False)
    ad.to_csv(a.out / "real_adanipower_detail.csv", index=False)
    def summ(d, name):
        cols = ["bs_iv", "sh_iv", "dh_iv", "unified_pre_iv", "unified_post_iv"]
        best = d[cols].idxmin(axis=1)
        return {"market": name, "dates": int(len(d)),
                "holdout_quotes": int(d.n_hold.sum()),
                "median_iv_rmse": {c: float(d[c].median()) for c in cols},
                "dates_best": {c: int((best == c).sum()) for c in cols},
                "ood": d.ood_status.value_counts().to_dict(),
                "median_latency_s": {"encoder": float(d.t_encoder.median()),
                                     "encoder_plus_physics": float(d.t_total.median()),
                                     "double_heston_cold": float(d.dh_s.median())},
                "uncertainty_by_expiry_count": {
                    str(k): {"n": int(len(v)), "sd_theta": float(v.sd_theta.median()),
                             "sd_kappa": float(v.sd_kappa.median()),
                             "sd_v0": float(v.sd_v0.median())}
                    for k, v in d.groupby("n_expiries")}}
    out = {"nifty": summ(nf, "NIFTY"), "adanipower": summ(ad, "ADANIPOWER"),
           "checkpoint": {"epoch": ck.get("epoch"), "phase": ck.get("phase")}}
    (a.out / "real_markets_summary.json").write_text(json.dumps(out, indent=2))
    print("\n" + json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
