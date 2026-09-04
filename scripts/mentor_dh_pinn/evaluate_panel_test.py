#!/usr/bin/env python3
"""Base vs fine-tuned on the FULL stock-panel test split -- a properly powered
single-expiry evaluation.

The ADANIPOWER test set has a median of four holdout quotes per date, which cannot resolve
the effect being measured. The panel's chronological test split covers eleven symbols and
4,662 quotes, is disjoint from the fine-tuning corpus (which draws only from
`split == "train"`), and is dominated by exactly the one- and two-expiry geometry that is
the open research problem.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
torch.set_default_dtype(torch.float64)
from src.mentor_dh_pinn.baselines import (bs_surface, dh_surface, fit_black_scholes,
                                          fit_double_heston, fit_single_heston, sh_surface)
from src.mentor_dh_pinn.nifty_panel import implied_vol
from src.mentor_dh_pinn.unified import UnifiedCalibrator

PANEL = Path("/Users/dhruvaambhaikar/Documents/Options pricing/outputs/"
             "pinn_single_heston/pinn_quote_panel.parquet")


def load_enc(p):
    ck = torch.load(p, weights_only=False); c = ck["config"]
    m = UnifiedCalibrator(d_model=c["d_model"], rounds=c["rounds"], node_count=c["nodes"])
    m.load_state_dict(ck["state_dict"]); m.eval(); return m


def cases(min_fit=8, min_hold=4):
    p = pd.read_parquet(PANEL); p = p[p.split == "test"]
    out = []
    for (sym, date), g in p.groupby(["symbol", "trade_date"]):
        g = g.copy()
        fp = g.market_price_adjusted/g.discount_factor
        fp = fp + np.where(~g.is_call, g.forward - g.strike, 0.0)
        g["fwd_price"] = fp
        cal, hol = g[g.fold == "calibration"], g[g.fold == "holdout"]
        if len(cal) < min_fit or len(hol) < min_hold: continue
        mk = lambda x: {"spot": x.forward.to_numpy(float), "strike": x.strike.to_numpy(float),
                        "tau": x.maturity.to_numpy(float), "rate": np.zeros(len(x)),
                        "carry": np.zeros(len(x))}
        iv = np.array([implied_vol(float(r.fwd_price), float(r.forward), float(r.strike),
                                   float(r.maturity)) for r in hol.itertuples()])
        ok = np.isfinite(iv)
        if ok.sum() < min_hold: continue
        gh = {k: v[ok] for k, v in mk(hol).items()}
        out.append({"label": f"{sym}|{date.date()}", "symbol": sym,
                    "geo": mk(cal), "price": cal.fwd_price.to_numpy(float),
                    "geo_h": gh, "iv_h": iv[ok],
                    "n_expiries": int(cal.expiry_date.nunique())})
    return out


def score(params, gh, ivh, kind="dh"):
    m = (dh_surface if kind == "dh" else sh_surface)(params, gh) if kind != "bs" \
        else bs_surface(params, gh)
    e = []
    for i in range(len(ivh)):
        v = implied_vol(float(m[i]), float(gh["spot"][i]), float(gh["strike"][i]),
                        float(gh["tau"][i]))
        if np.isfinite(v): e.append(v - ivh[i])
    return float(np.sqrt(np.mean(np.square(e)))) if e else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=ROOT/"outputs"/"unified_v6"/"unified.pt")
    ap.add_argument("--ft", type=Path, default=ROOT/"outputs"/"unified_v6"/"unified_ft.pt")
    ap.add_argument("--limit", type=int, default=140)
    ap.add_argument("--out", type=Path, default=ROOT/"outputs"/"unified_v6")
    a = ap.parse_args()
    base, ft = load_enc(a.base), load_enc(a.ft)
    cs = cases()
    print(f"{len(cs)} usable panel-test surfaces; scoring {min(a.limit, len(cs))}", flush=True)
    rng = np.random.default_rng(5)
    sel = rng.choice(len(cs), min(a.limit, len(cs)), replace=False)
    rows = []
    for n, i in enumerate(sel, 1):
        c = cs[i]; geo, obs = c["geo"], c["price"]
        sig = fit_black_scholes(geo, obs)
        r = (bs_surface(sig, geo)-obs)/np.maximum(obs, 1e-9)
        noise = float(np.clip(np.median(np.abs(r))*1.4826, 0.001, 0.08))
        t = lambda x: torch.tensor(np.asarray(x, float)).unsqueeze(0)
        nq = len(geo["tau"])
        b = {"spot": t(geo["spot"]), "strike": t(geo["strike"]), "tau": t(geo["tau"]),
             "rate": t(geo["rate"]), "carry": t(geo["carry"]), "price": t(obs),
             "clean": t(obs), "mask": t(np.ones(nq)),
             "noise_level": torch.tensor([noise]), "n_quotes": torch.tensor([nq])}
        rec = {"label": c["label"], "symbol": c["symbol"], "n_expiries": c["n_expiries"],
               "n_hold": len(c["iv_h"])}
        for nm, mdl in (("base", base), ("ft", ft)):
            with torch.no_grad(): o = mdl(b, refine_steps=3)
            rec[nm] = score(o["params"].numpy()[0], c["geo_h"], c["iv_h"])
        dh = fit_double_heston(geo, obs)
        rec["dh_cold"] = score(dh["params"], c["geo_h"], c["iv_h"]) if dh else np.nan
        sh = fit_single_heston(geo, obs)
        rec["sh"] = score(sh["params"], c["geo_h"], c["iv_h"], "sh") if sh else np.nan
        rows.append(rec)
        if n % 20 == 0: print(f"    {n}/{len(sel)}", flush=True)
    d = pd.DataFrame(rows).dropna(subset=["base", "ft", "dh_cold"])
    d.to_csv(a.out/"panel_test_detail.csv", index=False)
    w = int((d.ft < d.base).sum())
    out = {"surfaces": int(len(d)), "holdout_quotes": int(d.n_hold.sum()),
           "expiry_mix": d.n_expiries.value_counts().to_dict(),
           "median_iv_rmse": {c: float(d[c].median()) for c in ("sh","dh_cold","base","ft")},
           "ft_beats_base": f"{w}/{len(d)}",
           "wilcoxon_ft_vs_base": float(stats.wilcoxon(d.ft, d.base).pvalue),
           "wilcoxon_ft_vs_dh": float(stats.wilcoxon(d.ft, d.dh_cold).pvalue),
           "improvement_pct": float(100*(1-d.ft.median()/d.base.median()))}
    (a.out/"panel_test_summary.json").write_text(json.dumps(out, indent=2))
    print("\n"+json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
