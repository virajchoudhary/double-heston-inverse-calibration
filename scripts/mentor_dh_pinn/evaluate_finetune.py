#!/usr/bin/env python3
"""Base vs projection-fine-tuned encoder on the FROZEN real test sets.

Neither NIFTY test dates nor ADANIPOWER test dates appear anywhere in the fine-tuning
corpus: the stock panel contributes only `split == "train"`, and the 20 highest-realised-vol
NIFTY dates (test plus validation) are withheld by `build_real_corpus.py`.

Everything is scored on held-out quotes -- every third strike -- that no arm fitted.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"scripts"/"mentor_dh_pinn"))
torch.set_default_dtype(torch.float64)
from evaluate_real_markets import adanipower_cases, nifty_cases, as_batch, iv_rmse, load_model
from src.mentor_dh_pinn.baselines import (bs_surface, fit_black_scholes,
                                          fit_double_heston, fit_single_heston, sh_surface)
from src.mentor_dh_pinn.unified import UnifiedCalibrator


def load_enc(path: Path):
    ck = torch.load(path, weights_only=False); c = ck["config"]
    m = UnifiedCalibrator(d_model=c["d_model"], rounds=c["rounds"], node_count=c["nodes"])
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=ROOT/"outputs"/"unified_v6"/"unified.pt")
    ap.add_argument("--ft", type=Path, default=ROOT/"outputs"/"unified_v6"/"unified_ft.pt")
    ap.add_argument("--out", type=Path, default=ROOT/"outputs"/"unified_v6")
    a = ap.parse_args()
    base, _ = load_enc(a.base)
    ft, ckft = load_enc(a.ft)
    print(f"fine-tuned checkpoint: step {ckft.get('step')}, "
          f"real-val IV RMSE {ckft.get('real_iv_rmse'):.5f}", flush=True)
    sel = json.loads(Path("/Users/dhruvaambhaikar/Documents/Options pricing/outputs/"
                          "nifty_selection.json").read_text())["dates"]
    rows = []
    for tag, cases in (("ADANIPOWER", adanipower_cases()), ("NIFTY", nifty_cases(sel))):
        for c in cases:
            geo, obs = c["geo"], c["price"]
            sig = fit_black_scholes(geo, obs)
            r = (bs_surface(sig, geo) - obs)/np.maximum(obs, 1e-9)
            noise = float(np.clip(np.median(np.abs(r))*1.4826, 0.001, 0.08))
            b = as_batch(geo, obs, noise)
            rec = {"market": tag, "label": c["label"], "n_expiries": c["n_expiries"],
                   "n_hold": len(c["iv_h"])}
            for name, model in (("base", base), ("ft", ft)):
                for steps in (0, 3):
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        o = model(b, refine_steps=steps)
                    rec[f"{name}_{steps}"] = iv_rmse(o["params"].numpy()[0],
                                                     c["geo_h"], c["iv_h"], "dh")
                    rec[f"t_{name}_{steps}"] = time.perf_counter()-t0
            t0 = time.perf_counter(); dh = fit_double_heston(geo, obs)
            rec["dh_cold"] = iv_rmse(dh["params"], c["geo_h"], c["iv_h"], "dh") if dh else np.nan
            rec["t_dh"] = time.perf_counter()-t0
            sh = fit_single_heston(geo, obs)
            rec["sh"] = iv_rmse(sh["params"], c["geo_h"], c["iv_h"], "sh") if sh else np.nan
            rec["bs"] = iv_rmse(sig, c["geo_h"], c["iv_h"], "bs")
            rows.append(rec)
            print(f"  {tag} {c['label']}  base3 {rec['base_3']:.4f} -> ft3 {rec['ft_3']:.4f}"
                  f"   dh {rec['dh_cold']:.4f}", flush=True)
    d = pd.DataFrame(rows); d.to_csv(a.out/"finetune_real_detail.csv", index=False)
    summ = {}
    for tag, g in d.groupby("market"):
        cols = ["bs", "sh", "dh_cold", "base_0", "base_3", "ft_0", "ft_3"]
        best = g[cols].idxmin(axis=1)
        summ[tag] = {"dates": int(len(g)), "holdout_quotes": int(g.n_hold.sum()),
                     "median_iv_rmse": {c: float(g[c].median()) for c in cols},
                     "dates_best": {c: int((best == c).sum()) for c in cols},
                     "median_seconds": {"base_3": float(g.t_base_3.median()),
                                        "ft_3": float(g.t_ft_3.median()),
                                        "dh_cold": float(g.t_dh.median())},
                     "improvement_ft_vs_base_pct": float(
                         100*(1 - g.ft_3.median()/g.base_3.median()))}
    (a.out/"finetune_real_summary.json").write_text(json.dumps(summ, indent=2))
    print("\n"+json.dumps(summ, indent=2), flush=True)


if __name__ == "__main__":
    main()
