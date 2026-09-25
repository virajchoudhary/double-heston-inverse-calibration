"""Is the real-data gap an OPTIMISATION problem (bad starting point, too few steps) or a
PRIOR problem (the covariance anchors to a systematically wrong network prediction)?

Both hypotheses predict the same 3-step result but different behaviour as the prior is
weakened and the step budget raised."""
import sys, json, numpy as np, torch
from pathlib import Path
sys.path.insert(0,'.'); torch.set_default_dtype(torch.float64)
sys.path.insert(0, str(Path('scripts/mentor_dh_pinn').resolve()))
from evaluate_real_markets import (adanipower_cases, nifty_cases, as_batch, iv_rmse, load_model)
from src.mentor_dh_pinn.baselines import fit_black_scholes, fit_double_heston, bs_surface
from src.mentor_dh_pinn.params_v2 import decode

model, _ = load_model(Path("outputs/unified_v6/unified.pt"))
sel = json.loads(Path("/Users/dhruvaambhaikar/Documents/Options pricing/outputs/nifty_selection.json").read_text())["dates"]

for tag, cases in (("ADANIPOWER", adanipower_cases()), ("NIFTY", nifty_cases(sel))):
    print(f"\n===== {tag} =====", flush=True)
    rows = {}
    for c in cases:
        geo, obs = c["geo"], c["price"]
        sig = fit_black_scholes(geo, obs)
        resid = (bs_surface(sig, geo) - obs)/np.maximum(obs,1e-9)
        noise = float(np.clip(np.median(np.abs(resid))*1.4826, 0.001, 0.08))
        b = as_batch(geo, obs, noise)
        with torch.no_grad():
            h,pad = model.encode(b); p,_ = model.tokens_forward(h,pad,1); mu,L = model.gaussian_head(p)
        for label, steps, scale in (("3 steps, prior as trained",3,1.0),
                                    ("10 steps, prior as trained",10,1.0),
                                    ("30 steps, prior as trained",30,1.0),
                                    ("30 steps, prior x100 weaker",30,10.0),
                                    ("30 steps, prior x10000 weaker",30,100.0)):
            with torch.no_grad():
                z,_ = model.refine(mu, L*scale, b, steps=steps)
                pp = torch.stack(decode(z), dim=-1).numpy()[0]
            rows.setdefault(label, []).append(iv_rmse(pp, c["geo_h"], c["iv_h"], "dh"))
        # classical reference on the same quotes
        dh = fit_double_heston(geo, obs)
        rows.setdefault("classical DH cold 5-start", []).append(
            iv_rmse(dh["params"], c["geo_h"], c["iv_h"], "dh") if dh else np.nan)
    for k,v in rows.items():
        print(f"  {k:<32} median IV RMSE {np.nanmedian(v):.5f}", flush=True)
