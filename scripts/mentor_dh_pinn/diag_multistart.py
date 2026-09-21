"""Does sampling starts from the model's OWN predicted covariance escape the bad basin?

The network already emits Sigma_z. It has only ever been used as a prior stiffness, never
as a proposal distribution. If ADANIPOWER's failure is basin selection on a weakly identified
surface, then K draws from N(mu, Sigma), each refined, selecting by DATA FIT (not by truth),
should recover most of the gap to classical multi-start."""
import sys, json, numpy as np, torch
from pathlib import Path
sys.path.insert(0,'.'); torch.set_default_dtype(torch.float64)
sys.path.insert(0, str(Path('scripts/mentor_dh_pinn').resolve()))
from evaluate_real_markets import adanipower_cases, nifty_cases, as_batch, iv_rmse, load_model
from src.mentor_dh_pinn.baselines import fit_black_scholes, fit_double_heston, bs_surface, dh_surface
from src.mentor_dh_pinn.params_v2 import decode

model,_ = load_model(Path("outputs/unified_v6/unified.pt"))
sel = json.loads(Path("/Users/dhruvaambhaikar/Documents/Options pricing/outputs/nifty_selection.json").read_text())["dates"]
rng = np.random.default_rng(0)

for tag, cases in (("ADANIPOWER", adanipower_cases()), ("NIFTY", nifty_cases(sel))):
    print(f"\n===== {tag} =====", flush=True)
    res = {}
    for c in cases:
        geo, obs = c["geo"], c["price"]
        sig = fit_black_scholes(geo, obs)
        r = (bs_surface(sig,geo)-obs)/np.maximum(obs,1e-9)
        noise = float(np.clip(np.median(np.abs(r))*1.4826,0.001,0.08))
        b = as_batch(geo, obs, noise)
        with torch.no_grad():
            h,pad = model.encode(b); p,_ = model.tokens_forward(h,pad,1); mu,L = model.gaussian_head(p)
        # single start (current behaviour)
        with torch.no_grad():
            z,_ = model.refine(mu, L, b, steps=30)
            p1 = torch.stack(decode(z),dim=-1).numpy()[0]
        res.setdefault("network, single start, 30 steps",[]).append(iv_rmse(p1,c["geo_h"],c["iv_h"],"dh"))
        # K starts drawn from the model's OWN predicted covariance, selected by data fit
        for K in (4, 12):
            best, best_fit = None, np.inf
            for k in range(K):
                eps = torch.tensor(rng.normal(0,1,(1,10)))
                z0 = mu + (L @ eps.unsqueeze(-1)).squeeze(-1) if k else mu
                with torch.no_grad():
                    zk,_ = model.refine(z0, L, b, steps=30)
                    pk = torch.stack(decode(zk),dim=-1).numpy()[0]
                fit = float(np.sqrt(np.mean(((dh_surface(pk,geo)-obs)/geo["spot"])**2)))
                if fit < best_fit: best_fit, best = fit, pk
            res.setdefault(f"network, {K} starts from Sigma, 30 steps",[]).append(
                iv_rmse(best,c["geo_h"],c["iv_h"],"dh"))
        dh = fit_double_heston(geo,obs)
        res.setdefault("classical DH cold 5-start",[]).append(
            iv_rmse(dh["params"],c["geo_h"],c["iv_h"],"dh") if dh else np.nan)
    for k,v in res.items():
        print(f"  {k:<42} median IV RMSE {np.nanmedian(v):.5f}", flush=True)
