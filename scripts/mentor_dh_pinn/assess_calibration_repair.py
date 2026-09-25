#!/usr/bin/env python3
"""Reproducible recovery assessment: truth is used only by generation and scoring.

All model arms see the same calibration quotes, never reserved strikes or parameter
labels. A precision polish is identified explicitly as an optimiser-assisted hybrid.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.optimize import brentq
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
torch.set_default_dtype(torch.float64)
from src.mentor_dh_pinn.dataset_v6 import sample_parameters
from src.mentor_dh_pinn.params_v2 import CANONICAL, decode, encode
from src.mentor_dh_pinn.precise_calibration import exact_polish
from src.mentor_dh_pinn.torch_pricer import price_call
from src.mentor_dh_pinn.unified import UnifiedCalibrator


def prices(params, geo, nodes=128):
    with torch.no_grad():
        return price_call(torch.tensor(params), *(torch.tensor(geo[k]) for k in
                          ("spot", "strike", "tau", "rate", "carry")), node_count=nodes).numpy()


def iv_values(value, geo):
    values = []
    for c, s, k, t, r, q in zip(value, *(geo[key] for key in ("spot", "strike", "tau", "rate", "carry"))):
        f, disc = s*np.exp((r-q)*t), np.exp(-r*t)
        target = c/disc
        def black(sig):
            st = sig*np.sqrt(t)
            d = np.log(f/k)/st + st/2
            return f*ndtr(d)-k*ndtr(d-st)
        if not np.isfinite(target) or not max(f-k, 0) < target < f:
            values.append(np.nan); continue
        try:
            values.append(brentq(lambda v: black(v)-target, 1e-6, 8., xtol=1e-11))
        except ValueError:
            values.append(np.nan)
    return np.array(values)


def input_batch(geo, observed, noise):
    b = {k: torch.tensor(v)[None] for k,v in geo.items()}
    b.update(price=torch.tensor(observed)[None], mask=torch.ones(1,len(observed)),
             noise_level=torch.tensor([max(noise,.0015)]),
             n_quotes=torch.tensor([len(observed)]))
    # Known synthetic observation-noise level is not a structural parameter label.
    b["quote_sigma"] = torch.tensor(np.maximum(noise*np.abs(observed), 1e-6))[None]
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", action="append", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[905101,905102,905103])
    ap.add_argument("--cases-per-seed", type=int, default=8)
    ap.add_argument("--polish-max-nfev", type=int, default=500)
    ap.add_argument("--polish-starts", type=int, default=3)
    ap.add_argument("--no-polish", action="store_true")
    ap.add_argument("--polish-model", help="Optional exact model name (parent/stem) to polish")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/"manifest.json").exists():
        raise FileExistsError("Use a fresh assessment output directory")
    torch.set_num_threads(2)
    st = json.loads((ROOT/"outputs/unified_v6/latent_standardisation.json").read_text())
    zsd = np.asarray(st["sd"])
    models = []
    for path in args.checkpoint:
        ck = torch.load(path, weights_only=False); c = ck["config"]
        m = UnifiedCalibrator(d_model=c["d_model"], rounds=c["rounds"], node_count=c["nodes"])
        m.load_state_dict(ck["state_dict"]); m.eval()
        models.append((path.parent.name+"/"+path.stem, m))
    manifest = {"seeds": args.seeds, "cases_per_seed": args.cases_per_seed,
                "checkpoint_sha256": {str(p): hashlib.file_digest(p.open("rb"),"sha256").hexdigest()
                                      for p in args.checkpoint},
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "polish": {"max_nfev": args.polish_max_nfev, "starts": args.polish_starts},
                "polish_model": args.polish_model,
                "truth_nodes": 128, "crosscheck_nodes": 96, "refinement_steps": 3,
                "truth_used_in_calibration": False, "selection": "no model selection in this script",
                "noise": "0 or 1% multiplicative Gaussian, identical observations across arms",
                "holdout": "every third strike, constant across maturities",
                "evidence": "synthetic recovery; does not establish real-market true parameters"}
    (args.out/"manifest.json").write_text(json.dumps(manifest, indent=2))
    rows, parameters, details, fits, truth_rows = [], [], [], [], []
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        truths, regimes = sample_parameters(rng, args.cases_per_seed)
        for i, truth in enumerate(truths):
            case = f"{seed}_{i:02d}"
            truth_rows.append({"case":case, "regime":int(regimes[i]), **dict(zip(CANONICAL,truth))})
            for geometry, days in (("rich", [30.,60.,90.,180.,365.,730.]),("single_expiry",[90.])):
                strikes = np.linspace(.8,1.2,15)
                n = len(days)*len(strikes)
                geo = dict(spot=np.ones(n),strike=np.tile(strikes,len(days)),
                           tau=np.repeat(np.asarray(days)/365.,len(strikes)),
                           rate=np.full(n,.05),carry=np.full(n,.01))
                clean = prices(truth,geo)
                cross = prices(truth,geo,96)
                fit = np.tile(np.arange(len(strikes))%3 != 2,len(days)); hold=~fit
                gf={k:v[fit] for k,v in geo.items()}; gh={k:v[hold] for k,v in geo.items()}
                truth_iv = iv_values(clean[hold],gh)
                for noise in (0.,.01):
                    obs = clean*(1+noise*rng.normal(size=n))
                    b = input_batch(gf,obs[fit],noise)
                    for name,model in models:
                        t0=time.perf_counter()
                        with torch.no_grad(): o=model(b,refine_steps=0)
                        encoder_seconds=time.perf_counter()-t0
                        with torch.no_grad():
                            o["z"],_=model.refine(o["mu_z"],o["L"],b,steps=3)
                            o["params"]=torch.stack(decode(o["z"]),-1)
                        neural_seconds=time.perf_counter()-t0
                        candidates=[("encoder",o["mu_z"][0].numpy(),o["params_pre"][0].numpy(),encoder_seconds),
                                    ("refined",o["z"][0].numpy(),o["params"][0].numpy(),neural_seconds)]
                        if not args.no_polish and (args.polish_model is None or args.polish_model==name):
                            polished=exact_polish(gf,obs[fit],o["z"][0].numpy(),
                                quote_sigma=b["quote_sigma"][0].numpy() if noise else None,
                                node_count=128,max_nfev=args.polish_max_nfev,
                                latent_scale=zsd,n_starts=args.polish_starts,seed=71)
                            fits.append({"case":case,"geometry":geometry,"noise":noise,"model":name,
                                         **{k:v for k,v in polished.items() if k not in ("params","z")}})
                            if "params" in polished:
                                candidates.append(("hybrid_polish",polished["z"],polished["params"],
                                                   neural_seconds+polished["seconds"]))
                            else:
                                candidates.append(("hybrid_polish",np.full(10,np.nan),np.full(10,np.nan),
                                                   neural_seconds+polished["seconds"]))
                        for arm,z,p,seconds in candidates:
                            pred=prices(p,gh)
                            pred_cross=prices(p,gh,96)
                            discount=np.exp(-gh["rate"]*gh["tau"])
                            spot_disc=gh["spot"]*np.exp(-gh["carry"]*gh["tau"])
                            lower=np.maximum(spot_disc-gh["strike"]*discount,0.)
                            bound_fail=int(((pred<lower-1e-10)|(pred>spot_disc+1e-10)).sum())
                            iv=iv_values(pred,gh)
                            fail=int((~np.isfinite(pred)).sum())
                            iv_fail=int((~np.isfinite(iv) | ~np.isfinite(truth_iv)).sum())
                            price_rmse=float(np.sqrt(np.mean((pred-clean[hold])**2))) if not fail else np.inf
                            numerical_difference=float(np.max(np.abs(pred-pred_cross)))
                            truth_difference=float(np.max(np.abs(clean-cross)))
                            zerror=np.abs((z-encode(truth))/zsd)
                            rec={"case":case,"geometry":geometry,"noise":noise,"model":name,"arm":arm,
                                 "n_fit":int(fit.sum()),"n_hold":int(hold.sum()),"failed_prices":fail,
                                 "price_bound_violations":bound_fail,
                                 "invalid_iv_pairs":iv_fail,"price_rmse_spot":price_rmse,
                                 "iv_rmse":float(np.sqrt(np.mean((iv-truth_iv)**2))) if not iv_fail else np.nan,
                                 "z_mae":float(zerror.mean()),"z_max_error":float(zerror.max()),
                                 "parameter_gate":bool(np.isfinite(zerror).all() and zerror.max()<=.01),
                                 "price_gate":bool(price_rmse<=1e-6 and numerical_difference<=1e-6
                                                   and truth_difference<=1e-6 and bound_fail==0),"seconds":seconds,
                                 "truth_quadrature_max_difference":truth_difference,
                                 "predicted_quadrature_max_difference":numerical_difference,
                                 "failed_parameter_case":bool(not np.isfinite(z).all()
                                                               or not np.isfinite(p).all())}
                            rows.append(rec)
                            parameters.extend({**{k:rec[k] for k in ("case","geometry","noise","model","arm")},
                                               "parameter":key,"true":float(truth[j]),"estimated":float(p[j]),
                                               "absolute_error":float(abs(p[j]-truth[j]))}
                                              for j,key in enumerate(CANONICAL))
                            if arm=="hybrid_polish" and np.isfinite(z).all():
                                old_nodes=model.node_count; model.node_count=128
                                diag=model.local_identifiability(torch.tensor(z)[None],b,zsd)[0]
                                model.node_count=old_nodes
                                details.append({"case":case,"geometry":geometry,"noise":noise,"model":name,**diag})
                        print(case,geometry,noise,name,
                              [(r["arm"],f"{r['price_rmse_spot']:.2e}",f"{r['z_max_error']:.3f}")
                               for r in rows[-len(candidates):]],flush=True)
                    pd.DataFrame(rows).to_csv(args.out/"recovery_metrics.csv",index=False)
                    pd.DataFrame(parameters).to_csv(args.out/"parameter_recovery.csv",index=False)
                    pd.DataFrame(truth_rows).to_csv(args.out/"synthetic_truth.csv",index=False)
                    (args.out/"polish_runs.json").write_text(json.dumps(fits,indent=2))
                    (args.out/"identifiability.json").write_text(json.dumps(details,indent=2))
    d=pd.DataFrame(rows)
    summary=[]
    for keys,g in d.groupby(["model","arm","geometry","noise"]):
        summary.append(dict(zip(["model","arm","geometry","noise"],keys))|
            {"cases":len(g),"all_price_gates_pass":bool(g.price_gate.all()),
             "all_parameter_gates_pass":bool(g.parameter_gate.all()),
             "price_gate_passes":int(g.price_gate.sum()),"parameter_gate_passes":int(g.parameter_gate.sum()),
             "pooled_price_rmse":float(np.sqrt(np.sum(g.price_rmse_spot**2*g.n_hold)/g.n_hold.sum())),
             "successful_only_median_z_max_error":float(g.loc[~g.failed_parameter_case,"z_max_error"].median()),
             "failed_parameter_cases":int(g.failed_parameter_case.sum()),
             "failed_prices":int(g.failed_prices.sum()),
             "price_bound_violations":int(g.price_bound_violations.sum()),
             "invalid_iv_pairs":int(g.invalid_iv_pairs.sum()),"median_seconds":float(g.seconds.median())})
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
