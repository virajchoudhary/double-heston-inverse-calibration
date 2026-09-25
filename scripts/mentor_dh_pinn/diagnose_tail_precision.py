#!/usr/bin/env python3
"""Independent arbitrary-precision diagnostic of IV failures; never changes fitted models.

Carr–Madan damped inversion is evaluated at two decimal precisions. Original failed
float64 prices remain in the evaluation table. No price is clipped or imputed.
"""
import json
import sys
import time
from pathlib import Path
import pandas as pd
import mpmath as mp

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.mentor_dh_pinn.evaluate_repair_validation import digest, summarize
OUT=ROOT/"outputs/calibration_repair"
NAMES=("kappa_slow","theta_slow","sigma_slow","rho_slow","v0_slow",
       "kappa_fast","theta_fast","sigma_fast","rho_fast","v0_fast")


def price_and_iv(p,f,k,t,dps):
    if len(p)!=10 or f<=0 or k<=0 or t<=0 or dps<30:
        raise ValueError("Ten parameters, positive geometry, and >=30 decimal digits required")
    with mp.workdps(dps):
        p=list(map(lambda x:mp.mpf(str(x)),p))
        f,k,t=(mp.mpf(str(x)) for x in (f,k,t))
        def factor(u,pars):
            kap,theta,sigma,rho,v0=pars
            b=kap-rho*sigma*1j*u
            d=mp.sqrt(b*b+sigma*sigma*(u*u+1j*u))
            g=(b-d)/(b+d);e=mp.exp(-d*t)
            return kap*theta/(sigma*sigma)*((b-d)*t-2*mp.log((1-g*e)/(1-g))) \
                +v0*(b-d)/(sigma*sigma)*(1-e)/(1-g*e)
        def cf(u):
            return mp.exp(1j*u*mp.log(f)+factor(u,p[:5])+factor(u,p[5:]))
        a=mp.mpf("1.5")
        def integrand(u):
            return mp.re(mp.exp(-1j*u*mp.log(k))*cf(u-(a+1)*1j)
                         /(a*a+a-u*u+1j*(2*a+1)*u))
        cuts=list(range(0,501,25))+[750,1000,1500,2000]
        integrals=[mp.quad(integrand,[l,r]) for l,r in zip(cuts,cuts[1:])]
        price=mp.exp(-a*mp.log(k))*mp.fsum(integrals)/mp.pi
        if not max(f-k,0)<price<f:
            return {"dps":dps,"price":str(price),"iv":None,"status":"out_of_bounds"}
        def black(vol):
            st=vol*mp.sqrt(t);d=mp.log(f/k)/st+st/2
            return f*mp.erfc(-d/mp.sqrt(2))/2-k*mp.erfc(-(d-st)/mp.sqrt(2))/2
        lo,hi=mp.mpf("0.000001"),mp.mpf(8)
        if not black(lo)<price<black(hi):
            return {"dps":dps,"price":str(price),"iv":None,"status":"iv_not_bracketed"}
        for _ in range(180):
            mid=(lo+hi)/2
            if black(mid)>price:hi=mid
            else:lo=mid
        return {"dps":dps,"price":str(price),"iv":str((lo+hi)/2),
                "tail_interval_contribution":str(integrals[-1]*mp.exp(-a*mp.log(k))/mp.pi),
                "status":"finite_price_and_iv"}


def main():
    destination=OUT/"market_precision_checked"
    destination.mkdir(exist_ok=False)
    inputs=[OUT/"market_validation"/name for name in
            ("holdout_predictions.csv","surface_parameters_and_metrics.csv","summary.json")]
    hashes={str(p.relative_to(ROOT)):digest(p) for p in inputs}
    quotes=pd.read_csv(inputs[0])
    quotes["inference_error"]=quotes.inference_error.fillna("")
    surfaces=pd.read_csv(inputs[1])
    summary=json.loads(inputs[2].read_text())
    # A uniform numerical guard across ALL models, independent of residual accuracy.
    guard=(quotes.model_price_128_nodes-quotes.no_arbitrage_lower<=1e-10)|(quotes.status!="ok")
    policy={"evidence":"Numerical re-evaluation, not a new model or fresh market assessment",
            "guard":"Model time value <= 1e-10 normalized, or invalid original prediction; all models",
            "guard_chosen_after":"Tracing float64 cancellation in three original validation predictions",
            "method":"Independent Carr-Madan inversion, damping 1.5, integration 0..2000",
            "agreement":"40/60 decimal prices within 1e-12 relative; IV within 1e-10; last interval <1e-8 of price",
            "failure_policy":"Keep every quote; unresolved required IV makes the all-quote metric null",
            "source_sha256":{str(Path(__file__).relative_to(ROOT)):digest(__file__)},
            "input_sha256":hashes,"guarded_quotes":int(guard.sum())}
    (destination/"precision_protocol.json").write_text(json.dumps(policy,indent=2))
    quotes["original_status"]=quotes.status
    quotes["original_model_iv_inverted"]=quotes.model_iv_inverted
    quotes["original_iv_error"]=quotes.iv_error
    quotes["model_price_reevaluated"]=quotes.model_price_128_nodes
    quotes["pricing_method"]="128-node float64"
    records=[]
    started=time.perf_counter()
    for row in quotes[guard].itertuples():
        p=surfaces[(surfaces.model==row.model)&(surfaces.label==row.label)].iloc[0]
        rec={"model":row.model,"label":row.label,"quote_index":row.quote_index,
             "original_float64_price":row.model_price_128_nodes,"original_status":row.status,
             "observed_price":row.observed_price,"observed_iv":row.market_iv_inverted,"runs":[]}
        for dps in (40,60):
            run=price_and_iv([p[k] for k in NAMES],row.forward_normalized,
                             row.strike_normalized,row.tau_years,dps)
            rec["runs"].append(run);print(row.model,dps,run,flush=True)
        with mp.workdps(60):
            a,b=rec["runs"]
            valid=a["status"]==b["status"]=="finite_price_and_iv"
            if valid:
                relative=abs(mp.mpf(a["price"])-mp.mpf(b["price"]))/abs(mp.mpf(b["price"]))
                tail=max(abs(mp.mpf(x["tail_interval_contribution"]))/abs(mp.mpf(x["price"])) for x in (a,b))
                valid=bool(relative<=mp.mpf("1e-12") and tail<mp.mpf("1e-8")
                           and abs(mp.mpf(a["iv"])-mp.mpf(b["iv"]))<mp.mpf("1e-10"))
                rec.update(relative_precision_difference=str(relative),last_interval_relative=str(tail))
        rec["precision_check_pass"]=valid
        if valid:
            price,iv=float(b["price"]),float(b["iv"])
            vega=(row.model_price_128_nodes-row.observed_price)/row.first_order_vega_error
            updates={"model_price_reevaluated":price,"model_iv_inverted":iv,
                     "iv_error":iv-row.market_iv_inverted,"price_error":price-row.observed_price,
                     "first_order_vega_error":(price-row.observed_price)/vega,
                     "nonfinite_price":False,"no_arbitrage_violation":False,"iv_valid":True,
                     "status":"ok","pricing_method":"Carr-Madan 60 dps, checked against 40 dps"}
        else:
            updates={"status":"precision_check_failed","iv_valid":False,"iv_error":float("nan")}
        for key,value in updates.items():quotes.at[row.Index,key]=value
        records.append(rec)
        (destination/"precision_diagnostics.json").write_text(json.dumps(records,indent=2))
    quotes.to_csv(destination/"holdout_predictions.csv",index=False)
    for model in summary["models"]:
        subset=quotes[quotes.model==model["name"]]
        model["original_128_node_iv_rmse"]=model["iv_rmse_all_quotes"]
        model["original_128_node_invalid_quotes"]=model["invalid_iv_quotes"]
        model["precision_reevaluated_quotes"]=int((subset.pricing_method!="128-node float64").sum())
        model.update(summarize(subset.to_dict("records")))
        model["by_symbol"]={symbol:summarize(group.to_dict("records"))
                            for symbol,group in subset.groupby("symbol")}
    if any(digest(ROOT/p)!=h for p,h in hashes.items()):
        raise RuntimeError("Original evaluation changed during precision check")
    summary.update(precision_check=policy,additional_precision_seconds=time.perf_counter()-started,
                   precision_diagnostics_sha256=digest(destination/"precision_diagnostics.json"),
                   reevaluated_predictions_sha256=digest(destination/"holdout_predictions.csv"))
    (destination/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps({m["name"]:m["iv_rmse_all_quotes"] for m in summary["models"]},indent=2))


if __name__=="__main__":main()
