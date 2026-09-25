#!/usr/bin/env python3
"""Deliverable E: synthetic benchmark, geometry/noise/regime buckets, uncertainty
calibration, identifiability diagnostics, OOD behaviour, latency, and baselines.

Every price in every reported number comes from the exact Fourier engine. Parameter error
is range-scaled by the TRAINING PRIOR's 1st-99th percentile per parameter (there is no
PARAM_BOX any more), and that scale is written into the output so it can be audited.
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
torch.set_default_dtype(torch.float64)
from src.mentor_dh_pinn.collate import collate
from src.mentor_dh_pinn.ood import assess, build_reference
from src.mentor_dh_pinn.params_v2 import CANONICAL, decode, encode_batch
from src.mentor_dh_pinn.torch_pricer import price_call
from src.mentor_dh_pinn.unified import UnifiedCalibrator
from src.mentor_dh_pinn.baselines import (bs_surface, dh_surface, fit_black_scholes,
                                          fit_double_heston, fit_single_heston, sh_surface)


def load_model(ckpt):
    ck = torch.load(ckpt, weights_only=False); c = ck.get("config", {})
    m = UnifiedCalibrator(d_model=c.get("d_model", 128), rounds=c.get("rounds", 3),
                          node_count=c.get("nodes", 48))
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m, ck


def geo_of(d, i):
    n = int(d["n_quotes"][i])
    return {k: np.asarray(d[k][i, :n], dtype=float)
            for k in ("spot", "strike", "tau", "rate", "carry")}


def reprice_rmse(params, geo, clean):
    m = dh_surface(params, geo)
    return float(np.sqrt(np.mean(((m - clean) / geo["spot"]) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "outputs" / "unified_v6")
    ap.add_argument("--ckpt", type=Path, default=ROOT / "outputs" / "unified_v6" / "unified.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "unified_v6")
    ap.add_argument("--n-full", type=int, default=6000, help="surfaces for network metrics")
    ap.add_argument("--n-baseline", type=int, default=120, help="surfaces for classical arms")
    ap.add_argument("--refine", type=int, default=3)
    a = ap.parse_args()

    te = dict(np.load(a.data / "v6_test.npz", allow_pickle=True))
    tr = dict(np.load(a.data / "v6_train.npz", allow_pickle=True))
    keep = np.where(te["ok"])[0]
    for k in list(te):
        if isinstance(te[k], np.ndarray) and len(te[k]) == len(te["ok"]): te[k] = te[k][keep]
    ref = build_reference(tr, a.out / "ood_reference.json")
    prior = tr["params"][tr["ok"]]
    span = np.array([np.quantile(prior[:, j], 0.99) - np.quantile(prior[:, j], 0.01)
                     for j in range(10)])
    model, ck = load_model(a.ckpt)
    print(f"checkpoint: epoch {ck.get('epoch')} phase {ck.get('phase')}", flush=True)

    z_true = encode_batch(te["params"])
    N = min(a.n_full, len(te["params"]))
    idx = np.arange(N)

    # ---------------- network metrics, batched -------------------------------------
    rows = []
    t_enc = t_ref = 0.0
    for s in range(0, N, 64):
        ii = idx[s:s + 64]
        b = collate(te, ii)
        t0 = time.perf_counter()
        with torch.no_grad(): o0 = model(b, refine_steps=0)
        t_enc += time.perf_counter() - t0
        t0 = time.perf_counter()
        with torch.no_grad(): o = model(b, refine_steps=a.refine)
        t_ref += time.perf_counter() - t0
        Sig = (o["L"] @ o["L"].transpose(-1, -2)).numpy()
        mu = o["mu_z"].numpy(); zz = o["z"].numpy()
        pre = o["params_pre"].numpy(); post = o["params"].numpy()
        for j, i in enumerate(ii):
            g = geo_of(te, i); clean = np.asarray(te["clean"][i, :len(g["tau"])], dtype=float)
            e_pre = np.abs(pre[j] - te["params"][i]) / span
            e_post = np.abs(post[j] - te["params"][i]) / span
            d = z_true[i] - mu[j]
            sd = np.sqrt(np.diag(Sig[j]))
            rows.append({
                "i": int(i), "tag": str(te["tag"][i]), "regime": int(te["regime"][i]),
                "n_quotes": int(te["n_quotes"][i]), "noise": float(te["noise_level"][i]),
                "n_expiries": int(len(np.unique(g["tau"]))),
                "vol_now": float(np.sqrt(te["params"][i][4] + te["params"][i][9])),
                "param_pre": float(np.sqrt(np.mean(e_pre ** 2))),
                "param_post": float(np.sqrt(np.mean(e_post ** 2))),
                "within05": float(np.mean(e_post <= 0.05)),
                "within10": float(np.mean(e_post <= 0.10)),
                "reprice_pre": reprice_rmse(pre[j], g, clean),
                "reprice_post": reprice_rmse(post[j], g, clean),
                "z_err": d.tolist(), "z_sd": sd.tolist(),
                **{f"e_{k}": float(e_post[q]) for q, k in enumerate(CANONICAL)},
            })
    import pandas as pd
    df = pd.DataFrame(rows); df.to_csv(a.out / "unified_test_detail.csv", index=False)

    # ---------------- uncertainty calibration --------------------------------------
    Z = np.array([r["z_err"] for r in rows]); S = np.array([r["z_sd"] for r in rows])
    cover = {}
    for lvl, k in ((50, 0.6744897501960817), (90, 1.6448536269514722), (95, 1.959963984540054)):
        cover[f"coverage_{lvl}"] = float(np.mean(np.abs(Z) <= k * S))
        cover[f"coverage_{lvl}_per_param"] = [float(np.mean(np.abs(Z[:, j]) <= k * S[:, j]))
                                              for j in range(10)]
    cover["median_interval_width_90"] = float(np.median(2 * 1.6449 * S))
    # does predicted uncertainty track realised error?
    from scipy import stats
    cover["spearman_sd_vs_abs_error"] = float(
        stats.spearmanr(S.mean(1), np.abs(Z).mean(1)).statistic)

    # ---------------- buckets -------------------------------------------------------
    def bucket(mask, name):
        d2 = df[mask]
        if not len(d2): return None
        return {"name": name, "n": int(len(d2)),
                "param_pre": float(d2.param_pre.median()),
                "param_post": float(d2.param_post.median()),
                "reprice_pre": float(d2.reprice_pre.median()),
                "reprice_post": float(d2.reprice_post.median()),
                "within10": float(d2.within10.mean())}
    buckets = [b for b in [
        bucket(df.n_expiries == 1, "D single expiry"),
        bucket(df.n_expiries == 2, "E two expiries"),
        bucket(df.n_expiries >= 5, "dense expiries"),
        bucket(df.tag == "historical_5x9", "historical 5x9"),
        bucket(df.vol_now < 0.20, "B low-vol index regime"),
        bucket((df.vol_now >= 0.20) & (df.vol_now < 0.40), "ordinary equity"),
        bucket(df.vol_now >= 0.40, "C high-vol stock regime"),
        bucket(df.noise <= 0.005, "low noise <=0.5%"),
        bucket((df.noise > 0.005) & (df.noise <= 0.02), "moderate noise"),
        bucket(df.noise > 0.02, "J >2% noise"),
        bucket(df.n_quotes <= 10, "H sparse (<=10 quotes)"),
        bucket(df.n_quotes >= 50, "dense (>=50 quotes)"),
    ] if b]

    # ---------------- classical baselines on a subsample ---------------------------
    sub = np.random.default_rng(11).choice(N, min(a.n_baseline, N), replace=False)
    base = []
    for i in sub:
        g = geo_of(te, i); n = len(g["tau"])
        obs = np.asarray(te["noisy"][i, :n], float); cl = np.asarray(te["clean"][i, :n], float)
        rec = {"i": int(i), "n_expiries": int(len(np.unique(g["tau"]))),
               "vol_now": float(np.sqrt(te["params"][i][4] + te["params"][i][9]))}
        t0 = time.perf_counter(); sig = fit_black_scholes(g, obs); rec["bs_s"] = time.perf_counter() - t0
        rec["bs_reprice"] = float(np.sqrt(np.mean(((bs_surface(sig, g) - cl) / g["spot"]) ** 2)))
        t0 = time.perf_counter(); sh = fit_single_heston(g, obs); rec["sh_s"] = time.perf_counter() - t0
        rec["sh_reprice"] = (float(np.sqrt(np.mean(((sh_surface(sh["params"], g) - cl) / g["spot"]) ** 2)))
                             if sh else np.nan)
        t0 = time.perf_counter(); dh = fit_double_heston(g, obs); rec["dh_s"] = time.perf_counter() - t0
        if dh:
            rec["dh_reprice"] = reprice_rmse(dh["params"], g, cl)
            rec["dh_param"] = float(np.sqrt(np.mean((np.abs(dh["params"] - te["params"][i]) / span) ** 2)))
        base.append(rec)
    bdf = pd.DataFrame(base); bdf.to_csv(a.out / "unified_baselines_detail.csv", index=False)
    net_sub = df[df.i.isin(sub)]

    summary = {
        "checkpoint": {"epoch": ck.get("epoch"), "phase": ck.get("phase")},
        "n_surfaces": int(N),
        "range_scale_definition": "training-prior 1st-99th percentile per parameter",
        "range_scale": span.tolist(),
        "parameter_recovery": {
            "pre_refinement_median": float(df.param_pre.median()),
            "post_refinement_median": float(df.param_post.median()),
            "post_refinement_mean": float(df.param_post.mean()),
            "post_refinement_p90": float(df.param_post.quantile(0.90)),
            "fraction_within_0.05": float(df.within05.mean()),
            "fraction_within_0.10": float(df.within10.mean()),
            "per_parameter_median": {k: float(df[f"e_{k}"].median()) for k in CANONICAL},
        },
        "repricing_exact_engine": {
            "pre_refinement_median": float(df.reprice_pre.median()),
            "post_refinement_median": float(df.reprice_post.median()),
            "post_refinement_p95": float(df.reprice_post.quantile(0.95)),
            "post_refinement_worst": float(df.reprice_post.max()),
        },
        "uncertainty": cover,
        "buckets": buckets,
        "latency_seconds_per_surface": {
            "neural_encoder_only": t_enc / N,
            "encoder_plus_%d_physics_steps" % a.refine: t_ref / N,
            "note": "the second number is the latency of the answer actually used",
        },
        "baselines_on_%d_surfaces" % len(bdf): {
            "black_scholes_reprice_median": float(bdf.bs_reprice.median()),
            "single_heston_reprice_median": float(bdf.sh_reprice.median()),
            "double_heston_cold_reprice_median": float(bdf.dh_reprice.median()),
            "double_heston_cold_param_median": float(bdf.dh_param.median()),
            "unified_reprice_median": float(net_sub.reprice_post.median()),
            "unified_param_median": float(net_sub.param_post.median()),
            "seconds": {"black_scholes": float(bdf.bs_s.median()),
                        "single_heston": float(bdf.sh_s.median()),
                        "double_heston_cold": float(bdf.dh_s.median()),
                        "unified_total": t_ref / N},
        },
    }
    (a.out / "unified_evaluation.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2)[:4000], flush=True)


if __name__ == "__main__":
    main()
