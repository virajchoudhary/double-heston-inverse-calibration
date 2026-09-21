#!/usr/bin/env python3
"""Ablations that isolate which components produce the gains.

Honesty note carried into the report: only the ablations that can be run *at inference time*
on the trained model are run here. Those are the ones where the component can be switched off
without changing what was learned:

  C  parameter-communication rounds   -- round weights are shared, so 1 vs 2 vs 3 is a
                                         genuine inference-time switch
  D  covariance vs point output       -- replace Sigma with a scalar multiple of the identity
  E  the spring: full covariance vs diagonal vs global scalar
  I  number of exact-physics refinement steps: 0, 1, 3, 5

The remaining ablations in the brief (A fixed-vector encoder, B pooled head, F noise model,
G parameter prior, H surrogate vs exact physics, J sensitivity-informed routing) each require
training a separate model. At the measured cost of one training run they are not run here,
and the report says so rather than implying they were.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
torch.set_default_dtype(torch.float64)
from src.mentor_dh_pinn.baselines import dh_surface
from src.mentor_dh_pinn.collate import collate
from src.mentor_dh_pinn.params_v2 import encode_batch
from src.mentor_dh_pinn.unified import UnifiedCalibrator


def load(ckpt):
    ck = torch.load(ckpt, weights_only=False); c = ck.get("config", {})
    m = UnifiedCalibrator(d_model=c.get("d_model", 128), rounds=c.get("rounds", 3),
                          node_count=c.get("nodes", 48))
    m.load_state_dict(ck["state_dict"]); m.eval(); return m, ck


def metrics(model, te, idx, span, refine, sigma_mode="full", rounds=None):
    saved = model.rounds
    if rounds is not None: model.rounds = rounds
    P, R, T = [], [], 0.0
    for s in range(0, len(idx), 64):
        ii = idx[s:s + 64]; b = collate(te, ii)
        t0 = time.perf_counter()
        with torch.no_grad():
            h, pad = model.encode(b)
            p, _ = model.tokens_forward(h, pad, len(ii))
            mu, L = model.gaussian_head(p)
            if sigma_mode == "diagonal":
                L = torch.diag_embed(torch.diagonal(L, dim1=-2, dim2=-1))
            elif sigma_mode == "scalar":
                d = torch.diagonal(L, dim1=-2, dim2=-1).mean(-1, keepdim=True)
                L = torch.diag_embed(d.expand(-1, L.shape[-1]))
            z, _ = model.refine(mu, L, b, steps=refine)
            from src.mentor_dh_pinn.params_v2 import decode
            params = torch.stack(decode(z), dim=-1).numpy()
        T += time.perf_counter() - t0
        for j, i in enumerate(ii):
            n = int(te["n_quotes"][i])
            g = {k: np.asarray(te[k][i, :n], float)
                 for k in ("spot", "strike", "tau", "rate", "carry")}
            clean = np.asarray(te["clean"][i, :n], float)
            e = np.abs(params[j] - te["params"][i]) / span
            P.append(float(np.sqrt(np.mean(e ** 2))))
            R.append(float(np.sqrt(np.mean(((dh_surface(params[j], g) - clean) / g["spot"]) ** 2))))
    model.rounds = saved
    return {"param_median": float(np.median(P)), "reprice_median": float(np.median(R)),
            "reprice_mean": float(np.mean(R)), "seconds_per_surface": T / len(idx)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "outputs" / "unified_v6")
    ap.add_argument("--ckpt", type=Path, default=ROOT / "outputs" / "unified_v6" / "unified.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "unified_v6")
    ap.add_argument("--n", type=int, default=1500)
    a = ap.parse_args()
    te = dict(np.load(a.data / "v6_test.npz", allow_pickle=True))
    keep = np.where(te["ok"])[0]
    for k in list(te):
        if isinstance(te[k], np.ndarray) and len(te[k]) == len(te["ok"]): te[k] = te[k][keep]
    tr = dict(np.load(a.data / "v6_train.npz", allow_pickle=True))
    prior = tr["params"][tr["ok"]]
    span = np.array([np.quantile(prior[:, j], .99) - np.quantile(prior[:, j], .01)
                     for j in range(10)])
    model, ck = load(a.ckpt)
    idx = np.arange(min(a.n, len(te["params"])))
    res = {"checkpoint": {"epoch": ck.get("epoch"), "phase": ck.get("phase")},
           "n_surfaces": int(len(idx)), "ablations": {}}

    print("I — exact-physics refinement steps", flush=True)
    for r in (0, 1, 3, 5):
        m = metrics(model, te, idx, span, refine=r)
        res["ablations"][f"I_refine_{r}"] = m
        print(f"   steps {r}: param {m['param_median']:.5f}  reprice {m['reprice_median']:.4e}"
              f"  {m['seconds_per_surface']*1000:.1f} ms/surface", flush=True)

    print("E — the spring: full covariance vs diagonal vs global scalar", flush=True)
    for mode in ("full", "diagonal", "scalar"):
        m = metrics(model, te, idx, span, refine=3, sigma_mode=mode)
        res["ablations"][f"E_spring_{mode}"] = m
        print(f"   {mode:<9}: param {m['param_median']:.5f}  reprice {m['reprice_median']:.4e}",
              flush=True)

    print("C — parameter-communication rounds (weights shared, so switchable at inference)",
          flush=True)
    for r in (1, 2, 3):
        m = metrics(model, te, idx, span, refine=3, rounds=r)
        res["ablations"][f"C_rounds_{r}"] = m
        print(f"   rounds {r}: param {m['param_median']:.5f}  reprice {m['reprice_median']:.4e}",
              flush=True)

    res["not_run"] = {
        "A_fixed_vector_encoder": "requires training a separate model",
        "B_pooled_head_vs_parameter_tokens": "requires training a separate model",
        "F_iid_noise_vs_correlated": "requires regenerating data and retraining",
        "G_old_prior_vs_regime_balanced": "requires regenerating data and retraining",
        "H_surrogate_vs_exact_physics": "requires training a surrogate and a second model",
        "J_sensitivity_informed_routing": "requires training a separate model",
    }
    (a.out / "ablations.json").write_text(json.dumps(res, indent=2))
    print("\n" + json.dumps(res["ablations"], indent=2), flush=True)


if __name__ == "__main__":
    main()
