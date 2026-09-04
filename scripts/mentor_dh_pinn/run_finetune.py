#!/usr/bin/env python3
"""Run semi-supervised projection fine-tuning. Works with or without PyTorch Lightning."""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
torch.set_default_dtype(torch.float64)
from src.mentor_dh_pinn.collate import collate
from src.mentor_dh_pinn.finetune_projection import ProjectionFineTuner
from src.mentor_dh_pinn.params_v2 import decode, encode_batch
from src.mentor_dh_pinn.torch_pricer import price_call
from src.mentor_dh_pinn.unified import UnifiedCalibrator


def load_real(path: Path) -> dict:
    d = dict(np.load(path, allow_pickle=True))
    return {k: v for k, v in d.items()}


def real_batch(d: dict, idx: np.ndarray) -> dict:
    n = d["n_quotes"][idx]; m = int(n.max())
    t = lambda k: torch.tensor(d[k][idx][:, :m])
    b = {k: t(k) for k in ("spot", "strike", "tau", "rate", "carry",
                           "price", "vega", "quote_sigma", "mask")}
    b["n_quotes"] = torch.tensor(n)
    b["noise_level"] = torch.tensor(d["iv_noise"][idx])
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=ROOT/"outputs"/"unified_v6"/"unified.pt")
    ap.add_argument("--syn", type=Path, default=ROOT/"outputs"/"unified_v6")
    ap.add_argument("--real", type=Path, default=ROOT/"outputs"/"real_corpus")
    ap.add_argument("--out", type=Path, default=ROOT/"outputs"/"unified_v6")
    ap.add_argument("--tag", default="unified_ft")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch-syn", type=int, default=48)
    ap.add_argument("--batch-real", type=int, default=24)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--w-proj", type=float, default=1.0)
    ap.add_argument("--w-sc", type=float, default=0.1)
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--val-every", type=int, default=50)
    a = ap.parse_args()

    ck = torch.load(a.ckpt, weights_only=False); c = ck["config"]
    enc = UnifiedCalibrator(d_model=c["d_model"], rounds=c["rounds"], node_count=c["nodes"])
    enc.load_state_dict(ck["state_dict"])

    st = json.loads((a.syn/"latent_standardisation.json").read_text())
    model = ProjectionFineTuner(
        enc, decode, price_call,
        torch.tensor(st["mean"]), torch.tensor(st["sd"]),
        w_projection=a.w_proj, w_self_consistency=a.w_sc,
        n_consistency_draws=a.draws, lr=a.lr, max_steps=a.steps, node_count=c["nodes"])

    syn = dict(np.load(a.syn/"v6_train.npz", allow_pickle=True))
    keep = np.where(syn["ok"])[0]
    for k in list(syn):
        if isinstance(syn[k], np.ndarray) and len(syn[k]) == len(syn["ok"]): syn[k] = syn[k][keep]
    z_syn = torch.tensor(encode_batch(syn["params"]))
    syn_v = dict(np.load(a.syn/"v6_validation.npz", allow_pickle=True))
    kv = np.where(syn_v["ok"])[0]
    for k in list(syn_v):
        if isinstance(syn_v[k], np.ndarray) and len(syn_v[k]) == len(syn_v["ok"]): syn_v[k] = syn_v[k][kv]
    z_syn_v = torch.tensor(encode_batch(syn_v["params"]))

    rtr = load_real(a.real/"real_train.npz"); rva = load_real(a.real/"real_validation.npz")
    n_syn, n_real, n_rva = len(syn["params"]), len(rtr["n_quotes"]), len(rva["n_quotes"])
    print(f"synthetic {n_syn:,} | real train {n_real:,} | real val {n_rva:,}", flush=True)

    opt_cfg = model.configure_optimizers()
    opt, sched = opt_cfg["optimizer"], opt_cfg["lr_scheduler"]["scheduler"]
    rng = np.random.default_rng(3)

    def validate():
        model.eval(); pv, sv, cv, av = [], [], [], []
        with torch.no_grad():
            for s in range(0, min(n_rva, 600), 48):
                ii = np.arange(s, min(s+48, n_rva))
                b = real_batch(rva, ii)
                mu, L = model._encode(b)
                _, lp = model.projection_loss(b, mu)
                pv.append(float(lp["proj_iv_rmse"]))
            for s in range(0, 480, 48):
                ii = np.arange(s, s+48)
                b = collate(syn_v, ii); b["z_true"] = z_syn_v[ii]
                mu, L = model._encode(b)
                sv.append(float((model._std(mu)-model._std(b["z_true"])).abs().mean()))
                sd = torch.sqrt(torch.diagonal(L@L.transpose(-1,-2), dim1=-2, dim2=-1))
                cv.append(float(((b["z_true"]-mu).abs() <= 1.6448536*sd).double().mean()))
                from src.mentor_dh_pinn.finetune_projection import gaussian_nll_latent
                av.append(float(gaussian_nll_latent(b["z_true"], mu, L).mean()))
        model.train()
        return float(np.mean(pv)), float(np.mean(sv)), float(np.mean(cv)), float(np.mean(av))

    p0, s0, c0, a0 = validate()
    print(f"before: real IV RMSE {p0:.5f} | syn z-MAE {s0:.4f} | syn cov90 {c0:.3f} "
          f"| anchor NLL {a0:.3f}", flush=True)
    best, hist, t0 = p0, [], time.time()
    torch.save({"state_dict": enc.state_dict(), "config": c, "step": 0,
                "real_iv_rmse": p0}, a.out/f"{a.tag}.pt")

    model.train()
    for step in range(a.steps):
        si = rng.choice(n_syn, a.batch_syn, replace=False)
        ri = rng.choice(n_real, a.batch_real, replace=False)
        sb = collate(syn, si); sb["z_true"] = z_syn[si]
        rb = real_batch(rtr, ri)
        opt.zero_grad(set_to_none=True)
        loss, logs = model.compute_losses(sb, rb)
        if not torch.isfinite(loss):
            model.skipped += 1; sched.step(); continue
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        if not torch.isfinite(gn):
            opt.zero_grad(set_to_none=True); model.skipped += 1; sched.step(); continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), model.hp["grad_clip"])
        opt.step(); sched.step()
        if (step+1) % a.val_every == 0:
            p, s, cc, an = validate()
            rec = {"step": step+1, "loss": float(loss.detach()), "proj": float(logs["projection"]),
                   "sc": float(logs["self_consistency"]), "anchor": float(logs["anchor"]),
                   "real_iv_rmse": p, "syn_z_mae": s, "syn_cov90": cc, "syn_anchor_nll": an,
                   "grad_norm": float(gn), "lr": sched.get_last_lr()[0],
                   "skipped": model.skipped, "seconds": time.time()-t0}
            hist.append(rec)
            flag = ""
            if p < best:
                best = p; flag = "  <- best, saved"
                torch.save({"state_dict": enc.state_dict(), "config": c,
                            "step": step+1, "real_iv_rmse": p}, a.out/f"{a.tag}.pt")
            print(f"  {step+1:>4} loss {float(loss.detach()):7.4f} proj {float(logs['projection']):7.4f} "
                  f"sc {float(logs['self_consistency']):6.3f} | real IV {p:.5f} "
                  f"syn z-MAE {s:.4f} cov90 {cc:.3f} anchorNLL {an:6.2f} | "
                  f"|g| {float(gn):8.2f} skip {model.skipped} | {time.time()-t0:.0f}s{flag}",
                  flush=True)
            (a.out/f"{a.tag}_history.json").write_text(json.dumps(hist, indent=2))
    print(f"\nDONE in {time.time()-t0:.0f}s. best real IV RMSE {best:.5f} (was {p0:.5f}, "
          f"{(1-best/p0)*100:+.1f}%)", flush=True)


if __name__ == "__main__":
    main()
