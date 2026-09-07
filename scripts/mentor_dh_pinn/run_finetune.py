#!/usr/bin/env python3
"""Run semi-supervised projection fine-tuning. Works with or without PyTorch Lightning."""
from __future__ import annotations
import argparse, hashlib, json, math, platform, sys, time
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


def validation_batches(d, idx):
    """Keep all quote-derived calibration inputs outside the holdout scoring batch."""
    target = real_batch(d, idx)
    if "holdout_mask" not in d:
        raise ValueError("Validation corpus requires explicit holdout_mask; rebuild it.")
    hold = torch.tensor(d["holdout_mask"][idx][:, :target["mask"].shape[1]], dtype=torch.float64)
    fit = dict(target)
    fit["mask"] = target["mask"] * (1.0 - hold)
    fit["n_quotes"] = fit["mask"].sum(-1).to(torch.int64)
    target["mask"] = target["mask"] * hold
    if (fit["n_quotes"] < 3).any() or (target["mask"].sum(-1) < 1).any():
        raise ValueError("Each validation surface needs >=3 fit quotes and a holdout.")
    # Replace held-out features before encoding: a mask alone still evaluates their logs.
    first = fit["mask"].argmax(-1, keepdim=True)
    for key in ("spot", "strike", "tau", "rate", "carry", "price", "vega", "quote_sigma"):
        replacement = fit[key].gather(1, first).expand_as(fit[key])
        fit[key] = torch.where(fit["mask"] > 0.5, fit[key], replacement)
    return fit, target


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
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--refine", type=int, default=3)
    ap.add_argument("--syn-val-cap", type=int, default=480)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out / f"{a.tag}_run.json").exists():
        raise FileExistsError("Use a new output directory/tag; prior run evidence is preserved.")
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.use_deterministic_algorithms(True)
    inputs = [a.ckpt, a.syn/"latent_standardisation.json", a.syn/"v6_train.npz",
              a.syn/"v6_validation.npz", a.real/"real_train.npz", a.real/"real_validation.npz"]
    sources = [Path(__file__), ROOT/"src/mentor_dh_pinn/finetune_projection.py",
               ROOT/"src/mentor_dh_pinn/unified.py", ROOT/"src/mentor_dh_pinn/torch_pricer.py",
               ROOT/"src/mentor_dh_pinn/params_v2.py", ROOT/"src/mentor_dh_pinn/collate.py"]
    digest = lambda p: hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
    run = {"arguments": {k: str(v) if isinstance(v, Path) else v for k,v in vars(a).items()},
           "input_sha256": {str(p): digest(p) for p in inputs},
           "code_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
           "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                       "numpy": np.__version__, "platform": platform.platform()},
           "selection": "pooled heldout vega-price RMSE after refinement, synthetic gates",
           "evidence": "development; inherited base-checkpoint exposures remain"}
    (a.out/f"{a.tag}_run.json").write_text(json.dumps(run, indent=2))

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
    syn = {k: v[keep] for k, v in syn.items()}
    z_syn = torch.tensor(encode_batch(syn["params"]))
    syn_v = dict(np.load(a.syn/"v6_validation.npz", allow_pickle=True))
    kv = np.where(syn_v["ok"])[0]
    syn_v = {k: v[kv] for k, v in syn_v.items()}
    z_syn_v = torch.tensor(encode_batch(syn_v["params"]))

    rtr = load_real(a.real/"real_train.npz"); rva = load_real(a.real/"real_validation.npz")
    n_syn, n_real, n_rva = len(syn["params"]), len(rtr["n_quotes"]), len(rva["n_quotes"])
    print(f"synthetic {n_syn:,} | real train {n_real:,} | real val {n_rva:,}", flush=True)

    opt_cfg = model.configure_optimizers()
    opt, sched = opt_cfg["optimizer"], opt_cfg["lr_scheduler"]["scheduler"]
    rng = np.random.default_rng(a.seed)

    def validate():
        model.eval(); sv, cv, av = [], [], []
        sums = {"pre": 0., "post": 0.}; count = 0; failed = {"pre": 0, "post": 0}
        strata = {}
        with torch.no_grad():
            for s in range(0, n_rva, 16):
                ii = np.arange(s, min(s+16, n_rva))
                b, target = validation_batches(rva, ii)
                o = enc(b, refine_steps=a.refine)
                active = target["mask"] > .5
                count += int(active.sum())
                for name, z in (("pre", o["mu_z"]), ("post", o["z"])):
                    px = model._price(z, target)
                    err = (px-target["price"])/target["vega"]
                    failed[name] += int((~torch.isfinite(err) & active).sum())
                    sums[name] += float((err[active]**2).sum())
                    if name == "post":
                        for j, row in enumerate(ii):
                            symbol = str(rva["label"][row]).split("|")[0]
                            stat = strata.setdefault(symbol, [0., 0])
                            stat[0] += float((err[j, active[j]]**2).sum())
                            stat[1] += int(active[j].sum())
            for s in range(0, min(a.syn_val_cap, len(syn_v["params"])), 48):
                ii = np.arange(s, min(s+48, a.syn_val_cap, len(syn_v["params"])))
                b = collate(syn_v, ii); b["z_true"] = z_syn_v[ii]
                mu, L = model._encode(b)
                sv.extend((model._std(mu)-model._std(b["z_true"])).abs().mean(-1).tolist())
                sd = torch.sqrt(torch.diagonal(L@L.transpose(-1,-2), dim1=-2, dim2=-1))
                cv.extend(((b["z_true"]-mu).abs() <= 1.6448536*sd).double().mean(-1).tolist())
                from src.mentor_dh_pinn.finetune_projection import gaussian_nll_latent
                av.extend(gaussian_nll_latent(b["z_true"], mu, L).tolist())
        model.train()
        return {"real_iv_rmse": math.sqrt(sums["post"]/count) if not failed["post"] else math.inf,
                "pre_real_iv_rmse": math.sqrt(sums["pre"]/count) if not failed["pre"] else math.inf,
                "syn_z_mae": float(np.mean(sv)), "syn_cov90": float(np.mean(cv)),
                "syn_anchor_nll": float(np.mean(av)), "n_quotes": count,
                "failed_predictions": failed,
                "symbol_rmse": {k: math.sqrt(v[0]/v[1]) for k,v in strata.items()}}

    initial = validate()
    p0, s0, c0, a0 = (initial[k] for k in ("real_iv_rmse", "syn_z_mae", "syn_cov90", "syn_anchor_nll"))
    print(f"before: real IV RMSE {p0:.5f} | syn z-MAE {s0:.4f} | syn cov90 {c0:.3f} "
          f"| anchor NLL {a0:.3f}", flush=True)
    initial_eligible = math.isfinite(p0) and .85 <= c0 <= .95
    best, hist, t0 = p0 if initial_eligible else math.inf, [], time.time()
    def save_checkpoint(step, metrics, selected=False):
        checkpoint = {"state_dict": enc.state_dict(), "config": c, "step": step,
                      "seed": a.seed, "metrics": metrics,
                      "real_iv_rmse": metrics["real_iv_rmse"], "objective": "negative_elbo",
                      "refine_steps": a.refine, "loss_ema": model._loss_ema,
                      "optimizer_state": opt.state_dict(), "scheduler_state": sched.state_dict(),
                      "torch_rng_state": torch.get_rng_state(), "numpy_rng_state": rng.bit_generator.state}
        torch.save(checkpoint, a.out/f"{a.tag}_step_{step:04d}.pt")
        if selected:
            torch.save(checkpoint, a.out/f"{a.tag}.pt")
    initial["eligible"] = initial_eligible
    save_checkpoint(0, initial, selected=initial_eligible)
    (a.out/f"{a.tag}_initial.json").write_text(json.dumps(initial, indent=2))

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
        if (step+1) % a.val_every == 0 or step+1 == a.steps:
            metrics = validate()
            p, s, cc, an = (metrics[k] for k in ("real_iv_rmse", "syn_z_mae", "syn_cov90", "syn_anchor_nll"))
            rec = {"step": step+1, "loss": float(loss.detach()), "proj": float(logs["projection"]),
                   "sc": float(logs["self_consistency"]), "anchor": float(logs["anchor"]),
                   **metrics,
                   "grad_norm": float(gn), "lr": sched.get_last_lr()[0],
                   "skipped": model.skipped, "seconds": time.time()-t0}
            hist.append(rec)
            flag = ""
            eligible = math.isfinite(p) and s <= s0*1.05 and .85 <= cc <= .95
            rec["eligible"] = eligible
            save_checkpoint(step+1, metrics, selected=eligible and p < best)
            if eligible and p < best:
                best = p; flag = "  <- best, saved"
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
