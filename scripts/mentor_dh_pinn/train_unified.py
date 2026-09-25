#!/usr/bin/env python3
"""Train the unified physics-informed set calibrator, on the staged curriculum.

Objective (weights adapted, not hand-picked -- see `balance`):

    L = w_par L_parameter + w_unc L_uncertainty + w_phy L_clean_physics + w_ref L_refined

`L_clean_physics` and `L_refined` are computed with the EXACT Fourier pricer, never a
surrogate: the earlier study measured a 35% better surrogate producing *worse* parameter
recovery, so surrogate value accuracy is not a safe training criterion.

The curriculum exists because turning every objective on at full strength from epoch 1
reproduces the loss-domination failure this whole redesign is meant to remove.
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
torch.set_default_dtype(torch.float64)
from src.mentor_dh_pinn.collate import collate
from src.mentor_dh_pinn.params_v2 import decode, encode_batch, CANONICAL
from src.mentor_dh_pinn.torch_pricer import price_call
from src.mentor_dh_pinn.unified import UnifiedCalibrator, N_PARAMS

GROUPS = {"v0": (4, 9), "theta": (1, 6), "kappa": (0, 5), "sigma": (2, 7), "rho": (3, 8)}
# latent index groups, for the gradient diagnostic (z-coordinates, see params_v2)
Z_GROUPS = {"v0": (4, 5), "theta": (2, 3), "kappa": (0, 1), "eta": (6, 7), "rho": (8, 9)}

PHASES = [
    # name, epochs, use_noisy, refine, w_phy, w_ref, backprop_refine
    ("A_clean_inference",   3, False, 0, 0.0, 0.0, False),
    ("B_noise_robustness",  4, True,  0, 0.0, 0.0, False),
    ("C_physics",           4, True,  0, 1.0, 0.0, False),
    ("D_refinement",        3, True,  2, 1.0, 1.0, True),
]
import os
if os.environ.get("SMOKE"):
    PHASES = [(n, 1, un, rf, wp, wr, bp) for (n, e, un, rf, wp, wr, bp) in PHASES]


def masked_mse(pred, target, mask, scale):
    """Mean squared normalised residual over the real quotes.

    Non-finite prices are EXCLUDED rather than allowed to poison the batch. The
    characteristic function genuinely overflows for extreme parameters -- the production
    NumPy engine raises there -- and early in training the model will occasionally emit such
    a vector. Letting one surface NaN the batch loss stopped every subsequent update in the
    first training run: from epoch 9 on, every batch was skipped and nothing learned.

    Returns (loss, fraction_unpriceable) so the caller can penalise the region explicitly
    instead of making it free.
    """
    good = torch.isfinite(pred)
    m = mask * good.to(mask.dtype)
    r = torch.where(good, (pred - target) / scale, torch.zeros_like(target))
    loss = ((r ** 2) * m).sum() / m.sum().clamp(min=1.0)
    frac_bad = ((mask * (~good).to(mask.dtype)).sum() / mask.sum().clamp(min=1.0))
    return loss, frac_bad


def gaussian_nll(z_true, mu, L):
    """-log N(z* | mu, L L^T), using the Cholesky factor directly."""
    d = (z_true - mu).unsqueeze(-1)
    sol = torch.linalg.solve_triangular(L, d, upper=False)
    quad = (sol ** 2).sum((-2, -1))
    logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
    return 0.5 * (quad + logdet).mean()


def exact_prices(params, b, node_count):
    return price_call(params, b["spot"], b["strike"], b["tau"], b["rate"], b["carry"],
                      node_count=node_count)


class Balance:
    """Standardise each objective by its own running scale, then apply the phase weights.

    A single fixed weighted sum is what let high-sensitivity directions dominate before;
    dividing by an EMA of each term puts them on a common scale so the phase weights mean
    what they say. The ReLoBRaLo variant is available for the ablation.
    """
    def __init__(self, keys, beta=0.98, mode="standardise"):
        self.ema = {k: None for k in keys}; self.beta = beta; self.mode = mode
    def __call__(self, losses):
        out = {}
        for k, v in losses.items():
            x = float(v.detach())
            self.ema[k] = x if self.ema[k] is None else self.beta * self.ema[k] + (1 - self.beta) * x
            out[k] = v / max(abs(self.ema[k]), 1e-12) if self.mode == "standardise" else v
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "outputs" / "unified_v6")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "unified_v6")
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--nodes", type=int, default=48)
    ap.add_argument("--train-cap", type=int, default=0)
    ap.add_argument("--val-cap", type=int, default=1500)
    ap.add_argument("--refine-batches", type=int, default=120, help="batches/epoch in phase D")
    ap.add_argument("--tag", default="unified")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)

    tr = dict(np.load(a.data / "v6_train.npz", allow_pickle=True))
    va = dict(np.load(a.data / "v6_validation.npz", allow_pickle=True))
    for d in (tr, va):
        keep = np.where(d["ok"])[0]
        for k in list(d):
            if isinstance(d[k], np.ndarray) and len(d[k]) == len(d["ok"]):
                d[k] = d[k][keep]
    if a.train_cap: 
        for k in list(tr):
            if isinstance(tr[k], np.ndarray) and len(tr[k]) == len(tr["params"]):
                tr[k] = tr[k][:a.train_cap]
    # Standardise the latent coordinates for the loss. The brief requires parameter losses
    # to operate in standardised coordinates, and the audit of the first attempt showed why:
    # one badly scaled coordinate dominated the objective and the model barely beat a
    # constant predictor. The standardisation is a fixed affine map from the TRAINING set.
    _z_tr_raw = encode_batch(tr["params"])
    Z_MEAN = _z_tr_raw.mean(0); Z_SD = _z_tr_raw.std(0)
    (a.out / "latent_standardisation.json").write_text(json.dumps(
        {"mean": Z_MEAN.tolist(), "sd": Z_SD.tolist()}, indent=2))
    std = lambda z: (z - torch.tensor(Z_MEAN)) / torch.tensor(Z_SD)
    z_tr = torch.tensor(_z_tr_raw)
    z_va = torch.tensor(encode_batch(va["params"]))
    n = len(tr["params"]); print(f"train {n}  validation {len(va['params'])}", flush=True)

    model = UnifiedCalibrator(d_model=a.d_model, rounds=a.rounds, node_count=a.nodes)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-6)
    total_epochs = sum(p[1] for p in PHASES)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_epochs, eta_min=3e-6)
    bal = Balance(["par", "unc", "phy", "ref"])
    rng = np.random.default_rng(7)
    n_quotes_tr = tr["n_quotes"]

    def batches(pool, size):
        """Length-bucketed batches: group surfaces of similar quote count, then shuffle the
        batch order. Padding a 5-quote surface out to 85 wastes ~3x the attention compute."""
        pool = pool[np.argsort(n_quotes_tr[pool], kind="stable")]
        chunks = [pool[i:i + size] for i in range(0, len(pool), size)]
        rng.shuffle(chunks)
        return [c for c in chunks if len(c) >= 8]
    hist = []; t0 = time.time(); ep_global = 0; best_val = float("inf")

    va_idx = np.arange(min(a.val_cap, len(va["params"])))

    @torch.no_grad()
    def validate():
        model.eval(); errs = []; rep = []
        for s in range(0, len(va_idx), 64):
            idx = va_idx[s:s + 64]
            b = collate(va, idx); zt = z_va[idx]
            o = model(b, refine_steps=0)
            errs.append((std(o["mu_z"]) - std(zt)).abs())
            pr = exact_prices(o["params_pre"], b, a.nodes)
            l, _ = masked_mse(pr, b["clean"], b["mask"], b["spot"])
            rep.append(torch.sqrt(l).reshape(1))
        e = torch.cat(errs); model.train()
        return float(e.mean()), float(torch.cat(rep).mean())

    for (name, epochs, use_noisy, refine, w_phy, w_ref, bp) in PHASES:
        print(f"\n=== PHASE {name}  epochs {epochs}  noisy {use_noisy}  refine {refine} ===",
              flush=True)
        for ep in range(epochs):
            model.train(); pool = rng.permutation(n)
            if refine:
                pool = pool[:a.refine_batches * a.batch]
            acc = {}; steps = 0; n_skip = 0; gnorm = {k: 0.0 for k in Z_GROUPS}
            phy_ramp = min(1.0, (ep + 1) / 2.0) if w_phy else 0.0
            for idx in batches(pool, a.batch):
                b = collate(tr, idx, use_noisy=use_noisy); zt = z_tr[idx]
                opt.zero_grad(set_to_none=True)
                o = model(b, refine_steps=refine, create_graph=bp)
                o["mu_z"].retain_grad()
                losses = {
                    "par": torch.nn.functional.huber_loss(std(o["mu_z"]), std(zt), delta=1.0),
                    "unc": gaussian_nll(zt, o["mu_z"], o["L"]),
                }
                bad_frac = torch.zeros((), dtype=torch.float64)
                if w_phy:
                    l, bf = masked_mse(exact_prices(o["params_pre"], b, a.nodes),
                                       b["clean"], b["mask"], b["spot"])
                    losses["phy"] = l; bad_frac = bad_frac + bf
                if w_ref and refine:
                    l, bf = masked_mse(exact_prices(o["params"], b, a.nodes),
                                       b["clean"], b["mask"], b["spot"])
                    losses["ref"] = l + torch.nn.functional.huber_loss(std(o["z"]), std(zt),
                                                                        delta=1.0)
                    bad_frac = bad_frac + bf
                scaled = bal(losses)
                w = {"par": 1.0, "unc": 0.3, "phy": w_phy * phy_ramp, "ref": w_ref}
                loss = sum(w[k] * v for k, v in scaled.items())
                # an unpriceable prediction is not free: it is a region the exact engine
                # cannot evaluate, and the model should be pushed out of it
                loss = loss + 5.0 * bad_frac.detach() * losses["par"]
                if not torch.isfinite(loss):
                    n_skip += 1; continue
                loss.backward()
                bad_grad = any((g.grad is not None) and (not torch.isfinite(g.grad).all())
                               for g in model.parameters())
                if bad_grad:                      # finite loss can still yield NaN gradients
                    opt.zero_grad(set_to_none=True); n_skip += 1; continue
                if o["mu_z"].grad is not None:            # the acceptance diagnostic
                    g = o["mu_z"].grad.detach().abs().mean(0)
                    for k, ix in Z_GROUPS.items():
                        gnorm[k] += float(g[list(ix)].mean())
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                for k, v in losses.items(): acc[k] = acc.get(k, 0.0) + float(v.detach())
                steps += 1
            sched.step(); ep_global += 1
            ve, vr = validate()
            gn = {k: v / max(steps, 1) for k, v in gnorm.items()}
            ratio = gn["theta"] / max(gn["v0"], 1e-30)
            rec = {"phase": name, "epoch": ep_global,
                   **{f"loss_{k}": v / max(steps, 1) for k, v in acc.items()},
                   "val_z_mae": ve, "val_reprice": vr,
                   "grad": gn, "theta_over_v0": ratio, "skipped_batches": n_skip,
                   "seconds": time.time() - t0}
            hist.append(rec)
            print(f"  ep{ep_global:>2} {name[:12]:<12} " +
                  " ".join(f"{k}={v/max(steps,1):.4f}" for k, v in acc.items()) +
                  f" | val z-MAE {ve:.4f} reprice {vr:.3e}" +
                  f" | grad theta/v0 {ratio:.3f} kappa/v0 {gn['kappa']/max(gn['v0'],1e-30):.3f}"
                  f" | skip {n_skip} | {time.time()-t0:.0f}s", flush=True)
            cfg = {"d_model": a.d_model, "rounds": a.rounds, "nodes": a.nodes}
            torch.save({"state_dict": model.state_dict(), "epoch": ep_global, "phase": name,
                        "config": cfg}, a.out / f"{a.tag}_last.pt")
            score = ve + 30.0 * (vr if math.isfinite(vr) else 1.0)
            if math.isfinite(score) and score < best_val:
                best_val = score
                torch.save({"state_dict": model.state_dict(), "epoch": ep_global,
                            "phase": name, "config": cfg, "val_z_mae": ve, "val_reprice": vr},
                           a.out / f"{a.tag}.pt")
            (a.out / f"{a.tag}_history.json").write_text(json.dumps(hist, indent=2))
    print(f"\nDONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
