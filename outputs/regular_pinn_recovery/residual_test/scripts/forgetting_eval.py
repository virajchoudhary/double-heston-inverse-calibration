"""Parameter-forgetting evaluation across training checkpoints.

Calibrates the frozen network at every scheduled checkpoint to the SAME N synthetic
Double Heston truths (fixed seed 913311, never used in training or checkpoint selection),
exactly as the trainer's recovery_validation does -- same grid, every-third-strike holdout,
fit_network(starts=3, seed=906777, max_nfev=200), same scaled error (rho scaled by .5).
Records per-checkpoint, per-case, per-parameter scaled errors.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path("/Users/dhruvaambhaikar/Documents/Options pricing/double-heston-v2-controlled"); sys.path.insert(0, str(ROOT))
from src.mentor_dh_pinn.regular_pinn_data import teacher_labels, decode_unit

def truths(n, seed, path):
    if path.exists():
        d = np.load(path); return d["q"], d["w"], d["true"]
    rng = np.random.default_rng(seed); units = rng.uniform(.1, .9, (n, 10))
    x = np.tile(-np.log(np.linspace(.8, 1.2, 21)), 6)
    tau = np.repeat(np.array([30, 60, 90, 180, 365, 730]) / 365, 21)
    Q, W, T = [], [], []
    for u in units:
        q = np.column_stack([x, np.log(tau), np.broadcast_to(u, (len(x), 10))])
        lab = teacher_labels(q, 2, gradients=False)
        if not lab["usable"].all(): raise ValueError("unusable truth surface")
        Q.append(q); W.append(lab["w"]); T.append(decode_unit(u, 2))
    Q, W, T = np.array(Q), np.array(W), np.array(T)
    np.savez(path, q=Q, w=W, true=T, units=units, seed=seed); return Q, W, T

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path); ap.add_argument("--truths", type=Path, required=True)
    ap.add_argument("--n", type=int, default=24); ap.add_argument("--seed", type=int, default=913311)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    Q, W, T = truths(a.n, a.seed, a.truths)
    if a.run is None:
        print(f"truths ready: {len(T)} surfaces x {Q.shape[1]} quotes -> {a.truths}"); return
    from scripts.mentor_dh_pinn.assess_regular_pinn import fit_network, load_checkpoint
    fit = np.tile(np.arange(21) % 3 != 2, 6)
    steps, E, S = [], [], []
    for ck in sorted(a.run.glob("step_*.safetensors")):
        model, _ = load_checkpoint(ck); errs, stat = [], []
        for q, w, true_p in zip(Q, W, T):
            t = np.exp(q[:, 1]); iv = np.sqrt(w / t)
            r = fit_network(model, q[:, 0], t, iv, fit_mask=fit, starts=3, seed=906777, max_nfev=200)
            if r["status"] != "fitted":
                errs.append(np.full(10, np.nan)); stat.append(0); continue
            scale = np.abs(true_p).copy(); scale[3::5] = .5
            errs.append((np.asarray(r["physical"]) - true_p) / scale); stat.append(1)
        steps.append(int(ck.stem.split("_")[1])); E.append(errs); S.append(stat)
        print(json.dumps({"run": a.run.name, "step": steps[-1],
                          "rmse": float(np.sqrt(np.nanmean(np.square(errs)))), "fitted": int(sum(stat))}), flush=True)
    np.savez(a.out, steps=np.array(steps), E=np.array(E), status=np.array(S), true=T)

if __name__ == "__main__":
    main()
