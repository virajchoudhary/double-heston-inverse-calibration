"""Summarise parameter forgetting from forgetting_eval.py outputs, per run and per arm."""
import json, sys
from pathlib import Path
import numpy as np
NAMES = ["k_s","th_s","sg_s","rho_s","v0_s","k_f","th_f","sg_f","rho_f","v0_f"]

def metrics(E, steps, tau=0.10):
    A = np.abs(E)                                   # [ckpt, case, param]
    rmse = np.sqrt(np.nanmean(E**2, axis=(1, 2)))
    rec = A < tau
    prev, nxt = rec[:-1], rec[1:]
    forget = (prev & ~nxt).sum(); discover = (~prev & nxt).sum(); held = prev.sum()
    churn = np.nanmean(np.abs(np.diff(E, axis=0)))
    # how much of the best checkpoint's recovery survives to the end
    b = int(np.nanargmin(rmse))
    return dict(rmse=rmse, best_step=int(steps[b]), best=float(rmse[b]), final=float(rmse[-1]),
                regret=float(rmse[-1] - rmse[b]), churn=float(churn),
                forget=int(forget), discover=int(discover),
                forget_rate=float(forget / max(held, 1)),
                per_param_first=np.nanmean(A[0], 0), per_param_last=np.nanmean(A[-1], 0),
                fit_failures=int(np.isnan(E).any(-1).sum()))

def main(eval_dir, hist_root):
    runs = {p.stem: np.load(p) for p in sorted(Path(eval_dir).glob("*.npz"))}
    out = {}
    print(f"{'run':<24}{'best@step':>14}{'final':>8}{'regret':>8}{'churn':>7}{'forget':>7}{'discov':>7}{'f-rate':>7}  RMSE trajectory")
    for n, d in runs.items():
        m = metrics(d["E"], d["steps"]); out[n] = m
        traj = " ".join(f"{v:.3f}" for v in m["rmse"])
        print(f"{n:<24}{m['best']:>8.4f}@{m['best_step']:<5}{m['final']:>8.4f}{m['regret']:>+8.4f}{m['churn']:>7.3f}"
              f"{m['forget']:>7d}{m['discover']:>7d}{m['forget_rate']:>7.2f}  {traj}")
    arms = {}
    for n, m in out.items():
        arm = n.rsplit("_s", 1)[0]
        if not arm.startswith("prior"): arms.setdefault(arm, []).append(m)
    if arms:
        print("\nper arm, mean ± sd over seeds")
        for k in ("best", "final", "regret", "churn", "forget_rate"):
            print(f"  {k:<12}" + "  ".join(f"{a}: {np.mean([m[k] for m in ms]):.4f}±{np.std([m[k] for m in ms]):.4f} (n={len(ms)})"
                                           for a, ms in sorted(arms.items())))
        print("\nper-parameter mean |scaled error|, first -> last checkpoint, arm mean")
        print(f"  {'arm':<14}" + " ".join(f"{p:>13}" for p in NAMES))
        for a, ms in sorted(arms.items()):
            f = np.mean([m["per_param_first"] for m in ms], 0); l = np.mean([m["per_param_last"] for m in ms], 0)
            print(f"  {a:<14}" + " ".join(f"{x:.3f}->{y:.3f}" for x, y in zip(f, l)))
    return out

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
