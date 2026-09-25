"""Final residual-connection comparison: 3 arms x 3 seeds, scored on 24 fresh truths."""
import json, numpy as np
from pathlib import Path
from scipy import stats
BASE = Path(__file__).resolve().parents[2]; EV = BASE / "residual_test" / "eval"
N = ["k_s","th_s","sg_s","rho_s","v0_s","k_f","th_f","sg_f","rho_f","v0_f"]
TOL = np.array([.05, .05, .05, .10, .05] * 2)          # rho scaled by .5: .05 abs -> .10
ARMS = {"A": ("A_plain5", "plain 5x160, 107k"), "B": ("B_plain11", "plain 11x160, 262k"), "C": ("C_resid3", "5x160+3 residual, 262k")}
SEEDS = (17, 29, 43); R = {}
for arm, (pref, _) in ARMS.items():
    for s in SEEDS:
        rd = BASE / ("double_sobolev_v2_b0_s17" if (arm, s) == ("A", 17) else f"residual_test/{pref}_s{s}")
        d = np.load(EV / f"{pref}_s{s}.npz"); E, st, T = d["E"], list(d["steps"]), d["true"]
        rmse = np.sqrt(np.nanmean(E ** 2, axis=(1, 2))); b = int(np.nanargmin(rmse)); A = np.abs(E)
        rec = A < .10; held = rec[:-1].sum()
        h = json.load(open(rd / "history.json")); h = h if isinstance(h, list) else h.get("history", h)
        sel = json.load(open(rd / "selection.json"))["step"]
        S = np.abs(T).copy(); S[:, 3::5] = .5; est = T + E[-1] * S
        R[(arm, s)] = dict(rmse=rmse, best=rmse[b], best_step=st[b], final=rmse[-1], regret=rmse[-1] - rmse[b],
            churn=np.nanmean(np.abs(np.diff(E, axis=0))), forget_rate=(rec[:-1] & ~rec[1:]).sum() / max(held, 1),
            iv=h[-1]["validation_iv_rmse"], sel_step=sel, sel_rmse=rmse[st.index(sel)],
            passp=(A[-1] <= TOL).mean(0), allten=(A[-1] <= TOL).all(1).mean(),
            th_tot=np.mean(np.abs((est[:, 1] + est[:, 6]) / (T[:, 1] + T[:, 6]) - 1) <= .05),
            v0_tot=np.mean(np.abs((est[:, 4] + est[:, 9]) / (T[:, 4] + T[:, 9]) - 1) <= .05),
            fails=int(np.isnan(E).any(-1).sum()))
print(f"{'run':<9}{'recovery RMSE by checkpoint (2k..12k)':<44}{'best@step':>13}{'final':>7}{'regret':>8}{'churn':>7}{'forget':>7}{'IV err':>10}  4-case pick")
for k, r in sorted(R.items()):
    print(f"{k[0]}_s{k[1]:<6}{' '.join(f'{v:.3f}' for v in r['rmse']):<44}{r['best']:>7.3f}@{r['best_step']:<5}{r['final']:>7.3f}"
          f"{r['regret']:>+8.3f}{r['churn']:>7.3f}{r['forget_rate']:>7.2f}{r['iv']:>10.6f}  {r['sel_step']} ({r['sel_rmse']:.3f})")
print("\nPER ARM, mean ± sd over seeds 17/29/43")
for arm, (_, desc) in ARMS.items():
    g = lambda k: np.array([R[(arm, s)][k] for s in SEEDS])
    print(f"  {arm} {desc:<24} final recovery {g('final').mean():.3f}±{g('final').std():.3f}  best {g('best').mean():.3f}  "
          f"regret {g('regret').mean():+.3f}  churn {g('churn').mean():.3f}  forget rate {g('forget_rate').mean():.2f}  IV {g('iv').mean():.6f}")
print("\nPAIRED BY SEED (negative = first arm better)")
for x, y in (("C", "B"), ("C", "A"), ("B", "A")):
    for k in ("final", "churn", "forget_rate", "iv"):
        d = np.array([R[(x, s)][k] - R[(y, s)][k] for s in SEEDS])
        print(f"  {x} vs {y}  {k:<12} diff {d.mean():+.4f}  ({x} better on {(d < 0).sum()}/3 seeds)")
print("\nPASS AT PROTOCOL TOLERANCE, final checkpoint, arm mean over seeds")
print(f"  {'arm':<4}" + "".join(f"{n:>7}" for n in N) + f"{'all10':>7}{'THtot':>7}{'V0tot':>7}")
for arm in ARMS:
    p = np.mean([R[(arm, s)]["passp"] for s in SEEDS], 0)
    print(f"  {arm:<4}" + "".join(f"{v:>7.0%}" for v in p) + "".join(f"{np.mean([R[(arm, s)][k] for s in SEEDS]):>7.0%}" for k in ("allten", "th_tot", "v0_tot")))
iv = np.array([r["iv"] for r in R.values()]); fr = np.array([r["final"] for r in R.values()])
pr, pp = stats.pearsonr(iv, fr); sr, sp = stats.spearmanr(iv, fr)
print(f"\nPREDICTION TEST across all 9 runs: final IV error vs final recovery RMSE  Pearson r={pr:+.2f} (p={pp:.3f})  Spearman={sr:+.2f} (p={sp:.3f})")
bad = [(k, r["sel_rmse"] / r["best"] - 1) for k, r in R.items()]
print("4-CASE SELECTION vs 24-case best: picked a checkpoint worse by " + ", ".join(f"{k[0]}{k[1]} {v:.0%}" for k, v in sorted(bad)))
print(f"fit failures across all runs/checkpoints: {sum(r['fails'] for r in R.values())}")
json.dump({f"{k[0]}_s{k[1]}": {kk: (vv.tolist() if hasattr(vv, 'tolist') else vv) for kk, vv in r.items()} for k, r in R.items()},
          open(EV.parent / "final_results.json", "w"), indent=1, default=float)
