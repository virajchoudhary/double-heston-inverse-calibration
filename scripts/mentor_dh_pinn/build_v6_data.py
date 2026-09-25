#!/usr/bin/env python3
"""Generate the variable-geometry corpus. Deterministic seeds, geometry-disjoint splits."""
import sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
torch.set_default_dtype(torch.float64)
from src.mentor_dh_pinn.dataset_v6 import build

out = ROOT / "outputs" / "unified_v6"; out.mkdir(parents=True, exist_ok=True)
for name, n, seed in (("train", 150000, 101), ("validation", 25000, 202), ("test", 25000, 303)):
    f = out / f"v6_{name}.npz"
    if f.exists():
        print(f"{name}: exists, skipping", flush=True); continue
    t0 = time.time(); d = build(n, seed)
    np.savez_compressed(f, **d)
    print(f"{name}: {n} surfaces, valid {d['ok'].mean():.4f}, {time.time()-t0:.0f}s -> {f.name}",
          flush=True)
print("DATA DONE", flush=True)
