"""Turn stored variable-length surfaces into padded torch batches."""
from __future__ import annotations

import numpy as np
import torch


def collate(d: dict, idx: np.ndarray, *, use_noisy: bool = True, device="cpu") -> dict:
    """Pack rows `idx` of a dataset dict into a batch, trimmed to the longest surface."""
    n = d["n_quotes"][idx]
    m = int(n.max())
    t = lambda a: torch.tensor(np.asarray(a)[idx][:, :m], dtype=torch.float64, device=device)
    mask = (np.arange(m)[None, :] < n[:, None]).astype(float)
    price = d["noisy"] if use_noisy else d["clean"]
    b = {"spot": t(d["spot"]), "strike": t(d["strike"]), "tau": t(d["tau"]),
         "rate": t(d["rate"]), "carry": t(d["carry"]), "price": t(price),
         "clean": t(d["clean"]),
         "mask": torch.tensor(mask, dtype=torch.float64, device=device),
         "noise_level": torch.tensor(d["noise_level"][idx], dtype=torch.float64, device=device),
         "params": torch.tensor(d["params"][idx], dtype=torch.float64, device=device),
         "n_quotes": torch.tensor(n, device=device)}
    # padded slots must never be degenerate: copy the last real quote into them
    for k in ("spot", "strike", "tau", "rate", "carry", "price", "clean"):
        last = b[k].gather(1, (b["n_quotes"] - 1).clamp(min=0).unsqueeze(1)).expand(-1, m)
        b[k] = torch.where(b["mask"] > 0.5, b[k], last)
    return b
