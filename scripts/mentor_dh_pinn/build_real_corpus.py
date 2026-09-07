#!/usr/bin/env python3
"""Build a portable, calendar-embargoed DEVELOPMENT corpus from authentic local data.

Previously inspected market dates cannot become a fresh test by being relabelled.
This supports calibration development only; it is not a forward experiment.
Validation reserves strikes before carry fitting and estimates weights/noise from
calibration quotes. The original committed real_validation.npz is never replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.mentor_dh_pinn.nifty_panel import BHAV, bhav_path, surface as nifty_surface

PANEL = ROOT / "single_heston_pinn" / "outputs" / "pinn_quote_panel.parquet"
SELECTION = ROOT / "outputs" / "real_markets" / "nifty_selection.json"
MAX_QUOTES, MIN_QUOTES = 100, 6
# Preserve the previous study's entire reserved middle-of-period block, including
# its unscored dates. A calendar-day superset also covers exchange holidays.
OLD_NIFTY_RESERVED = set(pd.date_range("2026-03-25", "2026-04-27").strftime("%Y-%m-%d"))


def _vega(fwd, strike, tau, iv):
    root = np.maximum(iv, 1e-4) * np.sqrt(tau)
    d1 = np.log(fwd / strike) / root + 0.5 * root
    return fwd * np.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi) * np.sqrt(tau)


def _stratified_indices(strike, tau, holdout, maximum=MAX_QUOTES):
    """Round-robin expiry/fold allocation, evenly spaced strikes within each cell."""
    cells = [np.flatnonzero((tau == t) & (holdout == h))
             for t in np.unique(tau) for h in (False, True)]
    cells = [c[np.argsort(strike[c], kind="stable")] for c in cells if len(c)]
    if len(cells) > maximum:
        raise ValueError("Quote cap cannot preserve all expiry/fold cells")
    budgets = np.zeros(len(cells), int)
    for _ in range(min(maximum, len(tau))):
        available = [i for i, c in enumerate(cells) if budgets[i] < len(c)]
        i = min(available, key=lambda j: (budgets[j], j))
        budgets[i] += 1
    return np.sort(np.concatenate([c[np.linspace(0, len(c)-1, n, dtype=int)]
                                   for c, n in zip(cells, budgets)]))


def _pack(fwd, strike, tau, price, iv, label, holdout_mask=None):
    arrays = [np.asarray(x, float) for x in (fwd, strike, tau, price, iv)]
    fwd, strike, tau, price, iv = arrays
    held = (np.zeros(len(tau), bool) if holdout_mask is None
            else np.asarray(holdout_mask, bool))
    if any(a.shape != tau.shape for a in arrays) or held.shape != tau.shape:
        raise ValueError(f"Misaligned surface arrays: {label}")
    if pd.DataFrame({"tau": tau, "strike": strike}).duplicated().any():
        raise ValueError(f"Duplicate contracts in {label}")
    source_count = len(tau)
    ok = (np.logical_and.reduce([np.isfinite(a) for a in arrays]) & (fwd > 0)
          & (strike > 0) & (tau > 0) & (iv > 0)
          & (price > np.maximum(fwd-strike, 0)) & (price < fwd))
    fwd, strike, tau, price, iv, held = [a[ok] for a in (*arrays, held)]
    if len(tau) < MIN_QUOTES or (~held).sum() < MIN_QUOTES:
        return None
    clean_count = len(tau)
    keep = _stratified_indices(strike, tau, held)
    fwd, strike, tau, price, iv, held = [a[keep] for a in (fwd, strike, tau, price, iv, held)]
    fit = ~held
    if fit.sum() < MIN_QUOTES or (holdout_mask is not None and held.sum() < 2):
        return None
    scale = float(np.median(fwd[fit]))
    x = np.log(fwd / strike)
    weight_iv = np.empty(len(tau))
    residuals = []
    for t in np.unique(tau):
        expiry = tau == t
        anchors = expiry & fit
        if not anchors.any():
            return None
        degree = min(2, int(anchors.sum())-1)
        coef = np.polyfit(x[anchors], iv[anchors], degree)
        weight_iv[expiry] = np.clip(np.polyval(coef, x[expiry]), .01, 4.)
        residuals.extend(iv[anchors] - np.polyval(coef, x[anchors]))
    iv_noise = float(np.clip(1.4826*np.median(np.abs(residuals)), 2e-3, .15))
    # Heldout target IV must never determine the calibration objective's weight.
    veg = _vega(fwd, strike, tau, weight_iv if held.any() else iv) / scale
    return {"label": label, "n": len(tau), "spot": fwd / scale,
            "strike": strike / scale, "tau": tau, "price": price / scale,
            "iv": iv, "vega": np.maximum(veg, 1e-8),
            "quote_sigma": np.maximum(iv_noise*veg, 1e-8),
            "rate": np.zeros(len(tau)), "carry": np.zeros(len(tau)),
            "holdout_mask": held, "iv_noise": iv_noise,
            "quality_rows_excluded": source_count-clean_count,
            "cap_rows_excluded": clean_count-len(tau),
            "n_expiries": int(len(np.unique(tau)))}


def stock_surfaces(panel, split, excluded_dates, failures=None):
    out = []
    p = panel[(panel.split == split) & ~panel.trade_date.dt.strftime("%Y-%m-%d").isin(excluded_dates)]
    for (sym, date), g in p.groupby(["symbol", "trade_date"]):
        fp = g.market_price_adjusted/g.discount_factor + np.where(~g.is_call, g.forward-g.strike, 0.)
        held = g.fold.eq("holdout").to_numpy() if split == "validation" else None
        rec = _pack(g.forward.to_numpy(), g.strike.to_numpy(), g.maturity.to_numpy(),
                    fp.to_numpy(), g.market_iv.to_numpy(), f"{sym}|{date.date()}", held)
        if rec:
            out.append(rec)
        elif failures is not None:
            failures.append({"label": f"{sym}|{date.date()}",
                             "reason": "insufficient valid calibration or holdout quotes"})
    return out


def nifty_surfaces(dates, bhav_dir, *, validation=False):
    out, failures = [], []
    for date in dates:
        try:
            s = nifty_surface(date, bhav_dir=bhav_dir, holdout_fold=2 if validation else None)
            if s.empty:
                failures.append({"date": date, "reason": "no liquid valid NIFTY surface"})
                continue
            if (s.open_interest < 10000).any():
                raise ValueError("per-quote open-interest constraint failed")
            rec = _pack(s.forward.to_numpy(), s.strike.to_numpy(), s.tau.to_numpy(),
                        s.fwd_call.to_numpy(), s.iv.to_numpy(), f"NIFTY|{date}",
                        s.holdout_mask.to_numpy() if validation else None)
            if rec is None or rec["n_expiries"] < 3:
                failures.append({"date": date, "reason": "insufficient quotes or fewer than three expiries"})
            else:
                out.append(rec)
        except (OSError, ValueError, KeyError) as exc:
            failures.append({"date": date, "reason": f"{type(exc).__name__}: {exc}"})
    return out, failures


def save(recs, path):
    if not recs:
        raise ValueError(f"Empty corpus: {path}")
    m = max(r["n"] for r in recs)
    pad = lambda k: np.stack([np.pad(r[k], (0, m-r["n"]), constant_values=0)
                              for r in recs]).astype(np.float64)
    data = {k: pad(k) for k in ("spot", "strike", "tau", "rate", "carry", "price",
                                "iv", "vega", "quote_sigma", "holdout_mask")}
    # Padding has valid geometry for batched pricing and never enters a loss.
    for k in ("spot", "strike", "tau", "vega", "quote_sigma"):
        for i, r in enumerate(recs):
            data[k][i, r["n"]:] = r[k][-1]
    data["mask"] = np.stack([np.arange(m) < r["n"] for r in recs]).astype(float)
    for key, source in (("n_quotes", "n"), ("iv_noise", "iv_noise"),
                        ("n_expiries", "n_expiries"), ("label", "label")):
        data[key] = np.array([r[source] for r in recs])
    np.savez_compressed(path, **data)
    print(f"{path.name}: {len(recs)} surfaces / {sum(r['n'] for r in recs)} quotes", flush=True)


def source_hash(path):
    with Path(path).open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    return {"path": str(Path(path).resolve()), "sha256": digest, "bytes": Path(path).stat().st_size}


def corpus_summary(recs):
    return {"surfaces": len(recs), "quotes": sum(r["n"] for r in recs),
            "symbols": dict(Counter(r["label"].split("|")[0] for r in recs)),
            "expiry_histogram": dict(Counter(r["n_expiries"] for r in recs)),
            "dates": sorted({r["label"].split("|")[1] for r in recs}),
            "quality_rows_excluded": sum(r["quality_rows_excluded"] for r in recs),
            "cap_rows_excluded": sum(r["cap_rows_excluded"] for r in recs),
            "heldout_quotes": int(sum(r["holdout_mask"].sum() for r in recs))}


def evenly(dates, count):
    return [dates[i] for i in np.linspace(0, len(dates)-1, min(len(dates), count), dtype=int)] if dates and count else []


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT/"outputs"/"calibration_repair"/"real_corpus")
    ap.add_argument("--panel", type=Path, default=PANEL)
    ap.add_argument("--selection", type=Path, default=SELECTION)
    ap.add_argument("--bhav-dir", type=Path, default=BHAV,
                    help="Year subdirectories of authentic NSE bhavcopy ZIPs")
    ap.add_argument("--nifty-train", type=int, default=60)
    ap.add_argument("--nifty-validation", type=int, default=10)
    a = ap.parse_args()
    if a.out.resolve() == (ROOT/"outputs"/"real_corpus").resolve():
        raise ValueError("Preserve the original corpus; choose a new output directory")
    if not a.bhav_dir.is_dir():
        ap.error("Supply --bhav-dir pointing to your original NSE bhavcopies")
    p = pd.read_parquet(a.panel)
    p["trade_date"] = pd.to_datetime(p.trade_date)
    if p.duplicated(["symbol", "trade_date", "expiry_date", "strike", "option_type"]).any():
        raise ValueError("Duplicate source stock-panel contracts")
    held_stock = p[p.fold.eq("holdout")]
    if any(held_stock[c].fillna("").astype(str).ne("").any()
           for c in ("anchor_ce_row_key", "anchor_pe_row_key")):
        raise ValueError("Stock holdout quotes carry parity-anchor provenance")
    selected = set(json.loads(a.selection.read_text())["dates"])
    split_dates = {s: set(p[p.split == s].trade_date.dt.strftime("%Y-%m-%d"))
                   for s in ("train", "validation", "test")}
    test_dates = split_dates["test"] | selected | OLD_NIFTY_RESERVED
    validation_dates = split_dates["validation"] - test_dates
    raw_dates = {pd.Timestamp(f.name.split("_")[-3]).strftime("%Y-%m-%d")
                 for f in a.bhav_dir.glob("20*/BhavCopy_NSE_FO_0_0_0_*_F_0000.csv.zip")}
    old_nifty_pool = sorted(d for d in raw_dates if "2026-02-03" <= d <= "2026-06-03" and d not in test_dates)
    nv_dates = evenly(old_nifty_pool, a.nifty_validation)
    validation_dates |= set(nv_dates)
    nt_pool = sorted((raw_dates & split_dates["train"]) - test_dates - validation_dates)
    nt_dates = evenly(nt_pool, a.nifty_train)
    print(f"Global embargo: {len(test_dates)} reserved/test dates; {len(validation_dates)} validation dates", flush=True)
    print(f"NIFTY candidates: train {len(nt_dates)}/{len(nt_pool)}, validation {len(nv_dates)}", flush=True)
    ntr, ft = nifty_surfaces(nt_dates, a.bhav_dir)
    nva, fv = nifty_surfaces(nv_dates, a.bhav_dir, validation=True)
    stock_failures = []
    train = stock_surfaces(p, "train", test_dates | validation_dates, stock_failures) + ntr
    val = stock_surfaces(p, "validation", test_dates, stock_failures) + nva
    tr_dates = {r["label"].split("|")[1] for r in train}
    va_dates = {r["label"].split("|")[1] for r in val}
    assert not (tr_dates & va_dates or tr_dates & test_dates or va_dates & test_dates)
    notes = ["Development evidence only. Previously inspected market evaluation is not fresh.",
             "New validation NIFTY dates were within the previous fine-tune training period; old checkpoint exposure persists.",
             "Global calendar embargo applies to this new corpus, not historical checkpoints.",
             "Stock carry reuses shipped calibration-strike parity fits; heldout anchors are empty.",
             "No forward-market experiment performed; POWERGRID mentor decision remains pending."]
    if len(ntr) < 20 or len(nva) < 5:
        notes.append("Limited multi-expiry evidence after embargo/filtering; results are exploratory.")
        warnings.warn(notes[-1])
    if not ntr or not nva:
        raise ValueError(f"Multi-expiry corpus unavailable: train={len(ntr)}, val={len(nva)}, failures={ft+fv}")
    a.out.mkdir(parents=True, exist_ok=True)
    save(train, a.out/"real_train.npz")
    save(val, a.out/"real_validation.npz")
    sources = [source_hash(a.panel), source_hash(a.selection)]
    sources += [source_hash(bhav_path(d, a.bhav_dir)) for d in sorted(set(nt_dates+nv_dates))]
    manifest = {"purpose": "calibration_development", "notes": notes,
                "train": corpus_summary(train), "validation": corpus_summary(val),
                "reserved_test_dates": sorted(test_dates),
                "calendar_overlap_train_validation": sorted(tr_dates & va_dates),
                "calendar_overlap_train_test": sorted(tr_dates & test_dates),
                "calendar_overlap_validation_test": sorted(va_dates & test_dates),
                "nifty_train_requested_dates": nt_dates, "nifty_validation_requested_dates": nv_dates,
                "source_files": sources, "nifty_rejected_dates": ft+fv,
                "stock_rejected_surfaces": stock_failures,
                "stock_source_quotes_by_split": {str(k): int(v) for k, v in p.groupby("split").size().items()},
                "stock_calendar_embargoed_surfaces": {
                    split: int(p[(p.split == split) & p.trade_date.dt.strftime("%Y-%m-%d").isin(exclude)]
                               .groupby(["symbol", "trade_date"]).ngroups)
                    for split, exclude in (("train", test_dates | validation_dates), ("validation", test_dates))},
                "implementation_files": [source_hash(path) for path in (
                    Path(__file__), ROOT/"src/mentor_dh_pinn/nifty_panel.py",
                    ROOT/"single_heston_pinn/src/single_heston.py")],
                "quote_cap": MAX_QUOTES, "quote_cap_method": "expiry/fold stratified; evenly spaced strikes",
                "nifty_min_open_interest_per_quote": 10000,
                "holdout_mask_contract": "fit = mask & ~holdout_mask; score = mask & holdout_mask",
                "derived_weight_scope": "validation weights/noise fitted only to calibration-strike IV",
                "outputs": [source_hash(a.out/name) for name in ("real_train.npz", "real_validation.npz")]}
    (a.out/"manifest.json").write_text(json.dumps(manifest, indent=2))
    for failure in ft+fv:
        print(f"Excluded NIFTY {failure['date']}: {failure['reason']}", flush=True)
    print(f"Manifest: {a.out/'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
