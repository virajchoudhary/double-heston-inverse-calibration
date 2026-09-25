"""Read-only-to-source feasibility audit; no model fitting or favorable-date selection.

Outputs summarize archived NSE coverage and a preliminary parity quality gate.
They are not a fully released model-input dataset or a claim of full cleaning.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd

WINDOWS = {
    "covid_march_2020": ["2020-03-01", "2020-03-31"],
    "iran_june_2025": ["2025-06-01", "2025-06-30"],
    "iran_opening_march_2026": ["2026-02-28", "2026-03-31"],
    "iran_april_2026_later_window": ["2026-04-01", "2026-04-30"],
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parity_gate(paired, tau):
    """Hold every third entire strike pair out of carry estimation.

    No spot is invented for legacy files. F and D are inferred from anchor
    call-put differences C-P=D(F-K). No separate r/q or spot is estimated.
    Bounds follow the prior study; error and strike span use F instead of
    missing spot. This adaptation is disclosed, not an unmodified old filter.
    """
    hold = np.arange(len(paired)) % 3 == 1
    anchor = paired.iloc[~hold]
    k = anchor.index.to_numpy(float)
    y = (anchor.CE - anchor.PE).to_numpy(float)
    X = np.column_stack([np.ones(len(k)), k])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(6):
        error = y - X @ beta
        scale = max(np.median(np.abs(error)) * 1.4826, .05)
        weight = np.minimum(1., 1.345 * scale / np.maximum(abs(error), 1e-12))
        beta = np.linalg.lstsq(X * weight[:, None], y * weight, rcond=None)[0]
    discount = -float(beta[1])
    forward = float(beta[0] / discount) if discount > 0 else float("nan")
    if not np.isfinite(forward) or forward <= 0:
        return None
    nrmse = float(np.sqrt(np.mean((y - discount * (forward - k)) ** 2)) / forward)
    if not (math.exp(-.25 * tau) <= discount <= math.exp(.1 * tau)
            and nrmse <= .025 and np.ptp(k) / forward >= .04):
        return None
    return {"forward_from_anchor_parity": forward, "discount_from_anchor_parity": discount,
            "parity_nrmse": nrmse, "pairs": len(paired), "anchor_pairs": int((~hold).sum()),
            "potential_heldout_pairs_before_further_cleaning": int(hold.sum())}


def main(workspace, out):
    out.mkdir(parents=True, exist_ok=False)
    manifest = workspace / "outputs/019fc8a0/nse_source_manifest.csv"
    entries = pd.read_csv(manifest)
    results = {"purpose": "Feasibility only; no Single/Double Heston models fitted",
               "symbol_substitution_not_authorized": True,
               "source_manifest_sha256": sha(manifest), "script_sha256": sha(Path(__file__)),
               "limitations": ["Counts after preliminary filters are not fully cleaned model inputs",
                               "No market parameter truth is known",
                               "Event windows are not a measured ranking of peak stock volatility",
                               "Carry is anchor-inferred, not an exchange-supplied field"],
               "windows": {}}
    for name, (lo, hi) in WINDOWS.items():
        selected = entries[entries.date.between(lo, hi)]
        coverage = {"dates": [lo, hi], "source_status_counts": selected.status.value_counts().to_dict(),
                    "unavailable_manifest_dates": selected.loc[~selected.status.eq("downloaded"),
                                                              ["date", "status"]].to_dict("records"),
                    "source_files": [], "symbols": {}}
        counters = {s: Counter() for s in ("ADANIPOWER", "NIFTY")}
        eligible = {s: [] for s in counters}
        dates = {s: set() for s in counters}
        for row in selected[selected.status.eq("downloaded")].itertuples():
            path = workspace / row.file
            assert sha(path) == row.sha256, f"Archived source hash mismatch: {path}"
            with zipfile.ZipFile(path) as z:
                members = [n for n in z.namelist() if n.lower().endswith('.csv')]
                assert len(members) == 1
                with z.open(members[0]) as f:
                    raw = pd.read_csv(f)
            if 'SYMBOL' in raw:
                mapping = {'SYMBOL': 'symbol', 'EXPIRY_DT': 'expiry', 'STRIKE_PR': 'strike',
                           'OPTION_TYP': 'option', 'CLOSE': 'price', 'CONTRACTS': 'contracts',
                           'OPEN_INT': 'oi', 'TIMESTAMP': 'date'}
                a = raw[raw.INSTRUMENT.isin(['OPTSTK', 'OPTIDX'])].rename(columns=mapping)
            else:
                mapping = {'TckrSymb': 'symbol', 'FininstrmActlXpryDt': 'expiry', 'StrkPric': 'strike',
                           'OptnTp': 'option', 'ClsPric': 'price', 'TtlTradgVol': 'contracts',
                           'OpnIntrst': 'oi', 'TradDt': 'date'}
                a = raw[raw.FinInstrmTp.isin(['STO', 'IDO'])].rename(columns=mapping)
            a = a[a.symbol.isin(counters)].copy()
            a['expiry'] = pd.to_datetime(a.expiry, format='mixed')
            a['date'] = pd.to_datetime(a.date, format='mixed')
            assert a.date.eq(pd.Timestamp(row.date)).all()
            coverage['source_files'].append({'file': row.file, 'url': row.url,
                                             'sha256': row.sha256, 'hash_verified': True})
            for symbol in counters:
                c = counters[symbol]
                g = a[a.symbol.eq(symbol)]
                c['listed_option_rows'] += len(g)
                c['conflicting_key_duplicates'] += int(g.duplicated(['expiry','strike','option']).sum())
                assert not g.duplicated(['expiry','strike','option']).any()
                g = g[g.contracts.gt(0) & g.price.gt(0) & g.strike.gt(0) & g.option.isin(['CE','PE'])]
                c['positive_traded_option_rows'] += len(g)
                if len(g): dates[symbol].add(row.date)
                for expiry, block in g.groupby('expiry'):
                    tau = (expiry - pd.Timestamp(row.date)).days / 365
                    if not 7 / 365 <= tau <= 100 / 365:
                        c['expiry_groups_outside_7_100_days'] += 1
                        continue
                    pairs = block.pivot(index='strike', columns='option', values='price')
                    if not {'CE', 'PE'}.issubset(pairs.columns): continue
                    pairs = pairs[['CE','PE']].dropna().sort_index()
                    if len(pairs) < 6:
                        c['expiry_groups_under_six_traded_pairs'] += 1
                        continue
                    c['expiry_groups_six_or_more_pairs'] += 1
                    gate = parity_gate(pairs, tau)
                    if gate is None:
                        c['expiry_groups_rejected_parity'] += 1
                    else:
                        eligible[symbol].append({'date': row.date, 'expiry': str(expiry.date()), **gate})
        for symbol, c in counters.items():
            coverage['symbols'][symbol] = {**dict(c), 'days_with_positive_traded_options': len(dates[symbol]),
                'preliminary_eligible_dates': sorted({r['date'] for r in eligible[symbol]}),
                'preliminary_eligible_groups': eligible[symbol]}
        results['windows'][name] = coverage
        print(name, {s: {'rows': c['listed_option_rows'], 'eligible_dates': len({r['date'] for r in eligible[s]})}
                     for s,c in counters.items()}, flush=True)
    (out / 'coverage_audit.json').write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    main(args.workspace, args.out)
