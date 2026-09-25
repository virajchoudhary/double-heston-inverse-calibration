"""Apply the frozen rule in RULE.md, then score every model on every surface."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent; V5 = HERE.parent
sys.path.insert(0, str(V5)); sys.path.insert(0, str(V5.parent / 'btc_multifactor_v1'))
from track_b import prior_c, fit_scale
from evaluate import load_stage, locked
from engine import iv as inv_iv

NETS = {'PINN_OLD': locked(), 'PINN_NEW': load_stage('FINAL', 'final')}
MODELS = [('BS', 'BS', None), ('SH', 'SH', None), ('DH', 'DH', None), ('PINN_OLD', 'DH_PINN', 'PINN_OLD'), ('PINN_NEW', 'DH_PINN', 'PINN_NEW')]
BUCK = [(7, 30, '<=30d'), (30, 90, '30-90d'), (90, 365, '90-365d'), (365, 9999, '>365d')]
scan = json.loads((HERE / 'scan.json').read_text())
out = {}
for sym, info in scan.items():
    q = pd.read_csv(HERE / 'surfaces' / f'{sym}.csv')
    x, t = q.x.to_numpy(), q.tau.to_numpy()
    row = {'two_timescale_score': info['two_timescale_score'], 'expiries': info['expiries'], 'quotes': info['quotes'],
           'atm_iv_short': info['atm_iv_short'], 'atm_iv_long': info['atm_iv_long'], 'models': {}}
    for name, kind, net in MODELS:
        s = fit_scale(kind, q, NETS.get(net))
        c = prior_c(kind, x, t, s, NETS.get(net)); mi = inv_iv(c, x, t); ok = np.isfinite(mi) & np.isfinite(q.market_iv)
        e = 100 * (mi[ok] - q.market_iv.to_numpy()[ok]); days = q.days.to_numpy()[ok]
        row['models'][name] = {'scale': s, 'scale_at_bound': bool(s <= 0.7005 or s >= 3.1995), 'IV_RMSE': float(np.sqrt(np.mean(e ** 2))),
                               'buckets': {lab: float(np.sqrt(np.mean(e[(days >= a) & (days < b)] ** 2)))
                                           for a, b, lab in BUCK if ((days >= a) & (days < b)).any()}}
    row['in_pinn_domain'] = not row['models']['DH']['scale_at_bound']
    row['SH_minus_DH'] = row['models']['SH']['IV_RMSE'] - row['models']['DH']['IV_RMSE']
    out[sym] = row
    print(f"{sym:6} score {row['two_timescale_score']:.4f} domain {str(row['in_pinn_domain']):5} scale {row['models']['DH']['scale']:.2f} | "
          f"BS {row['models']['BS']['IV_RMSE']:6.2f} SH {row['models']['SH']['IV_RMSE']:6.2f} DH {row['models']['DH']['IV_RMSE']:6.2f} "
          f"PINNnew {row['models']['PINN_NEW']['IV_RMSE']:6.2f} | SH-DH {row['SH_minus_DH']:+.2f}", flush=True)
elig = {k: v for k, v in out.items() if v['in_pinn_domain'] and v['expiries'] >= 6}
pick = max(elig, key=lambda k: elig[k]['two_timescale_score']) if elig else None
res = {'rule': (HERE / 'RULE.md').read_text(), 'surfaces': out, 'eligible': sorted(elig, key=lambda k: -elig[k]['two_timescale_score']), 'selected': pick}
(HERE / 'results.json').write_text(json.dumps(res, indent=2, default=float) + '\n')
print('\nEligible by score:', [(k, round(elig[k]['two_timescale_score'], 4)) for k in res['eligible']])
print('SELECTED:', pick)
