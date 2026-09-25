"""Market tables for the frozen selected models. REPORTING/AUDIT ONLY.

--stage validation  The 800 held-out validation quotes on 14 dates. This is the SELECTION SAMPLE:
                    the SH and DH 14% candidates were chosen and the BS_TERM volatilities fitted on
                    these same quotes. It is NOT out-of-sample evidence (Table D).
--stage final       Reads market_final_predictions.csv written by the frozen `run.py market-final`
                    (Tables E, F, G). Refuses to run unless the final-test freeze manifest verifies.
One metric function serves both stages. Changes no model, filter, parameter or scoring rule.
"""
import argparse
import numpy as np
import pandas as pd
from run import OUT, verify, read, save, sha, stamp, nets, neural, metrics
from literature_exact import exact, bs_predict

FIXED = ['single_fixed_exact', 'double_fixed_exact', 'double_fixed_PINN', 'BS_fixed']
STATE = ['single_state_exact', 'double_state_exact', 'double_state_PINN', 'BS_state_flat', 'BS_state_term']


def masks(a):
    tau = a.tau.to_numpy(); kf = np.exp(-a.x.to_numpy())
    m = {'all': np.ones(len(a), bool), 'maturity_<=30d': tau <= 30 / 365, 'maturity_30-90d': (tau > 30 / 365) & (tau <= 90 / 365),
         'maturity_90-365d': (tau > 90 / 365) & (tau <= 1), 'maturity_>365d': tau > 1,
         'ATM_|K/F-1|<=.02': abs(kf - 1) <= .02, 'wing_K/F<.9_or_>1.1': (kf < .9) | (kf > 1.1)}
    for lo, hi in zip([.7, .9, .98, 1.02, 1.1], [.9, .98, 1.02, 1.1, 1.30000001]): m[f'K/F_{lo:.2f}-{hi:.2f}'] = (kf >= lo) & (kf < hi)
    return m


def score(a, names):
    rows = []; scale = (a.discount * a.forward).to_numpy(); y = a.target_forward_call.to_numpy()
    for label, mask in masks(a).items():
        for name in names:
            if name not in a: continue
            pred = a[name].to_numpy(); ok = mask & np.isfinite(pred)
            row = {'bucket': label, 'model': name, 'eligible_quotes': int(mask.sum()), 'scored_quotes': int(ok.sum())}
            if ok.any():
                e = pred[ok] - y[ok]; ep = e * scale[ok]; m = metrics(y[ok], pred[ok], a.x.to_numpy()[ok], a.tau.to_numpy()[ok])
                row.update({'price_RMSE_index_points': float(np.sqrt(np.mean(ep ** 2))), 'price_MAE_index_points': float(np.mean(abs(ep))),
                            'max_abs_error_index_points': float(abs(ep).max()), 'forward_normalized_RMSE': float(np.sqrt(np.mean(e ** 2))),
                            'equal_date_forward_RMSE': float(np.sqrt(pd.Series(e ** 2).groupby(a.date.to_numpy()[ok]).mean().mean())),
                            'IV_RMSE_volatility_points': m['IV_RMSE_volatility_points'], 'IV_valid_quotes': m['IV_valid_quotes']})
            rows.append(row)
    return pd.DataFrame(rows)


def decomposition(a, modes):
    rows = []; scale = (a.discount * a.forward).to_numpy(); y = a.target_forward_call.to_numpy()
    for mode in modes:
        ex, pn = a[f'double_{mode}_exact'].to_numpy(), a[f'double_{mode}_PINN'].to_numpy(); ok = np.isfinite(pn)
        for label, v in [('PINN_approximation = PINN - exact DH', (pn - ex) * scale), ('model_parameter = exact DH - market', (ex - y) * scale),
                         ('total = PINN - market', (pn - y) * scale)]:
            rows.append({'protocol': mode, 'component': label, 'quotes': int(ok.sum()), 'omitted_outside_PINN_domain': int((~ok).sum()),
                         'RMSE_index_points': float(np.sqrt(np.mean(v[ok] ** 2))), 'MAE_index_points': float(np.mean(abs(v[ok]))),
                         'max_abs_index_points': float(abs(v[ok]).max())})
        sx = a[f'single_{mode}_exact'].to_numpy()
        rows.append({'protocol': mode, 'component': 'Single Heston model_parameter = exact SH - market', 'quotes': len(a), 'omitted_outside_PINN_domain': 0,
                     'RMSE_index_points': float(np.sqrt(np.mean(((sx - y) * scale) ** 2))), 'MAE_index_points': float(np.mean(abs((sx - y) * scale))),
                     'max_abs_index_points': float(abs((sx - y) * scale).max())})
    return pd.DataFrame(rows)


def daily(a, names):
    scale = (a.discount * a.forward).to_numpy(); y = a.target_forward_call.to_numpy(); rows = []
    for name in names:
        if name not in a: continue
        err = (a[name].to_numpy() - y) * scale
        for date, idx in a.groupby('date').indices.items():
            z = err[idx]; ok = np.isfinite(z)
            rows.append({'date': date, 'model': name, 'quotes': len(idx), 'scored': int(ok.sum()), 'RMSE_index_points': float(np.sqrt(np.mean(z[ok] ** 2))) if ok.any() else np.nan})
    return pd.DataFrame(rows)


def validation_frame(c):
    sel_file = read(OUT / 'market_selection.json'); assert sel_file['validation_sha256'] == sha(OUT / 'market_validation/clean_quotes.csv')
    q = pd.read_csv(OUT / 'market_validation/clean_quotes.csv'); a = q[q.date_eligible & q.split.eq('test')].copy()
    assert len(a) == 800 and a.date.nunique() == 14
    sel = sel_file['selected']; x, t = a.x.to_numpy(), a.tau.to_numpy(); p = np.array(sel['double']['params'])
    lo_s, hi_s = c['pinn']['slow_state_domain']; lo_f, hi_f = c['pinn']['fast_state_domain']
    inside = (abs(x) <= .36 + 1e-12) & (t >= 7 / 365 - 1e-12) & (t <= 2. + 1e-12) & (lo_s <= p[4] <= hi_s) & (lo_f <= p[9] <= hi_f)
    a['target_forward_call'] = a.market_call_normalized.to_numpy() * np.exp(-x)
    a['single_fixed_exact'] = exact(sel['single']['params'], x, t); a['double_fixed_exact'] = exact(p, x, t)
    a['double_fixed_PINN'] = np.where(inside, neural(nets(c), p, x, t)[0], np.nan); a['BS_fixed'] = bs_predict(sel['bs'], x, t)
    return a


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', choices=['validation', 'final'], required=True); stage = ap.parse_args().stage
    c = verify(); dest = OUT / 'market_tables'; dest.mkdir(exist_ok=True)
    assert not (dest / f'{stage}_fixed_metrics.csv').exists(), 'stage already reported; outputs are not overwritten'
    if stage == 'validation':
        a = validation_frame(c); names, modes = FIXED, ['fixed']
        role = 'SELECTION_SAMPLE_NOT_OUT_OF_SAMPLE: SH/DH candidates chosen and BS_TERM fitted on these quotes'
    else:
        from final_freeze import verify_freeze
        verify_freeze(); a = pd.read_csv(OUT / 'market_final_predictions.csv'); names, modes = FIXED + STATE, ['fixed', 'state']
        role = 'UNTOUCHED_FINAL_TEST: models frozen before any final file was downloaded'
    a.to_csv(dest / f'{stage}_predictions.csv', index=False)
    score(a, FIXED).to_csv(dest / f'{stage}_fixed_metrics.csv', index=False)
    if stage == 'final': score(a, STATE).to_csv(dest / f'{stage}_STATE_ADAPTIVE_DIAGNOSTIC_NOT_FIXED_PARAMETER_PRIMARY_metrics.csv', index=False)
    decomposition(a, modes).to_csv(dest / f'{stage}_error_decomposition.csv', index=False)
    daily(a, names).to_csv(dest / f'{stage}_daily_errors.csv', index=False)
    save(dest / f'{stage}_manifest.json', {'utc': stamp(), 'stage': stage, 'sample_role': role, 'source_sha256': sha(__file__),
         'protocol_sha256': sha(OUT / 'manifest.json'), 'dates': sorted(a.date.unique().tolist()), 'quotes': len(a),
         'files': {p.name: sha(p) for p in sorted(dest.glob(f'{stage}_*')) if p.suffix == '.csv'}})
    print(stage, role, len(a), 'quotes', a.date.nunique(), 'dates', flush=True)


if __name__ == '__main__':
    main()
