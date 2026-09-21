"""Teacher-data quality diagnostics. REPORTING/AUDIT ONLY.

Reads the frozen teacher splits and controlled surfaces. Recomputes the primary 128/96-node
Torch prices to locate every fallback, spot-checks accepted and fallback points against the
independent adaptive reference, and checks bounds, coverage, split overlap, static arbitrage
and scenario representation. It changes no teacher value, model, threshold or selection and
records its own SHA-256.

Decision rule, fixed before this script was first run:
  TEACHER_UNSTABLE if any spot-check |teacher - adaptive| > 2e-6 (10% of the frozen PINN
  forward-price RMSE gate 2e-5), any non-finite price, any price outside [intrinsic-1e-9, 1+1e-9],
  or any unreliable adaptive reference. Exceedances of the protocol tolerance 1e-8 are reported
  separately and do not by themselves block training.
"""
import json, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def _adaptive_job(job):
    import sys; sys.path.insert(0, str(HERE))
    from literature_exact import adaptive
    idx, p, x, t, eps, label = job
    t0 = time.perf_counter()
    try:
        v, _ = adaptive(np.asarray(p, float), float(x), float(t), eps)
        return idx, label, float(v), None, time.perf_counter() - t0
    except Exception as e:
        return idx, label, float('nan'), repr(e)[:300], time.perf_counter() - t0


def main():
    import pandas as pd
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from run import OUT, verify, read, save, sha, stamp
    from src.mentor_dh_pinn.torch_pricer import price_call
    torch.set_num_threads(1)
    c = verify(); P = np.array(c['published_double_slow_first'])
    dest = OUT / 'teacher_quality'; dest.mkdir(exist_ok=False)
    STRICT, MATERIAL, BOUND = 1e-8, 2e-6, 1e-9
    TB = [(7, 30), (30, 90), (90, 365), (365, 731)]
    KB = [(0, .9), (.9, .98), (.98, 1.02), (1.02, 1.1), (1.1, 9)]
    rng = np.random.default_rng(20260917)
    report = {'utc': stamp(), 'source_sha256': sha(__file__), 'protocol_sha256': sha(OUT / 'manifest.json'),
              'decision_rule': 'UNSTABLE if spot-check diff > 2e-6, non-finite price, bound breach > 1e-9 or unreliable reference',
              'teacher_file_sha256': {f: sha(OUT / 'teacher' / f) for f in ['train.npz', 'development.npz', 'collocation.npz']}}
    splits = {k: np.load(OUT / f'teacher/{k}.npz') for k in ['train', 'development', 'collocation']}

    # ---------------- structure, ranges, overlap
    fp = {k: set(map(tuple, np.column_stack([z['coords'], z['params']]))) for k, z in splits.items()}
    report['split_rows'] = {k: len(z['coords']) for k, z in splits.items()}
    report['duplicates_within_split'] = {k: len(splits[k]['coords']) - len(fp[k]) for k in fp}
    report['exact_overlap'] = {'train_development': len(fp['train'] & fp['development']),
                               'train_collocation': len(fp['train'] & fp['collocation']),
                               'development_collocation': len(fp['development'] & fp['collocation'])}
    for k, z in splits.items():
        p = z['params']; s = p[:, 1] / P[1]
        assert np.allclose(p[:, [0, 5]], P[[0, 5]], rtol=0, atol=0) and np.allclose(p[:, [3, 8]], P[[3, 8]], rtol=0, atol=0)
        assert np.allclose(p[:, 6] / P[6], s, rtol=1e-12) and np.allclose(p[:, [2, 7]] / P[[2, 7]], np.sqrt(s)[:, None], rtol=1e-12)
        report.setdefault('ranges', {})[k] = {
            'x': [float(z['coords'][:, 0].min()), float(z['coords'][:, 0].max())],
            'strike_over_forward': [float(np.exp(-z['coords'][:, 0]).min()), float(np.exp(-z['coords'][:, 0]).max())],
            'tau_days': [float(365 * z['coords'][:, 3].min()), float(365 * z['coords'][:, 3].max())],
            'v_slow': [float(p[:, 4].min()), float(p[:, 4].max())], 'v_fast': [float(p[:, 9].min()), float(p[:, 9].max())],
            'scale': [float(s.min()), float(s.max())],
            'kappa_rho_fixed_to_published': True, 'theta_sigma_scaled_consistently': True,
            'fraction_fast_heavy_v_fast_gt_v_slow': float(np.mean(p[:, 9] > p[:, 4])),
            'fraction_short_maturity_le_30d': float(np.mean(365 * z['coords'][:, 3] <= 30)),
            'fraction_wing_KF_lt_0.9_or_gt_1.1': float(np.mean((np.exp(-z['coords'][:, 0]) < .9) | (np.exp(-z['coords'][:, 0]) > 1.1)))}
        if 'price' in z:
            report['ranges'][k]['price'] = [float(z['price'].min()), float(z['price'].max())]

    # ---------------- recompute primary resolutions, locate fallbacks, bounds
    spot_jobs, cell_rows = [], []
    for k in ['train', 'development']:
        z = splits[k]; coords, params, stored = z['coords'], z['params'], z['price']
        a, b = [], []
        with torch.no_grad():
            for s0 in range(0, len(coords), 256):
                zz = torch.tensor(coords[s0:s0 + 256], dtype=torch.float64); pp = torch.tensor(params[s0:s0 + 256], dtype=torch.float64)
                x, t = zz[:, 0:1], zz[:, -1:]; one = torch.ones_like(x)
                a.append(price_call(pp, one, torch.exp(-x), t, one * 0, one * 0, node_count=128)[:, 0].numpy())
                b.append(price_call(pp, one, torch.exp(-x), t, one * 0, one * 0, node_count=96)[:, 0].numpy())
        a, b = np.concatenate(a), np.concatenate(b); diff = abs(a - b)
        x = coords[:, 0]; tau = coords[:, 3]; kf = np.exp(-x); lo = np.maximum(1 - kf, 0)
        bad = (diff > 1e-8) | (a < lo - 1e-9) | (a > 1 + 1e-9)
        audit = read(OUT / f'teacher/{k}_audit.json')
        tv = stored - lo; invalid = ~z['iv_valid']
        report.setdefault('numerics', {})[k] = {
            'points': len(stored), 'rejected_points': 0,
            'integration_disagreements_128_vs_96_gt_1e-8': int((diff > 1e-8).sum()),
            'bound_triggered_fallbacks': int(((a < lo - 1e-9) | (a > 1 + 1e-9)).sum()),
            'fallback_count_recomputed': int(bad.sum()), 'fallback_count_audit': audit['fallback_count'],
            'fallback_count_matches_audit': int(bad.sum()) == audit['fallback_count'],
            'max_128_96_difference_recomputed': float(diff.max()), 'max_128_96_difference_audit': audit['pre_fallback_max_96_128_difference'],
            'diff_quantiles_50_90_99_999': [float(q) for q in np.quantile(diff, [.5, .9, .99, .999])],
            'stored_equals_128_node_on_accepted_points_bitwise': bool(np.array_equal(stored[~bad], a[~bad])),
            'max_abs_stored_minus_128_on_accepted': float(abs(stored[~bad] - a[~bad]).max()),
            'fallback_changed_price_median_abs': float(np.median(abs(stored[bad] - a[bad]))) if bad.any() else 0.,
            'fallback_changed_price_max_abs': float(abs(stored[bad] - a[bad]).max()) if bad.any() else 0.,
            'nonfinite_prices': int((~np.isfinite(stored)).sum()),
            'below_intrinsic_any': int((stored < lo).sum()), 'below_intrinsic_beyond_1e-9': int((stored < lo - BOUND).sum()),
            'worst_below_intrinsic': float(min(0., (stored - lo).min())),
            'above_forward_bound_beyond_1e-9': int((stored > 1 + BOUND).sum()), 'max_price': float(stored.max()),
            'invalid_IV': int(invalid.sum()), 'invalid_IV_price_at_or_below_intrinsic': int((invalid & (tv <= 0)).sum()),
            'invalid_IV_time_value_below_1e-12': int((invalid & (tv > 0) & (tv < 1e-12)).sum()),
            'invalid_IV_unexplained': int((invalid & (tv >= 1e-12)).sum())}
        days = 365 * tau
        for (t0, t1) in TB:
            for (k0, k1) in KB:
                m = (days >= t0) & (days < t1) & (kf >= k0) & (kf < k1)
                cell_rows.append({'split': k, 'tau_days': f'{t0}-{t1}', 'K/F': f'{k0}-{k1}', 'points': int(m.sum()),
                                  'fallbacks': int((m & bad).sum()), 'invalid_IV': int((m & invalid).sum()),
                                  'below_intrinsic_roundoff': int((m & (stored < lo)).sum()),
                                  'max_128_96_diff': float(diff[m].max()) if m.any() else np.nan})
                per_acc, per_fb = (40, 20) if k == 'train' else (10, 5)
                acc = np.flatnonzero(m & ~bad); fb = np.flatnonzero(m & bad)
                for j in rng.choice(acc, min(per_acc, len(acc)), replace=False):
                    spot_jobs.append((f'{k}:{j}', params[j].tolist(), x[j], tau[j], 1e-10, 'accepted_vs_adaptive_1e-10'))
                for j in rng.choice(fb, min(per_fb, len(fb)), replace=False):
                    spot_jobs.append((f'{k}:{j}', params[j].tolist(), x[j], tau[j], 1e-11, 'fallback_vs_adaptive_1e-11'))
        if k == 'train':
            fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), layout='constrained')
            ax[0].hist(np.log10(np.maximum(diff, 1e-18)), bins=80); ax[0].axvline(-8, color='r', ls='--', label='1e-8 fallback trigger')
            ax[0].set(xlabel='log10 |price(128 nodes) - price(96 nodes)|', ylabel='training points', title='Primary resolution agreement'); ax[0].legend()
            ax[1].scatter(days[~bad], kf[~bad], s=1, c='0.8', label='accepted'); ax[1].scatter(days[bad], kf[bad], s=3, c='r', label=f'fallback ({bad.sum()})')
            ax[1].set(xscale='log', xlabel='maturity (days)', ylabel='K/F', title='Where the independent fallback was used'); ax[1].legend(markerscale=4)
            fig.savefig(dest / 'teacher_resolution_agreement.png', dpi=140); plt.close(fig)
    pd.DataFrame(cell_rows).to_csv(dest / 'cells_maturity_x_moneyness.csv', index=False)

    # ---------------- independent spot checks
    stored = {k: splits[k]['price'] for k in ['train', 'development']}
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=8) as pool:
        out = list(pool.map(_adaptive_job, spot_jobs, chunksize=8))
    rows = []
    for idx, label, v, err, sec in out:
        k, j = idx.split(':'); j = int(j); z = splits[k]
        rows.append({'split': k, 'index': j, 'check': label, 'x': float(z['coords'][j, 0]), 'K/F': float(np.exp(-z['coords'][j, 0])),
                     'tau_days': float(365 * z['coords'][j, 3]), 'v_slow': float(z['params'][j, 4]), 'v_fast': float(z['params'][j, 9]),
                     'teacher': float(stored[k][j]), 'reference': v, 'abs_diff': abs(v - float(stored[k][j])) if np.isfinite(v) else np.nan,
                     'reference_error': err, 'seconds': sec})
    spot = pd.DataFrame(rows); spot.to_csv(dest / 'spotcheck_adaptive_reference.csv', index=False)
    report['spot_checks'] = {'seconds': time.perf_counter() - t0}
    for label, g in spot.groupby('check'):
        report['spot_checks'][label] = {'points': len(g), 'unreliable_reference': int(g.reference_error.notna().sum()),
            'max_abs_diff': float(g.abs_diff.max()), 'p99_abs_diff': float(g.abs_diff.quantile(.99)), 'median_abs_diff': float(g.abs_diff.median()),
            'exceed_protocol_1e-8': int((g.abs_diff > STRICT).sum()), 'exceed_material_2e-6': int((g.abs_diff > MATERIAL).sum()),
            'worst_point': g.loc[g.abs_diff.idxmax(), ['split', 'index', 'K/F', 'tau_days', 'v_slow', 'v_fast', 'teacher', 'reference']].to_dict() if g.abs_diff.notna().any() else None}

    # ---------------- controlled surfaces: static arbitrage + scenario representation in teacher
    cases = read(OUT / 'controlled_cases.json'); arb = []
    tr = splits['train']['params']; lo_s, hi_s = c['pinn']['slow_state_domain']; lo_f, hi_f = c['pinn']['fast_state_domain']; sd = c['pinn']['scale_domain']
    def norm(vs, vf, sc):
        return np.column_stack([np.log(vs / lo_s) / np.log(hi_s / lo_s), np.log(vf / lo_f) / np.log(hi_f / lo_f), (sc - sd[0]) / (sd[1] - sd[0])])
    U = norm(tr[:, 4], tr[:, 9], tr[:, 1] / P[1])
    for case in cases:
        s = np.load(OUT / 'surfaces' / (case['id'] + '.npz')); nt, nk = s['maturity_index'].max() + 1, s['strike_index'].max() + 1
        G = np.full((nt, nk), np.nan); G[s['maturity_index'], s['strike_index']] = s['price']; K = np.exp(-s['x'][s['maturity_index'] == 0][np.argsort(s['strike_index'][s['maturity_index'] == 0])])
        dK = np.diff(G, axis=1); slope = dK / np.diff(K); conv = np.diff(slope, axis=1); cal = np.diff(G, axis=0)
        p = np.array(case['params']); sc = p[1] / P[1]; u = norm(np.array([p[4]]), np.array([p[9]]), np.array([sc]))[0]
        near = int((np.abs(U - u).max(1) <= .05).sum())
        arb.append({'case': case['id'], 'family': case['family'], 'kind': case['kind'], 'v_slow': p[4], 'v_fast': p[9], 'scale': sc,
                    'fast_share': p[9] / (p[4] + p[9]), 'nonfinite': int((~np.isfinite(G)).sum()),
                    'strike_monotonicity_violations_gt_1e-9': int((dK > 1e-9).sum()), 'max_strike_monotonicity_violation': float(max(0, dK.max())),
                    'convexity_violations_gt_1e-7': int((conv < -1e-7).sum()), 'worst_convexity': float(min(0, conv.min())),
                    'calendar_violations_gt_1e-9': int((cal < -1e-9).sum()), 'worst_calendar': float(min(0, cal.min())),
                    'inside_teacher_domain': bool((u >= 0).all() and (u <= 1).all()), 'min_normalized_distance_to_domain_edge': float(min(u.min(), (1 - u).min())),
                    'teacher_training_points_within_0.05_cube': near})
    arb = pd.DataFrame(arb); arb.to_csv(dest / 'controlled_surfaces_arbitrage_and_representation.csv', index=False)
    report['controlled_surfaces'] = {'cases': len(arb), 'nonfinite': int(arb.nonfinite.sum()),
        'strike_monotonicity_violations': int(arb['strike_monotonicity_violations_gt_1e-9'].sum()),
        'convexity_violations': int(arb['convexity_violations_gt_1e-7'].sum()), 'worst_convexity': float(arb.worst_convexity.min()),
        'calendar_violations': int(arb['calendar_violations_gt_1e-9'].sum()), 'worst_calendar': float(arb.worst_calendar.min()),
        'all_inside_teacher_domain': bool(arb.inside_teacher_domain.all()), 'min_distance_to_domain_edge': float(arb.min_normalized_distance_to_domain_edge.min()),
        'teacher_points_near_case_by_family': arb.groupby('family')['teacher_training_points_within_0.05_cube'].agg(['min', 'median', 'max']).to_dict('index')}
    fig, ax = plt.subplots(figsize=(7.5, 6), layout='constrained')
    ax.scatter(tr[:, 4], tr[:, 9], s=.5, c='0.85', label='teacher training states')
    for fam, g in arb.groupby('family'):
        ax.scatter(g.v_slow, g.v_fast, s=28, label=fam, edgecolor='k', linewidth=.4)
    ax.set(xscale='log', yscale='log', xlabel='slow variance state v_slow', ylabel='fast variance state v_fast', title='Controlled scenarios inside the teacher state domain')
    ax.legend(markerscale=1.5, fontsize=8); fig.savefig(dest / 'teacher_state_coverage.png', dpi=140); plt.close(fig)

    # ---------------- decision
    num = report['numerics']; sp = report['spot_checks']
    unstable_reasons = []
    for k in num:
        if num[k]['nonfinite_prices']: unstable_reasons.append(f'{k}: non-finite prices')
        if num[k]['below_intrinsic_beyond_1e-9'] or num[k]['above_forward_bound_beyond_1e-9']: unstable_reasons.append(f'{k}: bound breach beyond 1e-9')
        if not num[k]['fallback_count_matches_audit']: unstable_reasons.append(f'{k}: fallback count does not reproduce audit')
    for label in ['accepted_vs_adaptive_1e-10', 'fallback_vs_adaptive_1e-11']:
        if label in sp:
            if sp[label]['unreliable_reference']: unstable_reasons.append(f'{label}: unreliable reference')
            if sp[label]['exceed_material_2e-6']: unstable_reasons.append(f'{label}: disagreement > 2e-6')
    if any(report['exact_overlap'].values()): unstable_reasons.append('split overlap')
    if report['controlled_surfaces']['nonfinite'] or not report['controlled_surfaces']['all_inside_teacher_domain']: unstable_reasons.append('controlled coverage/finite failure')
    report['decision'] = 'TEACHER_UNSTABLE_DO_NOT_TRAIN' if unstable_reasons else 'TEACHER_ACCEPTED_FOR_TRAINING'
    report['unstable_reasons'] = unstable_reasons
    save(dest / 'teacher_quality.json', json.loads(json.dumps(report, default=float)))
    save(dest / 'output_manifest.json', {'utc': stamp(), 'source_sha256': sha(__file__),
         'files': {p.name: sha(p) for p in sorted(dest.glob('*')) if p.is_file() and p.name != 'output_manifest.json'}})
    print(json.dumps(json.loads(json.dumps(report, default=float)), indent=1))


if __name__ == '__main__':
    main()
