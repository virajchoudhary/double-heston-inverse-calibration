"""Controlled structural analysis. REPORTING/AUDIT ONLY.

Written before `run.py evaluate` was first run, implementing the analyses requested in the
protocol brief: DH-PINN fidelity tables, controlled benchmark tables, the PINN-vs-SH error ratio,
fixed-total-variance analysis, fast- and slow-shock maturity tests, advantage and absolute-error
maps, and paired robustness summaries. Reads frozen evaluate() outputs; changes no model,
checkpoint, baseline, threshold or statistical rule.

The predeclared family-level paired bootstrap in controlled_results.json remains the PRIMARY
inference. Tests added here are SECONDARY/descriptive and their directions are fixed in code
before results were seen: H2 fast shocks concentrate SH error at short maturity relative to BASE;
H3 slow shocks move SH error mass to longer maturity relative to BASE.
Representative regimes for the combined map figure, fixed before results: BASE_0, FAST_SHOCK_1
(largest predeclared fast shock), SLOW_SHOCK_1 (largest predeclared slow shock),
FIXED_TOTAL_TWIST_0 (fast-heavy), FIXED_TOTAL_TWIST_1 (slow-heavy), SMIRK_WEIGHTS_1 (equal weights).
"""
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from run import OUT, verify, read, save, sha, stamp
from literature_exact import iv
from report import markdown

MODELS = ['BS_SELECTED', 'SH_BEST_FOUND', 'DH_PINN', 'DH_EXACT_TEACHER']
MAT = [('7-30d', 7, 30), ('30-90d', 30, 90), ('90-365d', 90, 365), ('365-730d', 365, 731)]
MON = [('K/F<0.90', 0, .9), ('0.90-0.98', .9, .98), ('0.98-1.02', .98, 1.02), ('1.02-1.10', 1.02, 1.1), ('K/F>=1.10', 1.1, 9)]
REPS = ['BASE_representative_0', 'FAST_SHOCK_representative_1', 'SLOW_SHOCK_representative_1',
        'FIXED_TOTAL_TWIST_representative_0', 'FIXED_TOTAL_TWIST_representative_1', 'SMIRK_WEIGHTS_representative_1']
FAMILIES = ['BASE', 'FAST_SHOCK', 'SLOW_SHOCK', 'FIXED_TOTAL_TWIST', 'SMIRK_WEIGHTS']


def err_stats(e, y=None, x=None, tau=None, pred=None):
    e = np.asarray(e, float); a = abs(e)
    out = {'n': int(len(e)), 'price_RMSE': float(np.sqrt(np.mean(e * e))), 'price_MAE': float(a.mean()),
           'price_P95': float(np.quantile(a, .95)), 'price_max': float(a.max())}
    if y is not None:
        u, v = iv(y, x, tau), iv(pred, x, tau); ok = np.isfinite(u) & np.isfinite(v)
        out['IV_RMSE_vol_points'] = float(100 * np.sqrt(np.mean((u[ok] - v[ok]) ** 2))) if ok.any() else np.nan
        out['IV_valid'] = int(ok.sum())
    return out


def grid(Q, case, col):
    a = Q[Q.case.eq(case)].sort_values(['maturity_index', 'strike_index'])
    ni, nj = int(a.maturity_index.max() + 1), int(a.strike_index.max() + 1)
    return a[col].to_numpy().reshape(ni, nj), (~a.calibration).to_numpy().reshape(ni, nj), np.sort(a.strike_over_forward.unique()), np.sort(a.tau.unique()) * 365


def expected_avg_vol(p, T):
    p = np.asarray(p).reshape(2, 5); w = np.zeros_like(T)
    for k, th, _, _, v in p: w = w + th + (v - th) * (1 - np.exp(-k * T)) / (k * T)
    return np.sqrt(w)


def main():
    c = verify()
    assert (OUT / 'controlled_results.json').exists(), 'Run run.py evaluate first'
    lock = read(OUT / 'evaluation_lock.json'); assert lock['manifest_sha256'] == sha(OUT / 'manifest.json')
    for s, d in lock['models'].items(): assert d == sha(OUT / f'pinn_s{s}/weights.pt')
    for n, d in lock['baselines'].items(): assert d == sha(OUT / 'baselines' / n)
    dest = OUT / 'controlled_analysis'; dest.mkdir(exist_ok=False); figs = dest / 'figures'; figs.mkdir()
    R = read(OUT / 'controlled_results.json'); fid = R['fidelity']; S = pd.DataFrame(R['statistics'])
    Q = pd.read_csv(OUT / 'controlled_predictions.csv'); M = pd.read_csv(OUT / 'controlled_metrics.csv')
    B = pd.read_csv(OUT / 'bucket_metrics.csv'); F = pd.read_csv(OUT / 'fidelity_predictions.csv')
    cases = {r['id']: r for r in read(OUT / 'controlled_cases.json')}
    meta = pd.DataFrame(cases.values())[['id', 'family', 'kind']].rename(columns={'id': 'case'})
    B = B.merge(meta, on='case', validate='many_to_one')
    summary = {'utc': stamp(), 'source_sha256': sha(__file__), 'protocol_sha256': sha(OUT / 'manifest.json'),
               'controlled_results_sha256': sha(OUT / 'controlled_results.json'), 'inference_note': 'family bootstrap in controlled_results.json is primary; tests here are secondary'}
    md = ['# Controlled structural analysis', 'Reporting only. Primary inference: the predeclared family-level paired bootstrap (Table C). Other tests are secondary and descriptive.']

    # ---------------- Table A: DH-PINN fidelity
    keys = ['price_RMSE', 'price_MAE', 'price_P95', 'price_max', 'IV_RMSE_volatility_points', 'IV_valid_quotes', 'quotes']
    A1 = pd.DataFrame([{'model': 'DH-PINN two-seed mean', **{k: fid[k] for k in keys}, 'all_frozen_gates_pass': fid['pass']}] +
                      [{'model': f'DH-PINN seed {s}', **{k: m[k] for k in keys}, 'all_frozen_gates_pass': ''} for s, m in zip(c['pinn']['seeds'], fid['seed_metrics'])])
    A1.to_csv(dest / 'tableA1_fidelity_heldout_overall.csv', index=False)
    x, tau, y, pr = F.x.to_numpy(), F.tau.to_numpy(), F.teacher.to_numpy(), F.pinn.to_numpy(); days = 365 * tau; kf = np.exp(-x)
    rows = []
    for lab, a, b in MAT:
        m = (days >= a) & (days < b); rows.append({'type': 'maturity', 'bucket': lab, **err_stats(pr[m] - y[m], y[m], x[m], tau[m], pr[m])})
    for lab, a, b in MON:
        m = (kf >= a) & (kf < b); rows.append({'type': 'moneyness', 'bucket': lab, **err_stats(pr[m] - y[m], y[m], x[m], tau[m], pr[m])})
    for lab, m in [('fast-heavy (v_fast>v_slow)', (F.v_fast > F.v_slow).to_numpy()), ('slow-heavy (v_slow>=v_fast)', (F.v_fast <= F.v_slow).to_numpy())]:
        rows.append({'type': 'state allocation', 'bucket': lab, **err_stats(pr[m] - y[m], y[m], x[m], tau[m], pr[m])})
    A2 = pd.DataFrame(rows); A2.to_csv(dest / 'tableA2_fidelity_heldout_by_bucket.csv', index=False)
    held = Q[~Q.calibration]; rows = []
    for (fam, kind), g in held.groupby(['family', 'kind']):
        rows.append({'family': fam, 'kind': kind, 'surfaces': g.case.nunique(),
                     **err_stats(g.DH_PINN_residual.to_numpy(), g.DH_EXACT_TEACHER.to_numpy(), g.x.to_numpy(), g.tau.to_numpy(), g.DH_PINN.to_numpy())})
    A3 = pd.DataFrame(rows); A3.to_csv(dest / 'tableA3_fidelity_by_structural_scenario.csv', index=False)
    md += ['## Table A — DH-PINN fidelity (PINN vs exact Double Heston)', '### A1. Held-out continuous fidelity set (8,192 points, seed 93104); frozen gates: RMSE<=2e-5, P95<=5e-5, max<=2e-4, IV RMSE<=0.2 vol pts',
           markdown(A1), f"Fresh PDE residual RMSE by seed: {fid['fresh_scaled_PDE_RMSE_by_seed']}",
           '### A2. By maturity, moneyness and state allocation (same held-out set)', markdown(A2),
           '### A3. On the controlled surfaces, held-out cells only, pooled per family and kind', markdown(A3)]

    # ---------------- Table B/C and the E_PINN << E_SH condition
    ind = M[M.kind.eq('independent')]; hi = held[held.kind.eq('independent')]
    Brows = []
    for m in MODELS:
        g = ind[ind.model.eq(m)]
        Brows.append({'model': m, 'surfaces': len(g), 'mean_price_RMSE': g.price_RMSE.mean(), 'median_price_RMSE': g.price_RMSE.median(),
                      'mean_price_MAE': g.price_MAE.mean(), 'mean_IV_RMSE_vol_points': g.IV_RMSE_volatility_points.mean(),
                      'pooled_heldout_price_RMSE': float(np.sqrt(np.mean(hi[m + '_residual'].to_numpy() ** 2)))})
    TB = pd.DataFrame(Brows); TB.to_csv(dest / 'tableB_controlled_benchmark_overall.csv', index=False)
    Crows = []
    for fam in FAMILIES:
        for m in MODELS:
            g = ind[ind.family.eq(fam) & ind.model.eq(m)]; h = hi[hi.family.eq(fam)]
            Crows.append({'family': fam, 'model': m, 'mean_price_RMSE': g.price_RMSE.mean(), 'mean_price_MAE': g.price_MAE.mean(),
                          'mean_IV_RMSE_vol_points': g.IV_RMSE_volatility_points.mean(), 'pooled_heldout_price_RMSE': float(np.sqrt(np.mean(h[m + '_residual'].to_numpy() ** 2)))})
    TC = pd.DataFrame(Crows); TC.to_csv(dest / 'tableC_controlled_benchmark_by_regime.csv', index=False)
    piv = ind.pivot(index='case', columns='model', values='price_RMSE').join(meta.set_index('case'))
    piv['PINN_over_SH'] = piv.DH_PINN / piv.SH_BEST_FOUND; piv['SH_minus_PINN'] = piv.SH_BEST_FOUND - piv.DH_PINN
    ratio = piv.groupby('family').agg(surfaces=('PINN_over_SH', 'size'), median_PINN_over_SH=('PINN_over_SH', 'median'),
                                      max_PINN_over_SH=('PINN_over_SH', 'max'), fraction_ratio_le_0_2=('PINN_over_SH', lambda r: float((r <= .2).mean())),
                                      PINN_beats_SH_fraction=('SH_minus_PINN', lambda d: float((d > 0).mean()))).reindex(FAMILIES).reset_index()
    ratio.to_csv(dest / 'table_condition_PINN_error_vs_SH_error.csv', index=False)
    S.to_csv(dest / 'tableC_primary_paired_statistics.csv', index=False)
    rng = np.random.default_rng(93120); d = piv.SH_minus_PINN.to_numpy(); boot = rng.choice(d, (5000, len(d))).mean(1)
    pooled = {'surfaces': len(d), 'mean_SH_minus_PINN': float(d.mean()), 'median': float(np.median(d)), 'PINN_wins_fraction': float((d > 0).mean()),
              'bootstrap95_SECONDARY_seed93120': np.quantile(boot, [.025, .975]).tolist(), 'median_PINN_over_SH': float(piv.PINN_over_SH.median())}
    summary['all_independent_surfaces_pooled_SECONDARY'] = pooled
    md += ['## Table B — Controlled structural benchmark, all 40 independent surfaces (held-out cells)',
           'The exact DH teacher generated these surfaces, so its zero error is tautological, not a model-comparison victory. The meaningful comparison is DH-PINN vs best-found SH.',
           markdown(TB), '## Table C — By regime', markdown(TC),
           '### C-primary. Predeclared paired statistics (positive advantage = SH RMSE minus PINN RMSE favours the PINN)',
           markdown(S[['family', 'surfaces', 'mean_advantage', 'median_advantage', 'DH_win_fraction', 'paired_bootstrap95', 'paired_bootstrap99_bonferroni', 'DH_over_SH_pooled_RMSE', 'reliable_structural_advantage']]),
           '### Condition E_PINN->DH << E_SH*->DH, per surface', markdown(ratio),
           f'Secondary, all 40 surfaces pooled: `{pooled}`']

    # ---------------- H2 / H3: where the SH error mass sits along maturity
    sh = B[B.model.eq('SH_BEST_FOUND') & B.kind.eq('independent') & B.bucket.isin(['short', 'medium', 'long', 'very_long'])].copy()
    sh['SSE'] = sh.quotes * sh.price_RMSE ** 2
    mass = sh.pivot_table(index=['case', 'family'], columns='bucket', values='SSE').reset_index()
    tot = mass[['short', 'medium', 'long', 'very_long']].sum(1)
    mass['share_short_le30d'] = mass.short / tot; mass['share_long_gt90d'] = (mass.long + mass.very_long) / tot
    shr = sh.pivot_table(index=['case', 'family'], columns='bucket', values='price_RMSE').reset_index()
    mass = mass.merge(shr, on=['case', 'family'], suffixes=('_SSE', '_RMSE'))
    mass.to_csv(dest / 'maturity_error_mass_SH_by_surface.csv', index=False)
    def mw(fam, col, alt):
        a = mass[mass.family.eq(fam)][col]; b = mass[mass.family.eq('BASE')][col]
        return {'family': fam, 'metric': col, 'family_median': float(a.median()), 'BASE_median': float(b.median()),
                'alternative': alt, 'mann_whitney_p_one_sided': float(stats.mannwhitneyu(a, b, alternative=alt).pvalue), 'n': [len(a), len(b)]}
    H = pd.DataFrame([mw('FAST_SHOCK', 'share_short_le30d', 'greater'), mw('SLOW_SHOCK', 'share_long_gt90d', 'greater'),
                      mw('FAST_SHOCK', 'share_long_gt90d', 'less'), mw('SLOW_SHOCK', 'share_short_le30d', 'less')])
    H.to_csv(dest / 'tests_H2_H3_secondary.csv', index=False)
    bk = B[B.kind.eq('independent') & B.model.isin(['SH_BEST_FOUND', 'DH_PINN', 'BS_SELECTED']) & B.bucket.isin(['short', 'medium', 'long', 'very_long', 'ATM', 'wing'] + [f'K/F_{a:.2f}_{b:.2f}' for a, b in zip([.7, .9, .98, 1.02, 1.1], [.9, .98, 1.02, 1.1, 1.30000001])])]
    BT = bk.pivot_table(index=['family', 'bucket'], columns='model', values='price_RMSE', aggfunc='mean').reset_index()
    BT['SH_over_PINN'] = BT.SH_BEST_FOUND / BT.DH_PINN; BT.to_csv(dest / 'table_bucket_errors_by_family.csv', index=False)
    curves = held[held.kind.eq('independent')].assign(days=lambda d: d.tau * 365).groupby(['family', 'case', 'days']).agg(
        SH=('SH_BEST_FOUND_residual', lambda e: float(np.sqrt(np.mean(e ** 2)))), PINN=('DH_PINN_residual', lambda e: float(np.sqrt(np.mean(e ** 2))))).reset_index()
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    for fam in FAMILIES:
        g = curves[curves.family.eq(fam)].groupby('days')[['SH', 'PINN']].mean()
        ln = ax[0].plot(g.index, g.SH, label=f'{fam} SH', lw=1.8)[0]; ax[0].plot(g.index, g.PINN, ls=':', c=ln.get_color(), lw=1.2)
        z = curves[curves.family.eq(fam)].copy(); z['SHn'] = z.SH / z.groupby('case').SH.transform('max'); gz = z.groupby('days').SHn.mean()
        ax[1].plot(gz.index, gz, label=fam, c=ln.get_color(), lw=1.8)
    for a in ax: a.set(xscale='log', xlabel='days to expiry'); a.axvline(np.log(2) / 10.7526 * 365, c='0.5', ls='--', lw=.8); a.axvline(np.log(2) / .9491 * 365, c='0.5', ls='-.', lw=.8)
    ax[0].set(yscale='log', ylabel='held-out price RMSE per maturity (mean of 8 surfaces)', title='Solid: best-found SH; dotted: DH-PINN'); ax[0].legend(fontsize=7, ncol=2)
    ax[1].set(ylabel='SH error / its own maximum (shape only)', title='Where SH error sits along maturity (dashed: fast half-life 24d; dash-dot: slow 267d)'); ax[1].legend(fontsize=8)
    fig.savefig(figs / 'maturity_profiles_by_regime.png', dpi=150); plt.close(fig)
    md += ['## Fast- and slow-shock maturity analysis (secondary)', 'Fast factor half-life ln2/10.7526 = 23.5 days; slow ln2/0.9491 = 267 days. The share of SH held-out squared error falling in each maturity range is compared with BASE (8 vs 8 surfaces, one-sided Mann-Whitney, directions fixed before results).',
           markdown(H), '![maturity profiles](figures/maturity_profiles_by_regime.png)',
           '### Mean held-out price RMSE by bucket and regime (8 independent surfaces each)', markdown(BT)]

    # ---------------- Fixed-total-variance twist (central figure)
    tw = ['FIXED_TOTAL_TWIST_representative_0', 'FIXED_TOTAL_TWIST_representative_1']; lab = {tw[0]: 'fast-heavy (v_fast .035, v_slow .005)', tw[1]: 'slow-heavy (v_fast .005, v_slow .035)'}
    base = {k: read(OUT / 'baselines' / (k + '.json')) for k in tw}
    fig, ax = plt.subplots(2, 2, figsize=(14, 10), layout='constrained'); T = np.geomspace(7 / 365, 2, 200)
    twrows = []
    for k, col in zip(tw, ['C0', 'C3']):
        p = np.array(cases[k]['params']); a = Q[Q.case.eq(k)]; atm = a[np.isclose(a.strike_over_forward, 1.)].sort_values('tau')
        ax[0, 0].plot(T * 365, 100 * expected_avg_vol(p, T), c=col, label=lab[k])
        for name, ls in [('DH_EXACT_TEACHER', '-'), ('SH_BEST_FOUND', '--'), ('DH_PINN', ':')]:
            ax[0, 1].plot(atm.tau * 365, 100 * iv(atm[name].to_numpy(), atm.x.to_numpy(), atm.tau.to_numpy()), c=col, ls=ls, label=f'{name} {lab[k].split()[0]}')
        cur = a[~a.calibration].groupby('tau').agg(SH=('SH_BEST_FOUND_residual', lambda e: np.sqrt(np.mean(e ** 2))), PINN=('DH_PINN_residual', lambda e: np.sqrt(np.mean(e ** 2))))
        ax[1, 0].plot(cur.index * 365, cur.SH, c=col, label=f'SH, {lab[k].split()[0]}'); ax[1, 0].plot(cur.index * 365, cur.PINN, c=col, ls=':', label=f'PINN, {lab[k].split()[0]}')
        ax[1, 1].plot(atm.tau * 365, 1e4 * (atm.SH_BEST_FOUND - atm.DH_EXACT_TEACHER), c=col, label=f'SH - teacher, {lab[k].split()[0]}')
        bb = B[B.case.eq(k) & B.bucket.isin(['short', 'medium', 'long', 'very_long'])].pivot(index='bucket', columns='model', values='price_RMSE')
        shp = base[k]['SH']['best']['params']
        twrows.append({'case': k, 'allocation': lab[k], **{f'SH_RMSE_{b}': float(bb.loc[b, 'SH_BEST_FOUND']) for b in ['short', 'medium', 'long', 'very_long']},
                       **{f'PINN_RMSE_{b}': float(bb.loc[b, 'DH_PINN']) for b in ['short', 'medium', 'long', 'very_long']},
                       'SH_kappa': shp[0], 'SH_theta': shp[1], 'SH_sigma': shp[2], 'SH_rho': shp[3], 'SH_v0': shp[4],
                       'SH_near_best_starts': base[k]['SH']['converged_near_best'], 'ATM_SH_residual_sign_changes_along_maturity': int((np.diff(np.sign(atm.SH_BEST_FOUND - atm.DH_EXACT_TEACHER)) != 0).sum())})
    ind_tw = piv[piv.family.eq('FIXED_TOTAL_TWIST')].copy(); ind_tw['fast_share'] = [cases[i]['params'][9] / (cases[i]['params'][4] + cases[i]['params'][9]) for i in ind_tw.index]
    ax[0, 0].set(xscale='log', xlabel='days', ylabel='sqrt(expected average total variance) %', title='Same current total variance (4%), different future variance'); ax[0, 0].legend()
    ax[0, 1].set(xscale='log', xlabel='days', ylabel='ATM implied vol %', title='ATM term structure: teacher (solid), refit SH (dashed), PINN (dotted)'); ax[0, 1].legend(fontsize=7)
    ax[1, 0].set(xscale='log', yscale='log', xlabel='days', ylabel='held-out price RMSE per maturity', title='Best-found SH (solid) vs DH-PINN (dotted)'); ax[1, 0].legend(fontsize=8)
    ax[1, 1].axhline(0, c='k', lw=.6); ax[1, 1].set(xscale='log', xlabel='days', ylabel='ATM price residual x 1e4 (forward units)', title='Signed SH misfit along maturity (all ATM cells)'); ax[1, 1].legend(fontsize=8)
    fig.savefig(figs / 'CENTRAL_fixed_total_variance_twist.png', dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5), layout='constrained')
    ax.scatter(ind_tw.fast_share, ind_tw.SH_BEST_FOUND, label='best-found SH'); ax.scatter(ind_tw.fast_share, ind_tw.DH_PINN, marker='x', label='DH-PINN')
    ax.set(yscale='log', xlabel='fast-factor share of current total variance', ylabel='held-out price RMSE', title='8 independent fixed-total twist surfaces'); ax.legend()
    fig.savefig(figs / 'twist_error_vs_fast_share.png', dpi=150); plt.close(fig)
    TW = pd.DataFrame(twrows); TW.to_csv(dest / 'table_fixed_total_twist_representatives.csv', index=False)
    ind_tw[['family', 'fast_share', 'SH_BEST_FOUND', 'DH_PINN', 'BS_SELECTED', 'PINN_over_SH']].to_csv(dest / 'table_fixed_total_twist_independent.csv')
    md += ['## Fixed-total-variance twist (central diagnostic)', 'Both representatives start at total variance 0.04 (20% vol); only the fast/slow allocation differs. SH is refit separately to each.',
           '![central](figures/CENTRAL_fixed_total_variance_twist.png)', markdown(TW),
           '![twist share](figures/twist_error_vs_fast_share.png)', markdown(ind_tw[['fast_share', 'SH_BEST_FOUND', 'DH_PINN', 'PINN_over_SH']].reset_index())]

    # ---------------- Advantage maps: representatives and family means (held-out cells only)
    def three(row_axes, esh, edh, hold, k, days, title):
        lo = max(float(np.nanmin(np.where(hold, np.minimum(esh, edh), np.nan))), 1e-12); hi_ = float(np.nanmax(np.where(hold, np.maximum(esh, edh), np.nan)))
        adv = esh - edh; lim = float(np.nanmax(abs(np.where(hold, adv, np.nan)))) or 1e-12
        for ax, z, t, norm, cmap in [(row_axes[0], esh, 'best-found SH |error|', LogNorm(lo, hi_), 'viridis'), (row_axes[1], edh, 'DH-PINN |error|', LogNorm(lo, hi_), 'viridis'),
                                     (row_axes[2], adv, 'A = |SH err| - |PINN err|', SymLogNorm(linthresh=max(lim * 1e-3, 1e-12), vmin=-lim, vmax=lim), 'coolwarm')]:
            im = ax.pcolormesh(k, days, np.where(hold, np.maximum(z, 1e-14) if 'error|' in t else z, np.nan), shading='nearest', norm=norm, cmap=cmap)
            ax.set(yscale='log', xlabel='K/F', ylabel='days', title=f'{title}: {t}'); plt.colorbar(im, ax=ax)
        return {'held_out_cells': int(hold.sum()), 'fraction_A_positive': float((adv[hold] > 0).mean()), 'mean_A': float(adv[hold].mean()),
                'mean_SH_abs_error': float(esh[hold].mean()), 'mean_PINN_abs_error': float(edh[hold].mean())}
    fig, axes = plt.subplots(len(REPS), 3, figsize=(17, 4 * len(REPS)), layout='constrained'); maprows = []
    for i, k in enumerate(REPS):
        esh, hold, kk, days = grid(Q, k, 'SH_BEST_FOUND_residual'); edh = grid(Q, k, 'DH_PINN_residual')[0]
        maprows.append({'map': k, **three(axes[i], abs(esh), abs(edh), hold, kk, days, k.replace('_representative_', ' rep '))})
    fig.savefig(figs / 'advantage_maps_representatives.png', dpi=110); plt.close(fig)
    fig, axes = plt.subplots(len(FAMILIES), 3, figsize=(17, 4 * len(FAMILIES)), layout='constrained')
    for i, fam in enumerate(FAMILIES):
        ids = [k for k, r in cases.items() if r['family'] == fam and r['kind'] == 'independent']
        esh = np.mean([abs(grid(Q, k, 'SH_BEST_FOUND_residual')[0]) for k in ids], 0); edh = np.mean([abs(grid(Q, k, 'DH_PINN_residual')[0]) for k in ids], 0)
        _, hold, kk, days = grid(Q, ids[0], 'SH_BEST_FOUND_residual')
        maprows.append({'map': f'{fam} mean of {len(ids)} independent surfaces', **three(axes[i], esh, edh, hold, kk, days, f'{fam} (mean of {len(ids)})')})
    fig.savefig(figs / 'advantage_maps_family_means.png', dpi=110); plt.close(fig)
    MP = pd.DataFrame(maprows); MP.to_csv(dest / 'table_advantage_maps.csv', index=False)
    md += ['## Advantage maps', 'A(K,T) = |SH_best - DH_exact| - |DH_PINN - DH_exact| on held-out cells; A>0 means the PINN is closer to the two-factor teacher. Absolute-error maps share one log colour scale per row.',
           '![representatives](figures/advantage_maps_representatives.png)', '![family means](figures/advantage_maps_family_means.png)', markdown(MP)]

    # ---------------- Paired robustness figure
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    data = [piv[piv.family.eq(f)].SH_minus_PINN.to_numpy() for f in FAMILIES]; ratios = [np.log10(piv[piv.family.eq(f)].PINN_over_SH.to_numpy()) for f in FAMILIES]
    for j, (dd, rr) in enumerate(zip(data, ratios)):
        ax[0].scatter(np.full(len(dd), j) + np.linspace(-.15, .15, len(dd)), dd, s=18); ax[1].scatter(np.full(len(rr), j) + np.linspace(-.15, .15, len(rr)), rr, s=18)
        ci = S[S.family.eq(FAMILIES[j])].paired_bootstrap95.iloc[0]; ax[0].errorbar(j + .3, np.mean(dd), yerr=[[np.mean(dd) - ci[0]], [ci[1] - np.mean(dd)]], fmt='ks', capsize=4)
    ax[0].boxplot(data, positions=range(len(FAMILIES)), widths=.5, showfliers=False); ax[0].axhline(0, c='k', ls='--')
    ax[0].set(xticks=range(len(FAMILIES)), xticklabels=FAMILIES, ylabel='SH RMSE - PINN RMSE (per surface)', title='Paired differences; black: mean and predeclared bootstrap 95% CI')
    ax[1].axhline(np.log10(.2), c='r', ls='--', label='predeclared ratio gate 0.2'); ax[1].set(xticks=range(len(FAMILIES)), xticklabels=FAMILIES, ylabel='log10(PINN RMSE / SH RMSE)', title='Error ratio per surface'); ax[1].legend()
    for a in ax: a.tick_params(axis='x', rotation=20)
    fig.savefig(figs / 'paired_robustness.png', dpi=150); plt.close(fig)
    md += ['## Paired robustness', '![paired](figures/paired_robustness.png)']

    # ---------------- SH optimizer diagnostics
    rows = []
    for k, r in cases.items():
        b = read(OUT / 'baselines' / (k + '.json'))['SH']; p = b['best']['params']
        rows.append({'case': k, 'family': r['family'], 'kind': r['kind'], 'best_MSE': b['best']['objective'], 'near_best_starts_of_12': b['converged_near_best'],
                     'best_at_bound': b['best']['near_bound'], 'failed_starts': sum(s['objective'] is None for s in b['starts']), 'kappa': p[0], 'theta': p[1], 'sigma': p[2], 'rho': p[3], 'v0': p[4]})
    SHD = pd.DataFrame(rows); SHD.to_csv(dest / 'table_SH_optimizer_diagnostics.csv', index=False)
    SHS = SHD.groupby('family').agg(min_near_best=('near_best_starts_of_12', 'min'), at_bound=('best_at_bound', 'sum'), failed=('failed_starts', 'sum'),
                                    kappa_min=('kappa', 'min'), kappa_max=('kappa', 'max'), rho_min=('rho', 'min'), rho_max=('rho', 'max')).reset_index()
    md += ['## Single-Heston optimizer diagnostics (global search + 12 local starts per surface)', markdown(SHS)]
    summary['tables'] = sorted(p.name for p in dest.glob('*.csv'))
    save(dest / 'analysis_summary.json', summary)
    (dest / 'CONTROLLED_ANALYSIS.md').write_text('\n\n'.join(md) + '\n')
    save(dest / 'output_manifest.json', {'utc': stamp(), 'source_sha256': sha(__file__),
         'files': {str(p.relative_to(dest)): sha(p) for p in sorted(dest.rglob('*')) if p.is_file() and p.name != 'output_manifest.json'}})
    print('CONTROLLED ANALYSIS', dest / 'CONTROLLED_ANALYSIS.md', flush=True)


if __name__ == '__main__':
    main()
