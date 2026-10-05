"""Retrospective fixed-parameter C3 pricing test; never trains or fits parameters.

ponytail: reuse saved C3 and the checked numerical reference; no new architecture.
Freeze before run. Earlier studies and source trade archives remain read-only.
"""
import argparse
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "artifacts"
BTC = ROOT / "experiments/btc_multifactor_v1"
V4 = ROOT / "experiments/nifty_multifactor_v4"
sys.path.insert(0, str(V4))
sys.path.insert(0, str(ROOT))
from literature_exact import exact, black, admissible, iv
from src.mentor_dh_pinn.regular_pinn_torch import TorchRegularVariancePINN


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def scenario_bank():
    cfg = read(V4 / "config.json")
    base = np.array(cfg["published_double_slow_first"]).reshape(2, 5)
    sh = np.array(cfg["published_single"])
    rows = []
    for vol in [.28, .45, .60]:
        # All scenarios declared before scoring; NOT claimed published BTC fits.
        scale = vol**2 / base[:, 4].sum() if vol == .28 else 3.2
        p = base.copy()
        p[:, 1] *= scale
        p[:, 2] *= np.sqrt(scale)
        p[:, 4] *= scale
        if vol != .28:
            p[:, 4] = vol**2 * np.array([.4, .6])
        s = sh.copy()
        theta_ratio = p[:, 1].sum() / s[1]
        s[1] *= theta_ratio
        s[2] *= np.sqrt(theta_ratio)
        s[4] = vol**2
        admissible(p.ravel()); admissible(s)
        assert .7 <= scale <= 3.2
        assert .00015 <= p[0, 4] <= .18 and .001 <= p[1, 4] <= .22
        rows.append({"id": f"fixed_{round(vol * 100)}pct", "initial_vol": vol,
                     "structural_scale": scale, "double_slow_first": p.ravel().tolist(),
                     "single": s.tolist(), "BS_vol": vol})
    return rows


def freeze():
    assert not OUT.exists(), "Do not overwrite a frozen experiment"
    OUT.mkdir()
    rows = [r for r in read(BTC / "artifacts/data_audit.json")["dates"] if r["stage"] == "test"]
    assert len({r['date'] for r in rows}) == len(rows)
    raw = {str((BTC / "artifacts/raw" / (r['date'] + '.json.gz')).relative_to(ROOT)): r['raw_sha256'] for r in rows}
    for path, digest in raw.items():
        assert sha(ROOT / path) == digest
    selection = read(V4 / "artifacts/pinn_selection.json")
    assert selection['chosen'] == 'C3'
    weights = {str((V4 / f'artifacts/pinn_s{s}/weights.pt').relative_to(ROOT)): selection['checkpoint_sha256'][str(s)] for s in [17, 43]}
    for path, digest in weights.items():
        assert sha(ROOT / path) == digest
    sources = [Path(__file__), V4 / 'literature_exact.py', V4 / 'config.json',
               ROOT / 'src/mentor_dh_pinn/regular_pinn_torch.py', ROOT / 'src/double_heston.py',
               ROOT / 'src/double_heston_reference.py', ROOT / 'src/mentor_dh_pinn/regular_pinn_data.py',
               ROOT / 'src/mentor_dh_pinn/torch_pricer.py']
    protocol = {
        'utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'role': 'RETROSPECTIVE_EXPLORATORY_NOT_UNTOUCHED_CONFIRMATION',
        'date_policy': 'All archived original stage=test dates, including previously unusable dates. No replacements.',
        'dates': [{k: r[k] for k in ['date', 'regime', 'stage']} for r in rows],
        'scenarios': scenario_bank(), 'parameter_order': ['kappa_slow','theta_slow','sigma_slow','rho_slow','v0_slow','kappa_fast','theta_fast','sigma_fast','rho_fast','v0_fast'],
        'selection': 'No fitting, ranking-based selection, training, or parameter adjustment. Report all three scenarios.',
        'scenario_provenance': 'Existing repository Chang/Wang/Zhang 2021 literature shape. 28% uses existing variance scaling. 45% and 60% are proposed stress states, 40/60 slow/fast share and structural scale 3.2, NOT published Bitcoin estimates.',
        'comparators': 'SH numerical: published one-factor shape rescaled to match each DH initial and long-run total variance. BS: same fixed initial volatility. Neither comparator is a PINN or optimally calibrated.',
        'forward_rule': 'First hour [06:00,07:00) UTC only; latest trade/instrument. Three call-put pairs nearest anchor median index (max log K/index .36), timestamps within 10 minutes, index discrepancy <=0.5%. Median of (K+C_BTC*S_C-P_BTC*S_P)/mean(S_C,S_P). F at later trade = frozen ratio * its index. No target IV or price enters forward.',
        'score_rule': 'Latest trade/instrument in [07:00,08:00), excluding both option types of anchor strikes. Finite nonnegative premium, positive index, 7<=days<=730, |log(F/K)|<=.36, OTM only. No market IV, vega, or model-error filter. Report every exclusion.',
        'pricing_units': 'Observed USD premium = BTC premium * contemporaneous index. Model USD premium = F * normalized numerical call/put. Zero USD discount rate assumed, same for all models. F/index basis is carried forward from first hour.',
        'official_convention_source': 'https://support.deribit.com/hc/en-us/articles/31424939096093-Inverse-Options',
        'source_endpoint': read(BTC / 'config.json')['sources']['trades'],
        'primary_metric': 'Per scenario, pooled and median-date forward-normalized market price RMSE; USD RMSE also reported. Separate PINN-vs-exact fidelity. No p-value superiority claims.',
        'fidelity_gates_inherited': read(V4 / 'config.json')['pinn']['fidelity_gate'],
        'caveats': ['Prior results on these dates have been viewed; not pristine test data.', 'Asynchronous last trades, not synchronized bid/ask quotes; no execution or spread claims.', 'Anchor forward estimates and assumed zero discounting can introduce model-independent pricing errors.', 'No direct historic futures prices in archive; anchor basis held constant for the following hour.', 'Three fixed hypotheses are not an optimization over Bitcoin parameters; no universal superiority or recovery claim.'],
        'hashes': {**raw, **weights, **{str(p.relative_to(ROOT)): sha(p) for p in sources}},
    }
    save(OUT / 'protocol.json', protocol)
    (OUT / 'protocol.sha256').write_text(sha(OUT / 'protocol.json') + '\n')
    print('FROZEN', len(rows), 'historical dates; three fixed scenarios', flush=True)


def verify():
    assert sha(OUT / 'protocol.json') == (OUT / 'protocol.sha256').read_text().strip()
    p = read(OUT / 'protocol.json')
    for path, digest in p['hashes'].items():
        assert sha(ROOT / path) == digest, path
    return p


def parse(trades, date):
    rows, seen = [], {}
    audit = {'raw_trades': len(trades), 'identical_duplicate_trades': 0, 'invalid_rows': 0}
    for t in trades:
        key = str(t['trade_id'])
        if key in seen:
            assert seen[key] == t, 'Conflicting duplicate trade ID'
            audit['identical_duplicate_trades'] += 1
            continue
        seen[key] = t
        try:
            currency, expcode, strike, cp = t['instrument_name'].split('-')
            assert currency == 'BTC' and cp in ['C','P']
            exp = dt.datetime.strptime(expcode, '%d%b%y').replace(hour=8, tzinfo=dt.timezone.utc)
            ts = dt.datetime.fromtimestamp(t['timestamp']/1000, dt.timezone.utc)
            price, index, strike = float(t['price']), float(t['index_price']), float(strike)
            assert str(ts.date()) == date and 6 <= ts.hour < 8
            assert np.isfinite([price,index,strike]).all() and price >= 0 and index > 0 and strike > 0
            tau = (exp-ts).total_seconds()/(365*86400)
            assert tau > 0
        except (KeyError, ValueError, AssertionError):
            audit['invalid_rows'] += 1
            continue
        rows.append({'trade_id':key, 'instrument':t['instrument_name'], 'expiry':str(exp.date()), 'strike':strike,
                     'cp':cp, 'timestamp':t['timestamp'], 'hour':ts.hour, 'tau':tau, 'index':index, 'price_btc':price})
    return pd.DataFrame(rows), audit


def prepare(trades, date):
    a, audit = parse(trades,date)
    audit['date'] = date
    if a.empty:
        return a, [], {**audit, 'scored':0, 'reason':'No valid trades'}
    earlier = a[a.hour.eq(6)].sort_values(['timestamp','trade_id']).groupby('instrument').tail(1)
    later = a[a.hour.eq(7)].sort_values(['timestamp','trade_id']).groupby('instrument').tail(1).copy()
    audit.update(earlier_instruments=len(earlier), later_instruments=len(later))
    forward_map, excluded, anchors, failures = {}, set(), [], []
    for expiry, group in earlier.groupby('expiry'):
        c = group[group.cp.eq('C')]; p = group[group.cp.eq('P')]
        pairs = c.merge(p,on=['expiry','strike'],suffixes=('_c','_p'))
        if pairs.empty:
            failures.append({'expiry':expiry,'reason':'no paired earlier trades'}); continue
        pairs['index_mean'] = (pairs.index_c+pairs.index_p)/2
        eligible = (abs(pairs.timestamp_c-pairs.timestamp_p)<=600000) & (abs(pairs.index_c/pairs.index_p-1)<=.005)
        eligible &= abs(np.log(pairs.strike/pairs.index_mean)) <= .36
        pairs = pairs[eligible].copy()
        if len(pairs)<3:
            failures.append({'expiry':expiry,'reason':'fewer than three synchronized-enough pairs','pairs':len(pairs)}); continue
        pairs['distance'] = abs(np.log(pairs.strike/float(group['index'].median())))
        pairs = pairs.sort_values(['distance','strike']).head(3)
        ratios = (pairs.strike+pairs.price_btc_c*pairs.index_c-pairs.price_btc_p*pairs.index_p)/pairs.index_mean
        if not np.isfinite(ratios).all() or (ratios<=0).any():
            failures.append({'expiry':expiry,'reason':'nonpositive or nonfinite anchor forward'}); continue
        forward_map[expiry] = float(ratios.median())
        for (_,row), ratio in zip(pairs.iterrows(),ratios):
            excluded.add((expiry,row.strike))
            anchors.append({'date':date,'expiry':expiry,'strike':row.strike,'call_id':row.trade_id_c,'put_id':row.trade_id_p,
                            'call_timestamp':int(row.timestamp_c),'put_timestamp':int(row.timestamp_p),'ratio':float(ratio)})
    audit['anchor_pairs'] = len(anchors); audit['anchor_expiry_failures'] = failures
    later['forward'] = later.expiry.map(forward_map)*later['index']
    # Sequential reasons, recorded once per candidate; no target-dependent IV cut.
    later['reason'] = ''
    def reject(mask, reason):
        later.loc[mask & later.reason.eq(''),'reason'] = reason
    reject(~np.isfinite(later.forward),'no_earlier_anchor_forward')
    reject(pd.Series([(e,k) in excluded for e,k in zip(later.expiry,later.strike)],index=later.index),'anchor_strike_reserved')
    reject(~later.tau.between(7/365,2),'outside_maturity_training_domain')
    later['x'] = np.log(later.forward/later.strike)
    reject(later.x.abs()>.36,'outside_moneyness_training_domain')
    reject(~np.where(later.cp.eq('C'),later.strike>=later.forward,later.strike<later.forward),'not_OTM')
    audit['exclusions'] = {str(k):int(v) for k,v in later[later.reason.ne('')].reason.value_counts().items()}
    q = later[later.reason.eq('')].copy().sort_values('instrument').reset_index(drop=True)
    q['date'] = date; q['price_usd'] = q.price_btc*q['index']
    q['market_call_fwd'] = q.price_usd/q.forward + np.where(q.cp.eq('P'),1-q.strike/q.forward,0)
    assert not q.instrument.duplicated().any()
    assert set(q.trade_id).isdisjoint({a[k] for a in anchors for k in ['call_id','put_id']})
    assert all(t < q.timestamp.min() for a in anchors for t in [a['call_timestamp'],a['put_timestamp']]) if len(q) else True
    audit['scored'] = len(q)
    return q, anchors, audit


def load_nets():
    nets=[]
    for seed in [17,43]:
        net=TorchRegularVariancePINN(factors=2,width=256,depth=5,tau_min=7/365,tau_max=2,x_half_width=.36)
        net.load_state_dict(torch.load(V4/f'artifacts/pinn_s{seed}/weights.pt',weights_only=True))
        net.eval();net.requires_grad_(False); nets.append(net)
    return nets


def neural(nets,p,x,t):
    coords=torch.tensor(np.column_stack([x,np.full(len(x),p[4]),np.full(len(x),p[9]),t]),dtype=torch.float64)
    structural=torch.tensor(np.array(p).reshape(2,5)[:,:4],dtype=torch.float64)
    with torch.no_grad():
        values=[np.concatenate([(net.price(z,structural)*torch.exp(-z[:,0])).numpy() for z in coords.split(512)]) for net in nets]
    return np.mean(values,axis=0)


def rmse(a):
    return float(np.sqrt(np.mean(np.asarray(a)**2)))


def run():
    protocol=verify(); assert not (OUT/'predictions.csv').exists(), 'Results already exist; never silently rerun'
    torch.set_num_threads(1)
    nets=load_nets(); panels=[]; audits=[]; anchor_rows=[]; probes=[]
    for row in protocol['dates']:
        date=row['date']; trades=json.loads(gzip.decompress((BTC/f'artifacts/raw/{date}.json.gz').read_bytes()))
        q,anchors,audit=prepare(trades,date)
        audits.append(audit); anchor_rows+=anchors
        if len(q):
            # Change ALL later prices and exchange IV/mark fields; model inputs and membership must stay identical.
            changed=[{**t,'price':t['price']*1.1,'iv':9999.,'mark_price':9999.} if dt.datetime.fromtimestamp(t['timestamp']/1000,dt.timezone.utc).hour==7 else dict(t) for t in trades]
            z,za,_=prepare(changed,date)
            cols=['instrument','trade_id','strike','tau','index','forward','x','cp']
            pd.testing.assert_frame_equal(q[cols],z[cols],check_exact=True)
            assert anchors==za
            # Verify actual neural predictions, not just input equality.
            p=protocol['scenarios'][0]['double_slow_first']
            np.testing.assert_array_equal(neural(nets,p,q.x.to_numpy(),q.tau.to_numpy()),neural(nets,p,z.x.to_numpy(),z.tau.to_numpy()))
            probes.append({'date':date,'target_perturbation_input_and_PINN_invariance':True,'quotes':len(q)})
            q['regime']=row['regime']; panels.append(q)
        print('PREPARED',date,'scored',len(q),flush=True)
    save(OUT/'cleaning_audit.json',audits);save(OUT/'leakage_probes.json',probes)
    pd.DataFrame(anchor_rows).to_csv(OUT/'forward_anchors.csv',index=False)
    assert panels,'No eligible data; do not weaken frozen rules'
    panel=pd.concat(panels,ignore_index=True);panel.to_csv(OUT/'scoring_quotes.csv',index=False)
    assert not panel.duplicated(['date','instrument']).any()
    x=panel.x.to_numpy();t=panel.tau.to_numpy();f=panel.forward.to_numpy()
    parity=np.where(panel.cp.eq('P'),1-np.exp(-x),0)
    miv=iv(panel.market_call_fwd.to_numpy(),x,t)
    records=[]; diagnostics=[]
    for scenario in protocol['scenarios']:
        p=scenario['double_slow_first']; reference_audit=[]
        dh=exact(p,x,t,audit=reference_audit); pred=neural(nets,p,x,t)
        sh=exact(scenario['single'],x,t,audit=reference_audit)
        bs=black(x,t,scenario['BS_vol'])
        diff=pred-dh; piv=iv(pred,x,t); div=iv(dh,x,t); valid=np.isfinite(piv)&np.isfinite(div)
        gates=protocol['fidelity_gates_inherited']
        fidelity={'scenario':scenario['id'],'quotes':len(panel),'RMSE_forward':rmse(diff),'P95_forward':float(np.quantile(abs(diff),.95)),
                  'max_forward':float(abs(diff).max()),'RMSE_USD':rmse(diff*f),'MAE_USD':float(np.mean(abs(diff*f))),
                  'IV_RMSE_vol_points':rmse((piv-div)[valid])*100 if valid.any() else None,'valid_IV_quotes':int(valid.sum()),
                  'reference_fallbacks':len(reference_audit)}
        fidelity['all_inherited_gates_pass']=bool(fidelity['RMSE_forward']<=gates['forward_price_RMSE_max'] and fidelity['P95_forward']<=gates['forward_price_P95_max'] and fidelity['max_forward']<=gates['forward_price_max_max'] and valid.any() and rmse((piv-div)[valid])<=gates['IV_RMSE_max'])
        diagnostics.append(fidelity)
        save(OUT/f"reference_{scenario['id']}.json",reference_audit)
        for name,call in [('DH_PINN',pred),('DH_numerical',dh),('SH_numerical',sh),('BS_fixed',bs)]:
            assert np.isfinite(call).all()
            r=panel[['date','regime','instrument','cp','tau','x','index','forward','price_btc','price_usd']].copy()
            r['scenario']=scenario['id'];r['model']=name;r['prediction_usd']=(call-parity)*f
            r['prediction_btc']=r.prediction_usd/r['index'];r['error_usd']=r.prediction_usd-r.price_usd
            r['error_fwd']=r.error_usd/r.forward;r['market_iv']=miv;r['model_iv']=iv(call,x,t)
            r['iv_error_vol_points']=(r.model_iv-r.market_iv)*100
            r['model_bound_violation']=(call<np.maximum(1-np.exp(-x),0)-1e-9)|(call>1+1e-9)
            records.append(r)
        print('SCORED',scenario['id'],json.dumps(fidelity),flush=True)
    predictions=pd.concat(records,ignore_index=True)
    predictions.to_csv(OUT/'predictions.csv',index=False);save(OUT/'fidelity.json',diagnostics)
    daily=[]
    for keys,g in predictions.groupby(['scenario','model','regime','date']):
        daily.append(dict(zip(['scenario','model','regime','date'],keys))|{'quotes':len(g),'RMSE_USD':rmse(g.error_usd),'RMSE_forward':rmse(g.error_fwd)})
    daily=pd.DataFrame(daily);daily.to_csv(OUT/'daily_metrics.csv',index=False)
    metrics=[]
    for regime in ['all','shock','calm']:
        group=predictions if regime=='all' else predictions[predictions.regime.eq(regime)]
        for (scenario,model),g in group.groupby(['scenario','model']):
            d=daily[(daily.scenario==scenario)&(daily.model==model)]
            if regime!='all':d=d[d.regime==regime]
            good=np.isfinite(g.iv_error_vol_points)
            metrics.append({'regime':regime,'scenario':scenario,'model':model,'quotes':len(g),'dates':g.date.nunique(),
                            'RMSE_USD':rmse(g.error_usd),'MAE_USD':float(g.error_usd.abs().mean()),'RMSE_forward':rmse(g.error_fwd),
                            'median_date_RMSE_forward':float(d.RMSE_forward.median()),'median_date_RMSE_USD':float(d.RMSE_USD.median()),
                            'IV_RMSE_vol_points':rmse(g.loc[good,'iv_error_vol_points']) if good.any() else None,
                            'IV_valid_quotes':int(good.sum()),'model_bound_violations':int(g.model_bound_violation.sum())})
    metrics=pd.DataFrame(metrics);metrics.to_csv(OUT/'metrics.csv',index=False)
    verify()
    save(OUT/'integrity.json',{'all_frozen_hashes_unchanged':True,'protocol_sha256':sha(OUT/'protocol.json'),
         'dates_requested':len(protocol['dates']),'dates_scored':int(panel.date.nunique()),'quotes_scored':len(panel),
         'target_perturbation_checks_passed':len(probes),'network_retrained':False,'parameters_fitted':False,
         'previously_exposed_dates':True,'quote_eligibility_identical_for_all_models':True})
    make_report(protocol, panel, metrics, diagnostics)
    print(metrics[metrics.regime.eq('all')].to_string(index=False),flush=True)


def make_report(protocol,panel,metrics,fidelity):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    models=['BS_fixed','SH_numerical','DH_numerical','DH_PINN']
    names=['Black–Scholes fixed','Single Heston numerical','Double Heston numerical','Saved C3 PINN']
    colors=['#7d8792','#d9903c','#4178a8','#247e71']
    scenarios=[s['id'] for s in protocol['scenarios']]
    fig,axes=plt.subplots(1,2,figsize=(13,5.2))
    for ax,regime in zip(axes,['shock','calm']):
        for i,(model,name,color) in enumerate(zip(models,names,colors)):
            g=metrics[(metrics.regime==regime)&(metrics.model==model)].set_index('scenario').reindex(scenarios)
            ax.bar(np.arange(3)+(i-1.5)*.19,g.median_date_RMSE_forward,.19,label=name,color=color)
        ax.set_xticks(range(3),['28% initial vol','45% initial vol','60% initial vol'])
        ax.set_title(regime.title()+' dates');ax.set_ylabel('Median-date forward-normalized price RMSE')
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].legend(fontsize=8,frameon=False)
    fig.suptitle('Bitcoin: three hardcoded scenarios, no parameter fitting or retraining')
    fig.text(.5,.015,'Retrospective • separate earlier forward anchors • identical scored quotes across models • lower is better',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.055,1,.94]);fig.savefig(OUT/'bitcoin_fixed_market_errors.png',dpi=220);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,4.5))
    bars=ax.bar([s['id'] for s in fidelity],[s['RMSE_USD'] for s in fidelity],color='#247e71')
    ax.bar_label(bars,fmt='%.3f',padding=3);ax.set_ylabel('PINN minus numerical DH: RMSE in USD')
    ax.set_title('Neural approximation error — not market prediction error');ax.set_ylim(bottom=0)
    fig.tight_layout();fig.savefig(OUT/'bitcoin_fixed_pinn_fidelity.png',dpi=220);plt.close(fig)
    lines=['# Fixed-parameter saved PINN on Bitcoin options','',
           '**Retrospective exploratory test. No retraining, parameter fitting, winner selection, or future-price forecasting.**','',
           f"Scored {len(panel):,} authentic archived trade quotes over {panel.date.nunique()} of {len(protocol['dates'])} requested dates. Date range: {panel.date.min()} to {panel.date.max()}. Maturities: {panel.tau.min()*365:.2f}–{panel.tau.max()*365:.2f} days.",'',
           '## Method','',protocol['forward_rule'], '', protocol['score_rule'], '',protocol['pricing_units'],'',
           'USD conversion follows [Deribit inverse-option documentation]('+protocol['official_convention_source']+'). Zero discounting and a stable first-hour forward/index ratio are explicit approximations.','',
           '## Parameters fixed before this run','',protocol['scenario_provenance'],'',protocol['comparators'],'',
           'Complete parameter values, source/weight hashes, dates and rules are in `protocol.json`. All three scenarios remain reported regardless of error.','',
           '## Market price error (all scored quotes)','',
           '| Fixed initial vol | Model | USD RMSE | USD MAE | Median-date normalized RMSE |','|---|---|---:|---:|---:|']
    for _,r in metrics[metrics.regime.eq('all')].iterrows():
        lines.append(f"| {r.scenario} | {r.model} | {r.RMSE_USD:.2f} | {r.MAE_USD:.2f} | {r.median_date_RMSE_forward:.6f} |")
    lines+=['','RMSE in dollars is per one-BTC underlying option premium, not position P&L. Daily medians give each date equal weight; pooled errors do not.','',
            '## PINN versus numerical Double Heston at identical parameters','',
            '| Scenario | USD RMSE | Forward-normalized RMSE | IV RMSE (vol points) | All inherited fidelity gates pass |','|---|---:|---:|---:|---|']
    for r in fidelity:
        lines.append(f"| {r['scenario']} | {r['RMSE_USD']:.4f} | {r['RMSE_forward']:.8f} | {r['IV_RMSE_vol_points']} | {r['all_inherited_gates_pass']} |")
    lines+=['','This is a separate check from matching actual Bitcoin prices. Domain inclusion alone does not guarantee local fidelity.','',
            '## Integrity and limitations','',
            '- Every archived raw file and both saved C3 checkpoints are SHA-256 checked before and after the run.',
            '- All second-hour targets and IV/mark fields were perturbed in memory: scored membership, forwards, network inputs and actual PINN predictions remained exactly unchanged on every scored date.',
            '- Anchors precede every scored trade and anchor strikes are excluded from scoring. No original leaked forward columns or previous fitted Bitcoin parameter values are used.',
            '- Numerical reference uses 96/128-node agreement and adaptive fallback. No model-price clipping; IV failures are counted without removing prices.',
            *['- '+c for c in protocol['caveats']], '',
            'The earlier BTC calibrated results are not comparable: this test changes parameter policy, units, forwards, scoring universe and time separation. A lower error here cannot prove universal model superiority.','',
            '## Figures','', '![Market errors](bitcoin_fixed_market_errors.png)','',
            'Interpretation: compare models within each fixed scenario and regime; every model sees the same quotes. Do not interpret the best displayed setting as independently validated.','',
            '![Neural fidelity](bitcoin_fixed_pinn_fidelity.png)','',
            'Interpretation: this isolates the network approximation from the economic parameter mismatch. Small values here do not imply small market errors.','',
            '## Reproduce','', '`python experiments/btc_fixed_pinn_v1/run.py self-test` checks preprocessing and units. The original evaluation is frozen in artifacts/protocol.json. Do not overwrite it; a revised protocol requires a new experiment directory.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))


def self_test():
    # Synthetic fixtures are test code only, NEVER market records or scored data.
    date='2024-01-01'; index=100.; forward=102.; expiration=dt.datetime(2024,4,1,8,tzinfo=dt.timezone.utc)
    trades=[]
    for hour in [6,7]:
        ts=dt.datetime(2024,1,1,hour,20,tzinfo=dt.timezone.utc)
        tau=(expiration-ts).total_seconds()/(365*86400)
        for strike in [75,80,90,100,110,120,130]:
            c=float(black(np.log(forward/strike),tau,.6))*forward
            for cp,usd in [('C',c),('P',c-forward+strike)]:
                trades.append({'trade_id':str(len(trades)), 'instrument_name':f'BTC-1APR24-{strike}-{cp}',
                               'timestamp':int(ts.timestamp()*1000),'price':usd/index,'index_price':index})
    q,anchors,_=prepare(trades,date)
    assert len(q)>0 and len(anchors)==3
    np.testing.assert_allclose(q.forward,forward,atol=1e-12)
    np.testing.assert_allclose(q.price_usd,q.price_btc*index,atol=1e-12)
    changed=[{**t,'price':t['price']*2,'iv':999} if dt.datetime.fromtimestamp(t['timestamp']/1000,dt.timezone.utc).hour==7 else t for t in trades]
    z,za,_=prepare(changed,date)
    assert anchors==za
    pd.testing.assert_frame_equal(q[['instrument','x','tau','forward']],z[['instrument','x','tau','forward']])
    assert (q.price_usd!=z.price_usd).all()
    q2,_,a2=prepare(trades+[trades[0]],date)
    pd.testing.assert_frame_equal(q,q2);assert a2['identical_duplicate_trades']==1
    try:
        prepare(trades+[{**trades[0],'price':999}],date)
    except AssertionError:
        pass
    else:
        raise AssertionError('Conflicting duplicate not rejected')
    assert len(scenario_bank())==3
    print('PASS: forward parity, BTC/USD units, temporal and strike separation, target invariance, duplicate handling, parameter bounds',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run','self-test']);args=parser.parse_args()
    {'freeze':freeze,'run':run,'self-test':self_test}[args.action]()
