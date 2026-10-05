"""Peak-DVOL stress test of the unchanged fixed-parameter saved C3 PINN.

ponytail: reuse the prior frozen cleaner, evaluators and leakage probes through
an explicit in-process adapter. No prior experiment files are modified.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'artifacts'
PRIOR=ROOT/'experiments/btc_fixed_pinn_v1'
BTC=ROOT/'experiments/btc_multifactor_v1'
DVOL=BTC/'artifacts/dvol_btc_daily.json'
sys.path.insert(0,str(PRIOR))
import run as base
sys.path.insert(0,str(BTC))
from data import fetch as fetch_trades

sha,read,save=base.sha,base.read,base.save


def select_peaks(rows,n=10,gap=30):
    dates=[dt.date.fromisoformat(r['date']) for r in rows]
    assert len(set(dates))==len(dates)
    assert all(np.isfinite(r['dvol_close']) and r['dvol_close']>0 for r in rows)
    selected=[]
    for row in sorted(rows,key=lambda r:(-r['dvol_close'],r['date'])):
        day=dt.date.fromisoformat(row['date'])
        if all(abs((day-dt.date.fromisoformat(r['date'])).days)>=gap for r in selected):
            selected.append(row)
            if len(selected)==n:break
    assert len(selected)==n
    return selected


def freeze():
    assert not OUT.exists(),'Never overwrite an existing experiment'
    parent=base.verify()
    dvol=read(DVOL)
    # Ignore the potentially incomplete candle on the archive fetch date.
    complete=[r for r in dvol['rows'] if r['date']<dvol['fetched_utc'][:10]]
    peaks=select_peaks(complete)
    dates=[]
    for rank,row in enumerate(peaks,1):
        for offset in [1,2,3]:
            day=(dt.date.fromisoformat(row['date'])+dt.timedelta(days=offset)).isoformat()
            dates.append({'date':day,'regime':'shock','stage':'retrospective_peak_stress',
                          'peak_rank':rank,'peak_day':row['date'],'peak_DVOL':row['dvol_close'],'offset':offset,
                          'already_in_original_option_archive':(BTC/f'artifacts/raw/{day}.json.gz').exists()})
    assert len({r['date'] for r in dates})==30
    inherited={k:v for k,v in parent.items() if k not in ['dates','hashes','utc','date_policy']}
    # Exclude old raw market files from this protocol's input hashes, retain all
    # inherited model and source hashes. Parent verification still protects them.
    hashes={k:v for k,v in parent['hashes'].items() if '/artifacts/raw/' not in k}
    hashes.update({str(Path(__file__).relative_to(ROOT)):sha(__file__),str(DVOL.relative_to(ROOT)):sha(DVOL),
                   str((BTC/'data.py').relative_to(ROOT)):sha(BTC/'data.py'),
                   str((BTC/'engine.py').relative_to(ROOT)):sha(BTC/'engine.py')})
    protocol={**inherited,'utc':dt.datetime.now(dt.timezone.utc).isoformat(),
              'role':'RETROSPECTIVE_INDEPENDENT_DVOL_PEAK_SELECTION_NOT_UNTOUCHED_VALIDATION',
              'date_policy':'Rank complete archived daily DVOL closes descending; greedily retain the highest ten peaks at least 30 calendar days apart. Evaluate offsets +1,+2,+3. No replacements or model-error date selection.',
              'selection_measure':'DVOL annualized 30-day implied volatility, NOT realized volatility or observed future forecast accuracy.',
              'ranking_is_retrospective':True,'peak_ranking_uses_full_indicator_history':True,
              'indicator_available_before_scoring_day':True,
              'DVOL_source':dvol['source'],'DVOL_definition':'https://insights.deribit.com/exchange-updates/dvol-deribit-implied-volatility-index/',
              'DVOL_coverage':[complete[0]['date'],complete[-1]['date']],
              'peaks':peaks,'dates':dates,'hashes':hashes,
              'same_hardcoded_parameters_as_previous_test':True,'same_filters_and_units_as_previous_test':True,
              'primary_summary':'Median-episode forward-normalized RMSE. Within each episode average squared error equally over its scored dates, then take square root. Also report pooled, median-date, USD metrics and per-episode paired differences for all scenarios.',
              'minimum_coverage_for_confirmatory_claim':'No confirmatory claim regardless of coverage; stress test is descriptive. Missing episodes and dates must remain visible.',
              'domain_warning':'Saved C3 parameter family is restricted. The three initial-vol settings (28/45/60%) are not calibrated for 100%+ DVOL. DVOL is not identical to initial volatility, but this flags a large potential level mismatch.',
              'parent_protocol_sha256':sha(PRIOR/'artifacts/protocol.json')}
    OUT.mkdir();save(OUT/'protocol.json',protocol)
    (OUT/'protocol.sha256').write_text(sha(OUT/'protocol.json')+'\n')
    pd.DataFrame(dates).to_csv(OUT/'selected_dates.csv',index=False)
    print('FROZEN 10 episodes / 30 days, before scoring or fetching new option data',flush=True)
    print(pd.DataFrame(peaks).to_string(index=False),flush=True)


def verify(require_raw=False):
    assert sha(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip()
    p=read(OUT/'protocol.json')
    assert sha(PRIOR/'artifacts/protocol.json')==p['parent_protocol_sha256']
    for path,digest in p['hashes'].items():assert sha(ROOT/path)==digest,path
    if require_raw:
        assert sha(OUT/'downloads.json')==(OUT/'downloads.sha256').read_text().strip()
        for row in read(OUT/'downloads.json'):
            if row['status']=='available':assert sha(OUT/'raw'/f"{row['date']}.json.gz")==row['sha256']
    return p


def download():
    p=verify();assert not (OUT/'downloads.json').exists()
    (OUT/'raw').mkdir(exist_ok=True)
    old={r['date']:r['raw_sha256'] for r in read(BTC/'artifacts/data_audit.json')['dates'] if 'raw_sha256' in r}
    def one(row):
        day=row['date']; path=OUT/'raw'/f'{day}.json.gz';existing=BTC/'artifacts/raw'/path.name
        try:
            if existing.exists():
                assert sha(existing)==old[day]
                shutil.copy2(existing,path)
                result={**row,'status':'available','sha256':sha(path),'source':'verified existing official archive','bytes':path.stat().st_size}
            else:
                trades,digest=fetch_trades(day,[6,8],p['source_endpoint'],OUT/'raw')
                result={**row,'status':'available','sha256':digest,'source':'new official Deribit download','trades':len(trades),'bytes':path.stat().st_size}
        except Exception as e:
            result={**row,'status':'failed','error':repr(e)}
        print('DOWNLOAD',day,result['status'],result.get('source',result.get('error')),flush=True)
        return result
    records=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(one,r) for r in p['dates']]):records.append(future.result())
    save(OUT/'downloads.json',sorted(records,key=lambda r:r['date']))
    (OUT/'downloads.sha256').write_text(sha(OUT/'downloads.json')+'\n')
    verify(True)


def evaluate():
    protocol=verify(True)
    available={r['date'] for r in read(OUT/'downloads.json') if r['status']=='available'}
    # Explicit adapter of the prior evaluator, in this process only. The saved
    # protocol retains all requested dates; unavailable dates remain in audit.
    def evaluation_protocol():
        p=verify(True)
        return {**p,'dates':[r for r in p['dates'] if r['date'] in available]}
    base.OUT=OUT
    base.BTC=HERE
    base.verify=evaluation_protocol
    base.make_report=report
    base.run()
    verify(True)


def report(protocol,panel,metrics,fidelity):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=verify(True)
    dates=pd.DataFrame(p['dates'])
    daily=pd.read_csv(OUT/'daily_metrics.csv').merge(dates[['date','peak_rank','peak_day','peak_DVOL']],on='date',validate='many_to_one')
    episode=[]
    for keys,g in daily.groupby(['scenario','model','peak_rank','peak_day','peak_DVOL']):
        episode.append(dict(zip(['scenario','model','peak_rank','peak_day','peak_DVOL'],keys))|{
            'days':len(g),'quotes':int(g.quotes.sum()),'equal_date_RMSE_forward':base.rmse(g.RMSE_forward),
            'equal_date_RMSE_USD':base.rmse(g.RMSE_USD)})
    episode=pd.DataFrame(episode);episode.to_csv(OUT/'episode_metrics.csv',index=False)
    summaries=[]
    for scenario in [s['id'] for s in p['scenarios']]:
        e=episode[episode.scenario.eq(scenario)].pivot(index='peak_rank',columns='model',values='equal_date_RMSE_forward')
        summaries.append({'scenario':scenario,'episodes':len(e),
                          'DH_PINN_beats_SH_episodes':int((e.DH_PINN<e.SH_numerical).sum()),
                          'DH_PINN_beats_BS_episodes':int((e.DH_PINN<e.BS_fixed).sum()),
                          **{f'median_episode_RMSE_{m}':float(e[m].median()) for m in e.columns}})
    save(OUT/'episode_summary.json',summaries)
    audit=read(OUT/'cleaning_audit.json'); counts={r['date']:r['scored'] for r in audit}
    coverage=dates.copy();coverage['scored_quotes']=coverage.date.map(counts).fillna(0).astype(int)
    coverage.to_csv(OUT/'date_coverage.csv',index=False)
    n_episodes=int(coverage[coverage.scored_quotes>0].peak_rank.nunique())
    models=['BS_fixed','SH_numerical','DH_numerical','DH_PINN']
    names=['BS fixed','SH numerical','DH numerical','Saved C3 PINN'];colors=['#7d8792','#d9903c','#4178a8','#247e71']
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    ds=read(DVOL)['rows'];d=pd.DataFrame(ds);d['date']=pd.to_datetime(d.date)
    axes[0].plot(d.date,d.dvol_close,lw=.8,color='#5f6b76')
    peaks=pd.DataFrame(p['peaks']);axes[0].scatter(pd.to_datetime(peaks.date),peaks.dvol_close,color='#c66a27',s=35,zorder=3)
    axes[0].set(title='Selection uses DVOL, not model errors',ylabel='DVOL: annualized implied volatility (%)',xlabel='Date (UTC)')
    for i,(model,name,color) in enumerate(zip(models,names,colors)):
        values=[r[f'median_episode_RMSE_{model}'] for r in summaries]
        bars=axes[1].bar(np.arange(3)+(i-1.5)*.19,values,.19,label=name,color=color)
    axes[1].set_xticks(range(3),['28% initial vol','45% initial vol','60% initial vol'])
    axes[1].set(title='Unchanged hardcoded parameter scenarios',ylabel='Median-episode forward-normalized RMSE')
    axes[1].legend(frameon=False,fontsize=9)
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle(f'Bitcoin peak-volatility stress test: {len(panel)} quotes, {panel.date.nunique()} dates, {n_episodes}/10 selected episodes')
    fig.text(.5,.015,'Retrospective peak ranking • next-three-day scoring • earlier disjoint forward anchors • no retraining or parameter fitting',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.06,1,.94]);fig.savefig(OUT/'peak_volatility_results.png',dpi=220);plt.close(fig)
    # Every selected peak remains shown, including those with zero usable quotes.
    peaks['usable_dates']=[int(((coverage.peak_day==r['date'])&(coverage.scored_quotes>0)).sum()) for _,r in peaks.iterrows()]
    lines=['# Bitcoin peak-volatility stress test','',
           '**Descriptive retrospective stress test. All parameters, weights and cleaning rules unchanged. No best-setting selection.**','',
           f"Independent indicator coverage: {p['DVOL_coverage'][0]} through {p['DVOL_coverage'][1]}. {p['date_policy']}",'',
           'DVOL is 30-day annualized implied volatility, not realized returns volatility. Ranking the full historical indicator is hindsight event selection, not a deployable trading rule. The selected peak close itself precedes each scoring day.','',
           f"Coverage: **{len(panel)} quotes on {panel.date.nunique()}/30 requested dates, from {n_episodes}/10 episodes**. No missing date was replaced and no eligibility rule was relaxed.",'',
           '## Selected independent volatility peaks','', '| Peak day | DVOL close (%) | Usable next-three-day dates |','|---|---:|---:|']
    for _,r in peaks.iterrows():lines.append(f"| {r.date} | {r.dvol_close:.2f} | {r.usable_dates}/3 |")
    lines+=['','## Frozen-scenario market errors','', '| Scenario | Model | Pooled USD RMSE | Pooled normalized RMSE | Median-episode normalized RMSE |','|---|---|---:|---:|---:|']
    for _,r in metrics[metrics.regime.eq('all')].iterrows():
        es=next(s for s in summaries if s['scenario']==r.scenario)
        lines.append(f"| {r.scenario} | {r.model} | {r.RMSE_USD:.2f} | {r.RMSE_forward:.6f} | {es['median_episode_RMSE_'+r.model]:.6f} |")
    lines+=['','## Episode-level consistency','', '| Scenario | PINN beats SH | PINN beats BS |','|---|---:|---:|']
    for r in summaries:lines.append(f"| {r['scenario']} | {r['DH_PINN_beats_SH_episodes']}/{r['episodes']} | {r['DH_PINN_beats_BS_episodes']}/{r['episodes']} |")
    lines+=['','## Numerical fidelity, separate from market fit','', '| Scenario | PINN minus numerical DH USD RMSE | All inherited fidelity gates pass |','|---|---:|---|']
    for r in fidelity:lines.append(f"| {r['scenario']} | {r['RMSE_USD']:.4f} | {r['all_inherited_gates_pass']} |")
    lines+=['','## Limitations and honesty','',
            '- Peak dates were selected only from DVOL, before loading or scoring their new option trades. Some dates were already present in earlier experiments; this is not pristine confirmation.',
            '- The three hardcoded settings are unchanged, in-domain hypotheses, not Bitcoin-calibrated estimates. Initial volatility of 60% is far below many selected DVOL peaks, although the two quantities are not identical.',
            '- The saved PINN was trained on a restricted literature-shaped family. Testing a much higher-volatility family requires a separately specified training protocol, not silently extrapolating this network.',
            '- All data provenance, target-perturbation checks and checkpoint hashes are retained. Forward anchors are first-hour trades, disjoint from later scored strikes. No target IV enters forward estimation.',
            '- Same-day forward basis and zero USD discounting are approximations; prices are asynchronous last trades, not bid/ask midpoints.',
            '- A positive median difference is not proof of consistent superiority. Episode counts are descriptive; no post-hoc p-value claim is made.',
            '- This selection cannot cover the March 2020 crash because DVOL history begins in March 2021.',
            '', '![Peak stress results](peak_volatility_results.png)','',
            'Interpretation: orange markers identify the independently selected volatility peaks. On the right, lower bars indicate smaller market-price errors, aggregated with equal weight per episode. Similar DH numerical and PINN bars mean faithful model approximation, not necessarily good market fit.','',
            '## Sources and reproduction','',
            '[Deribit DVOL definition]('+p['DVOL_definition']+') · [Inverse-option pricing units]('+p['official_convention_source']+')','',
            '`run_high_vol.py self-test` checks selection independence and the inherited preprocessing invariants. Frozen protocol and dates: `protocol.json`; original downloaded trade hashes: `downloads.json`; raw files: `raw/`; all scores: `predictions.csv`, `metrics.csv`, `episode_metrics.csv`; exclusions: `date_coverage.csv`, `cleaning_audit.json`.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    save(OUT/'peak_integrity.json',{'requested_dates':30,'available_download_dates':sum(r['status']=='available' for r in read(OUT/'downloads.json')),
                                 'scored_dates':int(panel.date.nunique()),'requested_episodes':10,'scored_episodes':n_episodes,
                                 'source_and_checkpoint_hashes_verified':True,'scenarios_unchanged':p['scenarios']==base.scenario_bank(),
                                 'no_missing_day_replacement':True,'retrospective_not_confirmatory':True})


def self_test():
    rows=[{'date':(dt.date(2020,1,1)+dt.timedelta(days=i)).isoformat(),'dvol_close':float(1000-i)} for i in range(100)]
    a=select_peaks(rows,n=3,gap=30)
    assert [r['date'] for r in a]==['2020-01-01','2020-01-31','2020-03-01']
    assert a==select_peaks(list(reversed(rows)),n=3,gap=30)
    assert a==select_peaks([{**r,'model_error':9999-i} for i,r in enumerate(rows)],n=3,gap=30) or [r['date'] for r in a]==[r['date'] for r in select_peaks([{**r,'model_error':9999-i} for i,r in enumerate(rows)],n=3,gap=30)]
    base.self_test()
    print('PASS: deterministic peak selection, episode separation, input-order independence, no model-error date ranking')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','download','evaluate','self-test']);args=parser.parse_args()
    {'freeze':freeze,'download':download,'evaluate':evaluate,'self-test':self_test}[args.action]()
