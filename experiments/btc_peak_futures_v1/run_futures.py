"""Coverage amendment: same peak dates/settings, independent as-of futures inputs.

This is separately labelled after sparse anchor results, not a silent repair.
ponytail: reuse frozen pricing/scoring and its actual-network leakage probes.
"""
import argparse
import calendar
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
from functools import lru_cache
import gzip
import json
from pathlib import Path
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'artifacts'
PEAK=ROOT/'experiments/btc_peak_volatility_v1'
sys.path.insert(0,str(PEAK))
import run_high_vol as peak
base=peak.base
sha,read,save=base.sha,base.read,base.save
UTC=dt.timezone.utc


def quarters(day):
    day=dt.date.fromisoformat(day);rows=[]
    for year in [day.year,day.year+1]:
        for month in [3,6,9,12]:
            last=dt.date(year,month,calendar.monthrange(year,month)[1])
            expiry=last-dt.timedelta(days=(last.weekday()-4)%7)
            if expiry>=day:
                rows.append({'instrument':'BTC-'+expiry.strftime('%d%b%y').lstrip('0').upper(),
                             'expiry':expiry.isoformat()})
    return rows[:3]


def freeze():
    p=peak.verify(True);assert not OUT.exists()
    hashes=dict(p['hashes']);hashes[str(Path(__file__).relative_to(ROOT))]=sha(__file__)
    for r in read(PEAK/'artifacts/downloads.json'):
        assert r['status']=='available'
        path=PEAK/f"artifacts/raw/{r['date']}.json.gz";hashes[str(path.relative_to(ROOT))]=sha(path)
    protocol={**p,'utc':dt.datetime.now(UTC).isoformat(),'role':'RETROSPECTIVE_COVERAGE_AMENDMENT_INDEPENDENT_FUTURES',
              'amendment_reason':'Strict option-anchor stress test retained 27 quotes on two dates. Replace only forward input construction/anchor reservation using independent as-of futures, preserving all 30 selected dates, weights and parameter scenarios. Prior results have been seen; no confirmatory claim.',
              'parent_peak_protocol_sha256':sha(PEAK/'artifacts/protocol.json'),
              'forward_rule':'Download the nearest three quarterly inverse BTC futures (last Friday each quarter), 06:00-08:00 UTC. For each scored option use last strictly earlier trade for each future, maximum age 15 minutes. Interpolate log(futures price / futures index) versus remaining expiry time, including (0,0). Beyond last maturity keep its annualized log carry constant. Multiply resulting ratio by the option trade index. No option premium or IV is used.',
              'score_rule':'Same second-hour latest OTM quote/instrument and 7-730 day / |x|<=.36 filters, no price/IV/model-error exclusions. No anchor strikes need reservation because inputs now come from a different futures market.',
              'futures_requests':[{**r,**f} for r in p['dates'] for f in quarters(r['date'])],
              'futures_endpoint':'https://history.deribit.com/api/v2/public/get_last_trades_by_instrument_and_time',
              'max_future_age_seconds':900,'futures_source':'https://support.deribit.com/hc/en-us/articles/31424938981533-Inverse-Futures',
              'coverage_method_changed_after_prior_scores':True,'parameters_changed':False,'hashes':hashes}
    OUT.mkdir();(OUT/'futures_raw').mkdir();save(OUT/'protocol.json',protocol)
    (OUT/'protocol.sha256').write_text(sha(OUT/'protocol.json')+'\n')
    print('FROZEN separate independent-futures amendment: same 30 days / ten peaks / three scenarios')


def verify(require_futures=False):
    assert sha(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip()
    p=read(OUT/'protocol.json')
    assert sha(PEAK/'artifacts/protocol.json')==p['parent_peak_protocol_sha256']
    for path,digest in p['hashes'].items():assert sha(ROOT/path)==digest,path
    if require_futures:
        assert sha(OUT/'futures_downloads.json')==(OUT/'futures_downloads.sha256').read_text().strip()
        for r in read(OUT/'futures_downloads.json'):
            if r['status']=='available':assert sha(OUT/'futures_raw'/r['file'])==r['sha256']
    return p


def download():
    p=verify();assert not (OUT/'futures_downloads.json').exists()
    def one(row):
        date=row['date'];inst=row['instrument'];path=OUT/'futures_raw'/f'{date}_{inst}.json.gz'
        start=int(dt.datetime.fromisoformat(date).replace(hour=6,tzinfo=UTC).timestamp()*1000)
        end=start+7200000-1;trades=[];seen={};pages=0
        try:
            while True:
                url=f"{p['futures_endpoint']}?instrument_name={inst}&start_timestamp={start}&end_timestamp={end}&count=10000&sorting=asc&include_old=true"
                for attempt in range(3):
                    try:
                        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=25) as r:payload=json.load(r)
                        if 'error' in payload:raise ValueError(payload['error'])
                        result=payload['result'];break
                    except Exception:
                        if attempt==2:raise
                        time.sleep(1+attempt)
                pages+=1;new=[]
                for t in result['trades']:
                    key=str(t['trade_id'])
                    if key in seen:assert seen[key]==t,'Conflicting raw duplicate'
                    else:seen[key]=t;new.append(t)
                trades+=new
                if not result['has_more']:break
                assert new,'Pagination stalled; no silent timestamp skipping'
                assert pages<200,'Unexpectedly many pages; do not return incomplete data'
                start=result['trades'][-1]['timestamp']
            path.write_bytes(gzip.compress(json.dumps(trades).encode()))
            out={**row,'status':'available','file':path.name,'sha256':sha(path),'trades':len(trades),'pages':pages}
        except Exception as e:out={**row,'status':'failed','error':repr(e),'pages_completed':pages}
        print('FUTURES',date,inst,out['status'],out.get('trades',out.get('error')),flush=True)
        return out
    rows=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(one,r) for r in p['futures_requests']]):rows.append(future.result())
    save(OUT/'futures_downloads.json',sorted(rows,key=lambda r:(r['date'],r['instrument'])))
    (OUT/'futures_downloads.sha256').write_text(sha(OUT/'futures_downloads.json')+'\n');verify(True)


@lru_cache(maxsize=32)
def future_arrays(date):
    result=[]
    for row in read(OUT/'futures_downloads.json'):
        if row['date']!=date or row['status']!='available':continue
        trades=json.loads(gzip.decompress((OUT/'futures_raw'/row['file']).read_bytes()))
        expiry=int(dt.datetime.fromisoformat(row['expiry']).replace(hour=8,tzinfo=UTC).timestamp()*1000)
        good=[]
        for t in trades:
            assert t['instrument_name']==row['instrument']
            if np.isfinite([t['timestamp'],t['price'],t['index_price']]).all() and t['price']>0 and t['index_price']>0:
                good.append((t['timestamp'],t['price']/t['index_price'],t['trade_seq']))
        good=sorted(good,key=lambda t:(t[0],t[2]))
        if good:result.append((row['instrument'],expiry,np.array(good,dtype=float)))
    return result


def asof_forward(timestamp,tau,index,arrays):
    knots=[(0.,0.)];used=[]
    for instrument,expiry,a in arrays:
        pos=np.searchsorted(a[:,0],timestamp,side='left')-1
        if pos<0:continue
        ts,ratio,seq=a[pos];age=(timestamp-ts)/1000
        maturity=(expiry-timestamp)/(365*86400000)
        if age>900 or maturity<=0:continue
        assert 0<age<=900
        knots.append((maturity,float(np.log(ratio))));used.append({'instrument':instrument,'trade_seq':int(seq),'age_seconds':age})
    if not used:return np.nan,[],False
    times,logbasis=np.array(sorted(knots)).T
    extrapolated=bool(tau>times[-1])
    value=logbasis[-1]*tau/times[-1] if extrapolated else np.interp(tau,times,logbasis)
    return float(index*np.exp(value)),used,extrapolated


def prepare(trades,date):
    a,audit=base.parse(trades,date);audit['date']=date
    if a.empty:return a,[],{**audit,'scored':0}
    later=a[a.hour.eq(7)].sort_values(['timestamp','trade_id']).groupby('instrument').tail(1).copy()
    audit['later_instruments']=len(later);arrays=future_arrays(date)
    vals=[asof_forward(r.timestamp,r.tau,r.index,arrays) for r in later.itertuples(index=False)]
    later['forward']=[v[0] for v in vals];later['future_sources']=[json.dumps(v[1]) for v in vals]
    later['carry_extrapolated']=[v[2] for v in vals];later['reason']=''
    def reject(mask,reason):later.loc[mask&later.reason.eq(''),'reason']=reason
    reject(~np.isfinite(later.forward),'no_fresh_strictly_earlier_future')
    reject(~later.tau.between(7/365,2),'outside_maturity_training_domain')
    later['x']=np.log(later.forward/later.strike)
    reject(later.x.abs()>.36,'outside_moneyness_training_domain')
    reject(~np.where(later.cp.eq('C'),later.strike>=later.forward,later.strike<later.forward),'not_OTM')
    audit['exclusions']={str(k):int(v) for k,v in later[later.reason.ne('')].reason.value_counts().items()}
    q=later[later.reason.eq('')].copy().sort_values('instrument').reset_index(drop=True)
    q['date']=date;q['price_usd']=q.price_btc*q['index']
    q['market_call_fwd']=q.price_usd/q.forward+np.where(q.cp.eq('P'),1-q.strike/q.forward,0)
    assert not q.instrument.duplicated().any()
    audit['scored']=len(q);audit['carry_extrapolated_scored']=int(q.carry_extrapolated.sum())
    return q,[],audit


def evaluate():
    verify(True)
    base.OUT=OUT;base.BTC=PEAK;base.prepare=prepare;base.verify=lambda:verify(True);base.make_report=report
    base.run();verify(True)


def report(protocol,panel,metrics,fidelity):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dates=pd.DataFrame(protocol['dates'])
    daily=pd.read_csv(OUT/'daily_metrics.csv').merge(dates[['date','peak_rank','peak_day','peak_DVOL']],on='date',validate='many_to_one')
    groups=['scenario','model','peak_rank','peak_day','peak_DVOL'];episodes=[]
    for keys,g in daily.groupby(groups):
        episodes.append(dict(zip(groups,keys))|{'days':len(g),'quotes':int(g.quotes.sum()),'RMSE_forward':base.rmse(g.RMSE_forward),'RMSE_USD':base.rmse(g.RMSE_USD)})
    episodes=pd.DataFrame(episodes);episodes.to_csv(OUT/'episode_metrics.csv',index=False)
    summary=[]
    for s in protocol['scenarios']:
        e=episodes[episodes.scenario.eq(s['id'])].pivot(index='peak_rank',columns='model',values='RMSE_forward')
        summary.append({'scenario':s['id'],'episodes':len(e),'PINN_beats_SH':int((e.DH_PINN<e.SH_numerical).sum()),'PINN_beats_BS':int((e.DH_PINN<e.BS_fixed).sum()),
                        **{m:float(e[m].median()) for m in e.columns}})
    save(OUT/'episode_summary.json',summary)
    coverage=dates.copy();coverage['scored_quotes']=coverage.date.map(panel.groupby('date').size()).fillna(0).astype(int)
    coverage.to_csv(OUT/'date_coverage.csv',index=False)
    models=['BS_fixed','SH_numerical','DH_numerical','DH_PINN'];names=['BS fixed','SH numerical','DH numerical','Saved C3 PINN']
    colors=['#7d8792','#d9903c','#4178a8','#247e71']
    fig,ax=plt.subplots(figsize=(10,5))
    for i,(model,name,color) in enumerate(zip(models,names,colors)):
        ax.bar(np.arange(3)+(i-1.5)*.19,[s[model] for s in summary],.19,label=name,color=color)
    ax.set_xticks(range(3),['28% initial vol','45% initial vol','60% initial vol'])
    ax.set_ylabel('Median-episode forward-normalized price RMSE')
    ax.set_title(f'Bitcoin volatility peaks: {len(panel)} quotes, {panel.date.nunique()} days, {episodes.peak_rank.nunique()}/10 episodes')
    ax.legend(frameon=False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.text(.5,.015,'Independent earlier futures inputs • unchanged hardcoded parameters • retrospective coverage amendment • lower is better',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(OUT/'peak_futures_market_errors.png',dpi=220);plt.close(fig)
    lines=['# Bitcoin peak-volatility test: independent futures inputs','',
           '**Retrospective coverage amendment, not independent confirmation. Same dates and hardcoded parameters as the strict-anchor peak test.**','',
           protocol['amendment_reason'],'',protocol['date_policy'],'',
           f"Scored **{len(panel):,} authentic option trades on {panel.date.nunique()}/30 days, spanning {episodes.peak_rank.nunique()}/10 selected episodes**. Maturities: {panel.tau.min()*365:.2f}–{panel.tau.max()*365:.2f} days.",'',
           '## Forward inputs and units','',protocol['forward_rule'],'',
           'Observed USD option premium = reported BTC premium × contemporaneous index. Model prices use the independent futures-based forward and zero USD discounting. Log-basis interpolation is an estimated input, not a fabricated observed quote. Every supporting futures trade and its age is recorded in `scoring_quotes.csv`.','',
           f"Carry extrapolation was required for {int(panel.carry_extrapolated.sum())}/{len(panel)} scored quotes. Asynchronous trading, stale basis up to 15 minutes and zero-discounting assumptions remain limitations.",'',
           '## Aggregate market error','', '| Fixed scenario | Model | Pooled USD RMSE | Pooled normalized RMSE | Median-episode normalized RMSE |','|---|---|---:|---:|---:|']
    for _,r in metrics[metrics.regime.eq('all')].iterrows():
        es=next(s for s in summary if s['scenario']==r.scenario)
        lines.append(f"| {r.scenario} | {r.model} | {r.RMSE_USD:.2f} | {r.RMSE_forward:.6f} | {es[r.model]:.6f} |")
    lines+=['','## Episode consistency','', '| Scenario | PINN beats SH | PINN beats BS |','|---|---:|---:|']
    for s in summary:lines.append(f"| {s['scenario']} | {s['PINN_beats_SH']}/{s['episodes']} | {s['PINN_beats_BS']}/{s['episodes']} |")
    lines+=['','## Neural fidelity, not market error','', '| Scenario | PINN minus numerical DH USD RMSE | All inherited fidelity gates pass |','|---|---:|---|']
    for f in fidelity:lines.append(f"| {f['scenario']} | {f['RMSE_USD']:.4f} | {f['all_inherited_gates_pass']} |")
    lines+=['','## Interpretation and limitations','',
            '- High volatility does not guarantee that Double Heston beats simpler models. Compare all retained episodes and all three fixed settings; do not select favorable dates.',
            '- The settings are fixed in-domain hypotheses, not Bitcoin market calibrations. The peak-DVOL range is 92.31–156.20%, much higher than the 28/45/60% initial-state scenarios. DVOL and initial volatility are different quantities; the mismatch is a warning, not an identity.',
            '- The original saved PINN remains unchanged. Parameter recovery, optimal calibration, future-price forecasting and universal market superiority are not established.',
            '- Existing historical results had been seen before this coverage amendment. Parameter values, peak dates and scoring rules were frozen before the new futures-input scores; no settings were changed after those scores.',
            '- The actual-network target-perturbation probes check that changing scored option targets cannot change their inputs or predictions. Additional tests check future-timestamp exclusion and 15-minute freshness. This does not prove absence of every bias.',
            '- No synchronized bid/ask prices: no executable trading or spread-level accuracy claim.',
            '- Ranking historical DVOL is hindsight stress selection, not a causal signal backtest. Peak closes precede scoring dates; whole-period ranks use the full history.',
            '', '![Stress-test market errors](peak_futures_market_errors.png)','',
            'Interpretation: bars summarize market error equally by episode. Similar numerical-DH and PINN bars indicate that the neural calculator is reproducing Double Heston; they do not establish a market edge.','',
            '## Sources','',
            '[Deribit DVOL definition]('+protocol['DVOL_definition']+') · [Deribit inverse futures]('+protocol['futures_source']+') · [Inverse option units]('+protocol['official_convention_source']+')','',
            'Full values and hashes: `protocol.json`. Futures provenance: `futures_downloads.json`, `futures_raw/`. Scores: `predictions.csv`, `metrics.csv`, `daily_metrics.csv`, `episode_metrics.csv`. Coverage and targeted invariance checks: `cleaning_audit.json`, `date_coverage.csv`, `leakage_probes.json`.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    save(OUT/'futures_integrity.json',{'all_inputs_strictly_before_option_trade':True,'max_age_seconds':900,'carry_extrapolated_quotes':int(panel.carry_extrapolated.sum()),
                                    'scored_quotes':len(panel),'scored_dates':int(panel.date.nunique()),'scored_episodes':int(episodes.peak_rank.nunique()),
                                    'weights_and_sources_unchanged':True,'parameters_unchanged':protocol['scenarios']==base.scenario_bank()})


def self_test():
    assert quarters('2021-05-24')[0]['instrument']=='BTC-25JUN21'
    ts=1000000000;tau=.1;exp=ts+int(.2*365*86400000)
    arrays=[('BTC-test',exp,np.array([[ts-1000,1.02,1],[ts+1000,9,2]],dtype=float))]
    f,used,extra=asof_forward(ts,tau,100,arrays)
    np.testing.assert_allclose(f,100*np.sqrt(1.02));assert len(used)==1 and not extra
    arrays[0][2][1,1]=1000
    assert f==asof_forward(ts,tau,100,arrays)[0]
    assert np.isnan(asof_forward(ts+902000,tau,100,arrays)[0])
    assert np.isnan(asof_forward(ts-1000,tau,100,arrays)[0])
    peak.self_test()
    print('PASS: quarterly dates, log-basis interpolation, strict as-of timing, future-data invariance, stale rejection')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','download','evaluate','self-test']);args=parser.parse_args()
    {'freeze':freeze,'download':download,'evaluate':evaluate,'self-test':self_test}[args.action]()
