"""Reproduce existing BTC scores read-only and expose preprocessing dependency."""
import gzip
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
BTC=ROOT/'experiments/btc_multifactor_v1'
sys.path.insert(0,str(BTC))
import amend01
from data import clean


def main():
    out=Path(__file__).resolve().parent/'btc_audit';out.mkdir(exist_ok=False)
    cfg=amend01.check()  # checks both original and BS-amendment frozen code hashes
    rows=json.loads((BTC/'artifacts/data_audit.json').read_text())['dates']
    for row in rows:
        path=BTC/'artifacts/raw'/f"{row['date']}.json.gz"
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['raw_sha256'],path
    stored=pd.read_csv(BTC/'artifacts/amend01/test_scores.csv')
    actual=amend01.scores('test')
    keys=['date','design','model'];stored=stored.set_index(keys).sort_index();actual=actual.set_index(keys).sort_index()
    assert stored.index.equals(actual.index) and not stored.index.duplicated().any()
    metrics=['IV_RMSE_vol_points','price_RMSE_USD','price_MAE_USD','fwd_price_RMSE']
    np.testing.assert_allclose(stored[metrics],actual[metrics],rtol=1e-9,atol=1e-9,equal_nan=True)
    s=actual.reset_index();s=s[s.model.isin(['BS_EXPIRY','SH','DH'])]
    assert s.groupby(['date','design'])['quotes'].nunique().eq(1).all()

    # Dependency probe, NOT new market data: perturb held-out targets in memory
    # and check whether preprocessing changes their purported input forwards.
    date=sorted(s.date.unique())[0]
    trades=json.loads(gzip.decompress((BTC/'artifacts/raw'/f'{date}.json.gz').read_bytes()))
    q,_=clean(trades,date,cfg)
    expiry=sorted(q[q.heldout_B].expiry.unique())[0]
    instruments=set(q[q.expiry.eq(expiry)].instrument)
    changed=[{**t,'price':t['price']*1.05} if t['instrument_name'] in instruments else dict(t) for t in trades]
    q2,_=clean(changed,date,cfg)
    pair=q[['instrument','forward']].merge(q2[['instrument','forward']],on='instrument',suffixes=('_before','_after'))
    pair=pair[pair.instrument.isin(instruments)]
    shift=float(np.max(abs(pair.forward_after/pair.forward_before-1)))
    assert shift>0, 'Probe did not expose the inspected dependence'
    result={'source_and_amendment_hashes_verified':True,'raw_hashes_verified':len(rows),
            'score_rows_reproduced':len(actual),'models_displayed':['BS_EXPIRY','SH','DH'],
            'preprocessing_target_dependency':True,'probe_date':date,'probe_expiry':expiry,
            'probe_change':'Held-out expiry trade prices +5%, fixed exchange IV; memory only',
            'maximum_relative_forward_change':shift,
            'interpretation':'Numerically reproduced historical results, not certified leakage-free; not PINN results or future-date forecasts.'}
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    cells=[('shock','B'),('shock','A'),('calm','B'),('calm','A')]
    models=['BS_EXPIRY','SH','DH'];names=['Black–Scholes per expiry','Single Heston','Double Heston']
    fig,ax=plt.subplots(figsize=(11,5.5))
    records=[]
    for i,(model,name,color) in enumerate(zip(models,names,['#6c757d','#d9822b','#1f5fa8'])):
        vals=[]
        for regime,design in cells:
            v=s[s.regime.eq(regime)&s.design.eq(design)&s.model.eq(model)]
            vals.append(float(v.IV_RMSE_vol_points.median()))
            records.append({'regime':regime,'design':design,'model':model,'dates':len(v),'median_IV_RMSE_vol_points':vals[-1]})
        bars=ax.bar(np.arange(4)+(i-1)*.25,vals,.25,label=name,color=color)
        ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
    ax.set_xticks(range(4),['Shock\nheld-out expiries','Shock\nheld-out strikes','Calm\nheld-out expiries','Calm\nheld-out strikes'])
    ax.set_ylabel('Median held-out IV RMSE (volatility percentage points)')
    ax.set_title('Existing Bitcoin options comparison — numerical calibration, not PINN')
    ax.set_ylim(bottom=0,top=10);ax.legend(frameon=False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.text(.5,.025,'Caution: forwards depend on option targets before splitting. Ranking is reproduced, not certified leakage-free.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.07,1,1]);fig.savefig(out/'bitcoin_error_comparison.png',dpi=300);plt.close(fig)
    pd.DataFrame(records).to_csv(out/'displayed_metrics.csv',index=False)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
