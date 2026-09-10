#!/usr/bin/env python3
"""Recompute completed factor-PINN price scores and display expiry-level errors.

Post-fit only. Exact prices are used for independent repricing here, never for
optimization. All cases and full-precision prices are retained in the output.
"""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.mentor_dh_pinn.assess_regular_pinn import _exact,_score_price_iv,sha256
from scripts.mentor_dh_pinn.report_affine_factor_pinn import audit_assessment
from src.mentor_dh_pinn.affine_factor_pricing import neural_call_prices
from src.mentor_dh_pinn.conjugate_factor_pinn import build_factor_pinn
from src.mentor_dh_pinn.regular_pinn_data import decode_unit


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--assessment',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    manifest,rows,_=audit_assessment(args.assessment)
    for name,digest in manifest['source_sha256'].items():
        assert sha256(ROOT/name)==digest, f'Assessment source changed: {name}'
    torch.set_num_threads(1)
    weights=Path(manifest['checkpoint']);model=build_factor_pinn(manifest['training_config'])
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True));model.requires_grad_(False)
    geometry=manifest['geometry'];x=np.asarray(geometry['x']);tau=np.asarray(geometry['tau'])
    holdout=np.asarray(geometry['holdout'],bool);expiries=np.unique(tau)
    ratios=np.full((2,len(rows),len(expiries)),np.nan);saved=[]
    finite_list=lambda a:[float(v) if np.isfinite(v) else None for v in a]
    for i,row in enumerate(rows):
        record={'case':row['case'],'status':row['status']};saved.append(record)
        if row['status']!='fitted':continue
        np.testing.assert_array_equal(decode_unit(np.asarray(row['true_unit']),2),row['true_physical'])
        np.testing.assert_array_equal(decode_unit(np.asarray(row['fit']['unit']),2),row['fit']['physical'])
        reference=_exact(x,tau,np.asarray(row['true_unit']),2)
        np.testing.assert_array_equal(reference,row['reference_price'])
        with torch.no_grad():
            neural=neural_call_prices(model,torch.tensor(row['fit']['physical'],dtype=torch.float64),
                                      torch.tensor(x,dtype=torch.float64),torch.tensor(tau,dtype=torch.float64)).numpy()
        exact=_exact(x,tau,np.asarray(row['fit']['unit']),2)
        record['reference_price_spot']=finite_list(reference*np.exp(-x))
        for j,(label,prices) in enumerate((('neural',neural),('exact_reprice',exact))):
            score,_=_score_price_iv(prices,reference,np.asarray(row['observed_iv']),x,tau,holdout)
            assert score==row[label], f'Recomputed price/IV score changed: case {row["case"]}, {label}'
            record[label]={'price_spot':finite_list(prices*np.exp(-x)),'scores':score}
            for k,expiry in enumerate(expiries):
                mask=holdout&(tau==expiry)
                ratios[j,i,k]=np.sqrt(np.mean(((prices-reference)*np.exp(-x))[mask]**2))/1e-5
            record[label]['expiry_heldout_price_rmse_gate_units']=finite_list(ratios[j,i])
    assert sha256(weights)==manifest['checkpoint_sha256']
    fig,axes=plt.subplots(1,2,figsize=(12,7),layout='constrained')
    for j,(ax,title) in enumerate(zip(axes,('Learned PINN prices at fitted parameters','Exact repricing of the same fitted parameters'))):
        grid=ratios[j]
        im=ax.imshow(np.ma.masked_invalid(np.ma.masked_less_equal(grid,0)),
                     norm=LogNorm(vmin=.01,vmax=100),cmap='RdYlGn_r',aspect='auto')
        ax.set_xticks(range(len(expiries)),[f'{v*365:.0f}' for v in expiries])
        ax.set_yticks(range(len(rows)),[r['case'] for r in rows])
        ax.set(xlabel='Synthetic expiry (days)',ylabel='Development case',title=title)
        for i in range(len(rows)):
            for k in range(len(expiries)):
                value=grid[i,k]
                label=(f'{value:.2g}'+(' !' if value>1 else '')) if np.isfinite(value) else 'failed'
                ax.text(k,i,label,ha='center',va='center',fontsize=8)
    fig.colorbar(im,ax=axes,label='Held-out price RMSE / 0.00001 of spot (1 = threshold)')
    fig.suptitle('Expiry-level price errors: these are not parameter-recovery scores')
    fig.savefig(args.out/'expiry_price_errors.png',dpi=160);plt.close(fig)
    output={'status':'passed','purpose':'Post-fit artifact repricing audit; no inverse optimization',
            'assessment':str(args.assessment),'checkpoint_sha256':manifest['checkpoint_sha256'],
            'assessment_input_sha256':{name:sha256(args.assessment/name) for name in ('manifest.json','cases.json','summary.json')},
            'all_recorded_price_iv_scores_reproduced':True,'geometry':geometry,
            'expiries_years':expiries.tolist(),'cases':saved,'source_sha256':sha256(Path(__file__))}
    (args.out/'audit.json').write_text(json.dumps(output,indent=2,allow_nan=False))
    (args.out/'source_snapshot.py').write_text(Path(__file__).read_text())
    (args.out/'REPORT.md').write_text(
        '# Post-fit expiry-level price audit\n\n'
        '![Expiry-level price errors](expiry_price_errors.png)\n\n'
        'Interpretation: each cell is held-out price RMSE divided by 0.00001 of spot. '
        'Values at or below 1 meet that price threshold for the expiry; above 1 fail. '
        'An exclamation mark flags failure using the full-precision value, even when a label rounds to 1. '
        'The left panel uses learned prices; the right independently prices the SAME fitted parameters '
        'with the canonical Double Heston reference. Neither panel establishes correct parameter recovery. '
        'These per-expiry diagnostics do not replace or change the predeclared aggregate price gate.\n\n'
        'Colors saturate below 0.01 and above 100; cell labels and JSON values are not clipped. '
        'Zeros are labelled 0, failed cases remain labelled failed. '
        'These are clean synthetic, previously exposed development cases, not NSE observations or unseen evidence.\n\n'
        'The audit re-evaluated every fitted case without optimization and exactly reproduced its archived '
        'price/IV metrics. It also checked physical parameter decoding, reference quotes, source and checkpoint hashes. '
        'This verifies artifact consistency, not a second independent implementation of all pricing mathematics.\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ('cases','geometry')}),flush=True)


if __name__=='__main__':main()
