#!/usr/bin/env python3
"""Replay one completed fit with corrupt held-out quotes and exact pricers blocked.

This is a bounded runtime isolation test of an actual trained checkpoint, not a
claim that all possible leakage or development-set overfitting has been ruled out.
"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.mentor_dh_pinn.assess_regular_pinn import sha256
from src.mentor_dh_pinn.affine_factor_calibration import fit_factor_pinn
from src.mentor_dh_pinn.conjugate_factor_pinn import build_factor_pinn


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--assessment',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--case',type=int,default=0)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((args.assessment/'manifest.json').read_text())
    assert manifest['status']=='complete' and manifest['frozen_hashes_rechecked']
    rows=json.loads((args.assessment/'cases.json').read_text())
    row=next(r for r in rows if r['case']==args.case)
    assert row['status']=='fitted'
    weights=Path(manifest['checkpoint'])
    assert sha256(weights)==manifest['checkpoint_sha256']
    # Require the same implementation that produced the archived fit.
    for name,digest in manifest['source_sha256'].items():
        assert sha256(ROOT/name)==digest, f'Assessment source changed: {name}'
    torch.set_num_threads(1)
    model=build_factor_pinn(manifest['training_config'])
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
    original={k:v.clone() for k,v in model.state_dict().items()}
    geometry=manifest['geometry'];holdout=np.asarray(geometry['holdout'],bool)
    x,tau,iv=map(np.asarray,(geometry['x'],geometry['tau'],row['observed_iv']))
    x=x.copy();tau=tau.copy();iv=iv.copy()
    x[holdout]=np.nan;tau[holdout]=-1.;iv[holdout]=np.inf
    seed=906777 if manifest['case_set']=='development4' else 907931+100+args.case
    blocked=['src.mentor_dh_pinn.affine_factor_reference.reference_coefficients',
             'src.mentor_dh_pinn.torch_pricer.price_call',
             'src.mentor_dh_pinn.torch_pricer.price_call_single',
             'src.mentor_dh_pinn.regular_pinn_data.price_call',
             'src.mentor_dh_pinn.regular_pinn_data.price_call_single',
             'scripts.mentor_dh_pinn.assess_regular_pinn._exact']
    with ExitStack() as stack:
        for name in blocked:
            stack.enter_context(patch(name,side_effect=AssertionError('Exact reference called during inverse fit')))
        replay=fit_factor_pinn(model,x,tau,iv,fit_mask=~holdout,starts=manifest['starts'],
                              max_nfev=manifest['max_nfev_per_start'],seed=seed)
    def timeless(value):
        if isinstance(value,dict):return {k:timeless(v) for k,v in value.items() if k!='seconds'}
        if isinstance(value,list):return [timeless(v) for v in value]
        return value
    assert timeless(replay)==timeless(row['fit']), 'Corrupted holdout changed the fit'
    assert all(torch.equal(original[k],v) for k,v in model.state_dict().items()), 'Neural weights changed'
    assert sha256(weights)==manifest['checkpoint_sha256']
    result={'status':'passed','assessment':str(args.assessment),'case':args.case,
            'checkpoint_sha256':manifest['checkpoint_sha256'],
            'assessment_input_sha256':{name:sha256(args.assessment/name) for name in ('manifest.json','cases.json')},
            'corrupted_holdout_quotes':int(holdout.sum()),'replayed_starts':manifest['starts'],
            'all_fit_and_start_fields_identical_except_seconds':True,'weights_unchanged':True,
            'blocked_reference_entry_points':blocked,'source_sha256':sha256(Path(__file__)),
            'limitation':'One actual checkpoint/case runtime check; not an unseen recovery test or universal leakage proof'}
    (args.out/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    (args.out/'replay.json').write_text(json.dumps(replay,indent=2,allow_nan=False))
    (args.out/'source_snapshot.py').write_text(Path(__file__).read_text())
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
