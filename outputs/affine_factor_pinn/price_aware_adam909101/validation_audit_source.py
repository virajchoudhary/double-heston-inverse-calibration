#!/usr/bin/env python3
"""Post-selection actual-IV audit of ALL reserved price-training validation quotes."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.mentor_dh_pinn.assess_regular_pinn import sha256
from scripts.mentor_dh_pinn.finetune_factor_prices import load_surfaces,surface_loss
from src.mentor_dh_pinn.affine_factor_pricing import neural_call_prices
from src.mentor_dh_pinn.conjugate_factor_pinn import build_factor_pinn
from src.mentor_dh_pinn.regular_pinn_data import black_call,invert_total_variance


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--checkpoint',type=Path,required=True)
    args=ap.parse_args();folder=args.checkpoint
    assert (folder/'complete.json').is_file(), 'Training must finish first'
    out=folder/'validation_audit.json';source=folder/'validation_audit_source.py'
    assert not out.exists() and not source.exists(), 'Preserve the existing audit'
    cfg=json.loads((folder/'config.json').read_text());selection=json.loads((folder/'selection.json').read_text())
    path=Path(cfg['price_surfaces']['path']);assert sha256(path)==cfg['price_surfaces']['sha256']
    torch.set_num_threads(1);weights=folder/'model.pt';digest=sha256(weights)
    model=build_factor_pinn(cfg);model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
    model.requires_grad_(False);data=load_surfaces(path,cfg['price_surfaces']['validation'])
    with np.load(path,allow_pickle=False) as archive:reference=archive['iv']
    errors=[];failures=[];inverse_errors=[]
    x,tau=data['x'].numpy(),data['tau'].numpy()
    with torch.no_grad():
        for i in range(data['training_count'],len(data['unit'])):
            price=neural_call_prices(model,data['physical'][i],data['x'],data['tau']).numpy()
            w=invert_total_variance(price,x);iv=np.sqrt(w/tau)
            residual=np.abs(black_call(x,w)-price)
            bad=~np.isfinite(iv)|~np.isfinite(residual)|(residual>1e-10)
            failures.extend({'surface_index':i,'quote_index':int(j)} for j in np.flatnonzero(bad))
            errors.extend(iv-reference[i]);inverse_errors.extend(residual[np.isfinite(residual)])
        score=float(surface_loss(model,data,range(data['training_count'],len(data['unit']))).sqrt())
    assert score==selection['validation_vega_price_rmse'], 'Selection score did not reproduce'
    assert sha256(weights)==digest and sha256(path)==cfg['price_surfaces']['sha256']
    result={'purpose':'Post-selection validation diagnostic; no checkpoint reselection or inverse calibration',
        'checkpoint_sha256':digest,'surface_corpus_sha256':sha256(path),'quotes':len(errors),
        'invalid_iv_quotes':len(failures),'invalid_locations':failures,
        'actual_iv_rmse':None if failures else float(np.sqrt(np.mean(np.asarray(errors)**2))),
        'maximum_finite_inverse_price_residual':float(max(inverse_errors)) if inverse_errors else None,
        'weighted_price_rmse_reproduced':score,'source_sha256':sha256(Path(__file__)),
        'limitation':'Same reserved selection set, not unseen evidence; undefined whole-set IV RMSE when any quote fails'}
    source.write_text(Path(__file__).read_text());out.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in result.items() if k!='invalid_locations'}),flush=True)


if __name__=='__main__':main()
