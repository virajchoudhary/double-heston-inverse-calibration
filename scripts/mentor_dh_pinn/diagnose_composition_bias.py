"""Training-only local parameter-bias proxy; no calibration or checkpoint selection."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.mentor_dh_pinn.assess_regular_pinn import load_checkpoint,sha256
from scripts.mentor_dh_pinn.compare_composition_pinn import neural_only
from src.mentor_dh_pinn.regular_pinn_data import coordinates


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',type=Path,action='append',required=True)
    ap.add_argument('--surfaces',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--count',type=int,default=32)
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    torch.set_num_threads(1)
    data=dict(np.load(args.surfaces));n=min(args.count,len(data['q']))
    if n<1:raise ValueError('At least one diagnostic surface is required')
    report={'data_sha256':sha256(args.surfaces),'data':str(args.surfaces),'indices':list(range(n)),
        'scope':'first archived TRAINING surfaces, including unusable/failing ones; not unseen assessment',
        'metric':'B @ (neural_IV - reference_IV), in physical recovery-tolerance coordinates',
        'interpretation':'local linear bias proxy, NOT recovered parameters; truncated directions unmeasured',
        'selection_use':'none; both checkpoints already frozen','models':[]}
    for checkpoint in args.checkpoint:
        model,info=load_checkpoint(checkpoint);rows=[];errors=[];shifts=[]
        for i in range(n):
            row={'surface':i,'reference_usable':bool(data['usable'][i]),'retained_rank':int(data['rank'][i])}
            if not row['reference_usable']:
                row['status']='unusable_reference';rows.append(row);continue
            try:
                with neural_only():
                    c,p=coordinates(torch.tensor(data['q'][i],dtype=torch.float64),2,torch)
                    prediction=model.iv(c,p).detach().numpy()
                error=prediction-data['iv'][i]
                shift=data['preconditioner'][i]@error
                if not np.isfinite(error).all() or not np.isfinite(shift).all():raise FloatingPointError('Nonfinite diagnostic')
                row.update(status='scored',iv_rmse=float(np.sqrt(np.mean(error**2))),
                    linear_tolerance_shift=shift.tolist(),max_abs_linear_shift=float(np.abs(shift).max()))
                errors.append(error);shifts.append(shift)
            except (FloatingPointError,ValueError,RuntimeError) as exc:
                row.update(status='failed',error=f'{type(exc).__name__}: {exc}')
            rows.append(row)
        complete=len(errors)==n
        summary={'checkpoint':info,'surfaces':rows,'complete_scoring':complete,
            'whole_subset_iv_rmse':float(np.sqrt(np.mean(np.array(errors)**2))) if complete else None,
            'whole_subset_linear_shift_rmse':float(np.sqrt(np.mean(np.array(shifts)**2))) if complete else None,
            'median_max_abs_linear_shift':float(np.median(np.abs(shifts).max(axis=1))) if complete else None}
        report['models'].append(summary)
        assert sha256(info['checkpoint'])==info['sha256']
        print(json.dumps({k:v for k,v in summary.items() if k not in ('checkpoint','surfaces')}),flush=True)
    report['source_sha256']={str(p.relative_to(ROOT)):sha256(p) for p in (Path(__file__),
        ROOT/'src/mentor_dh_pinn/convolution_pinn.py',ROOT/'scripts/mentor_dh_pinn/compare_composition_pinn.py')}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':main()
