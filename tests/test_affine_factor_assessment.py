import numpy as np
import torch
import pytest

from src.mentor_dh_pinn.conjugate_factor_pinn import build_factor_pinn
from scripts.mentor_dh_pinn.assess_affine_factor_pinn import evaluate_cases


def without_timing(value):
    if isinstance(value,dict):return {k:without_timing(v) for k,v in value.items() if k!='seconds'}
    if isinstance(value,list):return [without_timing(v) for v in value]
    return value


@pytest.mark.parametrize('constraints',[{}, {'conjugate':True,'moment':True,'two_moment':True}])
def test_spawned_assessment_preserves_serial_results(tmp_path,constraints):
    torch.manual_seed(83)
    config={'width':8,'depth':2,'integrated':True,**constraints}
    model=build_factor_pinn(config)
    with torch.no_grad():model.head.weight.normal_(0,.001)
    weights=tmp_path/'model.pt';torch.save(model.state_dict(),weights)
    x=np.tile([-.08,0.,.08],2);tau=np.repeat([.3,1.],3);holdout=np.arange(6)%3==2
    jobs=[(i,np.full(10,.4+i*.03),x,tau,holdout,1,2,71+i,weights,
           config) for i in range(2)]
    serial=list(evaluate_cases(jobs,1));parallel=list(evaluate_cases(jobs,2))
    assert all(row['status']=='fitted' for row in serial)
    assert without_timing(serial)==without_timing(parallel)
