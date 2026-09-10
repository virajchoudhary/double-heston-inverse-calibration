import numpy as np
import torch
import pytest
from src.mentor_dh_pinn.affine_factor_pinn import AffineFactorPINN
from src.mentor_dh_pinn.integrated_factor_pinn import IntegratedFactorPINN
from src.mentor_dh_pinn.conjugate_factor_pinn import ConjugateFactorPINN
from src.mentor_dh_pinn.moment_factor_pinn import MomentFactorPINN
from src.mentor_dh_pinn.two_moment_factor_pinn import TwoMomentFactorPINN
from src.mentor_dh_pinn.affine_factor_pricing import neural_call_prices
from src.mentor_dh_pinn.affine_factor_calibration import fit_factor_pinn
from src.mentor_dh_pinn.regular_pinn_data import decode_unit,invert_total_variance


@pytest.mark.parametrize('model_class', [AffineFactorPINN, IntegratedFactorPINN, ConjugateFactorPINN, MomentFactorPINN, TwoMomentFactorPINN])
def test_neural_fit_is_heldout_blind_and_has_no_exact_pricer(monkeypatch, model_class):
    import src.mentor_dh_pinn.affine_factor_reference as reference
    import src.mentor_dh_pinn.torch_pricer as pricer
    import src.mentor_dh_pinn.regular_pinn_data as data
    def forbidden(*a,**kw):raise AssertionError('Exact reference in neural inverse fitting')
    monkeypatch.setattr(reference,'reference_coefficients',forbidden)
    monkeypatch.setattr(pricer,'price_call',forbidden)
    monkeypatch.setattr(pricer,'price_call_single',forbidden)
    monkeypatch.setattr(data,'price_call',forbidden)
    monkeypatch.setattr(data,'price_call_single',forbidden)
    torch.set_num_threads(1)
    torch.manual_seed(81)
    model=model_class(width=8,depth=2).double()
    with torch.no_grad():model.head.weight.normal_(0,.001)
    model.requires_grad_(False)
    original={key:value.clone() for key,value in model.state_dict().items()}
    x=np.tile([-.08,0.,.08],2);tau=np.repeat([.3,1.],3)
    p=decode_unit(torch.full((10,),.45,dtype=torch.float64),2,torch)
    price=neural_call_prices(model,p,torch.tensor(x),torch.tensor(tau)).numpy()
    iv=np.sqrt(invert_total_variance(price,x)/tau);mask=np.arange(len(x))%3!=2
    first=fit_factor_pinn(model,x,tau,iv,fit_mask=mask,starts=1,max_nfev=2,seed=31)
    assert first['status']=='fitted'
    x[~mask]=np.nan;tau[~mask]=-1;iv[~mask]=np.inf
    second=fit_factor_pinn(model,x,tau,iv,fit_mask=mask,starts=1,max_nfev=2,seed=31)
    assert first['unit']==second['unit'] and first['calibration_iv_sse']==second['calibration_iv_sse']
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,original[key],rtol=0,atol=0)
    changed=iv.copy();changed[mask]*=1.01
    third=fit_factor_pinn(model,x,tau,changed,fit_mask=mask,starts=1,max_nfev=2,seed=31)
    assert third['status']=='fitted' and third['unit']!=first['unit']
