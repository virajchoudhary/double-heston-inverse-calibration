import numpy as np
import torch

from scripts.mentor_dh_pinn.compare_composition_pinn import neural_only,without_timing,geometry
from scripts.mentor_dh_pinn.assess_regular_pinn import fit_network
from src.mentor_dh_pinn.convolution_pinn import ConvolutionPINN
from src.mentor_dh_pinn.regular_pinn_data import coordinates
from src.mentor_dh_pinn.regular_pinn_torch import TorchRegularVariancePINN


def test_composition_calibration_isolated_from_heldout_and_exact_pricer():
    torch.set_num_threads(1)
    torch.manual_seed(910061)
    component=TorchRegularVariancePINN(factors=1,width=8,depth=2)
    with torch.no_grad():component.head.weight.normal_(0,.001)
    model=ConvolutionPINN(component,32).eval().requires_grad_(False)
    x=np.tile(np.array([-.1,0,.1]),2);tau=np.repeat([.25,1.],3)
    q=torch.tensor(np.column_stack([x,np.log(tau),np.full((6,10),.5)]),dtype=torch.float64)
    c,p=coordinates(q,2,torch)
    observed=model.iv(c,p).detach().numpy()
    mask=np.array([True,False,True,True,False,True])
    with neural_only():
        actual=fit_network(model,x,tau,observed,fit_mask=mask,starts=2,max_nfev=2,seed=910062)
        corrupted=[a.copy() for a in (x,tau,observed)]
        for a in corrupted:a[~mask]=np.nan
        replay=fit_network(model,*corrupted,fit_mask=mask,starts=2,max_nfev=2,seed=910062)
    assert actual['status']=='fitted'
    assert actual['fit_quotes']==4 and len(actual['starts'])==2
    assert without_timing(actual)==without_timing(replay)


def test_fixed_common_quote_geometry():
    x,tau,holdout=geometry()
    assert len(x)==126 and holdout.sum()==42
    assert len(np.unique(tau))==6
    assert np.all(tau>0) and np.max(tau)==2.


def test_raw_composition_pde_in_deterministic_variance_limit():
    from scripts.mentor_dh_pinn.check_composition_physics import raw_price_residual
    component=TorchRegularVariancePINN(factors=1,width=4,depth=1)
    with torch.no_grad():component.head.weight.zero_();component.head.bias.zero_()
    model=ConvolutionPINN(component,64).requires_grad_(False)
    state=torch.tensor([[-.1,.04,.03,.5],[.1,.04,.03,1.]],dtype=torch.float64)
    structural=torch.tensor([[[1.,.06,0.,-.3],[4.,.08,0.,-.4]]],dtype=torch.float64).expand(2,2,4)
    residual,diagnostics=raw_price_residual(model,state,structural)
    torch.testing.assert_close(residual,torch.zeros_like(residual),rtol=0,atol=1e-10)
    assert (diagnostics['convexity']>=0).all() and (diagnostics['calendar']>=0).all()


def test_raw_pde_matches_stochastic_fourier_reference():
    # Assessment-only independent representation; never an inference component.
    from scripts.mentor_dh_pinn.check_composition_physics import raw_price_residual
    from src.mentor_dh_pinn.torch_pricer import price_call
    class Reference:
        def price(self,c,p):
            physical=torch.cat([p[:,0],c[:,1:2],p[:,1],c[:,2:3]],-1)
            return price_call(physical,torch.exp(c[:,0:1]),torch.ones_like(c[:,0:1]),
                c[:,-1:],0.,0.,node_count=128).squeeze(-1)
    state=torch.tensor([[-.1,.06,.04,.25],[.1,.06,.04,1.]],dtype=torch.float64)
    p=torch.tensor([[[.8,.04,.2,-.5],[5.,.05,.4,.2]]],dtype=torch.float64).expand(2,2,4)
    residual,_=raw_price_residual(Reference(),state,p)
    torch.testing.assert_close(residual,torch.zeros_like(residual),rtol=0,atol=1e-10)
