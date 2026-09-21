import numpy as np
import pytest
import torch
import scripts.mentor_dh_pinn.assess_regular_pinn as assess
from src.mentor_dh_pinn.regular_pinn_torch import TorchRegularVariancePINN
from src.mentor_dh_pinn.surrogate_error import SurrogateError


def test_regularized_jacobian_objective_and_holdout_isolation(monkeypatch):
    torch.set_num_threads(1);torch.manual_seed(12)
    net=TorchRegularVariancePINN(factors=1,width=8,depth=2).double().requires_grad_(False)
    x=np.tile(np.linspace(-.1,.1,6),3);tau=np.repeat([.1,.5,1.],6);mask=np.arange(18)%3!=2
    iv=assess._network_iv(net,x,tau,np.array([.3,.4,.5,.6,.7]))
    stats=SurrogateError(x,tau,np.zeros(18),np.eye(18)*1e-6,'test')
    original=assess.least_squares
    def checked(fun,u,jac,**kwargs):
        h=1e-6;eye=np.eye(len(u))*h
        numeric=np.column_stack([(fun(u+d)-fun(u-d))/(2*h) for d in eye])
        np.testing.assert_allclose(jac(u),numeric,rtol=1e-5,atol=1e-6)
        return original(fun,u,jac=jac,**kwargs)
    monkeypatch.setattr(assess,'least_squares',checked)
    monkeypatch.setattr(assess,'exact_prices',lambda *a,**kw:pytest.fail('Exact trial pricing forbidden'))
    kw=dict(fit_mask=mask,starts=1,max_nfev=50,surrogate_error=stats,prior_strength=10.)
    a=assess.fit_network(net,x,tau,iv,**kw);iv[~mask]=np.nan
    b=assess.fit_network(net,x,tau,iv,**kw)
    np.testing.assert_array_equal(a['unit'],b['unit'])
    assert a['prior_penalty']==pytest.approx(10*np.sum((np.array(a['unit'])-.5)**2))
    assert a['calibration_objective']==min(r['sse'] for r in a['starts'])
    assert a['calibration_objective']==pytest.approx(a['calibration_iv_sse']/1e-6+a['prior_penalty'])
    with pytest.raises(ValueError,match='prior_strength'):
        assess.fit_network(net,x,tau,iv,fit_mask=mask,prior_strength=-1)
