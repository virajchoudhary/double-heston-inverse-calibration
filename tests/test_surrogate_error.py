import numpy as np
import pytest
from src.mentor_dh_pinn.surrogate_error import SurrogateError


def test_whitening_and_masked_uncertainty():
    x=np.arange(4.);t=np.ones(4);mask=np.array([True,False,True,True])
    a=np.array([[1.,.2,.1,0],[.2,2,0,.1],[.1,0,3,.2],[0,.1,.2,4]])
    stats=SurrogateError(x,t,np.zeros(4),a,'test')
    mean,w=stats.transform(x,t,mask,1e-4,np.array([.1,np.nan,.2,.3]))
    cov=a[np.ix_(mask,mask)]+np.diag([.01,.04,.09])
    np.testing.assert_allclose(w@cov@w.T,np.eye(3),atol=1e-14)
    assert mean.shape==(3,)
    with pytest.raises(ValueError,match='geometry'):
        stats.transform(x+.01,t,mask)
    with pytest.raises(ValueError,match='floor'):
        stats.transform(x,t,mask,0)


def test_gls_is_neural_only_and_holdout_blind(monkeypatch):
    import torch
    import scripts.mentor_dh_pinn.assess_regular_pinn as assess
    from src.mentor_dh_pinn.regular_pinn_torch import TorchRegularVariancePINN
    torch.set_num_threads(1);torch.manual_seed(2)
    net=TorchRegularVariancePINN(factors=1,width=8,depth=2).double().requires_grad_(False)
    x=np.tile(np.linspace(-.1,.1,6),3);tau=np.repeat([.1,.5,1.],6)
    mask=np.arange(18)%3!=2
    iv=assess._network_iv(net,x,tau,np.full(5,.5))
    stats=SurrogateError(x,tau,np.zeros(18),np.eye(18)*1e-6,'test')
    def forbidden(*a,**kw):raise AssertionError('Exact pricing during fit')
    monkeypatch.setattr(assess,'exact_prices',forbidden)
    a=assess.fit_network(net,x,tau,iv,fit_mask=mask,starts=1,max_nfev=30,surrogate_error=stats)
    iv[~mask]=np.nan
    b=assess.fit_network(net,x,tau,iv,fit_mask=mask,starts=1,max_nfev=30,surrogate_error=stats)
    np.testing.assert_array_equal(a['unit'],b['unit'])
    assert a['calibration_objective']==min(r['sse'] for r in a['starts'])
    assert a['objective_kind']=='training_error_GLS'
