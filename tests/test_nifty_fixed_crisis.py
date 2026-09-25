import numpy as np
import torch
from scripts.mentor_dh_pinn.nifty_fixed_crisis import single_match, normalized_call, raw_option, state
from src.mentor_dh_pinn.regular_pinn_torch import TorchRegularVariancePINN, residual

P = np.array([.9,.1,.36,-.5,.36,1.2,.15,.2,-.5,.2])


def test_single_matches_initial_total_moments():
    k,t,s,r,v = P.reshape(2,5).T
    a,b,c,d,e = single_match(P)
    np.testing.assert_allclose([e,b,a*(b-e),c*c*e,d*c*e],
                              [v.sum(),t.sum(),np.sum(k*(t-v)),np.sum(s*s*v),np.sum(r*s*v)])
    assert 2*a*b>c*c


def test_put_call_normalization_roundtrip():
    x = np.array([-.1,.1]); opt = np.array(['CE','PE'])
    price = np.array([20.,30.]); K=np.array([1000.,1000.]); D=.99
    c=normalized_call(price,opt,x,D,K)
    np.testing.assert_allclose(raw_option(c,opt,x,D,K),price)


def test_pde_updates_network_not_fixed_parameters_and_payoff_exact():
    for p in [P,single_match(P)]:
        st=torch.tensor(p.reshape(-1,5)[:,:4])
        net=TorchRegularVariancePINN(factors=len(p)//5,width=8,depth=2)
        coords=state(np.array([[-.1,.1],[.1,.2]]),p)
        eq,_=residual(net,coords,st)
        eq.square().mean().backward()
        assert st.grad is None and not st.requires_grad
        assert any(w.grad is not None and torch.isfinite(w.grad).all() and w.grad.abs().sum()>0 for w in net.parameters())
        coords[:,-1]=0
        np.testing.assert_allclose(net.price(coords,st).detach().numpy(),np.maximum(np.expm1([-.1,.1]),0),atol=1e-14)


def test_scaled_residual_matches_direct_price_autodifferentiation():
    for p in [P,single_match(P)]:
        st=torch.tensor(p.reshape(-1,5)[:,:4])
        torch.manual_seed(111)
        net=TorchRegularVariancePINN(factors=len(p)//5,width=8,depth=2)
        coords=state(np.array([[-.1,.1],[.1,.2]]),p).requires_grad_(True)
        derivative=lambda y: torch.autograd.grad(y.sum(),coords,create_graph=True)[0]
        c=net.price(coords,st)
        first=derivative(c); second=derivative(first[:,0])
        gen=.5*coords[:,1:-1].sum(-1)*(second[:,0]-first[:,0])
        for i,(k,t,s,r) in enumerate(st):
            j=i+1; v=coords[:,j]
            gen=gen+k*(t-v)*first[:,j]+r*s*v*second[:,j]+.5*s*s*v*derivative(first[:,j])[:,j]
        res,diag=residual(net,coords,st)
        w=diag['w']; d2=coords[:,0]/w.sqrt()-.5*w.sqrt()
        cw=torch.exp(-.5*d2*d2)/(np.sqrt(2*np.pi)*2*w.sqrt())
        np.testing.assert_allclose(res.detach(),((first[:,-1]-gen)/(cw*diag['scale'])).detach(),atol=2e-11,rtol=2e-10)


def test_published_dh1_matches_independent_adaptive_pricer():
    from scripts.mentor_dh_pinn.nifty_fixed_crisis import reference
    from src.double_heston_reference import reference_double_heston_call
    xt=np.array([[x,t] for x in [-.3,0.,.3] for t in [7/365,30/365,100/365]])
    actual=reference(P,xt)
    for (x,t),price in zip(xt,actual):
        expected,diag=reference_double_heston_call(np.exp(x),1.,t,0.,0.,P)
        assert diag['reliable']
        np.testing.assert_allclose(price,expected,atol=2e-10,rtol=0)
