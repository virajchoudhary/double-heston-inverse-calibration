import numpy as np
import torch

from src.mentor_dh_pinn.multiscale_pinn import TorchMultiscaleVariancePINN
from src.mentor_dh_pinn.regular_pinn_torch import TorchRegularVariancePINN, residual


def setup():
    torch.manual_seed(911)
    base = TorchRegularVariancePINN(width=12, depth=2)
    model = TorchMultiscaleVariancePINN(base, expert_width=8)
    coords = torch.tensor([[.04,.05,.04,.1],[-.07,.08,.05,.7]], dtype=torch.float64)
    structural = torch.tensor([[.8,.04,.19,-.5],[4.5,.05,.35,-.3]], dtype=torch.float64)
    return base, model, coords, structural


def test_zero_initialized_extension_preserves_price_and_pde():
    base, model, z, s = setup()
    torch.testing.assert_close(model.price(z,s),base.price(z,s),rtol=0,atol=0)
    torch.testing.assert_close(residual(model,z,s)[0],residual(base,z,s)[0],rtol=1e-12,atol=1e-12)
    assert not any(p.requires_grad for p in base.parameters())


def test_smooth_gate_partition_and_derivative():
    _, model, _, _ = setup()
    tau = torch.logspace(-3,1,51,dtype=torch.float64,requires_grad=True)
    weights = model.maturity_weights(tau)
    assert (weights >= 0).all()
    torch.testing.assert_close(weights.sum(-1),torch.ones_like(tau))
    derivative = torch.autograd.grad(weights[:,1].sum(),tau)[0]
    eps=1e-7
    fd=(model.maturity_weights(tau.detach()+eps)[:,1]-model.maturity_weights(tau.detach()-eps)[:,1])/(2*eps)
    torch.testing.assert_close(derivative,fd,rtol=1e-6,atol=1e-7)


def test_price_bounds_terminal_and_pde_parameter_gradient():
    _, model, z, s = setup()
    with torch.no_grad():
        for expert in model.experts:
            expert[-1].weight.normal_(std=.1)
    price=model.price(z,s)
    assert (price >= torch.clamp_min(torch.expm1(z[:,0]),0)).all()
    assert (price <= z[:,0].exp()).all()
    terminal=z.clone();terminal[:,-1]=0
    torch.testing.assert_close(model.price(terminal,s),torch.clamp_min(torch.expm1(z[:,0]),0),atol=0,rtol=0)
    loss=residual(model,z,s)[0].square().mean();loss.backward()
    assert all(p.grad is None for p in model.base.parameters())
    weight=model.experts[0][-1].weight
    analytic=weight.grad[0,0].item();original=weight[0,0].item();values=[]
    for delta in [1e-5,-1e-5]:
        with torch.no_grad():weight[0,0]=original+delta
        values.append(residual(model,z,s)[0].square().mean().item())
    np.testing.assert_allclose(analytic,(values[0]-values[1])/2e-5,rtol=1e-5,atol=1e-9)


def test_nonzero_gated_correction_agrees_with_independent_raw_price_pde():
    import math
    _, model, z, s = setup()
    with torch.no_grad():
        for expert in model.experts:
            expert[-1].weight.normal_(std=.2)
    z.requires_grad_(True)
    scaled, diagnostic = residual(model,z,s)
    def derivative(value):
        return torch.autograd.grad(value.sum(),z,create_graph=True)[0]
    first=derivative(model.price(z,s));xx=derivative(first[:,0])
    raw=first[:,-1]-.5*z[:,1:-1].sum(-1)*(xx[:,0]-first[:,0])
    for i in range(2):
        j=i+1;k,theta,sigma,rho=s[i]
        raw=raw-k*(theta-z[:,j])*first[:,j]-rho*sigma*z[:,j]*xx[:,j]-.5*sigma**2*z[:,j]*derivative(first[:,j])[:,j]
    w=diagnostic['w'];d2=z[:,0]/torch.sqrt(w)-.5*torch.sqrt(w)
    derivative_w=torch.exp(-.5*d2**2)/(math.sqrt(2*math.pi)*2*torch.sqrt(w))
    torch.testing.assert_close(scaled,raw/(derivative_w*diagnostic['scale']),rtol=1e-9,atol=1e-11)


def test_market_parameter_jacobian_matches_finite_differences():
    from experiments.nifty_multiscale_v5.market import predictor
    _,model,_,_=setup()
    fn=predictor([model],np.array([-.1,0.,.1]),np.array([.1,.25,.5]))
    z=torch.tensor(np.log([1.,.02,.03]),dtype=torch.float64)
    actual=torch.func.jacfwd(fn)(z).detach().numpy();eps=1e-5
    eye=torch.eye(3,dtype=torch.float64)
    expected=np.column_stack([(fn(z+eps*eye[j]).detach().numpy()-fn(z-eps*eye[j]).detach().numpy())/(2*eps) for j in range(3)])
    np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-9)


def test_outer_quote_perturbation_cannot_change_anchor_bs_fit():
    import pandas as pd
    from experiments.nifty_multiscale_v5.market import arrays,choose_bs,split_day
    from experiments.nifty_multifactor_v4.literature_exact import black
    x=np.linspace(-.2,.2,18);tau=np.repeat([.1,.3],9)
    day=pd.DataFrame({'expiry':np.repeat(['a','b'],9),'strike':np.exp(-x),
        'x':x,'tau':tau,'market_call_normalized':black(x,tau,.2)*np.exp(x),
        'split':np.where(np.arange(18)%3==1,'test','anchor')})
    anchor,_=split_day(day);first=choose_bs(*arrays(anchor))
    day.loc[day.split.eq('test'),'market_call_normalized']=999
    anchor,_=split_day(day);second=choose_bs(*arrays(anchor))
    assert first==second
