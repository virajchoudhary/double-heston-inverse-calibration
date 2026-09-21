import numpy as np
import torch
from src.mentor_dh_pinn.research_modified_pinn import ModifiedMLP,ResearchCorrectionPINN
from src.mentor_dh_pinn.regular_pinn_torch import TorchRegularVariancePINN,residual
from src.mentor_dh_pinn.multiscale_pinn import TorchMultiscaleVariancePINN


def test_published_modified_recurrence_against_numpy():
    torch.manual_seed(21);model=ModifiedMLP(inputs=3,width=7,depth=3)
    with torch.no_grad():model.head.weight.normal_()
    x=np.random.default_rng(22).normal(size=(8,3))
    def linear(layer,x):return x@layer.weight.detach().numpy().T+layer.bias.detach().numpy()
    u=np.tanh(linear(model.u,x));v=np.tanh(linear(model.v,x));h=x
    for layer in model.layers:
        gate=np.tanh(linear(layer,h));h=gate*u+(1-gate)*v
    expected=linear(model.head,h)[:,0]
    np.testing.assert_allclose(model(torch.tensor(x)).detach().numpy(),expected,rtol=1e-12,atol=1e-12)


def example():
    torch.manual_seed(29)
    base=TorchMultiscaleVariancePINN(TorchRegularVariancePINN(width=12,depth=2),expert_width=8,gated=False)
    net=ResearchCorrectionPINN(base,np.zeros(26),np.ones(26))
    z=torch.tensor([[.02,.02,.03,.1],[-.1,.04,.08,.5]],dtype=torch.float64)
    s=torch.tensor([[.9491,.0257,.0517,.7009],[10.7526,.033,.3613,-.8916]],dtype=torch.float64)
    return base,net,z,s


def test_pretrained_price_pde_and_terminal_preservation():
    base,net,z,s=example()
    torch.testing.assert_close(base.price(z,s),net.price(z,s),rtol=0,atol=0)
    torch.testing.assert_close(residual(base,z,s)[0],residual(net,z,s)[0],rtol=1e-12,atol=1e-12)
    z[:,-1]=0
    torch.testing.assert_close(net.price(z,s),torch.clamp_min(torch.expm1(z[:,0]),0),rtol=0,atol=0)


def test_modified_pde_parameter_derivative():
    _,net,z,s=example()
    with torch.no_grad():net.core.head.weight.normal_(std=.02)
    loss=residual(net,z,s)[0].square().mean();loss.backward()
    weight=net.core.u.weight;actual=weight.grad[0,0].item();original=weight[0,0].item();values=[]
    for delta in [1e-5,-1e-5]:
        with torch.no_grad():weight[0,0]=original+delta
        values.append(residual(net,z,s)[0].square().mean().item())
    np.testing.assert_allclose(actual,(values[0]-values[1])/2e-5,rtol=1e-5,atol=1e-9)
    assert all(p.grad is None for p in net.base.parameters())


def test_stratified_minibatches_preserve_all_marginal_bins():
    from experiments.pinn_architecture_v6.train import strata,sample_indices
    from experiments.pinn_architecture_v6.common import old,read
    z,_=old.sample(read(old.HERE/'config.json'),1000,982)
    groups=strata(z);rng=np.random.default_rng(881)
    for _ in range(20):
        selected=sample_indices(rng,groups,total=1000).numpy()
        assert all(np.isin(selected,group).any() for group in groups)


def test_modified_network_transformed_pde_matches_raw_price_pde():
    import math
    _,net,z,s=example()
    with torch.no_grad():net.core.head.weight.normal_(std=.03)
    z.requires_grad_(True)
    scaled,dg=residual(net,z,s)
    def d(value):return torch.autograd.grad(value.sum(),z,create_graph=True)[0]
    first=d(net.price(z,s));xx=d(first[:,0])
    raw=first[:,-1]-.5*z[:,1:-1].sum(-1)*(xx[:,0]-first[:,0])
    for i in range(2):
        j=i+1;k,t,sigma,rho=s[i]
        raw=raw-k*(t-z[:,j])*first[:,j]-rho*sigma*z[:,j]*xx[:,j]-.5*sigma**2*z[:,j]*d(first[:,j])[:,j]
    w=dg['w'];d2=z[:,0]/w.sqrt()-.5*w.sqrt()
    cw=torch.exp(-.5*d2**2)/(2*math.sqrt(2*math.pi)*w.sqrt())
    torch.testing.assert_close(scaled,raw/(cw*dg['scale']),rtol=1e-9,atol=1e-10)


def test_unchanged_fidelity_gates_reject_each_failure():
    from experiments.pinn_architecture_v6.evaluate import passed
    good={'price_RMSE':2e-5,'price_P95':5e-5,'price_max':2e-4,'IV_RMSE_volatility_points':.2}
    assert passed(good)
    for key in good:
        bad=dict(good);bad[key]*=1.000001;assert not passed(bad)
        bad[key]=float('nan');assert not passed(bad)
    bad=dict(good);bad['IV_RMSE_volatility_points']=None;assert not passed(bad)
