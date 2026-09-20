import pytest
import math
import torch
from src.mentor_dh_pinn.maturity_dual_pinn import MaturityDualPINN, european_price
from src.mentor_dh_pinn.regular_pinn_torch import residual

P = [.8,.1,.4,-.4,.08,8.,.2,1.,.3,.12]


def test_dual_price_physics_and_adapter():
    torch.manual_seed(1)
    net = MaturityDualPINN(width=12, depth=2)
    p = torch.tensor([P]*3, dtype=torch.float64)
    st = p.reshape(-1,2,5)[...,:4]
    coords = torch.tensor([[-.2,.08,.12,.02],[0,.08,.12,.25],[.2,.08,.12,1.]], dtype=torch.float64)
    eq, diag = residual(net,coords,st)
    (eq.square().mean()+diag['convexity'].square().mean()).backward()
    for name, par in net.named_parameters():
        assert par.grad is not None and torch.isfinite(par.grad).all(), name
    assert net.long_head.weight.grad.abs().sum()>0
    kwargs = dict(forward=[80,100,120],strike=100,tau=.25,discount=.99,params=P)
    c = european_price(net,**kwargs)
    put = european_price(net,option_type='put',**kwargs)
    torch.testing.assert_close(c-put,.99*(torch.tensor([80.,100.,120.],dtype=torch.float64)-100))
    assert (c>=0).all() and (put>=0).all() and (c<=.99*torch.tensor([80,100,120])).all()
    larger = european_price(net,**{**kwargs,'forward':[8000,10000,12000],'strike':10000})
    torch.testing.assert_close(larger,100*c)
    terminal = european_price(net,**{**kwargs,'tau':0,'discount':1})
    torch.testing.assert_close(terminal,torch.tensor([0.,0.,20.],dtype=torch.float64))
    # Arbitrary quote counts/order, no fixed legacy strike/expiry grid.
    reversed_c = european_price(net,**{**kwargs,'forward':[120,100,80]})
    torch.testing.assert_close(reversed_c,c.flip(0))


@pytest.mark.parametrize('change',[
    {'exercise':'american'}, {'option_type':'barrier'}, {'forward':0},
    {'tau':-.1}, {'tau':3.}, {'forward':1000}, {'params':[1]*9},
    {'params':[8,.1,.4,1,.08,.8,.2,1,.3,.12]}, {'discount':float('nan')}])
def test_adapter_rejects_unsupported_data(change):
    args=dict(forward=100,strike=100,tau=.25,discount=1,params=P)
    with pytest.raises(ValueError): european_price(MaturityDualPINN(width=8,depth=1),**{**args,**change})


def test_dual_residual_matches_independent_price_pde_including_gate_derivatives():
    torch.manual_seed(41)
    net=MaturityDualPINN(width=12,depth=2)
    # Around the transition, tau derivatives of the mixing gate matter most.
    z=torch.tensor([[.04,.08,.12,.23],[-.1,.06,.1,.28]],dtype=torch.float64,requires_grad=True)
    st=torch.tensor(P,dtype=torch.float64).reshape(2,5)[:,:4]
    result,diag=residual(net,z,st)
    def d(value):return torch.autograd.grad(value.sum(),z,create_graph=True)[0]
    c=net.price(z,st);first=d(c);xx=d(first[:,0])
    raw=first[:,-1]-.5*z[:,1:3].sum(-1)*(xx[:,0]-first[:,0])
    for i in range(2):
        j=i+1;k,theta,sigma,rho=st[i]
        raw=raw-k*(theta-z[:,j])*first[:,j]-rho*sigma*z[:,j]*xx[:,j]-.5*sigma**2*z[:,j]*d(first[:,j])[:,j]
    w=diag['w'];d2=z[:,0]/w.sqrt()-.5*w.sqrt()
    cw=torch.exp(-.5*d2*d2)/(math.sqrt(2*math.pi)*2*w.sqrt())
    torch.testing.assert_close(result,raw/(cw*diag['scale']),atol=2e-11,rtol=2e-9)


def test_shared_structure_broadcast_matches_batch_and_training_snapshot():
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'experiments/portable_dual_pinn_v1/model_training_snapshot.py'
    spec=importlib.util.spec_from_file_location('src.mentor_dh_pinn._training_snapshot',path)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    torch.manual_seed(29)
    net=MaturityDualPINN(width=12,depth=2);prior=old.MaturityDualPINN(width=12,depth=2)
    prior.load_state_dict(net.state_dict())
    z=torch.tensor([[.04,.08,.12,.23],[-.1,.06,.1,.28]],dtype=torch.float64)
    shared=torch.tensor(P,dtype=torch.float64).reshape(2,5)[:,:4]
    batch=shared.expand(2,-1,-1)
    torch.testing.assert_close(net.price(z,shared),net.price(z,batch),atol=0,rtol=0)
    torch.testing.assert_close(net.price(z,batch),prior.price(z,batch),atol=0,rtol=0)
