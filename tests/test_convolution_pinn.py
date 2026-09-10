import numpy as np
import torch
import json
import pytest

from src.mentor_dh_pinn.convolution_pinn import ConvolutionPINN,ImplicitBlackIV
from src.mentor_dh_pinn.regular_pinn_torch import TorchRegularVariancePINN,black_call
from src.mentor_dh_pinn.regular_pinn_data import coordinates,draw_points


def test_independent_black_composition_and_parameter_gradients():
    component=TorchRegularVariancePINN(factors=1,width=8,depth=2)
    with torch.no_grad():component.head.weight.zero_();component.head.bias.zero_()
    model=ConvolutionPINN(component,nodes=96)
    q=torch.tensor(draw_points(12,2,910001),dtype=torch.float64,requires_grad=True)
    c,p=coordinates(q,2,torch)
    value,diagnostics=model.price_and_diagnostics(c,p)
    from src.mentor_dh_pinn.regular_pinn_torch import expected_variance
    target=black_call(c[:,0],expected_variance(c,p)*c[:,-1])
    torch.testing.assert_close(value,target,rtol=1e-9,atol=1e-11)
    torch.testing.assert_close(diagnostics['mass'],torch.ones(12,dtype=torch.float64),rtol=1e-10,atol=1e-10)
    torch.testing.assert_close(diagnostics['martingale_moment'],torch.ones(12,dtype=torch.float64),rtol=1e-10,atol=1e-10)
    actual_grad=torch.autograd.grad(value.sum(),q,retain_graph=True)[0]
    expected_grad=torch.autograd.grad(target.sum(),q)[0]
    torch.testing.assert_close(actual_grad,expected_grad,rtol=1e-7,atol=1e-10)


def test_implicit_iv_first_derivatives():
    x=torch.tensor([-.15,.04,.2],dtype=torch.float64,requires_grad=True)
    tau=torch.tensor([.1,.4,1.],dtype=torch.float64,requires_grad=True)
    sigma=torch.tensor([.22,.4,.5],dtype=torch.float64,requires_grad=True)
    price=black_call(x,sigma*sigma*tau)
    recovered=ImplicitBlackIV.apply(price,x,tau)
    torch.testing.assert_close(recovered,sigma,rtol=1e-10,atol=1e-11)
    gx,gt,gs=torch.autograd.grad(recovered.sum(),(x,tau,sigma))
    torch.testing.assert_close(gx,torch.zeros_like(gx),rtol=0,atol=1e-9)
    torch.testing.assert_close(gt,torch.zeros_like(gt),rtol=0,atol=1e-9)
    torch.testing.assert_close(gs,torch.ones_like(gs),rtol=1e-9,atol=1e-9)


def test_learned_price_parameter_jacobian_and_checkpoint(tmp_path):
    from scripts.mentor_dh_pinn.assess_regular_pinn import load_checkpoint
    torch.manual_seed(910051)
    component=TorchRegularVariancePINN(factors=1,width=8,depth=2)
    with torch.no_grad():
        component.head.weight.normal_(0,.002)
        component.head.bias.fill_(.001)
    model=ConvolutionPINN(component,96).eval().requires_grad_(False)
    q=torch.tensor(np.column_stack([[-.1,0,.1], np.log([.2,.5,1]),
        np.full((3,10),.5)]),dtype=torch.float64,requires_grad=True)
    c,p=coordinates(q,2,torch)
    actual=torch.autograd.grad(model.iv(c,p).sum(),q)[0]
    for j in range(q.shape[1]):
        h=1e-5
        plus=q.detach().clone();minus=plus.clone()
        plus[:,j]+=h;minus[:,j]-=h
        cp,pp=coordinates(plus,2,torch);cm,pm=coordinates(minus,2,torch)
        fd=(model.iv(cp,pp)-model.iv(cm,pm))/(2*h)
        torch.testing.assert_close(actual[:,j],fd,rtol=2e-5,atol=2e-8)
    config={'checkpoint_format':'convolution_price_pinn_float64',
        'component_architecture':{'factors':1,'width':8,'depth':2},'quadrature_nodes':96}
    (tmp_path/'config.json').write_text(json.dumps(config))
    torch.save(model.state_dict(),tmp_path/'model.pt')
    restored,_=load_checkpoint(tmp_path)
    torch.testing.assert_close(model.price(c,p),restored.price(c,p),rtol=0,atol=0)


def test_component_domain_is_feasible_and_covers_factors():
    from src.mentor_dh_pinn.component_pinn_data import decode_component_unit,component_coordinates
    u=np.random.default_rng(910052).uniform(0,1,(100,5))
    p=decode_component_unit(u,1)
    assert (2*p[:,0]*p[:,1]>p[:,2]**2).all()
    assert (np.abs(p[:,3])<1).all()
    assert p[:,1].min()<.03 and p[:,4].min()<.03
    torch.testing.assert_close(torch.tensor(p),decode_component_unit(torch.tensor(u),1,torch))
    q=torch.tensor(draw_points(8,1,910053,decoder=decode_component_unit),dtype=torch.float64,requires_grad=True)
    c,structural=component_coordinates(q,1,torch)
    assert c.shape==(8,3) and structural.shape==(8,1,4)
    assert torch.isfinite(torch.autograd.grad((c.sum()+structural.sum()),q)[0]).all()


def test_exact_reference_composition_identity():
    # Reference-only identity test. This component is NEVER used by the PINN.
    from src.mentor_dh_pinn.torch_pricer import price_call_single,price_call
    class ExactReference(torch.nn.Module):
        factors=1
        def price(self,c,p):
            physical=torch.cat([p[:,0],c[:,1:2]],-1)
            return price_call_single(physical,torch.exp(c[:,0:1]),torch.ones_like(c[:,0:1]),
                c[:,2:3],0.,0.,node_count=128).squeeze(-1)
    physical=torch.tensor([1.,.06,.15,-.4,.05,5.,.04,.25,-.2,.06],dtype=torch.float64)
    c=torch.tensor([[-.1,.05,.06,.5],[.1,.05,.06,1.]],dtype=torch.float64)
    p=physical.reshape(2,5)[:,:4].expand(2,2,4)
    target=price_call(physical,torch.exp(c[:,0]),torch.ones(2),c[:,-1],0.,0.,node_count=128)
    # More GH nodes extend |x| into extreme tails where this fixed 128-node
    # Fourier REFERENCE aliases. Use two converged moderate GH rules here;
    # production neural quadrature remains 96 and is independently audited at128.
    for nodes in (32,48):
        composed,diagnostics=ConvolutionPINN(ExactReference(),nodes).price_and_diagnostics(c,p)
        torch.testing.assert_close(composed,target,rtol=1e-6,atol=1e-8)
        torch.testing.assert_close(diagnostics['mass'],torch.ones(2,dtype=torch.float64),rtol=1e-8,atol=1e-8)


def test_composition_terminal_payoff_and_invalid_maturities():
    model=ConvolutionPINN(TorchRegularVariancePINN(factors=1,width=8,depth=2),32)
    q=torch.tensor(np.column_stack([[-.2,0,.2],np.log([.2,.2,.2]),np.full((3,10),.5)]),dtype=torch.float64,requires_grad=True)
    c,p=coordinates(q,2,torch)
    terminal=torch.cat([c[:,:-1],torch.zeros_like(c[:,-1:])],-1)
    price=model.price(terminal,p)
    torch.testing.assert_close(price,torch.clamp_min(torch.expm1(c[:,0]),0),rtol=0,atol=0)
    grad=torch.autograd.grad(price.sum(),q)[0]
    torch.testing.assert_close(grad[:,2:],torch.zeros_like(grad[:,2:]),rtol=0,atol=0)
    with pytest.raises(ValueError,match='positive maturity'):model.price_and_diagnostics(terminal,p)
    with pytest.raises(ValueError,match='positive maturity'):model.iv(terminal,p)
    invalid=terminal.detach().clone();invalid[0,-1]=-.1
    with pytest.raises(ValueError,match='nonnegative maturity'):model.price(invalid,p)
