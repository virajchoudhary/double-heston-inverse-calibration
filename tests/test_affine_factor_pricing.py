from types import SimpleNamespace
import numpy as np
import torch
from src.mentor_dh_pinn.affine_factor_pinn import AffineFactorPINN,ratios
from src.mentor_dh_pinn.affine_factor_reference import reference_coefficients
from src.mentor_dh_pinn.affine_factor_pricing import neural_call_prices
from src.mentor_dh_pinn.regular_pinn_data import black_call,decode_unit
from src.mentor_dh_pinn.torch_pricer import price_call


def geometry():
    return (torch.tensor(np.tile(-np.log([.8,.95,1.,1.05,1.2]),3),dtype=torch.float64),
            torch.tensor(np.repeat([.1,.5,1.5],5),dtype=torch.float64))


def test_integral_with_reference_coefficients_matches_canonical_engine():
    p=decode_unit(torch.full((10,),.45,dtype=torch.float64),2,torch)
    x,tau=geometry();provider=SimpleNamespace(coefficients=reference_coefficients)
    actual=neural_call_prices(provider,p,x,tau)
    one=torch.ones_like(x)
    expected=price_call(p,torch.exp(x),one,tau,one*0,one*0,node_count=128)
    torch.testing.assert_close(actual,expected,atol=2e-11,rtol=1e-10)


def test_neural_deterministic_limit_and_forward_parameter_jacobian(monkeypatch):
    import src.mentor_dh_pinn.affine_factor_reference as reference
    import src.mentor_dh_pinn.torch_pricer as pricer
    def forbidden(*a,**kw):raise AssertionError('Exact pricer entered neural quadrature')
    monkeypatch.setattr(reference,'reference_coefficients',forbidden)
    monkeypatch.setattr(pricer,'price_call',forbidden)
    model=AffineFactorPINN(width=8,depth=2).double()
    model.requires_grad_(False)
    p=torch.tensor([.8,.06,0.,-.4,.04,5.,.03,0.,-.1,.02],dtype=torch.float64)
    x,tau=geometry()
    actual=neural_call_prices(model,p,x,tau)
    integrated=sum(tau*(p[i+1]+(p[i+4]-p[i+1])*ratios(p[i]*tau)[0]) for i in (0,5))
    np.testing.assert_allclose(actual.numpy(),black_call(x.numpy(),integrated.numpy()),atol=1e-9)
    p=decode_unit(torch.full((10,),.45,dtype=torch.float64),2,torch)
    fun=lambda value:neural_call_prices(model,value,x,tau)
    jac=torch.func.jacfwd(fun)(p)
    for j in (0,2,3,4,6):
        bump=torch.zeros_like(p);bump[j]=1e-5
        torch.testing.assert_close(jac[:,j],(fun(p+bump)-fun(p-bump))/(2e-5),rtol=1e-4,atol=1e-8)
