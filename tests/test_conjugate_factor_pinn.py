import torch
import pytest
from src.mentor_dh_pinn.moment_factor_pinn import MomentFactorPINN
from src.mentor_dh_pinn.two_moment_factor_pinn import TwoMomentFactorPINN
from src.mentor_dh_pinn.conjugate_factor_pinn import ConjugateFactorPINN,unshifted_coordinates
from src.mentor_dh_pinn.affine_factor_pinn import riccati_residual
from src.mentor_dh_pinn.affine_factor_reference import draw_factor_points,reference_coefficients
from src.mentor_dh_pinn.affine_factor_pricing import neural_call_prices
from src.mentor_dh_pinn.regular_pinn_data import decode_unit


def test_coefficient_conjugacy_matches_reference_riccati_equation():
    torch.set_num_threads(1)
    q=torch.tensor(draw_factor_points(1024,908521),dtype=torch.float64,requires_grad=True)
    def mapped(q):
        d,a=reference_coefficients(unshifted_coordinates(q))
        return tuple(torch.complex(v.real,v.imag*(1-2*q[...,5])) for v in (d,a))
    for actual,expected in zip(mapped(q),reference_coefficients(q)):
        torch.testing.assert_close(actual,expected,atol=1e-11,rtol=1e-11)
    assert riccati_residual(mapped,q).abs().max()<1e-9


@pytest.mark.parametrize('model_class',[ConjugateFactorPINN,MomentFactorPINN,TwoMomentFactorPINN])
def test_neural_conjugacy_is_exact_and_parameter_gradients_survive(model_class):
    torch.manual_seed(84)
    model=model_class(width=8,depth=2).double()
    with torch.no_grad():model.head.weight.normal_(0,.001)
    q=torch.tensor(draw_factor_points(20,908522),dtype=torch.float64,requires_grad=True)
    q0=unshifted_coordinates(q)
    for actual,unshifted in zip(model.coefficients(q),model.coefficients(q0)):
        torch.testing.assert_close(actual,torch.complex(unshifted.real,unshifted.imag*(1-2*q[...,5])),atol=1e-10,rtol=1e-11)
    loss=riccati_residual(model.coefficients,q).square().mean();loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.requires_grad_(False)
    x=torch.tensor([-.05,0.,.05],dtype=torch.float64);tau=torch.full_like(x,.4)
    unit=torch.full((10,),.4,dtype=torch.float64)
    fun=lambda u:neural_call_prices(model,decode_unit(u,2,torch),x,tau)
    jac=torch.func.jacfwd(fun)(unit)
    for j in range(10):
        bump=torch.zeros_like(unit);bump[j]=1e-5
        torch.testing.assert_close(jac[:,j],(fun(unit+bump)-fun(unit-bump))/(2e-5),rtol=1e-4,atol=1e-8)
