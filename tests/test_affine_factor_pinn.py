import numpy as np
import torch
import pytest
from src.mentor_dh_pinn.affine_factor_pinn import AffineFactorPINN,riccati_residual,ratios
from src.mentor_dh_pinn.integrated_factor_pinn import IntegratedFactorPINN
from src.mentor_dh_pinn.conjugate_factor_pinn import ConjugateFactorPINN
from src.mentor_dh_pinn.moment_factor_pinn import MomentFactorPINN
from src.mentor_dh_pinn.two_moment_factor_pinn import TwoMomentFactorPINN
from src.mentor_dh_pinn.affine_factor_reference import draw_factor_points,reference_coefficients
from src.mentor_dh_pinn.torch_pricer import _heston_log_cf


def test_reference_matches_existing_cf_and_riccati():
    q=torch.tensor(draw_factor_points(1024,908231),dtype=torch.float64,requires_grad=True)
    d,a=reference_coefficients(q)
    tau,k,s,r,u,shift=q.unbind(-1);z=torch.complex(u,-shift)
    actual=k*.06*a+.04*d
    expected=_heston_log_cf(z,tau,k,torch.full_like(k,.06),s,r,torch.full_like(k,.04))
    torch.testing.assert_close(actual,expected,rtol=1e-9,atol=1e-10)
    residual=riccati_residual(reference_coefficients,q)
    assert residual.abs().max().item()<1e-9


@pytest.mark.parametrize('model_class', [AffineFactorPINN, IntegratedFactorPINN, ConjugateFactorPINN, MomentFactorPINN, TwoMomentFactorPINN])
def test_exact_initial_martingale_and_deterministic_limits(model_class):
    model=model_class(width=8,depth=2).double()
    q=torch.tensor(draw_factor_points(20,908232),dtype=torch.float64)
    at_zero=q.clone();at_zero[:,0]=0
    d,a=model.coefficients(at_zero)
    torch.testing.assert_close(d,torch.zeros_like(d),rtol=0,atol=0)
    torch.testing.assert_close(a,torch.zeros_like(a),rtol=0,atol=0)
    at_zero=q.clone();at_zero[:,4]=0
    d,a=model.coefficients(at_zero)
    torch.testing.assert_close(d,torch.zeros_like(d),rtol=0,atol=0)
    torch.testing.assert_close(a,torch.zeros_like(a),rtol=0,atol=0)
    deterministic=q.clone();deterministic[:,2]=0
    tau,k,_,_,u,shift=deterministic.unbind(-1);z=torch.complex(u,-shift)
    r,_=ratios(k*tau);integrated=tau*(.06+(.04-.06)*r)
    torch.testing.assert_close(model.log_cf(deterministic,.06,.04),-.5*(z*z+1j*z)*integrated)


@pytest.mark.parametrize('model_class', [AffineFactorPINN, IntegratedFactorPINN, ConjugateFactorPINN, MomentFactorPINN, TwoMomentFactorPINN])
def test_neural_physics_gradients_and_no_reference_in_forward(monkeypatch, model_class):
    import src.mentor_dh_pinn.affine_factor_reference as reference
    import src.mentor_dh_pinn.torch_pricer as pricer
    def forbidden(*a,**kw):raise AssertionError('Exact reference entered neural evaluation')
    monkeypatch.setattr(reference,'reference_coefficients',forbidden)
    monkeypatch.setattr(pricer,'_heston_log_cf',forbidden)
    model=model_class(width=8,depth=2).double()
    q=torch.tensor(draw_factor_points(8,908233),dtype=torch.float64,requires_grad=True)
    loss=riccati_residual(model.coefficients,q).square().mean()
    gradient=torch.autograd.grad(loss,model.head.weight)[0]
    assert torch.isfinite(gradient).all() and gradient.norm()>0
    before=model.log_cf(q,.06,.04).detach().clone()
    with torch.no_grad():model.head.bias.add_(.1)
    assert not torch.equal(before,model.log_cf(q,.06,.04).detach())
