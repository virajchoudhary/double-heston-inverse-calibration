import numpy as np
import torch
from src.mentor_dh_pinn.affine_factor_pinn import AffineFactorPINN,riccati_residual
from src.mentor_dh_pinn.affine_factor_reference import draw_factor_points,correction_targets
from src.mentor_dh_pinn.integrated_factor_pinn import IntegratedFactorPINN,linear_coefficient_targets
from src.mentor_dh_pinn.affine_factor_pricing import neural_call_prices
from src.mentor_dh_pinn.regular_pinn_data import decode_unit


def test_primitive_identity_physics_and_nested_parameter_jacobian():
    torch.set_num_threads(1)
    model=IntegratedFactorPINN(width=8,depth=2).double()
    q=torch.tensor(draw_factor_points(6,908301),dtype=torch.float64,requires_grad=True)
    d,a=model.coefficients(q)
    ad=torch.complex(torch.autograd.grad(a.real.sum(),q,create_graph=True,retain_graph=True)[0][:,0],
                     torch.autograd.grad(a.imag.sum(),q,create_graph=True,retain_graph=True)[0][:,0])
    torch.testing.assert_close(d,ad,atol=1e-11,rtol=1e-12)
    loss=riccati_residual(model.coefficients,q).square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.requires_grad_(False)
    x=torch.tensor([-.05,0.,.05],dtype=torch.float64);tau=torch.full_like(x,.4)
    unit=torch.full((10,),.4,dtype=torch.float64)
    fun=lambda u:neural_call_prices(model,decode_unit(u,2,torch),x,tau)
    jac=torch.func.jacfwd(fun)(unit)
    bump=torch.zeros_like(unit);bump[6]=1e-5
    torch.testing.assert_close(jac[:,6],(fun(unit+bump)-fun(unit-bump))/(2e-5),rtol=1e-4,atol=1e-8)


def test_primitive_warm_start_and_reference_feature_conversion():
    base=AffineFactorPINN(width=8,depth=2).double()
    with torch.no_grad():base.head.bias.copy_(torch.tensor([.01,.02,.03,.04]))
    model=IntegratedFactorPINN(width=8,depth=2).double();model.load_primitive_from(base)
    q=torch.tensor(draw_factor_points(10,908302),dtype=torch.float64)
    torch.testing.assert_close(model.primitive(q),base.coefficients(q)[1])
    labels=linear_coefficient_targets(q,correction_targets(q))
    assert torch.isfinite(labels).all()
    zero=q.clone();zero[:,0]=0
    d,a=model.coefficients(zero)
    torch.testing.assert_close(d,torch.zeros_like(d),atol=0,rtol=0)
    torch.testing.assert_close(a,torch.zeros_like(a),atol=0,rtol=0)
