import torch
import pytest
from src.mentor_dh_pinn.affine_factor_pinn import riccati_residual,ratios
from src.mentor_dh_pinn.affine_factor_reference import draw_factor_points
from src.mentor_dh_pinn.moment_factor_pinn import MomentFactorPINN
from src.mentor_dh_pinn.two_moment_factor_pinn import TwoMomentFactorPINN
from src.mentor_dh_pinn.conjugate_factor_pinn import ConjugateFactorPINN


def mean_error(model):
    q=torch.tensor(draw_factor_points(32,908541),dtype=torch.float64)
    q[:,4]=0;q[:,5]=0;q.requires_grad_(True)
    theta,v0=.06,.04
    logcf=model.log_cf(q,theta,v0)
    slope=torch.autograd.grad(logcf.imag.sum(),q)[0][:,4]
    tau,k=q[:,0],q[:,1]
    expected=-.5*tau*(theta+(v0-theta)*ratios(k*tau)[0])
    return slope-expected


@pytest.mark.parametrize('model_class',[MomentFactorPINN,TwoMomentFactorPINN])
def test_expected_integrated_variance_is_exact_for_arbitrary_neural_weights(model_class):
    torch.set_num_threads(1);torch.manual_seed(85)
    original=ConjugateFactorPINN(width=8,depth=2).double()
    with torch.no_grad():original.head.bias.fill_(.1)
    model=model_class(width=8,depth=2).double();model.load_state_dict(original.state_dict())
    assert mean_error(original).abs().max()>1e-5
    torch.testing.assert_close(mean_error(model),torch.zeros(32,dtype=torch.float64),atol=1e-13,rtol=0)
    with torch.no_grad():model.head.weight.normal_(0,.2);model.head.bias.fill_(-.3)
    torch.testing.assert_close(mean_error(model),torch.zeros(32,dtype=torch.float64),atol=1e-13,rtol=0)
    q=torch.tensor(draw_factor_points(12,908542),dtype=torch.float64,requires_grad=True)
    loss=riccati_residual(model.coefficients,q).square().mean();loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
