import torch
from src.mentor_dh_pinn.affine_factor_pinn import riccati_residual
from src.mentor_dh_pinn.affine_factor_reference import draw_factor_points,reference_coefficients
from src.mentor_dh_pinn.moment_factor_pinn import MomentFactorPINN
from src.mentor_dh_pinn.two_moment_factor_pinn import TwoMomentFactorPINN


def second_cumulant_error(model):
    q=torch.tensor(draw_factor_points(64,909021),dtype=torch.float64);q[:,4]=0;q.requires_grad_(True)
    d,a=reference_coefficients(q);truth=q[:,1]*.06*a+.04*d
    actual=model.log_cf(q,.06,.04)
    def second(value):
        first=torch.autograd.grad(value.real.sum(),q,create_graph=True,retain_graph=True)[0][:,4]
        return torch.autograd.grad(first.sum(),q,retain_graph=True)[0][:,4]
    return second(actual)-second(truth)


def test_second_cumulant_is_independent_of_neural_weights():
    torch.set_num_threads(1);torch.manual_seed(86)
    original=MomentFactorPINN(width=8,depth=2).double()
    with torch.no_grad():original.head.bias.fill_(.1)
    model=TwoMomentFactorPINN(width=8,depth=2).double();model.load_state_dict(original.state_dict())
    assert second_cumulant_error(original).abs().max()>1e-5
    torch.testing.assert_close(second_cumulant_error(model),torch.zeros(64,dtype=torch.float64),atol=1e-10,rtol=0)
    with torch.no_grad():model.head.weight.normal_(0,.2);model.head.bias.fill_(-.3)
    torch.testing.assert_close(second_cumulant_error(model),torch.zeros(64,dtype=torch.float64),atol=1e-10,rtol=0)
    q=torch.tensor(draw_factor_points(12,909022),dtype=torch.float64,requires_grad=True)
    riccati_residual(model.coefficients,q).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
