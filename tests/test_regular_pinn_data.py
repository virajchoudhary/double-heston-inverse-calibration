"""Reference sensitivity and float64 export checks, independent of training."""
import numpy as np
import torch
import mlx.core as mx
from src.mentor_dh_pinn.regular_pinn_data import draw_points,teacher_labels,coordinates
from src.mentor_dh_pinn.regular_pinn import RegularVariancePINN
from src.mentor_dh_pinn.regular_pinn_torch import TorchRegularVariancePINN


def test_implicit_iv_parameter_derivative_matches_reference_difference():
    torch.set_num_threads(1)
    q=draw_points(8,2,713);labels=teacher_labels(q,2)
    assert labels["usable"].all()
    for j in (2,5,8,11):
        plus,minus=q.copy(),q.copy();h=2e-4
        plus[:,j]+=h;minus[:,j]-=h
        fd=(teacher_labels(plus,2,gradients=False)["g"]-teacher_labels(minus,2,gradients=False)["g"])/(2*h)
        np.testing.assert_allclose(fd,labels["dg_du"][:,j-2],atol=2e-6,rtol=2e-3)


def test_float64_copy_has_same_prices_and_correct_parameter_jacobian():
    mx.random.seed(71);model=RegularVariancePINN(2,width=16,depth=2)
    model.head.weight=.1*mx.random.normal(model.head.weight.shape)
    q=draw_points(12,2,815)
    c,p=coordinates(mx.array(q,dtype=mx.float32),2,mx)
    a=np.asarray(model.iv(c,p))
    copied=TorchRegularVariancePINN.from_mlx(model)
    def forward(qt):
        c,p=coordinates(qt,2,torch);return copied.iv(c,p)
    qt=torch.tensor(q,dtype=torch.float64,requires_grad=True)
    b=forward(qt)
    np.testing.assert_allclose(b.detach().numpy(),a,atol=2e-6,rtol=5e-6)
    jac=torch.autograd.grad(b.sum(),qt)[0].detach().numpy()
    h=1e-5;plus=q.copy();minus=q.copy();plus[:,6]+=h;minus[:,6]-=h
    fd=((forward(torch.tensor(plus))-forward(torch.tensor(minus)))/(2*h)).detach().numpy()
    np.testing.assert_allclose(fd,jac[:,6],atol=1e-8,rtol=1e-6)
