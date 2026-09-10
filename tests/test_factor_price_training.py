import numpy as np
import pytest
import torch
from scripts.mentor_dh_pinn.finetune_factor_prices import load_surfaces,surface_loss
from src.mentor_dh_pinn.two_moment_factor_pinn import TwoMomentFactorPINN


def test_price_weight_gradient_and_reserved_surface_isolation(tmp_path,monkeypatch):
    torch.set_num_threads(1);torch.manual_seed(909101)
    unit=np.full((4,10),.4)+np.arange(4)[:,None]*.01
    x=np.array([-.06,0.,.06]);tau=np.full(3,.5)
    q=np.stack([np.column_stack([x,np.log(tau),np.broadcast_to(u,(3,10))]) for u in unit])
    path=tmp_path/'surfaces.npz'
    np.savez(path,q=q,iv=np.full((4,3),.3),unit=unit,usable=np.ones(4,bool))
    surfaces=load_surfaces(path,1)
    model=TwoMomentFactorPINN(width=8,depth=2).double()
    def forbidden(*a,**kw):raise AssertionError('Exact pricer in learned-price loss')
    for name in ('src.mentor_dh_pinn.torch_pricer.price_call',
                 'src.mentor_dh_pinn.regular_pinn_data.price_call',
                 'src.mentor_dh_pinn.affine_factor_reference.reference_coefficients'):
        monkeypatch.setattr(name,forbidden)
    def value_gradient():
        model.zero_grad(set_to_none=True);loss=surface_loss(model,surfaces,[0,1]);loss.backward()
        return float(loss.detach()),torch.cat([p.grad.flatten() for p in model.parameters()]).numpy().copy()
    value,gradient=value_gradient();assert np.isfinite(gradient).all() and np.linalg.norm(gradient)>0
    surfaces['target'][3]=float('nan');surfaces['physical'][3]=float('nan')
    other,other_gradient=value_gradient()
    assert other==value;np.testing.assert_array_equal(gradient,other_gradient)
    initial=torch.nn.utils.parameters_to_vector(model.parameters()).detach()
    direction=torch.tensor(gradient/np.linalg.norm(gradient));h=1e-5;values=[]
    for sign in (1,-1):
        torch.nn.utils.vector_to_parameters(initial+sign*h*direction,model.parameters())
        values.append(float(surface_loss(model,surfaces,[0,1]).detach()))
    np.testing.assert_allclose((values[0]-values[1])/(2*h),gradient@direction.numpy(),rtol=1e-5)
    unit[3]=unit[0];q[3]=q[0]
    np.savez(path,q=q,iv=np.full((4,3),.3),unit=unit,usable=np.ones(4,bool))
    with pytest.raises(ValueError,match='Duplicate'):load_surfaces(path,1)
