import numpy as np
import mlx.core as mx
import torch

from src.mentor_dh_pinn.regular_pinn import RegularVariancePINN
from src.mentor_dh_pinn.regular_pinn_data import coordinates, draw_points,teacher_labels
from src.mentor_dh_pinn.regular_pinn_torch import TorchRegularVariancePINN, residual
from src.mentor_dh_pinn.deep_regular_pinn import DeepRegularVariancePINN, TorchDeepRegularVariancePINN


def test_added_layers_preserve_initial_function_and_export_all_weights(tmp_path):
    mx.random.seed(90917)
    base=RegularVariancePINN(factors=2,width=12,depth=3)
    base.head.weight=mx.ones_like(base.head.weight)*.01
    mx.eval(base.parameters())
    path=tmp_path/'base.safetensors';base.save_weights(str(path))
    deep=DeepRegularVariancePINN(factors=2,width=12,depth=3,residual_blocks=2,residual_width=20)
    deep.load_weights(str(path),strict=False);mx.eval(deep.parameters())
    q=draw_points(8,2,90918);state,structural=coordinates(mx.array(q,dtype=mx.float32),2,mx)
    np.testing.assert_array_equal(np.asarray(base.iv(state,structural)),np.asarray(deep.iv(state,structural)))
    net=TorchRegularVariancePINN.from_mlx(deep)
    assert isinstance(net,TorchDeepRegularVariancePINN)
    assert net.hidden_layer_count==7
    c,p=coordinates(torch.tensor(q,dtype=torch.float64),2,torch)
    np.testing.assert_allclose(net.iv(c,p).detach().numpy(),np.asarray(deep.iv(state,structural)),rtol=2e-6,atol=2e-7)
    assert len(net.residual_blocks)==2


def test_deep_pde_loss_trains_added_layers_and_terminal_is_exact():
    torch.manual_seed(90919)
    net=TorchDeepRegularVariancePINN(factors=2,width=10,depth=3,residual_blocks=2,residual_width=16)
    q=torch.tensor(draw_points(4,2,90920),dtype=torch.float64)
    c,p=coordinates(q,2,torch)
    loss=residual(net,c,p)[0].square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(a.grad is not None and torch.isfinite(a.grad).all() for a in net.parameters())
    assert net.residual_blocks[0].down.weight.grad.abs().sum()>0
    c[:,-1]=0
    torch.testing.assert_close(net.price(c,p),torch.expm1(c[:,0]).clamp_min(0),rtol=0,atol=0)


def test_full_teacher_geometry_gradients_match_independent_finite_differences():
    q=draw_points(3,2,90921);q[:,0]=[.03,-.04,.06];q[:,1]=np.log([.15,.7,1.2])
    labels=teacher_labels(q,2,full_gradients=True)
    assert labels['usable'].all()
    np.testing.assert_allclose(labels['dg_dq'][:,2:],labels['dg_du'],rtol=1e-12,atol=1e-12)
    for j in [0,1]:
        a=q.copy();b=q.copy();a[:,j]+=1e-5;b[:,j]-=1e-5
        finite=(teacher_labels(a,2,gradients=False)['g']-teacher_labels(b,2,gradients=False)['g'])/2e-5
        np.testing.assert_allclose(labels['dg_dq'][:,j],finite,rtol=2e-5,atol=2e-8)


def test_deep_calibration_is_neural_only_and_does_not_read_holdouts(monkeypatch):
    from scripts.mentor_dh_pinn.assess_regular_pinn import fit_network
    import src.mentor_dh_pinn.regular_pinn_data as data
    torch.set_num_threads(1);torch.manual_seed(90922)
    net=TorchDeepRegularVariancePINN(factors=2,width=10,depth=2,residual_blocks=2,residual_width=16)
    net.requires_grad_(False)
    x=np.tile(np.linspace(-.15,.15,6),4);tau=np.repeat([.1,.3,.8,1.5],6)
    unit=np.full(10,.5)
    q=torch.tensor(np.column_stack([x,np.log(tau),np.broadcast_to(unit,(len(x),10))]),dtype=torch.float64)
    c,p=coordinates(q,2,torch);iv=net.iv(c,p).numpy();mask=np.arange(len(x))%3!=2
    def forbidden(*args,**kwargs):raise AssertionError('Exact pricing entered neural calibration')
    monkeypatch.setattr(data,'exact_prices',forbidden)
    a=fit_network(net,x,tau,iv,fit_mask=mask,starts=1,max_nfev=15)
    iv[~mask]=np.nan
    b=fit_network(net,x,tau,iv,fit_mask=mask,starts=1,max_nfev=15)
    assert a['status']=='fitted'
    np.testing.assert_array_equal(a['unit'],b['unit'])
