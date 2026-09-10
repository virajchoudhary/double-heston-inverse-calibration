import json

import numpy as np
import pytest
import torch

from scripts.mentor_dh_pinn.assess_regular_pinn import load_checkpoint
from scripts.mentor_dh_pinn.finetune_regular_torch import (
    anchor_loss, physics_loss, relevance_weights, tensor,
)
from src.mentor_dh_pinn.regular_pinn_data import draw_points
from src.mentor_dh_pinn.regular_pinn_torch import TorchRegularVariancePINN


def test_float64_checkpoint_keeps_sub_float32_precision(tmp_path):
    model = TorchRegularVariancePINN(width=8, depth=2)
    exact = 1.0 + 2.0**-40
    with torch.no_grad():
        model.head.bias.fill_(exact)
    cfg = {'width': 8, 'depth': 2, 'checkpoint_format': 'torch_state_dict_float64'}
    (tmp_path/'config.json').write_text(json.dumps(cfg))
    torch.save(model.state_dict(), tmp_path/'model.pt')
    loaded, _ = load_checkpoint(tmp_path)
    assert loaded.head.bias.item() == exact
    assert loaded.head.bias.item() != float(np.float32(exact))
    assert all(p.dtype == torch.float64 and not p.requires_grad for p in loaded.parameters())
    torch.save({k: v.float() for k, v in model.state_dict().items()}, tmp_path/'model.pt')
    with pytest.raises(ValueError, match='float64'):
        load_checkpoint(tmp_path)


@pytest.mark.parametrize('extra_blocks', [0, 6])
def test_double_precision_training_gradient_and_chunk_normalization(monkeypatch, extra_blocks):
    import scripts.mentor_dh_pinn.assess_regular_pinn as assessor
    import src.mentor_dh_pinn.regular_pinn_data as data
    import src.mentor_dh_pinn.torch_pricer as pricer

    def forbidden(*args, **kwargs):
        raise AssertionError('No exact pricing during archived-label/PDE training')

    for module, name in ((assessor, '_exact'), (assessor, 'exact_prices'),
                         (data, 'exact_prices'), (data, 'teacher_labels'),
                         (pricer, 'price_call'), (pricer, 'price_call_single')):
        monkeypatch.setattr(module, name, forbidden)
    torch.manual_seed(937)
    model = TorchRegularVariancePINN(width=8, depth=2)
    if extra_blocks:
        model = model.deepened(extra_blocks)
        # Check derivatives away from the identity initialization as well.
        with torch.no_grad():
            for _, last in model.extra:
                last.weight.normal_(0, .01)
    q = tensor(draw_points(8, 2, 939))
    pq = tensor(draw_points(8, 2, 941, collocation=True))
    # Numerical test labels only; this is not evidence of synthetic recovery.
    g, dg, scale = torch.zeros(8, dtype=torch.float64), torch.zeros((8, 10), dtype=torch.float64), torch.ones(10, dtype=torch.float64)
    pw = relevance_weights(model, pq)
    frozen = pw.clone()

    def loss(sl=slice(None)):
        return (anchor_loss(model, q[sl], g[sl], dg[sl], scale)
                + physics_loss(model, pq[sl], pw[sl], pw.mean()))

    full = loss()
    params = tuple(model.parameters())
    gradient = torch.autograd.grad(full, params)
    chunks = .5*(loss(slice(0, 4))+loss(slice(4, 8)))
    chunk_gradient = torch.autograd.grad(chunks, params)
    torch.testing.assert_close(full, chunks, rtol=1e-12, atol=1e-12)
    for a, b in zip(gradient, chunk_gradient):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)
        assert torch.isfinite(a).all()
    norm = torch.sqrt(sum((v*v).sum() for v in gradient))
    direction = [v/norm for v in gradient]
    original = [p.detach().clone() for p in params]

    def evaluate(delta):
        with torch.no_grad():
            for p, base, d in zip(params, original, direction):
                p.copy_(base+delta*d)
        return float(loss().detach())

    h = 1e-6
    fd = (evaluate(h)-evaluate(-h))/(2*h)
    np.testing.assert_allclose(fd, float(norm), rtol=2e-7, atol=1e-7)
    evaluate(0)
    torch.testing.assert_close(frozen, pw, rtol=0, atol=0)


def test_deepening_preserves_function_pde_and_checkpoint(tmp_path):
    from src.mentor_dh_pinn.regular_pinn_data import coordinates
    from src.mentor_dh_pinn.regular_pinn_torch import residual
    torch.manual_seed(709)
    shallow = TorchRegularVariancePINN(width=12, depth=5)
    deep = shallow.deepened(6)
    assert deep.total_hidden_layers == 17 and shallow.total_hidden_layers == 5
    assert sum(p.numel() for p in deep.parameters()) == sum(p.numel() for p in shallow.parameters())+12*(12*12+12)
    q = tensor(draw_points(8, 2, 715))
    c, p = coordinates(q, 2, torch)
    torch.testing.assert_close(deep.iv(c,p), shallow.iv(c,p), rtol=0, atol=0)
    torch.testing.assert_close(residual(deep,c,p)[0], residual(shallow,c,p)[0], rtol=0, atol=0)
    loss = anchor_loss(deep,q,torch.zeros(8),torch.zeros((8,10)),torch.ones(10))
    loss.backward()
    assert all(torch.isfinite(v.grad).all() for v in deep.parameters())
    assert all(float(last.weight.grad.norm())>0 for _,last in deep.extra)
    cfg = {'width':12, 'depth':5, 'residual_blocks':6, 'checkpoint_format':'torch_state_dict_float64'}
    (tmp_path/'config.json').write_text(json.dumps(cfg))
    torch.save(deep.state_dict(),tmp_path/'model.pt')
    loaded,_ = load_checkpoint(tmp_path)
    assert loaded.total_hidden_layers==17
    torch.testing.assert_close(deep.iv(c,p),loaded.iv(c,p),rtol=0,atol=0)
