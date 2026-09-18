"""Small checks for the separate fixed-truth pilot's most important guarantees."""
import ast
import inspect
import numpy as np
import torch
from scripts.mentor_dh_pinn import literature_fixed_truth_pilot as pilot


def test_constraints_gradients_and_same_start_for_every_case():
    raw = pilot.initial_raw(17).requires_grad_(True)
    p = pilot.decode(raw)
    assert p[0] < p[5]
    assert np.all(pilot.feller(p.detach().numpy()) > 0)
    assert torch.equal(raw.detach(), pilot.initial_raw(17))
    assert (p[[0, 1, 2, 4, 5, 6, 7, 9]] > 0).all()
    assert (p[[3, 8]].abs() < 1).all()
    model = pilot.TorchRegularVariancePINN(width=16, depth=2)
    xt = torch.tensor([[-.1, .2], [.1, 1.]])
    coords, structural = pilot.data_coords(xt, p)
    res, _ = pilot.residual(model, coords, structural)
    (model.iv(coords, structural).square().mean() + res.square().mean()).backward()
    assert torch.isfinite(raw.grad).all() and (raw.grad.abs() > 0).all()
    assert all(torch.isfinite(w.grad).all() for w in model.parameters())
    payoff_coords = coords.detach().clone()
    payoff_coords[:, -1] = 0
    expected = torch.expm1(xt[:, 0]).clamp_min(0)
    torch.testing.assert_close(model.price(payoff_coords, structural), expected)


def test_training_function_has_no_truth_holdout_or_pricer_access():
    # Static guard complements separate train process and allow-listed NPZ keys.
    tree = ast.parse(inspect.getsource(pilot.train))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not names & {"teacher", "price_call", "score", "truth"}
    strings = [node.value for node in ast.walk(tree)
               if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    assert not any(s.endswith("holdout.npz") or s.endswith("truth_and_sources.json") for s in strings)


def test_stated_truths_and_published_price_reference():
    import json
    cfg = json.loads((pilot.ROOT / "configs/literature_pinn_pilot.json").read_text())
    for case in cfg["cases"]:
        assert len(case["parameters"]) == 10
        assert np.all(pilot.feller(case["parameters"]) > 0)
    assert np.all(pilot.feller(cfg["excluded"][0]["parameters"]) < 0)
    p = cfg["cases"][0]["parameters"]
    computed = pilot.teacher(p, np.array([[.03, 1.]]), 128)[0] * 100 * np.exp(-.03)
    assert abs(computed - 26.9504) < .00005
