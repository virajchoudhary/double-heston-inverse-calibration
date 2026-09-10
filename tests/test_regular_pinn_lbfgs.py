import numpy as np
import mlx.core as mx
from src.mentor_dh_pinn.regular_pinn import RegularVariancePINN
from src.mentor_dh_pinn.regular_pinn_data import draw_points,teacher_labels
from scripts.mentor_dh_pinn.finetune_regular_lbfgs import FrozenObjective,directional_check


def test_frozen_objective_gradient_and_chunk_independence():
    mx.random.seed(901)
    model=RegularVariancePINN(2,width=10,depth=2)
    model.head.weight=.005*mx.random.normal(model.head.weight.shape)
    q=draw_points(8,2,813);labels=teacher_labels(q,2)
    scale=np.maximum(np.sqrt(np.mean(labels['dg_du']**2,axis=0)),.02)
    physics=draw_points(8,2,812,collocation=True)
    full=FrozenObjective(model,labels,physics,scale,chunk=8)
    chunks=FrozenObjective(model,labels,physics,scale,chunk=4)
    x=full.vector();a,ag=full(x);b,bg=chunks(x)
    np.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-7)
    np.testing.assert_allclose(ag,bg,rtol=2e-4,atol=1e-5)
    before=np.asarray(full.pw).copy()
    checks=directional_check(full,x)
    assert min(c['relative_error'] for c in checks)<.01
    full(x+.001)
    np.testing.assert_array_equal(before,np.asarray(full.pw))
