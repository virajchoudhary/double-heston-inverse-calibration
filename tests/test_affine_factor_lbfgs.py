import numpy as np
import torch
import pytest
from src.mentor_dh_pinn.affine_factor_pinn import AffineFactorPINN
from src.mentor_dh_pinn.integrated_factor_pinn import IntegratedFactorPINN,linear_coefficient_targets
from src.mentor_dh_pinn.conjugate_factor_pinn import ConjugateFactorPINN
from src.mentor_dh_pinn.moment_factor_pinn import MomentFactorPINN
from src.mentor_dh_pinn.two_moment_factor_pinn import TwoMomentFactorPINN
from src.mentor_dh_pinn.affine_factor_reference import draw_factor_points,correction_targets
from scripts.mentor_dh_pinn.finetune_affine_factor_lbfgs import FrozenFactorObjective,gradient_check


@pytest.mark.parametrize('model_class', [AffineFactorPINN, IntegratedFactorPINN, ConjugateFactorPINN, MomentFactorPINN, TwoMomentFactorPINN])
def test_float64_fixed_loss_gradient_and_chunk_invariance(model_class):
    torch.manual_seed(908291)
    model=model_class(width=8,depth=2).double()
    q=torch.tensor(draw_factor_points(8,908292),dtype=torch.float64)
    target=correction_targets(q)
    if isinstance(model,IntegratedFactorPINN):target=linear_coefficient_targets(q,target)
    pq=torch.tensor(draw_factor_points(8,908293),dtype=torch.float64)
    full=FrozenFactorObjective(model,(q,target),pq,8)
    chunks=FrozenFactorObjective(model,(q,target),pq,3)
    x=full.vector();a,ga=full(x);b,gb=chunks(x)
    np.testing.assert_allclose(a,b,atol=1e-14)
    np.testing.assert_allclose(ga,gb,atol=1e-12)
    assert min(r['relative_error'] for r in gradient_check(full,x))<1e-6
