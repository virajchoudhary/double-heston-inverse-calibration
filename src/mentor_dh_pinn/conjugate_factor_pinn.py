"""Enforce the shifted/unshifted affine-coefficient identity exactly.

For real u, the Riccati equations imply
  A(u-i; k,s,r) = conj(A(u; k-r*s,s,-r)), and likewise for D.
This is an algebraic change of inputs, not an exact coefficient evaluation.
The log-CF prefactor remains the ORIGINAL kappa*theta in the pricing module.
"""
import torch
from .integrated_factor_pinn import IntegratedFactorPINN
from .affine_factor_pinn import AffineFactorPINN


def unshifted_coordinates(q):
    tau,kappa,sigma,rho,u,shift=q.unbind(-1)
    return torch.stack([tau,kappa-rho*sigma*shift,sigma,rho*(1-2*shift),u,
                        torch.zeros_like(shift)],-1)


class ConjugateFactorPINN(IntegratedFactorPINN):
    def primitive(self,q):
        a=super().primitive(unshifted_coordinates(q))
        return torch.complex(a.real,a.imag*(1-2*q[...,5]))


def build_factor_pinn(config):
    """Shared checkpoint architecture dispatch for training, fitting and diagnostics."""
    if config.get('two_moment',False):
        from .two_moment_factor_pinn import TwoMomentFactorPINN
        return TwoMomentFactorPINN(config['width'],config['depth']).double()
    if config.get('moment',False):
        from .moment_factor_pinn import MomentFactorPINN
        return MomentFactorPINN(config['width'],config['depth']).double()
    cls=ConjugateFactorPINN if config.get('conjugate',False) else (
        IntegratedFactorPINN if config.get('integrated',False) else AffineFactorPINN)
    return cls(config['width'],config['depth']).double()
