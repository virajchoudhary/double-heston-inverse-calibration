"""First-cumulant-constrained primitive, still a learned factor PINN.

At unshifted u=0 the real normalized primitive must equal Q(kappa*tau).
Making its neural real correction O(u²), and the imaginary correction O(u),
preserves the exact expected integrated variance for EVERY parameter input.
Higher-frequency coefficients remain learned; no exact coefficient solver.
"""
import torch
from .affine_factor_pinn import AffineFactorPINN,scales
from .conjugate_factor_pinn import ConjugateFactorPINN,unshifted_coordinates


class MomentFactorPINN(ConjugateFactorPINN):
    def primitive(self,q):
        mapped=unshifted_coordinates(q)
        y=AffineFactorPINN.raw_corrections(self,mapped)
        _,_,frequency,_,base=scales(mapped)
        real_gate=frequency.square()/(1+frequency.square())
        imag_gate=frequency/(1+frequency)
        g=torch.complex(base*torch.exp(real_gate*y[...,0]),base*imag_gate*y[...,1])
        u=mapped[...,4];c=-.5*torch.complex(u*u,u)
        a=c*mapped[...,0]**2*g
        return torch.complex(a.real,a.imag*(1-2*q[...,5]))
