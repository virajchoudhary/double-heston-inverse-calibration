"""Learn higher-order coefficient corrections with the first two cumulants fixed.

The low-frequency coefficients follow by matching powers of u in the Riccati
equation. This is a finite moment backbone, NOT a full characteristic-function
formula or a pricing/ODE solver. Option values still require learned corrections.
"""
import torch
from .affine_factor_pinn import AffineFactorPINN,ratios,scales
from .conjugate_factor_pinn import unshifted_coordinates
from .moment_factor_pinn import MomentFactorPINN


def primitive_imaginary_slope(q):
    """g1 in A/(c*tau²)=Q(k*tau)+i*u*g1+O(u²), for unshifted q."""
    tau,k,s,r,_,_=q.unbind(-1);a=k*tau
    rr,qq=ratios(a);small=a.abs()<.02;safe=torch.where(small,torch.ones_like(a),a)
    j=(rr-torch.exp(-a))/safe
    difference=torch.where(small,a*(1/6+a*(-1/12+a*(1/40+a*(-1/180+a*(1/1008+a*(-1/6720+a/51840)))))),qq-j)
    combination=torch.where(small,a*a*(-1/12+a*(1/15+a*(-11/360+a*(13/1260+a*(-19/6720+a/1512))))),
                            -qq+2*j-.5*rr*rr)
    return r*s/k*difference+s*s/(4*k*k)*combination


class TwoMomentFactorPINN(MomentFactorPINN):
    def primitive(self,q):
        mapped=unshifted_coordinates(q)
        y=AffineFactorPINN.raw_corrections(self,mapped)
        _,_,frequency,_,base=scales(mapped)
        square=frequency.square();real_gate=square/(1+square)
        imag_gate=frequency**3/(1+square)**1.5
        u=mapped[...,4]
        imaginary=primitive_imaginary_slope(mapped)*u/(1+square)+base*imag_gate*y[...,1]
        g=torch.complex(base*torch.exp(real_gate*y[...,0]),imaginary)
        a=-.5*torch.complex(u*u,u)*mapped[...,0]**2*g
        return torch.complex(a.real,a.imag*(1-2*q[...,5]))
