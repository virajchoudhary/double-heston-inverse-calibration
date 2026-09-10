"""Factor-PINN refinement with A_tau=D enforced by differentiation.

Only the complex primitive A is learned; D is its automatic time derivative.
There is no ODE solver or exact coefficient call in inference. This removes
independent approximation errors between the two affine coefficient functions.
"""
import torch
from torch import nn
from .affine_factor_pinn import AffineFactorPINN,scales


class IntegratedFactorPINN(AffineFactorPINN):
    def __init__(self,width=64,depth=4):
        super().__init__(width,depth)
        self.head=nn.Linear(width,2)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

    def load_primitive_from(self,independent_model):
        """Warm start A only, not a calibration parameter vector."""
        self.hidden.load_state_dict(independent_model.hidden.state_dict())
        with torch.no_grad():
            self.head.weight.copy_(independent_model.head.weight[2:])
            self.head.bias.copy_(independent_model.head.bias[2:])

    def primitive(self,q):
        y=super().raw_corrections(q)
        _,size,frequency,_,base=scales(q)
        rg=size/(1+size);ig=frequency/(1+frequency)
        g=torch.complex(base*torch.exp(rg*y[...,0]),base*ig*y[...,1])
        z=torch.complex(q[...,4],-q[...,5]);c=-.5*(z*z+1j*z)
        return c*q[...,0]**2*g

    def coefficients(self,q):
        tangent=torch.zeros_like(q);tangent[...,0]=1.
        a,d=torch.func.jvp(self.primitive,(q,),(tangent,))
        return d,a

    def normalized_coefficients(self,q):
        d,a=self.coefficients(q)
        z=torch.complex(q[...,4],-q[...,5]);c=-.5*(z*z+1j*z)
        return d/(c*q[...,0]),a/(c*q[...,0]**2)

    def raw_corrections(self,q):
        """Linear (not logarithmic) normalized features for positive tau/u.

        Negative derivative coefficients stay finite and penalizable in training;
        they are never clipped into apparently valid option prices.
        """
        h,g=self.normalized_coefficients(q)
        _,size,frequency,bh,bg=scales(q)
        rg=size/(1+size);ig=frequency/(1+frequency)
        return torch.stack([(h.real/bh-1)/rg,h.imag/bh/ig,
                            (g.real/bg-1)/rg,g.imag/bg/ig],-1)


def linear_coefficient_targets(q,log_targets):
    """Re-express the same reference labels, without changing their information."""
    _,size,_,_,_=scales(q);rg=size/(1+size)
    return torch.stack([torch.expm1(rg*log_targets[...,0])/rg,log_targets[...,1],
                        torch.expm1(rg*log_targets[...,2])/rg,log_targets[...,3]],-1)
