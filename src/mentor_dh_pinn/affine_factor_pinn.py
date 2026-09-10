"""Experimental factor-structured PINN, separate from the regular price PINN.

Learn D and A in log CF = sum(kappa*theta*A + v0*D). The learned
coefficients satisfy the Riccati equations D_t=.5*sigma²*D²-b*D+c,
A_t=D, b=kappa-rho*sigma*i*z, c=-(z²+i*z)/2, z=u-i*shift.
No exact coefficient/pricer or numerical ODE solver is called here.
"""
import math
import torch
from torch import nn


def ratios(a):
    """R=(1-exp(-a))/a and Q=(1-R)/a, including their zero limits."""
    small=a.abs()<1e-3
    safe=torch.where(small,torch.ones_like(a),a)
    r=torch.where(small,1-a/2+a*a/6-a**3/24+a**4/120,-torch.expm1(-a)/safe)
    g=torch.where(small,.5-a/6+a*a/24-a**3/120+a**4/720,(1-r)/safe)
    return r,g


def scales(q):
    """q=[tau,kappa,sigma,rho,u,shift]; u>=0, shift is 0 or 1."""
    tau,kappa,sigma,rho,u,shift=q.unbind(-1)
    a=(kappa-rho*sigma*shift)*tau
    beta=sigma*tau
    size=beta*torch.sqrt(u*u+1)/(1+a)
    frequency=beta*u/(1+a)
    damping=torch.rsqrt(1+frequency*frequency)
    r,g=ratios(a)
    return a,size,frequency,r*damping,g*damping


class AffineFactorPINN(nn.Module):
    def __init__(self,width=64,depth=4):
        super().__init__()
        self.width,self.depth=width,depth
        layers=[]
        for i in range(depth):layers.extend([nn.Linear(5 if i==0 else width,width),nn.Tanh()])
        self.hidden=nn.Sequential(*layers)
        self.head=nn.Linear(width,4)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

    def raw_corrections(self,q):
        a,size,_,_,_=scales(q)
        features=torch.stack([2*torch.log1p(a)/math.log(25)-1,
            2*torch.log1p(size)/math.log(1001)-1,
            2*torch.log1p(q[...,4])/math.log(501)-1,q[...,3]/.8,2*q[...,5]-1],-1)
        return self.head(self.hidden(features))

    def normalized_coefficients(self,q):
        _,size,frequency,base_h,base_g=scales(q)
        y=self.raw_corrections(q)
        real_gate=size/(1+size);imag_gate=frequency/(1+frequency)
        h=torch.complex(base_h*torch.exp(real_gate*y[...,0]),base_h*imag_gate*y[...,1])
        g=torch.complex(base_g*torch.exp(real_gate*y[...,2]),base_g*imag_gate*y[...,3])
        return h,g

    def coefficients(self,q):
        h,g=self.normalized_coefficients(q)
        z=torch.complex(q[...,4],-q[...,5])
        c=-.5*(z*z+1j*z);tau=q[...,0]
        return c*tau*h,c*tau*tau*g

    def log_cf(self,q,theta,v0):
        d,a=self.coefficients(q)
        return q[...,1]*theta*a+v0*d


def riccati_residual(coefficient_function,q):
    """Real four-component, scale-normalized residual; retains weight gradients."""
    if not q.requires_grad:raise ValueError('Collocation q must require gradients')
    d,a=coefficient_function(q)
    derivative=[torch.autograd.grad(v.sum(),q,create_graph=True,retain_graph=True)[0][...,0]
                for v in (d.real,d.imag,a.real,a.imag)]
    dt=torch.complex(derivative[0],derivative[1]);at=torch.complex(derivative[2],derivative[3])
    tau,kappa,sigma,rho,u,shift=q.unbind(-1)
    z=torch.complex(u,-shift);c=-.5*(z*z+1j*z);b=kappa-rho*sigma*1j*z
    rd=(dt-(.5*sigma*sigma*d*d-b*d+c))/(1+c.abs())
    ra=(at-d)/(1+c.abs()*tau)
    return torch.stack([rd.real,rd.imag,ra.real,ra.imag],-1)
