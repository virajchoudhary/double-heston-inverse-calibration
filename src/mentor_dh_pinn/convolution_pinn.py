"""Experimental Double Heston composition of a single-factor price-PDE PINN.

For independent martingale log returns Y1,Y2, c12(x)=E[c1(x+Y2)].
The second factor's log-return density is c2_xx(-y)-c2_x(-y).
Only learned prices, automatic differentiation and fixed quadrature are used.
No exact Heston engine, density clipping or mass renormalization is permitted.
"""
import math

import numpy as np
import torch
from torch import nn

from .regular_pinn_data import invert_total_variance
from .regular_pinn_torch import expected_variance


class ImplicitBlackIV(torch.autograd.Function):
    """Black inversion with its implicit first derivative, not bisection gradients."""
    @staticmethod
    def forward(ctx, price, x, tau):
        w = invert_total_variance(price.detach().numpy(), x.detach().numpy())
        iv = torch.as_tensor(np.sqrt(w/tau.detach().numpy()),dtype=price.dtype)
        if not torch.isfinite(iv).all():
            raise FloatingPointError('Noninvertible composed neural price; no clipping/fallback')
        root=iv*torch.sqrt(tau);d1=x/root+.5*root;d2=d1-root
        vega=torch.exp(-.5*d2*d2)*torch.sqrt(tau)/math.sqrt(2*math.pi)
        dx=torch.exp(x)*.5*torch.erfc(-d1/math.sqrt(2))
        ctx.save_for_backward(iv,vega,dx,tau)
        return iv

    @staticmethod
    def backward(ctx, grad):
        iv,vega,dx,tau=ctx.saved_tensors
        return grad/vega,-grad*dx/vega,-grad*iv/(2*tau)


class ConvolutionPINN(nn.Module):
    """Two uses of shared single-factor neural weights; ten inverse parameters."""
    factors=2

    def __init__(self,component,nodes=96):
        super().__init__()
        if component.factors!=1 or nodes<8:
            raise ValueError('A single-factor price PINN and at least eight nodes are required')
        self.component=component;self.nodes=nodes
        points,weights=np.polynomial.hermite.hermgauss(nodes)
        self.register_buffer('quadrature_points',torch.tensor(points,dtype=torch.float64))
        self.register_buffer('quadrature_weights',torch.tensor(weights*np.exp(points**2),dtype=torch.float64))

    @torch.enable_grad()
    def price_and_diagnostics(self,coords,structural):
        x,v1,v2,tau=coords.unbind(-1)
        if not torch.isfinite(tau).all() or (tau<=0).any():
            raise ValueError('Density diagnostics require finite positive maturity; use price() for terminal payoff')
        p1,p2=structural[...,0,:],structural[...,1,:]
        c2=torch.stack([torch.zeros_like(x),v2,tau],-1)
        variance=expected_variance(c2,p2[...,None,:])*tau
        # Integration coordinates depend differentiably on factor-two parameters.
        spread=torch.sqrt(2*variance)
        y=-.5*variance[...,None]+spread[...,None]*self.quadrature_points
        negative_y=-y
        if not negative_y.requires_grad:negative_y=negative_y.detach().requires_grad_(True)
        expand=lambda a:a[...,None].expand_as(y)
        states2=torch.stack([negative_y,expand(v2),expand(tau)],-1)
        states1=torch.stack([x[...,None]+y,expand(v1),expand(tau)],-1)
        shape=y.shape
        factors2=p2[...,None,:].expand(*shape,4).reshape(-1,1,4)
        factors1=p1[...,None,:].expand(*shape,4).reshape(-1,1,4)
        call2=self.component.price(states2.reshape(-1,3),factors2).reshape(shape)
        first=torch.autograd.grad(call2.sum(),negative_y,create_graph=True)[0]
        second=torch.autograd.grad(first.sum(),negative_y,create_graph=True)[0]
        density=second-first
        integration=spread[...,None]*self.quadrature_weights
        call1=self.component.price(states1.reshape(-1,3),factors1).reshape(shape)
        value=(integration*density*call1).sum(-1)
        return value,{'mass':(integration*density).sum(-1),
                      'martingale_moment':(integration*density*torch.exp(y)).sum(-1),
                      'minimum_density':density.min(-1).values}

    def price(self,coords,structural):
        tau=coords[...,-1]
        if not torch.isfinite(tau).all() or (tau<0).any():
            raise ValueError('Price requires finite nonnegative maturity')
        positive=tau>0
        if positive.all():
            return self.price_and_diagnostics(coords,structural)[0]
        # At expiry the return law is a point mass, not a finite density.
        safe=torch.cat([coords[...,:-1],torch.where(positive,tau,torch.ones_like(tau))[...,None]],-1)
        value=self.price_and_diagnostics(safe,structural)[0]
        return torch.where(positive,value,torch.clamp_min(torch.expm1(coords[...,0]),0))

    def iv(self,coords,structural):
        if not torch.isfinite(coords[...,-1]).all() or (coords[...,-1]<=0).any():
            raise ValueError('Implied volatility is defined here only at finite positive maturity')
        return ImplicitBlackIV.apply(self.price(coords,structural),coords[...,0],coords[...,-1])
