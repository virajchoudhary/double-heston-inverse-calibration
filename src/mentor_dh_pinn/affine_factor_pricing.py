"""Fourier quadrature of LEARNED factor coefficients, not an exact DH pricer.

Expiry compression is for parameter calibration with fixed observed geometry;
do not use it to differentiate option prices with respect to maturity.
"""
from functools import lru_cache
import math
import numpy as np
import torch


@lru_cache(maxsize=4)
def quadrature(nodes):
    u,w=np.polynomial.laguerre.laggauss(nodes)
    return torch.tensor(u,dtype=torch.float64),torch.tensor(w*np.exp(u),dtype=torch.float64)


def neural_call_prices(model,physical,x,tau,*,nodes=128):
    """Normalized forward calls C/(K*discount), canonical 5/10 physical params."""
    if physical.ndim!=1 or physical.numel() not in (5,10):
        raise ValueError('One canonical Single/Double Heston parameter vector required')
    if x.ndim!=1 or x.shape!=tau.shape:raise ValueError('Fixed x/tau must be matching vectors')
    if tau.requires_grad:raise ValueError('Expiry compression is for fixed-geometry parameter calibration only')
    u,weights=(v.to(physical) for v in quadrature(nodes))
    times,index=torch.unique(tau,sorted=True,return_inverse=True)
    p=physical.reshape(-1,5)
    tt,uu,shift,kk,ss,rr=torch.broadcast_tensors(times[:,None,None,None],
        u[None,None,:,None],torch.tensor([0.,1.],dtype=physical.dtype,device=physical.device)[None,:,None,None],
        p[None,None,None,:,0],p[None,None,None,:,2],p[None,None,None,:,3])
    q=torch.stack([tt,kk,ss,rr,uu,shift],-1)
    d,a=model.coefficients(q.reshape(-1,6))
    d,a=d.reshape(q.shape[:-1]),a.reshape(q.shape[:-1])
    exponent=(kk*p[None,None,None,:,1]*a+p[None,None,None,:,4]*d).sum(-1)
    cf=torch.exp(exponent)[index]
    oscillation=torch.exp(1j*x[:,None]*u[None,:])/(1j*u[None,:])
    probability=.5+(weights[None,None,:]*(oscillation[:,None,:]*cf).real).sum(-1)/math.pi
    call=torch.exp(x)*probability[:,1]-probability[:,0]
    return torch.where(tau==0,torch.clamp_min(torch.expm1(x),0),call)
