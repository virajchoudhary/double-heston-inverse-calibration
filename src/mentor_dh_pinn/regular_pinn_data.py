"""Explicit synthetic domain and independent price/sensitivity labels for regular PINNs.

q = [log(F/K), log(tau), unit parameter coordinates]. Canonical physical parameters
are [kappa, theta, sigma, rho, v0] per factor. Bounds describe this experiment's
training domain, not the whole admissible Heston model class.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from scipy.special import ndtr
from scipy.stats import qmc

from .torch_pricer import price_call, price_call_single

DOMAIN = {
    "tau_years": [7/365, 2.], "single_kappa": [.2, 12.], "slow_kappa": [.2, 3.],
    "fast_kappa": "slow + 0.5 + (11.5 - slow) * u; maximum 12",
    "total_theta_and_v0": [.03, .30], "factor_shares": [.2, .8],
    "feller_ratio": [.15, .90], "rho_slow": [-.8, .3], "rho_fast": [-.6, .5],
    "sampling": "Latin hypercube; log scale for speeds and total variances; linear shares, correlations and eta",
}


def decode_unit(u, factors, lib=np):
    """Bounded differentiable map; unit inputs are not silently clipped."""
    logbox=lambda j,lo,hi:lo*lib.exp(u[...,j]*math.log(hi/lo))
    if factors==1:
        k=logbox(0,.2,12.);theta=logbox(1,.03,.3)
        eta=.15+.75*u[...,2];rho=-.8+1.1*u[...,3];v=logbox(4,.03,.3)
        values=[k,theta,eta*lib.sqrt(2*k*theta),rho,v]
    elif factors==2:
        ks=logbox(0,.2,3.);kf=ks+.5+(11.5-ks)*u[...,1]
        theta=logbox(2,.03,.3);share=.2+.6*u[...,3]
        v=logbox(4,.03,.3);vshare=.2+.6*u[...,5]
        ts,tf=theta*share,theta*(1-share)
        ss=(.15+.75*u[...,6])*lib.sqrt(2*ks*ts)
        sf=(.15+.75*u[...,7])*lib.sqrt(2*kf*tf)
        values=[ks,ts,ss,-.8+1.1*u[...,8],v*vshare,
                kf,tf,sf,-.6+1.1*u[...,9],v*(1-vshare)]
    else:raise ValueError("Only one or two factors are supported")
    return lib.stack(values,**({"dim":-1} if lib is torch else {"axis":-1}))


def coordinates(q, factors, lib=np):
    p=decode_unit(q[...,2:],factors,lib)
    stack=lambda v:lib.stack(v,**({"dim":-1} if lib is torch else {"axis":-1}))
    state=stack([q[...,0],*[p[...,5*i+4] for i in range(factors)],lib.exp(q[...,1])])
    structural=lib.stack([p[...,5*i:5*i+4] for i in range(factors)],
                         **({"dim":-2} if lib is torch else {"axis":-2}))
    return state,structural


def expected_variance(q,factors,lib=np):
    p=decode_unit(q[...,2:],factors,lib);tau=lib.exp(q[...,1]);total=0.
    for i in range(factors):
        k,t,v=p[...,5*i],p[...,5*i+1],p[...,5*i+4]
        a=k*tau
        # All training a >= .2*7/365. expm1 is precise and differentiable here.
        total=total+t+(v-t)*(-lib.expm1(-a)/a)
    return total


def draw_points(n,factors,seed,*,collocation=False):
    u=qmc.LatinHypercube(d=2+5*factors,seed=seed).random(n)
    q=u.copy();q[:,1]=math.log(7/365)+u[:,1]*math.log(2/(7/365))
    tau=np.exp(q[:,1])
    # Half continuous maturities; half evenly distributed over monthly/rich slices.
    slices=np.array([30,60,90,180,365,730])/365
    on=np.arange(n)%2==0
    tau[on]=slices[np.arange(on.sum())%6];q[:,1]=np.log(tau)
    vb=expected_variance(q,factors)
    q[:,0]=(u[:,0]*6-3)*np.sqrt(vb*tau)
    if collocation:
        wide=np.arange(n)%4==0
        q[wide,0]=3*(2*u[wide,0]-1)
    return q


def black_call(x,w):
    root=np.sqrt(w);d=x/root+root/2
    return np.exp(x)*ndtr(d)-ndtr(d-root)


def invert_total_variance(price,x):
    lower=np.maximum(np.expm1(x),0.)
    valid=np.isfinite(price)&(price>lower+1e-12)&(price<np.exp(x)-1e-12)
    lo=np.full_like(price,1e-12);hi=np.full_like(price,40.)
    for _ in range(75):
        mid=(lo+hi)/2;above=black_call(x,mid)>price
        hi=np.where(above,mid,hi);lo=np.where(above,lo,mid)
    return np.where(valid,(lo+hi)/2,np.nan)


def exact_prices(q,factors,nodes=128):
    p=decode_unit(q[...,2:],factors,torch)
    x,tau=q[...,0],torch.exp(q[...,1]);one=torch.ones_like(x)[:,None]
    engine=price_call if factors==2 else price_call_single
    return engine(p,torch.exp(x)[:,None],one,tau[:,None],one*0,one*0,
                  node_count=nodes)[:,0]


def teacher_labels(points,factors,*,chunk=512,gradients=True):
    """Prices and d(g)/du from independent float64 Fourier evaluation.

    IV differentiation uses the implicit Black inverse, not finite differences.
    Parameter derivatives keep x and tau fixed, including when sampled geometry
    was constructed using the generating parameters.
    """
    rows=[]
    for start in range(0,len(points),chunk):
        q=torch.tensor(points[start:start+chunk],dtype=torch.float64,requires_grad=gradients)
        c=exact_prices(q,factors);price=c.detach().numpy()
        with torch.no_grad():other=exact_prices(q,factors,96).numpy()
        w=invert_total_variance(price,q[:,0].detach().numpy())
        vb=expected_variance(q,factors,torch)
        total=(torch.exp(q[:,1])*vb)
        g=.5*(np.log(w)-np.log(total.detach().numpy()))
        valid=np.isfinite(g)&(np.abs(price-other)<=1e-9)&(np.abs(g)<1.5)
        block={"q":q.detach().numpy(),"price":price,"w":w,"g":g,"usable":valid,
               "quadrature_difference":np.abs(price-other)}
        if gradients:
            dprice=torch.autograd.grad(c.sum(),q,retain_graph=True)[0].detach().numpy()[:,2:]
            dback=torch.autograd.grad(torch.log(total).sum(),q)[0].detach().numpy()[:,2:]
            root=np.sqrt(w);d=q[:,0].detach().numpy()/root-root/2
            cw=np.exp(-d*d/2)/math.sqrt(2*math.pi)/(2*root)
            dg=.5*(dprice/(cw*w)[:,None]-dback)
            block["dg_du"]=dg
            block["usable"] &= np.isfinite(dg).all(axis=1)
        rows.append(block)
    return {key:np.concatenate([r[key] for r in rows]) for key in rows[0]}
