"""Exact synthetic TRAINING/validation references; never imported by neural core."""
import numpy as np
import torch
from scipy.stats import qmc
from .affine_factor_pinn import scales


def reference_coefficients(q):
    tau,kappa,sigma,rho,u,shift=q.unbind(-1)
    z=torch.complex(u,-shift);c=z*z+1j*z
    b=kappa-rho*sigma*1j*z
    disc=torch.sqrt(b*b+sigma*sigma*c)
    # Rationalized b-disc avoids loss of significant digits at small sigma/u.
    bm=-sigma*sigma*c/(b+disc)
    g=bm/(b+disc);one_minus_exp=-torch.expm1(-disc*tau)
    denom=1-g+g*one_minus_exp
    d=-c/(b+disc)*one_minus_exp/denom
    a=(bm*tau-2*torch.log1p(g*one_minus_exp/(1-g)))/(sigma*sigma)
    return d,a


def draw_factor_points(n,seed):
    """Independent synthetic factor states; theta only sets a feasible sigma range."""
    z=qmc.LatinHypercube(7,seed=seed).random(n)
    tau=(7/365)*np.exp(z[:,0]*np.log(2/(7/365)))
    kappa=.2*np.exp(z[:,1]*np.log(12/.2))
    theta=.006*np.exp(z[:,2]*np.log(.24/.006))
    sigma=(.15+.75*z[:,3])*np.sqrt(2*kappa*theta)
    rho=-.8+1.3*z[:,4]
    u=.01*np.exp(z[:,5]*np.log(500/.01))
    nodes=np.polynomial.laguerre.laggauss(128)[0]
    fixed=np.arange(n)%2==0;u[fixed]=nodes[np.arange(fixed.sum())%len(nodes)]
    shift=(z[:,6]>=.5).astype(float)
    return np.column_stack([tau,kappa,sigma,rho,u,shift])


def correction_targets(q):
    """Four real, dimensionless coefficient targets, not parameter labels."""
    d,a=reference_coefficients(q)
    z=torch.complex(q[...,4],-q[...,5]);c=-.5*(z*z+1j*z)
    h=d/(c*q[...,0]);g=a/(c*q[...,0]**2)
    _,size,frequency,bh,bg=scales(q)
    rg=size/(1+size);ig=frequency/(1+frequency)
    targets=torch.stack([torch.log(h.real/bh)/rg,h.imag/bh/ig,
                         torch.log(g.real/bg)/rg,g.imag/bg/ig],-1)
    if not torch.isfinite(targets).all():raise FloatingPointError('Invalid exact factor training targets')
    return targets
