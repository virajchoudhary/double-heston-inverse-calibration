"""Alternative optimizer coordinates with the existing exact training bounds."""
import numpy as np
from scipy.special import expit,logit


def to_optimizer(unit,mode):
    u=np.asarray(unit,dtype=float)
    if u.shape!=(10,) or not np.isfinite(u).all() or (u<=0).any() or (u>=1).any():
        raise ValueError('An interior ten-dimensional unit vector is required')
    if mode=='unit':return u.copy()
    if mode=='logit':return logit(u)
    if mode=='timescale':
        z=u.copy();ks=.2*np.exp(np.log(15)*u[0]);kf=ks+.5+(11.5-ks)*u[1]
        z[0]=(1/ks-1/3)/(5-1/3)
        z[1]=(1/kf-1/12)/(1/(ks+.5)-1/12)
        return z
    raise ValueError('Unknown optimizer parameterization')


def to_unit_and_jacobian(z,mode):
    z=np.asarray(z,dtype=float)
    if z.shape!=(10,) or not np.isfinite(z).all():raise ValueError('Finite ten-dimensional optimizer vector required')
    if mode=='unit':return z.copy(),np.eye(10)
    if mode=='logit':
        u=expit(z);return u,np.diag(u*(1-u))
    if mode=='timescale':
        if (z<0).any() or (z>1).any():raise ValueError('Timescale coordinates must lie in [0,1]')
        u=z.copy();jac=np.eye(10)
        ks=1/(1/3+z[0]*(5-1/3));dks=-(5-1/3)*ks**2
        kf=1/(1/12+z[1]*(1/(ks+.5)-1/12))
        dkf0=kf**2*z[1]*dks/(ks+.5)**2;dkf1=-kf**2*(1/(ks+.5)-1/12)
        numerator=kf-ks-.5;denominator=11.5-ks
        u[0]=np.log(ks/.2)/np.log(15);u[1]=numerator/denominator
        jac[0,0]=dks/ks/np.log(15)
        jac[1,0]=((dkf0-dks)*denominator+numerator*dks)/denominator**2
        jac[1,1]=dkf1/denominator
        return u,jac
    raise ValueError('Unknown optimizer parameterization')
