"""Independent exact forward/Jacobian diagnostics. Never used to update a fit."""
import numpy as np
import torch
from .torch_pricer import price_call


def price_and_jacobian(physical,x,tau,nodes=128):
    """Return C/S and its physical-parameter Jacobian, keeping quotes fixed."""
    x,tau=np.asarray(x),np.asarray(tau)
    p=torch.tensor(np.broadcast_to(physical,(len(x),10)).copy(),dtype=torch.float64,requires_grad=True)
    one=torch.ones((len(x),1),dtype=torch.float64)
    price=price_call(p,one,torch.tensor(np.exp(-x)[:,None]),torch.tensor(tau[:,None]),
                     one*0,one*0,node_count=nodes)[:,0]
    jac=torch.autograd.grad(price.sum(),p)[0]
    return price.detach().numpy(),jac.detach().numpy()


def physical_to_unit(physical):
    """Inverse of the current ordered training parameterization; no clipping."""
    p=np.asarray(physical,dtype=float);ks,ts,ss,rs,vs,kf,tf,sf,rf,vf=np.moveaxis(p,-1,0)
    return np.stack([np.log(ks/.2)/np.log(15),(kf-ks-.5)/(11.5-ks),
                     np.log((ts+tf)/.03)/np.log(10),(ts/(ts+tf)-.2)/.6,
                     np.log((vs+vf)/.03)/np.log(10),(vs/(vs+vf)-.2)/.6,
                     (ss/np.sqrt(2*ks*ts)-.15)/.75,(sf/np.sqrt(2*kf*tf)-.15)/.75,
                     (rs+.8)/1.1,(rf+.6)/1.1],axis=-1)


def information_diagnostics(jacobian,physical,*,standard_deviation=None,covariance=None):
    """Information in physical tolerance coordinates, with explicit noise units.

    With no uncertainty, this is an unweighted sensitivity Gram matrix, not a
    statistical Fisher estimate. SVD squares are reported rather than rounded
    negative Gram eigenvalues in severely ill-conditioned cases.
    """
    p=np.asarray(physical);tol=.05*np.abs(p);tol[3::5]=.05
    a=np.asarray(jacobian)*tol
    if standard_deviation is not None and covariance is not None:raise ValueError('Choose one uncertainty model')
    if standard_deviation is not None:
        std=np.asarray(standard_deviation)
        if (std<=0).any() or not np.isfinite(std).all():raise ValueError('Positive finite uncertainty required')
        a=a/std[:,None]
    if covariance is not None:a=np.linalg.solve(np.linalg.cholesky(covariance),a)
    _,s,right=np.linalg.svd(a,full_matrices=False)
    unique=[]
    for j in range(10):
        other=np.delete(a,j,axis=1)
        residual=a[:,j]-other@np.linalg.lstsq(other,a[:,j],rcond=None)[0]
        unique.append(residual)
    unique=np.column_stack(unique)
    return {'scaled_jacobian':a,'gram':a.T@a,'singular_values':s,'right_vectors':right,
            'fisher_eigenvalues':s**2,'condition_number':float(s[0]/max(s[-1],1e-300)),
            'numerical_rank':int(np.sum(s>s[0]*max(a.shape)*np.finfo(float).eps)),
            'relative_effective_rank_1e_8':int(np.sum(s>s[0]*1e-8)),
            'noise_resolved_tolerance_directions':int(np.sum(s>=1)) if (standard_deviation is not None or covariance is not None) else None,
            'conditional_sensitivity':np.linalg.norm(unique,axis=0),'conditional_quote_residuals':unique}
