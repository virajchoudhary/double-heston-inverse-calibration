"""Isolated four-shock convention; canonical validators are NEVER modified.

Reuse the existing factor CF and Torch pricer. Fast repeated single-surface
calibration caches geometry and evaluates the existing scalar factor exponent
once per distinct maturity, not separately for every strike.
"""
from functools import lru_cache
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.integrate import quad
from scipy.optimize import differential_evolution, least_squares, minimize_scalar
from scipy.special import ndtr
from scipy.stats import qmc

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.double_heston import heston_log_characteristic_exponent
from src.double_heston_reference import _factor_exponent, _quad_diagnostic
from src.mentor_dh_pinn.torch_pricer import price_call, price_call_single
from src.mentor_dh_pinn.regular_pinn_data import invert_total_variance


def admissible(p):
    a=np.asarray(p,float).reshape(-1,5)
    if len(a) not in [1,2] or not np.isfinite(a).all(): raise ValueError('Bad parameter shape/value')
    if np.any(a[:,[0,1,2,4]]<=0) or np.any(abs(a[:,3])>=1): raise ValueError('Economic bound failure')
    gaps=2*a[:,0]*a[:,1]-a[:,2]**2
    if np.any(gaps<=0): raise ValueError('Strict Feller contract failure')
    if len(a)==2 and a[0,0]>=a[1,0]: raise ValueError('Canonical slow-first order required')
    return gaps


@lru_cache(maxsize=4)
def rule(n):
    u,w=np.polynomial.laguerre.laggauss(n)
    return u,w*np.exp(u)/(np.pi*1j*u)


class Grid:
    def __init__(self,x,tau,nodes=128):
        self.x,self.tau=np.broadcast_arrays(np.asarray(x,float),np.asarray(tau,float))
        assert self.x.ndim==1 and np.isfinite(self.x).all() and np.isfinite(self.tau).all() and (self.tau>=0).all()
        self.times,self.ix=np.unique(self.tau,return_inverse=True)
        self.u,self.w=rule(nodes)
        self.osc=np.exp(1j*self.x[:,None]*self.u)*self.w
        self.k=np.exp(-self.x)

    def __call__(self,p):
        admissible(p)
        f=np.asarray(p).reshape(-1,5)
        # Existing NumPy reference exponent broadcasts over unique maturities.
        # Independence checks also use Torch and canonical production formulas.
        t=self.times[:,None]
        cs=np.exp(sum(_factor_exponent(self.u-1j,t,*a) for a in f))
        cu=np.exp(sum(_factor_exponent(self.u,t,*a) for a in f))
        p1=.5+np.real(self.osc*cs[self.ix]).sum(1)
        p2=.5+np.real(self.osc*cu[self.ix]).sum(1)
        result=p1-self.k*p2
        result=np.where(self.tau==0,np.maximum(1-self.k,0),result)
        if not np.isfinite(result).all(): raise FloatingPointError('Nonfinite Fourier price')
        return result


def adaptive(p,x,t,eps=1e-10):
    admissible(p)
    if t==0:return max(1-np.exp(-x),0.),{'reliable':True,'terminal':True}
    factors=np.asarray(p).reshape(-1,5)
    def cf(u): return np.exp(sum(_factor_exponent(u,t,*a) for a in factors))
    if t<1e-4:
        # Numerical amendment v2: scale the integration variable, then use
        # oscillatory quadrature. Same CF, not a Black/payoff approximation.
        frequency=x/np.sqrt(t)
        def probability(shift):
            def phi(z):return cf(z/np.sqrt(t)-1j*shift)
            if abs(frequency)<1e-12:
                value,error=quad(lambda z:float(np.imag(phi(z))/z),0,np.inf,epsabs=eps,epsrel=eps,limit=500)
            else:
                first,e0=quad(lambda z:float(np.real(np.exp(1j*frequency*z)*phi(z)/(1j*z))),0,1,epsabs=eps,epsrel=eps,limit=500)
                cos,e1=quad(lambda z:float(np.imag(phi(z))/z),1,np.inf,weight='cos',wvar=abs(frequency),epsabs=eps,limlst=500,limit=500)
                sin,e2=quad(lambda z:float(np.real(phi(z))/z),1,np.inf,weight='sin',wvar=abs(frequency),epsabs=eps,limlst=500,limit=500)
                value=first+cos+np.sign(frequency)*sin;error=e0+e1+e2
            if not np.isfinite(value) or error>max(5e-9,20*eps):raise FloatingPointError('Short-time oscillatory reference failed')
            return .5+value/np.pi,float(error)
        p1,e1=probability(1);p2,e2=probability(0)
        return float(p1-np.exp(-x)*p2),{'reliable':True,'method':'scaled_oscillatory_quad','errors':[e1,e2]}
    one,d1=_quad_diagnostic(lambda u:float(np.real(np.exp(1j*u*x)*cf(u-1j)/(1j*u))),epsabs=eps,epsrel=eps,limit=500)
    two,d2=_quad_diagnostic(lambda u:float(np.real(np.exp(1j*u*x)*cf(u)/(1j*u))),epsabs=eps,epsrel=eps,limit=500)
    value=.5+one/np.pi-np.exp(-x)*(.5+two/np.pi)
    ok=bool(d1.get('reliable') and d2.get('reliable') and np.isfinite(value))
    if not ok:raise FloatingPointError(f'Unreliable adaptive reference: {d1}, {d2}')
    return float(value),{'reliable':ok,'p1':d1,'p2':d2}


def exact(p,x,tau, audit=None):
    x,tau=np.asarray(x),np.asarray(tau)
    a=Grid(x,tau,128)(p); b=Grid(x,tau,96)(p)
    lower=np.maximum(1-np.exp(-x),0)
    bad=(abs(a-b)>1e-8)|(a<lower-1e-9)|(a>1+1e-9)
    for j in np.flatnonzero(bad):
        a[j],d=adaptive(p,float(x[j]),float(tau[j]))
        if audit is not None:audit.append({'x':float(x[j]),'tau':float(tau[j]),'params':np.asarray(p).tolist(),'reference':d})
    if np.any(a<lower-1e-9) or np.any(a>1+1e-9): raise FloatingPointError('Price bounds failure')
    return a


def batch_teacher(params,coords):
    """Independent points: [x,v_s,v_f,tau]; params each row slow-first."""
    result=[]; diffs=[]
    with torch.no_grad():
        for start in range(0,len(coords),256):
            z=torch.tensor(coords[start:start+256],dtype=torch.float64)
            p=torch.tensor(params[start:start+256],dtype=torch.float64)
            x,t=z[:,0:1],z[:,-1:]; one=torch.ones_like(x)
            fn=price_call if p.shape[1]==10 else price_call_single
            a=fn(p,one,torch.exp(-x),t,one*0,one*0,node_count=128)[:,0].numpy()
            b=fn(p,one,torch.exp(-x),t,one*0,one*0,node_count=96)[:,0].numpy()
            result.append(a);diffs.append(abs(a-b))
    a=np.concatenate(result);diff=np.concatenate(diffs)
    lo=np.maximum(1-np.exp(-coords[:,0]),0)
    bad=(diff>1e-8)|(a<lo-1e-9)|(a>1+1e-9)
    for j in np.flatnonzero(bad):a[j]=adaptive(params[j],float(coords[j,0]),float(coords[j,-1]))[0]
    if not np.isfinite(a).all() or np.any(a<lo-1e-9):raise FloatingPointError('Teacher failure')
    return a,{'fallback_count':int(bad.sum()),'pre_fallback_max_96_128_difference':float(diff.max()),'roundoff_bound_count':int((a<lo).sum())}


def iv(price,x,tau):
    return np.sqrt(invert_total_variance(np.asarray(price)*np.exp(x),np.asarray(x))/tau)


def black(x,tau,sigma):
    root=np.asarray(sigma)*np.sqrt(tau); d=x/root+root/2
    return ndtr(d)-np.exp(-x)*ndtr(d-root)


def fit_bs(x,tau,y,term=False):
    def fit(mask):
        r=minimize_scalar(lambda v:np.mean((black(x[mask],tau[mask],v)-y[mask])**2),bounds=(.02,2.5),method='bounded',options={'xatol':1e-12})
        return float(r.x)
    knots=np.unique(tau) if term else np.array([float(np.median(tau))])
    values=[fit(tau==t if term else np.ones(len(tau),bool)) for t in knots]
    return {'knots':knots.tolist(),'volatility':values,'term':term}


def bs_predict(fit,x,tau):
    t=np.asarray(fit['knots']); v=np.asarray(fit['volatility'])
    w=np.interp(tau,t,t*v*v)
    w=np.where(tau<t[0],tau*v[0]**2,np.where(tau>t[-1],tau*v[-1]**2,w))
    return black(x,tau,np.sqrt(w/tau))


def decode_sh(z):
    k,t,eta,r,v=np.asarray(z)
    k,t,v=np.exp([k,t,v])
    return np.array([k,t,eta*np.sqrt(2*k*t),r,v])


def fit_sh(x,tau,y,cfg):
    grid=Grid(x,tau,128)
    bounds=[np.log(cfg['bounds_log_kappa']),np.log(cfg['bounds_theta']),cfg['bounds_eta'],cfg['bounds_rho'],np.log(cfg['bounds_v0'])]
    lo,hi=np.array(bounds).T
    def resid(z):return (grid(decode_sh(z))-y)/.01
    glob=differential_evolution(lambda z:float(np.mean(resid(z)**2)),bounds,seed=cfg['seed'],maxiter=cfg['global_iterations'],popsize=cfg['global_population_multiplier'],polish=False,workers=1)
    u=qmc.LatinHypercube(5,seed=cfg['seed']+1).random(cfg['multistarts']-1)
    starts=[glob.x,*list(lo+(hi-lo)*u)]; records=[]
    for j,z in enumerate(starts):
        try:
            r=least_squares(resid,z,bounds=(lo,hi),max_nfev=cfg['max_nfev'],xtol=1e-11,ftol=1e-11,gtol=1e-11,x_scale='jac')
            records.append({'start':j,'success':bool(r.success),'status':int(r.status),'message':str(r.message),'nfev':int(r.nfev),
                'objective':float(np.mean((r.fun*.01)**2)),'params':decode_sh(r.x).tolist(),
                'near_bound':bool(np.any(np.minimum((r.x-lo)/(hi-lo),(hi-r.x)/(hi-lo))<.005))})
        except (ValueError,FloatingPointError) as e:records.append({'start':j,'success':False,'message':str(e),'objective':None})
    finite=[r for r in records if r['objective'] is not None]
    if not finite:raise RuntimeError('All SH starts failed')
    best=min(finite,key=lambda r:r['objective'])
    repeats=sum(r['success'] and r['objective']<=best['objective']+max(1e-12,.01*best['objective']) for r in finite)
    return {'best':best,'starts':records,'converged_near_best':repeats,'global':{'success':bool(glob.success),'message':str(glob.message),'nfev':int(glob.nfev)},'global_optimum_proven':False}
