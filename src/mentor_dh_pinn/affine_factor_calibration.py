"""Truth-blind inverse calibration of the frozen factor PINN, in IV units."""
import time
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.stats import qmc
from .affine_factor_pricing import neural_call_prices
from .regular_pinn_data import decode_unit,invert_total_variance


def fit_factor_pinn(model,x,tau,observed_iv,*,fit_mask=None,factors=2,starts=5,seed=71,max_nfev=400):
    x,tau,observed_iv=[np.asarray(a,dtype=float) for a in (x,tau,observed_iv)]
    if x.ndim!=1 or x.shape!=tau.shape or x.shape!=observed_iv.shape:
        raise ValueError('Matching one-dimensional quote arrays required')
    mask=np.ones(x.shape,bool) if fit_mask is None else np.asarray(fit_mask,dtype=bool)
    if mask.shape!=x.shape:raise ValueError('Invalid quote mask')
    x,tau,observed_iv=x[mask],tau[mask],observed_iv[mask]
    if not len(x) or not np.isfinite([x,tau,observed_iv]).all() or (tau<=0).any() or (observed_iv<=0).any():
        raise ValueError('Calibration quotes must be finite with positive tau and IV')
    if factors not in (1,2) or starts<1 or max_nfev<1:raise ValueError('Invalid calibration dimensions/budget')
    tx,tt=torch.tensor(x,dtype=torch.float64),torch.tensor(tau,dtype=torch.float64)
    model.eval();model.requires_grad_(False)
    def price_with_aux(unit):
        value=neural_call_prices(model,decode_unit(unit,factors,torch),tx,tt)
        return value,value
    price_jacobian=torch.func.jacfwd(price_with_aux,has_aux=True)
    cached_unit=cached=None
    def residual_jacobian(unit):
        nonlocal cached_unit,cached
        if cached_unit is None or not np.array_equal(unit,cached_unit):
            derivative,price=price_jacobian(torch.tensor(unit,dtype=torch.float64))
            price,derivative=price.detach().numpy(),derivative.detach().numpy()
            iv=np.sqrt(invert_total_variance(price,x)/tau)
            if not np.isfinite(iv).all() or not np.isfinite(derivative).all():
                raise FloatingPointError('Noninvertible neural option price; no clipping or exact fallback')
            root=iv*np.sqrt(tau);d2=x/root-root/2
            vega=np.exp(-d2*d2/2)/np.sqrt(2*np.pi)*np.sqrt(tau)
            jac=derivative/vega[:,None]
            if not np.isfinite(jac).all():raise FloatingPointError('Nonfinite neural IV Jacobian')
            cached_unit,cached=unit.copy(),(iv-observed_iv,jac)
        return cached
    initial=.05+.9*qmc.LatinHypercube(5*factors,seed=seed).random(starts)
    records=[];candidates=[];started=time.perf_counter()
    for i,unit in enumerate(initial):
        record={'start':i,'initial_unit':unit.tolist()};clock=time.perf_counter()
        try:
            r=residual_jacobian(unit)[0];record['initial_sse']=float(r@r)
            fit=least_squares(lambda u:residual_jacobian(u)[0],unit,jac=lambda u:residual_jacobian(u)[1],
                bounds=(1e-5,1-1e-5),method='trf',x_scale='jac',ftol=1e-12,xtol=1e-8,gtol=1e-12,max_nfev=max_nfev)
            r=residual_jacobian(fit.x)[0];sse=float(r@r)
            record.update(unit=fit.x.tolist(),physical=decode_unit(fit.x,factors).tolist(),sse=sse,
                optimizer_success=bool(fit.success),nfev=int(fit.nfev),message=str(fit.message))
            candidates.append((sse,i,fit.x.copy(),bool(fit.success)))
        except (FloatingPointError,ValueError,RuntimeError) as error:
            record.update(sse=None,optimizer_success=False,error=f'{type(error).__name__}: {error}')
        record['seconds']=time.perf_counter()-clock;records.append(record)
    result={'status':'all_starts_failed','starts':records,'seconds':time.perf_counter()-started,
            'fit_quotes':len(x),'inference':'float64 learned factor coefficients plus fixed Fourier quadrature; no exact pricer'}
    if candidates:
        sse,i,unit,success=min(candidates,key=lambda row:row[0])
        result.update(status='fitted',unit=unit.tolist(),physical=decode_unit(unit,factors).tolist(),
                      calibration_iv_sse=sse,selected_start=i,optimizer_success=success)
    return result
