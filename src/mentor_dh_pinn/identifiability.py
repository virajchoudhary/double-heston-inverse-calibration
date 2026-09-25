"""Factor symmetry and tolerance-aware recovery metrics; no pricing/optimization."""
import numpy as np

PARAMETERS=('kappa_s','theta_s','sigma_s','rho_s','v0_s',
            'kappa_f','theta_f','sigma_f','rho_f','v0_f')
SWAP=np.array([5,6,7,8,9,0,1,2,3,4])


def canonicalize_parameters(parameters):
    """Order complete factors lexicographically, primarily by increasing kappa.

    The existing decode_unit training map already guarantees kappa_s + .5 <=
    kappa_f. Lexicographic tie-breaking makes this utility deterministic even
    outside that restricted training domain. Never sort individual parameters.
    """
    p=np.asarray(parameters,dtype=float)
    if p.shape[-1]!=10 or not np.isfinite(p).all():
        raise ValueError('Expected finite ten-parameter vectors')
    swap=np.zeros(p.shape[:-1],dtype=bool);equal=np.ones_like(swap)
    for j in range(5):
        swap|=equal & (p[...,j]>p[...,j+5])
        equal&=p[...,j]==p[...,j+5]
    return np.where(swap[...,None],p[...,SWAP],p).copy()


def normalized_errors(prediction,truth):
    """Positive parameters: signed relative error; correlations: absolute units."""
    pred,true=np.broadcast_arrays(np.asarray(prediction,dtype=float),np.asarray(truth,dtype=float))
    if pred.shape[-1]!=10 or not np.isfinite(pred).all() or not np.isfinite(true).all():
        raise ValueError('Expected finite ten-parameter vectors')
    scale=np.abs(true).copy();scale[...,3::5]=1.
    if (scale<=0).any():raise ValueError('Positive-parameter truths must be nonzero')
    return (pred-true)/scale


def match_factors(prediction,truth):
    """Match the WHOLE factor permutation by minimum squared tolerance error.

    Both relative-positive and absolute-correlation errors have tolerance .05.
    Deterministic ties keep the direct assignment. This never modifies a fit.
    """
    pred,true=np.broadcast_arrays(np.asarray(prediction,dtype=float),np.asarray(truth,dtype=float))
    direct=normalized_errors(pred,true);reverse=normalized_errors(pred,true[...,SWAP])
    swap=np.sum(reverse**2,axis=-1)<np.sum(direct**2,axis=-1)
    return np.where(swap[...,None],true[...,SWAP],true).copy(),swap


def recovery_metrics(prediction,truth,*,permutation_invariant=True):
    matched,swapped=match_factors(prediction,truth) if permutation_invariant else (np.asarray(truth),False)
    error=normalized_errors(prediction,matched);tol_error=np.abs(error)/.05
    passed=tol_error<=1.;counts=passed.sum(axis=-1)
    # Preserve historical scaling in a distinct field: rho used scale .5.
    historical=error.copy();historical[...,3::5]*=2.
    return {'matched_truth':matched,'swapped':swapped,'error':error,
            'tolerance_error':tol_error,'parameter_pass':passed,'pass_count':counts,
            'all_ten':counts==10,'at_least_nine':counts>=9,'at_least_eight':counts>=8,
            'tolerance_rmse':np.sqrt(np.mean((error/.05)**2,axis=-1)),
            'historical_scaled_rmse':np.sqrt(np.mean(historical**2,axis=-1))}


def parameter_error_summary(predictions,truths,*,permutation_invariant=True):
    metrics=recovery_metrics(predictions,truths,permutation_invariant=permutation_invariant)
    error=metrics['error'];absolute=np.abs(error)
    if error.ndim!=2:raise ValueError('Expected a batch of predictions')
    rows=[]
    for j,name in enumerate(PARAMETERS):
        rows.append({'parameter':name,'units':'absolute correlation' if j%5==3 else 'relative fraction',
                     'cases':len(error),'mean_error':float(absolute[:,j].mean()),
                     'median_error':float(np.median(absolute[:,j])),
                     'rmse':float(np.sqrt(np.mean(error[:,j]**2))),
                     'p90_error':float(np.quantile(absolute[:,j],.9)),
                     'pass_percent':float(100*metrics['parameter_pass'][:,j].mean()),
                     'bias':float(error[:,j].mean()),'error_sd':float(error[:,j].std()),
                     'sole_failure_cases':int(((metrics['pass_count']==9)&~metrics['parameter_pass'][:,j]).sum()),
                     'failure_cases':int((~metrics['parameter_pass'][:,j]).sum())})
    return rows,np.corrcoef(error,rowvar=False),metrics
