"""Frozen training error statistics for neural-only generalized least squares."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SurrogateError:
    x: np.ndarray
    tau: np.ndarray
    mean: np.ndarray
    covariance: np.ndarray
    checkpoint_sha256: str

    def transform(self, x, tau, mask, relative_floor=1e-4, observation_iv_std=None):
        # Require the training geometry explicitly; never interpolate covariance.
        if not np.array_equal(x, self.x) or not np.array_equal(tau, self.tau):
            raise ValueError('Surrogate error statistics require their exact quote geometry')
        n=len(x)
        if self.mean.shape!=(n,) or self.covariance.shape!=(n,n):
            raise ValueError('Invalid error statistic shapes')
        if not 0 < relative_floor <= 1:
            raise ValueError('Covariance floor must be in (0, 1]')
        mean=self.mean[mask]
        cov=self.covariance[np.ix_(mask,mask)].copy()
        if not np.isfinite(mean).all() or not np.isfinite(cov).all():
            raise ValueError('Nonfinite active error statistics')
        if not np.allclose(cov,cov.T,atol=1e-15,rtol=1e-10):
            raise ValueError('Covariance must be symmetric')
        values,vectors=np.linalg.eigh(cov)
        if values[0] < -max(abs(values[-1])*1e-10,1e-16):
            raise ValueError('Covariance must be positive semidefinite')
        floor=max(values[-1]*relative_floor,1e-16)
        cov=(vectors*np.maximum(values,floor))@vectors.T
        if observation_iv_std is not None:
            std=np.broadcast_to(np.asarray(observation_iv_std,dtype=float),(n,))[mask]
            if not np.isfinite(std).all() or (std<0).any():
                raise ValueError('Active observation uncertainty must be finite and nonnegative')
            cov+=np.diag(std**2)
        # W.T @ W = inv(cov), hence transform BOTH residual and Jacobian.
        whitening=np.linalg.solve(np.linalg.cholesky(cov),np.eye(len(mean)))
        return mean,whitening
