from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES
from sklearn.externals.joblib import Parallel, delayed
from scipy.stats import boxcox

__all__ = ['SpatialSignTransformer']


class SpatialSignTransformer(BaseEstimator, TransformerMixin):
    """Project the feature space of a matrix into a multi-dimensional sphere
    by dividing each feature by its squared norm.
       
    Parameters
    ----------
    n_jobs : int, 1 by default
       The number of jobs to use for the computation. This works by
       estimating each of the feature lambdas in parallel.
       
       If -1 all CPUs are used. If 1 is given, no parallel computing code
       is used at all, which is useful for debugging. For n_jobs below -1,
       (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but
       one are used.


    Attributes
    ----------
    sq_nms_ : array_like, shape (n_features,)
       The squared norms for each feature
    """
    
    def __init__(self, n_jobs = 1):
        self.n_jobs = n_jobs
        
    def fit(self, X, y = None):
        """Estimate the squared norms for each feature, provided X
        
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used for estimating the lambdas
        
        y : Passthrough for Pipeline compatibility
        """
        X = check_array(X, accept_sparse = False, copy = True,
                       ensure_2d = True, warn_on_dtype = True,
                       estimator = self, dtype = FLOAT_DTYPES)
        
        ## Now estimate the lambdas in parallel
        self.sq_nms_ = np.array(Parallel(n_jobs=self.n_jobs)(
                    delayed(_sq_norm_single)
                    (y) for y in X.transpose()))

        ## What if a squared norm is zero? We want to avoid a divide-by-zero situation...
        self.sq_nms_[self.sq_nms_ == 0.0] = np.inf
        
        return self

    def fit_transform(self, X, y = None):
        """Estimate the squared norms, provided X, and then
        return the transformed copy of X.
        
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used for estimating the lambdas
        
        y : Passthrough for Pipeline compatibility
        """
        return self.fit(X, y).transform(X, y)

    def transform(self, X, y = None):
        """Perform spatial sign transformation
        
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to transform
        """
        check_is_fitted(self, 'sq_nms_')
        
        X = check_array(X, accept_sparse = False, copy = True,
                       ensure_2d = True, warn_on_dtype = True,
                       estimator = self, dtype = FLOAT_DTYPES)
        
        _, n_features = X.shape
        sq_nms_ = self.sq_nms_
        
        if not n_features == sq_nms_.shape[0]:
            raise ValueError('dim mismatch in n_features')
        
        ## Return scaled by norms
        return X / sq_nms_

    def inverse_transform(self, X):
        """Transform back to the original representation
        
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to inverse transform
        """
        check_is_fitted(self, 'sq_nms_')
        
        X = check_array(X, accept_sparse = False, copy = True,
                       ensure_2d = True, warn_on_dtype = True,
                       estimator = self, dtype = FLOAT_DTYPES)
        
        _, n_features = X.shape

        ## If we have any infs in the output data, it's because
        ## the squared norm was 0 and we forced it to Inf...
        ## This will only happen if all elements in the feature
        ## were 0 to begin with:
        sq_nms_ = np.array([s if not s == np.inf else 0.0 for s in self.sq_nms_])
        
        if not n_features == sq_nms_.shape[0]:
            raise ValueError('dim mismatch in n_features')
        
        ## Re-multiply by the norms
        return X * sq_nms_


def _sq_norm_single(x):
	return np.dot(x, x)
