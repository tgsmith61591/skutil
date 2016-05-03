from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES
from sklearn.externals.joblib import Parallel, delayed
from scipy.stats import boxcox

__all__ = ['BoxCoxTransformer']

class BoxCoxTransformer(BaseEstimator, TransformerMixin):
    """Estimate a lambda parameter for each feature, and transform
       it to a distribution more-closely resembling a Gaussian bell
       using the Box-Cox transformation.
       
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
    shift_ : ndarray, shape (n_features,)
       The shifts for each feature needed to shift the min value in 
       the feature up to at least 0.0, as every element must be positive

    lambda_ : ndarray, shape (n_features,)
       The lambda values corresponding to each feature
    """
    
    def __init__(self, n_jobs = 1):
        self.n_jobs = n_jobs
        
    def fit(self, X, y = None):
        """Estimate the lambdas, provided X
        
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used for estimating the lambdas
        
        y : Passthrough for Pipeline compatibility
        """
        X = check_array(X, accept_sparse = False, copy = True,
                       ensure_2d = True, warn_on_dtype = True,
                       estimator = self, dtype = FLOAT_DTYPES)
        
        n_samples, n_features = X.shape
        if n_samples < 2:
            raise ValueError('n_samples should be at least two, but was %i' % n_samples)

        ## First step is to compute all the shifts needed, then add back to X...
        min_Xs = X.min(axis = 0)
        self.shift_ = np.array([np.abs(x) + 1e-6 if x <= 0.0 else 0.0 for x in min_Xs])
        X += self.shift_
        
        
        ## Now estimate the lambdas in parallel
        self.lambda_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_estimate_lambda_single_y)
            (y) for y in X.transpose())
        
        return self
    
    def fit_transform(self, X, y = None):
        """Estimate the lambdas, provided X, and then
        return the transformed copy of X.
        
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used for estimating the lambdas
        
        y : Passthrough for Pipeline compatibility
        """
        return self.fit(X, y).transform(X, y)
    
    def transform(self, X, y = None):
        """Perform Box-Cox transformation
        
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to transform
        """
        check_is_fitted(self, 'shift_')
        
        X = check_array(X, accept_sparse = False, copy = True,
                       ensure_2d = True, warn_on_dtype = True,
                       estimator = self, dtype = FLOAT_DTYPES)
        
        _, n_features = X.shape
        lambdas_, shifts_ = self.lambda_, self.shift_
        
        if not n_features == shifts_.shape[0]:
            raise ValueError('dim mismatch in n_features')
            
        ## Add the shifts in, and if they're too low,
        ## we have to truncate at some low value: 1e-6
        X += shifts_
        
        ## If the shifts are too low, truncate...
        X[X <= 0.0] = 1e-16
        
        return np.array([_transform_y(X[:,i], lambdas_[i])\
                         for i in xrange(n_features)]).transpose()
    
    def inverse_transform(self, X):
        """Transform back to the original representation
        
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to inverse transform
        """
        check_is_fitted(self, 'shift_')
        
        X = check_array(X, accept_sparse = False, copy = True,
                       ensure_2d = True, warn_on_dtype = True,
                       estimator = self, dtype = FLOAT_DTYPES)
        
        _, n_features = X.shape
        lambdas_, shifts_ = self.lambda_, self.shift_
        
        if not n_features == shifts_.shape[0]:
            raise ValueError('dim mismatch in n_features')
        
        X = np.array([_inv_transform_y(X[:,i], lambdas_[i])\
                         for i in xrange(n_features)]).transpose()
        
        ## Remember to subtract out the shifts
        return X - shifts_

def _inv_transform_y(y, lam):
    """Inverse transform a single y, given a single 
    lambda value. No validation performed.
    
    Parameters
    ----------
    y : ndarray, shape (n_samples,)
       The vector being inverse transformed
       
    lam : ndarray, shape (n_lambdas,)
       The lambda value used for the inverse operation
    """
    ool = 1.0 / lam
    return np.array(map(lambda x: np.power(((x*lam)+1), ool) if not lam == 0 else np.exp(x), y))

def _transform_y(y, lam):
    """Transform a single y, given a single lambda value.
    No validation performed.
    
    Parameters
    ----------
    y : ndarray, shape (n_samples,)
       The vector being transformed
       
    lam : ndarray, shape (n_lambdas,)
       The lambda value used for the transformation
    """
    return np.array(map(lambda x: (np.power(x, lam)-1)/lam if not lam == 0 else np.log(x), y))
    
def _estimate_lambda_single_y(y):
    """Estimate lambda for a single y, given a range of lambdas
    through which to search. No validation performed.
    
    Parameters
    ----------
    y : ndarray, shape (n_samples,)
       The vector being estimated against
       
    lambdas : ndarray, shape (n_lambdas,)
       The vector of lambdas to estimate with
    """
    
    ## Use scipy's log-likelihood estimator
    b = boxcox(y, lmbda = None)
    
    ## Return lambda corresponding to maximum P
    return b[1]

