from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES
from sklearn.externals.joblib import Parallel, delayed
from scipy.stats import boxcox
from scipy import optimize
from .encode import get_unseen


__all__ = [
    'BoxCoxTransformer',
    'YeoJohnsonTransformer',
    'SpatialSignTransformer'
]

ZERO = 1e-16



## Helper funtions:
def _all_sparse(x):
    """Often, if all elements are sparse they are generated
    by the OneHotCategoricalTransformer, and we want to ignore them.
    This is challenging since the BoxCoxTransformer is casting everything
    to float already and we're going to get poor precision. If we merely
    cast everything straight to int, we may turn a -0.8 into a 0 (try on
    the last column of iris and see what happens).
    """
    us = get_unseen()
    sparse = np.array([0, 1, us]) ## have minus one in the mix as well since we shift down potentially...
    return all([any([np.isclose(v,i) for i in sparse]) for v in x])

def _eqls(lam, v):
    return np.abs(lam) <= v



class BoxCoxTransformer(BaseEstimator, TransformerMixin):
    """Estimate a lambda parameter for each feature, and transform
       it to a distribution more-closely resembling a Gaussian bell
       using the Box-Cox transformation. By default, will ignore sparse
       features that are generated via the OneHotCategoricalTransformer.
       
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
                       ensure_2d = True, warn_on_dtype = False,
                       estimator = self, dtype = FLOAT_DTYPES)
        
        n_samples, n_features = X.shape
        if n_samples < 2:
            raise ValueError('n_samples should be at least two, but was %i' % n_samples)
            
            
        ## Check for sparse BEFORE shift
        sparse_cols_ = Parallel(n_jobs = self.n_jobs)(
            delayed(_all_sparse)
            (i) for i in X.transpose())
            

        ## First step is to compute all the shifts needed, then add back to X...
        min_Xs = X.min(axis = 0)
        self.shift_ = np.array([np.abs(x) + 1e-6 if x <= 0.0 else 0.0 for x in min_Xs])
        X += self.shift_
        
        
        ## Now estimate the lambdas in parallel
        Xt = X.transpose()
        self.lambda_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_estimate_lambda_single_y)
            (Xt[i], sparse_cols_[i]) for i in range(n_features))
        
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

    ## If lam is nan, then it was a sparse column:
    if np.isnan(lam):
        return y

    ool = 1.0 / lam
    return np.array(map(lambda x: np.power(((x*lam)+1), ool) if not _eqls(lam,ZERO) else np.exp(x), y))

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

    ## If lam is nan, then it was a sparse column:
    if np.isnan(lam):
        return y

    return np.array(map(lambda x: (np.power(x, lam)-1)/lam if not _eqls(lam,ZERO) else np.log(x), y))
    
def _estimate_lambda_single_y(y, is_sparse):
    """Estimate lambda for a single y, given a range of lambdas
    through which to search. No validation performed.
    
    Parameters
    ----------
    y : ndarray, shape (n_samples,)
       The vector being estimated against
       
    lambdas : ndarray, shape (n_lambdas,)
       The vector of lambdas to estimate with
    """
    
    ## If not an array, then was identified as a sparse col
    if is_sparse:
        return np.nan
    
    ## Use scipy's log-likelihood estimator
    b = boxcox(y, lmbda = None)
    
    ## Return lambda corresponding to maximum P
    return b[1]



class YeoJohnsonTransformer(BaseEstimator, TransformerMixin):
    """Estimate a lambda parameter for each feature, and transform
       it to a distribution more-closely resembling a Gaussian bell
       using the Yeo-Johnson transformation.

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


        ## Now estimate the lambdas in parallel
        self.lambda_ = np.array(Parallel(n_jobs=self.n_jobs)(
            delayed(_yj_estimate_lambda_single_y)
            (y) for y in X.transpose()))

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
        """Perform Yeo-Johnson transformation

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to transform
        """
        check_is_fitted(self, 'lambda_')

        X = check_array(X, accept_sparse = False, copy = True,
                       ensure_2d = True, warn_on_dtype = True,
                       estimator = self, dtype = FLOAT_DTYPES)

        _, n_features = X.shape
        lambdas_ = self.lambda_

        if not n_features == lambdas_.shape[0]:
            raise ValueError('dim mismatch in n_features')

        return np.array([_yj_transform_y(X[:,i], lambdas_[i])\
                         for i in xrange(n_features)]).transpose()

    def inverse_transform(self, X):
        """Transform back to the original representation

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to inverse transform
        """
        check_is_fitted(self, 'lambda_')

        X = check_array(X, accept_sparse = False, copy = True,
                       ensure_2d = True, warn_on_dtype = True,
                       estimator = self, dtype = FLOAT_DTYPES)

        _, n_features = X.shape
        lambdas_ = self.lambda_

        if not n_features == lambdas_.shape[0]:
            raise ValueError('dim mismatch in n_features')

        return np.array([_yj_inv_transform_y(X[:,i], lambdas_[i])\
                         for i in xrange(n_features)]).transpose()

def _yj_inv_trans_single_x(x, lam):
    ## This is where it gets messy, but we can theorize that
    ## if the x is < 0 and the lambda meets the appropriate conditions,
    ## that the x was sub-zero to begin with.
    if x >= 0:
        ## Case 1: x >= 0 and lambda is not 0
        if not _eqls(lam, ZERO):
            x *= lam
            x += 1
            x = np.power(x, 1.0 / lam)
            return x - 1

        ## Case 2: x >= 0 and lambda is 0
        return np.exp(x) - 1
    else:
        ## Case 3: lambda does not equal 2
        if not lam == 2.0:
            x *= -(2.0 - lam)
            x += 1
            x = np.pow(x, 1.0 / (2.0 - lam))
            x -= 1
            return -x

        ## Case 4: lambda equals 2
        return -(np.exp(-x) - 1)


def _yj_inv_transform_y(y, lam):
    """Inverse transform a single y, given a single
    lambda value. No validation performed.

    Parameters
    ----------
    y : ndarray, shape (n_samples,)
       The vector being inverse transformed

    lam : ndarray, shape (n_lambdas,)
       The lambda value used for the inverse operation
    """
    ## If lam is nan, then it was a sparse column:
    if np.isnan(lam):
        return y

    return np.array([_yj_inv_trans_single_x(x, lam) for x in y])

def _yj_trans_single_x(x, lam):
    if x >= 0:
        ## Case 1: x >= 0 and lambda is not 0
        if not _eqls(lam, ZERO):
            return (np.power(x + 1, lam) - 1.0) / lam

        ## Case 2: x >= 0 and lambda is zero
        return np.log(x + 1)
    else:
        ## Case 2: x < 0 and lambda is not two
        if not lam == 2.0:
            denom = 2.0 - lam
            numer = np.power((-x + 1), (2.0 - lam)) - 1.0
            return -numer / denom

        ## Case 4: x < 0 and lambda is two
        return -np.log(-x + 1)

def _yj_transform_y(y, lam):
    """Transform a single y, given a single lambda value.
    No validation performed.

    Parameters
    ----------
    y : ndarray, shape (n_samples,)
       The vector being transformed

    lam : ndarray, shape (n_lambdas,)
       The lambda value used for the transformation
    """
    ## If lam is nan, then it was a sparse column:
    if np.isnan(lam):
        return y

    return np.array([_yj_trans_single_x(x, lam) for x in y])

def _yj_estimate_lambda_single_y(y):
    """Estimate lambda for a single y, given a range of lambdas
    through which to search. No validation performed.

    Parameters
    ----------
    y : ndarray, shape (n_samples,)
       The vector being estimated against

    lambdas : ndarray, shape (n_lambdas,)
       The vector of lambdas to estimate with
    """

    ## If all sparse, likely came from a OneHotCategoricalTransformer
    if _all_sparse(y):
        return np.nan

    ## Use customlog-likelihood estimator
    return _yj_normmax(y)

def _yj_normmax(x, brack = (-2, 2)):
    """Compute optimal YJ transform parameter for input data.

    Parameters
    ----------
    x : array_like
       Input array.
    brack : 2-tuple
       The starting interval for a downhill bracket search
    """

    ## Use MLE to compute the optimal YJ parameter
    def _mle_opt(x, brack):
        def _eval_mle(lmb, data):
            ## Function to minimize
            return -_yj_llf(data, lmb)
    
        return optimize.brent(_eval_mle, brack = brack, args = (x,))

    ## If we don't want to use the optimizer...
    def _mle(x, brack):
        rng = np.arange(brack[0], brack[1], 0.05)
        min_llf, best_lam = np.inf, None

        for lam in rng:
            llf = _yj_llf(x, lam)
            if llf < min_llf:
                min_llf = llf
                best_lam = lam
        return best_lam

    return _mle_opt(x, brack) #_mle(x, brack)

def _yj_llf(data, lmb):
    """Transform a y vector given a single lambda value,
    and compute the log-likelihood function. No validation
    is applied to the input.

    Parameters
    ----------
    data : array_like
       The vector to transform

    lmb : scalar
       The lambda value
    """

    data = np.asarray(data)
    N = data.shape[0]
    if 0 == N:
        raise ValueError('data is empty')
        #return np.nan

    y = _yj_transform_y(data, lmb)

    ## We can't take the log of data, as there could be
    ## zeros or negatives. Thus, we need to shift both distributions
    ## up by some artbitrary factor just for the LLF computation
    min_d, min_y = np.min(data), np.min(y)
    if min_d < ZERO:
        shift = np.abs(min_d) + 1
        data += shift

    ## Same goes for Y
    if min_y < ZERO:
        shift = np.abs(min_y) + 1
        y += shift

    ## Compute mean on potentially shifted data
    y_mean = np.mean(y, axis = 0)
    var = np.sum((y - y_mean)**2. / N, axis = 0)

    ## If var is 0.0, we'll get a warning. Means all the 
    ## values were nearly identical in y, so we will return
    ## NaN so we don't optimize for this value of lam
    if 0 == var:
        return np.nan

    llf = (lmb - 1) * np.sum(np.log(data), axis=0)
    llf -= N / 2.0 * np.log(var)

    return llf



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

