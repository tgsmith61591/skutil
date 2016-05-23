import six
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.externals.joblib import Parallel, delayed
from scipy.stats import boxcox
from scipy import optimize
from .encode import get_unseen
from ..utils import *
from ..base import *


__all__ = [
    'BoxCoxTransformer',
    'FunctionMapper',
    'SelectiveImputer',
    'SelectiveScaler',
    'SpatialSignTransformer',
    'YeoJohnsonTransformer',
]

ZERO = 1e-16


## Helper funtions:
def _eqls(lam, v):
    return np.abs(lam) <= v

def _validate_rows(X):
    m,n = X.shape
    if m < 2:
        raise ValueError('n_samples should be at least two, but got %i' % m)



###############################################################################
class FunctionMapper(BaseEstimator, TransformerMixin, SelectiveMixin):
    """Apply a function to a column or set of columns.

    Parameters
    ----------
    cols : string or array_like, default None
        The columns to apply a function to

    fun : function, default None
        The function to apply to the feature(s)
    """

    def __init__(self, cols=None, fun=None, **kwargs):
        self.cols = cols
        self.fun = fun
        self.kwargs = kwargs

    def fit(self, X, y = None):
        """Validate the args
        
        Parameters
        ----------
        X : pandas DF, shape [n_samples, n_features]
            The data used for estimating the lambdas
        
        y : Passthrough for Pipeline compatibility
        """
        # this function is a bit strange, because we can accept a single col:
        if isinstance(self.cols, str):
            self.cols = [self.cols]

        # Check this second in this case
        X, self.cols = validate_is_pd(X, self.cols)

        # validate the function. If none, make it a passthrough
        if not self.fun:
            def pass_through(x):
                return x

            self.fun = pass_through
        else:
            # check whether is function
            if not hasattr(self.fun, '__call__'):
                raise ValueError('passed fun arg is not a function')

        return self

    def transform(self, X, y = None):
        """Apply the function to the new data.
        
        Parameters
        ----------
        X : pandas DF, shape [n_samples, n_features]
            The data used for estimating the lambdas
        
        y : Passthrough for Pipeline compatibility
        """
        X, _ = validate_is_pd(X, self.cols)

        # apply the function
        X[self.cols] = X[self.cols].apply(lambda x: self.fun(x, **self.kwargs))
        return X



###############################################################################
class SelectiveImputer(BaseEstimator, TransformerMixin, SelectiveMixin):
    """An imputer class that can operate across a select
    group of columns. Useful for data that contains categorical features
    that have not yet been dummied, for dummied features that we
    may not want to scale, or for any already-in-scale features.

    Parameters
    ----------
    cols : array_like (string)
        names of columns on which to apply scaling

    missing_values : str, default 'NaN'
        the missing value representation

    strategy : str, default 'mean'
        the strategy for imputation

    as_df : boolean, default True
        Whether to return a dataframe

    Attributes
    ----------
    cols : array_like (string)
        the columns

    imputer_ : the fit imputer

    missing_values : see above
    strategy : see above

    """

    def __init__(self, cols=None, missing_values = 'NaN', strategy = 'mean', as_df=True):
        self.cols = cols
        self.missing_values = missing_values
        self.strategy = strategy
        self.as_df = as_df

    def fit(self, X, y = None):
        """Fit the imputers.
        
        Parameters
        ----------
        X : pandas DF, shape [n_samples, n_features]
            The data used for estimating the lambdas
        
        y : Passthrough for Pipeline compatibility
        """
        # check on state of X and cols
        X, self.cols = validate_is_pd(X, self.cols)

        ## fails if columns don't exist
        self.imputer_ = Imputer(missing_values=self.missing_values, strategy=self.strategy).fit(X[self.cols])
        return self

    def transform(self, X, y = None):
        """Transform the incoming data.
        
        Parameters
        ----------
        X : pandas DF, shape [n_samples, n_features]
            The data used for estimating the lambdas
        
        y : Passthrough for Pipeline compatibility
        """
        check_is_fitted(self, 'imputer_')
        # check on state of X and cols
        X, _ = validate_is_pd(X, self.cols)

        X[self.cols] = self.imputer_.transform(X[self.cols])
        return X if self.as_df else X.as_matrix()



###############################################################################
class SelectiveScaler(BaseEstimator, TransformerMixin, SelectiveMixin):
    """A class that will apply scaling only to a select group
    of columns. Useful for data that contains categorical features
    that have not yet been dummied, for dummied features that we
    may not want to scale, or for any already-in-scale features.
    Perhaps, even, there are some features you'd like to impute in
    a different manner than others. This, then, allows two back-to-back
    SelectiveScalers with different columns & strategies in a pipeline object.

    Parameters
    ----------
    cols : array_like (string)
        names of columns on which to apply scaling

    scaler : instance of a sklearn Scaler, default StandardScaler

    as_df : boolean, default True
        Whether to return a dataframe


    Attributes
    ----------
    cols : array_like (string)
        the columns

    scaler : instance of a sklearn Scaler
        the scaler
    """

    def __init__(self, cols=None, scaler = StandardScaler(), as_df=True):
        self.cols = cols
        self.scaler = scaler
        self.as_df = as_df

    def fit(self, X, y = None, **kwargs):
        """Fit the scaler"""
        # check on state of X and cols
        X, self.cols = validate_is_pd(X, self.cols)

        ## throws exception if the cols don't exist
        self.scaler.fit(X[self.cols])
        return self

    def transform(self, X, y = None):
        """Transform on new data, return a pd DataFrame"""
        # check on state of X and cols
        X, _ = validate_is_pd(X, self.cols)

        ## Fails through if cols don't exist or if the scaler isn't fit yet
        X[self.cols] = self.scaler.transform(X[self.cols])
        return X if self.as_df else X.as_matrix()



###############################################################################
class BoxCoxTransformer(BaseEstimator, TransformerMixin, SelectiveMixin):
    """Estimate a lambda parameter for each feature, and transform
       it to a distribution more-closely resembling a Gaussian bell
       using the Box-Cox transformation. By default, will ignore sparse
       features that are generated via the OneHotCategoricalTransformer.
       
    Parameters
    ----------
    cols : array_like, str
       The columns which to transform

    n_jobs : int, 1 by default
       The number of jobs to use for the computation. This works by
       estimating each of the feature lambdas in parallel.
       
       If -1 all CPUs are used. If 1 is given, no parallel computing code
       is used at all, which is useful for debugging. For n_jobs below -1,
       (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but
       one are used.

    as_df : boolean, def True
       Whether to return a dataframe


    Attributes
    ----------
    shift_ : dict
       The shifts for each feature needed to shift the min value in 
       the feature up to at least 0.0, as every element must be positive

    lambda_ : dict
       The lambda values corresponding to each feature
    """
    
    def __init__(self, cols=None, n_jobs=1, as_df=True):
        self.cols = cols
        self.n_jobs = n_jobs
        self.as_df = as_df
        
    def fit(self, X, y = None):
        """Estimate the lambdas, provided X
        
        Parameters
        ----------
        X : pandas DF, shape [n_samples, n_features]
            The data used for estimating the lambdas
        
        y : Passthrough for Pipeline compatibility
        """
        # check on state of X and cols
        X, self.cols = validate_is_pd(X, self.cols)
        
        # ensure enough rows
        _validate_rows(X)
        

        ## First step is to compute all the shifts needed, then add back to X...
        min_Xs = X[self.cols].min(axis = 0)
        shift = np.array([np.abs(x) + 1e-6 if x <= 0.0 else 0.0 for x in min_Xs])
        X[self.cols] += shift

        ## now put shift into a dict
        self.shift_ = dict(zip(self.cols, shift))
        
        ## Now estimate the lambdas in parallel
        self.lambda_ = dict(zip(self.cols, 
            Parallel(n_jobs=self.n_jobs)(
                delayed(_estimate_lambda_single_y)
                (X[i].tolist()) for i in self.cols)))

        return self
    
    def transform(self, X, y = None):
        """Perform Box-Cox transformation
        
        Parameters
        ----------
        X : pandas DF, shape [n_samples, n_features]
            The data to transform
        """
        check_is_fitted(self, 'shift_')
        # check on state of X and cols
        X, _ = validate_is_pd(X, self.cols)
        
        _, n_features = X.shape
        lambdas_, shifts_ = self.lambda_, self.shift_
            
        ## Add the shifts in, and if they're too low,
        ## we have to truncate at some low value: 1e-6
        for nm in self.cols:
            X[nm] += shifts_[nm]
        
        ## If the shifts are too low, truncate...
        X = X.apply(lambda x: x.apply(lambda y: np.maximum(1e-6, y)))

        ## do transformations
        for nm in self.cols:
            X[nm] = _transform_y(X[nm].tolist(), lambdas_[nm])

        return X if self.as_df else X.as_matrix()

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
    ## ensure np array
    y = np.array(y)
    y_prime = np.array(map(lambda x: (np.power(x, lam)-1)/lam if not _eqls(lam,ZERO) else np.log(x), y))

    ## rarely -- very rarely -- we can get a NaN. Why?
    return y_prime
    
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
    
    ## ensure is array
    y = np.array(y)

    ## Use scipy's log-likelihood estimator
    b = boxcox(y, lmbda = None)
    
    ## Return lambda corresponding to maximum P
    return b[1]






###############################################################################
class YeoJohnsonTransformer(BaseEstimator, TransformerMixin, SelectiveMixin):
    """Estimate a lambda parameter for each feature, and transform
       it to a distribution more-closely resembling a Gaussian bell
       using the Yeo-Johnson transformation.

    Parameters
    ----------
    cols : array_like, str
       The columns which to transform

    n_jobs : int, 1 by default
       The number of jobs to use for the computation. This works by
       estimating each of the feature lambdas in parallel.

       If -1 all CPUs are used. If 1 is given, no parallel computing code
       is used at all, which is useful for debugging. For n_jobs below -1,
       (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but
       one are used.

    as_df : boolean, def True
       Whether to return a dataframe


    Attributes
    ----------
    lambda_ : dict
       The lambda values corresponding to each feature
    """

    def __init__(self, cols=None, n_jobs=1, as_df=True):
        self.cols = cols
        self.n_jobs = n_jobs
        self.as_df = as_df

    def fit(self, X, y = None):
        """Estimate the lambdas, provided X

        Parameters
        ----------
        X : pandas DF, shape [n_samples, n_features]
            The data used for estimating the lambdas

        y : Passthrough for Pipeline compatibility
        """
        # check on state of X and cols
        X, self.cols = validate_is_pd(X, self.cols)

        # ensure enough rows
        _validate_rows(X)


        ## Now estimate the lambdas in parallel
        self.lambda_ = dict(zip(self.cols,
            Parallel(n_jobs=self.n_jobs)(
                delayed(_yj_estimate_lambda_single_y)
                (X[nm]) for nm in self.cols)))

        return self

    def transform(self, X, y = None):
        """Perform Yeo-Johnson transformation

        Parameters
        ----------
        X : pandas DF, shape [n_samples, n_features]
            The data to transform
        """
        check_is_fitted(self, 'lambda_')
        # check on state of X and cols
        X, _ = validate_is_pd(X, self.cols)

        lambdas_ = self.lambda_

        ## do transformations
        for nm in self.cols:
            X[nm] = _yj_transform_y(X[nm], lambdas_[nm])

        return X if self.as_df else X.as_matrix()


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
    y = np.array(y)
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
    y = np.array(y)
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






class SpatialSignTransformer(BaseEstimator, TransformerMixin, SelectiveMixin):
    """Project the feature space of a matrix into a multi-dimensional sphere
    by dividing each feature by its squared norm.
       
    Parameters
    ----------
    cols : array_like, str
       The columns which to transform

    n_jobs : int, 1 by default
       The number of jobs to use for the computation. This works by
       estimating each of the feature lambdas in parallel.
       
       If -1 all CPUs are used. If 1 is given, no parallel computing code
       is used at all, which is useful for debugging. For n_jobs below -1,
       (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but
       one are used.

    as_df : boolean, def True
       Whether to return a dataframe


    Attributes
    ----------
    sq_nms_ : dict
       The squared norms for each feature
    """
    
    def __init__(self, cols=None, n_jobs=1, as_df=True):
        self.cols = cols
        self.n_jobs = n_jobs
        self.as_df = as_df
        
    def fit(self, X, y = None):
        """Estimate the squared norms for each feature, provided X
        
        Parameters
        ----------
        X : pd DF, shape [n_samples, n_features]
            The data used for estimating the lambdas
        
        y : Passthrough for Pipeline compatibility
        """
        # check on state of X and cols
        X, self.cols = validate_is_pd(X, self.cols)
        
        ## Now estimate the lambdas in parallel
        self.sq_nms_ = dict(zip(self.cols,
            Parallel(n_jobs=self.n_jobs)(
                delayed(_sq_norm_single)
                (X[nm]) for nm in self.cols)))

        ## What if a squared norm is zero? We want to avoid a divide-by-zero situation...
        for k,v in six.iteritems(self.sq_nms_):
            if v == 0.0:
                self.sq_nms_[k] = np.inf
        
        return self

    def transform(self, X, y = None):
        """Perform spatial sign transformation
        
        Parameters
        ----------
        X : pd DF, shape [n_samples, n_features]
            The data to transform
        """
        check_is_fitted(self, 'sq_nms_')

        # check on state of X and cols
        X, _ = validate_is_pd(X, self.cols)
        sq_nms_ = self.sq_nms_

        ## scale by norms
        for nm in self.cols:
            X[nm] /= sq_nms_[nm]
        
        return X if self.as_df else X.as_matrix()


def _sq_norm_single(x):
    x = np.array(x)
    return np.dot(x, x)




