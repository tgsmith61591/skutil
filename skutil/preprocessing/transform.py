from __future__ import print_function, absolute_import, division
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import boxcox
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import six
from sklearn.externals.joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from skutil.base import *
from ..utils import *
from abc import ABCMeta

__all__ = [
    'BoxCoxTransformer',
    'FunctionMapper',
    'InteractionTermTransformer',
    'SelectiveScaler',
    'SpatialSignTransformer',
    'YeoJohnsonTransformer'
]

# A very small number used to measure differences.
# If the absolute difference between two numbers is
# <= EPS, it is considered equal.
EPS = 1e-12

# A very small number used to represent zero.
ZERO = 1e-16


# Helper funtions:
def _eqls(lam, v):
    return np.abs(lam - v) <= EPS


def _validate_rows(X):
    m, n = X.shape
    if m < 2:
        raise ValueError('n_samples should be at least two, but got %i' % m)


class _BaseSelectiveTransformer(six.with_metaclass(ABCMeta, BaseEstimator, 
                                                   TransformerMixin, 
                                                   SelectiveMixin)):
    """Base class for skutil transformers"""
    def __init__(self, cols=None, as_df=True):
        self.cols = cols
        self.as_df = as_df


class FunctionMapper(_BaseSelectiveTransformer):
    """Apply a function to a column or set of columns.

    Parameters
    ----------

    cols : array_like, optional (default=None)
        The columns on which the transformer will be ``fit``. In
        the case that ``cols`` is None, the transformer will be fit
        on all columns. Note that since this transformer can only operate
        on numeric columns, not explicitly setting the ``cols`` parameter
        may result in errors for categorical data.

    fun : function, default None
        The function to apply to the feature(s)
    """

    def __init__(self, cols=None, fun=None, **kwargs):
        super(FunctionMapper, self).__init__(cols=cols)

        self.fun = fun
        self.kwargs = kwargs

    def fit(self, X, y=None):
        """Validate the args
        
        Parameters
        ----------

        X : pandas DF, shape [n_samples, n_features]
            The data used for estimating the lambdas
        
        y : Passthrough for Pipeline compatibility

        Returns
        -------

        self
        """
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

    def transform(self, X, y=None):
        """Apply the transformation to test data.

        Parameters
        ----------

        X : Pandas DataFrame
            The Pandas frame to transform.

        y : None
            Passthrough for ``sklearn.pipeline.Pipeline``. Even
            if explicitly set, will not change behavior of ``fit``.

        Returns
        -------

        X : pd.DataFrame or np.ndarray
            The transformed matrix
        """
        X, cols = validate_is_pd(X, self.cols)
        cols = cols if not cols is None else X.columns

        # apply the function
        X[cols] = X[cols].apply(lambda x: self.fun(x, **self.kwargs))
        return X


def _mul(a, b):
    """Multiplies two series objects
    (no validation since internally used).

    Parameters
    ----------

    a : pd.Series
    b : pd.Series

    Returns
    -------

    product np.ndarray
    """
    return (a * b).values


class InteractionTermTransformer(_BaseSelectiveTransformer):
    """A class that will generate interaction terms between selected columns.
    An interaction captures some relationship between two independent variables
    in the form of In = (xi * xj).

    Parameters
    ----------

    cols : array_like, optional (default=None)
        The columns on which the transformer will be ``fit``. In
        the case that ``cols`` is None, the transformer will be fit
        on all columns. Note that since this transformer can only operate
        on numeric columns, not explicitly setting the ``cols`` parameter
        may result in errors for categorical data.

    as_df : bool, optional (default=True)
        Whether to return a Pandas DataFrame in the ``transform``
        method. If False, will return a NumPy ndarray instead. 
        Since most skutil transformers depend on explicitly-named
        DataFrame features, the ``as_df`` parameter is True by default.

    interaction : callable, optional (default=None)
        A callable for interactions. Default None will
        result in multiplication of two Series objects

    name_suffix : str, optional (default='I')
        The suffix to add to the new feature name in the form of
        <feature_x>_<feature_y>_<suffix>

    only_return_interactions : bool, optional (default=False)
        If set to True, will only return features in feature_names
        and their respective generated interaction terms.

    Attributes
    ----------

    fun_ : callable
        The interaction term function
    """

    def __init__(self, cols=None, as_df=True, interaction_function=None,
                 name_suffix='I', only_return_interactions=False):

        super(InteractionTermTransformer, self).__init__(cols=cols, as_df=as_df)
        self.interaction_function = interaction_function
        self.name_suffix = name_suffix
        self.only_return_interactions = only_return_interactions

    def fit(self, X, y=None):
        """Fit the transformer.

        Parameters
        ----------

        X : Pandas DataFrame
            The Pandas frame to fit. The frame will only
            be fit on the prescribed ``cols`` (see ``__init__``) or
            all of them if ``cols`` is None.

        y : None
            Passthrough for ``sklearn.pipeline.Pipeline``. Even
            if explicitly set, will not change behavior of ``fit``.

        Returns
        -------

        self
        """
        X, cols = validate_is_pd(X, self.cols)
        self.cols = X.columns if not self.cols else self.cols
        self.fun_ = self.interaction_function if not self.interaction_function is None else _mul

        # validate function
        if not hasattr(self.fun_, '__call__'):
            raise TypeError('require callable for interaction_function')

        # validate cols
        if len(self.cols) < 2:
            raise ValueError('need at least two columns')

        return self

    def transform(self, X):
        """Apply the transformation to test data.

        Parameters
        ----------

        X : Pandas DataFrame
            The Pandas frame to transform.

        Returns
        -------

        X : pd.DataFrame or np.ndarray
            The transformed matrix
        """
        check_is_fitted(self, 'fun_')
        X, cols = validate_is_pd(X, self.cols)
        n_features = len(cols)
        suff = self.name_suffix

        fun = self.fun_
        append_dict = {}
        interaction_names = [x for x in cols]

        # we can do this in N^2 or we can do it in an uglier N choose 2...
        for i in range(n_features - 1):
            for j in range(i + 1, n_features):
                col_i, col_j = cols[i], cols[j]
                new_nm = '%s_%s_%s' % (col_i, col_j, suff)
                append_dict[new_nm] = fun(X[col_i], X[col_j])
                interaction_names.append(new_nm)

        # create DF 2:
        df2 = pd.DataFrame.from_dict(append_dict)
        X = pd.concat([X, df2], axis=1)

        # if we only want to keep interaction names, filter now
        X = X if not self.only_return_interactions else X[interaction_names]

        # return matrix if needed
        return X if self.as_df else X.as_matrix()


class SelectiveScaler(_BaseSelectiveTransformer):
    """A class that will apply scaling only to a select group
    of columns. Useful for data that contains categorical features
    that have not yet been dummied, for dummied features that we
    may not want to scale, or for any already-in-scale features.
    Perhaps, even, there are some features you'd like to impute in
    a different manner than others. This, then, allows two back-to-back
    SelectiveScalers with different columns & strategies in a pipeline object.

    Parameters
    ----------

    cols : array_like, optional (default=None)
        The columns on which the transformer will be ``fit``. In
        the case that ``cols`` is None, the transformer will be fit
        on all columns. Note that since this transformer can only operate
        on numeric columns, not explicitly setting the ``cols`` parameter
        may result in errors for categorical data.

    scaler : instance of a sklearn Scaler, default StandardScaler

    as_df : bool, optional (default=True)
        Whether to return a Pandas DataFrame in the ``transform``
        method. If False, will return a NumPy ndarray instead. 
        Since most skutil transformers depend on explicitly-named
        DataFrame features, the ``as_df`` parameter is True by default.
    """

    def __init__(self, cols=None, scaler=StandardScaler(), as_df=True):
        super(SelectiveScaler, self).__init__(cols=cols, as_df=as_df)
        self.scaler = scaler

    def fit(self, X, y=None, **kwargs):
        """Fit the transformer.

        Parameters
        ----------

        X : Pandas DataFrame
            The Pandas frame to fit. The frame will only
            be fit on the prescribed ``cols`` (see ``__init__``) or
            all of them if ``cols`` is None.

        y : None
            Passthrough for ``sklearn.pipeline.Pipeline``. Even
            if explicitly set, will not change behavior of ``fit``.

        Returns
        -------

        self
        """
        # check on state of X and cols
        X, self.cols = validate_is_pd(X, self.cols)
        cols = X.columns if not self.cols else self.cols

        # throws exception if the cols don't exist
        self.scaler.fit(X[cols])
        return self

    def transform(self, X, y=None):
        """Apply the transformation to test data.

        Parameters
        ----------

        X : Pandas DataFrame
            The Pandas frame to transform.

        y : None
            Passthrough for ``sklearn.pipeline.Pipeline``. Even
            if explicitly set, will not change behavior of ``fit``.

        Returns
        -------

        X : pd.DataFrame or np.ndarray
            The transformed matrix
        """
        # check on state of X and cols
        X, _ = validate_is_pd(X, self.cols)
        cols = X.columns if not self.cols else self.cols

        # Fails through if cols don't exist or if the scaler isn't fit yet
        X[cols] = self.scaler.transform(X[cols])
        return X if self.as_df else X.as_matrix()


class BoxCoxTransformer(_BaseSelectiveTransformer):
    """Estimate a lambda parameter for each feature, and transform
       it to a distribution more-closely resembling a Gaussian bell
       using the Box-Cox transformation. By default, will ignore sparse
       features that are generated via the OneHotCategoricalTransformer.
       
    Parameters
    ----------

    cols : array_like, optional (default=None)
        The columns on which the transformer will be ``fit``. In
        the case that ``cols`` is None, the transformer will be fit
        on all columns. Note that since this transformer can only operate
        on numeric columns, not explicitly setting the ``cols`` parameter
        may result in errors for categorical data.

    n_jobs : int, 1 by default
       The number of jobs to use for the computation. This works by
       estimating each of the feature lambdas in parallel.
       
       If -1 all CPUs are used. If 1 is given, no parallel computing code
       is used at all, which is useful for debugging. For n_jobs below -1,
       (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but
       one are used.

    as_df : bool, optional (default=True)
        Whether to return a Pandas DataFrame in the ``transform``
        method. If False, will return a NumPy ndarray instead. 
        Since most skutil transformers depend on explicitly-named
        DataFrame features, the ``as_df`` parameter is True by default.


    Attributes
    ----------

    shift_ : dict
       The shifts for each feature needed to shift the min value in 
       the feature up to at least 0.0, as every element must be positive

    lambda_ : dict
       The lambda values corresponding to each feature
    """

    def __init__(self, cols=None, n_jobs=1, as_df=True):
        super(BoxCoxTransformer, self).__init__(cols=cols, as_df=as_df)
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Fit the transformer.

        Parameters
        ----------

        X : Pandas DataFrame
            The Pandas frame to fit. The frame will only
            be fit on the prescribed ``cols`` (see ``__init__``) or
            all of them if ``cols`` is None.

        y : None
            Passthrough for ``sklearn.pipeline.Pipeline``. Even
            if explicitly set, will not change behavior of ``fit``.

        Returns
        -------

        self
        """
        # check on state of X and cols
        X, self.cols = validate_is_pd(X, self.cols)
        cols = X.columns if not self.cols else self.cols

        # ensure enough rows
        _validate_rows(X)

        # First step is to compute all the shifts needed, then add back to X...
        min_Xs = X[cols].min(axis=0)
        shift = np.array([np.abs(x) + 1e-6 if x <= 0.0 else 0.0 for x in min_Xs])
        X[cols] += shift

        # now put shift into a dict
        self.shift_ = dict(zip(cols, shift))

        # Now estimate the lambdas in parallel
        self.lambda_ = dict(zip(cols,
                                Parallel(n_jobs=self.n_jobs)(
                                    delayed(_estimate_lambda_single_y)
                                    (X[i].tolist()) for i in cols)))

        return self

    def transform(self, X, y=None):
        """Apply the transformation to test data.

        Parameters
        ----------

        X : Pandas DataFrame
            The Pandas frame to transform.

        y : None
            Passthrough for ``sklearn.pipeline.Pipeline``. Even
            if explicitly set, will not change behavior of ``fit``.

        Returns
        -------

        X : pd.DataFrame or np.ndarray
            The transformed matrix
        """
        check_is_fitted(self, 'shift_')
        # check on state of X and cols
        X, _ = validate_is_pd(X, self.cols)

        _, n_features = X.shape
        lambdas_, shifts_ = self.lambda_, self.shift_
        cols = X.columns if not self.cols else self.cols

        # Add the shifts in, and if they're too low,
        # we have to truncate at some low value: 1e-6
        for nm in cols:
            X[nm] += shifts_[nm]

        # If the shifts are too low, truncate...
        X = X.apply(lambda x: x.apply(lambda y: np.maximum(1e-6, y)))

        # do transformations
        for nm in cols:
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
    # ensure np array
    y = np.array(y)
    y_prime = np.array(map(lambda x: (np.power(x, lam) - 1) / lam if not _eqls(lam, ZERO) else np.log(x), y))

    # rarely -- very rarely -- we can get a NaN. Why?
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

    # ensure is array
    y = np.array(y)

    # Use scipy's log-likelihood estimator
    b = boxcox(y, lmbda=None)

    # Return lambda corresponding to maximum P
    return b[1]


class YeoJohnsonTransformer(_BaseSelectiveTransformer):
    """Estimate a lambda parameter for each feature, and transform
       it to a distribution more-closely resembling a Gaussian bell
       using the Yeo-Johnson transformation.

    Parameters
    ----------

    cols : array_like, optional (default=None)
        The columns on which the transformer will be ``fit``. In
        the case that ``cols`` is None, the transformer will be fit
        on all columns. Note that since this transformer can only operate
        on numeric columns, not explicitly setting the ``cols`` parameter
        may result in errors for categorical data.

    n_jobs : int, 1 by default
       The number of jobs to use for the computation. This works by
       estimating each of the feature lambdas in parallel.

       If -1 all CPUs are used. If 1 is given, no parallel computing code
       is used at all, which is useful for debugging. For n_jobs below -1,
       (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but
       one are used.

    as_df : bool, optional (default=True)
        Whether to return a Pandas DataFrame in the ``transform``
        method. If False, will return a NumPy ndarray instead. 
        Since most skutil transformers depend on explicitly-named
        DataFrame features, the ``as_df`` parameter is True by default.


    Attributes
    ----------

    lambda_ : dict
       The lambda values corresponding to each feature
    """

    def __init__(self, cols=None, n_jobs=1, as_df=True):
        super(YeoJohnsonTransformer, self).__init__(cols=cols, as_df=as_df)
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Fit the transformer.

        Parameters
        ----------

        X : Pandas DataFrame
            The Pandas frame to fit. The frame will only
            be fit on the prescribed ``cols`` (see ``__init__``) or
            all of them if ``cols`` is None.

        y : None
            Passthrough for ``sklearn.pipeline.Pipeline``. Even
            if explicitly set, will not change behavior of ``fit``.

        Returns
        -------

        self
        """
        # check on state of X and cols
        X, self.cols = validate_is_pd(X, self.cols)
        cols = X.columns if not self.cols else self.cols

        # ensure enough rows
        _validate_rows(X)

        # Now estimate the lambdas in parallel
        self.lambda_ = dict(zip(cols,
                                Parallel(n_jobs=self.n_jobs)(
                                    delayed(_yj_estimate_lambda_single_y)
                                    (X[nm]) for nm in cols)))

        return self

    def transform(self, X, y=None):
        """Apply the transformation to test data.

        Parameters
        ----------

        X : Pandas DataFrame
            The Pandas frame to transform.

        y : None
            Passthrough for ``sklearn.pipeline.Pipeline``. Even
            if explicitly set, will not change behavior of ``fit``.

        Returns
        -------

        X : pd.DataFrame or np.ndarray
            The transformed matrix
        """
        check_is_fitted(self, 'lambda_')
        # check on state of X and cols
        X, cols = validate_is_pd(X, self.cols)
        cols = cols if not cols is None else X.columns

        lambdas_ = self.lambda_

        # do transformations
        for nm in cols:
            X[nm] = _yj_transform_y(X[nm], lambdas_[nm])

        return X if self.as_df else X.as_matrix()


def _yj_trans_single_x(x, lam):
    if x >= 0:
        # Case 1: x >= 0 and lambda is not 0
        if not _eqls(lam, ZERO):
            return (np.power(x + 1, lam) - 1.0) / lam

        # Case 2: x >= 0 and lambda is zero
        return log(x + 1)
    else:
        # Case 2: x < 0 and lambda is not two
        if not lam == 2.0:
            denom = 2.0 - lam
            numer = np.power((-x + 1), (2.0 - lam)) - 1.0
            return -numer / denom

        # Case 4: x < 0 and lambda is two
        return -log(-x + 1)


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
    # Use customlog-likelihood estimator
    return _yj_normmax(y)


def _yj_normmax(x, brack=(-2, 2)):
    """Compute optimal YJ transform parameter for input data.

    Parameters
    ----------

    x : array_like
       Input array.
    brack : 2-tuple
       The starting interval for a downhill bracket search
    """

    # Use MLE to compute the optimal YJ parameter
    def _mle_opt(x, brack):
        def _eval_mle(lmb, data):
            # Function to minimize
            return -_yj_llf(data, lmb)

        return optimize.brent(_eval_mle, brack=brack, args=(x,))

    return _mle_opt(x, brack)  # _mle(x, brack)


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

    # We can't take the canonical log of data, as there could be
    # zeros or negatives. Thus, we need to shift both distributions
    # up by some artbitrary factor just for the LLF computation
    min_d, min_y = np.min(data), np.min(y)
    if min_d < ZERO:
        shift = np.abs(min_d) + 1
        data += shift

    # Same goes for Y
    if min_y < ZERO:
        shift = np.abs(min_y) + 1
        y += shift

    # Compute mean on potentially shifted data
    y_mean = np.mean(y, axis=0)
    var = np.sum((y - y_mean) ** 2. / N, axis=0)

    # If var is 0.0, we'll get a warning. Means all the
    # values were nearly identical in y, so we will return
    # NaN so we don't optimize for this value of lam
    if 0 == var:
        return np.nan

    # Can't use canonical log due to maybe negatives, so use the truncated log function in utils
    llf = (lmb - 1) * np.sum(log(data), axis=0)
    llf -= N / 2.0 * log(var)

    return llf


class SpatialSignTransformer(_BaseSelectiveTransformer):
    """Project the feature space of a matrix into a multi-dimensional sphere
    by dividing each feature by its squared norm.
       
    Parameters
    ----------

    cols : array_like, optional (default=None)
        The columns on which the transformer will be ``fit``. In
        the case that ``cols`` is None, the transformer will be fit
        on all columns. Note that since this transformer can only operate
        on numeric columns, not explicitly setting the ``cols`` parameter
        may result in errors for categorical data.

    n_jobs : int, 1 by default
       The number of jobs to use for the computation. This works by
       estimating each of the feature lambdas in parallel.
       
       If -1 all CPUs are used. If 1 is given, no parallel computing code
       is used at all, which is useful for debugging. For n_jobs below -1,
       (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but
       one are used.

    as_df : bool, optional (default=True)
        Whether to return a Pandas DataFrame in the ``transform``
        method. If False, will return a NumPy ndarray instead. 
        Since most skutil transformers depend on explicitly-named
        DataFrame features, the ``as_df`` parameter is True by default.


    Attributes
    ----------

    sq_nms_ : dict
       The squared norms for each feature
    """

    def __init__(self, cols=None, n_jobs=1, as_df=True):
        super(SpatialSignTransformer, self).__init__(cols=cols, as_df=as_df)
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Fit the transformer.

        Parameters
        ----------

        X : Pandas DataFrame
            The Pandas frame to fit. The frame will only
            be fit on the prescribed ``cols`` (see ``__init__``) or
            all of them if ``cols`` is None.

        y : None
            Passthrough for ``sklearn.pipeline.Pipeline``. Even
            if explicitly set, will not change behavior of ``fit``.

        Returns
        -------

        self
        """
        # check on state of X and cols
        X, self.cols = validate_is_pd(X, self.cols)
        cols = X.columns if not self.cols else self.cols

        # Now get sqnms in parallel
        self.sq_nms_ = dict(zip(cols,
                                Parallel(n_jobs=self.n_jobs)(
                                    delayed(_sq_norm_single)
                                    (X[nm]) for nm in cols)))

        return self

    def transform(self, X, y=None):
        """Apply the transformation to test data.

        Parameters
        ----------

        X : Pandas DataFrame
            The Pandas frame to transform.

        y : None
            Passthrough for ``sklearn.pipeline.Pipeline``. Even
            if explicitly set, will not change behavior of ``fit``.

        Returns
        -------

        X : pd.DataFrame or np.ndarray
            The transformed matrix
        """
        check_is_fitted(self, 'sq_nms_')

        # check on state of X and cols
        X, _ = validate_is_pd(X, self.cols)
        sq_nms_ = self.sq_nms_

        # scale by norms
        for nm, the_norm in six.iteritems(sq_nms_):
            X[nm] /= the_norm

        return X if self.as_df else X.as_matrix()


def _sq_norm_single(x, zero_action=np.inf):
    x = np.asarray(x)
    nrm = np.dot(x, x)

    # What if a squared norm is zero? We want to
    # avoid a divide-by-zero situation...
    return nrm if not nrm == 0 else zero_action
