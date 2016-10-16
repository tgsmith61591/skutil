from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import warnings
from abc import ABCMeta, abstractmethod
from scipy import special
from sklearn.externals import six
from .split import *
from .select import BaseH2OFeatureSelector
from .util import _unq_vals_col, rbind_all
from .base import (BaseH2OTransformer, _check_is_frame, 
                   _retain_features, _frame_from_x_y, 
                   validate_x_y)

__all__ = [
    'h2o_f_classif',
    'h2o_f_oneway',
    'H2OFScoreSelector'
]

def _validate_alpha(alpha):
    """Ensure that alpha is a float greater than zero
    and less that 1.0.

    Parameters
    ----------

    alpha : float
        The alpha value to validate

    Returns
    -------

    alpha : float
        The validated alpha
    """
    if not 0 < alpha < 1.0:
        raise ValueError('alpha must be a float between 0 and 1')
    return alpha


def h2o_f_classif(X, feature_names, target_feature):
    """Compute the ANOVA F-value for the provided sample.
    This method is adapted from ``sklearn.feature_selection.f_classif``
    to function on H2OFrames.

    Parameters
    ----------

    X : H2OFrame, shape=(n_samples, n_features)
        The feature matrix. Each feature will be tested 
        sequentially.

    y : H2OFrame, shape=(n_samples,)
        The target feature. Should be int or enum, per
        the classification objective.

    Returns
    -------

    f : float
        The computed F-value of the test.

    prob : float
        The associated p-value from the F-distribution.
    """
    frame = _check_is_frame(X)

    # first, get unique values of y
    y = X[target_feature]
    _, unq = _unq_vals_col(y)

    # if y is enum, make the unq strings..
    unq = unq[_] if not y.isfactor()[0] else [str(i) for i in unq[_]]

    # get the masks
    args = [frame[y==k, :][feature_names] for k in unq]
    f, prob= h2o_f_oneway(*args)
    return f, prob


# The following function is a rewriting (of the sklearn rewriting) of 
# scipy.stats.f_oneway. Contrary to the scipy.stats.f_oneway implementation 
# it does not copy the data while keeping the inputs unchanged. Furthermore,
# contrary to the sklearn implementation, it does not use np.ndarrays, rather
# amending 1d H2OFrames inplace.
def h2o_f_oneway(*args):
    """Performs a 1-way ANOVA.
    The one-way ANOVA tests the null hypothesis that 2 or more groups have
    the same population mean. The test is applied to samples from two or
    more groups, possibly with differing sizes.

    Parameters
    ----------

    sample1, sample2, ... : array_like, H2OFrames, shape=(n_classes,)
        The sample measurements should be given as varargs (*args).
        A slice of the original input frame for each class in the
        target feature.

    Returns
    -------

    f : float
        The computed F-value of the test.

    prob : float
        The associated p-value from the F-distribution.

    Notes
    -----

    The ANOVA test has important assumptions that must be satisfied in order
    for the associated p-value to be valid.

    1. The samples are independent
    2. Each sample is from a normally distributed population
    3. The population standard deviations of the groups are all equal. This
       property is known as homoscedasticity.

    If these assumptions are not true for a given set of data, it may still be
    possible to use the Kruskal-Wallis H-test (`scipy.stats.kruskal`_) although
    with some loss of power.

    The algorithm is from Heiman[2], pp.394-7.
    See ``scipy.stats.f_oneway`` (that should give the same results while
    being less efficient) and ``sklearn.feature_selection.f_oneway``.

    References
    ----------

    .. [1] Lowry, Richard.  "Concepts and Applications of Inferential
           Statistics". Chapter 14.
           http://faculty.vassar.edu/lowry/ch14pt1.html

    .. [2] Heiman, G.W.  Research Methods in Statistics. 2002.
    """
    n_classes = len(args)

    # sklearn converts everything to float here. Rather than do so,
    # we will test for total numericism and fail out if it's not 100%
    # numeric.
    if not all([all([X.isnumeric() for X in args])]):
        raise ValueError("All features must be entirely numeric for F-test")


    n_samples_per_class = [X.shape[0] for X in args]
    n_samples = np.sum(n_samples_per_class)

    # compute the sum of squared values in each column, and then compute the column
    # sums of all of those intermittent rows rbound together
    ss_alldata = rbind_all(*[X.apply(lambda x: (x*x).sum()) for X in args]).apply(lambda x: x.sum())

    # compute the sum of each column for each X in args, then rbind them all
    # and sum them up, finally squaring them. Tantamount to the squared sum
    # of each complete column. Note that we need to add a tiny fraction to ensure
    # all are real numbers for the rbind...
    sum_args = [X.apply(lambda x: x.sum() + 1e-12).asnumeric() for X in args] # col sums
    square_of_sums_alldata = rbind_all(*sum_args).apply(lambda x: x.sum())
    square_of_sums_alldata *= square_of_sums_alldata

    square_of_sums_args = [s*s for s in sum_args]
    sstot = ss_alldata - square_of_sums_alldata / float(n_samples)

    ssbn = None # h2o frame
    for k, _ in enumerate(args):
        tmp = square_of_sums_args[k] / n_samples_per_class[k]
        ssbn = tmp if ssbn is None else (ssbn + tmp)

    ssbn -= square_of_sums_alldata / float(n_samples)
    sswn = sstot - ssbn
    dfbn = n_classes - 1
    dfwn = n_samples - n_classes
    msb = ssbn / float(dfbn)
    msw = sswn / float(dfwn)

    constant_feature_idx = (msw == 0)
    constant_feature_sum = constant_feature_idx.sum() # sum of ones
    nonzero_size = (msb != 0).sum()
    if (nonzero_size != msb.shape[1] and constant_feature_sum):
        warnings.warn("Features %s are constant." % np.arange(msw.shape[1])[constant_feature_idx],
            UserWarning)

    f = (msb / msw)

    # convert to numpy ndarray for special
    f = np.asarray(f.as_data_frame(use_pandas=False)[1]).astype(np.float)

    # compute prob
    prob = special.fdtrc(dfbn, dfwn, f)
    return f, prob


class _H2OBaseUnivariateSelector(six.with_metaclass(ABCMeta, 
                                                    BaseH2OFeatureSelector)):
    """The base class for all univariate feature selectors in H2O.

    Parameters
    ----------

    feature_names : array_like (str), optional (default=None)
        The list of names on which to fit the transformer.

    target_feature : str, optional (default=None)
        The name of the target feature (is excluded from the fit)
        for the estimator.

    exclude_features : iterable or None, optional (default=None)
        Any names that should be excluded from ``feature_names``

    cv : int or ``H2OBaseCrossValidator``, optional (default=3)
        Univariate feature selection can very easily remove
        features erroneously or cause overfitting. Using cross
        validation, we can more confidently select the features 
        to drop.

    alpha : float, optional (default=0.05)
        The significance level below which to consider a feature
        "significant."

    iid : bool, optional (default=True)
        Whether to consider each fold as IID. The fold scores
        are normalized at the end by the number of observations
        in each fold

    min_version : str, float (default 'any')
        The minimum version of h2o that is compatible with the transformer

    max_version : str, float (default None)
        The maximum version of h2o that is compatible with the transformer
    """
    @abstractmethod
    def __init__(self, feature_names=None, target_feature=None,
                 exclude_features=None, cv=3, alpha=0.05, iid=True,
                 min_version='any', max_version=None):
        super(_H2OBaseUnivariateSelector, self).__init__(
            feature_names=feature_names, target_feature=target_feature,
            exclude_features=exclude_features, min_version=self._min_version,
            max_version=self._max_version)

        # validate CV
        self.cv = cv
        self.alpha = alpha


def _repack_tuple(two, one):
    """Utility for ``_test_and_score``.
    Packs the scores, p-values and train-fold length
    into a single, flat tuple.

    Parameters
    ----------

    two : tuple, shape=(2,)
        The scores & p-values tuple

    one : int
        The train fold length

    Returns
    -------

    out : tuple, shape=(3,)
        The flattened tuple: (F-scores, p-values, 
        train-fold size)
    """
    return (two[0], two[1], one)


def _test_and_score(frame, fun, cv, feature_names, target_feature, iid):
    """Fit all the folds of some provided function, repack the scores
    tuple and adjust the fold score if ``iid`` is True.

    Parameters
    ----------

    frame : H2OFrame
            The frame to fit

    fun : callable
        The function to call

    cv : ``H2OBaseCrossValidator``
        The cross validation class

    feature_names : array_like (str), optional (default=None)
        The list of names on which to fit the transformer.

    target_feature : str, optional (default=None)
        The name of the target feature (is excluded from the fit)
        for the estimator.

    iid : bool, optional (default=True)
        Whether to consider each fold as IID. The fold scores
        are normalized at the end by the number of observations
        in each fold

    Returns
    -------

    all_scores : np.ndarray
        The normalized scores

    all_pvalues : np.ndarray
        The normalized p-values
    """

    fn, tf = feature_names, target_feature
    scores = [
        _repack_tuple(fun(frame[train,:], fn, tf), len(train))
        for train, _ in cv.split(frame, tf)
    ]

    # compute the mean F-score, p-value, adjust with IID
    n_folds = cv.get_n_splits()
    all_scores = 0.
    all_pvalues = 0.

    # adjust the fold scores
    for these_scores, p_vals, fold_size in scores:
        if iid:
            these_scores *= fold_size
            p_vals *= fold_size
        all_scores += these_scores
        all_pvalues += p_vals

    if iid:
        all_scores /= frame.shape[0]
        all_pvalues /= frame.shape[0]
    else:
        all_scores /= float(n_folds)
        all_pvalues /= float(n_folds)

    return all_scores, all_pvalues


class H2OFScoreSelector(_H2OBaseUnivariateSelector):
    """Select features based on the F-score, using the 
    ``h2o_f_classif`` method.

    Parameters
    ----------

    feature_names : array_like (str), optional (default=None)
        The list of names on which to fit the transformer.

    target_feature : str, optional (default=None)
        The name of the target feature (is excluded from the fit)
        for the estimator.

    exclude_features : iterable or None, optional (default=None)
        Any names that should be excluded from ``feature_names``

    cv : int or ``H2OBaseCrossValidator``, optional (default=3)
        Univariate feature selection can very easily remove
        features erroneously or cause overfitting. Using cross
        validation, we can more confidently select the features 
        to drop.

    alpha : float, optional (default=0.05)
        The significance level below which to consider a feature
        "significant."

    iid : bool, optional (default=True)
        Whether to consider each fold as IID. The fold scores
        are normalized at the end by the number of observations
        in each fold

    min_version : str, float (default 'any')
        The minimum version of h2o that is compatible with the transformer

    max_version : str, float (default None)
        The maximum version of h2o that is compatible with the transformer
    """

    _min_version = '3.8.2.9'
    _max_version = None

    def __init__(self, feature_names=None, target_feature=None,
                 exclude_features=None, cv=3, alpha=0.05, iid=True):
        super(H2OFScoreSelector, self).__init__(
            feature_names=feature_names, target_feature=target_feature,
            exclude_features=exclude_features, cv=cv, alpha=alpha,
            iid=iid, min_version=self._min_version, 
            max_version=self._max_version)

    def fit(self, X):
        """Fit the F-score feature selector.

        Parameters
        ----------

        X : H2OFrame
            The frame to fit

        Returns
        -------

        self
        """
        # we can use this to extract the feature names to pass...
        feature_names = _frame_from_x_y(
            X=X, x=feature_names=self.feature_names, 
            y=self.target_feature, 
            exclude=self.exclude_features).columns

        cv = check_cv(self.cv)
        _validate_alpha(self.alpha) # need to validate is in range

        # use the X frame (full frame) including target
        self.f_scores_, self.p_values_ = _test_and_score(
            frame=X, fun=h2o_f_classif, cv=cv, 
            feature_names=feature_names, # extracted above
            target_feature=self.target_feature, 
            iid=self.iid)


        # set self.drop_
        # TODO:
        return self

