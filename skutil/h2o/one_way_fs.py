from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import warnings
from scipy import special
from .util import _unq_vals_col, rbind_all
from .base import (BaseH2OTransformer, _check_is_frame, 
                   _retain_features, _frame_from_x_y, 
                   validate_x_y)

__all__ = [
    'h2o_f_classif',
    'h2o_f_oneway'
]


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

