from __future__ import print_function, division, absolute_import

from collections import namedtuple

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from .base import _BaseFeatureSelector
from ..utils import validate_is_pd, is_numeric

__all__ = [
    'FeatureDropper',
    'FeatureRetainer',
    'filter_collinearity',
    'MulticollinearityFilterer',
    'NearZeroVarianceFilterer',
    'SparseFeatureDropper'
]


def _validate_cols(cols):
    """Validate that there are at least two columns
    to evaluate. This is used for the MulticollinearityFilterer,
    as it requires there be at least two columns.

    Parameters
    ----------

    cols : array_like or None
        The columns to evaluate
    """

    if cols is not None and len(cols) < 2:
        raise ValueError('too few features')


class SparseFeatureDropper(_BaseFeatureSelector):
    """Retains features that are less sparse (NaN) than
    the provided threshold.

    Parameters
    ----------

    cols : array_like (string)
        The features from which to drop

    threshold : float (default=0.5)
        The threshold of sparsity above which to drop

    as_df : boolean, optional (True default)
        Whether to return a dataframe

    Attributes
    ----------

    sparsity_ : array_like, (n_cols,)
        The array of sparsity values
    
    drop : array_like
        The array of column names to drop
    """

    def __init__(self, cols=None, threshold=0.5, as_df=True):
        super(SparseFeatureDropper, self).__init__(cols=cols, as_df=as_df)
        self.threshold = threshold

    def fit(self, X, y=None):
        X, self.cols = validate_is_pd(X, self.cols)
        thresh = self.threshold

        # validate the threshold
        if not (is_numeric(thresh) and (0.0 <= thresh < 1.0)):
            raise ValueError('thresh must be a float between '
                             '0 (inclusive) and 1. Got %s' % str(thresh))

        # get cols
        cols = self.cols if self.cols is not None else X.columns.tolist()

        # assess sparsity
        self.sparsity_ = X[cols].apply(lambda x: x.isnull().sum() / x.shape[0]).values  # numpy array
        mask = self.sparsity_ > thresh  # numpy boolean array
        self.drop = X.columns[mask].tolist() if mask.sum() > 0 else None
        return self


###############################################################################
class FeatureDropper(_BaseFeatureSelector):
    """A very simple class to be used at the beginning or any stage of a 
    Pipeline that will drop the given features from the remainder of the pipe

    Parameters
    ----------

    cols : array_like (string)
        The features to drop

    as_df : boolean, optional (True default)
        Whether to return a dataframe
    """

    def __init__(self, cols=None, as_df=True):
        super(FeatureDropper, self).__init__(cols=cols, as_df=as_df)

    def fit(self, X, y=None):
        # check on state of X and cols
        _, self.cols = validate_is_pd(X, self.cols)
        self.drop = self.cols
        return self


class FeatureRetainer(_BaseFeatureSelector):
    """A very simple class to be used at the beginning of a Pipeline that will
    only propagate the given features throughout the remainder of the pipe

    Parameters
    ----------
    
    cols : array_like (string)
        The features to select

    as_df : boolean, optional (True default)
        Whether to return a dataframe
    """

    def __init__(self, cols=None, as_df=True):
        super(FeatureRetainer, self).__init__(cols=cols, as_df=as_df)

    def fit(self, X, y=None):
        # check on state of X and cols
        X, self.cols = validate_is_pd(X, self.cols)

        # set the drop as those not in cols
        cols = self.cols if not self.cols is None else []
        self.drop = X.drop(cols, axis=1).columns.tolist()  # these will be the left overs

        return self

    def transform(self, X, y=None):
        # check on state of X and cols
        X, _ = validate_is_pd(X, self.cols)  # copy X
        retained = X[self.cols or X.columns]  # if cols is None, returns all
        return retained if self.as_df else retained.as_matrix()


class _MCFTuple(namedtuple('_MCFTuple', ('feature_x',
                                         'feature_y',
                                         'abs_corr',
                                         'mac'))):
    """A raw namedtuple is very memory efficient as it packs the attributes
    in a struct to get rid of the __dict__ of attributes in particular it
    does not copy the string for the keys on each instance.
    By deriving a namedtuple class just to introduce the __repr__ method we
    would also reintroduce the __dict__ on the instance. By telling the
    Python interpreter that this subclass uses static __slots__ instead of
    dynamic attributes. Furthermore we don't need any additional slot in the
    subclass so we set __slots__ to the empty tuple. """
    __slots__ = tuple()

    def __repr__(self):
        """Simple custom repr to summarize the main info"""
        return "Dropped: {0}, Corr_feature: {1}, abs_corr: {2:.5f}, MAC: {3:.5f}".format(
            self.feature_x,
            self.feature_y,
            self.abs_corr,
            self.mac)


def filter_collinearity(c, threshold):
    """Performs the collinearity filtration for both the
    MulticollinearityFilterer as well as the H2OMulticollinearityFilterer

    Parameters
    ----------

    c : pandas DataFrame
        The correlation matrix

    threshold : float
        The threshold above which to filter

    Returns
    -------

    drops, macor, crrz
        (The drop list, the mean absolute correlations, and the correlation tuples)
    """
    # ensure symmetric
    if c.shape[0] != c.shape[1]:
        raise ValueError('input dataframe should be symmetrical in dimensions')

    # init drops list
    drops = []
    macor = []  # mean abs corrs
    corrz = []  # the correlations

    # Iterate over each feature
    finished = False
    while not finished:

        # Whenever there's a break, this loop will start over
        for i, nm in enumerate(c.columns):
            this_col = c[nm].drop(nm).sort_values(
                na_position='first')  # gets the column, drops the index of itself, and sorts
            this_col_nms = this_col.index.tolist()
            this_col = np.array(this_col)

            # check if last value is over thresh
            max_cor = this_col[-1]
            if pd.isnull(max_cor) or max_cor < threshold or this_col.shape[0] == 1:
                if i == c.columns.shape[0] - 1:
                    finished = True

                # control passes to next column name or end if finished
                continue

            # otherwise, we know the corr is over the threshold
            # gets the current col, and drops the same row, sorts asc and gets other col
            other_col_nm = this_col_nms[-1]
            that_col = c[other_col_nm].drop(other_col_nm)

            # get the mean absolute correlations of each
            mn_1, mn_2 = np.nanmean(this_col), np.nanmean(that_col)

            # we might get nans?
            # if pd.isnull(mn_1) and pd.isnull(mn_2):
            # this condition is literally impossible, as it would
            # require every corr to be NaN, and it wouldn't have
            # even gotten here without hitting the continue block.
            if pd.isnull(mn_1):
                drop_nm = other_col_nm
            elif pd.isnull(mn_2):
                drop_nm = nm
            else:
                drop_nm = nm if mn_1 > mn_2 else other_col_nm

            # drop the bad col, row
            c.drop(drop_nm, axis=1, inplace=True)
            c.drop(drop_nm, axis=0, inplace=True)

            # add the bad col to drops
            drops.append(drop_nm)
            macor.append(np.maximum(mn_1, mn_2))
            corrz.append(_MCFTuple(
                feature_x=drop_nm,
                feature_y=nm if not nm == drop_nm else other_col_nm,
                abs_corr=max_cor,
                mac=macor[-1]
            ))

            # if we get here, we have to break so the loop will 
            # start over from the first (non-popped) column
            break

            # if not finished, restarts loop, otherwise will exit loop

    # return
    return drops, macor, corrz


class MulticollinearityFilterer(_BaseFeatureSelector):
    """Filter out features with a correlation greater than the provided threshold.
    When a pair of correlated features is identified, the mean absolute correlation (MAC)
    of each feature is considered, and the feature with the highsest MAC is discarded.

    Parameters
    ----------

    cols : array_like, string
        The columns used to generate the correlation matrix

    threshold : float, default 0.85
        The threshold above which to filter correlated features

    method : str, one of ['pearson','kendall','spearman'], default 'pearson'
        The method used to compute the correlation

    as_df : boolean, default True
        Whether to return a pandas DataFrame

    Attributes
    ----------

    cols : the cols used to compute the correlation matrix

    drop : list, string
        The columns to drop

    mean_abs_correlations_ : list, float
        The corresponding mean absolute correlations of each drop_ name

    as_df : boolean
        Whether or not to return a dataframe
    """

    def __init__(self, cols=None, threshold=0.85, method='pearson', as_df=True):
        super(MulticollinearityFilterer, self).__init__(cols=cols, as_df=as_df)
        self.threshold = threshold
        self.method = method

    def fit(self, X, y=None):
        """Fit the multicollinearity filterer.

        Parameters
        ----------

        X : pandas DataFrame
            The frame to fit

        y : None, passthrough for pipeline
        """

        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y=None):
        """Fit the multicollinearity filterer and
        return the filtered frame.

        Parameters
        ----------

        X : pandas DataFrame
            The frame to fit

        y : None, passthrough for pipeline
        """

        # check on state of X and cols
        X, self.cols = validate_is_pd(X, self.cols, assert_all_finite=True)
        _validate_cols(self.cols)

        ## Generate correlation matrix
        c = X[self.cols or X.columns].corr(method=self.method).apply(lambda x: np.abs(x))

        ## get drops list
        d, mac, crz = filter_collinearity(c, self.threshold)
        self.drop = d if d else None
        self.mean_abs_correlations_ = mac if mac else None
        self.correlations_ = crz if crz else None

        # if drop is None, we need to just return X
        if not self.drop:
            return X if self.as_df else X.as_matrix()

        dropped = X.drop(self.drop, axis=1)
        return dropped if self.as_df else dropped.as_matrix()

    def transform(self, X, y=None):
        """Drops the highly-correlated features from the new
        input frame.

        Parameters
        ----------

        X : pandas DataFrame
            The frame to transform

        y : None, passthrough for pipeline
        """
        check_is_fitted(self, 'drop')
        # check on state of X and cols
        X, _ = validate_is_pd(X, self.cols)

        # ensure we don't drop None
        if not self.drop:
            return X

        dropped = X.drop(self.drop, axis=1)
        return dropped if self.as_df else dropped.as_matrix()


###############################################################################
class NearZeroVarianceFilterer(_BaseFeatureSelector):
    """Identify and remove any features that have a variance below
    a certain threshold.

    Parameters
    ----------

    cols : array_like, string
        The columns to evaluate for potential drops

    threshold : float, default 1e-6
        The threshold below which to declare "zero variance"

    as_df : boolean, default True
        Whether to return a pandas DataFrame
    """

    def __init__(self, cols=None, threshold=1e-6, as_df=True):
        super(NearZeroVarianceFilterer, self).__init__(cols=cols, as_df=as_df)
        self.threshold = threshold

    def fit(self, X, y=None):
        # check on state of X and cols
        X, self.cols = validate_is_pd(X, self.cols, assert_all_finite=True)
        cols = self.cols if not self.cols is None else X.columns

        # if cols is None, applies over everything
        variances = X[cols].var()
        mask = (variances < self.threshold).values
        self.var_ = variances[mask].tolist()
        self.drop = variances.index[mask].tolist()

        # I don't like making this None; it opens up bugs in pd.drop,
        # but it was the precedent the API set from early on, so don't
        # want to change it without a warning. #TODO: warn of change
        if not self.drop:
            self.drop = None

        return self
