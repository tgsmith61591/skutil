from __future__ import print_function, division, absolute_import
import warnings
import numpy as np
from sklearn.utils.validation import check_is_fitted
from h2o.frame import H2OFrame
from ..feature_selection import filter_collinearity
from ..utils import is_numeric
from .base import (BaseH2OTransformer, _check_is_frame, _retain_features, _frame_from_x_y)

__all__ = [
    'BaseH2OFeatureSelector',
    'H2OFeatureDropper',
    'H2OMulticollinearityFilterer',
    'H2ONearZeroVarianceFilterer',
    'H2OSparseFeatureDropper'
]


def _validate_use(X, use, na_warn):
    """For H2OMulticollinearityFilterer and H2ONearZeroVarianceFilterer,
    validate that our 'use' arg is appropriate given the presence of NA
    values in the H2OFrame.

    Parameters
    ----------

    X : H2OFrame
        The frame to evaluate. Since this is an internal method,
        no validation is done to ensure it is, in fact, an H2OFrame

    use : str, one of ('complete.obs', 'all.obs', 'everything')
        The 'use' argument passed to the transformer

    na_warn : bool
        Whether to warn if there are NAs present in the frame. If there are,
        and na_warn is set to False, the function will use the provided use,
        however, if na_warn is True and there are NA values present it will
        raise a warning and use 'complete.obs'

    Returns
    -------

    use : string
        The appropriate use string
    """
    # validate use
    _valid_use = ['complete.obs', 'all.obs', 'everything']
    if use not in _valid_use:
        raise ValueError('expected one of (%s) but got %s' % (', '.join(_valid_use), use))

    # check on NAs
    if use == 'complete.obs':
        pass
    elif na_warn:  # only warn if not using complete.obs
        nasum = X.isna().sum()
        if nasum > 0:
            warnings.warn('%i NA value(s) in frame; using "complete.obs"' % nasum)
            use = 'complete.obs'

    return use


class BaseH2OFeatureSelector(BaseH2OTransformer):
    """Base class for all H2O selectors.

    Parameters
    ----------

    feature_names : array_like (str), optional (default=None)
        The list of names on which to fit the transformer.

    target_feature : str, optional (default None)
        The name of the target feature (is excluded from the fit)
        for the estimator.

    exclude_features : iterable or None, optional (default=None)
        Any names that should be excluded from ``feature_names``

    min_version : str, float (default 'any')
        The minimum version of h2o that is compatible with the transformer

    max_version : str, float (default None)
        The maximum version of h2o that is compatible with the transformer
    """

    def __init__(self, feature_names=None, target_feature=None, exclude_features=None,
                 min_version='any', max_version=None):
        super(BaseH2OFeatureSelector, self).__init__(feature_names=feature_names,
                                                     target_feature=target_feature,
                                                     exclude_features=exclude_features,
                                                     min_version=min_version,
                                                     max_version=max_version)

    def transform(self, X):
        """Transform the test frame, after fitting
        the transformer.

        Parameters
        ----------

        X : H2OFrame
            The test frame

        Returns
        -------

        X : H2OFrame
            The transformed frame
        """
        # validate state, frame
        check_is_fitted(self, 'drop_')
        X = _check_is_frame(X)
        return X[_retain_features(X, self.drop_)]


class H2OFeatureDropper(BaseH2OFeatureSelector):
    """A very simple class to be used at the beginning or any stage of an
    H2OPipeline that will drop the given features from the remainder of the pipe.

    This is useful when you have many features, but only a few to drop.
    Rather than passing the feature_names arg as the delta between many
    features and the several to drop, this allows you to drop them and keep
    feature_names as None in future steps.

    Parameters
    ----------

    feature_names : array_like (str), optional (default=None)
        The list of names on which to fit the transformer.

    target_feature : str, optional (default None)
        The name of the target feature (is excluded from the fit)
        for the estimator.

    exclude_features : iterable or None, optional (default=None)
        Any names that should be excluded from ``feature_names``

    exclude_features : iterable or None
        Any names that should be excluded from ``feature_names``

    Attributes
    ----------

    feature_names
        These are the features that will be dropped by 
        the ``FeatureDropper``
    """

    def __init__(self, feature_names=None, target_feature=None, exclude_features=None):
        super(H2OFeatureDropper, self).__init__(feature_names=feature_names,
                                                target_feature=target_feature,
                                                exclude_features=exclude_features)

    def fit(self, X):
        """Fit the H2OTransformer.

        Parameters
        ----------

        X : H2OFrame
            The H2OFrame that will be fit.

        Returns
        -------

        return self
        """
        fn = self.feature_names
        if fn is None:
            fn = []

        # We validate the features_names is a list or iterable
        if hasattr(fn, '__iter__'):
            self.drop_ = [i for i in fn]
        else:
            raise ValueError('expected iterable for feature_names')

        return self


class H2OSparseFeatureDropper(BaseH2OFeatureSelector):
    """Retains features that are less sparse (NA) than
    the provided threshold.

    Parameters
    ----------

    feature_names : array_like (str), optional (default=None)
        The list of names on which to fit the transformer.

    target_feature : str, optional (default None)
        The name of the target feature (is excluded from the fit)
        for the estimator.

    exclude_features : iterable or None, optional (default=None)
        Any names that should be excluded from ``feature_names``

    threshold : float (default=0.5)
        The threshold of sparsity above which to drop

    Attributes
    ----------

    sparsity_ : array_like, (n_cols,)
        The array of sparsity values

    drop_ : array_like
        The array of column names to drop
    """

    _min_version = '3.8.2.9'
    _max_version = None

    def __init__(self, feature_names=None, target_feature=None, exclude_features=None, threshold=0.5):
        super(H2OSparseFeatureDropper, self).__init__(feature_names=feature_names,
                                                      target_feature=target_feature,
                                                      exclude_features=exclude_features,
                                                      min_version=self._min_version,
                                                      max_version=self._max_version)
        self.threshold = threshold

    def fit(self, X):
        """Fit the H2OTransformer.

        Parameters
        ----------

        X : H2OFrame
            The H2OFrame that will be fit.

        Returns
        -------

        return self
        """
        frame, thresh = _check_is_frame(X), self.threshold
        frame = _frame_from_x_y(frame, self.feature_names, self.target_feature, self.exclude_features)

        # validate the threshold
        if not (is_numeric(thresh) and (0.0 <= thresh < 1.0)):
            raise ValueError('thresh must be a float between '
                             '0 (inclusive) and 1. Got %s' % str(thresh))

        df = (frame.isna().apply(lambda x: x.sum()) / frame.shape[0]).as_data_frame(use_pandas=True)
        df.columns = frame.columns
        ser = df.T[0]

        self.drop_ = [str(x) for x in ser.index[ser > thresh]]
        self.sparsity_ = ser.values  # numpy array of sparsities

        return self


class H2OMulticollinearityFilterer(BaseH2OFeatureSelector):
    """Filter out features with a correlation greater than the provided threshold.
    When a pair of correlated features is identified, the mean absolute correlation (MAC)
    of each feature is considered, and the feature with the highsest MAC is discarded.

    Parameters
    ----------

    feature_names : array_like (str), optional (default=None)
        The list of names on which to fit the transformer.

    target_feature : str, optional (default None)
        The name of the target feature (is excluded from the fit)
        for the estimator.

    exclude_features : iterable or None, optional (default=None)
        Any names that should be excluded from ``feature_names``

    threshold : float, (default=0.85)
        The threshold above which to filter correlated features

    na_warn : bool (default True)
        Whether to warn if any NAs are present

    na_rm : bool (default False)
        Whether to remove NA values

    use : str (default "complete.obs"), one of {'complete.obs','all.obs','everything'}
        A string indicating how to handle missing values.

    Attributes
    ----------

    drop_ : list, string
        The columns to drop

    mean_abs_correlations_ : list, float
        The corresponding mean absolute correlations of each drop_ name

    correlations_ : named tuple
        A list of tuples with each tuple containing the two correlated features, 
        the level of correlation, the feature that was selected for dropping, and
        the mean absolute correlation of the dropped feature.
    """

    _min_version = '3.8.2.9'
    _max_version = None

    def __init__(self, feature_names=None, target_feature=None, exclude_features=None,
                 threshold=0.85, na_warn=True, na_rm=False, use='complete.obs'):
        super(H2OMulticollinearityFilterer, self).__init__(feature_names=feature_names,
                                                           target_feature=target_feature,
                                                           exclude_features=exclude_features,
                                                           min_version=self._min_version,
                                                           max_version=self._max_version)
        self.threshold = threshold
        self.na_warn = na_warn
        self.na_rm = na_rm
        self.use = use

    def fit(self, X):
        """Fit the H2OTransformer.

        Parameters
        ----------

        X : H2OFrame
            The H2OFrame that will be fit.

        Returns
        -------

        return self
        """
        self.fit_transform(X)
        return self

    def fit_transform(self, X):
        """Fit the multicollinearity filterer and
        return the transformed H2OFrame, X.

        Parameters
        ----------

        X : H2OFrame
            The frame to fit

        Returns
        -------

        X : H2OFrame
            The transformed frame
        """
        frame, thresh = _check_is_frame(X), self.threshold
        frame = _frame_from_x_y(frame, self.feature_names, self.target_feature, self.exclude_features)

        # validate use, check NAs
        use = _validate_use(frame, self.use, self.na_warn)

        # Generate absolute correlation matrix
        c = frame.cor(use=use, na_rm=self.na_rm).abs().as_data_frame(use_pandas=True)
        c.columns = frame.columns  # set the cols to the same names
        c.index = frame.columns

        # get drops list
        self.drop_, self.mean_abs_correlations_, self.correlations_ = filter_collinearity(c, self.threshold)
        return self.transform(X)


class H2ONearZeroVarianceFilterer(BaseH2OFeatureSelector):
    """Identify and remove any features that have a variance below
    a certain threshold.

    Parameters
    ----------

    feature_names : array_like (str), optional (default=None)
        The list of names on which to fit the transformer.

    target_feature : str, optional (default None)
        The name of the target feature (is excluded from the fit)
        for the estimator.

    exclude_features : iterable or None, optional (default=None)
        Any names that should be excluded from ``feature_names``

    threshold : float, default 1e-6
        The threshold below which to declare "zero variance"

    na_warn : bool (default True)
        Whether to warn if any NAs are present

    na_rm : bool (default False)
        Whether to remove NA values

    use : str (default "complete.obs"), one of {'complete.obs','all.obs','everything'}
        A string indicating how to handle missing values.

    Attributes
    ----------

    drop_ : list, string
        The columns to drop
    """

    _min_version = '3.8.2.9'
    _max_version = None

    def __init__(self, feature_names=None, target_feature=None, exclude_features=None,
                 threshold=1e-6, na_warn=True, na_rm=False, use='complete.obs'):
        super(H2ONearZeroVarianceFilterer, self).__init__(feature_names=feature_names,
                                                          target_feature=target_feature,
                                                          exclude_features=exclude_features,
                                                          min_version=self._min_version,
                                                          max_version=self._max_version)
        self.threshold = threshold
        self.na_warn = na_warn
        self.na_rm = na_rm
        self.use = use

    def fit(self, X):
        """Fit the near zero variance filterer,
        return the transformed X frame.

        Parameters
        ----------

        X : H2OFrame
            The frame to fit

        Returns
        -------

        self
        """
        self.fit_transform(X)
        return self

    def fit_transform(self, X):
        """Fit the near zero variance filterer.

        Parameters
        ----------

        X : H2OFrame
            The frame to fit

        Returns
        -------

        X : H2OFrame
            The transformed frame
        """
        frame, thresh = _check_is_frame(X), self.threshold
        frame = _frame_from_x_y(frame, self.feature_names, self.target_feature, self.exclude_features)

        # validate use, check NAs
        use = _validate_use(frame, self.use, self.na_warn)

        # get covariance matrix, extract the diagonal
        cols = np.asarray([str(n) for n in frame.columns])
        diag = np.diagonal(frame.var(na_rm=self.na_rm, use=use).as_data_frame(use_pandas=True).as_matrix())

        # create mask
        var_mask = diag < thresh

        self.drop_ = cols[var_mask].tolist()  # make list
        self.var_ = dict(zip(self.drop_, diag[var_mask]))

        return self.transform(X)
