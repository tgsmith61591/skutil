# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import
import warnings
import numpy as np
import pandas as pd
import numbers
import scipy.stats as st
from sklearn.datasets import load_iris, load_breast_cancer, load_boston
from sklearn.externals import six
from sklearn.metrics import confusion_matrix as cm
from ..base import suppress_warnings
from .fixes import _grid_detail, _is_integer, is_iterable, _cols_if_none

try:
    # this causes a UserWarning to be thrown by matplotlib... should we squelch this?
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        import matplotlib
        matplotlib.use('Agg')  # set backend
        from matplotlib import pyplot as plt

        # log it
        CAN_CHART_MPL = True
except ImportError as ie:
    CAN_CHART_MPL = False

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import seaborn as sns

        CAN_CHART_SNS = True
except ImportError as ie:
    CAN_CHART_SNS = False

__max_exp__ = 1e19
__min_log__ = -19
__all__ = [
    'corr_plot',
    'df_memory_estimate',
    'exp',
    'flatten_all',
    'flatten_all_generator',
    'get_numeric',
    'human_bytes',
    'is_entirely_numeric',
    'is_integer',
    'is_float',
    'is_numeric',
    'is_integer',
    'is_float',
    'load_boston_df',
    'load_breast_cancer_df',
    'load_iris_df',
    'log',
    'pd_stats',
    'report_confusion_matrix',
    'report_grid_score_detail',
    'shuffle_dataframe',
    'validate_is_pd'
]


@suppress_warnings
def _log_single(x):
    """Sanitized log function for a single element.
    Since this method internally calls np.log and carries
    the (very likely) possibility to overflow, the method
    suppresses all warnings.

    #XXX: at some point we might want to let ``suppress_warnings``
    # specify exactly which types of warnings it should filter.

    Parameters
    ----------

    x : float
        The number to log

    Returns
    -------

    val : float
        the log of x
    """
    x = np.maximum(0, x)
    val = __min_log__ if x == 0 else np.maximum(__min_log__, np.log(x))
    return val


@suppress_warnings
def _exp_single(x):
    """Sanitized exponential function.
    Since this method internally calls np.exp and carries
    the (very likely) possibility to overflow, the method
    suppresses all warnings.

    #XXX: at some point we might want to let ``suppress_warnings``
    # specify exactly which types of warnings it should filter.

    Parameters
    ----------

    x : float
        The number to exp


    Returns
    -------

    val : float
        the exp of x
    """
    val = np.minimum(__max_exp__, np.exp(x))
    return val


def _vectorize(fun, x):
    if is_iterable(x):
        return np.array([fun(p) for p in x])
    raise ValueError('Type %s is not iterable' % type(x))


def exp(x):
    """A safe mechanism for computing the exponential function
    while avoiding overflows.
    
    Parameters
    ----------

    x : float, number
        The number for which to compute the exp


    Returns
    -------

    exp(x)
    """
    # check on single exp
    if is_numeric(x):
        return _exp_single(x)
    # try vectorized
    try:
        return _vectorize(exp, x)
    except ValueError as v:
        # bail
        raise ValueError("don't know how to compute exp for type %s" % type(x))


def log(x):
    """A safe mechanism for computing a log while
    avoiding NaNs or exceptions.

    Parameters
    ----------

    x : float, number
        The number for which to compute the log


    Returns
    -------

    log(x)
    """
    # check on single log
    if is_numeric(x):
        return _log_single(x)
    # try vectorized
    try:
        return _vectorize(log, x)
    except ValueError as v:
        # bail
        raise ValueError("don't know how to compute log for type %s" % type(x))


def _val_cols(cols):
    # if it's None, return immediately
    if cols is None:
        return cols

    # try to make cols a list
    if not is_iterable(cols):
        if isinstance(cols, six.string_types):
            return [cols]
        else:
            raise ValueError('cols must be an iterable sequence')

    # if it is an index or a np.ndarray, it will have a built-in
    # (potentially more efficient tolist() method)
    if hasattr(cols, 'tolist') and hasattr(cols.tolist, '__call__'):
        return cols.tolist()

    # make it a list implicitly, make no guarantees about elements
    return [c for c in cols]


def _def_headers(X):
    m = X.shape[1] if hasattr(X, 'shape') else len(X)
    return ['V%i' % (i + 1) for i in range(m)]


def corr_plot(X, plot_type='cor', cmap='Blues_d', n_levels=5, corr=None,
              method='pearson', figsize=(11, 9), cmap_a=220, cmap_b=10, vmax=0.3,
              xticklabels=5, yticklabels=5, linewidths=0.5, cbar_kws={'shrink': 0.5}):
    """Create a simple correlation plot given a dataframe.
    Note that this requires all datatypes to be numeric and finite!

    Parameters
    ----------

    X : pd.DataFrame, shape=(n_samples, n_features)
        The pandas DataFrame on which to compute correlations,
        or if ``corr`` is 'precomputed', the correlation matrix.
        In the case that ``X`` is a correlation matrix, it must
        be square, i.e., shape=(n_features, n_features).

    plot_type : str, optional (default='cor')
        The type of plot, one of ('cor', 'kde', 'pair')

    cmap : str, optional (default='Blues_d')
        The color to use for the kernel density estimate plot
        if ``plot_type`` == 'kde'. Otherwise unused.

    n_levels : int, optional (default=5)
        The number of levels to use for the kde plot 
        if ``plot_type`` == 'kde'. Otherwise unused.

    corr : 'precomputed' or None, optional (default=None)
        If None, the correlation matrix is computed, otherwise
        if 'precomputed', ``X`` is treated as a correlation matrix.

    method : str, optional (default='pearson')
        The method to use for correlation

    figsize : tuple (int), shape=(w,h), optional (default=(11,9))
        The size of the image

    cmap_a : int, optional (default=220)
        The colormap start point

    cmap_b : int, optional (default=10)
        The colormap end point

    vmax : float, optional (default=0.3)
        Arg for seaborn heatmap

    xticklabels : int, optional (default=5)
        The spacing for X ticks

    yticklabels : int, optional (default=5)
        The spacing for Y ticks

    linewidths : float, optional (default=0.5)
        The width of the lines

    cbar_kws : dict, optional (default={'shrink':0.5})
        Any KWs to pass to seaborn's heatmap when ``plot_type`` = 'cor'
    """

    X, _ = validate_is_pd(X, None, assert_all_finite=True)
    valid_types = ('cor', 'kde', 'pair')
    if not plot_type in valid_types:
        raise ValueError('expected one of (%s), but got %s'
                         % (','.join(valid_types), plot_type))

    # seaborn is needed for all of these, so we have to check outside
    if not CAN_CHART_SNS:
        warnings.warn('Cannot plot (unable to import Seaborn)', ImportWarning)
        return None

    if plot_type == 'cor':
        # MPL is only needed for COR
        if not CAN_CHART_MPL:
            warnings.warn('Cannot plot (unable to import Matplotlib)')
            return None

        if not corr == 'precomputed':
            cols = X.columns.values
            X = X.corr(method=method)
            X.columns = cols
            X.index = cols

        # mask for upper triangle
        mask = np.zeros_like(X, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        # set up mpl figure
        f, ax = plt.subplots(figsize=figsize)
        color_map = sns.diverging_palette(cmap_a, cmap_b, as_cmap=True)
        sns.heatmap(X, mask=mask, cmap=color_map, vmax=vmax,
                    square=True, xticklabels=xticklabels, yticklabels=yticklabels,
                    linewidths=linewidths, cbar_kws=cbar_kws, ax=ax)

    elif plot_type == 'pair':
        sns.pairplot(X)
        sns.plt.show()

    else:
        g = sns.PairGrid(X)
        g.map_diag(sns.kdeplot)
        g.map_offdiag(sns.kdeplot, cmap=cmap, n_levels=n_levels)
        sns.plt.show()


def flatten_all(container):
    """Recursively flattens an arbitrarily nested iterable.
    WARNING: this function may produce a list of mixed types.

    Parameters
    ----------

    container : iterable, object
        The iterable to flatten. If the ``container`` is
        not iterable, it will be returned in a list as 
        ``[container]``


    Examples
    --------

    The example below produces a list of mixed results:

        >>> a = [[[],3,4],['1','a'],[[[1]]],1,2]
        >>> flatten_all(a)
        [3, 4, '1', 'a', 1, 1, 2]


    Returns
    -------

    l : list
        The flattened list
    """
    l = [x for x in flatten_all_generator(container)]
    return l


def flatten_all_generator(container):
    """Recursively flattens an arbitrarily nested iterable.
    WARNING: this function may produce a list of mixed types.

    Parameters
    ----------

    container : iterable, object
        The iterable to flatten.


    Examples
    --------

    The example below produces a list of mixed results:

        >>> a = [[[],3,4],['1','a'],[[[1]]],1,2]
        >>> flatten_all(a) # yields a generator for this iterable
        [3, 4, '1', 'a', 1, 1, 2]
    """
    if not is_iterable(container):
        yield container
    else:
        for i in container:
            if is_iterable(i):
                for j in flatten_all_generator(i):
                    yield j
            else:
                yield i


def shuffle_dataframe(X):
    """Shuffle the rows in a data frame without replacement.
    The random state used for shuffling is controlled by
    numpy's random state.

    Parameters
    ----------

    X : pd.DataFrame, shape=(n_samples, n_features)
        The dataframe to shuffle
    """
    X, _ = validate_is_pd(X, None, False)
    return X.iloc[np.random.permutation(np.arange(X.shape[0]))]


def validate_is_pd(X, cols, assert_all_finite=False):
    """Used within each SelectiveMixin fit method to determine whether
    the passed ``X`` is a dataframe, and whether the cols is appropriate.
    There are four scenarios (in the order in which they're checked):

    1) Names is not None, but X is not a dataframe.
        Resolution: the method will attempt to return a DataFrame from the
        args provided (with default names), but catches any
        exception and raises a ValueError. A common case where this would work
        may be a numpy.ndarray as X, and a list as cols (where the list is either
        int indices or default names that the dataframe will take on).

    2) X is a DataFrame, but cols is None.
        Resolution: return a copy of the dataframe, and use all column names.

    3) X is a DataFrame and cols is not None.
        Return a copy of the dataframe, and use only the names provided. This is
        the typical use case.

    4) X is not a DataFrame, and cols is None.
        Resolution: this case will only work if the X can be built into a DataFrame.
        Otherwise, there will be a ValueError thrown.

    Parameters
    ----------

    X : array_like, shape=(n_samples, n_features)
        The dataframe to validate. If ``X`` is not a DataFrame,
        but it can be made into one, no exceptions will be raised.
        However, if ``X`` cannot naturally be made into a DataFrame,
        a TypeError will be raised.

    cols : array_like (str), shape=(n_features,)
        The list of column names. Used particularly in SelectiveMixin
        transformers that validate column names.

    assert_all_finite : bool, optional (default=False)
        If True, will raise an AssertionError if any np.nan or np.inf
        values reside in ``X``.


    Returns
    -------

    X : pd.DataFrame, shape=(n_samples, n_features)
        A copy of the original input ``X``

    cols : list or None, shape=(n_features,)
        If ``cols`` was not None and did not raise a TypeError,
        it is converted into a list of strings and returned
        as a copy. Else None.
    """

    def _check(X, cols):
        # first check hard-to-detect case:
        if isinstance(X, pd.Series):
            raise ValueError('expected DataFrame but got Series')

        # validate the cols arg
        cols = _val_cols(cols)

        # if someone devious gave us an empty set of cols
        if cols is not None and len(cols) == 0:
            cols = None

        # avoid multiple isinstance checks
        is_df = isinstance(X, pd.DataFrame)

        # we do want to make sure the X at least is "array-like"
        if not is_iterable(X):
            raise TypeError('X (type=%s) cannot be cast to DataFrame' % type(X))

        # case 1, we have names but the X is not a frame
        if not is_df and cols is not None:
            # this is tough, because they only pass cols if it's a subset
            # and this frame is likely too large for the passed columns.
            # so, we hope they either passed what the col names WILL be
            # or that they passed numeric cols... they should handle that
            # validation on their end, though. If this fails, we'll just let
            # it fall through.
            return pd.DataFrame.from_records(data=X, columns=_def_headers(X)), cols

        # case 2, we have a DF but no cols, def behavior: use all
        elif is_df and cols is None:
            return X.copy(), None

        # case 3, we have a DF AND cols
        elif is_df and cols is not None:
            return X.copy(), cols

        # case 4, we have neither a frame nor cols (maybe JUST a np.array?)
        else:
            # we'll do two tests here... either that it's a np ndarray or a list of lists
            if isinstance(X, np.ndarray) or (is_iterable(X) and all(isinstance(elem, list) for elem in X)):
                return pd.DataFrame.from_records(data=X, columns=_def_headers(X)), None

            # bail out:
            raise ValueError('cannot handle data of type %s' % type(X))

    # do initial check
    X, cols = _check(X, cols)

    # we need to ensure all are finite
    if assert_all_finite:
        # if cols, we only need to ensure the specified columns are finite
        cols_tmp = _cols_if_none(X, cols)
        X_prime = X[cols_tmp]

        if X_prime.apply(lambda x: (~np.isfinite(x)).sum()).sum() > 0:
            raise ValueError('Expected all entries to be finite')

    return X, cols


def df_memory_estimate(X, unit='MB', index=False):
    """We estimate the memory footprint of an H2OFrame
    to determine whether it's capable of being held in memory 
    or not.

    Parameters
    ----------

    X : Pandas ``DataFrame`` or ``H2OFrame``, shape=(n_samples, n_features)
        The DataFrame in question

    unit : str, optional (default='MB')
        The units to report. One of ('MB', 'KB', 'GB', 'TB')

    index : bool, optional (default=False)
        Whether to also estimate the memory footprint of the index.


    Returns
    -------

    mb : str
        The estimated number of UNIT held in the frame
    """
    X, _ = validate_is_pd(X, None, False)
    return human_bytes(X.memory_usage(index=index).sum(), unit)


def _is_int(x, tp):
    """Determine whether a column can be cast to int
    without loss of data
    """
    if not any([tp.startswith(i) for i in ('float', 'int')]):
        return False

    # if there's no difference between the two, then it's an int.
    return (x - x.astype('int')).abs().sum() == 0


def pd_stats(X, col_type='all', na_str='--', hi_skew_thresh=1.0, mod_skew_thresh=0.5):
    """Get a descriptive report of the elements in the data frame.
    Builds on existing pandas ``describe`` method by adding counts of
    factor-level features, a skewness rating and several other helpful
    statistics.

    Parameters
    ----------

    X : Pandas ``DataFrame`` or ``H2OFrame``, shape=(n_samples, n_features)
        The DataFrame on which to compute stats.

    col_type : str, optional (default='all')
        The types of columns to analyze. One of ('all',
        'numeric', 'object'). If not all, will only return
        corresponding typed columns.

    hi_skew_thresh : float, optional (default=1.0)
        The threshold above which a skewness rating will
        be deemed "high."

    mod_skew_thresh : float, optional (default=0.5)
        The threshold above which a skewness rating will 
        be deemed "moderate," so long as it does not exceed
        ``hi_skew_thresh``


    Returns
    -------

    s : Pandas ``DataFrame`` or ``H2OFrame``, shape=(n_samples, n_features)
        The resulting stats dataframe
    """
    X, _ = validate_is_pd(X, None, False)
    raw_stats = X.describe()
    stats = raw_stats.to_dict()
    dtypes = X.dtypes

    # validate col_type
    valid_types = ('all', 'numeric', 'object')
    if not col_type in valid_types:
        raise ValueError('expected col_type in (%s), but got %s'
                         % (','.join(valid_types), col_type))

    # if user only wants part back, we can use this...
    type_dict = {}

    # the string to use when we don't want to populate a cell
    _nastr = na_str

    # objects are going to get dropped in the describe() call,
    # so we need to add them back in with dicts of nastr for all...
    object_dtypes = dtypes[dtypes == 'object']
    if object_dtypes.shape[0] > 0:
        obj_nms = object_dtypes.index.values

        for nm in obj_nms:
            obj_dct = {stat: _nastr for stat in raw_stats.index.values}
            stats[nm] = obj_dct

    # we'll add rows to the stats...
    for col, dct in six.iteritems(stats):
        # add the dtype
        _dtype = str(dtypes[col])
        is_numer = any([_dtype.startswith(x) for x in ('int', 'float')])
        dct['dtype'] = _dtype

        # populate type_dict
        type_dict[col] = 'numeric' if is_numer else 'object'

        # if the dtype is not a float, we can
        # get the count of uniques, then do a
        # ratio of majority : minority
        _isint = _is_int(X[col], _dtype)
        if _isint or _dtype == 'object':
            _unique = len(X[col].unique())
            _val_cts = X[col].value_counts().sort_values(ascending=True)
            _min_cls, _max_cls = _val_cts.index[0], _val_cts.index[-1]

            # if there's only one class...
            if _min_cls == _max_cls:
                _min_cls = _nastr
                _min_max_ratio = _nastr
            else:
                _min_max_ratio = _val_cts.values[0] / _val_cts.values[-1]

            # chance we didn't recognize it as an int before...
            if 'float' in dct['dtype']:
                dct['dtype'] = dct['dtype'].replace('float', 'int')

        else:
            _unique = _min_max_ratio = _nastr

        # populate the unique count and more
        dct['unique_ct'] = _unique
        dct['min_max_class_ratio'] = _min_max_ratio

        # get the skewness...
        if is_numer:
            _skew, _kurt = X[col].skew(), X[col].kurtosis()
            abs_skew = abs(_skew)
            hs, ms = hi_skew_thresh, mod_skew_thresh
            _skew_risk = 'high skew' if abs_skew > hs else 'mod. skew' if (ms < abs_skew < hs) else 'symmetric'
        else:
            _skew = _kurt = _skew_risk = _nastr

        dct['skewness'] = _skew
        dct['skewness rating'] = _skew_risk
        dct['kurtosis'] = _kurt

    # go through and pop the keys that might be filtered on
    if col_type != 'all':
        stat_out = {}
        for col, dtype in six.iteritems(type_dict):
            if col_type == dtype:
                stat_out[col] = stats[col]

    else:
        stat_out = stats

    s = pd.DataFrame.from_dict(stat_out)
    return s


def get_numeric(X):
    """Return list of indices of numeric dtypes variables

    Parameters
    ----------

    X : Pandas ``DataFrame`` or ``H2OFrame``, shape=(n_samples, n_features)
        The dataframe


    Returns
    -------

    list, int
        The list of indices which are numeric.
    """
    validate_is_pd(X, None)  # don't want warning
    return X.dtypes[X.dtypes.apply(lambda x: str(x).startswith(("float", "int")))].index.tolist()


def human_bytes(b, unit='MB'):
    """Get bytes in a human readable form

    Parameters
    ----------

    b : int
        The number of bytes

    unit : str, optional (default='MB')
        The units to report. One of ('MB', 'KB', 'GB', 'TB')


    Returns
    -------

    mb : str
        The estimated number of UNIT held in the frame
    """
    kb = float(1024)
    units = {
        'KB': kb,
        'MB': float(kb ** 2),
        'GB': float(kb ** 3),
        'TB': float(kb ** 4)
    }

    if not unit in units:
        raise ValueError('got %s, expected one of (%s)'
                         % (unit, ', '.join(units.keys())))

    return '%.3f %s' % (b / units[unit], unit)


def is_entirely_numeric(X):
    """Determines whether an entire pandas frame
    is numeric in dtypes.

    Parameters
    ----------

    X : Pandas ``DataFrame`` or ``H2OFrame``, shape=(n_samples, n_features)
        The dataframe to test


    Returns
    -------

    bool
        True if the entire pd.DataFrame 
        is numeric else False
    """
    return X.shape[1] == len(get_numeric(X))


def is_integer(x):
    """Determine whether some object ``x`` is an
    integer type (int, long, etc).

    Parameters
    ----------

    x : object
        The item to assess


    Returns
    -------

    bool
        True if ``x`` is an integer type
    """
    return _is_integer(x)


def is_float(x):
    """Determine whether some object ``x`` is a
    float type (float, np.float, etc).

    Parameters
    ----------

    x : object
        The item to assess


    Returns
    -------

    bool
        True if ``x`` is a float type
    """
    return isinstance(x, (float, np.float)) or \
        (not isinstance(x, (bool, np.bool)) and isinstance(x, numbers.Real))


def is_numeric(x):
    """Determine whether some object ``x`` is a
    numeric type (float, int, etc).

    Parameters
    ----------

    x : object
        The item to assess


    Returns
    -------

    bool
        True if ``x`` is a float or integer type
    """
    return is_float(x) or is_integer(x)


def load_iris_df(include_tgt=True, tgt_name="Species", shuffle=False):
    """Loads the iris dataset into a dataframe with the
    target set as the "Species" feature or whatever name
    is specified in ``tgt_name``.

    Parameters
    ----------

    include_tgt : bool, optional (default=True)
        Whether to include the target

    tgt_name : str, optional (default="Species")
        The name of the target feature

    shuffle : bool, optional (default=False)
        Whether to shuffle the rows on return


    Returns
    -------

    X : pd.DataFrame, shape=(n_samples, n_features)
        The loaded dataset
    """
    iris = load_iris()
    X = pd.DataFrame.from_records(data=iris.data, columns=iris.feature_names)

    if include_tgt:
        X[tgt_name] = iris.target

    return X if not shuffle else shuffle_dataframe(X)


def load_breast_cancer_df(include_tgt=True, tgt_name="target", shuffle=False):
    """Loads the breast cancer dataset into a dataframe with the
    target set as the "target" feature or whatever name
    is specified in ``tgt_name``.

    Parameters
    ----------

    include_tgt : bool, optional (default=True)
        Whether to include the target

    tgt_name : str, optional (default="target")
        The name of the target feature

    shuffle : bool, optional (default=False)
        Whether to shuffle the rows


    Returns
    -------

    X : pd.DataFrame, shape=(n_samples, n_features)
        The loaded dataset
    """
    bc = load_breast_cancer()
    X = pd.DataFrame.from_records(data=bc.data, columns=bc.feature_names)

    if include_tgt:
        X[tgt_name] = bc.target

    return X if not shuffle else shuffle_dataframe(X)


def load_boston_df(include_tgt=True, tgt_name="target", shuffle=False):
    """Loads the boston housing dataset into a dataframe with the
    target set as the "target" feature or whatever name
    is specified in ``tgt_name``.

    Parameters
    ----------

    include_tgt : bool, optional (default=True)
        Whether to include the target

    tgt_name : str, optional (default="target")
        The name of the target feature

    shuffle : bool, optional (default=False)
        Whether to shuffle the rows


    Returns
    -------

    X : Pandas ``DataFrame`` or ``H2OFrame``, shape=(n_samples, n_features)
        The loaded dataset
    """
    bo = load_boston()
    X = pd.DataFrame.from_records(data=bo.data, columns=bo.feature_names)

    if include_tgt:
        X[tgt_name] = bo.target

    return X if not shuffle else shuffle_dataframe(X)


def report_grid_score_detail(random_search, charts=True, sort_results=True,
                             ascending=True, percentile=0.975, y_axis='mean_test_score', 
                             sort_by='mean_test_score', highlight_best=True, highlight_col='red', 
                             def_color='blue', return_drops=False):
    """Return plots and dataframe of results, given a fitted grid search.
    Note that if Matplotlib is not installed, a warning will be thrown
    and no plots will be generated.

    Parameters
    ----------

    random_search : ``BaseSearchCV`` or ``BaseH2OSearchCV``
        The fitted grid search

    charts : bool, optional (default=True)
        Whether to plot the charts

    sort_results : bool, optional (default=True)
        Whether to sort the results based on score

    ascending : bool, optional (default=True)
        If ``sort_results`` is True, whether to use asc or desc
        in the sorting process.

    percentile : float, optional (default=0.975)
        The percentile point (0 < percentile < 1.0). The
        corresponding z-score will be multiplied
        by the cross validation score standard deviations.

    y_axis : str, optional (default='mean_test_score')
        The y-axis of the charts. One of ('score','std')

    sort_by : str, optional (default='mean_test_score')
        The column to sort by. This is not validated, in case
        the user wants to sort by a parameter column. If
        not ``sort_results``, this is unused.

    highlight_best : bool, optional (default=True)
        If set to True, charts is True, and sort_results is 
        also True, then highlights the point in the top
        position of the model DF.

    highlight_col : str, optional (default='red')
        What color to use for ``highlight_best`` if both
        ``charts`` and ``highlight_best``. If either is False,
        this is unused.

    def_color : str, optional (default='blue')
        What color to use for the points if ``charts`` is True.
        This should differ from ``highlight_col``, but no validation
        is performed.

    return_drops : bool, optional (default=False)
        If True, will return the list of names that can be dropped
        out (i.e., were generated by sklearn and are not parameters
        of interest).


    Returns
    -------

    result_df : Pandas ``DataFrame`` or ``H2OFrame``, shape=(n_samples, n_features)
        The grid search results

    drops : list
        List of sklearn-generated names. Only returned if
        ``return_drops`` is True.
    """
    valid_axes = ('mean_test_score', 'std_test_score')

    # these are produced in sklearn 0.18 but not 0.17 -- want to skip for now...
    ignore_axes = ('mean_fit_time', 'mean_score_time', 
                   'mean_train_score', 'std_fit_time', 
                   'std_score_time', 'std_train_score')

    # validate y-axis
    if not y_axis in valid_axes:
        raise ValueError('y-axis=%s must be one of (%s)' % (y_axis, ', '.join(valid_axes)))

    # validate percentile
    if not (0 < percentile < 1):
        raise ValueError('percentile must be > 0 and < 1, but got %.5f' % percentile)
    z_score = st.norm.ppf(percentile)

    # make into a data frame from search
    result_df, drops = _grid_detail(random_search, 
                                    z_score=z_score,
                                    sort_results=sort_results, 
                                    sort_by=sort_by, 
                                    ascending=ascending)

    # if the import failed, we won't be able to chart here
    if charts and CAN_CHART_MPL:
        for col in get_numeric(result_df):
            if col in ignore_axes:
                # don't plot these ones
                continue
            elif col not in valid_axes:  # skip score / std
                ser = result_df[col]
                color = [def_color for i in range(ser.shape[0])]

                # highlight if needed
                if sort_results and highlight_best:
                    color[0] = highlight_col

                # build scatter plot
                plt.scatter(ser, result_df[y_axis], color=color)
                plt.title(col)
                plt.ylabel(y_axis)

                # if there's a '__' in the col, split it
                x_lab = col if not '__' in col else col.split('__')[-1]
                plt.xlabel(x_lab)

                # render
                plt.show()

        for col in list(result_df.columns[result_df.dtypes == "object"]):
            cat_plot = result_df[y_axis].groupby(result_df[col]).mean()
            cat_plot.sort_values()
            cat_plot.plot(kind="barh", xlim=(.5, None), figsize=(7, cat_plot.shape[0] / 2))

            plt.show()

    elif charts and not CAN_CHART_MPL:
        warnings.warn('no module matplotlib, will not be able to display charts', ImportWarning)

    return result_df if not return_drops else (result_df, drops)


def report_confusion_matrix(actual, pred, return_metrics=True):
    """Return a dataframe with the confusion matrix, and a series
    with the classification performance metrics.

    Parameters
    ----------

    actual : np.ndarray, shape=(n_samples,)
        The array of actual values

    pred : np.ndarray, shape=(n_samples,)
        The array of predicted values

    return_metrics : bool, optional (default=True)
        Whether to return the metrics in a pd.Series. If False,
        index 1 of the returned tuple will be None.


    Returns
    -------

    conf : pd.DataFrame, shape=(2, 2)
        The confusion matrix

    ser : pd.Series or None
        The metrics if ``return_metrics`` else None
    """

    # ensure only two classes in each
    lens = [len(set(actual)), len(set(pred))]
    max_len = np.max(lens)
    if max_len > 2:
        raise ValueError('max classes is 2, but got %i' % max_len)

    cf = cm(actual, pred)
    # format: (col = pred, index = act)
    # array([[TN, FP],
    #        [FN, TP]])

    ser = None
    if return_metrics:
        total_pop = np.sum(cf)
        condition_pos = np.sum(cf[1, :])
        condition_neg = np.sum(cf[0, :])

        # alias the elements in the matrix
        tp = cf[1, 1]
        fp = cf[0, 1]
        tn = cf[0, 0]
        fn = cf[1, 0]

        # sums of the prediction cols
        pred_pos = tp + fp
        pred_neg = tn + fn

        acc = (tp + tn) / total_pop  # accuracy
        tpr = tp / condition_pos  # sensitivity, recall
        fpr = fp / condition_neg  # fall-out
        fnr = fn / condition_pos  # miss rate
        tnr = tn / condition_neg  # specificity
        prev = condition_pos / total_pop  # prevalence
        plr = tpr / fpr  # positive likelihood ratio, LR+
        nlr = fnr / tnr  # negative likelihood ratio, LR-
        dor = plr / nlr  # diagnostic odds ratio
        prc = tp / pred_pos  # precision, positive predictive value
        fdr = fp / pred_pos  # false discovery rate
        fomr = fn / pred_neg  # false omission rate
        npv = tn / pred_neg  # negative predictive value

        # define the series
        d = {
            'Accuracy': acc,
            'Diagnostic odds ratio': dor,
            'Fall-out': fpr,
            'False discovery rate': fdr,
            'False Neg. Rate': fnr,
            'False omission rate': fomr,
            'False Pos. Rate': fpr,
            'Miss rate': fnr,
            'Neg. likelihood ratio': nlr,
            'Neg. predictive value': npv,
            'Pos. likelihood ratio': plr,
            'Pos. predictive value': prc,
            'Precision': prc,
            'Prevalence': prev,
            'Recall': tpr,
            'Sensitivity': tpr,
            'Specificity': tnr,
            'True Pos. Rate': tpr,
            'True Neg. Rate': tnr
        }

        ser = pd.Series(data=d)
        ser.name = 'Metrics'

    # create the DF
    conf = pd.DataFrame.from_records(data=cf, columns=['Neg', 'Pos'])
    conf.index = ['Neg', 'Pos']

    return conf, ser
