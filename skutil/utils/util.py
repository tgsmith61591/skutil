from __future__ import print_function, division, absolute_import
import pandas as pd
import numpy as np
import warnings
import numbers
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix as cm
from sklearn.datasets import load_iris
from sklearn.externals import six
from ..base import SelectiveWarning, ModuleImportWarning

try:
    # this causes a UserWarning to be thrown by matplotlib... should we squelch this?
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # do actual import
        #import matplotlib as mpl
        #mpl.use('TkAgg') # set backend
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
    'is_numeric',
    'load_iris_df',
    'log',
    'pd_stats',
    'report_confusion_matrix',
    'report_grid_score_detail',
    'shuffle_dataframe',
    'validate_is_pd'
]



######## MATHEMATICAL UTILITIES #############    
def _log_single(x):
    """Sanitized log function for a single element
    Parameters
    ----------
    x : float
        The number to log
    Returns
    -------
    val : float
        the log of x
    """
    x = max(0, x)
    val = __min_log__ if x == 0 else max(__min_log__, np.log(x))  
    return val

def _exp_single(x):
    """Sanitized exponential function
    Parameters
    ----------
    x : float
        The number to exp
    Returns
    -------
    float
        the exp of x
    """
    return min(__max_exp__, np.exp(x))

def _vectorize(fun, x):
    if hasattr(x, '__iter__'):
        return np.array([fun(p) for p in x])
    raise ValueError('Type %s does not have attr __iter__' % type(x))

def exp(x):
    """A safe mechanism for computing the exponential function"""
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
    """A safe mechanism for computing a log"""
    # check on single log
    if is_numeric(x):
        return _log_single(x)
    # try vectorized
    try:
        return _vectorize(log, x)
    except ValueError as v:
        # bail
        raise ValueError("don't know how to compute log for type %s" % type(x))





######### GENERAL UTILITIES #################
def _val_cols(cols):
    # if it's None, return immediately
    if cols is None:
        return cols

    # try to make cols a list
    if not hasattr(cols, '__iter__'):
        raise ValueError('cols must be an iterable sequence')
    return [c for c in cols] # make it a list implicitly, make no guarantees about elements

def _def_headers(X):
    m = X.shape[1] if hasattr(X, 'shape') else len(X)
    return ['V%i' %  (i+1) for i in range(m)]



def corr_plot(X, plot_type='cor', cmap='Blues_d', n_levels=5, corr=None, 
        method='pearson', figsize=(11,9), cmap_a=220, cmap_b=10, vmax=0.3,
        xticklabels=5, yticklabels=5, linewidths=0.5, cbar_kws={'shrink':0.5}):
    """Create a simple correlation plot given a dataframe.
    Note that this requires all datatypes to be numeric and finite!

    Parameters
    ----------
    X : pd.DataFrame
        The pandas DataFrame

    plot_type : str, optional (default='cor')
        The type of plot, one of ('cor', 'kde', 'pair')

    cmap : str, optional (default='Blues_d')
        The color to use for the kernel density estimate plot
        if plot_type == 'kde'

    n_levels : int, optional (default=5)
        The number of levels to use for the kde plot 
        if plot_type == 'kde'

    corr : 'precomputed' or None, optional (default=None)
        If None, the correlation matrix is computed, otherwise
        if 'precomputed', X is treated as a correlation matrix.

    method : str, optional (default='pearson')
        The method to use for correlation

    figsize : tuple (int), optional (default=(11,9))
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

    cbar_kws : dict, optional
        Any KWs to pass to seaborn's heatmap when plot_type = 'cor'
    """

    X, _ = validate_is_pd(X, None, assert_all_finite=True)
    valid_types = ('cor', 'kde', 'pair')
    if not plot_type in valid_types:
        raise ValueError('expected one of (%s), but got %s'
                         % (','.join(valid_types), plot_type))

    # seaborn is needed for all of these, so we have to check outside
    if not CAN_CHART_SNS:
        warnings.warn('Cannot plot (unable to import Seaborn)')
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

    Usage:
    a = [[[],3,4],['1','a'],[[[1]]],1,2]
    >>> flatten_all(a)
    [3,4,'1','a',1,1,2]
    """
    return [x for x in flatten_all_generator(container)]

def flatten_all_generator(container):
    """Recursively flattens an arbitrarily nested iterable.
    WARNING: this function may produce a list of mixed types.

    Usage:
    a = [[[],3,4],['1','a'],[[[1]]],1,2]
    >>> flatten_all_generator(a)
    [3,4,'1','a',1,1,2] # returns a generator for this iterable
    """
    for i in container:
        if hasattr(i, '__iter__'):
            for j in flatten_all_generator(i):
                yield j
        else:
            yield i

def shuffle_dataframe(X):
    X, _ = validate_is_pd(X, None, False)
    return X.iloc[np.random.permutation(np.arange(X.shape[0]))]


def validate_is_pd(X, cols, assert_all_finite=False):
    """Used within each SelectiveMixin fit method to determine whether
    the passed X is a dataframe, and whether the cols is appropriate.
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

    Returns
    -------
    tuple, (DataFrame: X, list: cols)
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

        # avoid multiple isinstances
        is_df = isinstance(X, pd.DataFrame)

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
            if isinstance(X, np.ndarray) or (hasattr(X, '__iter__') and all(isinstance(elem, list) for elem in X)):
                return pd.DataFrame.from_records(data=X, columns=_def_headers(X)), None

            # bail out:
            raise ValueError('cannot handle data of type %s' % type(X))

    # do initial check
    X, cols = _check(X, cols)

    # we need to ensure all are finite
    if assert_all_finite:
        if X.apply(lambda x: (~np.isfinite(x)).sum()).sum() > 0:
            raise ValueError('Expected all entries to be finite')

    return X, cols


def df_memory_estimate(X, bit_est=32, unit='MB', index=False):
    """We estimate the memory footprint of an H2OFrame
    to determine, possibly, whether it's capable of being
    held in memory or not.

    Parameters
    ----------
    X : pandas DataFrame
        The DataFrame in question

    bit_est : int, optional (default=32)
        The estimated bit-size of each cell. The default
        assumes each cell is a signed 32-bit float

    unit : str, optional (default='MB')
        The units to report. One of ('MB', 'KB', 'GB', 'TB')

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

def pd_stats(X, col_type='all'):
    """Get a descriptive report of the elements in the data frame.
    Builds on existing pandas `describe` method.

    Parameters
    ----------
    X : pd.DataFrame
        The DataFrame

    col_type : str, optional (default='all')
        The types of columns to analyze. One of ('all',
        'numeric', 'object'). If not all, will only return
        corresponding typed columns.
    """
    X, _ = validate_is_pd(X, None, False)
    raw_stats = X.describe()
    stats = raw_stats.to_dict()
    dtypes = X.dtypes

    # validate col_type
    valid_types = ('all','numeric','object')
    if not col_type in valid_types:
        raise ValueError('expected col_type in (%s), but got %s'
            % (','.join(valid_types), col_type))

    # if user only wants part back, we can use this...
    type_dict = {}

    # the string to use when we don't want to populate a cell
    _nastr = '--'

    # objects are going to get dropped in the describe() call,
    # so we need to add them back in with dicts of nastr for all...
    object_dtypes = dtypes[dtypes=='object']
    if object_dtypes.shape[0] > 0:
        obj_nms = object_dtypes.index.values

        for nm in obj_nms:
            obj_dct = {stat:_nastr for stat in raw_stats.index.values}
            stats[nm] = obj_dct


    # we'll add rows to the stats...
    for col, dct in six.iteritems(stats):
        # add the dtype
        _dtype = str(dtypes[col])
        is_numer = any([_dtype.startswith(x) for x in ('int','float')])
        dct['dtype'] = _dtype

        # populate type_dict
        type_dict[col] = 'numeric' if is_numer else 'object'

        # if the dtype is not a float, we can
        # get the count of uniques, then do a
        # ratio of majority : minority
        _isint = _is_int(X[col], _dtype)
        if _isint or _dtype == 'object':
            _unique = len(X[col].unique())
            _val_cts= X[col].value_counts().sort_values(ascending=True)
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
            _skew_risk = 'high skew' if abs_skew > 1 else 'mod. skew' if (0.5 < abs_skew < 1) else 'symmetric'
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

    return pd.DataFrame.from_dict(stat_out)



def get_numeric(X):
    """Return list of indices of numeric dtypes variables

    Parameters
    ----------
    X : pandas DF
        The dataframe
    """
    validate_is_pd(X, None) # don't want warning
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
        'KB':kb, 
        'MB':float(kb ** 2), 
        'GB':float(kb ** 3), 
        'TB':float(kb ** 4)
    }

    if not unit in units:
        raise ValueError('got %s, expected one of (%s)'
                         % (unit, ', '.join(units.keys())))

    return '%.3f %s' % (b/units[unit], unit)


def is_entirely_numeric(X):
    """Determines whether an entire pandas frame
    is numeric in dtypes.

    Parameters
    ----------
    X : pd DataFrame
        The dataframe to test
    """
    return X.shape[1] == len(get_numeric(X))


def is_numeric(x):
    """Determines whether the arg is numeric

    Parameters
    ----------
    x : anytype
    """
    return isinstance(x, (numbers.Integral, int, float, long, np.int, np.float, np.long))


def load_iris_df(include_tgt=True, tgt_name="Species"):
    """Loads the iris dataset into a dataframe with the
    target set as the "Species" feature or whatever name
    is specified.

    Parameters
    ----------
    include_tgt : bool, optional (default=True)
        Whether to include the target

    tgt_name : str, optional (default="Species")
        The name of the target feature
    """
    iris = load_iris()
    X = pd.DataFrame.from_records(data=iris.data, columns=iris.feature_names)

    if include_tgt:
        X[tgt_name] = iris.target
        
    return X


def report_grid_score_detail(random_search, charts=True, sort_results=True, ascending=True):
    """Input fit grid search estimator. Returns df of scores with details"""
    df_list = []

    for line in random_search.grid_scores_:
        results_dict = dict(line.parameters)
        results_dict["score"] = line.mean_validation_score
        results_dict["std"] = line.cv_validation_scores.std()*1.96
        df_list.append(results_dict)

    result_df = pd.DataFrame(df_list)
    if sort_results:
        result_df = result_df.sort_values("score", ascending=ascending)
    
    # if the import failed, we won't be able to chart here
    if charts and CAN_CHART_MPL:
        for col in get_numeric(result_df):
            if col not in ["score", "std"]:
                plt.scatter(result_df[col], result_df.score)
                plt.title(col)
                plt.show()

        for col in list(result_df.columns[result_df.dtypes == "object"]):
            cat_plot = result_df.score.groupby(result_df[col]).mean()
            cat_plot.sort_values()
            cat_plot.plot(kind="barh", xlim=(.5, None), figsize=(7, cat_plot.shape[0]/2))
            plt.show()
    elif charts and not CAN_CHART:
        warnings.warn('no module matplotlib, will not be able to display charts', ModuleImportWarning)

    return result_df

def report_confusion_matrix(actual, pred, return_metrics=True):
    """Return a dataframe with the confusion matrix, and a series
    with the classification performance metrics.
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
        condition_pos = np.sum(cf[1,:])
        condition_neg = np.sum(cf[0,:])

        # alias the elements in the matrix
        tp = cf[1,1]
        fp = cf[0,1]
        tn = cf[0,0]
        fn = cf[1,0]

        # sums of the prediction cols
        pred_pos = tp + fp
        pred_neg = tn + fn

        acc = (tp+tn) / total_pop       # accuracy
        tpr = tp / condition_pos        # sensitivity, recall
        fpr = fp / condition_neg        # fall-out
        fnr = fn / condition_pos        # miss rate
        tnr = tn / condition_neg        # specificity
        prev= condition_pos / total_pop # prevalence
        plr = tpr / fpr                 # positive likelihood ratio, LR+
        nlr = fnr / tnr                 # negative likelihood ratio, LR-
        dor = plr / nlr                 # diagnostic odds ratio
        prc = tp / pred_pos             # precision, positive predictive value
        fdr = fp / pred_pos             # false discovery rate
        fomr= fn / pred_neg             # false omission rate
        npv = tn / pred_neg             # negative predictive value

        # define the series
        d = {
            'Accuracy'              : acc,
            'Diagnostic odds ratio' : dor,
            'Fall-out'              : fpr,
            'False discovery rate'  : fdr,
            'False Neg. Rate'       : fnr,
            'False omission rate'   : fomr,
            'False Pos. Rate'       : fpr,
            'Miss rate'             : fnr,
            'Neg. likelihood ratio' : nlr,
            'Neg. predictive value' : npv,
            'Pos. likelihood ratio' : plr,
            'Pos. predictive value' : prc,
            'Precision'             : prc,
            'Prevalence'            : prev,
            'Recall'                : tpr,
            'Sensitivity'           : tpr,
            'Specificity'           : tnr,
            'True Pos. Rate'        : tpr,
            'True Neg. Rate'        : tnr
        }

        ser = pd.Series(data=d)
        ser.name = 'Metrics'


    # create the DF
    conf = pd.DataFrame.from_records(data=cf, columns=['Neg','Pos'])
    conf.index = ['Neg','Pos']

    return conf, ser




    