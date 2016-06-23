from __future__ import print_function, division
import pandas as pd
import numpy as np
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix as cm
from ..base import SelectiveWarning, ModuleImportWarning


# check if matplotlib exists
__can_chart__ = True
try:
    # this causes a UserWarning to be thrown by matplotlib... should we squelch this?
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # do actual import
        import matplotlib as mpl
        mpl.use('TkAgg') # set backend
        from matplotlib import pyplot as plt
except ImportError as ie:
    __can_chart__ = False
    warnings.warn('no module matplotlib, will not be able to display charts', ModuleImportWarning)


__all__ = [
    'exp',
    'flatten_all',
    'flatten_all_generator',
    'get_numeric',
    'is_entirely_numeric',
    'is_numeric',
    'log',
    'report_confusion_matrix',
    'report_grid_score_detail',
    'validate_is_pd'
]

__max_exp__ = 1e19
__min_log__ = -19


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

def validate_is_pd(X, cols):
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




def get_numeric(X):
    """Return list of indices of numeric dtypes variables

    Parameters
    ----------
    X : pandas DF
        The dataframe
    """
    validate_is_pd(X, None) # don't want warning
    return X.dtypes[X.dtypes.apply(lambda x: str(x).startswith(("float", "int", "bool")))].index.tolist()


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
    return isinstance(x, (int, float, long, np.int, np.float, np.long))

def report_grid_score_detail(random_search, charts=True):
    """Input fit grid search estimator. Returns df of scores with details"""
    df_list = []

    for line in random_search.grid_scores_:
        results_dict = dict(line.parameters)
        results_dict["score"] = line.mean_validation_score
        results_dict["std"] = line.cv_validation_scores.std()*1.96
        df_list.append(results_dict)

    result_df = pd.DataFrame(df_list)
    result_df = result_df.sort_values("score", ascending=False)
    
    # if the import failed, we won't be able to chart here
    if charts and __can_chart__:
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




    