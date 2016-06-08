import pandas as pd
import numpy as np
import warnings
from sklearn.linear_model import LinearRegression
from ..base import SelectiveWarning, ModuleImportWarning


# check if matplotlib exists
__can_chart__ = True
try:
    # this causes a UserWarning to be thrown by matplotlib... should we squelch this?
    import matplotlib as mpl
    mpl.use('TkAgg') # set backend
    from matplotlib import pyplot as plt
except ImportError as ie:
    __can_chart__ = False
    warnings.warn('no module matplotlib, will not be able to display charts', ModuleImportWarning)


__all__ = [
    'flatten_all',
    'flatten_all_generator',
    'get_numeric',
    'is_numeric',
    'report_grid_score_detail',
    'validate_is_pd'
]

def _val_cols(cols):
    # if it's None, return immediately
    if cols is None:
        return cols

    # try to make cols a list
    if not hasattr(cols, '__iter__'):
        raise ValueError('cols must be an iterable sequence')
    return [c for c in cols] # make it a list implicitly, make no guarantees about elements

def _def_headers(X):
    return ['V%i' %  (i+1) for i in range(X.shape[1])]

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
        try:
            # this is tough, because they only pass cols if it's a subset
            # and this frame is likely too large for the passed columns.
            # so, we hope they either passed what the col names WILL be
            # or that they passed numeric cols... they should handle that
            # validation on their end, though.
            return pd.DataFrame.from_records(data=X, columns=_def_headers(X)), cols
        except Exception as e:
            print(e)
            raise ValueError('expected pandas DataFrame if passed cols arg')

    # case 2, we have a DF but no cols, def behavior: use all
    elif is_df and cols is None:
        return X.copy(), None

    # case 3, we have a DF AND cols
    elif is_df and cols is not None:
        return X.copy(), cols

    # case 4, we have neither a frame nor cols (maybe JUST a np.array?)
    else:
        # we'll do two tests here... either that it's a np ndarray or a list of lists
        if isinstance(X, np.ndarray):
            return pd.DataFrame.from_records(data=X, columns=_def_headers(X)), None

        # otherwise check for list of lists...
        if hasattr(X, '__iter__') and all(isinstance(elem, list) for elem in X):
            try:
                return pd.DataFrame.from_records(data=X, columns=_def_headers(X)), None
            except Exception as e:
                raise ValueError('cannot create dataframe from X')

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

    