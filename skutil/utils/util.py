import pandas as pd
import numpy as np
import warnings
from sklearn.linear_model import LinearRegression
from ..base import SelectiveWarning

__all__ = [
	'get_numeric',
	'is_numeric',
	'validate_is_pd'
]

def _val_cols(cols):
    # try to make cols a list
    if not hasattr(cols, '__iter__'):
        raise ValueError('cols must be an iterable sequence')
    return [c for c in cols] # make it a list implicitly

def validate_is_pd(X, cols, warn=True):
    """Used within each SelectiveMixin fit method to determine whether
    the passed X is a dataframe, and whether the cols is appropriate.
    There are four scenarios (in the order in which they're checked):

    1) Names is not None, but X is not a dataframe.
        Resolution: the method will attempt to return a DataFrame from the
        args provided (with the cols arg as the column names), but catches any
        exception and raises a ValueError. A common case where this would work
        may be a numpy.ndarray as X, and a list as cols.

    2) X is a DataFrame, but cols is None.
        Resolution: return a copy of the dataframe, and use all column names.

    3) X is a DataFrame and cols is not None.
        Return a copy of the dataframe, and use only the names provided.

    4) X is not a DataFrame, and cols is None.
        Resolution: this case will only work if the X can be built into a DataFrame.
        Otherwise, there will be a ValueError thrown.

    Returns
    -------
    tuple, (DataFrame: X, list: cols)
    """

    # case 1, we have names but the X is not a frame
    if not isinstance(X, pd.DataFrame) and cols is not None:
        try:
            cols = _val_cols(cols)
            return pd.DataFrame.from_records(data=X, columns=cols), cols
        except Exception as e:
            raise ValueError('expected pandas DataFrame if passed cols arg')

    # case 2, we have a DF but no cols
    elif not cols:
        try:
            return X.copy(), X.columns.values.tolist()
        except AttributeError as e:
            # this happens if the X is a series, and not actually a frame
            raise ValueError('got Series but expected DataFrame')

    # case 3, we have a DF AND cols
    elif cols is not None:
        cols = _val_cols(cols)
        return X.copy(), cols

    # case 4, we have neither a frame nor cols (maybe JUST a np.array?)
    else:
        # in balancers, the names won't matter so disable warn
        if warn:
            warnings.warn('X is not a DataFrame, and y is None', SelectiveWarning)

        try:
            df = pd.DataFrame.from_records(data=X)
            return df, df.columns.values.tolist()
        except Exception as e:
            raise ValueError('cannot create dataframe from X')




def get_numeric(X):
    """Return list of indices of numeric dtypes variables

    Parameters
    ----------
    X : pandas DF
        The dataframe
    """
    validate_is_pd(X, None, False) # don't want warning
    return X.dtypes[X.dtypes.apply(lambda x: str(x).startswith(("float", "int", "bool")))].index.tolist()


def is_numeric(x):
	"""Determines whether the X is numeric

    Parameters
    ----------
    x : anytype
    """
	return isinstance(x, (int, float, long, np.int, np.float, np.long))
