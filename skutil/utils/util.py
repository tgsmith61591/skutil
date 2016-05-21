import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

__all__ = [
	'get_numeric',
	'is_numeric',
	'validate_is_pd'
]

def validate_is_pd(X):
    if not isinstance(X, pd.DataFrame):
        raise ValueError('expected pandas DataFrame')


def get_numeric(X):
    """Return list of indices of numeric dtypes variables

    Parameters
    ----------
    X : pandas DF
        The dataframe
    """
    if not isinstance(X, pd.DataFrame):
        raise ValueError('expected pandas DF')

    return X.dtypes[X.dtypes.apply(lambda x: str(x).startswith(("float", "int", "bool")))].index.tolist()


def is_numeric(x):
	"""Determines whether the X is numeric

    Parameters
    ----------
    x : anytype
    """
	return isinstance(x, (int, float, long, np.int, np.float, np.long))
