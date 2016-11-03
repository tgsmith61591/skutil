from __future__ import absolute_import, division, print_function
from h2o.frame import H2OFrame
import pandas as pd
from .base import check_frame
from ..utils import flatten_all

__all__ = [
    '_check_is_1d_frame',
    'as_series',
    'is_numeric',
    'is_integer',
    'is_float'
]


def _check_is_1d_frame(X):
    """Check whether X is an H2OFrame
    and that it's a 1d column. If not, will
    raise an AssertionError

    Parameters
    ----------

    X : H2OFrame, shape=(n_samples, 1)
        The H2OFrame to check

    Raises
    ------

    AssertionError if the ``X`` variable
    is not a 1-dimensional H2OFrame.

    Returns
    -------

    X : H2OFrame, shape=(n_samples, 1)
        The frame if is 1d
    """
    X = check_frame(X, copy=False)
    assert X.shape[1] == 1, 'expected 1d H2OFrame'

    return X


def as_series(x):
    """Make a 1d H2OFrame into a pd.Series.

    Parameters
    ----------

    x : H2OFrame, shape=(n_samples, 1)
        The H2OFrame

    Returns
    -------

    x : pd.Series, shape=(n_samples,)
        The pandas series
    """
    x = _check_is_1d_frame(x)
    x = x.as_data_frame(use_pandas=True)[x.columns[0]]
    return x


def is_numeric(x):
    """Determine whether a 1d H2OFrame is numeric.

    Parameters
    ----------

    x : H2OFrame, shape=(n_samples, 1)
        The H2OFrame

    Returns
    -------

    bool : True if numeric, else False
    """
    _check_is_1d_frame(x)
    return flatten_all(x.isnumeric())[0]


def is_integer(x):
    """Determine whether a 1d H2OFrame is 
    made up of integers.

    Parameters
    ----------

    x : H2OFrame, shape=(n_samples, 1)
        The H2OFrame

    Returns
    -------

    bool : True if integers, else False
    """
    _check_is_1d_frame(x)
    if not is_numeric(x):
        return False
    return (x.round(digits=0) - x).sum() == 0


def is_float(x):
    """Determine whether a 1d H2OFrame is
    made up of floats.

    Parameters
    ----------

    x : H2OFrame, shape=(n_samples, 1)
        The H2OFrame

    Returns
    -------

    bool : True if float, else False
    """
    _check_is_1d_frame(x)
    return is_numeric(x) and not is_integer(x)
