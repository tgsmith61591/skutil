from __future__ import absolute_import, division, print_function
from h2o.frame import H2OFrame
from .base import _check_is_frame
from ..utils import flatten_all

__all__ = [
    '_check_is_1d_frame',
    'is_numeric',
    'is_integer',
    'is_float'
]


def _check_is_1d_frame(X):
    """Check whether X is an H2OFrame
    and that it's a 1d column.

    Parameters
    ----------

    X : H2OFrame
        The H2OFrame

    Returns
    -------

    X : H2OFrame
    """
    X = _check_is_frame(X)
    assert X.shape[1] == 1, 'expected 1d H2OFrame'

    return X


def is_numeric(x):
    """Determine whether a 1d H2OFrame is numeric.

    Parameters
    ----------

    x : H2OFrame, 1d
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

    x : H2OFrame, 1d
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

    x : H2OFrame, 1d
        The H2OFrame

    Returns
    -------

    bool : True if float, else False
    """
    _check_is_1d_frame(x)
    return is_numeric(x) and not is_integer(x)
