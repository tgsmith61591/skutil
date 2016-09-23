from __future__ import absolute_import, division, print_function
import numpy as np
from h2o.frame import H2OFrame
from sklearn.externals import six
from .base import _check_is_frame, check_version
from ..utils import flatten_all



__all__ = [
    '_check_is_1d_frame'
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
