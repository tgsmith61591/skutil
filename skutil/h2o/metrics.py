from __future__ import absolute_import, division, print_function
import numpy as np
from h2o.frame import H2OFrame
from .frame import _check_is_1d_frame
from ..metrics import GainsStatisticalReport
from ..base import overrides


__all__ = [
    'h2o_accuracy_score',
]


def _check_targets(*args):
    """Ensures all the args are H2OFrames,
    that each arg is 1 column, and that all
    of the lengths of the columns match.

    Parameters
    ----------
    *args : a collection of H2OFrames
    """
    frms = [_check_is_1d_frame(arg) for arg in args]
    shape = frms[0].shape

    # assert all the same length
    assert all([frame.shape==shape for frame in frms])
    



def _weighted_sum(sample_score, sample_weight):
    """Returns the weighted sum. Adapted from sklearn's 
    sklearn.metrics.classification._weighted_sum
    method for use with H2O frames.

    Parameters
    ----------
    sample_score : H2OFrame
        The binary vector

    sample_weight : H2OFrame
        A frame of weights and of matching dims as
        the sample_score frame.
    """
    if sample_weight is not None:
        _check_targets(sample_score, sample_weight)
        return (sample_score * sample_weight).sum()
    else:
        return sample_score.sum()



def h2o_accuracy_score(y_actual, y_predict, sample_weight=None):
    """Accuracy classification score for H2O

    Parameters
    ----------
    y_actual : 1d H2OFrame
        The ground truth

    y_predict : 1d H2OFrame
        The predicted labels

    sample_weight : 1d H2OFrame, optional (default=None)
        A frame of sample weights of matching dims with
        y_actual and y_predict.

    Returns
    -------
    score : float
    """
    _check_targets(y_actual, y_predict)
    return _weighted_sum(y_actual==y_predict, sample_weight)


