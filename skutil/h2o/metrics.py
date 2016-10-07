from __future__ import absolute_import, division, print_function
import numpy as np
import abc
from h2o.frame import H2OFrame
from sklearn.externals import six
from .frame import _check_is_1d_frame
from ..metrics import GainsStatisticalReport
from ..base import overrides
from ..utils import flatten_all


__all__ = [
    'h2o_accuracy_score',
    'h2o_mean_absolute_error',
    'h2o_mean_squared_error',
    'h2o_median_absolute_error',
    'h2o_r2_score',
    'make_h2o_scorer',
]


def _get_bool(x):
    """H2O is_() operations often return
    a list of booleans (even when one column),
    so we need to extract the True/False value

    Parameter
    ---------
    x : bool or iterable
        The boolean to extract
    """
    if hasattr(x, '__iter__'):
        return flatten_all(x)[0]
    return x


def _err_for_continuous(typ):
    if typ == 'continuous':
        raise ValueError('continuous response unsupported for classification metric')

def _err_for_discrete(typ):
    if typ != 'continuous':
        raise ValueError('discrete response unsupported for regression metric')

def _is_int(x):
    if not x.isnumeric():
        return False
    return (x.round(digits=0) - x).sum() == 0

def _get_mean(x):
    """Internal method. Gets the mean from
    an H2O frame (single col). Since the mean
    inconsistently returns a list in some versions,
    this extracts the value.
    """
    return flatten_all(x.mean())[0]


def _unique_labels(y_true, y_pred):
    df = y_true.unique().rbind(y_pred.unique()).unique()
    return df[df.columns[0]].tolist()


def _type_of_target(y):
    """Determine the type of data indicated by target `y`.
    Adapted from sklearn.utils.multiclass.type_of_target.
    If is int, will treat the column as a factor.

    Parameters
    ----------
    y : H2OFrame
        the y variable

    Returns
    -------
    target_type : string
        One of:
        * 'continuous'
        * 'binary'
        * 'multiclass'
        * 'unknown'
    """
    _check_is_1d_frame(y)
    if _get_bool(y.isfactor()) or _is_int(y):
        unq = y.unique()
        return 'unknown' if unq.shape[0] < 2 else\
            'binary' if unq.shape[0] == 2 else\
            'multiclass'
    return 'continuous'


def _check_targets(y_true, y_pred, y_type=None):
    """Ensures all the args are H2OFrames,
    that each arg is 1 column, and that all
    of the lengths of the columns match.

    Parameters
    ----------
    y_true, y_pred : both H2OFrames
    """
    frms = [_check_is_1d_frame(arg) for arg in (y_true, y_pred)]
    shape = frms[0].shape

    # assert all the same length
    assert all([frame.shape==shape for frame in frms])

    if y_type is None:
        # get type of truth
        y_type = _type_of_target(y_true)
        if y_type == 'unknown':
            raise ValueError('cannot determine datatype of y_true: is it all the same value?')


    # TODO: more?
    return y_type, y_true, y_pred
    


def _average(score, weights=None):
    if weights is not None:
        x = score * weights
    else:
        x = score
    return x.sum() / x.shape[0]


def _weighted_sum(sample_score, sample_weight, normalize):
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
    if normalize:
        return _average(sample_score, weights=sample_weight)
    if sample_weight is not None:
        return (sample_score * sample_weight).sum()
    else:
        return sample_score.sum()


def h2o_accuracy_score(y_actual, y_predict, normalize=True, 
                       sample_weight=None, y_type=None):
    """Accuracy classification score for H2O

    Parameters
    ----------
    y_actual : 1d H2OFrame
        The ground truth

    y_predict : 1d H2OFrame
        The predicted labels

    normalize : bool, optional (default=True)
        Whether to average the data

    sample_weight : 1d H2OFrame, optional (default=None)
        A frame of sample weights of matching dims with
        y_actual and y_predict.

    y_type : string, optional (default=None)
        The type of the column. If None, will be determined.

    Returns
    -------
    score : float
    """
    y_type, y_actual, y_predict = _check_targets(y_actual, y_predict, y_type)
    _err_for_continuous(y_type)
    return _weighted_sum(y_actual==y_predict, sample_weight, normalize)



def _h2o_ae(y_actual, y_predict, sample_weight=None, y_type=None):
    """Compute absolute difference between actual and predict"""
    y_type, y_actual, y_predict = _check_targets(y_actual, y_predict)
    _err_for_discrete(y_type)

    # compute abs diff
    abs_diff = (y_actual - y_predict).abs()

    # apply sample weight if necessary
    if sample_weight is not None:
        abs_diff *= sample_weight

    return abs_diff



def h2o_mean_absolute_error(y_actual, y_predict, sample_weight=None, y_type=None):
    """MAE score for H2O frames

    Parameters
    ----------
    y_actual : 1d H2OFrame
        The ground truth

    y_predict : 1d H2OFrame
        The predicted labels

    sample_weight : 1d H2OFrame, optional (default=None)
        A frame of sample weights of matching dims with
        y_actual and y_predict.

    y_type : string, optional (default=None)
        The type of the column. If None, will be determined.

    Returns
    -------
    score : float
    """
    return _get_mean(_h2o_ae(y_actual, y_predict, sample_weight, y_type))



def h2o_median_absolute_error(y_actual, y_predict, sample_weight=None, y_type=None):
    """Median abs error score for H2O frames

    Parameters
    ----------
    y_actual : 1d H2OFrame
        The ground truth

    y_predict : 1d H2OFrame
        The predicted labels

    sample_weight : 1d H2OFrame, optional (default=None)
        A frame of sample weights of matching dims with
        y_actual and y_predict.

    y_type : string, optional (default=None)
        The type of the column. If None, will be determined.

    Returns
    -------
    score : float
    """
    return flatten_all(_h2o_ae(y_actual, y_predict, sample_weight, y_type).median())[0]



def h2o_r2_score(y_actual, y_predict, sample_weight=None, y_type=None):
    """R^2 score for H2O frames

    Parameters
    ----------
    y_actual : 1d H2OFrame
        The ground truth

    y_predict : 1d H2OFrame
        The predicted labels

    sample_weight : 1d H2OFrame, optional (default=None)
        A frame of sample weights of matching dims with
        y_actual and y_predict.

    y_type : string, optional (default=None)
        The type of the column. If None, will be determined.

    Returns
    -------
    score : float
    """

    y_type, y_actual, y_predict = _check_targets(y_actual, y_predict)
    _err_for_discrete(y_type)

    # compute the numerator & denominator precursors
    diff = (y_actual - y_predict)
    sq_diff = diff * diff
    mean_centered = y_actual - _get_mean(y_actual)
    sq_mean_centered = mean_centered * mean_centered

    # compute the numerator and denominator
    if sample_weight is not None:
        numerator = (sample_weight * sq_diff).sum()
        denominator = (sample_weight * sq_mean_centered).sum()
    else:
        numerator = sq_diff.sum()
        denominator = sq_mean_centered.sum()


    nonzero_denom = denominator != 0
    nonzero_numer = numerator != 0
    valid_score = nonzero_numer & nonzero_denom

    # generate output
    return 1 - (numerator[valid_score] / denominator[valid_score])



def h2o_mean_squared_error(y_actual, y_predict, sample_weight=None, y_type=None):
    """MSE score for H2O frames

    Parameters
    ----------
    y_actual : 1d H2OFrame
        The ground truth

    y_predict : 1d H2OFrame
        The predicted labels

    sample_weight : 1d H2OFrame, optional (default=None)
        A frame of sample weights of matching dims with
        y_actual and y_predict.

    y_type : string, optional (default=None)
        The type of the column. If None, will be determined.

    Returns
    -------
    score : float
    """

    y_type, y_actual, y_predict = _check_targets(y_actual, y_predict)
    _err_for_discrete(y_type)

    # compute abs diff
    diff = (y_actual - y_predict)
    diff *= diff # square it...

    # apply sample weight if necessary
    if sample_weight is not None:
        abs_diff *= sample_weight

    return flatten_all(abs_diff.mean())[0]



def make_h2o_scorer(score_function, y_true):
    """Make a scoring function from a callable.
    The signature for the callable should resemble:

        ```some_function(y_true, y_pred, y_type=None...)```

    Parameters
    ----------
    score_function : callable
        The function

    y_true : H2OFrame
        An H2O frame (the ground truth). This is
        used to determine before hand whether the
        type is binary or multiclass.
    """
    return _H2OScorer(score_function, y_true).score


class _H2OScorer(six.with_metaclass(abc.ABCMeta)):
    """A class that wraps a custom scoring function for use
    with H2OFrames. The first two arguments in the scoring function
    signature should resemble the following: 

        ```some_function(y_true, y_pred, y_type=None...)```

    Any specific scoring kwargs should be passed to the ```score```
    function in the class instance.

    Parameters
    ----------
    score_function : callable
        The function

    y_true : H2OFrame
        An H2O frame (the ground truth). This is
        used to determine before hand whether the
        type is binary or multiclass.
    """

    def __init__(self, score_function, y_true):
        if not hasattr(score_function, '__call__'):
            raise TypeError('score_function must be callable')

        self.fun_ = score_function
        self.y_type = _type_of_target(y_true)

    def score(self, y_true, y_pred, **kwargs):
        # confirm are h2o frames
        for fr in (y_true, y_pred):
            _check_is_1d_frame(fr)

        return self.fun_(y_true, y_pred, y_type=self.y_type, **kwargs)
