# -*- coding: utf-8 -*-
"""Metrics for scoring H2O model predictions"""
# Author: Taylor Smith
# adapted from sklearn for use with skutil & H2OFrames

from __future__ import absolute_import, division, print_function
import abc
import warnings
import numpy as np
from sklearn.externals import six
from .frame import _check_is_1d_frame, is_integer
from .encode import H2OLabelEncoder
from .util import h2o_bincount, h2o_col_to_numpy
from ..utils import flatten_all
from ..utils.fixes import is_iterable
from ..base import since

__all__ = [
    'h2o_accuracy_score',
    'h2o_f1_score',
    'h2o_fbeta_score',
    'h2o_mean_absolute_error',
    'h2o_mean_squared_error',
    'h2o_median_absolute_error',
    'h2o_precision_score',
    'h2o_recall_score',
    'h2o_r2_score',
    'make_h2o_scorer'
]


def _get_bool(x):
    """H2O is_() operations often return
    a list of booleans (even when one column),
    so we need to extract the True/False value

    Parameters
    ----------

    x : bool or iterable
        The boolean to extract
    """
    if is_iterable(x):
        return flatten_all(x)[0]
    return x


def _err_for_continuous(typ):
    """Throw ValueError if typ is
    continuous. Used as a utility
    for type checking.
    """
    if typ == 'continuous':
        raise ValueError('continuous response unsupported for classification metric')


def _err_for_discrete(typ):
    """Throw ValueError if typ is
    not continuous. Used as a utility
    for type checking.
    """
    if typ != 'continuous':
        raise ValueError('discrete response unsupported for regression metric')


def _get_mean(x):
    """Internal method. Gets the mean from
    an H2O frame (single col). Since the mean
    inconsistently returns a list in some versions,
    this extracts the value.
    """
    return flatten_all(x.mean())[0]


def _type_of_target(y):
    """Determine the type of data indicated by target `y`.
    Adapted from sklearn.utils.multiclass.type_of_target.
    If is int, will treat the column as a factor.

    Note that this can be achieved using h2o frames' ``types``,
    however, if there is a 'real' type that is only "nominally
    real," i.e., [1.0, 2.0, 3.0], we will treat them as ints.

    Parameters
    ----------

    y : ``H2OFrame``, shape=(n_samples,)
        the y variable

    Returns
    -------

    string
        The target type. One of:
            * 'continuous'
            * 'binary'
            * 'multiclass'
            * 'unknown'
    """
    _check_is_1d_frame(y)
    if _get_bool(y.isfactor()) or is_integer(y):
        unq = y.unique()
        return 'unknown' if unq.shape[0] < 2 else \
            'binary' if unq.shape[0] == 2 else \
            'multiclass'
    return 'continuous'


def _check_targets(y_true, y_pred, y_type=None):
    """Ensures all the args are H2OFrames,
    that each arg is 1 column, and that all
    of the lengths of the columns match.

    Parameters
    ----------

    y_true : ``H2OFrame``, shape=(n_samples,)
        A 1d ``H2OFrame`` of the ground truth.

    y_pred : ``H2OFrame``, shape=(n_samples,)
        A 1d ``H2OFrame`` of the predictions.

    y_type : string, optional (default=None)
        If provided, will not test for type.
        If None, this method will determine the type
        of column (for ``y_true``). Note that this can
        be expensive. Thus, the ``make_h2o_scorer`` function
        caches this variable in the scoring instance.

    Returns
    -------

    y_type : string
        The type of frame. One of: 
        ('multiclass', 'continuous', 'binary', 'unknown')

    y_true : ``H2OFrame``, shape=(n_samples,)
        A 1d ``H2OFrame`` of the ground truth.

    y_pred : ``H2OFrame``, shape=(n_samples,)
        A 1d ``H2OFrame`` of the predictions.
    """
    frms = [_check_is_1d_frame(arg) for arg in (y_true, y_pred)]
    shape = frms[0].shape

    # assert all the same length
    assert all([frame.shape == shape for frame in frms])

    if y_type is None:
        # get type of truth
        y_type = _type_of_target(y_true)
        if y_type == 'unknown':
            raise ValueError('cannot determine datatype of y_true: is it all the same value?')

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

    sample_score : ``H2OFrame``, shape=(n_samples)
        The binary vector

    sample_weight : ``H2OFrame``, shape=(n_samples) or float
        A frame of weights and of matching dims as
        the sample_score frame.

    normalize : bool
        Whether or not to normalize the sum by
        the number of observations (equivalent to
        an average).

    Returns
    -------

    float
        The weighted sum
    """
    if normalize:
        return _average(sample_score, weights=sample_weight)
    if sample_weight is not None:
        return (sample_score * sample_weight).sum()
    else:
        return sample_score.sum()


@since('0.1.0')
def h2o_accuracy_score(y_actual, y_predict, normalize=True,
                       sample_weight=None, y_type=None):
    """Accuracy classification score for H2O

    Parameters
    ----------

    y_actual : ``H2OFrame``, shape=(n_samples,)
        The one-dimensional ground truth

    y_predict : ``H2OFrame``, shape=(n_samples,)
        The one-dimensional predicted labels

    normalize : bool, optional (default=True)
        Whether to average the data

    sample_weight : H2OFrame or float, optional (default=None)
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
    return _weighted_sum(y_actual == y_predict, sample_weight, normalize)


@since('0.1.0')
def h2o_f1_score(y_actual, y_predict, labels=None, pos_label=1, average='binary',
                 sample_weight=None, y_type=None):
    """Compute the F1 score, the weighted average of the precision 
    and the recall:

        ``F1 = 2 * (precision * recall) / (precision + recall)``

    Parameters
    ----------

    y_actual : ``H2OFrame``, shape=(n_samples,)
        The one-dimensional ground truth

    y_predict : ``H2OFrame``, shape=(n_samples,)
        The one-dimensional predicted labels

    labels : list, optional (default=None)
        The set of labels to include when ``average != 'binary'``, and their
        order if ``average is None``. By default all labels in ``y_actual`` and
        ``y_predict`` are used in sorted order.

    pos_label : str or int, optional (default=1)
        The class to report if ``average=='binary'`` and the data is binary.
        If the data are multiclass, this will be ignored.

    average : str, optional (default='binary')
        One of ('binary', 'micro', 'macro', 'weighted'). This parameter is
        required for multiclass targets. If ``None``, the scores for each 
        class are returned. Otherwise, this determines the type of averaging
        performed on the data:

        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.

        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.

        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.

        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.

    sample_weight : H2OFrame or float, optional (default=None)
        The sample weights

    y_type : string, optional (default=None)
        The type of the column. If None, will be determined.

    Returns
    -------

    f : float
        The F-1 score
    """
    return h2o_fbeta_score(y_actual, y_predict, 1.0, labels=labels,
                           pos_label=pos_label, average=average,
                           sample_weight=sample_weight, y_type=y_type)


@since('0.1.0')
def h2o_fbeta_score(y_actual, y_predict, beta, labels=None, pos_label=1,
                    average='binary', sample_weight=None, y_type=None):
    """Compute the F-beta score.  The F-beta score is the weighted harmonic 
    mean of precision and recall.

    Parameters
    ----------

    y_actual : ``H2OFrame``, shape=(n_samples,)
        The one-dimensional ground truth

    y_predict : ``H2OFrame``, shape=(n_samples,)
        The one-dimensional predicted labels

    beta : float
        The beta value for the F-score

    labels : list, optional (default=None)
        The set of labels to include when ``average != 'binary'``, and their
        order if ``average is None``. By default all labels in ``y_actual`` and
        ``y_predict`` are used in sorted order.

    pos_label : str or int, optional (default=1)
        The class to report if ``average=='binary'`` and the data is binary.
        If the data are multiclass, this will be ignored.

    average : str, optional (default='binary')
        One of ('binary', 'micro', 'macro', 'weighted'). This parameter is
        required for multiclass targets. If ``None``, the scores for each 
        class are returned. Otherwise, this determines the type of averaging
        performed on the data:

        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.

        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.

        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.

        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.

    sample_weight : H2OFrame or float, optional (default=None)
        The sample weights

    y_type : string, optional (default=None)
        The type of the column. If None, will be determined.

    Returns
    -------

    f : float
        The F-beta score
    """
    _, _, f, _ = h2o_precision_recall_fscore_support(y_actual, y_predict,
                                                     beta=beta,
                                                     labels=labels,
                                                     pos_label=pos_label,
                                                     average=average,
                                                     warn_for=('f-score',),
                                                     sample_weight=sample_weight,
                                                     y_type=y_type)
    return f


@since('0.1.0')
def h2o_precision_score(y_actual, y_predict, labels=None, pos_label=1,
                        average='binary', sample_weight=None, y_type=None):
    """Compute the precision.  Precision is the ratio ``tp / (tp + fp)`` where ``tp`` 
    is the number of true positives and ``fp`` the number of false positives.

    Parameters
    ----------

    y_actual : ``H2OFrame``, shape=(n_samples,)
        The one-dimensional ground truth

    y_predict : ``H2OFrame``, shape=(n_samples,)
        The one-dimensional predicted labels

    labels : list, optional (default=None)
        The set of labels to include when ``average != 'binary'``, and their
        order if ``average is None``. By default all labels in ``y_actual`` and
        ``y_predict`` are used in sorted order.

    pos_label : str or int, optional (default=1)
        The class to report if ``average=='binary'`` and the data is binary.
        If the data are multiclass, this will be ignored.

    average : str, optional (default='binary')
        One of ('binary', 'micro', 'macro', 'weighted'). This parameter is
        required for multiclass targets. If ``None``, the scores for each 
        class are returned. Otherwise, this determines the type of averaging
        performed on the data:

        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.

        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.

        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.

        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.

    sample_weight : H2OFrame or float, optional (default=None)
        The sample weights

    y_type : string, optional (default=None)
        The type of the column. If None, will be determined.

    Returns
    -------

    p : float
        The precision score
    """

    p, _, _, _ = h2o_precision_recall_fscore_support(y_actual, y_predict,
                                                     labels=labels,
                                                     pos_label=pos_label,
                                                     average=average,
                                                     warn_for=('precision',),
                                                     sample_weight=sample_weight,
                                                     y_type=y_type)

    return p


@since('0.1.0')
def h2o_recall_score(y_actual, y_predict, labels=None, pos_label=1,
                     average='binary', sample_weight=None, y_type=None):
    """Compute the recall

    Precision is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives.

    Parameters
    ----------

    y_actual : ``H2OFrame``, shape=(n_samples,)
        The one-dimensional ground truth

    y_predict : ``H2OFrame``, shape=(n_samples,)
        The one-dimensional predicted labels

    labels : list, optional (default=None)
        The set of labels to include when ``average != 'binary'``, and their
        order if ``average is None``. By default all labels in ``y_actual`` and
        ``y_predict`` are used in sorted order.

    pos_label : str or int, optional (default=1)
        The class to report if ``average=='binary'`` and the data is binary.
        If the data are multiclass, this will be ignored.

    average : str, optional (default='binary')
        One of ('binary', 'micro', 'macro', 'weighted'). This parameter is
        required for multiclass targets. If ``None``, the scores for each 
        class are returned. Otherwise, this determines the type of averaging
        performed on the data:

        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.

        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.

        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.

        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.

    sample_weight : H2OFrame, optional (default=None)
        The sample weights

    y_type : string, optional (default=None)
        The type of the column. If None, will be determined.

    Returns
    -------

    r : float
        The recall score
    """

    _, r, _, _ = h2o_precision_recall_fscore_support(y_actual, y_predict,
                                                     labels=labels,
                                                     pos_label=pos_label,
                                                     average=average,
                                                     warn_for=('precision',),
                                                     sample_weight=sample_weight,
                                                     y_type=y_type)

    return r


def h2o_precision_recall_fscore_support(y_actual, y_predict, beta=1.0, pos_label=1,
                                        sample_weight=None, y_type=None, average=None,
                                        labels=None, warn_for=('precision', 'recall',
                                                               'f-score')):
    average_options = (None, 'micro', 'macro', 'weighted')
    if average not in average_options and average != 'binary':
        raise ValueError('average should be one of %s'
                         % str(average_options))
    if beta <= 0:
        raise ValueError('beta should be >0 in the F-beta score')

    y_type, y_actual, y_predict = _check_targets(y_actual, y_predict, y_type)
    _err_for_continuous(y_type)

    # get all the unique labels
    present_labels = sorted(h2o_col_to_numpy(y_actual.unique().rbind(y_predict.unique()).unique()))

    if average == 'binary':
        if y_type == 'binary':
            if pos_label not in present_labels:
                if len(present_labels) < 2:
                    # only negative
                    return 0.0, 0.0, 0.0, 0.0
                else:
                    raise ValueError("pos_label=%r is not a valid label: %r"
                                     % (pos_label, present_labels))

            labels = [pos_label]
        else:
            raise ValueError('Target is %s but average="binary". Choose '
                             'another average setting' % y_type)
    elif pos_label not in (None, 1):
        warnings.warn('Note that pos_label (set to %r) is ignored when '
                      'average != "binary" (got %r). You may use '
                      'labels=[pos_label] to specify a single positive class.'
                      % (pos_label, average), UserWarning)

    if labels is None:
        labels = present_labels
        n_labels = None
    else:
        n_labels = len(labels)
        labels = np.hstack([labels, np.setdiff1d(present_labels, labels,
                                                 assume_unique=True)])

    # calculate tp_sum, pred_sum, true_sum

    le = H2OLabelEncoder()
    y_actual = le.fit_transform(y_actual)
    y_predict = le.transform(y_predict)
    sorted_labels = le.classes_

    # labels now from 0 to len(labels) - 1
    tp = y_actual == y_predict
    tp_bins = y_actual[tp]
    if sample_weight is not None:
        tp_bins_weights = sample_weight[tp]
    else:
        tp_bins_weights = None

    if tp_bins.shape[0]:
        tp_sum = h2o_bincount(tp_bins, weights=tp_bins_weights,
                              minlength=len(labels))
    else:
        true_sum = pred_sum = tp_sum = np.zeros(len(labels))

    if y_predict.shape[0]:
        pred_sum = h2o_bincount(y_predict, weights=sample_weight,
                                minlength=len(labels))
    if y_actual.shape[0]:
        true_sum = h2o_bincount(y_actual, weights=sample_weight,
                                minlength=len(labels))

    # Retain only selected labels
    indices = np.searchsorted(sorted_labels, labels[:n_labels])
    tp_sum = tp_sum[indices]
    true_sum = true_sum[indices]
    pred_sum = pred_sum[indices]

    if average == 'micro':
        tp_sum = np.array([tp_sum.sum()])
        pred_sum = np.array([pred_sum.sum()])
        true_sum = np.array([true_sum.sum()])

    # Now do divisions

    beta2 = beta ** 2
    with np.errstate(divide='ignore', invalid='ignore'):
        # Divide, and on zero-division, set scores to 0 and warn:

        # Oddly, we may get an "invalid" rather than a "divide" error
        # here.
        precision = _prf_divide(tp_sum, pred_sum,
                                'precision', 'predicted', average, warn_for)
        recall = _prf_divide(tp_sum, true_sum,
                             'recall', 'true', average, warn_for)
        # Don't need to warn for F: either P or R warned, or tp == 0 where pos
        # and true are nonzero, in which case, F is well-defined and zero
        f_score = ((1 + beta2) * precision * recall /
                   (beta2 * precision + recall))
        f_score[tp_sum == 0] = 0.0

    # averaging the results

    if average == 'weighted':
        weights = true_sum
        if weights.sum() == 0:
            return 0, 0, 0, None
    else:
        weights = None

    if average is not None:
        assert average != 'binary' or len(precision) == 1
        precision = np.average(precision, weights=weights)
        recall = np.average(recall, weights=weights)
        f_score = np.average(f_score, weights=weights)
        true_sum = None  # return no support

    return precision, recall, f_score, true_sum


def _prf_divide(numerator, denominator, metric, modifier, average, warn_for):
    """Adapted from sklearn.metrics for use with skutil and
    H2OFrames in particular.
    """

    result = numerator / denominator
    mask = denominator == 0.0
    if not np.any(mask):
        return result

    # remove infs
    result[mask] = 0.0

    axis0 = 'sample'
    axis1 = 'label'
    if average == 'samples':
        axis0, axis1 = axis1, axis0

    # build appropriate warning
    if metric in warn_for and 'f-score' in warn_for:
        msg_start = '{0} and F-score are'.format(metric.title())
    elif metric in warn_for:
        msg_start = '{0} is'.format(metric.title())
    elif 'f-score' in warn_for:
        msg_start = 'F-score is'
    else:
        return result

    msg = ('{0} ill-defined and being set to 0.0 {{0}} '
           'no {1} {2}s.'.format(msg_start, modifier, axis0))
    if len(mask) == 1:
        msg = msg.format('due to')
    else:
        msg = msg.format('in {0}s with'.format(axis1))
    warnings.warn(msg, UserWarning, stacklevel=2)
    return result


def _h2o_ae(y_actual, y_predict, sample_weight=None):
    """Compute absolute difference between actual and predict"""
    y_type, y_actual, y_predict = _check_targets(y_actual, y_predict)
    _err_for_discrete(y_type)

    # compute abs diff
    abs_diff = (y_actual - y_predict).abs()

    # apply sample weight if necessary
    if sample_weight is not None:
        abs_diff *= sample_weight

    return abs_diff


@since('0.1.0')
def h2o_mean_absolute_error(y_actual, y_predict, sample_weight=None, y_type=None):
    """Mean absolute error score for H2O frames. Provides fast computation
    in a distributed fashion without loading all of the data into memory.

    Parameters
    ----------

    y_actual : ``H2OFrame``, shape=(n_samples,)
        The one-dimensional ground truth

    y_predict : ``H2OFrame``, shape=(n_samples,)
        The one-dimensional predicted labels

    sample_weight : H2OFrame or float, optional (default=None)
        A frame of sample weights of matching dims with
        y_actual and y_predict.

    y_type : string, optional (default=None)
        The type of the column. If None, will be determined.

    Returns
    -------

    score : float
        The mean absolute error
    """
    _err_for_discrete(y_type)
    score = _get_mean(_h2o_ae(y_actual, y_predict, sample_weight))
    return score


@since('0.1.0')
def h2o_median_absolute_error(y_actual, y_predict, sample_weight=None, y_type=None):
    """Median absolute error score for H2O frames. Provides fast computation
    in a distributed fashion without loading all of the data into memory.

    Parameters
    ----------

    y_actual : ``H2OFrame``, shape=(n_samples,)
        The one-dimensional ground truth

    y_predict : ``H2OFrame``, shape=(n_samples,)
        The one-dimensional predicted labels

    sample_weight : H2OFrame or float, optional (default=None)
        A frame of sample weights of matching dims with
        y_actual and y_predict.

    y_type : string, optional (default=None)
        The type of the column. If None, will be determined.

    Returns
    -------

    score : float
        The median absolute error score
    """
    _err_for_discrete(y_type)
    score = flatten_all(_h2o_ae(y_actual, y_predict, sample_weight).median())[0]
    return score


@since('0.1.0')
def h2o_r2_score(y_actual, y_predict, sample_weight=None, y_type=None):
    """R^2 score for H2O frames. Provides fast computation
    in a distributed fashion without loading all of the data into memory.

    Parameters
    ----------

    y_actual : ``H2OFrame``, shape=(n_samples,)
        The one-dimensional ground truth

    y_predict : ``H2OFrame``, shape=(n_samples,)
        The one-dimensional predicted labels

    sample_weight : H2OFrame or float, optional (default=None)
        A frame of sample weights of matching dims with
        y_actual and y_predict.

    y_type : string, optional (default=None)
        The type of the column. If None, will be determined.

    Returns
    -------

    score : float
        The R^2 score
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
    score = 1 - (numerator / denominator)
    return score


@since('0.1.0')
def h2o_mean_squared_error(y_actual, y_predict, sample_weight=None, y_type=None):
    """Mean squared error score for H2O frames. Provides fast computation
    in a distributed fashion without loading all of the data into memory.

    Parameters
    ----------

    y_actual : ``H2OFrame``, shape=(n_samples,)
        The one-dimensional ground truth

    y_predict : ``H2OFrame``, shape=(n_samples,)
        The one-dimensional predicted labels

    sample_weight : H2OFrame or float, optional (default=None)
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
    diff *= diff  # square it...

    # apply sample weight if necessary
    if sample_weight is not None:
        diff *= sample_weight

    score = flatten_all(diff.mean())[0]
    return score


@since('0.1.0')
def make_h2o_scorer(score_function, y_actual):
    """Make a scoring function from a callable.
    The signature for the callable should resemble:

        ``some_function(y_actual=y_actual, y_predict=y_pred, y_type=None, **kwargs)``

    Parameters
    ----------

    score_function : callable
        The function

    y_actual : ``H2OFrame``, shape=(n_samples,)
        A one-dimensional ``H2OFrame`` (the ground truth). This is
        used to determine before hand whether the type is 
        binary or multiclass.

    Returns
    -------
    score_class : ``_H2OScorer``
        An instance of ``_H2OScorer`` whose ``score`` method
        will be used for scoring in the ``skutil.h2o.grid_search`` 
        module.
    """
    score_class = _H2OScorer(score_function, y_actual)
    return score_class


class _H2OScorer(six.with_metaclass(abc.ABCMeta)):
    """A class that wraps a custom scoring function for use
    with H2OFrames. The first two arguments in the scoring function
    signature should resemble the following: 

        ``some_function(y_true, y_pred, y_type=None...)``

    Any specific scoring kwargs should be passed to the ``score``
    function in the class instance.

    Parameters
    ----------

    score_function : callable
        The function

    y_true : ``H2OFrame``, shape=(n_samples,)
        A one-dimensional ``H2OFrame`` (the ground truth). This is
        used to determine before hand whether the type is 
        binary or multiclass.
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

        return self.fun_(y_actual=y_true, y_predict=y_pred,
                         y_type=self.y_type, **kwargs)
