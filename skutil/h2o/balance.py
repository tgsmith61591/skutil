from __future__ import absolute_import, division, print_function
import abc
import numpy as np
import pandas as pd

from sklearn.externals import six
from h2o.frame import H2OFrame

from .base import _check_is_frame, BaseH2OFunctionWrapper, _frame_from_x_y
from ..preprocessing import BalancerMixin
from ..preprocessing.balance import (_validate_ratio, _validate_target, 
    _validate_num_classes, _OversamplingBalancePartitioner,
    _UndersamplingBalancePartitioner)
from ..base import overrides


__all__ = [
    'H2OOversamplingClassBalancer',
    'H2OUndersamplingClassBalancer'
]


def _validate_x_y_ratio(X, y, ratio):
    """Validates the following, given that X is
    already a validated H2OFrame:

    1. That y is a string
    2. That the number of classes does not exceed _max_classes
       as defined by the BalancerMixin class
    3. That the number of classes is at least 2
    4. That ratio is a float that falls between 0.0 (exclusive) and
       1.0 (inclusive)

    Return
    ------
    (cts, n_classes), a tuple with the sorted class value_counts and the number of classes
    """
    # validate ratio, if the current ratio is >= the ratio, it's "balanced enough"
    ratio = _validate_ratio(ratio)
    y = _validate_target(y) # cast to string type

    # generate cts. Have to get kludgier in h2o...
    unq_vals = X[y].unique()
    unq_vals = unq_vals.as_data_frame(use_pandas=True)[unq_vals.columns[0]].values # numpy array of unique vals
    unq_cts = dict([(val, X[y][X[y]==val].shape[0]) for val in unq_vals])

    # validate is < max classes
    cts = pd.Series(unq_cts).sort_values()
    n_classes = _validate_num_classes(cts)

    return cts, n_classes


class _BaseH2OBalancer(BaseH2OFunctionWrapper, BalancerMixin):
    """Base class for all H2O balancers. Provides _min_version
    and _max_version for BaseH2OFunctionWrapper constructor.
    """

    def __init__(self, target_feature, ratio=BalancerMixin._def_ratio, min_version='any', max_version=None):
        super(_BaseH2OBalancer, self).__init__(target_feature=target_feature, 
                                               min_version=min_version,
                                               max_version=max_version)
        self.ratio = ratio


class H2OOversamplingClassBalancer(_BaseH2OBalancer):
    """Oversample the minority classes until they are represented
    at the target proportion to the majority class.

    Parameters
    ----------
    target_feature : str
        The name of the response column. The response column must be
        biclass, no more or less.

    ratio : float, def 0.2
        The target ratio of the minority records to the majority records. If the
        existing ratio is >= the provided ratio, the return value will merely be
        a copy of the input frame
    """

    def __init__(self, target_feature, ratio=BalancerMixin._def_ratio):
        # as of now, no min/max version; it's simply uncompatible...
        super(H2OOversamplingClassBalancer, self).__init__(target_feature, ratio)

    @overrides(BalancerMixin)
    def balance(self, X):
        """Apply the oversampling balance operation. Oversamples
        the minority class to the provided ratio of minority
        class(es) : majority class.
        
        Parameters
        ----------
        X : H2OFrame, shape [n_samples, n_features]
            The data to balance
        """

        # check on state of X
        frame = _check_is_frame(X)

        # get the partitioner
        partitioner = _OversamplingBalancePartitioner(frame, 
            self.target_feature, self.ratio, _validate_x_y_ratio)
        sample_idcs = partitioner.get_indices()

        # since H2O won't allow us to resample (it's considered rearranging)
        # we need to rbind at each point of duplication... this can be pretty
        # inefficient, so we might need to get clever about this...
        new_frame = None

        for i in sample_idcs:
            row = frame[i, :]
            if new_frame is None:
                new_frame = row
            else:
                new_frame = new_frame.rbind(row)

        return new_frame


class H2OUndersamplingClassBalancer(_BaseH2OBalancer):
    """Undersample the majority class until it is represented
    at the target proportion to the most-represented minority class.
    For example, given the follow pd.Series (index = class, and values = counts):

    0  150
    1  30
    2  10

    and the ratio 0.5, the majority class (0) will be undersampled until
    the second most-populous class (1) is represented at a ratio of 0.5:

    0  60
    1  30
    2  10

    Parameters
    ----------
    target_feature : str
        The name of the response column. The response column must be
        biclass, no more or less.

    ratio : float, def 0.2
        The target ratio of the minority records to the majority records. If the
        existing ratio is >= the provided ratio, the return value will merely be
        a copy of the input frame
    """

    _min_version = '3.8.2.9'
    _max_version = None

    def __init__(self, target_feature, ratio=BalancerMixin._def_ratio):
        super(H2OUndersamplingClassBalancer, self).__init__(
            target_feature, ratio, self._min_version, self._max_version)

    @overrides(BalancerMixin)
    def balance(self, X):
        """Apply the undersampling balance operation. Undersamples
        the majority class to the provided ratio of minority
        class(es) : majority class
        
        Parameters
        ----------
        X : H2OFrame, shape [n_samples, n_features]
            The data to balance
        """

        # check on state of X
        frame = _check_is_frame(X)

        # get the partitioner
        partitioner = _UndersamplingBalancePartitioner(frame, 
            self.target_feature, self.ratio, _validate_x_y_ratio)

        # since there are no feature_names, we can just slice
        # the h2o frame as is, given the indices:
        return frame[partitioner.get_indices(), :]

