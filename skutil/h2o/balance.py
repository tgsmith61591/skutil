from __future__ import absolute_import, division, print_function
import pandas as pd
from abc import ABCMeta
from sklearn.externals import six
from skutil.base import overrides
from .util import reorder_h2o_frame
from .base import check_frame, BaseH2OFunctionWrapper
from ..preprocessing.balance import (_validate_ratio, _validate_target, _validate_num_classes,
                                     _OversamplingBalancePartitioner, _UndersamplingBalancePartitioner,
                                     BalancerMixin, SamplingWarning)

__all__ = [
    'H2OOversamplingClassBalancer',
    'H2OUndersamplingClassBalancer'
]


def _validate_x_y_ratio(X, y, ratio):
    """Validates the following, given that X is
    already a validated pandas DataFrame:

    1. That y is a string
    2. That the number of classes does not exceed _max_classes
       as defined by the BalancerMixin class
    3. That the number of classes is at least 2
    4. That ratio is a float that falls between 0.0 (exclusive) and
       1.0 (inclusive)

    Parameters
    ----------

    X : ``H2OFrame``, shape=(n_samples, n_features)
        The frame from which to sample

    y : str
        The name of the column that is the response class.
        This is the column on which ``value_counts`` will be
        executed to determine imbalance.

    ratio : float
        The ratio at which the balancing operation will 
        be performed. Used to determine whether balancing is
        required.

    Returns
    -------

    out_tup : tuple, shape=(3,)
        a length-3 tuple with the following args:
            [0] - cts (pd.Series), the ascending sorted ``value_counts`` 
                  of the class, where the index is the class label.
            [1] - n_classes (int), the number of unique classes
            [2] - needs_balancing (bool), whether the least populated class
                  is represented at a rate lower than the demanded ratio.
    """
    # validate ratio, if the current ratio is >= the ratio, it's "balanced enough"
    ratio = _validate_ratio(ratio)
    y = _validate_target(y)  # cast to string type

    # generate cts. Have to get kludgier in h2o...
    unq_vals = X[y].unique()
    unq_vals = unq_vals.as_data_frame(use_pandas=True)[unq_vals.columns[0]].values  # numpy array of unique vals
    unq_cts = dict([(val, X[y][X[y] == val].shape[0]) for val in unq_vals])

    # validate is < max classes
    cts = pd.Series(unq_cts).sort_values(ascending=True)
    n_classes = _validate_num_classes(cts)
    needs_balancing = (cts.values[0] / cts.values[-1]) < ratio

    out_tup = (cts, n_classes, needs_balancing)
    return out_tup


class _BaseH2OBalancer(six.with_metaclass(ABCMeta, 
                                          BaseH2OFunctionWrapper, 
                                          BalancerMixin)):
    """Base class for all H2O balancers. Provides _min_version
    and _max_version for BaseH2OFunctionWrapper constructor.
    """

    def __init__(self, target_feature, ratio=BalancerMixin._def_ratio, 
                 min_version='any', max_version=None, shuffle=True):
        super(_BaseH2OBalancer, self).__init__(target_feature=target_feature,
                                               min_version=min_version,
                                               max_version=max_version)
        self.ratio = ratio
        self.shuffle = shuffle


class H2OOversamplingClassBalancer(_BaseH2OBalancer):
    """Oversample the minority classes until they are represented
    at the target proportion to the majority class.

    Parameters
    ----------

    target_feature : str
        The name of the response column. The response column must be
        more than a single class and less than 
        ``skutil.preprocessing.balance.BalancerMixin._max_classes``

    ratio : float, optional (default=0.2)
        The target ratio of the minority records to the majority records. If the
        existing ratio is >= the provided ratio, the return value will merely be
        a copy of the input frame

    shuffle : bool, optional (default=True)
        Whether or not to shuffle rows on return


    Examples
    --------

    Consider the following example: with a ``ratio`` of 0.5, the 
    minority classes (1, 2) will be oversampled until they are represented 
    at a ratio of at least 0.5 * the prevalence of the majority class (0)

        >>> def example():
        ...     import h2o
        ...     import pandas as pd
        ...     import numpy as np
        ...     from skutil.h2o.frame import value_counts
        ...     from skutil.h2o import from_pandas
        ...     
        ...     # initialize h2o
        ...     h2o.init()
        ...
        ...     # read into pandas
        ...     x = pd.DataFrame(np.concatenate([np.zeros(100), np.ones(30), np.ones(25)*2]), columns=['A'])
        ...     
        ...     # load into h2o
        ...     X = from_pandas(x)
        ...     
        ...     # initialize sampler
        ...     sampler = H2OOversamplingClassBalancer(target_feature="A", ratio=0.5)
        ...     
        ...     # do balancing
        ...     X_balanced = sampler.balance(X)
        ...     value_counts(X_balanced)
        >>>
        >>> example() # doctest: +SKIP
        0    100
        1     50
        2     50
        Name A, dtype: int64
        
    """

    def __init__(self, target_feature, ratio=BalancerMixin._def_ratio, shuffle=True):
        # as of now, no min/max version; it's simply compatible with all...
        super(H2OOversamplingClassBalancer, self).__init__(
            target_feature=target_feature, ratio=ratio, shuffle=shuffle)

    @overrides(BalancerMixin)
    def balance(self, X):
        """Apply the oversampling balance operation. Oversamples
        the minority class to the provided ratio of minority
        class(es) : majority class.
        
        Parameters
        ----------

        X : ``H2OFrame``, shape=(n_samples, n_features)
            The imbalanced dataset.

        Returns
        -------

        Xb : ``H2OFrame``, shape=(n_samples, n_features)
            The balanced H2OFrame
        """
        # check on state of X
        frame = check_frame(X, copy=False)

        # get the partitioner
        partitioner = _OversamplingBalancePartitioner(
            X=frame, y_name=self.target_feature, 
            ratio=self.ratio, validation_function=_validate_x_y_ratio)
        sample_idcs = partitioner.get_indices(self.shuffle)

        # since H2O won't allow us to resample (it's considered rearranging)
        # we need to rbind at each point of duplication... this can be pretty
        # inefficient, so we might need to get clever about this...
        Xb = reorder_h2o_frame(frame, sample_idcs)
        return Xb


class H2OUndersamplingClassBalancer(_BaseH2OBalancer):
    """Undersample the majority class until it is represented
    at the target proportion to the most-represented minority class.

    Parameters
    ----------

    target_feature : str
        The name of the response column. The response column must be
        more than a single class and less than 
        ``skutil.preprocessing.balance.BalancerMixin._max_classes``

    ratio : float, optional (default=0.2)
        The target ratio of the minority records to the majority records. If the
        existing ratio is >= the provided ratio, the return value will merely be
        a copy of the input frame

    shuffle : bool, optional (default=True)
        Whether or not to shuffle rows on return


    Examples
    --------
    
    Consider the following example: with a ``ratio`` of 0.5, the 
    majority class (0) will be undersampled until the second most-populous 
    class (1) is represented at a ratio of 0.5.

        >>> def example():
        ...     import h2o
        ...     import pandas as pd
        ...     import numpy as np
        ...     from skutil.h2o.frame import value_counts
        ...     from skutil.h2o import from_pandas
        ...
        ...     # initialize h2o
        ...     h2o.init()
        ...     
        ...     # read into pandas
        ...     x = pd.DataFrame(np.concatenate([np.zeros(100), np.ones(30), np.ones(25)*2]), columns=['A'])
        ...     
        ...     # load into h2o
        ...     X = from_pandas(x) # doctest:+ELLIPSIS
        ...     
        ...     # initialize sampler
        ...     sampler = H2OUndersamplingClassBalancer(target_feature="A", ratio=0.5)
        ...     
        ...     X_balanced = sampler.balance(X)
        ...     value_counts(X_balanced)
        ...
        >>> example() # doctest: +SKIP
        0    60
        1    30
        2    10
        Name A, dtype: int64

    """

    _min_version = '3.8.2.9'
    _max_version = None

    def __init__(self, target_feature, ratio=BalancerMixin._def_ratio, shuffle=True):
        super(H2OUndersamplingClassBalancer, self).__init__(
            target_feature=target_feature, ratio=ratio, min_version=self._min_version, 
            max_version=self._max_version, shuffle=shuffle)

    @overrides(BalancerMixin)
    def balance(self, X):
        """Apply the undersampling balance operation. Undersamples
        the majority class to the provided ratio of minority
        class(es) : majority class
        
        Parameters
        ----------

        X : ``H2OFrame``, shape=(n_samples, n_features)
            The imbalanced dataset.


        Returns
        -------

        Xb : ``H2OFrame``, shape=(n_samples, n_features)
            The balanced H2OFrame
        """

        # check on state of X
        frame = check_frame(X, copy=False)

        # get the partitioner
        partitioner = _UndersamplingBalancePartitioner(
            X=frame, y_name=self.target_feature, ratio=self.ratio, 
            validation_function=_validate_x_y_ratio)

        # since there are no feature_names, we can just slice
        # the h2o frame as is, given the indices:
        idcs = partitioner.get_indices(self.shuffle)
        Xb = frame[idcs, :] if not self.shuffle else reorder_h2o_frame(frame, idcs)
        return Xb
