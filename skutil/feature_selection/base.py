# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import
from abc import ABCMeta, abstractmethod
from sklearn.utils.validation import check_is_fitted
from sklearn.externals import six
from sklearn.base import TransformerMixin
from skutil.base import BaseSkutil
from ..utils import validate_is_pd
import warnings

__all__ = [
    '_BaseFeatureSelector'
]


class _BaseFeatureSelector(six.with_metaclass(ABCMeta, BaseSkutil, TransformerMixin)):
    """The base class for all skutil feature selectors, the _BaseFeatureSelector
    should adhere to the following behavior:

        * The ``fit`` method should only fit the specified columns
          (since it's also a ``SelectiveMixin``), fitting all columns
          only when ``cols`` is None.

        * The ``fit`` method should not change the state of the training frame.

        * The transform method should return a copy of the test frame,
          dropping the columns identified as "bad" in the ``fit`` method.

    Parameters
    ----------

    cols : array_like, shape=(n_features,), optional (default=None)
        The names of the columns on which to apply the transformation.
        If no column names are provided, the transformer will be ``fit``
        on the entire frame. Note that the transformation will also only
        apply to the specified columns, and any other non-specified
        columns will still be present after transformation.

    as_df : bool, optional (default=True)
        Whether to return a Pandas ``DataFrame`` in the ``transform``
        method. If False, will return a Numpy ``ndarray`` instead. 
        Since most skutil transformers depend on explicitly-named
        ``DataFrame`` features, the ``as_df`` parameter is True by default.


    Attributes
    ----------

    drop_ : array_like, shape=(n_features,)
        Assigned after calling ``fit``. These are the features that
        are designated as "bad" and will be dropped in the ``transform``
        method.
    """

    @abstractmethod
    def __init__(self, cols=None, as_df=True):
        super(_BaseFeatureSelector, self).__init__(cols=cols, as_df=as_df)

    def transform(self, X):
        """Transform a test matrix given the already-fit transformer.

        Parameters
        ----------

        X : Pandas ``DataFrame``, shape=(n_samples, n_features)
            The Pandas frame to transform. The prescribed
            ``drop_`` columns will be dropped and a copy of
            ``X`` will be returned.


        Returns
        -------

        dropped : Pandas ``DataFrame`` or np.ndarray, shape=(n_samples, n_features)
            The test data with the prescribed ``drop_`` columns removed.
        """
        check_is_fitted(self, 'drop_')

        # check on state of X and cols
        X, _ = validate_is_pd(X, self.cols)

        if not self.drop_:  # empty or None
            return X if self.as_df else X.as_matrix()
        else:
            # what if we don't want to throw this key error for a non-existent
            # column that we hope to drop anyways? We need to at least inform the
            # user...
            drops = [x for x in self.drop_ if x in X.columns]
            if len(drops) != len(self.drop_):
                warnings.warn('one or more features to drop not contained '
                              'in input data feature names', UserWarning)

            dropped = X.drop(drops, axis=1)
            return dropped if self.as_df else dropped.as_matrix()
