from __future__ import print_function
import warnings
from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.externals import six
from skutil.base import SelectiveMixin
from ..utils import validate_is_pd

__all__ = [
    '_BaseFeatureSelector'
]


class _BaseFeatureSelector(six.with_metaclass(ABCMeta, BaseEstimator, 
                                              TransformerMixin, 
                                              SelectiveMixin)):
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

    cols : array_like, optional (default=None)
        The columns on which the transformer will be ``fit``. In
        the case that ``cols`` is None, the transformer will be fit
        on all columns.

    as_df : bool, optional (default=True)
        Whether to return a Pandas DataFrame in the ``transform``
        method. If False, will return a NumPy ndarray instead. 
        Since most skutil transformers depend on explicitly-named
        DataFrame features, the ``as_df`` parameter is True by default.

    Attributes
    ----------

    drop_ : array_like, shape=(n_features,)
        Assigned after calling ``fit``. These are the features that
        are designated as "bad" and will be dropped in the ``transform``
        method.
    """

    @abstractmethod
    def __init__(self, cols=None, as_df=True, **kwargs):
        self.cols = cols
        self.as_df = as_df

    def transform(self, X, y=None):
        check_is_fitted(self, 'drop_')
        # check on state of X and cols
        X, _ = validate_is_pd(X, self.cols)

        if self.drop_ is None:
            return X if self.as_df else X.as_matrix()
        else:
            # what if we don't want to throw this key error for a non-existent
            # column that we hope to drop anyways? We need to at least inform the
            # user...
            drops = [x for x in self.drop_ if x in X.columns]
            if not len(drops) == len(self.drop_):
                warnings.warn('one of more features to drop not contained '
                              'in input data feature names')

            dropped = X.drop(drops, axis=1)
            return dropped if self.as_df else dropped.as_matrix()
