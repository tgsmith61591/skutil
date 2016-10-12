from __future__ import print_function

import warnings
from abc import ABCMeta, abstractmethod

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from skutil.base import SelectiveMixin
from ..utils import validate_is_pd

__all__ = [
    '_BaseFeatureSelector'  
]

###############################################################################
class _BaseFeatureSelector(BaseEstimator, TransformerMixin, SelectiveMixin):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, cols=None, as_df=True, **kwargs):
        self.cols = cols
        self.as_df = as_df
        self.drop = None

    def transform(self, X, y=None):
        check_is_fitted(self, 'drop')
        # check on state of X and cols
        X, _ = validate_is_pd(X, self.cols)

        if self.drop is None:
            return X if self.as_df else X.as_matrix()
        else:
            # what if we don't want to throw this key error for a non-existent
            # column that we hope to drop anyways? We need to at least inform the
            # user...
            drops = [x for x in self.drop if x in X.columns]
            if not len(drops) == len(self.drop):
                warnings.warn('one of more features to drop not contained in input data feature names')

            dropped = X.drop(drops, axis=1)
            return dropped if self.as_df else dropped.as_matrix()