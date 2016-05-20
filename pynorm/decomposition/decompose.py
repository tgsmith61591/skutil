import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_is_fitted
from ..utils import *


__all__ = [
    'SelectivePCA'
]



###############################################################################
class SelectivePCA(BaseEstimator, TransformerMixin):
    """A class that will apply PCA only to a select group
    of columns. Useful for data that contains categorical features
    that have not yet been dummied, for dummied features that we
    may not want to scale, or for any already-in-scale features.

    Parameters
    ----------
    cols : array_like (string)
        names of columns on which to apply scaling

    n_components : int, float, None or string
        Number of components to keep.
        if n_components is not set all components are kept:

            n_components == min(n_samples, n_features)

        if n_components == 'mle' and svd_solver == 'full', Minka\'s MLE is used
        to guess the dimension
        if ``0 < n_components < 1`` and svd_solver == 'full', select the number
        of components such that the amount of variance that needs to be
        explained is greater than the percentage specified by n_components
        n_components cannot be equal to n_features for svd_solver == 'arpack'.

    whiten : bool, optional (default False)
        When True (False by default) the `components_` vectors are multiplied
        by the square root of n_samples and then divided by the singular values
        to ensure uncorrelated outputs with unit component-wise variances.
        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometime
        improve the predictive accuracy of the downstream estimators by
        making their data respect some hard-wired assumptions.


    Attributes
    ----------
    cols_ : array_like (string)
        the columns

    pca_ : the PCA object
    """

    def __init__(self, cols=None, n_components=None, whiten=False):
        self.cols_ = cols
        self.n_components = n_components
        self.whiten = whiten

    def fit(self, X, y = None):
        validate_is_pd(X)

        ## If cols is None, then apply to all by default
        if not self.cols_:
            self.cols_ = X.columns

        ## fails thru if names don't exist:
        self.pca_ = PCA(
            n_components=self.n_components,
            whiten=self.whiten).fit(X[self.cols_])

        return self

    def transform(self, X, y = None):
        check_is_fitted(self, 'pca_')
        validate_is_pd(X)

        X = X.copy()
        other_nms = [nm for nm in X.columns if not nm in self.cols_]

        ## don't check fit, does internally in PCA object
        transform = self.pca_.transform(X[self.cols_])
        left = pd.DataFrame.from_records(data=transform, columns=[('PC%i'%(i+1)) for i in range(transform.shape[1])])

        return pd.concat([left, X[other_nms]], axis=1)

