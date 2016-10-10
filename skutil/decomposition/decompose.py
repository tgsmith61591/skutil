from __future__ import print_function, division, absolute_import
import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.utils.validation import check_is_fitted

from ..utils import *
from ..base import *
from ..base import overrides



__all__ = [
    'SelectivePCA',
    'SelectiveTruncatedSVD'
]


###############################################################################
class _BaseSelectiveDecomposer(BaseEstimator, TransformerMixin, SelectiveMixin):
    """Base class for selective decompositional transformers."""
    __metaclass__ = ABCMeta

    def __init__(self, cols=None, n_components=None, as_df=True):
        self.cols = cols
        self.n_components = n_components
        self.as_df = as_df

    @abstractmethod
    def get_decomposition(self):
        """This needs to be overridden by subclasses.
        As of now, it will just raise a NotImplementedError
        """
        raise NotImplementedError('this should be implemented by a subclass')



###############################################################################
class SelectivePCA(_BaseSelectiveDecomposer):
    """A class that will apply PCA only to a select group
    of columns. Useful for data that contains categorical features
    that have not yet been dummied, or for dummied features we don't want
    decomposed.

    Parameters

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

    as_df : boolean, default True
        Whether to return a pandas DataFrame

    whiten : bool, optional (default False)
        When True (False by default) the `components_` vectors are multiplied
        by the square root of n_samples and then divided by the singular values
        to ensure uncorrelated outputs with unit component-wise variances.
        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometime
        improve the predictive accuracy of the downstream estimators by
        making their data respect some hard-wired assumptions.

    weight : bool, optional (default False)
        When True (False by default) the `explained_variance_` vector is used to weight
        the features post-transformation. This is especially useful in clustering contexts,
        where features are all implicitly assigned the same importance, even though PCA
        by nature orders the features by importance (i.e., not all components are created equally).
        When True, weighting will subtract the median variance from the weighting vector, and add one
        (so as not to down sample or upsample everything), then multiply the weights across the
        transformed features.


    Attributes

    cols : array_like (string)
        the columns

    pca_ : the PCA object
    """

    def __init__(self, cols=None, n_components=None, whiten=False, weight=False, as_df=True):
        super(SelectivePCA, self).__init__(cols=cols, n_components=n_components, as_df=as_df)
        self.whiten = whiten
        self.weight = weight

    def fit(self, X, y = None):
        # check on state of X and cols
        X, self.cols = validate_is_pd(X, self.cols)
        cols = X.columns if not self.cols else self.cols

        ## fails thru if names don't exist:
        self.pca_ = PCA(
            n_components=self.n_components,
            whiten=self.whiten).fit(X[cols])

        return self

    def transform(self, X, y = None):
        check_is_fitted(self, 'pca_')
        # check on state of X and cols
        X, _ = validate_is_pd(X, self.cols)
        cols = X.columns if not self.cols else self.cols

        other_nms = [nm for nm in X.columns if not nm in cols]
        transform = self.pca_.transform(X[cols])

        # do weighting if necessary
        if self.weight:
            # get the weight vals
            weights = self.pca_.explained_variance_ratio_
            weights -= np.median(weights)
            weights += 1

            # now add to the transformed features
            transform *= weights

        left = pd.DataFrame.from_records(data=transform, columns=[('PC%i'%(i+1)) for i in range(transform.shape[1])])

        # concat if needed
        x = pd.concat([left, X[other_nms]], axis=1) if other_nms else left
        return x if self.as_df else x.as_matrix()

    @overrides(_BaseSelectiveDecomposer)
    def get_decomposition(self):
        return self.pca_ if hasattr(self, 'pca_') else None


###############################################################################
class SelectiveTruncatedSVD(_BaseSelectiveDecomposer):
    """A class that will apply truncated SVD (LSA) only to a select group
    of columns. Useful for data that contains categorical features
    that have not yet been dummied, or for dummied features we don't want
    decomposed. TruncatedSVD is the equivalent of Latent Semantic Analysis,
    and returns the "concept space" of the decomposed features.

    Parameters

    cols : array_like (string)
        names of columns on which to apply scaling

    n_components : int, default = 2
        Desired dimensionality of output data.
        Must be strictly less than the number of features.
        The default value is useful for visualisation. For LSA, a value of
        100 is recommended.

    algorithm : string, default = "randomized"
        SVD solver to use. Either "arpack" for the ARPACK wrapper in SciPy
        (scipy.sparse.linalg.svds), or "randomized" for the randomized
        algorithm due to Halko (2009).

    n_iter : int, optional (default 5)
        Number of iterations for randomized SVD solver. Not used by ARPACK.
        The default is larger than the default in `randomized_svd` to handle
        sparse matrices that may have large slowly decaying spectrum.

    as_df : boolean, default True
        Whether to return a pandas DataFrame


    Attributes

    cols : array_like (string)
        the columns

    svd_ : the SVD object
    """

    def __init__(self, cols=None, n_components=2, algorithm='randomized', n_iter=5, as_df=True):
        super(SelectiveTruncatedSVD, self).__init__(cols=cols, n_components=n_components, as_df=as_df)
        self.algorithm = algorithm
        self.n_iter = n_iter

    def fit(self, X, y = None):
        # check on state of X and cols
        X, self.cols = validate_is_pd(X, self.cols)

        ## fails thru if names don't exist:
        self.svd_ = TruncatedSVD(
            n_components=self.n_components,
            algorithm=self.algorithm,
            n_iter=self.n_iter).fit(X[self.cols or X.columns])

        return self

    def transform(self, X, y = None):
        check_is_fitted(self, 'svd_')
        # check on state of X and cols
        X, _ = validate_is_pd(X, self.cols)
        cols = X.columns if not self.cols else self.cols

        other_nms = [nm for nm in X.columns if not nm in cols]
        transform = self.svd_.transform(X[cols])
        left = pd.DataFrame.from_records(data=transform, columns=[('Concept%i'%(i+1)) for i in range(transform.shape[1])])

        # concat if needed
        x = pd.concat([left, X[other_nms]], axis=1) if other_nms else left

        return x if self.as_df else x.as_matrix()

    @overrides(_BaseSelectiveDecomposer)
    def get_decomposition(self):
        return self.svd_ if hasattr(self, 'svd_') else None
