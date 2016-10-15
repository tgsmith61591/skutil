from __future__ import print_function, division, absolute_import

from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.utils.validation import check_is_fitted
from sklearn.externals import six

from skutil.base import *
from skutil.base import overrides
from ..utils import *

__all__ = [
    'SelectivePCA',
    'SelectiveTruncatedSVD'
]


class _BaseSelectiveDecomposer(six.with_metaclass(ABCMeta, BaseEstimator, 
                                                  TransformerMixin, 
                                                  SelectiveMixin)):
    """Base class for selective decompositional transformers.
    Each of these transformers should adhere to the :class:`skutil.base.SelectiveMixin`
    standard of accepting a ``cols`` parameter in the ``__init__`` method, and
    only applying the transformation to the defined columns, if any.

    Parameters
    ----------

    cols : array_like (string), optional (default=None)
        The names of the columns on which to apply the transformation.
        If no column names are provided, the decomposition will be ``fit``
        on the entire frame. Note that the transormation will also only
        apply to the specified columns, and any other non-specified
        columns will still be present after transformation.

    n_components : int, float, None or string, optional (default=None)
        ``n_components`` is specific to the type of transformation
        being fit, and determines the number of components to extract
        in the transformation.

    as_df : bool, optional (default=None)
        Whether or not to return a pandas DataFrame object. If
        False, will return a np.ndarray instead.
    """

    def __init__(self, cols=None, n_components=None, as_df=True):
        self.cols = cols
        self.n_components = n_components
        self.as_df = as_df

    @abstractmethod
    def get_decomposition(self):
        """This method needs to be overridden by subclasses.
        It is intended to act as a property to return the specific
        decomposition. For `SelectivePCA`, this will return the `pca_`
        attribute; for `SelectiveTruncatedSVD`, this will return the
        `svd_` attribute.
        """
        raise NotImplementedError('this should be implemented by a subclass')


class SelectivePCA(_BaseSelectiveDecomposer):
    """A class that will apply PCA only to a select group
    of columns. Useful for data that may contain a mix of columns 
    that we do and don't want to decompose.

    Parameters
    ----------

    cols : array_like (string), optional (default=None)
        The names of the columns on which to apply the transformation.
        If no column names are provided, the decomposition will be ``fit``
        on the entire frame. Note that the transormation will also only
        apply to the specified columns, and any other non-specified
        columns will still be present after transformation.

    n_components : int, float, None or string, optional (default=None)
        The number of components to keep, per sklearn:

        * if n_components is not set, all components are kept:

            n_components == min(n_samples, n_features)

        * if n_components == 'mle' and svd_solver == 'full', Minka's MLE is used
        to guess the dimension.

        * if ``0 < n_components < 1`` and svd_solver == 'full', select the number
        of components such that the amount of variance that needs to be
        explained is greater than the percentage specified by ``n_components``

        * ``n_components`` cannot be equal to ``n_features`` for ``svd_solver`` == 'arpack'.

    as_df : bool, optional (default=None)
        Whether or not to return a pandas DataFrame object. If
        False, will return a np.ndarray instead.

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
    ----------

    cols : array_like (string)
        The names of the columns on which to apply 
        the transformation.

    pca_ : the PCA object
    """

    def __init__(self, cols=None, n_components=None, whiten=False, weight=False, as_df=True):
        super(SelectivePCA, self).__init__(cols=cols, n_components=n_components, as_df=as_df)
        self.whiten = whiten
        self.weight = weight

    def fit(self, X, y=None):
        """Fit the transformer to the provided dataset.

        Parameters
        ----------

        X: pd.DataFrame, shape(n_samples, n_features)
            The data to fit.

        y: None
            Pass through for grid search and pipeline.

        Returns
        -------

        self : SelectivePCA
            The fit transformer
        """
        # check on state of X and cols
        X, self.cols = validate_is_pd(X, self.cols)
        cols = X.columns if not self.cols else self.cols

        # fails thru if names don't exist:
        self.pca_ = PCA(
            n_components=self.n_components,
            whiten=self.whiten).fit(X[cols])

        return self

    def transform(self, X, y=None):
        """Transform the given dataset, provided the transformer
        has already been fit.

        Parameters
        ----------

        X: pd.DataFrame, shape(n_samples, n_features)
            The data to fit.

        y: None
            Pass through for grid search and pipeline.

        Returns
        -------

        x : pd.DataFrame or np.ndarray
            The transformed matrix. pd.DataFrame if ``as_df``
            is True, else np.ndarray.
        """
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

        left = pd.DataFrame.from_records(data=transform,
                                         columns=[('PC%i' % (i + 1)) for i in range(transform.shape[1])])

        # concat if needed
        x = pd.concat([left, X[other_nms]], axis=1) if other_nms else left
        return x if self.as_df else x.as_matrix()

    @overrides(_BaseSelectiveDecomposer)
    def get_decomposition(self):
        """Overridden from the :class:``_BaseSelectiveDecomposer`` class,
        this method returns the internal decomposition class: 
        ``sklearn.decomposition.PCA``

        Returns
        -------
        self.pca_ : sklearn.decomposition.PCA
            The fit internal decomposition class
        """
        return self.pca_ if hasattr(self, 'pca_') else None

    def score(self, X, y=None):
        """Return the average log-likelihood of all samples.
        This calls sklearn.decomposition.PCA's score method
        on the specified columns.

        See. "Pattern Recognition and Machine Learning"
        by C. Bishop, 12.2.1 p. 574
        or http://www.miketipping.com/papers/met-mppca.pdf

        Parameters
        ----------

        X: pd.DataFrame, shape(n_samples, n_features)
            The data to score.

        y: None
            Passthrough for pipeline/gridsearch

        Returns
        -------

        ll: float
            Average log-likelihood of the samples under the fit
            PCA model (`self.pca_`)
        """
        check_is_fitted(self, 'pca_')
        X, _ = validate_is_pd(X, self.cols)
        cols = X.columns if not self.cols else self.cols

        ll = self.pca_.score(X[cols], y)
        return ll


###############################################################################
class SelectiveTruncatedSVD(_BaseSelectiveDecomposer):
    """A class that will apply truncated SVD (LSA) only to a select group
    of columns. Useful for data that contains categorical features
    that have not yet been dummied, or for dummied features we don't want
    decomposed. TruncatedSVD is the equivalent of Latent Semantic Analysis,
    and returns the "concept space" of the decomposed features.

    Parameters
    ----------

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
    ----------

    cols : array_like (string)
        the columns

    svd_ : the SVD object
    """

    def __init__(self, cols=None, n_components=2, algorithm='randomized', n_iter=5, as_df=True):
        super(SelectiveTruncatedSVD, self).__init__(cols=cols, n_components=n_components, as_df=as_df)
        self.algorithm = algorithm
        self.n_iter = n_iter

    def fit(self, X, y=None):
        """Fit the transformer to the provided dataset.

        Parameters
        ----------

        X: pd.DataFrame, shape(n_samples, n_features)
            The data to fit.

        y: None
            Pass through for grid search and pipeline.

        Returns
        -------

        self
        """
        # check on state of X and cols
        X, self.cols = validate_is_pd(X, self.cols)

        # fails thru if names don't exist:
        self.svd_ = TruncatedSVD(
            n_components=self.n_components,
            algorithm=self.algorithm,
            n_iter=self.n_iter).fit(X[self.cols or X.columns])

        return self

    def transform(self, X, y=None):
        """Transform the given dataset, provided the transformer
        has already been fit.

        Parameters
        ----------

        X: pd.DataFrame, shape(n_samples, n_features)
            The data to fit.

        y: None
            Pass through for grid search and pipeline.

        Returns
        -------

        x : pd.DataFrame or np.ndarray
            The transformed matrix. pd.DataFrame if ``as_df``
            is True, else np.ndarray.
        """
        check_is_fitted(self, 'svd_')
        # check on state of X and cols
        X, _ = validate_is_pd(X, self.cols)
        cols = X.columns if not self.cols else self.cols

        other_nms = [nm for nm in X.columns if not nm in cols]
        transform = self.svd_.transform(X[cols])
        left = pd.DataFrame.from_records(data=transform,
                                         columns=[('Concept%i' % (i + 1)) for i in range(transform.shape[1])])

        # concat if needed
        x = pd.concat([left, X[other_nms]], axis=1) if other_nms else left

        return x if self.as_df else x.as_matrix()

    @overrides(_BaseSelectiveDecomposer)
    def get_decomposition(self):
        """Overridden from the :class:``_BaseSelectiveDecomposer`` class,
        this method returns the internal decomposition class: 
        ``sklearn.decomposition.TruncatedSVD``

        Returns
        -------
        self.svd_ : sklearn.decomposition.TruncatedSVD
            The fit internal decomposition class
        """
        return self.svd_ if hasattr(self, 'svd_') else None
