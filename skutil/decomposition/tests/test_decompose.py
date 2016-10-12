import numpy as np
from numpy.testing import (assert_array_equal, assert_array_almost_equal)
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.datasets import load_iris
from skutil.decomposition import *
from skutil.utils import assert_fails
from skutil.utils import load_iris_df
from skutil.decomposition.decompose import _BaseSelectiveDecomposer

# Def data for testing
iris = load_iris()
X = load_iris_df(False)


def test_selective_pca():
    original = X
    cols = [original.columns[0]]  # Only perform on first...
    compare_cols = np.array(original[['sepal width (cm)', 'petal length (cm)',
                                      'petal width (cm)']].as_matrix())  # should be the same as the trans cols

    transformer = SelectivePCA(cols=cols, n_components=0.85).fit(original)
    transformed = transformer.transform(original)

    untouched_cols = np.array(transformed[['sepal width (cm)', 'petal length (cm)', 'petal width (cm)']].as_matrix())
    assert_array_almost_equal(compare_cols, untouched_cols)
    assert 'PC1' in transformed.columns
    assert transformed.shape[1] == 4
    assert isinstance(transformer.get_decomposition(), PCA)
    assert SelectivePCA().get_decomposition() is None

    # test the selective mixin
    assert isinstance(transformer.get_features(), list)
    transformer.set_features(cols=None)
    assert transformer.get_features() is None

    # what if we want to weight it?
    pca_df = SelectivePCA(weight=True, n_components=0.99, as_df=False).fit_transform(original)
    pca_arr = SelectivePCA(weight=True, n_components=0.99, as_df=False).fit_transform(iris.data)
    assert_array_equal(pca_df, pca_arr)

    # hack to assert they are not equal if weighted
    pca_arr = SelectivePCA(weight=False, n_components=0.99, as_df=False).fit_transform(iris.data)
    assert_fails(assert_array_equal, AssertionError, pca_df, pca_arr)


def test_selective_tsvd():
    original = X
    cols = [original.columns[0], original.columns[1]]  # Only perform on first two columns...
    compare_cols = np.array(
        original[['petal length (cm)', 'petal width (cm)']].as_matrix())  # should be the same as the trans cols

    transformer = SelectiveTruncatedSVD(cols=cols, n_components=1).fit(original)
    transformed = transformer.transform(original)

    untouched_cols = np.array(transformed[['petal length (cm)', 'petal width (cm)']].as_matrix())
    assert_array_almost_equal(compare_cols, untouched_cols)
    assert 'Concept1' in transformed.columns
    assert transformed.shape[1] == 3
    assert isinstance(transformer.get_decomposition(), TruncatedSVD)
    assert SelectiveTruncatedSVD().get_decomposition() is None  # default None

    # test the selective mixin
    assert isinstance(transformer.get_features(), list)
    transformer.set_features(cols=None)
    assert transformer.get_features() is None


def test_not_implemented_failure():
    # define anon decomposer
    class AnonDecomposer(_BaseSelectiveDecomposer):
        def __init__(self, cols=None, n_components=None, as_df=True):
            super(AnonDecomposer, self).__init__(cols, n_components, as_df)

        def get_decomposition(self):
            return super(AnonDecomposer, self).get_decomposition()

    assert_fails(AnonDecomposer().get_decomposition, NotImplementedError)
