import numpy as np
import pandas as pd
from numpy.testing import (assert_array_equal, assert_almost_equal, assert_array_almost_equal)
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA, TruncatedSVD
from skutil.decomposition import *



## Def data for testing
iris = load_iris()
X = pd.DataFrame(data=iris.data, columns=iris.feature_names)

def test_selective_pca():
	original = X
	cols = [original.columns[0]] ## Only perform on first...
	compare_cols = np.array(original[['sepal width (cm)','petal length (cm)','petal width (cm)']].as_matrix()) ## should be the same as the trans cols

	transformer = SelectivePCA(cols=cols, n_components=0.85).fit(original)
	transformed = transformer.transform(original)

	untouched_cols = np.array(transformed[['sepal width (cm)','petal length (cm)','petal width (cm)']].as_matrix())
	assert_array_almost_equal(compare_cols, untouched_cols)
	assert 'PC1' in transformed.columns
	assert transformed.shape[1] == 4
	assert isinstance(transformer.get_decomposition(), PCA)

	# test the selective mixin
	assert isinstance(transformer.get_features(), list)
	transformer.set_features(cols=None)
	assert transformer.get_features() is None


def test_selective_tsvd():
	original = X
	cols = [original.columns[0], original.columns[1]] ## Only perform on first two columns...
	compare_cols = np.array(original[['petal length (cm)','petal width (cm)']].as_matrix()) ## should be the same as the trans cols

	transformer = SelectiveTruncatedSVD(cols=cols, n_components=1).fit(original)
	transformed = transformer.transform(original)

	untouched_cols = np.array(transformed[['petal length (cm)','petal width (cm)']].as_matrix())
	assert_array_almost_equal(compare_cols, untouched_cols)
	assert 'Concept1' in transformed.columns
	assert transformed.shape[1] == 3
	assert isinstance(transformer.get_decomposition(), TruncatedSVD)

	# test the selective mixin
	assert isinstance(transformer.get_features(), list)
	transformer.set_features(cols=None)
	assert transformer.get_features() is None

	
