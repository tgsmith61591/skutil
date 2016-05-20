import numpy as np
import pandas as pd
from numpy.testing import (assert_array_equal, assert_almost_equal, assert_array_almost_equal)
from sklearn.datasets import load_iris
from pynorm.decomposition import *


__all__ = [
	'test_selective_pca'
]



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
