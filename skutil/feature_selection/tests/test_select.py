from __future__ import print_function
import numpy as np
import pandas as pd
from numpy.testing import (assert_array_equal, assert_almost_equal, assert_array_almost_equal)
from sklearn.datasets import load_iris
from skutil.feature_selection import *

__all__ = [
	'test_feature_selector',
	'test_multi_collinearity'
]

## Def data for testing
iris = load_iris()
X = pd.DataFrame.from_records(data = iris.data, columns = iris.feature_names)


def test_feature_selector():
	transformer = FeatureSelector().fit(X)
	assert transformer.transform(X).shape[1] == 4

	cols = ['sepal length (cm)', 'sepal width (cm)']
	transformer = FeatureSelector(cols=cols).fit(X)
	assert transformer.transform(X).shape[1] == 2

	# test the selective mixin
	assert isinstance(transformer.get_features(), list)
	transformer.set_features(cols=None)
	assert transformer.get_features() is None


def test_multi_collinearity():
	transformer = MulticollinearityFilterer()

	x = transformer.fit_transform(X)
	assert x.shape[1] == 3

	col_nms = x.columns
	assert col_nms[0] == 'sepal length (cm)'
	assert col_nms[1] == 'sepal width (cm)'
	assert col_nms[2] == 'petal width (cm)'
	assert len(transformer.drop_) == 1

	# test the selective mixin
	assert isinstance(transformer.get_features(), list)
	transformer.set_features(cols=None)
	assert transformer.get_features() is None