import numpy as np
from numpy.testing import (assert_array_equal, assert_almost_equal, assert_array_almost_equal)
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from skutil.preprocessing import *
from skutil.decomposition import *
from skutil.feature_selection import *
import pandas as pd


__all__ = [
	'test_pipeline_basic',
	'test_pipeline_complex'
]

## Def data for testing
iris = load_iris()
X = pd.DataFrame(data=iris.data, columns=iris.feature_names)

def test_pipeline_basic():
	pipe = Pipeline([
			('selector', FeatureRetainer(cols=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)'])),
			('scaler', SelectiveScaler()),
			('model', RandomForestClassifier())
		])

	pipe.fit(X, iris.target)


def test_pipeline_complex():
	pipe = Pipeline([
			('selector', FeatureRetainer(cols=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)'])),
			('scaler', SelectiveScaler()),
			('boxcox', BoxCoxTransformer()),
			('pca', SelectivePCA()),
			('svd', SelectiveTruncatedSVD()),
			('model', RandomForestClassifier())
		])

	pipe.fit(X, iris.target)
