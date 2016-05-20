import numpy as np
import pandas as pd
from numpy.testing import (assert_array_equal, assert_almost_equal, assert_array_almost_equal)
from sklearn.datasets import load_iris
from pynorm.preprocessing import *

__all__ = [
	'test_bc',
	'test_yj',
	'test_ss',
	'test_selective_impute',
	'test_selective_scale',
	'test_selective_pca'
]


## Def data for testing
iris = load_iris()
X = iris.data


def test_bc():
	transformer = BoxCoxTransformer().fit(X)

	## Assert similar lambdas
	assert_array_almost_equal(transformer.lambda_,
		np.array([-0.14475082666963388, 0.26165380763371671, 0.93129521538860016, 0.64441777772515185]))

	## Assert exact shifts
	assert_array_equal(transformer.shift_, np.array([ 0.,  0.,  0.,  0.]))

	## Now subtract out some fixed amt from X, assert we get different values:
	x = X - 10
	transformer = BoxCoxTransformer().fit(x)

	## Assert similar lambdas
	assert_array_almost_equal(transformer.lambda_,
		np.array([0.59843688208993162, 0.69983717204250795, 0.5928185584100969, 0.42501980692063013]))

	## Assert exact shifts
	assert_array_equal(transformer.shift_, np.array([ 5.700001,  8.000001,  9.000001,  9.900001]))

	## If we inverse transform, it should be nearly the same as the input matrix
	transformed = transformer.transform(X)
	inversed = transformer.inverse_transform(transformed)
	assert_array_almost_equal(X, inversed)




def test_yj():
	transformer = YeoJohnsonTransformer().fit(X)

	## Assert transform works...
	transformed = transformer.transform(X)

	inverse = transformer.inverse_transform(transformed)
	assert inverse is NotImplemented, 'expected NotImplemented'

	## TODO: more




def test_ss():
	transformer = SpatialSignTransformer().fit(X)

	## Assert transform and inverse yields original
	transformed = transformer.transform(X)
	inverse = transformer.inverse_transform(transformed)
	assert_array_almost_equal(X, inverse)

	l = len(transformer.sq_nms_[transformer.sq_nms_ == np.inf])
	assert l == 0, 'expected len == 0, but got %i' % l

	## Force inf as the sq norm
	x = np.zeros((5,5))
	transformer.fit(x)

	## Assert transform and inverse yields original
	transformed = transformer.transform(x)
	inverse = transformer.inverse_transform(transformed) ## returns to zero internally
	assert_array_almost_equal(x, inverse)

	## Assert all Inf
	l = len(transformer.sq_nms_[transformer.sq_nms_ == np.inf])
	assert l == 5, 'expected len == 5, but got %i' % l


def test_selective_impute():
	a = np.random.rand(5, 5)

	## add some missing vals
	a[0, 3] = np.nan
	a[1, 2] = np.nan

	## throw into a DF
	df = pd.DataFrame.from_records(data=a, columns=['a','b','c','d','e'])
	transformer = SelectiveImputer(cols=['d']).fit(df)
	df = transformer.transform(df)

	assert not pd.isnull(df.iloc[0, 3])
	assert pd.isnull(df.iloc[1, 2])


def test_selective_pca():
	original = pd.DataFrame.from_records(data = X, columns = iris.feature_names)
	cols = [original.columns[0]] ## Only perform on first...
	compare_cols = np.array(original[['sepal width (cm)','petal length (cm)','petal width (cm)']].as_matrix()) ## should be the same as the trans cols

	transformer = SelectivePCA(cols=cols, n_components=0.85).fit(original)
	transformed = transformer.transform(original)

	untouched_cols = np.array(transformed[['sepal width (cm)','petal length (cm)','petal width (cm)']].as_matrix())
	assert_array_almost_equal(compare_cols, untouched_cols)
	assert 'PC1' in transformed.columns
	assert transformed.shape[1] == 4



def test_selective_scale():
	original = pd.DataFrame.from_records(data = X, columns = iris.feature_names)
	cols = [original.columns[0]] ## Only perform on first...

	original_means = np.mean(X, axis=0) ## array([ 5.84333333,  3.054     ,  3.75866667,  1.19866667])
	original_std   = np.std(X, axis=0)  ## array([ 0.82530129,  0.43214658,  1.75852918,  0.76061262])

	transformer = SelectiveScaler(cols=cols).fit(original)
	transformed = transformer.transform(original)

	new_means = np.array(np.mean(transformed, axis = 0).tolist()) ## expected: array([ 0.  ,  3.054     ,  3.75866667,  1.19866667])
	new_std   = np.array(np.std(transformed, axis = 0).tolist())  ## expected: array([ 1.  ,  0.43214658,  1.75852918,  0.76061262])

	assert_array_almost_equal(new_means, np.array([ 0.  ,  3.054     ,  3.75866667,  1.19866667]))
	assert_array_almost_equal(new_std,   np.array([ 1.  ,  0.43214658,  1.75852918,  0.76061262]))


