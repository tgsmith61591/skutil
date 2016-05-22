from __future__ import print_function
import numpy as np
import pandas as pd
from numpy.testing import (assert_array_equal, assert_almost_equal, assert_array_almost_equal)
from sklearn.datasets import load_iris
from skutil.preprocessing import *

__all__ = [
	'test_boxcox',
	'test_function_mapper',
	'test_yeo_johnson',
	'test_spatial_sign',
	'test_selective_impute',
	'test_selective_scale'
]


## Def data for testing
iris = load_iris()
X = pd.DataFrame(data=iris.data, columns=iris.feature_names)



def test_boxcox():
	transformer = BoxCoxTransformer().fit(X) ## Will fit on all cols

	## Assert similar lambdas
	assert_array_almost_equal(sorted(transformer.lambda_.values()),
		np.array([-0.14475082666963388, 0.26165380763371671, 0.64441777772515185, 0.93129521538860016]))

	## Assert exact shifts
	assert_array_equal(transformer.shift_.values(), np.array([ 0.,  0.,  0.,  0.]))

	## Now subtract out some fixed amt from X, assert we get different values:
	x = X - 10
	transformer = BoxCoxTransformer().fit(x)

	## Assert similar lambdas
	assert_array_almost_equal(sorted(transformer.lambda_.values()),
		np.array([0.42501980692063013, 0.5928185584100969, 0.59843688208993162, 0.69983717204250795]))

	## Assert exact shifts
	assert_array_equal(sorted(transformer.shift_.values()), np.array([ 5.700001,  8.000001,  9.000001,  9.900001]))

	## assert transform works
	transformed = transformer.transform(X)
	assert isinstance(transformed, pd.DataFrame)

	## assert as df false yields array
	assert isinstance(BoxCoxTransformer(as_df=False).fit_transform(X), np.ndarray)

	# test the selective mixin
	assert isinstance(transformer.get_features(), list)
	transformer.set_features(cols=None)
	assert transformer.get_features() is None


def test_function_mapper():
	Y = np.array([['USA','RED','a'],
                  ['MEX','GRN','b'],
                  ['FRA','RED','b']])
	y = pd.DataFrame.from_records(data = Y, columns = ['A','B','C'])
	# Tack on a pseudo-numeric col
	y['D'] = np.array(['$5,000','$6,000','$7'])
	y['E'] = np.array(['8%','52%','0.3%'])

	def fun(x):
		return x.replace('[\$,%]', '', regex=True).astype(float)

	transformer = FunctionMapper(cols=['D','E'], fun=fun).fit(y)
	transformed = transformer.transform(y)
	assert transformed['D'].dtype == float



def test_yeo_johnson():
	transformer = YeoJohnsonTransformer().fit(X) ## will fit on all cols

	## Assert transform works...
	transformed = transformer.transform(X)
	assert isinstance(transformed, pd.DataFrame)

	## assert as df false yields array
	assert isinstance(YeoJohnsonTransformer(as_df=False).fit_transform(X), np.ndarray)

	# test the selective mixin
	assert isinstance(transformer.get_features(), list)
	transformer.set_features(cols=None)
	assert transformer.get_features() is None
	
	## TODO: more




def test_spatial_sign():
	transformer = SpatialSignTransformer().fit(X) ## will fit to all cols

	## Assert transform works
	transformed = transformer.transform(X)

	vals = np.array(transformer.sq_nms_.values())
	l = len(vals[vals == np.inf])
	assert l == 0, 'expected len == 0, but got %i' % l

	## Force inf as the sq norm
	x = np.zeros((5,5))
	xdf= pd.DataFrame.from_records(data=x)
	transformer = SpatialSignTransformer().fit(xdf)

	## Assert transform works
	transformed = transformer.transform(xdf)
	assert isinstance(transformed, pd.DataFrame)

	## Assert all Inf
	vals = np.array(transformer.sq_nms_.values())
	l = len(vals[vals == np.inf])
	assert l == 5, 'expected len == 5, but got %i' % l

	## assert as df false yields array
	assert isinstance(SpatialSignTransformer(as_df=False).fit_transform(X), np.ndarray)

	# test the selective mixin
	assert isinstance(transformer.get_features(), list)
	transformer.set_features(cols=None)
	assert transformer.get_features() is None


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

	# test the selective mixin
	assert isinstance(transformer.get_features(), list)
	transformer.set_features(cols=None)
	assert transformer.get_features() is None


def test_selective_scale():
	original = X
	cols = [original.columns[0]] ## Only perform on first...

	original_means = np.mean(X, axis=0) ## array([ 5.84333333,  3.054     ,  3.75866667,  1.19866667])
	original_std   = np.std(X, axis=0)  ## array([ 0.82530129,  0.43214658,  1.75852918,  0.76061262])

	transformer = SelectiveScaler(cols=cols).fit(original)
	transformed = transformer.transform(original)

	new_means = np.array(np.mean(transformed, axis = 0).tolist()) ## expected: array([ 0.  ,  3.054     ,  3.75866667,  1.19866667])
	new_std   = np.array(np.std(transformed, axis = 0).tolist())  ## expected: array([ 1.  ,  0.43214658,  1.75852918,  0.76061262])

	assert_array_almost_equal(new_means, np.array([ 0.  ,  3.054     ,  3.75866667,  1.19866667]))
	assert_array_almost_equal(new_std,   np.array([ 1.  ,  0.43214658,  1.75852918,  0.76061262]))

	# test the selective mixin
	assert isinstance(transformer.get_features(), list)
	transformer.set_features(cols=None)
	assert transformer.get_features() is None


