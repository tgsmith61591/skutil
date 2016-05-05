import numpy as np
from numpy.testing import (assert_array_equal, assert_almost_equal, assert_array_almost_equal)
from sklearn.datasets import load_iris
from pynorm.preprocessing import *

## Def data for testing
X = load_iris().data




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

	## Assert transform and inverse yields original
	transformed = transformer.transform(X)
	inverse = transformer.inverse_transform(transformed)
	assert_array_almost_equal(X, inverse)
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

if __name__ == '__main__':
	test_bc()
	test_yj()
	test_ss()
