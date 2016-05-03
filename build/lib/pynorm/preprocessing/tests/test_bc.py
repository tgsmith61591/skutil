import numpy as np
from numpy.testing import (assert_array_equal, assert_almost_equal, assert_array_almost_equal)
from sklearn.datasets import load_iris
from pynorm.preprocessing import BoxCoxTransformer

## Def data for testing
X = load_iris().data

def test_basic():
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

if __name__ == '__main__':
	test_basic()