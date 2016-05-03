import numpy as np
from numpy.testing import (assert_array_equal, assert_almost_equal, assert_array_almost_equal)
from sklearn.datasets import load_iris
from pynorm.preprocessing import SpatialSignTransformer

## Def data for testing
X = load_iris().data

def test_basic():
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
	test_basic()