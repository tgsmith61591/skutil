import numpy as np
from numpy.testing import (assert_array_equal, assert_almost_equal, assert_array_almost_equal)
from sklearn.datasets import load_iris
from pynorm.preprocessing import YeoJohnsonTransformer

## Def data for testing
X = load_iris().data

def test_basic():
	transformer = YeoJohnsonTransformer().fit(X)

	## Assert transform and inverse yields original
	transformed = transformer.transform(X)
	inverse = transformer.inverse_transform(transformed)
	assert_array_almost_equal(X, inverse)

	## TODO: more

if __name__ == '__main__':
	test_basic()