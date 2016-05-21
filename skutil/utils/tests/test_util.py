import numpy as np
import pandas as pd
from numpy.testing import (assert_array_equal, assert_almost_equal, assert_array_almost_equal)
from sklearn.datasets import load_iris
from skutil.utils import *

__all__ = [
	'test_perfect_collinearity',
	'test_is_numeric',
	'test_get_numeric'
]


## Def data for testing
iris = load_iris()
X = pd.DataFrame.from_records(data = iris.data, columns = iris.feature_names)

# ensure things work with a categorical feature
X['target'] = ['A' if x == 1 else 'B' if x == 2 else 'C' for x in iris.target]

# exact copy of second col
X['perfect'] = X[[1]]


def test_perfect_collinearity():
	## This interally ensures that this method can handle categorical data
	series = perfect_collinearity_check(X)
	assert series.perfect == 1.0, 'expected perfect collinearity'

	## Test that all categorical won't work
	a = pd.Series(['a','b','b'])
	b = pd.DataFrame(a)
	failed = False

	try:
		perfect_collinearity_check(b)
	except ValueError as v:
		failed = True
	assert failed, 'expected the collinearity test to fail'

	## Test that adding just two numeric cols will make it work
	b['b'] = [1,2,3]
	b['c'] = [3,2,1]
	series = perfect_collinearity_check(b)

	assert series.b == 1.0, 'expected perfect collinearity'
	assert series.c == 1.0, 'expected perfect collinearity'


def test_is_numeric():
	assert is_numeric(1)
	assert is_numeric(1.)
	assert is_numeric(1L)
	assert is_numeric(np.int(1.0))
	assert is_numeric(np.float(1))
	assert is_numeric(1e-12)
	assert not is_numeric('a')


def test_get_numeric():
	a = pd.Series(['a','b','b'])
	b = pd.DataFrame(a)

	assert len(get_numeric(b)) == 0, 'expected empty'

