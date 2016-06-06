import numpy as np
import pandas as pd
from numpy.testing import (assert_array_equal, assert_almost_equal, assert_array_almost_equal)
from sklearn.datasets import load_iris
from skutil.utils import *


## Def data for testing
iris = load_iris()
X = pd.DataFrame.from_records(data = iris.data, columns = iris.feature_names)

# ensure things work with a categorical feature
X['target'] = ['A' if x == 1 else 'B' if x == 2 else 'C' for x in iris.target]

# exact copy of second col
X['perfect'] = X[[1]]

def _check_equal(L1, L2):
    return len(L1) == len(L2) and sorted(L1) == sorted(L2)


def test_flatten():
	a = [[[],3,4],['1','a'],[[[1]]],1,2]
	b = flatten_all(a)
	assert _check_equal(b, [3,4,'1','a',1,1,2])

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

def test_validate_on_non_df():
	x = iris.data
	validate_is_pd(x, None)