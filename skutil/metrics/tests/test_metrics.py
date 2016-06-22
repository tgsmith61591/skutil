from skutil.metrics import *
from skutil.metrics import pairwise
import numpy as np
from numpy.testing import (assert_array_equal, assert_almost_equal, assert_array_almost_equal)

def test_linear_kernel():
	X = np.reshape(np.arange(1,13), (4,3))
	assert_array_equal(linear_kernel(X=X),
		np.array([[  14.,   32.,   50.,   68.],
			      [  32.,   77.,  122.,  167.],
			      [  50.,  122.,  194.,  266.],
			      [  68.,  167.,  266.,  365.]]))