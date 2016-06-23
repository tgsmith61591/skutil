from skutil.metrics import *
import numpy as np
import timeit
from skutil.metrics._kernel import _hilbert_dot, _hilbert_matrix
from numpy.testing import (assert_array_equal, assert_almost_equal, assert_array_almost_equal)

def test_linear_kernel():
	X = np.reshape(np.arange(1,13), (4,3))
	assert_array_equal(linear_kernel(X=X),
		np.array([[  14.,   32.,   50.,   68.],
			      [  32.,   77.,  122.,  167.],
			      [  50.,  122.,  194.,  266.],
			      [  68.,  167.,  266.,  365.]]))


def test_poly_kernel():
	X = np.array([
			[0., 1.],
			[2., 3.],
			[2., 4.]
		])

	assert_array_equal(polynomial_kernel(X),
		np.array([[2, 4, 5],
				  [4,14,17],
				  [5,17,21]]))

def test_hilbert():
	X = np.array([10.0, 2.0, 3.0, 4.0])
	Y = np.array([5.0 , 6.0, 7.0, 8.0])

	answ = _hilbert_dot(X, Y)
	assert answ == -73.0, 'expected -73.0 but got %.3f' % answ


	Z = np.array([X, Y])
	answ = _hilbert_matrix(Z)
	assert_array_equal(answ, np.array([
			[0   , -73],
			[-73 ,   0]
		]))

	# speed...
	if False:
		def wrap(fun, *args):
			def wrapped():
				return fun(*args)
			return wrapped

		X = np.random.rand(1000)
		print(timeit.timeit(wrap(_hilbert_dot, X, X)))
