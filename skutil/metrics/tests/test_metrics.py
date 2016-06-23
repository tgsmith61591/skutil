from skutil.metrics import *
import numpy as np
import timeit
from skutil.metrics._kernel import (_hilbert_dot, 
									_hilbert_matrix)
from numpy.testing import (assert_array_equal, assert_almost_equal, assert_array_almost_equal)

sigma = 0.05

def _get_train_array():
	return np.array([
			[0., 1.],
			[2., 3.],
			[2., 4.]
		])

def test_linear_kernel():
	X = np.reshape(np.arange(1,13), (4,3))
	assert_array_equal(linear_kernel(X=X),
		np.array([[  14.,   32.,   50.,   68.],
			      [  32.,   77.,  122.,  167.],
			      [  50.,  122.,  194.,  266.],
			      [  68.,  167.,  266.,  365.]]))


def test_poly_kernel():
	X = _get_train_array()
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

def test_exp():
	X = _get_train_array()
	answ = exponential_kernel(X)
	assert_array_almost_equal(answ, np.array([
		[   1.          , 54.59815003 , 665.14163304],
 		[  54.59815003  ,  1.         ,   1.64872127],
 		[ 665.14163304  ,  1.64872127 ,   1.        ]]))

def test_laplace():
	X = _get_train_array()
	answ = laplace_kernel(X)
	assert_array_almost_equal(answ, np.array([
		[  1.00000000e+00  , 2.98095799e+03  , 4.42413392e+05],
 		[  2.98095799e+03  , 1.00000000e+00  , 2.71828183e+00],
 		[  4.42413392e+05  , 2.71828183e+00  , 1.00000000e+00]]), 4)

def test_rbf():
	X = _get_train_array()
	answ = rbf_kernel(X, sigma=sigma)
	assert_array_almost_equal(answ, np.array([
		[ 1.         , 0.67032004 , 0.52204577],
		[ 0.67032004 , 1.         , 0.95122942],
		[ 0.52204577 , 0.95122942 , 1.        ]]))

def test_tanh():
	X = _get_train_array()
	answ = tanh_kernel(X)
	assert_array_almost_equal(answ, np.array([
		[ 0.76159416 , 0.99505475 , 0.9993293 ],
 		[ 0.99505475 , 1.         , 1.        ],
 		[ 0.9993293  , 1.         , 1.        ]]))

