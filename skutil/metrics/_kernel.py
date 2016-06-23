import numpy as np
from skutil import exp, log
from ._kernel_fast import _hilbert_dot_fast, _hilbert_matrix_fast
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.metrics.pairwise import (check_pairwise_arrays,
									  linear_kernel as lk)


__all__ = [
	'exponential_kernel',
	'laplace_kernel',
	'linear_kernel',
	'polynomial_kernel',
	'rbf_kernel',
	'tanh_kernel'
]



### Cython proxies
def _hilbert_dot(x, y, scalar=1.0):
	#return 2 * safe_sparse_dot(x, y) - safe_sparse_dot(x, x.T) - safe_sparse_dot(y, y.T)
	x, y = x.astype(np.double, order='C'), y.astype(np.double, order='C')
	return _hilbert_dot_fast(x, y, scalar)


def _hilbert_matrix(X, Y=None, scalar=1.0):
	X, Y = check_pairwise_arrays(X, Y)
	X, Y = X.astype(np.double, order='C'), Y.astype(np.double, order='C').T # transposing Y here!

	res = np.zeros((X.shape[0], Y.shape[1]), dtype=X.dtype)
	_hilbert_matrix_fast(X, Y, res, np.double(scalar))
	return res


## Kernel functions
def exponential_kernel(X, Y=None, sigma=1.0):
	return exp(_hilbert_matrix(X, Y, scalar = -1.0) / 2*np.power(sigma, 2) )


def laplace_kernel(X, Y=None, sigma=1.0):
	return exp(_hilbert_matrix(X, Y, scalar = -1.0) / sigma)


def linear_kernel(X, Y=None, constant=0.0):
	"""Compute the sklearn linear_kernel but
	allow for constant scalars"""
	l = lk(X,Y)
	return l + constant


def polynomial_kernel(X, Y=None, alpha=1.0, degree=1.0, constant=1.0):
	lc = linear_kernel(X=X, Y=Y, constant=0.0)
	return np.power(lc * alpha + constant, degree)


def rbf_kernel(X, Y=None, sigma=1.0):
	return exp(_hilbert_matrix(X, Y, scalar=sigma))


def tanh_kernel(X, Y=None, constant=0.0, alpha=1.0):
	lc = linear_kernel(X=X, Y=Y, constant=0.0) # don't add it here
	return np.tanh(alpha * lc + constant)      # add it here
