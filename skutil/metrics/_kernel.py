import numpy as np
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.metrics.pairwise import check_pairwise_arrays
from ._kernel_fast import _linear_kernel_fast


__all__ = [
	'linear_kernel'
]


def linear_kernel(X, Y=None, constant=0.0):
	"""Performs the linear kernel computation,
	but unlike sklearn allows for a constant scaling term
	that is added at the same time as the computation rather
	than requiring an additional pass of O(N*M).
	"""
	X, Y = check_pairwise_arrays(X, Y)
	result = np.zeros((X.shape[0], Y.shape[1]), dtype=X.dtype)
	_linear_kernel_fast(X, Y.T, constant)
	return result
