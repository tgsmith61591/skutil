#cython: boundscheck=False
#cython: cdivision=True
#cython: wraparound=False

from libc.string cimport memset
from libc.math cimport pow
import numpy as np
cimport numpy as np

ctypedef float [:, :] float_array_2d_t
ctypedef double [:, :] double_array_2d_t
ctypedef np.npy_intp INTP
ctypedef np.npy_float FLOAT

cdef fused floating1d:
	float[::1]
	double[::1]

cdef fused floating_array_2d_t:
	float_array_2d_t
	double_array_2d_t


np.import_array()


def _hilbert_matrix_fast(floating_array_2d_t X,
						floating_array_2d_t Y,
						floating_array_2d_t res,
						FLOAT scalar):
	cdef INTP i, j, k
	cdef double sxy, sx, sy
	cdef INTP n_samples_X = X.shape[0]
	cdef INTP n_features_X= X.shape[1]
	cdef INTP n_features_Y= Y.shape[1]

	with nogil:
		for i in range(n_samples_X):
			for j in range(n_features_Y):
				sx = 0 # reset for every xi
				sy = 0 # reset for every yi... might be a better way to keep track of this?
				sxy = 0 # reset for every unique XiYi combo
				for k in range(n_features_X): # k also equals the number of rows in Y
					sx += X[i, k] * X[i, k]
					sy += Y[k, j] * Y[k, j]
					sxy += X[i, k] * Y[k, j]

				res[i, j] = ((2 * sxy) - sx - sy) * scalar


def _hilbert_dot_fast(np.ndarray[np.float_t, ndim=1, mode='c'] x,
					np.ndarray[np.float_t, ndim=1, mode='c'] y,
					FLOAT scalar):
	cdef int i
	cdef double s1 = 0, s2 = 0, s3 = 0 # initialize the sums
	cdef int len_x = x.shape[0]

	with nogil:
		for i in range(len_x):
			s1 += x[i] * y[i]
			s2 += x[i] * x[i]
			s3 += y[i] * y[i]

	return scalar * (2 * s1 - s2 - s3)


def _spline_kernel_fast(floating_array_2d_t X, 
						floating_array_2d_t Y,
						floating_array_2d_t res):
	cdef int i, j, k
	cdef int m = X.shape[0]
	cdef int n = X.shape[1]
	cdef int n_features_Y = Y.shape[1]
	cdef double prod, front, mid, back, a, b, min_el

	with nogil:
		for i in range(m):
			for j in range(n_features_Y):

				## Reinitialize this for each vector dot
				prod = 1

				for k in range(n):
					a = X[i, k] # the element in the X matrix
					b = Y[j, k] # the element in the Y matrix

					# get the min between the two
					if a < b:
						min_el = a
					else:
						min_el = b

					# compute the three parts
					front = a * b * (min_el + 1)
					mid = ((a + b) / 2.0) * (min_el * min_el)
					back = pow(min_el, 3) / 3.0
					prod *= (((front + 1) - mid) + back)

				## assign to output matrix
				res[i, j] = prod





