#cython: boundscheck=False
#cython: cdivision=True
#cython: wraparound=False

from libc.string cimport memset
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
				sxy = 0 # reset for every unique xiyi combo
				for k in range(n_features_X):
					sx += X[i, k] * X[i, k]
					sy += Y[k, j] * Y[k, j]
					sxy += X[i, k] * Y[k, j]

				res[i, j] = (2 * sxy - sx - sy) * scalar


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
