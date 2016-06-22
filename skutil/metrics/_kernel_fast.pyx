import numpy as np
cimport numpy as np

ctypedef float [:, :] float_array_2d_t
ctypedef double [:, :] double_array_2d_t
ctypedef np.npy_intp INTP

cdef fused floating1d:
	float[::1]
	double[::1]

cdef fused floating_array_2d_t:
	float_array_2d_t
	double_array_2d_t


np.import_array()


def _linear_kernel_fast(floating_array_2d_t X,
						floating_array_2d_t Y,
						INTP constant,
						floating_array_2d_t result):
	cdef INTP i, j, k
	cdef INTP n_samples_X = X.shape[0]
	cdef INTP n_features_X= X.shape[1]
	cdef INTP n_features_Y= Y.shape[1]

	with nogil:
		for i in range(n_samples_X):
			for j in range(n_features_Y):
				for k in range(n_features_X):
					result[i, j] = (X[i, k] * Y[k, j]) + constant

