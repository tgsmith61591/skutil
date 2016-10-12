from __future__ import print_function
import numpy as np
import warnings
from skutil import exp
from sklearn.metrics.pairwise import (check_pairwise_arrays,
                                      linear_kernel as lk)
from ._kernel_fast import (_hilbert_dot_fast, _hilbert_matrix_fast, _spline_kernel_fast)


__all__ = [
    'exponential_kernel',
    'gaussian_kernel',
    'inverse_multiquadric_kernel',
    'laplace_kernel',
    'linear_kernel',
    'multiquadric_kernel',
    'polynomial_kernel',
    'power_kernel',
    'rbf_kernel',
    'spline_kernel',
    'tanh_kernel'
]


# Utils
def _div(num, div):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # do division operation -- might throw runtimewarning
        return num / div


def _prep_X_Y_for_cython(X, Y):
    X, Y = check_pairwise_arrays(X, Y)
    X, Y = X.astype(np.double, order='C'), Y.astype(np.double, order='C').T # transposing Y here!
    res = np.zeros((X.shape[0], Y.shape[1]), dtype=X.dtype)
    return X, Y, res


# Cython proxies
def _hilbert_dot(x, y, scalar=1.0):
    # return 2 * safe_sparse_dot(x, y) - safe_sparse_dot(x, x.T) - safe_sparse_dot(y, y.T)
    x, y = x.astype(np.double, order='C'), y.astype(np.double, order='C')
    return _hilbert_dot_fast(x, y, scalar)


def _hilbert_matrix(X, Y=None, scalar=1.0):
    X, Y, res = _prep_X_Y_for_cython(X, Y)
    _hilbert_matrix_fast(X, Y, res, np.double(scalar))
    return res


# Kernel functions
def exponential_kernel(X, Y=None, sigma=1.0):
    return exp(_hilbert_matrix(X, Y, scalar=-1.0) / 2*np.power(sigma, 2) )


def gaussian_kernel(X, Y=None, sigma=1.0):
    return exp(-np.power(_hilbert_matrix(X, Y), 2.0) / 2*np.power(sigma, 2) )


def inverse_multiquadric_kernel(X, Y=None, constant=1.0):
    return _div(1.0, multiquadric_kernel(X, Y, constant))


def laplace_kernel(X, Y=None, sigma=1.0):
    return exp(_hilbert_matrix(X, Y, scalar=-1.0) / sigma)


def linear_kernel(X, Y=None, constant=0.0):
    """Compute the sklearn linear_kernel but
    allow for constant scalars"""
    l = lk(X,Y)
    return l + constant


def multiquadric_kernel(X, Y=None, constant=0.0):
    hs = _hilbert_matrix(X=X, Y=Y, scalar=1.0)
    hs = np.power(hs, 2.0)
    return np.sqrt(hs + np.power(constant, 2.0))


def polynomial_kernel(X, Y=None, alpha=1.0, degree=1.0, constant=1.0):
    lc = linear_kernel(X=X, Y=Y, constant=0.0)
    return np.power(lc * alpha + constant, degree)


def power_kernel(X, Y=None, degree=1.0):
    return -np.power(_hilbert_matrix(X, Y), degree)


def rbf_kernel(X, Y=None, sigma=1.0):
    return exp(_hilbert_matrix(X, Y, scalar=sigma))


def spline_kernel(X, Y=None):
    X, Y, res = _prep_X_Y_for_cython(X, Y)
    _spline_kernel_fast(X, Y, res)
    return res


def tanh_kernel(X, Y=None, constant=0.0, alpha=1.0):
    lc = linear_kernel(X=X, Y=Y, constant=0.0)  # don't add it here
    return np.tanh(alpha * lc + constant)     # add it here
