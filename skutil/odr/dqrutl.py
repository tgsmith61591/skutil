from __future__ import print_function, division
import numpy as np
import dqrsl
from sklearn.utils import check_array
from numpy.linalg import matrix_rank

# WARNING: there is little-to-no validation of input in these functions,
# and crashes may be caused by inappropriate usage. Use with care.

__all__ = [
    'qr_decomposition',
    'QRDecomposition'
]


def _validate_matrix_size(n, p):
    if n * p > 2147483647:
        raise ValueError('too many elements for Fortran LINPACK routine')


def _safecall(fun, name, *args, **kwargs):
    """A method to call a LAPACK or LINPACK subroutine internally"""
    ret = fun(*args, **kwargs)

    # since we're operating on arrays in place, we don't need this
    # if ret[-1] < 0:
    #   raise ValueError("illegal value in %d-th argument of internal %s"
    #       % (-ret[-1], name))


def qr_decomposition(X, job=1):
    """Performs the QR decomposition using LINPACK, BLAS and LAPACK
    Fortran subroutines.

    Parameters
    ----------

    X : array_like, shape (n_samples, n_features)
        The matrix to decompose

    job : int, optional (default=1)
        Whether to perform pivoting. 0 is False, any other value
        will be coerced to 1 (True).
    """

    X = check_array(X, dtype='numeric', order='F', copy=True)
    n, p = X.shape

    # check on size
    _validate_matrix_size(n, p)
    rank = matrix_rank(X)

    # validate job:
    job_ = 0 if not job else 1

    qraux, pivot, work = (np.zeros(p, dtype=np.double, order='F'),
                          # can't use arange, because need fortran order ('order' not kw in arange)
                          np.array([i for i in range(1, p + 1)], dtype=np.int, order='F'),
                          np.zeros(p, dtype=np.double, order='F'))

    # sanity checks
    assert qraux.shape[0] == p, 'expected qraux to be of length %i' % p
    assert pivot.shape[0] == p, 'expected pivot to be of length %i' % p
    assert work.shape[0] == p, 'expected work to be of length %i' % p

    # call the fortran module IN PLACE
    _safecall(dqrsl.dqrdc, 'dqrdc', X, n, n, p, qraux, pivot, work, job_)

    # do returns
    return (X,
            rank,
            qraux,
            (pivot - 1) if job_ else None)  # subtract one because pivot started at 1 for the fortran


def _qr_R(qr):
    """Extract the R matrix from a QR decomposition"""
    min_dim = min(qr.shape)
    return qr[:min_dim + 1, :]


class QRDecomposition():
    """Performs the QR decomposition using LINPACK, BLAS and LAPACK
    Fortran subroutines, and provides an interface for other useful
    QR utility methods.

    Parameters
    ----------

    X : array_like, shape (n_samples, n_features)
        The matrix to decompose

    pivot : int, optional (default=1)
        Whether to perform pivoting. 0 is False, any other value
        will be coerced to 1 (True).

    Attributes
    ----------

    qr : array_like, shape (n_samples, n_features)
        The decomposed matrix

    qraux : array_like, shape (n_features,)
        qraux contains further information required to recover
        the orthogonal part of the decomposition.

    pivot : array_like, shape (n_features,)
        The pivots, if pivot was set to 1, else None

    rank : int
        The rank of the input matrix
    """

    def __init__(self, X, pivot=1):
        self.job_ = 0 if not pivot else 1
        self._decompose(X)

    def _decompose(self, X):
        """Decomposes the matrix"""
        # perform the decomposition
        self.qr, self.rank, self.qraux, self.pivot = qr_decomposition(X, self.job_)

    def get_coef(self, X):
        qr, qraux = self.qr, self.qraux
        n, p = qr.shape

        # sanity check
        assert isinstance(qr, np.ndarray), 'internal error: QR should be a np.ndarray but got %s' % type(qr)
        assert isinstance(qraux, np.ndarray), 'internal error: qraux should be a np.ndarray but got %s' % type(qraux)

        # validate input array
        X = check_array(X, dtype='numeric', copy=True, order='F')
        nx, ny = X.shape
        if nx != n:
            raise ValueError('qr and X must have same number of rows')

        # check on size
        _validate_matrix_size(n, p)

        # get the rank of the decomposition
        k = self.rank

        # get ix vector
        # if p > n:
        #   ix = np.ones(n + (p - n)) * np.nan
        #   ix[:n] = np.arange(n) # i.e., array([0,1,2,nan,nan,nan])
        # else:
        #   ix = np.arange(n)

        # set up the structures to alter
        coef, info = (np.zeros((k, ny), dtype=np.double, order='F'),
                      np.zeros(1, dtype=np.int, order='F'))

        # call the fortran module IN PLACE
        _safecall(dqrsl.dqrcf, 'dqrcf', qr, n, k, qraux, X, ny, coef, 0)

        # post-processing
        # if k < p:
        #   cf = np.ones((p,ny)) * np.nan
        #   cf[self.pivot[np.arange(k)], :] = coef
        return coef if not k < p else coef[self.pivot[np.arange(k)], :]

    def get_rank(self):
        """Get the rank of the decomposition"""
        return self.rank

    def get_R(self):
        """Get the R matrix from the decomposition"""
        return _qr_R(self.qr)

    def get_R_rank(self):
        """Get the rank of the R matrix"""
        return matrix_rank(self.get_R())
