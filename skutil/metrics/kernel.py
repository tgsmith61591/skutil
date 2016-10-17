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


def _div(num, div):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # do division operation -- might throw runtimewarning
        return num / div


def _prep_X_Y_for_cython(X, Y):
    X, Y = check_pairwise_arrays(X, Y)
    X, Y = X.astype(np.double, order='C'), Y.astype(np.double, order='C').T  # transposing Y here!
    res = np.zeros((X.shape[0], Y.shape[1]), dtype=X.dtype)
    return X, Y, res


# Cython proxies
def _hilbert_dot(x, y, scalar=1.0):
    # return ``2 * safe_sparse_dot(x, y) - safe_sparse_dot(x, x.T) - safe_sparse_dot(y, y.T)``
    x, y = x.astype(np.double, order='C'), y.astype(np.double, order='C')
    return _hilbert_dot_fast(x, y, scalar)


def _hilbert_matrix(X, Y=None, scalar=1.0):
    X, Y, res = _prep_X_Y_for_cython(X, Y)
    _hilbert_matrix_fast(X, Y, res, np.double(scalar))
    return res


def exponential_kernel(X, Y=None, sigma=1.0):
    """The ``exponential_kernel`` is closely related to the ``gaussian_kernel``, 
    with only the square of the norm left out. It is also an ``rbf_kernel``. Note that
    the adjustable parameter, ``sigma``, plays a major role in the performance of the
    kernel and should be carefully tuned. If overestimated, the exponential will behave 
    almost linearly and the higher-dimensional projection will start to lose its non-linear 
    power. In the other hand, if underestimated, the function will lack regularization and 
    the decision boundary will be highly sensitive to noise in training data.

    The kernel is given by:

        :math:`k(x, y) = exp( -||x-y|| / 2\\sigma^2 )`

    Parameters
    ----------

    X : array_like (float), shape=(n_samples, n_features)
        The array of pandas DataFrame on which to compute 
        the kernel. If ``Y`` is None, the kernel will be computed
        with ``X``.

    Y : array_like (float), shape=(n_samples, n_features), optional (default=None)
        The array of pandas DataFrame on which to compute 
        the kernel. If ``Y`` is None, the kernel will be computed
        with ``X``.

    sigma : float, optional (default=1.0)
        The exponential tuning parameter.

    Returns
    -------

    c : float
        The result of the kernel computation.

    References
    ----------

    Souza, Cesar R., Kernel Functions for Machine Learning Applications
    http://crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html
    """
    c = exp(_hilbert_matrix(X, Y, scalar=-1.0) / 2 * np.power(sigma, 2))
    return c


def gaussian_kernel(X, Y=None, sigma=1.0):
    """The ``gaussian_kernel`` is closely related to the ``exponential_kernel``.
    It is also an ``rbf_kernel``. Note that the adjustable parameter, ``sigma``, 
    plays a major role in the performance of the kernel and should be carefully 
    tuned. If overestimated, the exponential will behave almost linearly and 
    the higher-dimensional projection will start to lose its non-linear 
    power. In the other hand, if underestimated, the function will lack regularization and 
    the decision boundary will be highly sensitive to noise in training data.

    The kernel is given by:

        :math:`k(x, y) = exp( -||x-y||^2 / 2\\sigma^2 )`

    Parameters
    ----------

    X : array_like (float), shape=(n_samples, n_features)
        The array of pandas DataFrame on which to compute 
        the kernel. If ``Y`` is None, the kernel will be computed
        with ``X``.

    Y : array_like (float), shape=(n_samples, n_features), optional (default=None)
        The array of pandas DataFrame on which to compute 
        the kernel. If ``Y`` is None, the kernel will be computed
        with ``X``.

    sigma : float, optional (default=1.0)
        The exponential tuning parameter.

    Returns
    -------

    c : float
        The result of the kernel computation.

    References
    ----------

    Souza, Cesar R., Kernel Functions for Machine Learning Applications
    http://crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html
    """
    c = exp(-np.power(_hilbert_matrix(X, Y), 2.0) / 2 * np.power(sigma, 2))
    return c


def inverse_multiquadric_kernel(X, Y=None, constant=1.0):
    """The ``inverse_multiquadric_kernel``, as with the ``gaussian_kernel``, 
    results in a kernel matrix with full rank (Micchelli, 1986) and thus forms 
    an infinite dimension feature space.

    The kernel is given by:

        :math:`k(x, y) = 1 / sqrt( -||x-y||^2 + c^2 )`

    Parameters
    ----------

    X : array_like (float), shape=(n_samples, n_features)
        The array of pandas DataFrame on which to compute 
        the kernel. If ``Y`` is None, the kernel will be computed
        with ``X``.

    Y : array_like (float), shape=(n_samples, n_features), optional (default=None)
        The array of pandas DataFrame on which to compute 
        the kernel. If ``Y`` is None, the kernel will be computed
        with ``X``.

    constant : float, optional (default=1.0)
        The linear tuning parameter.

    Returns
    -------

    c : float
        The result of the kernel computation.

    References
    ----------

    Souza, Cesar R., Kernel Functions for Machine Learning Applications
    http://crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html
    """
    c = _div(1.0, multiquadric_kernel(X, Y, constant))
    return c


def laplace_kernel(X, Y=None, sigma=1.0):
    """The ``laplace_kernel`` is completely equivalent to the ``exponential_kernel``, 
    except for being less sensitive for changes in the ``sigma`` parameter. 
    Being equivalent, it is also an ``rbf_kernel``.

    The kernel is given by:

        :math:`k(x, y) = exp( -||x-y|| / \\sigma )`

    Parameters
    ----------

    X : array_like (float), shape=(n_samples, n_features)
        The array of pandas DataFrame on which to compute 
        the kernel. If ``Y`` is None, the kernel will be computed
        with ``X``.

    Y : array_like (float), shape=(n_samples, n_features), optional (default=None)
        The array of pandas DataFrame on which to compute 
        the kernel. If ``Y`` is None, the kernel will be computed
        with ``X``.

    sigma : float, optional (default=1.0)
        The exponential tuning parameter.

    Returns
    -------

    c : float
        The result of the kernel computation.

    References
    ----------

    Souza, Cesar R., Kernel Functions for Machine Learning Applications
    http://crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html
    """
    c = exp(_hilbert_matrix(X, Y, scalar=-1.0) / sigma)
    return c


def linear_kernel(X, Y=None, constant=0.0):
    """The ``linear_kernel`` is the simplest kernel function. It is 
    given by the inner product <x,y> plus an optional ``constant`` parameter. 
    Kernel algorithms using a linear kernel are often equivalent to their non-kernel 
    counterparts, i.e. KPCA with a ``linear_kernel`` is the same as standard PCA.

    The kernel is given by:

        :math:`k(x, y) = x^Ty + c`

    Parameters
    ----------

    X : array_like (float), shape=(n_samples, n_features)
        The array of pandas DataFrame on which to compute 
        the kernel. If ``Y`` is None, the kernel will be computed
        with ``X``.

    Y : array_like (float), shape=(n_samples, n_features), optional (default=None)
        The array of pandas DataFrame on which to compute 
        the kernel. If ``Y`` is None, the kernel will be computed
        with ``X``.

    constant : float, optional (default=0.0)
        The linear tuning parameter.

    Returns
    -------

    c : float
        The result of the kernel computation.

    References
    ----------

    Souza, Cesar R., Kernel Functions for Machine Learning Applications
    http://crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html
    """
    c = lk(X, Y) + constant
    return c


def multiquadric_kernel(X, Y=None, constant=0.0):
    """The ``multiquadric_kernel`` can be used in the same situations 
    as the Rational Quadratic kernel. As is the case with the Sigmoid kernel, 
    it is also an example of an non-positive definite kernel.

    The kernel is given by:

        :math:`k(x, y) = sqrt( -||x-y||^2 + c^2 )`

    Parameters
    ----------

    X : array_like (float), shape=(n_samples, n_features)
        The array of pandas DataFrame on which to compute 
        the kernel. If ``Y`` is None, the kernel will be computed
        with ``X``.

    Y : array_like (float), shape=(n_samples, n_features), optional (default=None)
        The array of pandas DataFrame on which to compute 
        the kernel. If ``Y`` is None, the kernel will be computed
        with ``X``.

    constant : float, optional (default=0.0)
        The linear tuning parameter.

    Returns
    -------

    c : float
        The result of the kernel computation.

    References
    ----------

    Souza, Cesar R., Kernel Functions for Machine Learning Applications
    http://crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html
    """
    hs = _hilbert_matrix(X=X, Y=Y, scalar=1.0)
    hs = np.power(hs, 2.0)
    c = np.sqrt(hs + np.power(constant, 2.0))
    return c


def polynomial_kernel(X, Y=None, alpha=1.0, degree=1.0, constant=1.0):
    """The ``polynomial_kernel`` is a non-stationary kernel. Polynomial 
    kernels are well suited for problems where all the training data is normalized.
    Adjustable parameters are the slope (``alpha``), the constant term (``constant``), 
    and the polynomial degree (``degree``).

    The kernel is given by:

        :math:`k(x, y) = ( \\alpha x^Ty + c)^d`

    Parameters
    ----------

    X : array_like (float), shape=(n_samples, n_features)
        The array of pandas DataFrame on which to compute 
        the kernel. If ``Y`` is None, the kernel will be computed
        with ``X``.

    Y : array_like (float), shape=(n_samples, n_features), optional (default=None)
        The array of pandas DataFrame on which to compute 
        the kernel. If ``Y`` is None, the kernel will be computed
        with ``X``.

    alpha : float, optional (default=1.0)
        The slope tuning parameter.

    degree : float, optional (default=1.0)
        The polynomial degree tuning parameter.

    constant : float, optional (default=1.0)
        The linear tuning parameter.

    Returns
    -------

    c : float
        The result of the kernel computation.

    References
    ----------

    Souza, Cesar R., Kernel Functions for Machine Learning Applications
    http://crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html
    """
    lc = linear_kernel(X=X, Y=Y, constant=0.0)
    c = np.power(lc * alpha + constant, degree)
    return c


def power_kernel(X, Y=None, degree=1.0):
    """The ``power_kernel`` is also known as the (unrectified) triangular kernel. 
    It is an example of scale-invariant kernel (Sahbi and Fleuret, 2004) and is 
    also only conditionally positive definite.

    The kernel is given by:

        :math:`k(x, y) = -||x-y||^d`

    Parameters
    ----------

    X : array_like (float), shape=(n_samples, n_features)
        The array of pandas DataFrame on which to compute 
        the kernel. If ``Y`` is None, the kernel will be computed
        with ``X``.

    Y : array_like (float), shape=(n_samples, n_features), optional (default=None)
        The array of pandas DataFrame on which to compute 
        the kernel. If ``Y`` is None, the kernel will be computed
        with ``X``.

    degree : float, optional (default=1.0)
        The polynomial degree tuning parameter.

    Returns
    -------

    c : float
        The result of the kernel computation.

    References
    ----------

    Souza, Cesar R., Kernel Functions for Machine Learning Applications
    http://crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html
    """
    c = -np.power(_hilbert_matrix(X, Y), degree)
    return c


def rbf_kernel(X, Y=None, sigma=1.0):
    """The ``rbf_kernel`` is closely related to the ``exponential_kernel`` and
    ``gaussian_kernel``. Note that the adjustable parameter, ``sigma``, 
    plays a major role in the performance of the kernel and should be carefully 
    tuned. If overestimated, the exponential will behave almost linearly and 
    the higher-dimensional projection will start to lose its non-linear 
    power. In the other hand, if underestimated, the function will lack regularization and 
    the decision boundary will be highly sensitive to noise in training data.

    The kernel is given by:

        :math:`k(x, y) = exp(- \\gamma * ||x-y||^2)`

    where:

        :math:`\\gamma = 1/( \\sigma ^2)`

    Parameters
    ----------

    X : array_like (float), shape=(n_samples, n_features)
        The array of pandas DataFrame on which to compute 
        the kernel. If ``Y`` is None, the kernel will be computed
        with ``X``.

    Y : array_like (float), shape=(n_samples, n_features), optional (default=None)
        The array of pandas DataFrame on which to compute 
        the kernel. If ``Y`` is None, the kernel will be computed
        with ``X``.

    sigma : float, optional (default=1.0)
        The exponential tuning parameter.

    Returns
    -------

    c : float
        The result of the kernel computation.

    References
    ----------

    Souza, Cesar R., Kernel Functions for Machine Learning Applications
    http://crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html
    """
    c = exp(_hilbert_matrix(X, Y, scalar=sigma))
    return c


def spline_kernel(X, Y=None):
    """
    The ``spline_kernel`` is given as a piece-wise cubic polynomial,
    as derived in the works by Gunn (1998).

   The kernel is given by:

        :math:`k(x, y) = 1 + xy + xy * min(x,y) - (1/2 * (x+y)) * min(x,y)^2 + 1/3 * min(x,y)^3`

    Parameters
    ----------

    X : array_like (float), shape=(n_samples, n_features)
        The array of pandas DataFrame on which to compute 
        the kernel. If ``Y`` is None, the kernel will be computed
        with ``X``.

    Y : array_like (float), shape=(n_samples, n_features), optional (default=None)
        The array of pandas DataFrame on which to compute 
        the kernel. If ``Y`` is None, the kernel will be computed
        with ``X``.

    Returns
    -------

    res : float
        The result of the kernel computation.

    References
    ----------

    Souza, Cesar R., Kernel Functions for Machine Learning Applications
    http://crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html
    """
    X, Y, res = _prep_X_Y_for_cython(X, Y)
    _spline_kernel_fast(X, Y, res)
    return res


def tanh_kernel(X, Y=None, constant=0.0, alpha=1.0):
    """The ``tanh_kernel`` (Hyperbolic Tangent Kernel) is also known as the Sigmoid 
    Kernel and as the Multilayer Perceptron (MLP) kernel. The Sigmoid Kernel comes 
    from the Neural Networks field, where the bipolar sigmoid function is often used 
    as an activation function for artificial neurons. 

    The kernel is given by:

        :math:`k(x, y) = tanh (\\alpha x^T y + c)`

    It is interesting to note that a SVM model using a sigmoid kernel function is 
    equivalent to a two-layer, perceptron neural network. This kernel was quite popular 
    for support vector machines due to its origin from neural network theory. Also, despite 
    being only conditionally positive definite, it has been found to perform well in practice.

    There are two adjustable parameters in the sigmoid kernel, the slope ``alpha`` and the 
    intercept ``constant``. A common value for alpha is 1/N, where N is the data dimension. 
    A more detailed study on sigmoid kernels can be found in the works by Hsuan-Tien and Chih-Jen.

    Parameters
    ----------

    X : array_like (float), shape=(n_samples, n_features)
        The array of pandas DataFrame on which to compute 
        the kernel. If ``Y`` is None, the kernel will be computed
        with ``X``.

    Y : array_like (float), shape=(n_samples, n_features), optional (default=None)
        The array of pandas DataFrame on which to compute 
        the kernel. If ``Y`` is None, the kernel will be computed
        with ``X``.

    constant : float, optional (default=0.0)
        The linear tuning parameter.

    alpha : float, optional (default=1.0)
        The slope tuning parameter.

    Returns
    -------

    c : float
        The result of the kernel computation.

    References
    ----------

    Souza, Cesar R., Kernel Functions for Machine Learning Applications
    http://crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html
    """
    lc = linear_kernel(X=X, Y=Y, constant=0.0)  # don't add it here
    c = np.tanh(alpha * lc + constant)  # add it here
    return c
