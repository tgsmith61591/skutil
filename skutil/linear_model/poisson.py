from __future__ import print_function, division
import numbers, warnings
import numpy as np
from scipy import optimize, sparse
from sklearn.base import RegressorMixin
from sklearn.linear_model.base import LinearModel
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils.validation import check_X_y

__all__ = [
    'PoissonRegressor'
]

__max__ = 1e19
__min_log__ = -19


def _log_single(x):
    """Sanitized log function for a single element

    Parameters
    ----------
    x : float
        The number to log

    Returns
    -------
    val : float
        the log of x
    """
    x = max(0, x)
    val = __min_log__ if x == 0 else max(__min_log__, np.log(x))


def _log(x):
    """Sanitized log function for a vector

    Parameters
    ----------
    x : array_like, float (n_samples,)
        The numbers to log

    Returns
    -------
    array_like, float (n_samples,)
        the log of x vals
    """
    return np.array([_log_single(e) for e in x])


def _exp_single(x):
    """Sanitized exponential function

    Parameters
    ----------
    x : float
        The number to exp

    Returns
    -------
    float
        the exp of x
    """
    return min(__max__, np.exp(x))


def _exp(x):
    """Sanitized exp function for a vector

    Parameters
    ----------
    x : array_like, float (n_samples,)
        The numbers to exp

    Returns
    -------
    array_like, float (n_samples,)
        the exp of x vals
    """
    return np.array([_exp_single(e) for e in x])


def _exposure_or_zeros(exposure, n_obs):
    """Return exposure or a vector of zeros

    Parameters
    ----------
    exposure : array_like, float (n_samples,)
        Might be None.

    n_obs : int
        The number of observations
    """
    return exposure if not exposure is None else np.zeros(n_obs)


def _poisson_loss(w, X, y, exposure, alpha):
    """Computes the Poisson loss.

    Parameters
    ----------
    w : ndarray, shape (n_features,) or (n_features + 1,)
        Coefficient vector.

    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.

    y : ndarray, shape (n_samples,)
        Array of outcome variables.

    exposure : ndarray, shape (n_samples,)
        Array of exposure variables.

    alpha : float
        Regularization parameter. alpha is equal to 1 / C.

    Returns
    -------
    out : float
        Poisson loss.
    """
    # Poisson loss:
    # -\sum_{i=1}^{n} y_i * w * x_i - exp(w * x_i + ln(exposure))
    Xw = np.dot(X, w)
    n_obs = len(y)

    # the back portion of the equation
    z = np.exp(Xw + exposure)

    out = -np.sum(y * Xw - z) + .5 * alpha * np.dot(w, w)
    return out


def _poisson_grad(w, X, y, exposure, alpha):
    """Computes the Poisson loss.
    Parameters
    ----------
    w : ndarray, shape (n_features,) or (n_features + 1,)
        Coefficient vector.

    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.

    y : ndarray, shape (n_samples,)
        Array of outcome variables.

    exposure : ndarray, shape (n_samples,)
        Array of exposure variables.

    alpha : float
        Regularization parameter. alpha is equal to 1 / C.

    Returns
    -------
    out : float
        Poisson loss.
    """
    n_samples, n_features = X.shape
    n_obs = len(y)

    # Grad of Poisson loss:
    # -\sum_{i=1}^{n} (y_i - exp(w*x_i + ln(exposure)))*x_i
    Xw = np.dot(X, w)

    z = np.exp(Xw + exposure)

    grad = -np.dot(y - z, X)
    return grad



OPTIMIZERS = {
    'cg'     : {
        'opt'    : optimize.fmin_cg,
        'fprime' : None,
        'tol_kw' : 'gtol',
        'verbose': 'full_output'
    },

    'l_bfgs' : {
        'opt'    : optimize.fmin_l_bfgs_b,
        'fprime' : _poisson_grad,
        'tol_kw' : 'pgtol',
        'verbose': 'iprint'
    },

    'powell' : {
        'opt'    : optimize.fmin_powell,
        'fprime' : None,
        'tol_kw' : 'ftol',
        'verbose': 'full_output'
    }
}


def poisson_regression(X, y, exposure=None, alpha=None, solver='cg', max_iter=1000, 
                       tol=1e-4, verbose=0, fit_intercept=False):
    """Perform Poisson regression.

    Parameters
    ----------
    X : array_like, (n_samples, n_features)
        The training data

    y : array_like, (n_samples,)
        The training response

    exposure : array_like, (n_samples,), optional (default None)
        The exposure vector.

    alpha : float, optional (default None)
        The alpha

    solver : str, optional (default 'cg')
        One of ('cg', 'l_bfgs', 'powell')

    max_iter : int (default 1000)
        The max iters for the optimizer

    tol : float (default 1e-4)
        The tolerance

    verbose : int (default 0)
        Whether to be loud

    fit_intercept : boolean (default False)
        Whether to fit the intercept
    """
    # SAG needs X and y columns to be C-contiguous and np.float64
    X = check_array(X, dtype=np.float64)
    y = check_array(y, dtype='numeric', ensure_2d=False)


    # ensure shapes match
    check_consistent_length(X, y)
    n_samples, n_features = X.shape
    if y.ndim > 2:
        raise ValueError("Target y has the wrong shape %s" % str(y.shape))

    # validate exposure
    if exposure is not None:
        exposure = check_array(exposure, dtype='numeric', ensure_2d=False)
        check_consistent_length(X, exposure)
        if exposure.ndim > 2:
            raise ValueError("Target exposure has the wrong shape %s" % str(exposure.shape))

        exposure = _log(exposure)
    else:
        exposure = _exposure_or_zeros(None, n_samples)

    # validate alpha
    if alpha is None:
        alpha = 1 / n_samples

    # validate solver
    if not solver in OPTIMIZERS:
        raise ValueError('%s is not a valid solver' % solver)

    # validate intercept
    if not isinstance(fit_intercept, bool):
        raise ValueError('fit_intercept must be a boolean but encountered %s' % type(fit_intercept))

    # create intercept if needed
    w0 = np.zeros(n_features + int(fit_intercept))

    # validate max_iter
    if not isinstance(max_iter, numbers.Number) or max_iter < 0:
        raise ValueError("Maximum number of iteration must be positive; got (max_iter=%r)" % max_iter)

    # validate tol
    if not isinstance(tol, numbers.Number) or tol < 0:
        raise ValueError("Tolerance for stopping criteria must be positive; got (tol=%r)" % tol)

    # get the optimizer dict
    optimizer_dict = OPTIMIZERS[solver]
    optimizer = optimizer_dict['opt']
    optkwargs = {
        'fprime' : optimizer_dict['fprime'],
        optimizer_dict['tol_kw'] : tol,
        optimizer_dict['verbose'] : (verbose > 0) - 1
    }

    result = optimizer(_poisson_loss, w0, 
        args=(X, y, exposure, alpha),
        maxiter=max_iter,
        **optkwargs) # provides fprime, tolerance and verbose

    return w0


class PoissonRegressor(LinearModel, RegressorMixin):
    """Poisson Regression.

    exposure : array_like, (n_samples,), optional (default None)
        The exposure vector.

    alpha : float
        Small positive values of alpha improve the conditioning of the problem
        and reduce the variance of the estimates.  Alpha corresponds to
        ``C^-1`` in other linear models such as LogisticRegression or
        LinearSVC.

    solver : {'cg', 'lbfgs', 'powell'}
        Solver to use in the computational routines

    max_iter : int, optional
        Maximum number of iterations for conjugate gradient solver.
        The default value is determined by scipy.sparse.linalg.

    tol : float
        Precision of the solution.

    verbose : int (default 0)
        Whether to be loud

    fit_intercept : boolean (default False)
        Whether to fit the intercept
    """

    def __init__(self, exposure=None, alpha=0., solver='cg', max_iter=1000, 
        tol=1e-4, verbose=0, fit_intercept=False):

    	# set numpy stderr low, because can overflow on log/exp
    	np.seterr(all='ignore')

    	self.exposure = exposure
        self.alpha = alpha
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        """Fit the model according to the given training data.
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self : returns an instance of self.
        """
        X, y = check_X_y(X, y, 
            accept_sparse='csr', 
            dtype=np.float64,
            order="C")

        n_samples, n_features = X.shape
        coeff = poisson_regression(X, y, exposure=self.exposure, alpha=self.alpha,
                    solver=self.solver, max_iter=self.max_iter, tol=self.tol,
                    verbose=self.verbose, fit_intercept=self.fit_intercept)

        self.coef_ = coeff
        if self.fit_intercept:
            self.intercept_ = self.coef_[-1]
            self.coef_ = self.coef_[:-1]
        else:
        	self.intercept_ = 0

        return self