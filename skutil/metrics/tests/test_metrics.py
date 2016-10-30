from skutil.metrics import *
import numpy as np
import timeit
from skutil.metrics.kernel import (_hilbert_dot,
                                   _hilbert_matrix)
from skutil.metrics import GainsStatisticalReport
from skutil.utils.tests.utils import assert_fails
from numpy.testing import (assert_array_equal, assert_almost_equal, assert_array_almost_equal)

sigma = 0.05


def _get_train_array():
    return np.array([
        [0., 1.],
        [2., 3.],
        [2., 4.]
    ])


# this is the transpose of the train array:
# [0., 2., 2.]
# [1., 3., 4.]

def test_linear_kernel():
    X = np.reshape(np.arange(1, 13), (4, 3))
    assert_array_equal(linear_kernel(X=X),
                       np.array([[14., 32., 50., 68.],
                                 [32., 77., 122., 167.],
                                 [50., 122., 194., 266.],
                                 [68., 167., 266., 365.]]))


def test_poly_kernel():
    X = _get_train_array()
    assert_array_equal(polynomial_kernel(X),
                       np.array([[2, 4, 5],
                                 [4, 14, 17],
                                 [5, 17, 21]]))


def test_power_kernel():
    X = _get_train_array()
    assert_array_almost_equal(power_kernel(X=X, degree=2.0),
                              np.array([
                                  [0.0, -64.0, -169],
                                  [-64.0, 0.0, -1.0],
                                  [-169.0, -1.0, 0.0]
                              ]))


def test_hilbert():
    X = np.array([10.0, 2.0, 3.0, 4.0])
    Y = np.array([5.0, 6.0, 7.0, 8.0])

    answ = _hilbert_dot(X, Y)
    assert answ == -73.0, 'expected -73.0 but got %.3f' % answ

    Z = np.array([X, Y])
    answ = _hilbert_matrix(Z)
    assert_array_equal(answ, np.array([
        [0, -73],
        [-73, 0]
    ]))


def test_exp():
    X = _get_train_array()
    answ = exponential_kernel(X)
    assert_array_almost_equal(answ, np.array([
        [1., 54.59815003, 665.14163304],
        [54.59815003, 1., 1.64872127],
        [665.14163304, 1.64872127, 1.]]))


def test_laplace():
    X = _get_train_array()
    answ = laplace_kernel(X)
    assert_array_almost_equal(answ, np.array([
        [1.00000000e+00, 2.98095799e+03, 4.42413392e+05],
        [2.98095799e+03, 1.00000000e+00, 2.71828183e+00],
        [4.42413392e+05, 2.71828183e+00, 1.00000000e+00]]), 4)


def test_inverse_multiquadric():
    X = _get_train_array()
    answ = inverse_multiquadric_kernel(X)
    assert_array_almost_equal(answ, np.array([
        [1., 0.12403473, 0.0766965],
        [0.12403473, 1., 0.70710678],
        [0.0766965, 0.70710678, 1.]]))


def test_gaussian_kernel():
    X = _get_train_array()
    answ = gaussian_kernel(X)
    assert_array_almost_equal(answ, np.array([
        [1, 0, 0],
        [0, 1, 6.065307e-01],
        [0, 6.065307e-01, 1]
    ]))


def test_multiquadric():
    X = _get_train_array()
    answ = multiquadric_kernel(X)
    assert_array_equal(answ, np.array([
        [0., 8., 13.],
        [8., 0., 1.],
        [13., 1., 0.]
    ]))


def test_rbf():
    X = _get_train_array()
    answ = rbf_kernel(X, sigma=sigma)
    assert_array_almost_equal(answ, np.array([
        [1., 0.67032004, 0.52204577],
        [0.67032004, 1., 0.95122942],
        [0.52204577, 0.95122942, 1.]]))


def test_spline_kernel():
    X = _get_train_array()
    answ = spline_kernel(X)
    assert_array_almost_equal(answ, np.array([
        [3.83333333, 5.33333333, 3.83333333],
        [11.66666667, 72.83333333, 89.44444444],
        [15.66666667, 101.58333333, 120.11111111]]))


def test_tanh():
    X = _get_train_array()
    answ = tanh_kernel(X)
    assert_array_almost_equal(answ, np.array([
        [0.76159416, 0.99505475, 0.9993293],
        [0.99505475, 1., 1.],
        [0.9993293, 1., 1.]]))


def test_act_stats():
    pred = [0.0, 1.0, 1.5]
    loss = [0.5, 0.5, 1.0]
    expo = [1.0, 0.5, 1.0]

    a = GainsStatisticalReport().fit_fold(pred=pred, expo=expo, loss=loss)
    # now see if we can get one to fail...
    assert_fails(a.fit_fold, TypeError, **{'pred': pred, 'expo': expo, 'loss': loss, 'prem': 12})

    # this one will work:
    a.fit_fold(pred=pred, expo=expo, loss=loss, prem=[1.0, 1.0, 1.0])

    # initializing with a bad 'score_by' will fail
    assert_fails(GainsStatisticalReport, ValueError, **{'score_by': 'accuracy'})

    # purposefully set n_folds and not set n_iter
    assert_fails(GainsStatisticalReport, ValueError, **{'n_folds': 10})

    # purposefully set wrong error_behavior
    assert_fails(GainsStatisticalReport.fit_fold,
                 ValueError,
                 **{'error_behavior': '', 'pred': pred, 'expo': expo, 'loss': loss})

    # purposefully set n_folds so that n_obs is not be divisible by n_folds and n_iter
    assert_fails(GainsStatisticalReport.as_data_frame, ValueError,
                 **{'n_folds': 121, 'n_iter': 111})

    # assert this is two in length...
    d = a.as_data_frame()
    assert d.shape[0] == 2
