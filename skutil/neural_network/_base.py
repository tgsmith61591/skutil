"""Utilities for the neural network modules
"""

# Part of the sklearn.neural_network module (which is
# not visible to import, so it's been copied here)
# Author: Issam H. Laradji <issam.laradji@gmail.com>
# License: BSD 3 clause

import numpy as np
from sklearn.utils.fixes import expit as logistic_sigmoid


def identity(X):
    """Simply return the input array.
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Data, where n_samples is the number of samples
        and n_features is the number of features.
    Returns
    -------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Same as the input data.
    """
    return X


def logistic(X):
    """Compute the logistic function inplace.
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.
    Returns
    -------
    X_new : {array-like, sparse matrix}, shape (n_samples, n_features)
        The transformed data.
    """
    return logistic_sigmoid(X, out=X)


def tanh(X):
    """Compute the hyperbolic tan function inplace.
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.
    Returns
    -------
    X_new : {array-like, sparse matrix}, shape (n_samples, n_features)
        The transformed data.
    """
    return np.tanh(X, out=X)


def relu(X):
    """Compute the rectified linear unit function inplace.
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.
    Returns
    -------
    X_new : {array-like, sparse matrix}, shape (n_samples, n_features)
        The transformed data.
    """
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X


def softmax(X):
    """Compute the K-way softmax function inplace.
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.
    Returns
    -------
    X_new : {array-like, sparse matrix}, shape (n_samples, n_features)
        The transformed data.
    """
    tmp = X - X.max(axis=1)[:, np.newaxis]
    np.exp(tmp, out=X)
    X /= X.sum(axis=1)[:, np.newaxis]

    return X


ACTIVATIONS = {
	'identity': identity, 
	'tanh': tanh, 
	'logistic': logistic,
	'relu': relu, 
	'softmax': softmax
}