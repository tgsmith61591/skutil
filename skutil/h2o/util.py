from __future__ import print_function, division, absolute_import
import numpy as np
import h2o
import warnings
import pandas as pd

from ..utils import validate_is_pd, human_bytes, corr_plot
from .transform import _unq_vals_col
from .frame import _check_is_1d_frame
from .select import _validate_use
from .base import _check_is_frame

from h2o.frame import H2OFrame
from sklearn.utils.validation import check_array



__all__ = [
    'from_array',
    'from_pandas',
    'h2o_bincount',
    'h2o_col_to_numpy',
    'h2o_corr_plot',
    'h2o_frame_memory_estimate'
]


def h2o_col_to_numpy(column):
    """Return a 1d numpy array from a
    single H2OFrame column.

    Returns
    -------
    np.ndarray (1d)
    """
    x = _check_is_1d_frame(column)
    _1d = x[x.columns[0]].as_data_frame(use_pandas=True)
    return _1d[_1d.columns[0]].values


def h2o_bincount(bins, weights=None, minlength=None):
    """Given a 1d column of non-negative ints, ``bins``, return
    a np.ndarray of positional counts of each int.

    Parameters
    ----------
    bins : H2OFrame
        The values

    weights : list or H2OFrame, optional (default=None)
        The weights with which to weight the output

    minlength : int, optional (default=None)
        The min length of the output array
    """
    bins = _check_is_1d_frame(bins)
    _, unq = _unq_vals_col(bins)

    # ensure all positive
    unq_arr = h2o_col_to_numpy(unq)
    if any(unq_arr < 0):
        raise ValueError('values must be positive')

    # make sure they're all ints
    if (unq_arr.astype(np.int) - unq_arr).abs() > 0:
        raise ValueError('values must be ints')

    # check weights
    if weights is not None:
        if isinstance(weights, (list, tuple)):
            weights = np.asarray(weights)
        elif isinstance(weights, H2OFrame):
            weights = h2o_col_to_numpy(weights)

        if weights.shape[0] != all_vals.shape[0]:
            raise ValueError('dim mismatch in weights and bins')
    else:
        weights = np.ones(bins.shape[0])

    # adjust minlength
    if minlength is None:
        minlength = 1
    elif minlength < 0:
        raise ValueError('minlength must be positive')


    # create our output array
    all_vals = h2o_col_to_numpy(bins)
    output = np.zeros(np.maximum(minlength, unq_arr.max()+1))

    # update our bins
    for val in unq_arr:
        mask = all_vals == val
        output[val] = (all_vals[mask].sum() * weights[mask])

    return output


def from_pandas(X):
    """A simple wrapper for H2OFrame.from_python. This takes
    a pandas dataframe and returns an H2OFrame with all the 
    default args (generally enough) plus named columns.

    Parameters
    ----------
    X : pd.DataFrame
        The dataframe to convert.

    Returns
    -------
    H2OFrame
    """
    pd, _ = validate_is_pd(X, None)

    # if h2o hasn't started, we'll let this fail through
    return H2OFrame.from_python(X, header=0, column_names=X.columns.tolist())


def from_array(X, column_names=None):
    """A simple wrapper for H2OFrame.from_python. This takes a
    numpy array (or 2d array) and returns an H2OFrame with all 
    the default args.

    Parameters
    ----------
    X : ndarray
        The array to convert.

    column_names : list, tuple (default=None)
        the names to use for your columns

    Returns
    -------
    H2OFrame
    """
    X = check_array(X, force_all_finite=False)
    return from_pandas(pd.DataFrame.from_records(data=X, columns=column_names))


def h2o_corr_plot(X, plot_type='cor', cmap='Blues_d', n_levels=5, 
        figsize=(11,9), cmap_a=220, cmap_b=10, vmax=0.3,
        xticklabels=5, yticklabels=5, linewidths=0.5, 
        cbar_kws={'shrink':0.5}, use='complete.obs', 
        na_warn=True, na_rm=False):

    """Create a simple correlation plot given a dataframe.
    Note that this requires all datatypes to be numeric and finite!

    Parameters
    ----------
    X : pd.DataFrame
        The pandas DataFrame

    plot_type : str, optional (default='cor')
        The type of plot, one of ('cor', 'kde', 'pair')

    cmap : str, optional (default='Blues_d')
        The color to use for the kernel density estimate plot
        if plot_type == 'kde'

    n_levels : int, optional (default=5)
        The number of levels to use for the kde plot 
        if plot_type == 'kde'

    figsize : tuple (int), optional (default=(11,9))
        The size of the image

    cmap_a : int, optional (default=220)
        The colormap start point

    cmap_b : int, optional (default=10)
        The colormap end point

    vmax : float, optional (default=0.3)
        Arg for seaborn heatmap

    xticklabels : int, optional (default=5)
        The spacing for X ticks

    yticklabels : int, optional (default=5)
        The spacing for Y ticks

    linewidths : float, optional (default=0.5)
        The width of the lines

    cbar_kws : dict, optional
        Any KWs to pass to seaborn's heatmap when plot_type = 'cor'

    use : str, optional (default='complete.obs')
        The "use" to compute the correlation matrix

    na_warn : bool, optional (default=True)
        Whether to warn in the presence of NA values

    na_rm : bool, optional (default=False)
        Whether to remove NAs
    """
    X = _check_is_frame(X)
    corr = None

    if plot_type == 'cor':
        use = _validate_use(X, use, na_warn)
        cols = [str(u) for u in X.columns]

        X = X.cor(use=use, na_rm=na_rm).as_data_frame(use_pandas=True)
        X.columns = cols # set the cols to the same names
        X.index = cols
        corr = 'precomputed'

    else:
        # WARNING! This pulls everything into memory...
        X = X.as_data_frame(use_pandas=True)
    
    corr_plot(X, plot_type=plot_type, cmap=cmap, n_levels=n_levels, 
        figsize=figsize, cmap_a=cmap_a, cmap_b=cmap_b, 
        vmax=vmax, xticklabels=xticklabels, corr=corr,
        yticklabels=yticklabels, linewidths=linewidths, 
        cbar_kws=cbar_kws)



def h2o_frame_memory_estimate(X, bit_est=32, unit='MB'):
    """We estimate the memory footprint of an H2OFrame
    to determine, possibly, whether it's capable of being
    held in memory or not.

    Parameters
    ----------
    X : H2OFrame
        The H2OFrame in question

    bit_est : int, optional (default=32)
        The estimated bit-size of each cell. The default
        assumes each cell is a signed 32-bit float

    unit : str, optional (default='MB')
        The units to report. One of ('MB', 'KB', 'GB', 'TB')

    Returns
    -------
    mb : str
        The estimated number of UNIT held in the frame
    """
    X = _check_is_frame(X)

    n_samples, n_features = X.shape
    n_bits = (n_samples * n_features) * bit_est
    n_bytes = n_bits // 8

    return human_bytes(n_bytes, unit)


