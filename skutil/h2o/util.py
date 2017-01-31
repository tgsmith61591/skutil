from __future__ import print_function, division, absolute_import
import numpy as np
import h2o
import pandas as pd

import warnings
from collections import Counter
from pkg_resources import parse_version
from ..utils import (validate_is_pd, human_bytes, corr_plot,
                     load_breast_cancer_df, load_iris_df,
                     load_boston_df)
from .frame import _check_is_1d_frame
from .select import _validate_use
from .base import check_frame
from .fixes import rbind_all

from h2o.frame import H2OFrame
from sklearn.utils.validation import check_array

__all__ = [
    'from_array',
    'from_pandas',
    'h2o_bincount',
    'h2o_col_to_numpy',
    'h2o_corr_plot',
    'h2o_frame_memory_estimate',
    'load_iris_h2o',
    'load_boston_h2o',
    'load_breast_cancer_h2o',
    'reorder_h2o_frame',
    'shuffle_h2o_frame'
]


def load_iris_h2o(include_tgt=True, tgt_name="Species", shuffle=False):
    """Load the iris dataset into an H2OFrame

    Parameters
    ----------

    include_tgt : bool, optional (default=True)
        Whether or not to include the target

    tgt_name : str, optional (default="Species")
        The name of the target column.

    shuffle : bool, optional (default=False)
        Whether or not to shuffle the data
    """

    X = from_pandas(load_iris_df(include_tgt, tgt_name, shuffle))
    if include_tgt:
        X[tgt_name] = X[tgt_name].asfactor()

    return X


def load_breast_cancer_h2o(include_tgt=True, tgt_name="target", shuffle=False):
    """Load the breast cancer dataset into an H2OFrame

    Parameters
    ----------

    include_tgt : bool, optional (default=True)
        Whether or not to include the target

    tgt_name : str, optional (default="target")
        The name of the target column.

    shuffle : bool, optional (default=False)
        Whether or not to shuffle the data

    """
    X = from_pandas(load_breast_cancer_df(include_tgt, tgt_name, shuffle))
    if include_tgt:
        X[tgt_name] = X[tgt_name].asfactor()

    return X


def load_boston_h2o(include_tgt=True, tgt_name="target", shuffle=False):
    """Load the boston housing dataset into an H2OFrame


    Parameters
    ----------

    include_tgt : bool, optional (default=True)
        Whether or not to include the target

    tgt_name : str, optional (default="target")
        The name of the target column.

    shuffle : bool, optional (default=False)
        Whether or not to shuffle the data

    """
    X = from_pandas(load_boston_df(include_tgt, tgt_name, shuffle))
    return X


def h2o_col_to_numpy(column):
    """Return a 1d numpy array from a
    single H2OFrame column.

    Parameters
    ----------

    column : H2OFrame column, shape=(n_samples, 1)
        A column from an H2OFrame

    Returns
    -------

    np.ndarray, shape=(n_samples,)
    """
    x = _check_is_1d_frame(column)
    _1d = x[x.columns[0]].as_data_frame(use_pandas=True)
    return _1d[_1d.columns[0]].values


def _unq_vals_col(column):
    """Get the unique values and column name
    from a column.

    Returns
    -------

    str, np.ndarray : tuple
        (c1_nm, unq)
    """
    unq = column.unique().as_data_frame(use_pandas=True)
    c1_nm = unq.columns[0]
    unq = unq[unq.columns[0]].sort_values().reset_index()

    return c1_nm, unq


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
    unq_arr = unq[_].values
    if any(unq_arr < 0):
        raise ValueError('values must be positive')

    # make sure they're all ints
    if np.abs((unq_arr.astype(np.int) - unq_arr).sum()) > 0:
        raise ValueError('values must be ints')

    # adjust minlength
    if minlength is None:
        minlength = 1
    elif minlength < 0:
        raise ValueError('minlength must be positive')

    # create our output array
    all_vals = h2o_col_to_numpy(bins)
    output = np.zeros(np.maximum(minlength, unq_arr.max() + 1))

    # check weights
    if weights is not None:
        if isinstance(weights, (list, tuple)):
            weights = np.asarray(weights)
        elif isinstance(weights, H2OFrame):
            weights = h2o_col_to_numpy(weights)

        if weights.shape[0] != all_vals.shape[0]:
            raise ValueError('dim mismatch in weights and bins')
    else:
        weights = np.ones(all_vals.shape[0])

    # update our bins
    for val in unq_arr:
        mask = all_vals == val
        array_ones = np.ones(mask.sum())
        weight_vals = weights[mask]
        output[val] = np.dot(array_ones, weight_vals)

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

    # older version of h2o are super funky with this
    if parse_version(h2o.__version__) < parse_version('3.10.0.7'):
        h = 1
    else:
        h = 0

    # if h2o hasn't started, we'll let this fail through
    return H2OFrame.from_python(X, header=h, column_names=X.columns.tolist())


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
                  figsize=(11, 9), cmap_a=220, cmap_b=10, vmax=0.3,
                  xticklabels=5, yticklabels=5, linewidths=0.5,
                  cbar_kws={'shrink': 0.5}, use='complete.obs',
                  na_warn=True, na_rm=False):
    """Create a simple correlation plot given a dataframe.
    Note that this requires all datatypes to be numeric and finite!

    Parameters
    ----------

    X : H2OFrame, shape=(n_samples, n_features)
        The H2OFrame

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
    X = check_frame(X, copy=False)
    corr = None

    if plot_type == 'cor':
        use = _validate_use(X, use, na_warn)
        cols = [str(u) for u in X.columns]

        X = X.cor(use=use, na_rm=na_rm).as_data_frame(use_pandas=True)
        X.columns = cols  # set the cols to the same names
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
    X = check_frame(X, copy=False)

    n_samples, n_features = X.shape
    n_bits = (n_samples * n_features) * bit_est
    n_bytes = n_bits // 8

    return human_bytes(n_bytes, unit)


def _gen_optimized_chunks(idcs):
    """Given the list of indices, create more efficient chunks to minimize
    the number of rbind operations required for the H2OFrame ExprNode cache.
    """
    idcs = sorted(idcs)
    counter = Counter(idcs)
    counts = counter.most_common()  # order desc

    # the first index is the number of chunks we'll need to create.
    n_chunks = counts[0][1]
    chunks = [[] for _ in range(n_chunks)]  # gen the number of chunks we'll need

    # 1. populate the chunks each with their first idx (the most common)
    # 2. pop from the counter
    # 3. re-generate the most_common(), repeat
    while counts:
        val, n_iter = counts[0]  # the one at the head of the list is the most common
        for i in range(n_iter):
            chunks[i].append(val)
        counts.pop(0)  # pop out the first idx...
    # sort them
    return [sorted(chunk) for chunk in chunks]


def reorder_h2o_frame(X, idcs):
    """Currently, H2O does not allow us to reorder
    frames. This is a hack to rbind rows together in the
    order prescribed.

    Parameters
    ----------

    X : H2OFrame
        The H2OFrame to reorder

    idcs : iterable
        The order of the H2OFrame rows to be returned.

    Returns
    -------

    new_frame : H2OFrame
        The reordered H2OFrame
    """
    # hack... slow but functional
    X = check_frame(X, copy=False)  # we're rbinding. no need to copy

    # to prevent rbinding rows over, and over, and over
    # create chunks. Rbind chunks that are progressively increasing.
    # once we hit an index that decreases, rbind, and then start the next chunk
    last_index = np.inf
    chunks = []  # all of the chunks
    chunk = []  # the current chunk being built

    for i in idcs:
        # if it's a chunk from balancer:
        if hasattr(i, '__iter__'):  # probably a list of indices
            chunks.append(X[i, :])

        # otherwise chunks have not been computed
        else:
            # while the indices increase adjacently
            if i < last_index:
                last_index = i
                chunk.append(i)

            # otherwise, they are no longer increasing
            else:
                # if a chunk exists
                if chunk:  # there should ALWAYS be a chunk
                    rows = X[chunk, :]
                else:
                    rows = X[i, :]

                # append the chunk and reset the list
                chunks.append(rows)
                chunk = []
                last_index = i

    # print([type(c) for c in chunks])  # couldn't figure out an issue for a while...
    return chunks[0] if len(chunks) == 1 else rbind_all(*chunks)


def shuffle_h2o_frame(X):
    """Currently, H2O does not allow us to shuffle 
    frames. This is a hack to rbind rows together in the
    order prescribed.

    Parameters
    ----------

    X : H2OFrame
        The H2OFrame to reorder

    Returns
    -------

    shuf : H2OFrame
        The shuffled H2OFrame
    """
    warnings.warn('Shuffling H2O frames will eventually be deprecated, as H2O '
                  'does not allow re-ordering of frames by row. The current work-around '
                  '(rbinding the rows) is known to cause issues in the H2O ExprNode '
                  'cache for very large frames.', DeprecationWarning)

    X = check_frame(X, copy=False)
    idcs = np.random.permutation(np.arange(X.shape[0]))
    shuf = reorder_h2o_frame(X, idcs)  # do not generate optimized chunks here...
    return shuf
