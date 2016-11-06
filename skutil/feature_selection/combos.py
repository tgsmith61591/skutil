from __future__ import division, print_function
import numpy as np
from sklearn.externals import six
from sklearn.utils.validation import check_is_fitted
from skutil.odr import QRDecomposition
from .base import _BaseFeatureSelector
from .select import _validate_cols
from ..utils import flatten_all, validate_is_pd


__all__ = [
    'LinearCombinationFilterer'
]


class LinearCombinationFilterer(_BaseFeatureSelector):
    """The ``LinearCombinationFilterer will resolve linear combinations in a numeric matrix. 
    The QR decomposition is used to determine whether the matrix is full rank, and then 
    identify the sets of columns that are involved in the dependencies. This class is adapted 
    from the implementation in the R package, caret.

    Parameters
    ----------

    cols : array_like, shape=(n_features,), optional (default=None)
        The names of the columns on which to apply the transformation.
        If no column names are provided, the transformer will be ``fit``
        on the entire frame. Note that the transformation will also only
        apply to the specified columns, and any other non-specified
        columns will still be present after transformation. Note that 
        since this transformer can only operate on numeric columns, not 
        explicitly setting the ``cols`` parameter may result in errors 
        for categorical data.

    as_df : bool, optional (default=True)
        Whether to return a Pandas ``DataFrame`` in the ``transform``
        method. If False, will return a Numpy ``ndarray`` instead. 
        Since most skutil transformers depend on explicitly-named
        ``DataFrame`` features, the ``as_df`` parameter is True by default.


    Examples
    --------

        >>> from skutil.utils import load_iris_df
        >>>
        >>> X = load_iris_df(include_tgt=False)
        >>> filterer = LinearCombinationFilterer()
        >>> X_transform = filterer.fit_transform(X)
        >>> assert X_transform.shape[1] == 4 # no combos


    Attributes
    ----------

    drop_ : array_like, shape=(n_features,)
        Assigned after calling ``fit``. These are the features that
        are designated as "bad" and will be dropped in the ``transform``
        method.
    """

    def __init__(self, cols=None, as_df=True):
        super(LinearCombinationFilterer, self).__init__(cols=cols, as_df=as_df)

    def fit(self, X, y=None):
        """Fit the linear combination filterer.

        Parameters
        ----------

        X : Pandas DataFrame
            The Pandas frame to fit. The frame will only
            be fit on the prescribed ``cols`` (see ``__init__``) or
            all of them if ``cols`` is None.

        y : None
            Passthrough for ``sklearn.pipeline.Pipeline``. Even
            if explicitly set, will not change behavior of ``fit``.

        Returns
        -------

        self
        """
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y=None):
        """Fit the linear combination filterer and return
        the transformed matrix or DataFrame.

        Parameters
        ----------

        X : Pandas DataFrame
            The Pandas frame to fit. The frame will only
            be fit on the prescribed ``cols`` (see ``__init__``) or
            all of them if ``cols`` is None.

        y : None
            Passthrough for ``sklearn.pipeline.Pipeline``. Even
            if explicitly set, will not change behavior of ``fit``.

        Returns
        -------

        dropped : Pandas DataFrame or NumPy ndarray
            The training frame sans "bad" columns
        """

        # check on state of X and cols
        X, self.cols = validate_is_pd(X, self.cols, assert_all_finite=True)
        _validate_cols(self.cols)

        # init drops list
        drops = []

        # Generate sub matrix for qr decomposition
        cols = [n for n in (self.cols if self.cols is not None else X.columns)]  # get a copy of the cols
        x = X[cols].as_matrix()
        cols = np.array(cols)  # so we can do boolean indexing

        # do subroutines
        lc_list = _enumLC(QRDecomposition(x))

        if lc_list is not None:
            while lc_list is not None:
                # we want the first index in each of the keys in the dict
                bad = np.array([p for p in set([v[0] for _, v in six.iteritems(lc_list)])])

                # get the corresponding bad names
                bad_nms = cols[bad]
                drops.extend(bad_nms)

                # update our X, and then our cols
                x = np.delete(x, bad, axis=1)
                cols = np.delete(cols, bad)

                # keep removing linear dependencies until it resolves
                lc_list = _enumLC(QRDecomposition(x))

                # will break when lc_list returns None

        # Assign attributes, return
        self.drop_ = [p for p in set(drops)] # a list from the a set of the drops
        dropped = X.drop(self.drop_, axis=1)

        return dropped if self.as_df else dropped.as_matrix()

    def transform(self, X):
        """Drops the linear combination features from the new
        input frame.

        Parameters
        ----------

        X : Pandas DataFrame
            The Pandas frame to transform.

        Returns
        -------

        dropped : Pandas DataFrame or NumPy ndarray
            The test frame sans "bad" columns
        """
        check_is_fitted(self, 'drop_')
        # check on state of X and cols
        X, _ = validate_is_pd(X, self.cols)

        dropped = X.drop(self.drop_, axis=1)
        return dropped if self.as_df else dropped.as_matrix()
        

def _enumLC(decomp):
    """Perform a single iteration of linear combo scoping.

    Parameters
    ----------

    decomp : a QRDecomposition object
        The QR decomposition of the matrix
    """
    qr = decomp.qr  # the decomposition matrix

    # extract the R matrix
    R = decomp.get_R()         # the R matrix
    n_features = R.shape[1]    # number of columns in R
    is_zero = n_features == 0  # whether there are no features
    rank = decomp.get_rank()   # the rank of the original matrix, or num of independent cols

    if not (rank == n_features):
        pivot = decomp.pivot        # the pivot vector
        X = R[:rank, :rank]         # extract the independent cols
        Y = R[:rank, rank:]  # +1?  # extract the dependent columns

        new_qr = QRDecomposition(X) # factor the independent columns
        b = new_qr.get_coef(Y)      # get regression coefficients of dependent cols

        # if b is None, then there were no dependent columns
        if b is not None:
            b[np.abs(b) < 1e-6] = 0  # zap small values
            
            # will return a dict of {dim : list of bad idcs}
            d = {}
            row_idcs = np.arange(b.shape[0])
            for i in range(Y.shape[1]):  # should only ever be 1, right?
                nested = [ 
                            pivot[rank+i],
                            pivot[row_idcs[b[:, i] != 0]]
                         ]
                d[i] = flatten_all(nested)

            return d

    # if we get here, there are no linear combos to discover
    return None
