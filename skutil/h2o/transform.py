from __future__ import print_function, division, absolute_import
from .base import BaseH2OTransformer, _frame_from_x_y, check_frame
from ..utils import is_numeric, flatten_all
from .frame import _check_is_1d_frame
from .util import h2o_col_to_numpy, _unq_vals_col
from ..utils.fixes import is_iterable, dict_values
from ..preprocessing import ImputerMixin
from sklearn.externals import six
import pandas as pd
from sklearn.utils.validation import check_is_fitted

__all__ = [
    'H2OInteractionTermTransformer',
    'H2OSelectiveImputer',
    'H2OSelectiveScaler'
]


def _flatten_one(x):
    """There is a bug in some versions of h2o
    where a scalar is not returned by mean, but
    a list is. This will determine the proper 
    type for each item in the vec.
    """
    return x[0] if is_iterable(x) else x


def _transform_col(col, val):
    """If an imputation value does not match column
    type, we'll get some errors. So this is going to manipulate
    the column type based on the value type. This is
    necessary as opposed to the opposite way because an 
    int column might still have a 'mean' fill. Thus, we'll
    just treat everything as float.
    """
    if dict_values(col.types)[0] in ('int', 'real'):
        return col.asnumeric(), float(val)

    # for enums, character, etc...
    return col, val


class _H2OBaseImputer(BaseH2OTransformer, ImputerMixin):
    """A base class for all H2O imputers"""

    def __init__(self, feature_names=None, target_feature=None, exclude_features=None,
                 min_version='any', max_version=None, def_fill=None):
        super(_H2OBaseImputer, self).__init__(feature_names=feature_names,
                                              target_feature=target_feature,
                                              exclude_features=exclude_features,
                                              min_version=min_version,
                                              max_version=max_version)
        self.fill_ = self._def_fill if def_fill is None else def_fill


def _mode(x, def_fill=ImputerMixin._def_fill):
    """Get the most common value in a 1d
    H2OFrame. Ties will be handled in a non-specified
    manner.

    Parameters
    ----------

    x : ``H2OFrame``, shape=(n_samples, 1)
        The 1d frame from which to derive the mode
    """
    idx = x.as_data_frame(use_pandas=True)[x.columns[0]].value_counts().index

    # if the most common is null, then return the next most common.
    # if there is no next common (i.e., 100% null) then we return the def_fill
    return idx[0] if not pd.isnull(idx[0]) else idx[1] if idx.shape[0] > 1 else def_fill


class H2OSelectiveImputer(_H2OBaseImputer):
    """The selective imputer provides extreme flexibility and simplicity
    in imputation tasks. Rather than imposing one strategy across an entire
    frame, different strategies can be mapped to respective features.

    Parameters
    ----------

    feature_names : array_like (str), optional (default=None)
        The list of names on which to fit the transformer.

    target_feature : str, optional (default None)
        The name of the target feature (is excluded from the fit)
        for the estimator.

    exclude_features : iterable or None, optional (default=None)
        Any names that should be excluded from ``feature_names``

    def_fill : str, int or iterable, optional (default='mean')
        The fill strategy. If an int, the int value will be applied
        to all missing values in the H2OFrame. If a string, must be one of
        ('mean', 'median', 'mode') - note that 'mode' is still under
        development. If an iterable (list, tuple, array, etc.), the length must
        match the column dimensions. However, if a dict, the strategies
        will be applied to the mapped columns.

    Attributes
    ----------

    fill_val_ : int, float or iterable
        The fill value(s) provided or 
        derived in the ``fit`` method.
    """

    _min_version = '3.8.2.9'
    _max_version = None

    def __init__(self, feature_names=None, target_feature=None, exclude_features=None, def_fill='mean'):
        super(H2OSelectiveImputer, self).__init__(feature_names=feature_names,
                                                  target_feature=target_feature,
                                                  exclude_features=exclude_features,
                                                  min_version=self._min_version,
                                                  max_version=self._max_version,
                                                  def_fill=def_fill)

    def fit(self, X):
        """Fit the imputer.

        Parameters
        ----------

        X : H2OFrame, shape=(n_samples, n_features)
            The training data on which to fit.

        Returns
        -------

        self
        """
        X = check_frame(X, copy=False)
        frame = _frame_from_x_y(X, self.feature_names, self.target_feature, self.exclude_features)

        # at this point, the entirety of frame can be operated on...
        cols = [str(u) for u in frame.columns]  # convert to string...

        # validate the fill, do fit
        fill = self.fill_
        if isinstance(fill, six.string_types):
            fill = str(fill)
            if not fill in ('mode', 'mean', 'median'):
                raise TypeError('self.fill must be either "mode", "mean", "median", None, '
                                'a number, or an iterable. Got %s' % fill)

            if fill == 'mode':
                # for each column to impute, we go through and get the value counts
                # of each, sorting by the max...
                self.fill_val_ = dict(zip(cols, [_mode(X[c]) for c in cols]))

            elif fill == 'median':
                self.fill_val_ = dict(zip(cols, flatten_all([X[c].median(na_rm=True) for c in cols])))

            else:
                self.fill_val_ = dict(zip(cols, flatten_all([X[c].mean(na_rm=True) for c in cols])))

        elif is_iterable(fill):

            # if fill is a dictionary
            if isinstance(fill, dict):
                # if it's a dict, we can assume that these are the cols...
                cols, fill = zip(*fill.items())

            # we need to get the length of the iterable,
            # make sure it matches the len of cols
            if not len(fill) == len(cols):
                raise ValueError('len of fill does not match that of cols')

            # make sure they're all ints
            if not all(
                    [(is_numeric(i) or (isinstance(i, six.string_types)) and i in ('mode', 'mean', 'median')) for i in
                     fill]):
                raise TypeError('All values in self.fill must be numeric or in ("mode", "mean", "median"). '
                                'Got: %s' % ', '.join(fill))

            d = {}
            for ind, c in enumerate(cols):
                f = fill[ind]

                if is_numeric(f):  # if we fill with a single value...
                    d[c] = f
                else:
                    the_col = X[c]
                    if f == 'mode':
                        d[c] = _mode(the_col)
                        # d[c] = _col_mode(the_col)
                    elif f == 'median':
                        d[c] = _flatten_one(the_col.median(na_rm=True))
                    else:
                        d[c] = _flatten_one(the_col.mean(na_rm=True))

            self.fill_val_ = d

        else:
            if not is_numeric(fill):
                raise TypeError('self.fill must be either "mode", "mean", "median", None, '
                                'a number, or an iterable. Got %s' % str(fill))

            # either the fill is an int, or it's something the user provided...
            # if it's not an int or float, we'll let it go and not catch it because
            # the it's their fault they were dumb.
            self.fill_val_ = fill

        return self

    def transform(self, X):
        """Transform an H2OFrame given the fit imputer.

        Parameters
        ----------

        X : H2OFrame, shape=(n_samples, n_features)
            The test data to transform.

        Returns
        -------
        X : H2OFrame, shape=(n_samples, n_features)
            The transformed (imputed) test data.
        """
        check_is_fitted(self, 'fill_val_')
        X = check_frame(X, copy=True) # make a copy

        # get the fills
        fill_val = self.fill_val_

        # we get the subset frame just to retrieve the column names. We affect
        # X in place anyways, so no use using the slice...
        frame = _frame_from_x_y(X, self.feature_names, self.target_feature)
        cols = [str(u) for u in frame.columns]  # the cols we'll ultimately impute
        X_columns = [str(u) for u in X.columns]  # used for index lookup

        # get the frame of NAs
        na_frame = frame.isna()
        na_frame.columns = cols

        # iter over cols
        is_int = isinstance(fill_val, int)  # is it an int?
        for _, col in enumerate(cols):
            if not is_int and col not in fill_val:  # then it's a dict and this col doesn't exist in it...
                continue

            # get the column index
            col_idx = X_columns.index(col)

            # if it's a single int, easy, otherwise query dict
            col_imp_value = fill_val if is_int else fill_val[col]

            # reassign the column itself, as we might need to make it
            # a float column for imputation to avoid numpy int64 bug
            X[col], col_imp_value = _transform_col(X[col], col_imp_value)

            # unfortunately, since we can't boolean index the
            # h2oframe, we have to convert pandas
            the_na_col = na_frame[col].as_data_frame(use_pandas=True)[col]
            na_mask_idcs = the_na_col.index[the_na_col == 1].tolist()

            for na_row in na_mask_idcs:
                X[na_row, col_idx] = col_imp_value

        # return the copy
        return X


class H2OSelectiveScaler(BaseH2OTransformer):
    """A class that will scale selected features in the H2OFrame.

    Parameters
    ----------

    feature_names : array_like (str), optional (default=None)
        The list of names on which to fit the transformer.

    target_feature : str, optional (default=None)
        The name of the target feature (is excluded from the fit)
        for the estimator.

    exclude_features : iterable or None, optional (default=None)
        Any names that should be excluded from ``feature_names``

    with_mean : bool, optional (default=True)
        should subtract mean?

    with_std : bool, optional (default=True)
        should divide by std?

    Attributes
    -------
    
    means : dict (string:float)
        The mapping of column names to column means

    stds : dict (string:float)
        The mapping of column names to column standard deviations
    """

    _min_version = '3.8.2.9'
    _max_version = None

    def __init__(self, feature_names=None, target_feature=None, exclude_features=None,
                 with_mean=True, with_std=True):
        super(H2OSelectiveScaler, self).__init__(feature_names=feature_names,
                                                 target_feature=target_feature,
                                                 exclude_features=exclude_features,
                                                 min_version=self._min_version,
                                                 max_version=self._max_version)

        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X):
        """Fit the transformer.

        Parameters
        ----------

        X : H2OFrame, shape=(n_samples, n_features)
            The training data on which to fit
        """
        X = check_frame(X, copy=False)
        frame = _frame_from_x_y(X, self.feature_names, self.target_feature)
        self.cols_ = [str(i) for i in frame.columns]

        # get and std
        if self.with_mean:
            self.means = dict(zip(self.cols_, flatten_all(frame.mean())))

        if self.with_std:
            self.stds = dict(zip(self.cols_, flatten_all(frame.sd())))

        return self

    def transform(self, X):
        """Do the transformation

        Parameters
        ----------

        X : H2OFrame, shape=(n_samples, n_features)
            The test data to transform

        Returns
        -------

        frame : H2OFrame, shape=(n_samples, n_features)
            The transformed test data.
        """
        check_is_fitted(self, 'cols_')
        frame = check_frame(X, copy=True) # get a copy...

        if (not self.with_mean) and (not self.with_std):
            return frame  # nothing to change...

        for nm in self.cols_:
            if self.with_mean:
                frame[nm] -= self.means[nm]
            if self.with_std:
                frame[nm] /= self.stds[nm]

        return frame


def _mul(a, b):
    """Multiplies two H2OFrame objects
    (no validation since internally used).

    Parameters
    ----------

    a : H2OFrame, shape=(n_samples, 1)
        The first feature

    b : H2OFrame, shape=(n_samples, 1)
        The second feature

    Returns
    -------

    ``a`` * ``b`` : H2OFrame
        The product of ``a`` and ``b``
    """
    return a * b


class H2OInteractionTermTransformer(BaseH2OTransformer):
    """A class that will generate interaction terms between selected columns.
    An interaction captures some relationship between two independent variables
    in the form of:

    :math:`In = (x_i * x_j)`

    Note that the ``H2OInteractionTermTransformer`` will only operate on the feature_names,
    and at the transform point will return ALL features plus the newly generated ones
    unless otherwise specified in the ``only_return_interactions`` parameter.

    Parameters
    ----------

    feature_names : array_like (str), optional (default=None)
        The list of names on which to fit the transformer.

    target_feature : str, optional (default=None)
        The name of the target feature (is excluded from the fit)
        for the estimator.

    exclude_features : iterable or None, optional (default=None)
        Any names that should be excluded from ``feature_names``

    interaction : callable, optional (default=None)
        A callable for interactions. Default None will
        result in multiplication of two Series objects

    name_suffix : str, optional (default='I')
        The suffix to add to the new feature name in the form of
        <feature_x>_<feature_y>_<suffix>

    only_return_interactions : bool, optional (default=False)
        If set to True, will only return features in feature_names
        and their respective generated interaction terms.

    Attributes
    ----------

    fun_ : callable
        The interaction term function assigned 
        in the ``fit`` method.
    """

    _min_version = '3.8.2.9'
    _max_version = None

    def __init__(self, feature_names=None, target_feature=None, exclude_features=None,
                 interaction_function=None, name_suffix='I', only_return_interactions=False):

        super(H2OInteractionTermTransformer, self).__init__(feature_names=feature_names,
                                                            target_feature=target_feature,
                                                            exclude_features=exclude_features,
                                                            min_version=self._min_version,
                                                            max_version=self._max_version)

        self.interaction_function = interaction_function
        self.name_suffix = name_suffix
        self.only_return_interactions = only_return_interactions

    def fit(self, frame):
        """Fit the transformer.

        Parameters
        ----------
        
        frame : H2OFrame, shape=(n_samples, n_features)
            The training data on which to fit.

        Returns
        -------

        self
        """
        frame = _frame_from_x_y(frame, self.feature_names, self.target_feature, self.exclude_features)
        self.cols = [str(u) for u in frame.columns]  # the cols we'll ultimately operate on
        self.fun_ = self.interaction_function if not self.interaction_function is None else _mul

        # validate function
        if not hasattr(self.fun_, '__call__'):
            raise TypeError('require callable for interaction_function')

        # validate features
        if len(self.cols) < 2:
            raise ValueError('need at least two features')

        return self

    def transform(self, X):
        """Perform the interaction term expansion.
        
        Parameters
        ----------

        X : H2OFrame, shape=(n_samples, n_features)
            The test data to transform.

        Returns
        -------

        frame : H2OFrame, shape=(n_samples, n_features)
            The expanded (interacted) test data.
        """
        check_is_fitted(self, 'fun_')
        frame = check_frame(X, copy=True) # get a copy
        
        cols, fun, suff = self.cols, self.fun_, self.name_suffix
        n_features = len(cols)

        # these are the names to return if only_return_interactions
        interaction_names = [x for x in cols]

        # we can do this in N^2 or we can do it in an uglier N choose 2...
        for i in range(n_features - 1):
            for j in range(i + 1, n_features):
                col_i, col_j = cols[i], cols[j]

                new_col_nm = '%s_%s_%s' % (col_i, col_j, suff)
                new_col = fun(frame[col_i], frame[col_j])
                new_col.columns = [new_col_nm]

                # add the new col nm to the list of interaction names
                interaction_names.append(new_col_nm)

                # cbind
                frame = frame.cbind(new_col)

        # return matrix if needed
        return frame if not self.only_return_interactions else frame[interaction_names]
