from __future__ import print_function, absolute_import, division
import pandas as pd
import numpy as np
import h2o
from h2o.frame import H2OFrame
from sklearn.externals import six
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import LabelEncoder
from ..preprocessing.encode import _get_unseen
from .frame import _check_is_1d_frame
from .base import (BaseH2OTransformer, check_frame, _frame_from_x_y)
from .util import h2o_col_to_numpy, _unq_vals_col
from ..utils.fixes import dict_values

__all__ = [
    'H2OLabelEncoder',
    'H2OSafeOneHotEncoder'
]


def _val_vec(y):
    _check_is_1d_frame(y)
    return y


class H2OLabelEncoder(BaseH2OTransformer):
    """Encode categorical values in a H2OFrame (single column)
    into ordinal labels 0 - len(column) - 1.

    Parameters
    ----------

    feature_names : array_like (str), optional (default=None)
        The list of names on which to fit the transformer.

    target_feature : str, optional (default None)
        The name of the target feature (is excluded from the fit)
        for the estimator.

    exclude_features : iterable or None, optional (default=None)
        Any names that should be excluded from ``feature_names``


    Examples
    --------

        >>> def example():
        ...     import pandas as pd
        ...     import numpy as np
        ...     from skutil.h2o import from_pandas
        ...     from sktuil.h2o.transform import H2OLabelEncoder
        ...     
        ...     x = pd.DataFrame.from_records(data=[
        ...                 [5, 4],
        ...                 [6, 2],
        ...                 [5, 1],
        ...                 [7, 9],
        ...                 [7, 2]], columns=['C1', 'C2'])
        ...     
        ...     X = from_pandas(x)
        ...     encoder = H2OLabelEncoder()
        ...     encoder.fit_transform(X['C1'])
        >>>
        >>> example() # doctest: +SKIP
          C1
        ----
           0
           1
           0
           2
           2
        [5 rows x 1 column]


    Attributes
    ----------

    classes_ : np.ndarray
        The unique class levels
    """
    _min_version = '3.8.2.9'
    _max_version = None

    def __init__(self):
        super(H2OLabelEncoder, self).__init__(feature_names=None,
                                              target_feature=None,
                                              exclude_features=None,
                                              min_version=self._min_version,
                                              max_version=self._max_version)

    def fit(self, column):
        column = h2o_col_to_numpy(_check_is_1d_frame(column))
        self.encoder_ = LabelEncoder().fit(column)
        return self

    def transform(self, column):
        check_is_fitted(self, 'encoder_')
        column = h2o_col_to_numpy(_check_is_1d_frame(column))

        # transform
        trans = self.encoder_.transform(column)
        trans_T = trans.reshape(trans.shape[0], 1)

        # I don't like that we have to re-upload... but we do...
        return H2OFrame.from_python(trans_T)



class _H2OVecSafeOneHotEncoder(BaseH2OTransformer):
    """Safely one-hot encodes an H2OVec into an ``H2OFrame`` of
    one-hot encoded dummies. Whereas H2O's default behavior for
    previously-unseen factor levels is to error, the 
    ``_H2OVecSafeOneHotEncoder`` skips previously-unseen levels
    in the ``transform`` section, returning 'nan' (which H2O
    interprets as ``NA``).

    Parameters
    ----------

    feature_names : array_like (str) shape=(n_features,), optional (default=None)
        The list of names on which to fit the transformer.

    target_feature : str, optional (default=None)
        The name of the target feature (is excluded from the fit)
        for the estimator.

    exclude_features : array_like (str) shape=(n_features,), optional (default=None)
        Any names that should be excluded from ``feature_names``
    """

    _min_version = '3.8.2.9'
    _max_version = None

    def __init__(self):
        super(_H2OVecSafeOneHotEncoder, self).__init__(feature_names=None,
                                                       target_feature=None,
                                                       exclude_features=None,
                                                       min_version=self._min_version,
                                                       max_version=self._max_version)

    def fit(self, y):
        """Fit the encoder.

        Parameters
        ----------

        X : ``H2OFrame``, shape=(n_samples, 1)
            The training frame on which to fit. Should
            be a single column ``H2OFrame``

        Returns
        -------

        self
        """
        # validate y
        y = _val_vec(y)

        # get the unique count
        clz = y.unique().as_data_frame().T.iloc[0].tolist()

        # max class check:
        max_classes = _get_unseen()
        if len(clz) > max_classes:
            raise ValueError('max_classes=%i, but got %i'
                             % (max_classes, len(clz)))

        # set internal
        self.classes_ = clz

        return self

    def transform(self, y):
        """Transform a new 1d frame after fit.

        Parameters
        ----------

        X : ``H2OFrame``, shape=(n_samples, 1)
            The 1d ``H2OFrame`` to transform

        Returns
        -------

        output : ``H2OFrame``, shape=(n_samples, 1)
            The transformed ``H2OFrame``
        """
        # make sure is fitted, validate y
        check_is_fitted(self, 'classes_')
        y = _val_vec(y)

        # get col name
        col_name = str(y.columns[0])

        # the frame output
        output = None

        # iterate over the classes
        for clz in self.classes_:
            isnan = False
            rep = clz  # we copy for sake of NaN preservation

            # if the clz is np.nan, then the actual rep is 'NA'
            if pd.isnull(clz):
                isnan = True
                rep = 'NA'

            # returns int vec of 1s and 0s
            dummies = (y == rep)
            dummies.columns = ['%s.%s' % (col_name, clz if not isnan else 'nan')]

            # cbind
            output = dummies if output is None else output.cbind(dummies)

        return output


class H2OSafeOneHotEncoder(BaseH2OTransformer):
    """Given a set of feature_names, one-hot encodes (dummies)
    a set of vecs into an expanded set of dummied columns. Will
    drop the original columns after transformation, unless otherwise 
    specified.

    Parameters
    ----------

    feature_names : array_like (str) shape=(n_features,), optional (default=None)
        The list of names on which to fit the transformer.

    target_feature : str, optional (default=None)
        The name of the target feature (is excluded from the fit)
        for the estimator.

    exclude_features : array_like (str) shape=(n_features,), optional (default=None)
        Any names that should be excluded from ``feature_names``

    drop_after_encoded : bool (default=True)
        Whether to drop the original columns after transform
    """

    _min_version = '3.8.2.9'
    _max_version = None

    def __init__(self, feature_names=None, target_feature=None, exclude_features=None, drop_after_encoded=True):
        super(H2OSafeOneHotEncoder, self).__init__(feature_names=feature_names,
                                                   target_feature=target_feature,
                                                   exclude_features=exclude_features,
                                                   min_version=self._min_version,
                                                   max_version=self._max_version)

        self.drop_after_encoded = drop_after_encoded

    def fit(self, X):
        """Fit the one hot encoder.

        Parameters
        ----------

        X : ``H2OFrame``, shape=(n_samples, n_features)
            The training frame to fit

        Returns
        -------

        self
        """
        X = check_frame(X, copy=False)

        # these are just the features to encode
        cat = _frame_from_x_y(X, self.feature_names, self.target_feature, self.exclude_features)

        # do fit
        self.encoders_ = {
            str(k): _H2OVecSafeOneHotEncoder().fit(cat[str(k)])
            for k in cat.columns
        }

        return self

    def transform(self, X):
        """Transform a new frame after fit.

        Parameters
        ----------

        X : ``H2OFrame``, shape=(n_samples, n_features)
            The frame to transform

        Returns
        -------

        X : ``H2OFrame``, shape=(n_samples, n_features)
            The transformed H2OFrame
        """
        check_is_fitted(self, 'encoders_')
        X = check_frame(X, copy=True)
        enc = self.encoders_

        # these are just the features to encode. (we will return the 
        # entire frame unless told not to...)
        cat = _frame_from_x_y(X, self.feature_names, self.target_feature, self.exclude_features)

        output = None
        for name in cat.columns:
            name = str(name)
            dummied = enc[name].transform(cat[name])

            # duplicative of R's cbind (bind columns together)
            output = dummied if output is None else output.cbind(dummied)

        # if we need to drop the original columns, we do that here:
        if self.drop_after_encoded:
            keep_nms = [str(n) for n in X.columns if n not in cat.columns]
            X = X[keep_nms]

        # cbind the dummies at the end
        X = X.cbind(output)

        return X
