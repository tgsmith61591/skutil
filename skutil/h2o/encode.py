from __future__ import print_function, absolute_import, division
import pandas as pd
import h2o
from h2o.frame import H2OFrame
from sklearn.utils.validation import check_is_fitted
from ..preprocessing.encode import _get_unseen
from .frame import _check_is_1d_frame
from .base import (BaseH2OTransformer, _check_is_frame, _frame_from_x_y)

__all__ = [
    'H2OSafeOneHotEncoder'
]


def _val_vec(y):
    _check_is_1d_frame(y)
    return y


class _H2OVecSafeOneHotEncoder(BaseH2OTransformer):
    """Safely one-hot encodes an H2OVec into an H2OFrame of
    one-hot encoded dummies. Skips previously unseen levels
    in the transform section.
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

    feature_names : array_like (str), optional (default=None)
        The list of names on which to fit the transformer.

    target_feature : str, optional (default None)
        The name of the target feature (is excluded from the fit)
        for the estimator.

    exclude_features : iterable or None, optional (default=None)
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

        X : H2OFrame
            The frame to fit

        Returns
        -------

        self
        """

        frame = _check_is_frame(X)

        # these are just the features to encode
        cat = _frame_from_x_y(frame, self.feature_names, self.target_feature, self.exclude_features)

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

        X : H2OFrame
            The frame to transform

        Returns
        -------

        X_transform : H2OFrame
            The transformed H2OFrame
        """
        check_is_fitted(self, 'encoders_')
        frame = _check_is_frame(X)
        enc = self.encoders_

        # these are just the features to encode. (we will return the 
        # entire frame unless told not to...)
        cat = _frame_from_x_y(frame, self.feature_names, self.target_feature, self.exclude_features)

        output = None
        for name in cat.columns:
            name = str(name)
            dummied = enc[name].transform(cat[name])

            # duplicative of R's cbind (bind columns together)
            output = dummied if output is None else output.cbind(dummied)

        # if we need to drop the original columns, we do that here:
        if self.drop_after_encoded:
            keep_nms = [str(n) for n in X.columns if not n in cat.columns]
            X = X[keep_nms]

        # cbind the dummies at the end
        X = X.cbind(output)

        return X
