from __future__ import print_function, division, absolute_import
import abc
import numpy as np
import warnings
import os

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.externals import six

import h2o
from h2o.frame import H2OFrame

# in different versions, we get different exceptions
try:
    from h2o.backend.connection import H2OServerError
except ImportError as e:
    H2OServerError = EnvironmentError


try:
    from h2o.exceptions import H2OConnectionError
except ImportError as e:
    H2OConnectionError = EnvironmentError


from pkg_resources import parse_version
from ..utils import is_numeric

try:
    import cPickle as pickle
except ImportError as e:
    import pickle


__all__ = [
    'check_version',
    'NAWarning',
    'BaseH2OFunctionWrapper',
    'BaseH2OTransformer',
    'validate_x',
    'validate_x_y',
    'VizMixin'
]



class NAWarning(UserWarning):
    """Custom warning used to notify user that an NA exists
    within an h2o frame (h2o can handle NA values)
    """

def _frame_from_x_y(X, x, y, exclude_features=None, return_x_y=False):
    """Subset the H2OFrame if necessary. This is used in
    transformers where a target feature and feature names are
    provided.

    Parameters
    ----------
    X : H2OFrame
        The frame from which to drop

    x : array_like
        The feature names. These will be retained in the frame

    y : str
        The target feature. This will be dropped from the frame

    exclude_features : iterable or None
        Any names that should be excluded from ``x``

    return_x_y : bool, optional (default=False)
        Whether to return the sanitized ``x``, ``y`` variables.
        If False, will only return ``X``.


    Returns
    -------
    X : pd.DataFrame
        The sanitized dataframe
    """
    x, y = validate_x_y(X, x, y, exclude_features)
    X =_check_is_frame(X)[x] # make a copy

    return X if not return_x_y else (X, x, y)


def _check_is_frame(X):
    """Returns X if X is a frame else throws a TypeError

    Parameters
    ----------
    X : H2OFrame
        The frame to evaluate

    Returns
    -------
    X
    """

    if not isinstance(X, H2OFrame):
        raise TypeError('expected H2OFrame but got %s' % type(X))
    return X


def _retain_features(X, exclude):
    """Returns the features to retain. Used in
    conjunction with H2OTransformer classes that
    identify features that should be dropped.

    Parameters
    ----------
    X : H2OFrame
        The frame from which to drop

    exclude : array_like
        The columns to exclude

    Returns
    -------
    The names of the features to keep
    """
    return _retain_from_list(X.columns, exclude)


def _retain_from_list(x, exclude):
    """Returns the features to retain. Used in
    conjunction with H2OTransformer classes that
    identify features that should be dropped.

    Parameters
    ----------
    x : iterable of lists
        The list from which to exclude

    exclude : array_like
        The columns to exclude

    Returns
    -------
    The names of the features to keep
    """
    return [i for i in x if not i in exclude]


def validate_x(x):
    """Given an iterable or None, ``x``, validate that if
    it is an iterable, it only contains string types.

    Parameters
    ----------
    x : None, iterable
        The feature names

    Returns
    -------
    x : iterable or None
        The feature names
    """
    if x is not None:
        # validate feature_names
        if not (hasattr(x, '__iter__') and all([isinstance(i, six.string_types) for i in x])):
            raise TypeError('x must be an iterable of strings. '
                            'Got %s' % str(x))

    return x


def validate_x_y(X, feature_names, target_feature, exclude_features=None):
    """Validate the feature_names and target_feature arguments
    passed to an H2OTransformer.

    Parameters
    ----------
    feature_names : iterable or None
        The feature names to be used in a transformer. If feature_names
        is None, the transformer will use all of the frame's column names.
        However, if the feature_names are an iterable, they must all be
        either strings or unicode names of columns in the frame.

    target_feature : str, unicode or None
        The target name to exclude from the transformer analysis. If None,
        unsupervised is assumed, otherwise must be string or unicode.

    exclude_features : iterable or None
        Any names that should be excluded from ``x``

    Returns
    -------
    (feature_names, target_feature)
    """
    if feature_names is not None:
        # validate feature_names
        feature_names = validate_x(feature_names)
    else:
        X = _check_is_frame(X)
        feature_names = X.columns

    # validate exclude_features
    exclude_features = validate_x(exclude_features)
    if not exclude_features:
        exclude_features = []

    # we can allow it to be None...
    if target_feature is None:
        pass
    elif not isinstance(target_feature, six.string_types):
        raise TypeError('target_feature should be a single string. '
                        'Got %s (type=%s)' % (str(target_feature), type(target_feature)))
    else:
        # it is either a string or unicode...
        target_feature = str(target_feature)

    # make list of strings, return target_feature too
    # we know feature_names are not none, here so remove
    # the target_feature from the feature_names
    return (
        _retain_from_list([
            str(i) for i in feature_names 
            if not str(i)==target_feature
        ], exclude_features), 
        target_feature
    )



class VizMixin:
    """This mixin class provides the interface to plot
    an H2OEstimator's fit performance over a timestep.
    Any structure that wraps an H2OEstimator's fitting
    functionality should derive from this mixin.
    """

    def plot(self, timestep, metric):
        """Plot an H2OEstimator's performance over a
        given timestep.

        Parameters

        timestep : str
            A timestep as defined in the H2O API. Examples
            include number_of_trees, epochs

        metric : str
            The performance metric to evaluate, i.e., MSE
        """

        # Initially were raising but now want to just return NI.
        # It should be perfectly valid for a class not to implement
        # something from a mixin, but not necessarily from an abstract
        # parent. Thus, any mixins should just return the NI singleton
        return NotImplemented



def check_version(min_version, max_version):
    """Ensures the currently installed/running version
    of h2o is compatible with the min_version and max_version
    the function in question calls for.

    Parameters

    min_version : str, float
        The minimum version of h2o that is compatible with the transformer

    max_version : str, float
        The maximum version of h2o that is compatible with the transformer
    """
    h2ov = h2o.__version__

    # won't enter this block if passed at 'any'
    if is_numeric(min_version): # then int or float
        min_version = str(min_version)
    
    if isinstance(min_version, six.string_types):
        min_version = str(min_version) # in case is raw or unicode

        if min_version == 'any':
            pass # anything goes
        else:
            if parse_version(h2ov) < parse_version(min_version):
                raise EnvironmentError('your h2o version (%s) '
                                       'does not meet the minimum ' 
                                       'requirement for this transformer (%s)'
                                       % (h2ov, str(min_version)))
    
    else:
        raise ValueError('min_version must be a float, '
                         'a string in the form of "X.x" '
                         'or "any", but got %s: %s' % (type(min_version), str(min_version)))

    # validate max version
    if not max_version:
        pass
    elif is_numeric(max_version):
        max_version = str(max_version)

    if isinstance(max_version, six.string_types):
        max_version = str(max_version) # in case is raw or unicode

        if parse_version(h2ov) > parse_version(max_version):
            raise EnvironmentError('your h2o version (%s) '
                                   'exceeds the maximum permitted ' 
                                   'version for this transformer (%s)'
                                   % (h2ov, str(max_version)))
    elif not max_version is None: # remember we allow None
        raise ValueError('max_version must be a float, '
                         'a string in the form of "X.x" '
                         'or None, but got %s: %s' % (type(max_version), str(max_version)))




class BaseH2OFunctionWrapper(BaseEstimator):
    """Base class for all H2O estimators or functions.

    Parameters

    target_feature : str (default None)
        The name of the target feature (is excluded from the fit)

    min_version : str, float (default 'any')
        The minimum version of h2o that is compatible with the transformer

    max_version : str, float (default None)
        The maximum version of h2o that is compatible with the transformer
    """
    def __init__(self, target_feature=None, min_version='any', max_version=None):
        self.target_feature = target_feature

        # ensure our version is compatible
        check_version(min_version, max_version)

        # test connection, warn where needed
        try:
            g = h2o.frames() # returns a dict of frames
        except (EnvironmentError, ValueError, H2OServerError, H2OConnectionError) as v:
            warnings.warn('h2o has not been started; '
                          'initializing an H2O transformer without '
                          'a connection will not cause any issues, '
                          'but it will throw a ValueError if the '
                          'H2O cloud is not started prior to fitting')


    @property
    def max_version(self):
        try:
            mv = self.__max_version__
            return mv if not mv else str(mv)
        except AttributeError as n:
            return None
    

    @property
    def min_version(self):
        try:
            return str(self.__min_version__)
        except AttributeError as n:
            return 'any'

    @staticmethod
    def load(location):
        with open(location) as f:
            return pickle.load(f)

    def save(self, location, warn_if_exists=True, **kwargs):
        """Save the transformer"""
        if warn_if_exists and os.path.exists(location):
            warnings.warn('Overwriting existing path: %s' %location, UserWarning)

        # models that have H2OEstimators
        if hasattr(self, '_save_internal'):
            kwargs = {} if not kwargs else kwargs
            kwargs['location'] = location
            kwargs['warn_if_exists'] = warn_if_exists

            if not 'force' in kwargs:
                kwargs['force'] = True

            if not 'model_location' in kwargs:
                ops = os.path.sep
                loc_pts = location.split(ops)
                model_loc = '%s.mdl' % loc_pts[-1]
                kwargs['model_location'] = model_loc

            self._save_internal(**kwargs)

        else:
            with open(location, 'wb') as output:
                pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
    

class BaseH2OTransformer(BaseH2OFunctionWrapper, TransformerMixin):
    """Base class for all H2OTransformers.

    Parameters
    ----------
    feature_names : array_like (str)
        The list of names on which to fit the feature selector.

    target_feature : str (default None)
        The name of the target feature (is excluded from the fit)
        for the estimator.

    exclude_features : iterable or None
        Any names that should be excluded from ``feature_names``

    min_version : str, float (default 'any')
        The minimum version of h2o that is compatible with the transformer

    max_version : str, float (default None)
        The maximum version of h2o that is compatible with the transformer
    """
    def __init__(self, feature_names=None, target_feature=None, exclude_features=None, 
                 min_version='any', max_version=None):
        super(BaseH2OTransformer, self).__init__(target_feature=target_feature,
                                                 min_version=min_version,
                                                 max_version=max_version)
        # the column names
        self.feature_names = feature_names
        self.exclude_features = exclude_features

    def fit_transform(self, frame):
        return self.fit(frame).transform(frame)
