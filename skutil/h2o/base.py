from __future__ import print_function, division, absolute_import
import warnings
import h2o
import os
from ..utils.fixes import is_iterable
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import six
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
    'check_frame',
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

    X : H2OFrame, shape=(n_samples, n_features)
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

    X : H2OFrame, shape=(n_samples, n_features)
        The sanitized H2OFrame
    """
    x, y = validate_x_y(X, x, y, exclude_features)
    X = check_frame(X, copy=False) # don't copy here
    X = X[x] # make a copy of only the x features

    return X if not return_x_y else (X, x, y)


def check_frame(X, copy=False):
    """Returns ``X`` if ``X`` is an H2OFrame
    else raises a TypeError. If ``copy`` is True,
    will return a copy of ``X`` instead.

    Parameters
    ----------

    X : H2OFrame, shape=(n_samples, n_features)
        The frame to evaluate

    copy : bool, optional (default=False)
        Whether to return a copy of the H2OFrame.

    Returns
    -------

    X : H2OFrame, shape=(n_samples, n_features)
        The frame or the copy
    """
    if not isinstance(X, H2OFrame):
        raise TypeError('expected H2OFrame but got %s' % type(X))
    return X if not copy else X[X.columns]


def _retain_features(X, exclude):
    """Returns the features to retain. Used in
    conjunction with H2OTransformer classes that
    identify features that should be dropped.

    Parameters
    ----------

    X : H2OFrame, shape=(n_samples, n_features)
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
    return [i for i in x if i not in exclude]


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
        if not (is_iterable(x) and all([isinstance(i, six.string_types) for i in x])):
            raise TypeError('x must be an iterable of strings. '
                            'Got %s' % str(x))

    return x


def validate_x_y(X, feature_names, target_feature, exclude_features=None):
    """Validate the feature_names and target_feature arguments
    passed to an H2OTransformer.

    Parameters
    ----------

    X : H2OFrame, shape=(n_samples, n_features)
        The frame from which to drop

    feature_names : iterable or None
        The feature names to be used in a transformer. If feature_names
        is None, the transformer will use all of the frame's column names.
        However, if the feature_names are an iterable, they must all be
        either strings or unicode names of columns in the frame.

    target_feature : str, unicode or None
        The target name to exclude from the transformer analysis. If None,
        unsupervised is assumed, otherwise must be string or unicode.

    exclude_features : iterable or None, optional (default=None)
        Any names that should be excluded from ``x``

    Returns
    -------

    feature_names : list, str
        A list of the ``feature_names`` as strings

    target_feature : str or None
        The ``target_feature`` as a string if it is not 
        None, else None
    """
    if feature_names is not None:
        # validate feature_names
        feature_names = validate_x(feature_names)
    else:
        X = check_frame(X, copy=False)
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
              if not str(i) == target_feature
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
        given ``timestep`` (x-axis) against a provided 
        ``metric`` (y-axis).

        Parameters
        ----------

        timestep : str
            A timestep as defined in the H2O API. One of
            ("AUTO", "duration", "number_of_trees").

        metric : str
            The performance metric to evaluate. One of
            ("log_likelihood", "objective", "MSE", "AUTO")
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
    ----------

    min_version : str, float
        The minimum version of h2o that is compatible with the transformer

    max_version : str, float
        The maximum version of h2o that is compatible with the transformer
    """
    h2ov = h2o.__version__

    # won't enter this block if passed at 'any'
    if is_numeric(min_version):  # then int or float
        min_version = str(min_version)

    if isinstance(min_version, six.string_types):
        min_version = str(min_version)  # in case is raw or unicode

        if min_version == 'any':
            pass  # anything goes
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
        max_version = str(max_version)  # in case is raw or unicode

        if parse_version(h2ov) > parse_version(max_version):
            raise EnvironmentError('your h2o version (%s) '
                                   'exceeds the maximum permitted '
                                   'version for this transformer (%s)'
                                   % (h2ov, str(max_version)))
    elif max_version is not None:  # remember we allow None
        raise ValueError('max_version must be a float, '
                         'a string in the form of "X.x" '
                         'or None, but got %s: %s' % (type(max_version), str(max_version)))


class BaseH2OFunctionWrapper(BaseEstimator):
    """Base class for all H2O estimators or functions.

    Parameters
    ----------

    target_feature : str, optional (default=None)
        The name of the target feature (is excluded from the fit).

    min_version : str or float, optional (default='any')
        The minimum version of h2o that is compatible with the transformer

    max_version : str or float, optional (default=None)
        The maximum version of h2o that is compatible with the transformer
    """

    def __init__(self, target_feature=None, min_version='any', max_version=None):
        self.target_feature = target_feature

        # ensure our version is compatible
        check_version(min_version, max_version)

        # test connection, warn where needed
        try:
            g = h2o.frames()  # returns a dict of frames
        except (EnvironmentError, ValueError, H2OServerError, H2OConnectionError) as v:
            warnings.warn('h2o has not been started; '
                          'initializing an H2O transformer without '
                          'a connection will not cause any issues, '
                          'but it will throw a ValueError if the '
                          'H2O cloud is not started prior to fitting')

    @property
    def max_version(self):
        """Returns the max version of H2O that is compatible
        with the BaseH2OFunctionWrapper instance. Some classes
        differ in their support for H2O versions, due to changes
        in the underlying API.

        Returns
        -------

        mv : string, or None
            If there is a max version associated with
            the BaseH2OFunctionWrapper, returns it
            as a string, otherwise returns None.
        """
        try:
            mv = self._max_version
            return mv if not mv else str(mv)
        except AttributeError as n:
            return None

    @property
    def min_version(self):
        """Returns the min version of H2O that is compatible
        with the BaseH2OFunctionWrapper instance. Some classes
        differ in their support for H2O versions, due to changes
        in the underlying API.

        Returns
        -------

        mv : string
            If there is a min version associated with
            the BaseH2OFunctionWrapper, returns it
            as a string, otherwise returns 'any'
        """
        try:
            mv = str(self._min_version)
        except AttributeError as n:
            mv = 'any'
        return mv

    @staticmethod
    def load(location):
        """Loads a persisted state of an instance of BaseH2OFunctionWrapper
        from disk. If the instance is of a more complex class, i.e., one that contains
        an H2OEstimator, this method will handle loading these models separately 
        and outside of the constraints of the pickle package. 

        Note that this is a static method and should be called accordingly:

            >>> def load_and_transform():
            ...     from skutil.h2o.select import H2OMulticollinearityFilterer
            ...     mcf = H2OMulticollinearityFilterer.load(location='example/path.pkl')
            ...     return mcf.transform(X)
            >>>
            >>> load_and_transform() # doctest: +SKIP

        Some classes define their own load functionality, and will not
        work as expected if called in the following manner:

            >>> def load_pipe():
            ...     return BaseH2OFunctionWrapper.load('path/to/h2o/pipeline.pkl')
            >>>
            >>> pipe = load_pipe() # doctest: +SKIP

        This is because of the aforementioned situation wherein some classes
        handle saves and loads of H2OEstimator objects differently. Thus, any
        class that is being loaded should be statically referenced at the level of
        lowest abstraction possible:

            >>> def load_pipe():
            ...     from skutil.h2o.pipeline import H2OPipeline
            ...     return H2OPipeline.load('path/to/h2o/pipeline.pkl')
            >>>
            >>> pipe = load_pipe() # doctest: +SKIP

        Parameters
        ----------

        location : str
            The location where the persisted model resides.

        Returns
        -------

        m : BaseH2OFunctionWrapper
            The unpickled instance of the model
        """
        with open(location) as f:
            m = pickle.load(f)
        return m

    def save(self, location, warn_if_exists=True, **kwargs):
        """Saves the BaseH2OFunctionWrapper to disk. If the 
        instance is of a more complex class, i.e., one that contains
        an H2OEstimator, this method will handle saving these 
        models separately and outside of the constraints of the 
        pickle package. Any key-word arguments will be passed to
        the _save_internal method (if it exists).


        Parameters
        ----------

        location : str
            The absolute path of location where the transformer 
            should be saved.

        warn_if_exists :  bool, optional (default=True)
            Warn the user that ``location`` exists if True.
        """
        if warn_if_exists and os.path.exists(location):
            warnings.warn('Overwriting existing path: %s' % location, UserWarning)

        # models that have H2OEstimators
        if hasattr(self, '_save_internal'):
            kwargs = {} if not kwargs else kwargs
            kwargs['location'] = location
            kwargs['warn_if_exists'] = warn_if_exists

            if 'force' not in kwargs:
                kwargs['force'] = True

            if 'model_location' not in kwargs:
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

    feature_names : array_like, str
        The list of names on which to fit the feature selector.

    target_feature : str, optional (default=None)
        The name of the target feature (is excluded from the fit)
        for the estimator.

    exclude_features : iterable or None, optional (default=None)
        Any names that should be excluded from ``feature_names``
        during the fit.

    min_version : str or float, optional (default='any')
        The minimum version of h2o that is compatible with the transformer

    max_version : str or float, optional (default=None)
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
        """Fit the model and then immediately transform
        the input (training) frame with the fit parameters.

        Parameters
        ----------

        frame : H2OFrame, shape=(n_samples, n_features)
            The training frame

        Returns
        -------

        ft : H2OFrame, shape=(n_samples, n_features)
            The transformed training frame
        """
        ft = self.fit(frame).transform(frame)
        return ft
