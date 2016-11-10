# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import six
from abc import ABCMeta
import warnings

__all__ = [
    'overrides',
    'suppress_warnings',
    'BaseSkutil',
    'SelectiveMixin'
]


def overrides(interface_class):
    """Decorator for methods that override super methods. Provides
    runtime validation that the method is, in fact, inherited from the
    superclass. If not, will raise an ``AssertionError``.

    Examples
    --------
    
    The following is valid use:

        >>> class A():
        ...     def a(self):
        ...         return 1

        >>> class B(A):
        ...     @overrides(A)
        ...     def a(self):
        ...         return 2
        ...
        ...     def b(self):
        ...         return 0

    The following would be an invalid ``overrides`` statement, since
    ``A`` does not have a ``b`` method to override.

        >>> class C(B): # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     @overrides(A) # should override B, not A
        ...     def b(self):
        ...         return 1
        Traceback (most recent call last):  
        AssertionError: A.b must override a super method!

    """

    def overrider(method):
        assert (method.__name__ in dir(interface_class)), '%s.%s must override a super method!' % (
            interface_class.__name__, method.__name__)
        return method

    return overrider


def suppress_warnings(func):
    """Decorator that forces a method to suppress
    all warnings it may raise. This should be used with caution,
    as it may complicate debugging. For internal purposes, this is
    used for imports that cause consistent warnings (like pandas or
    matplotlib)

    Parameters
    ----------

    func : callable
        Automatically passed to the decorator. This
        function is run within the context of the warning
        filterer.


    Examples
    --------

    When any function is decorated with the ``suppress_warnings``
    decorator, any warnings that are raised will be suppressed.

        >>> import warnings
        >>>
        >>> @suppress_warnings
        ... def fun_that_warns():
        ...     warnings.warn("This is a warning", UserWarning)
        ...     return 1
        >>>
        >>> fun_that_warns()
        1
    """

    def suppressor(*args, **kwargs):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)

    return suppressor


class SelectiveMixin:
    """A mixin class that all selective transformers
    should implement. All ``SelectiveMixin`` implementers
    should only apply their ``fit`` method on the defined columns.
    """
    # at one time, this contained methods. But They've since
    # been weeded out one-by-one... do we want to keep it?
    # TODO: in future versions, remove this mixin or add
    # concrete functionality


class BaseSkutil(six.with_metaclass(ABCMeta, BaseEstimator, 
                                    TransformerMixin, SelectiveMixin)):
    """Provides the base class for all non-h2o skutil transformers.
    Implements both ``TransformerMixin`` and ``SelectiveMixin``. Also
    provides the "pretty-print" ``__repr__`` implemented in sklearn's
    ``BaseEstimator``.

    Parameters
    ----------

    cols : array_like, shape=(n_features,), optional (default=None)
        The names of the columns on which to apply the transformation.
        If no column names are provided, the transformer will be ``fit``
        on the entire frame. Note that the transformation will also only
        apply to the specified columns, and any other non-specified
        columns will still be present after transformation.

    as_df : bool, optional (default=True)
        Whether to return a Pandas ``DataFrame`` in the ``transform``
        method. If False, will return a Numpy ``ndarray`` instead. 
        Since most skutil transformers depend on explicitly-named
        ``DataFrame`` features, the ``as_df`` parameter is True by default.


    Examples
    --------

        >>> from skutil.base import BaseSkutil
        >>> class A(BaseSkutil):
        ...     def __init__(self, cols=None, as_df=None):
        ...             super(A, self).__init__(cols, as_df)
        ...
        >>> A()
        A(as_df=None, cols=None)

    """
    
    def __init__(self, cols=None, as_df=True):
        self.cols = cols
        self.as_df = as_df
