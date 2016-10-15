from __future__ import absolute_import, division, print_function
import warnings

__all__ = [
    'ModuleImportWarning',
    'overrides',
    'SamplingWarning',
    'SelectiveMixin',
    'SelectiveWarning',
    'suppress_warnings'
]


def overrides(interface_class):
    """Decorator for methods that override super methods.
    Nice syntactic sugar and easy to follow OOP on sources like Git.
    """
    def overrider(method):
        assert (method.__name__ in dir(interface_class)), '%s.%s must override a super method!' % (
            interface_class.__name__, method.__name__)
        return method

    return overrider


def suppress_warnings(func):
    """Decorator to force a method to suppress
    all warnings it may raise.
    """

    def suppressor(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)

    return suppressor


class ModuleImportWarning(UserWarning):
    """Custom warning used to notify user a non-critical import failed, and to
    suggest the installation of the module for optimal results.
    """


class SamplingWarning(UserWarning):
    """Custom warning used to notify the user that sub-optimal sampling behavior
    has occurred. For instance, performing oversampling on a minority class with only
    one instance will cause this warning to be thrown.
    """


class SelectiveWarning(UserWarning):
    """Custom warning used to notify user when a structure implementing SelectiveMixin
    operates improperly. A common usecase is when the fit method receives a non-DataFrame
    X, and no cols.
    """


class SelectiveMixin:
    """A mixin class that all selective transformers
    should implement. All ``SelectiveMixin`` implementers
    should only apply their ``fit`` method on the defined columns.
    """
