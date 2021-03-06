"""Utilities for meta-estimators"""
# Author: Joel Nothman
#         Andreas Mueller
# Adapted by Taylor Smith for use with skutil
# License: BSD

from __future__ import print_function, absolute_import, division
from functools import update_wrapper
from operator import attrgetter
from ..base import since

__all__ = [
    'if_delegate_has_method',
    'if_delegate_isinstance'
]


class _IffHasAttrDescriptor(object):
    """Implements a conditional property using the descriptor protocol.
    Using this class to create a decorator will raise an ``AttributeError``
    if none of the delegates (specified in ``delegate_names``) is an attribute
    of the base object or the first found delegate does not have an attribute
    ``attribute_name``. THIS IS ADAPTED FROM SKLEARN 0.18, AS IT ADDS NEW
    FUNCTIONALITY AND ALLOWS US TO SEARCH FOR A SPECIFIC ATTRIBUTE NAME.

    This allows ducktyping of the decorated method based on
    ``delegate.attribute_name``. Here ``delegate`` is the first item in
    ``delegate_names`` for which ``hasattr(object, delegate) is True``.

    See https://docs.python.org/3/howto/descriptor.html for an explanation of
    descriptors.
    """

    def __init__(self, fn, delegate_names, attribute_name):
        self.fn = fn
        self.delegate_names = delegate_names
        self.attribute_name = attribute_name

        # update the docstring of the descriptor
        update_wrapper(self, fn)

    def __get__(self, obj, type=None):
        # raise an AttributeError if the attribute is not present on the object
        if obj is not None:
            # delegate only on instances, not the classes.
            # this is to allow access to the docstrings.
            for delegate_name in self.delegate_names:
                try:
                    delegate = attrgetter(delegate_name)(obj)
                except AttributeError:
                    continue
                else:
                    getattr(delegate, self.attribute_name)
                    break
            else:
                attrgetter(self.delegate_names[-1])(obj)

        # lambda, but not partial, allows help() to work with update_wrapper
        out = lambda *args, **kwargs: self.fn(obj, *args, **kwargs)
        # update the docstring of the returned function
        update_wrapper(out, self.fn)
        return out


def if_delegate_has_method(delegate, method=None):
    """Create a decorator for methods that are delegated to a sub-estimator
    This enables ducktyping by ``hasattr`` returning True according to the
    sub-estimator. This is an adaptation of the sklearn if_delegate_has_method
    where this implementation allows specific method naming.

    Parameters
    ----------

    delegate : string, list of strings or tuple of strings
        Name of the sub-estimator that can be accessed as an attribute of the
        base object. If a list or a tuple of names are provided, the first
        sub-estimator that is an attribute of the base object  will be used.

    method : string, optional (default=None)
        The name of the method delegated to the decorator. If None,
        will use the name of the current function.
    """
    if isinstance(delegate, list):
        delegate = tuple(delegate)
    if not isinstance(delegate, tuple):
        delegate = (delegate,)

    return lambda fn: _IffHasAttrDescriptor(fn, delegate,
                                            attribute_name=(method if method is not None else fn.__name__))


class _IffIsInstanceDescriptor(object):
    """Implements a conditional property using the descriptor protocol.
    Using this class to create a decorator will raise a ``AttributeError``
    if none of the delegates (specified in ``delegate_names``) is an attribute
    of the base object or a ``TypeError`` if the first found delegate is not an
    instance of ``instance_type``.

    This allows ducktyping of the decorated method based on
    ``delegate.instance_type``. Here ``delegate`` is the first item in
    ``delegate_names`` for which ``hasattr(object, delegate) is True``.
    """

    def __init__(self, fn, delegate_names, instance_type):
        self.fn = fn
        self.delegate_names = delegate_names
        self.instance_type = instance_type

        # update the docstring of the descriptor
        update_wrapper(self, fn)

    def __get__(self, obj, type=None):
        # raise an AttributeError if the attribute is not present on the object
        if obj is not None:
            # delegate only on instances, not the classes.
            # this is to allow access to the docstrings.
            for delegate_name in self.delegate_names:
                try:
                    delegate = getattr(obj, delegate_name)
                except AttributeError:
                    continue
                else:
                    if not isinstance(delegate, self.instance_type):
                        raise TypeError('delegate (%s) is not an instance of %s' 
                                        % (delegate, self.instance_type))
                    break
            else:
                attrgetter(self.delegate_names[-1])(obj)

        # lambda, but not partial, allows help() to work with update_wrapper
        out = lambda *args, **kwargs: self.fn(obj, *args, **kwargs)
        # update the docstring of the returned function
        update_wrapper(out, self.fn)
        return out


@since('0.1.2')
def if_delegate_isinstance(delegate, instance_type):
    """Create a decorator for methods that are delegated to a sub-estimator
    of a given type. This enables ducktyping by ``isinstance`` returning True 
    according to the sub-estimator.

    Parameters
    ----------

    delegate : string, list of strings or tuple of strings
        Name of the sub-estimator that can be accessed as an attribute of the
        base object. If a list or a tuple of names are provided, the first
        sub-estimator that is an attribute of the base object  will be used.

    instance_type : type
        The type of object to check for instance.
    """
    if isinstance(delegate, list):
        delegate = tuple(delegate)
    if not isinstance(delegate, tuple):
        delegate = (delegate,)

    return lambda fn: _IffIsInstanceDescriptor(fn, delegate, instance_type=instance_type)
