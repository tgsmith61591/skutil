from __future__ import print_function, absolute_import, division
from sklearn.utils.metaestimators import _IffHasAttrDescriptor


__all__ = [
    'if_delegate_has_method'
]

def if_delegate_has_method(delegate, method=None):
    """Create a decorator for methods that are delegated to a sub-estimator
    This enables ducktyping by hasattr returning True according to the
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
        attribute_name=method if not method is None else fn.__name__)