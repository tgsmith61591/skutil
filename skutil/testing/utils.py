# Utilities for testing

from __future__ import print_function, absolute_import, division
import warnings

__all__ = [
    'assert_fails'
]


def assert_fails(fun, expected=ValueError, *args, **kwargs):
    failed = False
    try:
        fun(*args, **kwargs)
    except expected:
        failed = True
        failing_class = None
    except Exception as e:
        failing_class = type(e)
        failed = False
    assert failed, 'expected %s in function but got %s' % (expected, failing_class)
