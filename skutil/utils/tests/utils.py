# Utilities for testing

from __future__ import print_function, absolute_import, division
import warnings

__all__ = [
    'assert_fails',
    'suppress_test_warnings'
]


def assert_fails(fun, expected=ValueError, *args, **kwargs):
    failed = False
    try:
        fun(*args, **kwargs)
    except expected:
        failed = True
    assert failed, 'expected %s in function' % expected
