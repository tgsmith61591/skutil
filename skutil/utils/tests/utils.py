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


def suppress_test_warnings(func):
    def test_warning_suppressor(*args, **kwargs):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)

    return test_warning_suppressor
