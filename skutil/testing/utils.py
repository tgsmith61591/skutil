# Utilities for testing

from __future__ import print_function, absolute_import, division

__all__ = [
    'assert_fails',
    'assert_elements_almost_equal'
]


def assert_fails(fun, expected=ValueError, *args, **kwargs):
    failed = False
    failing_class = None
    try:
        fun(*args, **kwargs)
    except expected:
        failed = True
    except Exception as e:
        failing_class = type(e)
        failed = False
    assert failed, 'expected %s in function but got %s' % (expected, failing_class)


def assert_elements_almost_equal(a, b, tolerance=1e-6):
    try:
        a = float(a)
        b = float(b)
    except:
        assert a == b, 'A (%r) != B (%r)' % (a, b)
    assert abs(a - b) < tolerance, '%r != %r within %r' % (a, b, tolerance)
