from __future__ import print_function
from skutil.preprocessing.tests import *
from skutil.decomposition.tests import *
from skutil.utils.tests import *
from skutil.tests import *


__test_modules = [
	test_decompose,
	test_encode,
	test_transform,
	test_util,
	test_pipe
]

def _test_runner():
	"""Runs all tests in the suite"""
	from time import time
	for module in __test_modules:
		for a in module.__all__:
			method = getattr(module, a)

			pfx, msg = 'PASS', 'passed'
			t = time()
			try:
				method()
			except AssertionError as ae:
				pfx = 'FAIL'
				msg = 'FAILED! (%s)' % (ae.message if not ae.message == '' else 'assertion error; no message provided')


			## Print just so we can see that it passed...
			print('[%s]  %.5f (sec) - %s %s' % (pfx, time() - t, a, msg))

if __name__ == '__main__':
	_test_runner()
