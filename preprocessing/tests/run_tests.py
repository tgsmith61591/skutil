from pynorm.preprocessing.tests import *

__test_modules = [
	test_util,
	test_encode,
	test_transform
]

def _test_runner():
	"""Runs all tests in the suite"""
	from time import time
	for module in __test_modules:
		for a in module.__all__:
			method = getattr(module, a)

			pfx, msg = 'INFO ', 'passed'
			t = time()
			try:
				method()
			except AssertionError as ae:
				pfx = 'ERROR'
				msg = 'FAILED! (%s)' % (ae.message if not ae.message == '' else 'no message provided')


			## Print just so we can see that it passed...
			print '[%s]  %.5f (sec) - %s %s' % (pfx, time() - t, a, msg)

if __name__ == '__main__':
	_test_runner()