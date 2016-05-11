import os
from os.path import join
import warnings


## DEFINE CONFIG
def configuration(parent_package = '', top_path = None):
	from numpy.distutils.misc_util import Configuration
	from numpy.distutils.system_info import get_info, BlasNotFoundError
	import numpy


	libs = []
	if os.name == 'posix':
		libs.append('m')

	config = Configuration('pynorm', parent_package, top_path)
	config.add_subpackage('preprocessing')
	config.add_subpackage('preprocessing/tests')

	return config

if __name__ == '__main__':
	try:
		from numpy.distutils.core import setup
		setup(**configuration(top_path='').todict())
		print 'setup completed successfully.'
	except Exception as e:
		print 'setup failed: '
		print e
