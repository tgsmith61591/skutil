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

	config = Configuration('skutil', parent_package, top_path)

	## modules
	config.add_subpackage('decomposition')
	config.add_subpackage('feature_selection')
	config.add_subpackage('odr')
	config.add_subpackage('preprocessing')
	config.add_subpackage('utils')

	## module tests
	config.add_subpackage('decomposition/tests')
	config.add_subpackage('feature_selection/tests')
	config.add_subpackage('odr/tests')
	config.add_subpackage('preprocessing/tests')
	config.add_subpackage('utils/tests')

	## modules that have their own setup.py
	config.add_subpackage('metrics')

	## misc repo tests
	config.add_subpackage('tests')

	return config

if __name__ == '__main__':
	from numpy.distutils.core import setup
	setup(**configuration(top_path='').todict())
