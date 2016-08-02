from __future__ import print_function, division, absolute_import
from numpy.distutils.core import Extension

import os
from os.path import join

def configuration(parent_package='', top_path=None):
	from numpy.distutils.misc_util import Configuration

	'''
	from distutils.sysconfig import get_python_inc
	from numpy.distutils.system_info import get_info, NotFoundError, numpy_info
	from numpy.distutils.misc_util import get_numpy_include_dirs
	from scipy._build_utils import (get_sgemv_fix, get_g77_abi_wrappers, split_fortran_files)

	config = Configuration('odr', parent_package, top_path)

	lapack_opt = get_info('lapack_opt')

	if not lapack_opt:
		raise NotFoundError('no lapack/blas resources found')

	atlas_version = ([v[3:-3] for k, v in lapack_opt.get('define_macros', []) 
		if k == 'ATLAS_INFO']+[None])[0]
	if atlas_version:
		print(('ATLAS version: %s' % atlas_version))

	sources = ['dqrsl.pyf.src']
	sources += get_g77_abi_wrappers(lapack_opt)
	sources += get_sgemv_fix(lapack_opt)

	# add the fortran module
	config.add_extension('dqrsl',
						 sources=sources,
						 depends=[],
						 extra_info=lapack_opt
						 )

	return config
	'''

	config = Configuration('odr', parent_package, top_path)
	config.add_extension('dqrsl',
						 sources=['dqrsl.f'])

	return config


if __name__ == '__main__':
	from numpy.distutils.core import setup
	setup(**configuration().todict())