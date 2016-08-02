from __future__ import print_function
import os, sys, shutil, glob
from os.path import join
import warnings
import subprocess
from pkg_resources import parse_version
from distutils.extension import Extension

## For cleaning build artifacts
from distutils.command.clean import clean as Clean


if sys.version_info[0] < 3:
	import __builtin__ as builtins
else:
	import builtins


try:
	from Cython.Build import cythonize
	ext = 'pyx'
except ImportError as e:
	warnings.warn('Cython needs to be installed')
	raise e
	#ext = 'c'



## Hacky, adopted from sklearn
builtins.__SKUTIL_SETUP__ = True

## Metadata
DISTNAME = 'skutil'
DESCRIPTION = 'A set of sklearn-esque extension modules'
MAINTAINER = 'Taylor G. Smith'
MAINTAINER_EMAIL = 'tgsmith61591@gmail.com'


## Import the restricted version that doesn't need compiled code
import skutil
VERSION = skutil.__version__


## Version requirements
pandas_min_version = '0.18'
sklearn_min_version= '0.16'
numpy_min_version  = '1.6'
scipy_min_version  = '0.17'
h2o_min_version    = '3.8'

# optional, but if installed and lower version, warn
matplotlib_version = '1.5'


## Define setup tools early
SETUPTOOLS_COMMANDS = set([
	'develop','release','bdist_egg','bdist_rpm',
	'bdist_wininst','install_egg_info','build_sphinx',
	'egg_info','easy_install','upload','bdist_wheel',
	'--single-version-externally-managed'
])

if SETUPTOOLS_COMMANDS.intersection(sys.argv):
	import setuptools
	extra_setuptools_args = dict(zip_safe=False, include_package_data=True)
else:
	extra_setuptools_args = dict()


def _clean_compiled(suffixes):
	# check on compiled files
	for dirpath, dirnames, filenames in os.walk('skutil'):
		for filename in filenames:
			flnm = os.path.join(dirpath, filename)

			# rm compiled files
			if any(filename.endswith(suffix) for suffix in suffixes):
				print('Removing %s' % flnm)
				os.unlink(flnm)
				continue

			extension = os.path.splitext(filename)[1]
		for dirname in dirnames:
			if dirname == '__pycache__':
				shutil.rmtree(os.path.join(dirpath, dirname))

def _clean_fortran():
	print('Cleaning existing compiled Fortran files')
	# check on fortran dirs
	fortran_dirs = ['odr']
	for dr in fortran_dirs:
		fortrans = glob.glob(os.path.join('skutil',dr,'*.so.*'))
		for fortran in fortrans:
			print('Removing %s' % fortran)
			shutil.rmtree(fortran)

	# clean the compiled files
	_clean_compiled(('.so', '.pyf'))


def generate_fortran():
	print("Generating Fortran modules")
	cwd = os.path.abspath(os.path.dirname(__file__))
	p = subprocess.call([sys.executable, os.path.join(cwd, 'build_tools', 'fortranize.py'), 'skutil'], cwd=cwd)
	if p != 0:
		raise RuntimeError("Running fortranize failed!")


def generate_cython():
	print("Generating Cython modules")
	cwd = os.path.abspath(os.path.dirname(__file__))
	p = subprocess.call([sys.executable, os.path.join(cwd, 'build_tools', 'cythonize.py'), 'skutil'], cwd=cwd)
	if p != 0:
		raise RuntimeError("Running cythonize failed!")

def _clean_all():
	print('Removing existing build artifacts')

	if os.path.exists('build'):
		shutil.rmtree('build')
	if os.path.exists('dist'):
		shutil.rmtree('dist')
	if os.path.exists('%s.egg-info' % DISTNAME):
		shutil.rmtree('%s.egg-info' % DISTNAME)


	# check on fortran dirs
	_clean_fortran() # takes care of .so files

	# check on other compiled files
	_clean_compiled(('.pyd','.dll','.pyc','.DS_Store'))


## Custom class to clean build artifacts
class CleanCommand(Clean):
	description = 'Remove build artifacts from the source tree'
	
	def run(self):
		Clean.run(self)
		_clean_all()


cmdclass = {'clean' : CleanCommand}


WHEELHOUSE_UPLOADER_COMMANDS = set(['fetch_artifacts','upload_all'])
if WHEELHOUSE_UPLOADER_COMMANDS.intersection(sys.argv):
	import wheelhouse_uploader.cmd
	cmdclass.update(vars(wheelhouse_uploader.cmd))


def get_pandas_status():
	pd_status = {}
	try:
		import pandas as pd
		pd_version = str(pd.__version__) ## pandas uses a unicode string...
		pd_status['up_to_date'] = parse_version(pd_version) >= parse_version(pandas_min_version)
		pd_status['version'] = pd_version
	except ImportError:
		pd_status['up_to_date'] = False
		pd_status['version'] = ""
	return pd_status

def get_sklearn_status():
	sk_status = {}
	try:
		import sklearn as sk
		sk_version = sk.__version__
		sk_status['up_to_date'] = parse_version(sk_version) >= parse_version(sklearn_min_version)
		sk_status['version'] = sk_version
	except ImportError:
		sk_status['up_to_date'] = False
		sk_status['version'] = ""
	return sk_status

def get_numpy_status():
	np_status = {}
	try:
		import numpy as np
		np_version = np.__version__
		np_status['up_to_date'] = parse_version(np_version) >= parse_version(numpy_min_version)
		np_status['version'] = np_version
	except ImportError:
		np_status['up_to_date'] = False
		np_status['version'] = ""
	return np_status

def get_scipy_status():
	sc_status = {}
	try:
		import scipy as sc
		sc_version = sc.__version__
		sc_status['up_to_date'] = parse_version(sc_version) >= parse_version(scipy_min_version)
		sc_status['version'] = sc_version
	except ImportError:
		sc_status['up_to_date'] = False
		sc_status['version'] = ""
	return sc_status

def get_h2o_status():
	h2_status = {}
	try:
		import h2o
		h2_version = h2o.__version__
		h2_status['up_to_date'] = parse_version(h2_version) >= parse_version(h2o_min_version)
		h2_status['version'] = h2_version
	except ImportError:
		h2_status['up_to_date'] = False
		h2_status['version'] = ""
	return h2_status

## DEFINE CONFIG
def configuration(parent_package = '', top_path = None):
	from numpy.distutils.misc_util import Configuration
	config = Configuration(None, parent_package, top_path)

	## Avoid non-useful msg
	config.set_options(ignore_setup_xxx_py=True, assume_default_configuration=True, delegate_options_to_subpackages=True, quiet=True)

	config.add_subpackage(DISTNAME)
	return config


def check_statuses(pkg_nm, status, rs):
	if status['up_to_date'] is False:
		if status['version']:
			warning_msg = 'Your installation of {0} {1} is out-of-date.\n{2}'.format(pkg_nm, status['version'], rs)
			try:
				print(warning_msg, 'Attempting to upgrade.\n')
				subprocess.call(['pip', 'install', '--upgrade', ('%s' % pkg_nm)])
			except:
				raise ValueError('cannot upgrade')
		else:
			warning_msg = '{0} is not installed.\n{1}'.format(pkg_nm, rs)
			try:
				print(warning_msg, 'Attempting to install.\n')
				subprocess.call(['pip', 'install', ('%s' % pkg_nm)])
			except:
				raise ImportError('cannot install')


def setup_package():
	metadata = dict(name=DISTNAME, 
			maintainer=MAINTAINER, 
			maintainer_email=MAINTAINER_EMAIL, 
			description=DESCRIPTION, 
			version=VERSION,
			classifiers=['Intended Audience :: Science/Research',
						 'Intended Audience :: Developers',
						 'Intended Audience :: Scikit-learn users',
						 'Programming Language :: C',
						 'Programming Language :: Fortran',
						 'Programming Language :: Python',
						 'Topic :: Machine Learning',
						 'Topic :: Software Development',
						 'Topic :: Scientific/Engineering',
						 'Operating System :: Microsoft :: Windows',
						 'Operating System :: POSIX',
						 'Operating System :: Unix',
						 'Operating System :: MacOS',
						 'Programming Language :: Python :: 2.7'
						 ],
			keywords='sklearn smote caret h2o',
			cmdclass=cmdclass,
			**extra_setuptools_args)


	if len(sys.argv) == 1 or (
		len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or
								sys.argv[1] in ('--help-commands',
												'egg-info',
												'--version',
												'clean'))):
		# For these actions, NumPy is not required, nor Cythonization
		#
		# They are required to succeed without Numpy for example when
		# pip is used to install skutil when Numpy is not yet present in
		# the system.
		try:
			from setuptools import setup
		except ImportError as impe:
			from distutils.core import setup

		metadata['version'] = VERSION

	else:
		# check on MPL
		try:
			import matplotlib
			mplv = matplotlib.__version__
			mpl_uptodate = parse_version(mplv) >= parse_version(matplotlib_version)
			
			if not mpl_uptodate:
				warnings.warn('Consider upgrading matplotlib (current version=%s, recommended=1.5)' % mplv)
		except ImportError as i:
			pass # not required, doesn't matter


		pandas_status = get_pandas_status()
		sklearn_status= get_sklearn_status()
		numpy_status  = get_numpy_status()
		scipy_status  = get_scipy_status()
		h2o_status    = get_h2o_status()

		pdrs = 'skutil requires Pandas >= {0}.\n'.format(pandas_min_version)
		skrs = 'skutil requires sklearn >= {0}.\n'.format(sklearn_min_version)
		nprs = 'skutil requires NumPy >= {0}.\n'.format(numpy_min_version)
		scrs = 'skutil requires SciPy >= {0}.\n'.format(scipy_min_version)
		h2rs = 'skutil requires h2o >= {0}.\n'.format(h2o_min_version)
		
		check_statuses('numpy', numpy_status, nprs) ## Needs to happen before anything
		check_statuses('scipy', scipy_status, scrs) ## Needs to happen before sklearn
		check_statuses('pandas', pandas_status, pdrs)
		check_statuses('scikit-learn',sklearn_status, skrs)
		check_statuses('h2o', h2o_status, h2rs)

		## We know numpy is installed at this point
		import numpy
		from numpy.distutils.core import setup

		metadata['configuration'] = configuration


		# we need to build our fortran and cython
		if len(sys.argv) >= 2 and sys.argv[1] not in 'config': #and sys.argv[1] in ('build_ext'): 
			# clean up the .so files
			# _clean_all()


			# Clean existing .so files
			cwd = os.path.abspath(os.path.dirname(__file__))
			for dirpath, dirnames, filenames in os.walk(os.path.join(cwd, DISTNAME)):
				for filename in filenames:
					extension = os.path.splitext(filename)[1]

					if extension in (".so.dSYM", ".so", ".pyd", ".dll"):
						for e in ('.f', '.f90', '.pyx'):
							pyx_file = str.replace(filename, extension, e)
							print(pyx_file)

							if not os.path.exists(os.path.join(dirpath, pyx_file)):
								delpath = os.path.join(dirpath, filename)

								if os.path.isfile(delpath):
									os.unlink(delpath)
								elif os.path.isdir(delpath):
									shutil.rmtree(delpath)


			# gen fortran modules
			#generate_fortran()

			# gen cython sources (compile the .pyx files if needed)
			print('Generating cython files')

			# sklearn method...
			if not os.path.exists(os.path.join(cwd, 'PKG-INFO')):
				# Generate Cython sources, unless building from source release
				generate_cython()
		


	setup(**metadata)


if __name__ == '__main__':
	setup_package()
