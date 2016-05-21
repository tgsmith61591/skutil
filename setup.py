import os, sys, shutil
from os.path import join
import warnings
import subprocess
from pkg_resources import parse_version

## For cleaning build artifacts
from distutils.command.clean import clean as Clean


if sys.version_info[0] < 3:
	import __builtin__ as builtins
else:
	import builtins


## Hacky, adopted from sklearn
builtins.__SKUTIL_SETUP__ = True

## Metadata
DISTNAME = 'skutil'
DESCRIPTION = 'A set of sklearn-esque extension modules'
MAINTAINER = 'Taylor Smith'
MAINTAINER_EMAIL = 'tgsmith61591@gmail.com'


## Import the restricted version that doesn't need compiled code
import skutil
VERSION = skutil.__version__


## Version requirements
pandas_min_version = '0.18'
sklearn_min_version= '0.16'
numpy_min_version  = '1.6'
scipy_min_version  = '0.17'


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

## Custom class to clean build artifacts
class CleanCommand(Clean):
	description = 'Remove build artifacts from the source tree'
	
	def run(self):
		Clean.run(self)

		if os.path.exists('build'):
			shutil.rmtree('build')
		for dirpath, dirnames, filenames in os.walk('skutil'):
			for filename in filenames:
				if any(filename.endswith(suffix) for suffix in ('.so','.pyd','.dll','.pyc', '.DS_Store')):
					os.unlink(os.path.join(dirpath, filename))
					continue
				extension = os.path.splitext(filename)[1]
			for dirname in dirnames:
				if dirname == '__pycache__':
					shutil.rmtree(os.path.join(dirpath, dirname))


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

## DEFINE CONFIG
def configuration(parent_package = '', top_path = None):
	from numpy.distutils.misc_util import Configuration
	config = Configuration(None, parent_package, top_path)

	## Avoid non-useful msg
	config.set_options(ignore_setup_xxx_py=True, assume_default_configuration=True, delegate_options_to_subpackages=True, quiet=True)

	config.add_subpackage('skutil')
	return config


def check_statuses(pkg_nm, status, rs):
	if status['up_to_date'] is False:
		if status['version']:
			try:
				subprocess.call(['pip', 'install', '--upgrade', ('%s' % pkg_nm)])
			except:
				raise ValueError('Your installation of {0} {1} is out-of-date.\n{2}'.format(pkg_nm, status['version'], rs))
		else:
			try:
				subprocess.call(['pip', 'install', ('%s' % pkg_nm)])
			except:
				raise ImportError('{0} is not installed.\n{1}'.format(pkg_nm, rs))


def setup_package():
	metadata = dict(name=DISTNAME, 
			maintainer=MAINTAINER, 
			maintainer_email=MAINTAINER_EMAIL, 
			description=DESCRIPTION, 
			version=VERSION,
			cmdclass=cmdclass,
			**extra_setuptools_args)

	pandas_status = get_pandas_status()
	sklearn_status=get_sklearn_status()
	numpy_status  = get_numpy_status()
	scipy_status  = get_scipy_status()

	pdrs = 'skutil requires Pandas >= {0}.\n'.format(pandas_min_version)
	skrs = 'skutil requires sklearn >= {0}.\n'.format(sklearn_min_version)
	nprs = 'skutil requires NumPy >= {0}.\n'.format(numpy_min_version)
	scrs = 'skutil requires SciPy >= {0}.\n'.format(scipy_min_version)

	
	check_statuses('Pandas', pandas_status, pdrs)
	check_statuses('sklearn',sklearn_status, skrs)
	check_statuses('NumPy', numpy_status, nprs)
	check_statuses('SciPy', scipy_status, scrs)

	## We know numpy is installed at this point
	from numpy.distutils.core import setup
	metadata['configuration'] = configuration

	setup(**metadata)

if __name__ == '__main__':
	setup_package()
