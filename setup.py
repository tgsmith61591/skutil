import os, sys
from os.path import join
import warnings
import subprocess
from pkg_resources import parse_version


if sys.version_info[0] < 3:
	import __builtin__ as builtins
else:
	import builtins


## Hacky, adopted from sklearn
builtins.__PYNORM_SETUP__ = True

## Metadata
DISTNAME = 'pynorm'
DESCRIPTION = 'A set of sklearn-esque extension modules'
MAINTAINER = 'Taylor Smith'
MAINTAINER_EMAIL = 'tgsmith61591@gmail.com'


## Import the restricted version that doesn't need compiled code
import pynorm
VERSION = pynorm.__version__


## Version requirements
pandas_min_version = '0.18'
sklearn_min_version= '0.17'
numpy_min_version  = '1.6'
scipy_min_version  = '0.17'


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

	config.add_subpackage('pynorm')
	return config


def check_statuses(pkg_nm, status, rs):
	if status['up_to_date'] is False:
		if status['version']:
			raise ImportError('Your installation of {0} {1} is out-of-date.\n{2}'.format(pkg_nm, status['version'], rs))
		else:
			raise ImportError('{0} is not installed.\n{1}'.format(pkg_nm, rs))


def setup_package():
	metadata = dict(name=DISTNAME, maintainer=MAINTAINER, maintainer_email=MAINTAINER_EMAIL, description=DESCRIPTION, version=VERSION)

	pandas_status = get_pandas_status()
	sklearn_status=get_sklearn_status()
	numpy_status  = get_numpy_status()
	scipy_status  = get_scipy_status()

	pdrs = 'pynorm requires Pandas >= {0}.\n'.format(pandas_min_version)
	skrs = 'pynorm requires sklearn >= {0}.\n'.format(sklearn_min_version)
	nprs = 'pynorm requires NumPy >= {0}.\n'.format(numpy_min_version)
	scrs = 'pynorm requires SciPy >= {0}.\n'.format(scipy_min_version)

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
