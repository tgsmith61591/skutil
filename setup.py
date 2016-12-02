from __future__ import print_function
import os
import sys
import shutil
import glob
import traceback
import warnings
import subprocess
from pkg_resources import parse_version

# For cleaning build artifacts
from distutils.command.clean import clean

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

# Hacky, adopted from sklearn. This sets a global variable
# so skutil __init__ can detect if it's being loaded in the setup
# routine, so it won't load submodules that haven't yet been built.
builtins.__SKUTIL_SETUP__ = True

# Metadata
DISTNAME = 'skutil'
DESCRIPTION = 'A set of sklearn-esque extension modules'
MAINTAINER = 'Taylor G. Smith'
MAINTAINER_EMAIL = 'tgsmith61591@gmail.com'
LICENSE = 'new BSD'

# Import the restricted version that doesn't need compiled code
import skutil

VERSION = skutil.__version__

# Version requirements
pandas_min_version = '0.18'
sklearn_min_version = '0.17'
numpy_min_version = '1.6'
scipy_min_version = '0.17'
h2o_min_version = '3.8.2.9'

# optional, but if installed and lower version, warn
matplotlib_version = '1.5'

# Define setup tools early
SETUPTOOLS_COMMANDS = {  # this is a set literal, not a dict
    'develop', 'release', 'bdist_egg', 'bdist_rpm',
    'bdist_wininst', 'install_egg_info', 'build_sphinx',
    'egg_info', 'easy_install', 'upload', 'bdist_wheel',
    '--single-version-externally-managed'
}

if SETUPTOOLS_COMMANDS.intersection(sys.argv):
    import setuptools

    extra_setuptools_args = dict(
        zip_safe=False,  # the package can run out of an .egg file
        include_package_data=True,
        extras_require={
            'alldeps': (
                'pandas >= {0}'.format(pandas_min_version),
                'scikit-learn >= {0}'.format(sklearn_min_version),
                'numpy >= {0}'.format(numpy_min_version),
                'scipy >= {0}'.format(scipy_min_version),
                'h2o >= {0}'.format(h2o_min_version)
            ),
        },
    )
else:
    extra_setuptools_args = dict()


# Custom clean command to remove build artifacts -- adopted from sklearn
class CleanCommand(clean):
    description = "Remove build artifacts from the source tree"

    def run(self):
        clean.run(self)
        # Remove c files if we are not within a sdist package
        cwd = os.path.abspath(os.path.dirname(__file__))
        remove_c_files = not os.path.exists(os.path.join(cwd, 'PKG-INFO'))
        if remove_c_files:
            cython_hash_file = os.path.join(cwd, 'cythonize.dat')
            if os.path.exists(cython_hash_file):
                os.unlink(cython_hash_file)
            print('Will remove generated .c & .so files')
        if os.path.exists('build'):
            shutil.rmtree('build')
        for dirpath, dirnames, filenames in os.walk('skutil'):
            for filename in filenames:
                if any(filename.endswith(suffix) for suffix in
                       (".so", ".pyd", ".dll", ".pyc")):
                    print('Removing file: %s' % filename)
                    os.unlink(os.path.join(dirpath, filename))
                    continue
                extension = os.path.splitext(filename)[1]
                if remove_c_files and extension in ['.c', '.cpp']:
                    pyx_file = str.replace(filename, extension, '.pyx')
                    if os.path.exists(os.path.join(dirpath, pyx_file)):
                        os.unlink(os.path.join(dirpath, filename))
            for dirname in dirnames:
                if dirname == '__pycache__' or dirname.endswith('.so.dSYM'):
                    print('Removing directory: %s' % dirname)
                    shutil.rmtree(os.path.join(dirpath, dirname))

cmdclass = {'clean': CleanCommand}

# This is the optional wheelhouse-uploader feature
# that sklearn includes in its setup. We can use this to both
# fetch artifacts as well as upload to PyPi (eventually).
# The URLs are set up in the setup.cfg file that sklearn defined
# and we modified.

WHEELHOUSE_UPLOADER_COMMANDS = {'fetch_artifacts', 'upload_all'}  # set literal
if WHEELHOUSE_UPLOADER_COMMANDS.intersection(sys.argv):
    import wheelhouse_uploader
    cmdclass.update(vars(wheelhouse_uploader.cmd))


# DEFINE CONFIG
def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)

    # Avoid non-useful msg
    # "Ignoring attempt to set 'name' (from ... "
    config.set_options(ignore_setup_xxx_py=True, 
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True, 
                       quiet=True)

    config.add_subpackage(DISTNAME)
    return config


# the default dict for a non-up-to-date package
default_status = {'up_to_date': False, 'version': ""}


def _check_version(current, required):
    crnt = str(current)
    return {
        'up_to_date': parse_version(crnt) >= parse_version(required),
        'version':    crnt
    }


def get_pandas_status():
    try:
        import pandas as pd
        return _check_version(pd.__version__, pandas_min_version)
    except ImportError:
        traceback.print_exc()
        return default_status


def get_sklearn_status():
    try:
        import sklearn as sk
        return _check_version(sk.__version__, sklearn_min_version)
    except ImportError:
        traceback.print_exc()
        return default_status


def get_numpy_status():
    try:
        import numpy as np
        return _check_version(np.__version__, numpy_min_version)
    except ImportError:
        traceback.print_exc()
        return default_status


def get_scipy_status():
    try:
        import scipy as sc
        return _check_version(sc.__version__, scipy_min_version)
    except ImportError:
        traceback.print_exc()
        return default_status


def get_h2o_status():
    try:
        import h2o
        return _check_version(h2o.__version__, h2o_min_version)
    except ImportError:
        traceback.print_exc()
        return default_status


def generate_cython():
    cwd = os.path.abspath(os.path.dirname(__file__))
    print("Generating Cython modules")
    p = subprocess.call([sys.executable, os.path.join(cwd, 
                                                      'build_tools', 
                                                      'cythonize.py'), 
                        'skutil'], cwd=cwd)

    if p != 0:
        raise RuntimeError("Running cythonize failed!")


def check_statuses(pkg_nm, status, rs):
    if status['up_to_date'] is False:
        if status['version']:
            warning_msg = 'Your installation of {0} {1} is out-of-date.\n{2}'.format(
                pkg_nm, status['version'], rs)
        else:
            warning_msg = '{0} is not installed.\n{1}'.format(pkg_nm, rs)
        raise ImportError(warning_msg)


def setup_package():
    metadata = dict(name=DISTNAME,
                    maintainer=MAINTAINER,
                    maintainer_email=MAINTAINER_EMAIL,
                    description=DESCRIPTION,
                    license=LICENSE,
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
        except ImportError:
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
        except ImportError:
            # not required, doesn't matter really
            warnings.warn('Matplotlib is not installed. Some functions may not work as expected.',
                          ImportWarning)

        pandas_status = get_pandas_status()
        sklearn_status = get_sklearn_status()
        numpy_status = get_numpy_status()
        scipy_status = get_scipy_status()
        h2o_status = get_h2o_status()

        pdrs = 'skutil requires Pandas >= {0}.\n'.format(pandas_min_version)
        skrs = 'skutil requires sklearn >= {0}.\n'.format(sklearn_min_version)
        nprs = 'skutil requires NumPy >= {0}.\n'.format(numpy_min_version)
        scrs = 'skutil requires SciPy >= {0}.\n'.format(scipy_min_version)
        h2rs = 'skutil requires h2o >= {0}.\n'.format(h2o_min_version)

        check_statuses('numpy', numpy_status, nprs)  # Needs to happen before anything
        check_statuses('scipy', scipy_status, scrs)  # Needs to happen before sklearn
        check_statuses('pandas', pandas_status, pdrs)
        check_statuses('scikit-learn', sklearn_status, skrs)
        check_statuses('h2o', h2o_status, h2rs)

        # We know numpy is installed at this point
        from numpy.distutils.core import setup

        metadata['configuration'] = configuration

        # we need to build our fortran and cython
        if len(sys.argv) >= 2 and sys.argv[1] not in 'config':  # and sys.argv[1] in ('build_ext'):
            # cythonize, fortranize

            print('Generating cython files')

            cwd = os.path.abspath(os.path.dirname(__file__))
            if not os.path.exists(os.path.join(cwd, 'PKG-INFO')):
                # Generate Cython sources, unless building from source release
                generate_cython()

            # sklearn cleans up .so files here... but we won't for now...

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
