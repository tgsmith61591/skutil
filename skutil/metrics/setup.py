import os
import os.path

import numpy, warnings
from numpy.distutils.misc_util import Configuration
from sklearn._build_utils import get_blas_info


try:
    import Cython
    HAVE_CYTHON = True
except ImportError as e:
    warnings.warn(e.message)
    HAVE_CYTHON = False


def configuration(parent_package="", top_path=None):
    config = Configuration("metrics", parent_package, top_path)
    _, blas_info = get_blas_info()

    # pop it from blas because we want to use numpy's instead
    blas_info.pop('include_dirs', [])

    config.add_extension(
            name="_kernel_fast",
            sources='_kernel_fast.c', #["_kernel_fast.%s" % ('c' if not HAVE_CYTHON else 'pyx')],
            include_dirs=[numpy.get_include(), '.'],
            extra_compile_args=blas_info.pop('extra_compile_args',[]),
            **blas_info
        )

    config.add_subpackage('tests')
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(**configuration().todict())