'''
sklearn-esque transformers for python
'''
import sys, warnings

__version__ = '0.0.1'

try:
    # This variable is injected in the __builtins__ by the build
    # process. It used to enable importing subpackages of pynorm when
    # the binaries are not built
    __PYNORM_SETUP__
except NameError:
    __PYNORM_SETUP__ = False



if __PYNORM_SETUP__:
    sys.stderr.write('Partial import of pynorm during the build process.\n')
else:
    __all__ = ['preprocessing']


def setup_module(module):
    import os, numpy as np, random

    _random_seed = int(np.random.uniform() * (2 ** 31 - 1))
    np.random.seed(_random_seed)
    random.seed(_random_seed)
