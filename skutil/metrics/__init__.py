"""
Pairwise matrix ops
"""

try:
    # This variable is injected in the __builtins__ by the build
    # process. It used to enable importing subpackages of skutil when
    # the binaries are not built
    __SKUTIL_SETUP__
except NameError:
    __SKUTIL_SETUP__ = False



if __SKUTIL_SETUP__:
    sys.stderr.write('Partial import of skutil during the metrics build process.\n')
else:
	# import statements
    from .pairwise import *