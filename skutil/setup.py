import os


# DEFINE CONFIG
def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    libs = []
    if os.name == 'posix':
        libs.append('m')

    config = Configuration('skutil', parent_package, top_path)

    # modules with build utils
    config.add_subpackage('_build_utils')

    # modules
    config.add_subpackage('decomposition')
    config.add_subpackage('odr')  # needs to happen before feature selection -- has its own setup...
    config.add_subpackage('feature_selection')
    config.add_subpackage('linear_model')
    config.add_subpackage('model_selection')
    config.add_subpackage('preprocessing')
    config.add_subpackage('utils')
    config.add_subpackage('h2o')

    # module tests -- must be added after others!
    config.add_subpackage('decomposition/tests')
    config.add_subpackage('feature_selection/tests')
    config.add_subpackage('linear_model/tests')
    config.add_subpackage('odr/tests')
    config.add_subpackage('preprocessing/tests')
    config.add_subpackage('utils/tests')
    config.add_subpackage('h2o/tests')

    # Modules with cython
    config.add_subpackage('metrics')
    # config.add_subpackage('metrics/tests') # added in the setup...

    # misc repo tests
    config.add_subpackage('tests')

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup

    setup(**configuration(top_path='').todict())
