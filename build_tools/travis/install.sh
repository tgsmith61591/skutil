#!/bin/bash
# This script is meant to be called by the 'install' step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.

set -e

# Fix the compilers to workaround having the python 3.4 build
# lookup for g++44 unexpectedly.
export CC=gcc
export CXX=g++

echo 'List files from cached directories'
echo 'pip:'
ls $HOME/.cache/pip

if [[ "$DISTRIB" == "conda" ]]; then
    # Deactivate travis-provided environment and set up a
    # conda-based environment instead
    deactivate
    
    # Use miniconda installer for faster download/install of conda
    # itself
    pushd .
    cd
    mkdir -p download
    cd download
    echo "Cached in $HOME/download :"
    ls -l
    echo
    if [[ ! -f miniconda.sh ]]
        then
        wget http://repo.continuum.io/miniconda/Miniconda-3.6.0-Linux-x86_64.sh \
            -O miniconda.sh
        fi
    chmod +x miniconda.sh && ./miniconda.sh -b
    cd ..
    export PATH=/home/travis/miniconda/bin:$PATH
    conda update --yes conda
    popd
    
    # Configure the conda environment and put it in the path using the
    # provided versions
    if [[ "$INSTALL_MKL" == "true" ]]; then
        conda create -n testenv --yes python=$PYTHON_VERSION pip nose \
            numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION sklearn=$SKLEARN_VERSION \
            pandas=$PANDAS_VERSION numpy scipy sklearn pandas
    else
        conda create -n testenv --yes python=$PYTHON_VERSION pip nose \
            numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION sklearn=$SKLEARN_VERSION \
            pandas=$PANDAS_VERSION
    fi
    source activate testenv
    
    ## Install nose timer
    pip install nose-timer

elif [[ "$DISTRIB" == "ubuntu" ]]; then
    deactivate
    virtualenv --system-site-packages testvenv
    source testvenv/bin/activate
    pip install nose nose-timer cython

elif [[ "$DISTRIB" == "scipy-dev-wheels" ]]; then
    virtualenv --python=python ~/testvenv
    source ~/testvenv/bin/activate
    pip install --upgrade pip setuptools
    
    echo "Installing numpy master wheel"
    pip install --pre --upgrade --no-index --timeout=60 \
        --trusted-host travis-dev-wheels.scipy.org \
        -f http://travis-dev-wheels.scipy.org/ numpy scipy sklearn pandas
    pip install nose nose-timer cython
fi

if [[ "$COVERAGE" == "true" ]]; then
    pip install coverage coveralls
fi

if [[ ! -d "$CACHED_BUILD_DIR" ]]; then
    mkdir -p $CACHED_BUILD_DIR
fi

rsync -av --exclude '.git/' --exclude='testvenv/' \
    $TRAVIS_BUILD_DIR $CACHED_BUILD_DIR

cd $CACHED_BUILD_DIR/skutil

# Build skutil in the install.sh script to collapse the verbose output
# in the travis output when it succeeds.
python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"
python -c "import pandas; print('pandas %s' % pandas.__version__)"
python -c "import sklearn; print('sklearn %s' % sklearn.__version__)"
python setup.py develop
