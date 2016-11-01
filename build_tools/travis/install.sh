#!/bin/bash

case "${TRAVIS_PYTHON_VERSION}" in
    "2.7")
        wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O $HOME/miniconda.sh;
        ;;
    "3.5")
        wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $HOME/miniconda.sh;
        ;;
esac

bash $HOME/miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda

# Useful for debugging any issues with conda
conda info -a

# Dependencies
conda create -q -n test-environment python=$TRAVIS_PYTHON_VERSION numpy scipy pandas coverage cython
source activate test-environment

# Coverage packages
if [[ "$TRAVIS_PYTHON_VERSION" == "2.7" ]]; then
      conda install --yes -c dan_blanchard python-coveralls nose-cov;
fi

case "${COMBINATION}" in
    "sklearn-0-17-matplotlib-seaborn")
        pip install scikit-learn==0.17.1
        pip install matplotlib
        pip install seaborn
        ;;
    "sklearn-0-17-without-matplotlib-seaborn")
        pip install scikit-learn==0.17.1
        ;;
    "sklearn-0-18-matplotlib-seaborn")
        pip install scikit-learn==0.18
        pip install matplotlib
        pip install seaborn
        ;;
    "sklearn-0-18-without-matplotlib-seaborn")
        pip install scikit-learn==0.18
        ;;
esac

pip install coveralls
pip install http://h2o-release.s3.amazonaws.com/h2o/rel-turchin/9/Python/h2o-3.8.2.9-py2.py3-none-any.whl
python setup.py develop