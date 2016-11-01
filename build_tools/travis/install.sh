#!/bin/bash

echo -e "\nA : `date` \n"
case "${TRAVIS_PYTHON_VERSION}" in
    "2.7")
        wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O $HOME/miniconda.sh;
        ;;
    "3.5")
        wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $HOME/miniconda.sh;
        ;;
esac

echo -e "\nB : `date` \n"
bash $HOME/miniconda.sh -b -p $HOME/miniconda
echo -e "\nC : `date` \n"
export PATH="$HOME/miniconda/bin:$PATH"
echo -e "\nD : `date` \n"
hash -r
echo -e "\nE : `date` \n"
conda config --set always_yes yes --set changeps1 no
echo -e "\nF : `date` \n"
conda update -q conda

# Useful for debugging any issues with conda
echo -e "\nG : `date` \n"
conda info -a

# Dependencies
echo -e "\nH : `date` \n"
conda create -q -n test-environment python=$TRAVIS_PYTHON_VERSION numpy scipy pandas coverage cython
echo -e "\nI : `date` \n"
source activate test-environment

# Coverage packages
echo -e "\nJ : `date` \n"
if [[ "$TRAVIS_PYTHON_VERSION" == "2.7" ]]; then
      conda install --yes -c dan_blanchard python-coveralls nose-cov;
fi
echo -e "\nK : `date` \n"
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
echo -e "\nL : `date` \n"
pip install coveralls
echo -e "\nM : `date` \n"
pip install http://h2o-release.s3.amazonaws.com/h2o/rel-turchin/9/Python/h2o-3.8.2.9-py2.py3-none-any.whl
echo -e "\nN : `date` \n"
python setup.py develop