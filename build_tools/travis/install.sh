#!/bin/bash

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
