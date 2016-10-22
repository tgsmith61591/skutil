#!/bin/bash

case "${COMBINATION}" in
    python-2-7-sklearn-0-17)
        pip install scikit-learn==0.17.1
        ;;
    python-2-7-sklearn-0-18)
        pip install scikit-learn==0.18
        ;;
esac