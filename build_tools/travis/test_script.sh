#!/bin/bash

set -e

mkdir -p $TEST_DIR
cp setup.cfg $TEST_DIR
cd $TEST_DIR

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"
python -c "import pandas; print('pandas %s' % pandas.__version__)"
python -c "import sklearn; print('sklearn %s' % sklearn.__version__)"
python -c "import multiprocessing as mp; print('%d CPUs' % mp.cpu_count())"

# Skip tests that require large downloads
export PYNORM_SKIP_NETWORK_TESTS=1

if [[ "$COVERAGE" == "true" ]]; then
    nosetests -s --with-coverage --with-timer --timer-top-n 20 skutil
else
    nosetests -s --with-timer --timer-top-n 20 skutil
fi

# Is dir still empty?
ls -ltra

# Test doc
cd $CACHED_BUILD_DIR/skutil
make test-doc test-sphinxext
