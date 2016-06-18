from __future__ import print_function, division
import numpy as np
from numpy.testing import (assert_array_equal, assert_almost_equal, assert_array_almost_equal)
from sklearn.datasets import load_iris
from skutil.odr import *

X = load_iris().data

def test_qr():
	# test just the decomp first
	q = QRDecomposition(X)
	aux = q.qraux
	assert_array_almost_equal(aux, np.array([ 1.07056264,  1.0559255,   1.03857984,  1.04672249]))

	# test that we can get the rank
	assert q.get_rank() == 4

	# test that we can get the R matrix and that it's rank 4
	assert q.get_R_rank() == 4

	# next, let's test that we can get the coefficients:
	#q.get_coef(X)
