from __future__ import print_function
import pandas as pd
import numpy as np
from numpy.random import choice
from sklearn.datasets import load_iris
from skutil.preprocessing import *
import warnings

def _random_X(m,n,cols):
	return pd.DataFrame.from_records(
		data=np.random.rand(m,n), 
		columns=cols)


def test_bagged_imputer():
	nms = ['a','b','c','d','e','f','g','h','i','j']
	X = _random_X(500,10,nms)
	Z = _random_X(200,10,nms)

	# test works on 100 full...
	imputer = BaggedImputer()
	imputed = imputer.fit_transform(X)
	null_ct = imputed.isnull().sum().sum()
	assert null_ct == 0, 'expected no nulls but got %i' % null_ct

	# operates in place
	def fill_with_nas(x):
		# make some of them NaN - only 10% tops
		ceil = int(x.shape[0] * 0.1)
		for col in nms:
			n_missing = max(1, choice(ceil)) # at least one in each
			missing_idcs = choice(range(x.shape[0]), n_missing)

			# fill with some NAs
			x.loc[missing_idcs, col] = np.nan

	# throw some NAs in
	fill_with_nas(X)
	fill_with_nas(Z)

	# ensure there are NAs now
	null_ct = X.isnull().sum().sum()
	assert null_ct > 0, 'expected some missing values but got %i' % null_ct

	# now fit the imputer on ALL with some missing:
	imputed = imputer.fit_transform(X)
	null_ct = imputed.isnull().sum().sum()
	assert null_ct == 0, 'expected no nulls but got %i' % null_ct

	# test the transform method on new data
	z = imputer.transform(Z)
	null_ct = z.isnull().sum().sum()
	assert null_ct == 0, 'expected no nulls but got %i' % null_ct


def test_bagged_imputer_errors():
	nms = ['a','b','c','d','e']
	X = _random_X(500,5,nms)

	# ensure works on just fit
	BaggedImputer().fit(X)

	# make all of a NaN
	X.a = np.nan

	# test that all nan will fail
	failed = False
	try:
		imputer = BaggedImputer().fit(X)
	except ValueError as v:
		failed = True
	assert failed, 'Expected imputation on fully missing feature to fail'

	# test on just one col
	failed = False
	try:
		u = pd.DataFrame()
		u['b'] = X.b
		imputer = BaggedImputer().fit(u)
	except ValueError as v:
		failed = True
	assert failed, 'Expected fitting on one col to fail'

	# test with a categorical column
	f = ['a' if choice(4)%2==0 else 'b' for i in range(X.shape[0])]
	X['f'] = f
	failed = False
	try:
		imputer = BaggedImputer().fit(X[['d','e','f']])
	except ValueError as v:
		failed = True
	assert failed, 'Expected imputation with categorical feature to fail'

