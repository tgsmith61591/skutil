from __future__ import print_function
import numpy as np
import pandas as pd
import warnings
from skutil.odr import QRDecomposition
from skutil.feature_selection import combos
from numpy.testing import (assert_array_equal, assert_almost_equal, assert_array_almost_equal)
from sklearn.datasets import load_iris
from skutil.feature_selection import *
from skutil.utils.tests.utils import assert_fails


## Def data for testing
iris = load_iris()
X = pd.DataFrame.from_records(data = iris.data, columns = iris.feature_names)
y = np.array(
	[[ 0.41144380 ,  1  ,  2],
	[ 0.20002043  ,  1  ,  2],
	[ 1.77615427  ,  1  ,  2],
	[-0.88393494  ,  1  ,  2],
	[ 1.03053577  ,  1  ,  2],
	[ 0.10348028  ,  1  ,  2],
	[-2.63301012  ,  1  ,  2],
	[-0.09411449  ,  1  ,  2],
	[-0.37090572  ,  1  ,  2],
	[ 3.67912713  ,  1  ,  2],
	[-1.11889106  ,  1  ,  2],
	[-0.16339222  ,  1  ,  2],
	[-1.68642994  ,  1  ,  2],
	[ 0.01475935  ,  1  ,  2],
	[-0.71178462  ,  1  ,  2],
	[-0.07375506  ,  1  ,  2],
	[ 1.67680864  ,  1  ,  2],
	[ 1.08437155  ,  1  ,  2],
	[ 0.42135106  ,  1  ,  2],
	[ 0.23891404  ,  1  ,  2],
	[-0.67025244  ,  1  ,  2],
	[-0.74780315  ,  1  ,  2],
	[ 1.53795249  ,  1  ,  2],
	[ 2.24940846  ,  1  ,  2],
	[-1.33077619  ,  1  ,  2],
	[-1.23597935  ,  1  ,  2],
	[-1.10603714  ,  1  ,  2],
	[ 0.06115450  ,  1  ,  2],
	[ 2.33540909  ,  1  ,  2],
	[-0.20694138  ,  1  ,  2],
	[ 1.34077119  ,  1  ,  2],
	[ 1.19347871  ,  1  ,  2],
	[ 0.23480672  ,  1  ,  2],
	[-1.48948507  ,  1  ,  2],
	[ 1.00529241  ,  1  ,  2],
	[ 1.72366825  ,  1  ,  2],
	[ 4.14722011  ,  1  ,  2],
	[-0.66620106  ,  1  ,  2],
	[ 1.45597498  ,  1  ,  2],
	[-0.39631565  ,  1  ,  2],
	[ 0.80971318  ,  1  ,  2],
	[ 0.71547389  ,  1  ,  2],
	[-0.17342195  ,  1  ,  2],
	[-1.18399696  ,  1  ,  2],
	[ 1.77178761  ,  1  ,  2],
	[-0.94494203  ,  1  ,  2],
	[-1.47486102  ,  1  ,  2],
	[ 0.35748476  ,  1  ,  2],
	[-1.29096329  ,  1  ,  2],
	[ 0.61611613  ,  1  ,  2],
	[ 0.92048145  ,  1  ,  2],
	[ 0.56870638  ,  1  ,  2],
	[ 0.06455932  ,  1  ,  2],
	[ 0.20987525  ,  1  ,  2],
	[ 0.60659611  ,  1  ,  2],
	[ 0.43715853  ,  1  ,  2],
	[-0.06136566  ,  1  ,  2],
	[-1.75842912  ,  1  ,  2],
	[-1.03648110  ,  1  ,  2],
	[-2.72359130  ,  1  ,  2],
	[ 1.80935039  ,  1  ,  2],
	[ 1.27240976  ,  1  ,  2],
	[-2.74477429  ,  1  ,  2],
	[ 0.34654907  ,  1  ,  2],
	[-1.90913461  ,  1  ,  2],
	[-3.42357727  ,  1  ,  2],
	[-1.28010016  ,  1  ,  2],
	[ 3.17908952  ,  1  ,  2],
	[-1.54936824  ,  1  ,  2],
	[-1.37700148  ,  1  ,  2],
	[ 0.41881648  ,  1  ,  2],
	[ 0.22241198  ,  1  ,  2],
	[-0.78960214  ,  1  ,  2],
	[ 0.28105782  ,  1  ,  2],
	[ 2.58817288  ,  1  ,  2],
	[ 0.88948762  ,  1  ,  2],
	[ 1.25544532  ,  1  ,  2],
	[-0.50838470  ,  1  ,  2],
	[ 1.13062450  ,  1  ,  2],
	[ 2.41422771  ,  1  ,  2],
	[-0.86262900  ,  1  ,  2],
	[-2.16937438  ,  1  ,  2],
	[-0.57198596  ,  1  ,  2],
	[-0.07023331  ,  1  ,  2],
	[ 2.34332545  ,  1  ,  2],
	[-0.71221171  ,  1  ,  2],
	[-0.18585408  ,  1  ,  2],
	[-2.81586156  ,  1  ,  2],
	[-0.86356504  ,  1  ,  2],
	[-0.01727535  ,  1  ,  2],
	[-3.15966711  ,  1  ,  2],
	[-0.84387501  ,  1  ,  2],
	[-1.73471525  ,  1  ,  2],
	[ 2.74981014  ,  1  ,  2],
	[ 0.28114847  ,  1  ,  2],
	[-1.66076523  ,  1  ,  2],
	[-0.62953126  ,  1  ,  2],
	[-1.90627065  ,  1  ,  2],
	[-0.38711584  ,  1  ,  2],
	[ 0.84237942  ,  1  ,  2],
	[ 0.35066088  ,  1  ,  2],
	[-0.47789289  ,  1  ,  2],
	[-1.72405119  ,  1  ,  2],
	[ 0.78935913  ,  1  ,  2],
	[ 3.03339661  ,  1  ,  2],
	[-2.68912845  ,  1  ,  2],
	[ 0.22600963  ,  1  ,  2],
	[ 3.72403170  ,  1  ,  2],
	[ 0.25115682  ,  1  ,  2],
	[ 2.51450226  ,  1  ,  2],
	[-2.52882830  ,  1  ,  2],
	[-1.60614569  ,  1  ,  2],
	[-0.74095083  ,  1  ,  2],
	[ 0.78927670  ,  1  ,  2],
	[ 2.35876839  ,  1  ,  2],
	[ 0.84019398  ,  1  ,  2],
	[-2.49124992  ,  1  ,  2],
	[-1.36854708  ,  1  ,  2],
	[ 0.59393289  ,  1  ,  2],
	[-0.82345534  ,  1  ,  2],
	[ 1.16502458  ,  1  ,  2],
	[-0.28916165  ,  1  ,  2],
	[ 0.56981198  ,  1  ,  2],
	[ 1.26863563  ,  1  ,  2],
	[-2.88717380  ,  1  ,  2],
	[ 0.01525054  ,  1  ,  2],
	[-1.62951432  ,  1  ,  2],
	[ 0.45031432  ,  1  ,  2],
	[ 0.75238069  ,  1  ,  2],
	[ 0.73113016  ,  1  ,  2],
	[ 1.52144045  ,  1  ,  2],
	[ 0.54123604  ,  1  ,  2],
	[-3.18827503  ,  1  ,  2],
	[-0.31185831  ,  1  ,  2],
	[ 0.77786948  ,  1  ,  2],
	[ 0.96769255  ,  1  ,  2],
	[ 2.01435274  ,  1  ,  2],
	[-0.86995262  ,  1  ,  2],
	[ 1.63125106  ,  1  ,  2],
	[-0.49056004  ,  1  ,  2],
	[-0.17913921  ,  1  ,  2],
	[ 1.55363112  ,  1  ,  2],
	[-1.83564770  ,  1  ,  2],
	[-1.22079526  ,  1  ,  2],
	[-1.69420452  ,  1  ,  2],
	[ 0.54327665  ,  1  ,  2],
	[-2.07883607  ,  1  ,  2],
	[ 0.52608135  ,  1  ,  2],
	[-0.89157428  ,  1  ,  2],
	[-1.07971739  ,  1  ,  2]])

Z = pd.DataFrame.from_records(data=y, columns=['A','B','C'])


def test_feature_dropper():
	transformer = FeatureDropper().fit(X)
	assert not transformer.cols
	assert transformer.transform(X).shape[1] == 4
	assert FeatureDropper(['sepal length (cm)', 'sepal width (cm)']).fit_transform(X).shape[1] == 2

	# test the selective mixin
	assert transformer.get_features() is None


def test_feature_selector():
	transformer = FeatureRetainer().fit(X)
	assert transformer.transform(X).shape[1] == 4

	cols = ['sepal length (cm)', 'sepal width (cm)']
	transformer = FeatureRetainer(cols=cols).fit(X)
	assert transformer.transform(X).shape[1] == 2

	# test the selective mixin
	assert isinstance(transformer.get_features(), list)


def test_multi_collinearity():
	transformer = MulticollinearityFilterer()

	# Test fit_transform
	x = transformer.fit_transform(X)
	assert x.shape[1] == 3

	col_nms = x.columns
	assert col_nms[0] == 'sepal length (cm)'
	assert col_nms[1] == 'sepal width (cm)'
	assert col_nms[2] == 'petal width (cm)'
	assert len(transformer.drop) == 1

	# test the selective mixin
	assert transformer.get_features() is None

	# Test fit, then transform
	transformer = MulticollinearityFilterer().fit(X)
	x = transformer.transform(X)
	assert x.shape[1] == 3

	col_nms = x.columns
	assert col_nms[0] == 'sepal length (cm)'
	assert col_nms[1] == 'sepal width (cm)'
	assert col_nms[2] == 'petal width (cm)'
	assert len(transformer.drop) == 1

	# Check as_df false
	transformer.as_df = False
	assert isinstance(transformer.transform(X), np.ndarray)


def test_nzv_filterer():
	transformer = NearZeroVarianceFilterer().fit(X)
	assert transformer.drop is None

	y = X.copy()
	y['zeros'] = np.zeros(150)

	transformer = NearZeroVarianceFilterer().fit(y)
	assert len(transformer.drop) == 1
	assert transformer.drop[0] == 'zeros'
	assert transformer.transform(y).shape[1] == 4

	# test the selective mixin
	assert transformer.get_features() is None

	# see what happens if we have a nan or inf in the mix:
	a = pd.DataFrame.from_records(data=np.reshape(np.arange(25), (5,5)))
	a.iloc[0,0] = np.inf
	a.iloc[0,1] = np.nan

	# expect a valueerror
	assert_fails(NearZeroVarianceFilterer().fit, ValueError, a)


def test_feature_dropper_warning():
	x = np.array([
			[1,2,3],
			[1,2,3],
			[1,2,4]
		])

	df = pd.DataFrame.from_records(data=x, columns=['a','b','c'])

	# catch the warning
	with warnings.catch_warnings(record=True) as w:
		warnings.simplefilter('always')
		FeatureDropper(cols=['d']).fit_transform(df)
		assert len(w) == 1

def test_linear_combos():
	lcf = LinearCombinationFilterer().fit(Z)
	assert_array_equal(lcf.drop, ['C'])

	z = lcf.transform(Z)
	assert_array_equal(z.columns.values, ['A','B'])
	assert (z.B == 1).all()

	# test on no linear combos
	lcf = LinearCombinationFilterer(cols=['A','B']).fit(Z)
	assert not lcf.drop
	assert Z.equals(lcf.transform(Z))

	# test too few features
	assert_fails(LinearCombinationFilterer(cols=['A']).fit, ValueError, Z)


def test_sparsity():
	x = np.array([
			[1,      2,      3],
			[1,      np.nan, np.nan],
			[1,      2,      np.nan]
		])

	df = pd.DataFrame.from_records(data=x, columns=['a','b','c'])


	# test at .33 level
	filt = SparseFeatureDropper(threshold=0.3).fit(df)
	assert len(filt.drop) == 2
	assert all([i in filt.drop for i in ('b','c')]), 'expected "b" and "c" but got %s' % ', '.join(filt.drop)

	# test at 2/3 level
	filt = SparseFeatureDropper(threshold=0.6).fit(df)
	assert len(filt.drop) == 1
	assert 'c' in filt.drop, 'expected "c" but got %s' % filt.drop

	# test with a bad value
	assert_fails(SparseFeatureDropper(threshold=  1.0).fit, ValueError, df)
	assert_fails(SparseFeatureDropper(threshold= -0.1).fit, ValueError, df)
	assert_fails(SparseFeatureDropper(threshold=  'a').fit, ValueError, df)



def test_enumLC():
	Y = np.array([
			[1, 2, 3 ],
			[4, 5, 6 ],
			[7, 8, 9 ],
			[10,11,12]
		])

	a, b = combos._enumLC(QRDecomposition(Y))[0], np.array([2,0,1])
	assert (a==b).all(), 'should be [2,0,1] but got %s' % a
	assert not combos._enumLC(QRDecomposition(iris.data))

	assert_array_equal( combos._enumLC(QRDecomposition(y))[0], np.array([2, 1]) )
