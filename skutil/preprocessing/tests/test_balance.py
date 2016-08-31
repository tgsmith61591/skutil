from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from skutil.preprocessing import *
from skutil.preprocessing.balance import _BaseBalancer
from skutil.utils.tests.utils import assert_fails
import warnings


## Def data for testing
iris = load_iris()
X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
X['target'] = iris.target


def _get_three_results(sampler):
	x = X.iloc[:60] # 50 zeros, 10 ones
	y = pd.concat([x, X.iloc[140:150]])
	a,b = sampler.balance(x), sampler.balance(y)
	sampler.ratio = 0.2
	return a, b, sampler.balance(y)

def test_oversample():
	a, b, c = _get_three_results(OversamplingClassBalancer(y='target', ratio=0.5))

	expected_1_ct = 25
	cts = a.target.value_counts()
	assert cts[1] == expected_1_ct

	cts = b.target.value_counts()
	assert cts[1] == expected_1_ct
	assert cts[2] == expected_1_ct

	expected_2_ct = 10
	cts = c.target.value_counts()
	assert cts[1] == expected_2_ct
	assert cts[2] == expected_2_ct

	# test what happens when non-string passed as col name
	failed = False
	try:
		OversamplingClassBalancer(y=1).balance(X)
	except ValueError as v:
		failed = True
	assert failed

	# test with too many classes
	Y = X.copy()
	Y['class'] = np.arange(Y.shape[0])
	failed = False
	try:
		OversamplingClassBalancer(y='class').balance(Y)
	except ValueError as v:
		failed = True
	assert failed

	# test with one class
	Y['class'] = np.zeros(Y.shape[0])
	failed = False
	try:
		OversamplingClassBalancer(y='class').balance(Y)
	except ValueError as v:
		failed = True
	assert failed

	# test with bad ratio
	for r in [0.0, 1.1, 'string']:
		failed = False
		try:
			OversamplingClassBalancer(y='target', ratio=r).balance(X)
		except ValueError as v:
			failed=True
		assert failed


	# test where two classes are equally represented, and one has only a few
	Y = X.iloc[:105]
	d = OversamplingClassBalancer(y='target', ratio=1.0).balance(Y)
	assert d.shape[0] == 150

	cts= d.target.value_counts()
	assert cts[0] == 50
	assert cts[1] == 50
	assert cts[2] == 50

def test_oversample_warning():
	x = np.array([
			[1,2,3],
			[1,2,3],
			[1,2,4]
		])

	df = pd.DataFrame.from_records(data=x, columns=['a','b','c'])

	# catch the warning
	with warnings.catch_warnings(record=True) as w:
		warnings.simplefilter('always')
		OversamplingClassBalancer(y='c').balance(df)
		assert len(w) == 1

def test_smote_error():
	x = np.array([
			[1,2,3],
			[1,2,3],
			[1,2,4]
		])

	df = pd.DataFrame.from_records(data=x, columns=['a','b','c'])
	failed = False
	try:
		SMOTEClassBalancer(y='c').balance(df)
	except ValueError as e:
		failed = True
	assert failed


def test_smote():
	a, b, c = _get_three_results(SMOTEClassBalancer(y='target', ratio=0.5))

	expected_1_ct = 25
	cts = a.target.value_counts()
	assert cts[1] == expected_1_ct

	cts = b.target.value_counts()
	assert cts[1] == expected_1_ct
	assert cts[2] == expected_1_ct

	expected_2_ct = 10
	cts = c.target.value_counts()
	assert cts[1] == expected_2_ct
	assert cts[2] == expected_2_ct

def test_undersample():
	# since all classes are equal, should be no change here
	b = UndersamplingClassBalancer(y='target').balance(X)
	assert b.shape[0] == X.shape[0]

	x = X.iloc[:60] # 50 zeros, 10 ones
	b = UndersamplingClassBalancer(y='target', ratio=0.5).balance(x)

	assert b.shape[0] == 30
	cts = b.target.value_counts()
	assert cts[0] == 20
	assert cts[1] == 10

	b = UndersamplingClassBalancer(y='target', ratio=0.25).balance(x)

	assert b.shape[0] == 50
	cts = b.target.value_counts()
	assert cts[0] == 40
	assert cts[1] == 10

def test_superclass_not_implemented():
	# anon balancer
	class AnonBalancer(_BaseBalancer):
		def __init__(self, ratio=0.2, y=None, as_df=True):
			super(AnonBalancer, self).__init__(ratio, y, as_df)

		def balance(self, X):
			return super(AnonBalancer, self).balance(X)

	assert_fails(AnonBalancer().balance, NotImplementedError, X)




