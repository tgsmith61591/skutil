from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from skutil.preprocessing import *


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
