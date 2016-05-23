from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from skutil.preprocessing import SMOTEStratifiedBalancer


__all__ = [
	'test_smote'
]

## Def data for testing
iris = load_iris()
X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
X['target'] = iris.target

def test_smote():
	x = X.iloc[:60] # 50 zeros, 10 ones
	t = SMOTEStratifiedBalancer(y='target', ratio=0.5).transform(x)

	expected_1_ct = 25
	cts = t.target.value_counts()
	assert cts[1] == expected_1_ct
