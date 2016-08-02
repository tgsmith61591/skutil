from __future__ import print_function, division
import h2o
import warnings
from h2o.frame import H2OFrame
from skutil.h2o.select import *
from sklearn.datasets import load_iris
import pandas as pd


# if we can't start an h2o instance, let's just pass all these tests
try:
	h2o.init()

	iris = load_iris()
	X = pd.DataFrame.from_records(data=iris.data, columns=iris.feature_names)
	X = H2OFrame.from_python(X)
	X.columns = X.columns.values

except Exception as e:
	warnings.warn('could not successfully start H2O instance', UserWarning)
	X = None


def test_multicollinearity():
	if X:
		filterer = H2OMulticollinearityFilterer(threshold=0.6)
		x = filterer.fit_transform(X)
	else:
		pass