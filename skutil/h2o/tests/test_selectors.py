from __future__ import print_function, division
import h2o
import warnings
import numpy as np
from h2o.frame import H2OFrame
from skutil.h2o.select import *
from sklearn.datasets import load_iris
import pandas as pd


iris = load_iris()
F = pd.DataFrame.from_records(data=iris.data, columns=iris.feature_names)


# if we can't start an h2o instance, let's just pass all these tests
def test_h2o():
	try:
		h2o.init(ip='localhost', port=54321)
		X = H2OFrame.from_python(F, header=1, column_names=F.columns.tolist())

		# weirdness sometimes.
		if not 'sepal length (cm)' in X.columns:
			X.columns = F.columns.tolist()

		if X.shape[0] > F.shape[0]:
			X = X[1:,:]
	except Exception as e:
		warnings.warn('could not successfully start H2O instance', UserWarning)
		X = None


	def catch_warning_assert_thrown(fun, kwargs):
		with warnings.catch_warnings(record=True) as w:
			warnings.simplefilter("always")

			ret = fun(**kwargs)
			assert len(w) > 0 if X is None else True, 'expected warning to be thrown'
			return ret



	def multicollinearity():
		# one way or another, we can initialize it
		filterer = catch_warning_assert_thrown(H2OMulticollinearityFilterer, {'threshold':0.6})
		assert filterer.min_version == '3.8.3'
		assert not filterer.max_version

		if X is not None:
			x = filterer.fit_transform(X)
			assert x.shape[1] == 2
		else:
			pass


	def nzv():
		filterer = catch_warning_assert_thrown(H2ONearZeroVarianceFilterer, {'threshold':1e-8})
		assert filterer.min_version == '3.8.3'
		assert not filterer.max_version

		# let's add a zero var feature to F
		f = F.copy()
		f['zerovar'] = np.zeros(F.shape[0])

		try:
			Y = H2OFrame.from_python(f, header=1, column_names=f.columns)
			# weirdness sometimes.
			if not 'sepal length (cm)' in Y.columns:
				Y.columns = f.columns.tolist()

			if Y.shape[0] > f.shape[0]:
				Y = Y[1:,:]
		except Exception as e:
			Y = None


		if Y is not None:
			y = filterer.fit_transform(Y)
			assert len(filterer.drop_) == 1
			assert y.shape[1] == 4
		else:
			pass

	# run them
	multicollinearity()
	nzv()


