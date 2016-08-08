from __future__ import print_function, division
import warnings
import numpy as np
import h2o
from h2o.frame import H2OFrame
from h2o.estimators import H2ORandomForestEstimator
from skutil.h2o.select import *
from skutil.h2o.pipeline import *
from sklearn.datasets import load_iris
import pandas as pd

# for split
try:
	from sklearn.model_selection import train_test_split
except ImportError as i:
	from sklearn.cross_validation import train_test_split


iris = load_iris()
F = pd.DataFrame.from_records(data=iris.data, columns=iris.feature_names)



def new_h2o_frame(X):
	Y = H2OFrame.from_python(X, header=1, column_names=X.columns.tolist())
	# weirdness sometimes.
	if not 'sepal length (cm)' in Y.columns:
		Y.columns = X.columns.tolist()

	if Y.shape[0] > X.shape[0]:
		Y = Y[1:,:]
	return  Y


# if we can't start an h2o instance, let's just pass all these tests
def test_h2o():
	try:
		h2o.init(ip='localhost', port=54321) # this might throw a warning
		X = new_h2o_frame(F)
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

		# test some exceptions...
		if X is not None:
			failed = False
			try:
				filterer.fit(F) # thrown here
			except TypeError as t:
				failed = True
			assert failed, 'Expected failure when passing a dataframe'
		else:
			pass

		# test with a target feature
		if X is not None:
			tgt = 'sepal length (cm)'
			new_filterer = catch_warning_assert_thrown(H2OMulticollinearityFilterer, {'threshold':0.6, 'target_feature':tgt})
			x = new_filterer.fit_transform(X)

			# h2o throws weird error because it override __contains__, so use the comprehension
			assert tgt in [c for c in x.columns], 'target feature was accidentally dropped...'

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
			Y = new_h2o_frame(f)
		except Exception as e:
			Y = None


		if Y is not None:
			y = filterer.fit_transform(Y)
			assert len(filterer.drop_) == 1
			assert y.shape[1] == 4
		else:
			pass

		# test with a target feature
		if X is not None:
			tgt = 'sepal length (cm)'
			new_filterer = catch_warning_assert_thrown(H2ONearZeroVarianceFilterer, {'threshold':1e-8, 'target_feature':tgt})
			y = new_filterer.fit_transform(Y)

			assert len(new_filterer.drop_) == 1
			assert y.shape[1] == 4

			# h2o throws weird error because it override __contains__, so use the comprehension
			assert tgt in [c for c in y.columns], 'target feature was accidentally dropped...'

		else:
			pass

	def pipeline():
		f = F.copy()

		# do split
		X_train, X_test, y_train, y_test = train_test_split(f, iris.target, train_size=0.7)
		
		# add the y into the matrix for h2o's sake -- pandas will throw a warning here...
		with warnings.catch_warnings(record=True) as w:
			warnings.simplefilter("ignore")
			X_train['species'] = y_train
			X_test['species'] = y_test

		try:
			train = new_h2o_frame(X_train)
			test  = new_h2o_frame(X_test)
		except Exception as e:
			train = None
			test  = None


		if train is not None:
			# define pipe
			pipe = H2OPipeline([
					('nzv', H2ONearZeroVarianceFilterer()),
					('mc',  H2OMulticollinearityFilterer(threshold=0.9)),
					('rf',  H2ORandomForestEstimator())
				], 
				feature_names=F.columns.tolist(),
				target_feature='species'
			)

			# fit pipe...
			pipe.fit(train)

			# try predicting
			pipe.predict(test)
		else:
			pass


	# run them
	multicollinearity()
	nzv()
	pipeline()


