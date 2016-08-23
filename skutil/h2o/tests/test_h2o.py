from __future__ import print_function, division
import warnings
import numpy as np
import pandas as pd
import time

import h2o
from h2o.frame import H2OFrame
from h2o.estimators import (H2ORandomForestEstimator,
							H2OGeneralizedLinearEstimator,
							H2OGradientBoostingEstimator,
							H2ODeepLearningEstimator,
							H2OPrincipalComponentAnalysisEstimator)

from skutil.h2o import from_pandas, from_array
from skutil.h2o.base import *
from skutil.h2o.select import *
from skutil.h2o.pipeline import *
from skutil.h2o.grid_search import *
from skutil.h2o.grid_search import _as_numpy
from skutil.utils import load_iris_df
from skutil.utils.tests.utils import assert_fails
from skutil.feature_selection import NearZeroVarianceFilterer
from skutil.h2o.split import (check_cv, H2OKFold, 
	H2OStratifiedKFold, h2o_train_test_split, 
	_validate_shuffle_split_init, _validate_shuffle_split,
	_val_y)

from sklearn.datasets import load_iris, load_boston
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import randint, uniform

from numpy.random import choice

# for split
try:
	from sklearn.model_selection import train_test_split
except ImportError as i:
	from sklearn.cross_validation import train_test_split


def new_h2o_frame(X):
	Y = H2OFrame.from_python(X, header=1, 
		column_names=X.columns.tolist())
	return  Y


def new_estimators():
	"""Returns a tuple of newly initialized estimators to test all of them
	with the skutil framework. This ensures it will work with all the estimators...
	"""
	return (
			H2ORandomForestEstimator(),
			#H2OGeneralizedLinearEstimator(family='multinomial'),
			H2OGradientBoostingEstimator(distribution='multinomial'),
			H2ODeepLearningEstimator(distribution='multinomial')
		)


# if we can't start an h2o instance, let's just pass all these tests
def test_h2o():
	iris = load_iris()
	F = pd.DataFrame.from_records(data=iris.data, columns=iris.feature_names)
	#F = load_iris_df(include_tgt=False)
	X = None

	try:
		h2o.init()
		#h2o.init(ip='localhost', port=54321) # this might throw a warning

		# sleep before trying this:
		time.sleep(10)
		X = new_h2o_frame(F)
	except Exception as e:
		#raise #for debugging

		# if we can't start on localhost, try default (127.0.0.1)
		# we'll try it a few times, since H2O is SUPER finnicky
		max_tries = 3
		count = 0

		while (count < max_tries) and (X is None):
			try:
				h2o.init()
				X = new_h2o_frame(F)
			except Exception as e:
				count += 1

		if X is None:
			warnings.warn('could not successfully start H2O instance, tried %d times' % max_tries, UserWarning)


	def catch_warning_assert_thrown(fun, kwargs):
		with warnings.catch_warnings(record=True) as w:
			warnings.simplefilter("always")

			ret = fun(**kwargs)
			#assert len(w) > 0 if X is None else True, 'expected warning to be thrown'
			return ret



	def multicollinearity():
		# one way or another, we can initialize it
		filterer = catch_warning_assert_thrown(H2OMulticollinearityFilterer, {'threshold':0.6})
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
		targ = iris.target
		targ = ['a' if x == 0 else 'b' if x == 1 else 'c' for x in targ]

		# do split
		X_train, X_test, y_train, y_test = train_test_split(f, targ, train_size=0.7)
		
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
			for estimator in new_estimators():
				# define pipe
				pipe = H2OPipeline([
						('nzv', H2ONearZeroVarianceFilterer()),
						('mc',  H2OMulticollinearityFilterer(threshold=0.9)),
						('est', estimator)
					], 
					feature_names=F.columns.tolist(),
					target_feature='species'
				)

				# fit pipe...
				pipe.fit(train)

				# try predicting
				pipe.predict(test)

				# coverage:
				fe = pipe._final_estimator
				ns = pipe.named_steps


			# test some failures -- first, Y
			for y in [None, 1]:
				pipe = H2OPipeline([
						('nzv', H2ONearZeroVarianceFilterer()),
						('mc',  H2OMulticollinearityFilterer(threshold=0.9)),
						('est', H2OGradientBoostingEstimator(distribution='multinomial'))
					], 
					feature_names=F.columns.tolist(),
					target_feature=y
				)
				
				excepted = False
				try:
					pipe.fit(train)
				except (TypeError, ValueError, EnvironmentError) as e:
					excepted = True
				assert excepted, 'expected failure for y=%s'%str(y)


			# now X
			pipe = H2OPipeline([
					('nzv', H2ONearZeroVarianceFilterer()),
					('mc',  H2OMulticollinearityFilterer(threshold=0.9)),
					('est', H2OGradientBoostingEstimator(distribution='multinomial'))
				], 
				feature_names=1,
				target_feature='species'
			)

			assert_fails(pipe.fit, TypeError, train)



			# now non-unique names
			failed = False
			try:
				pipe = H2OPipeline([
						('nzv', H2ONearZeroVarianceFilterer()),
						('mc',  H2OMulticollinearityFilterer(threshold=0.9)),
						('mc', H2OGradientBoostingEstimator(distribution='multinomial'))
					], 
					feature_names=F.columns.tolist(),
					target_feature='species'
				)

				# won't even get here...
				pipe.fit(train)
			except ValueError as v:
				failed = True
			assert failed


			# fails for non-h2o transformers
			failed = False
			try:
				pipe = H2OPipeline([
						('nzv', NearZeroVarianceFilterer()),
						('mc',  H2OMulticollinearityFilterer(threshold=0.9)),
						('est', H2OGradientBoostingEstimator(distribution='multinomial'))
					], 
					feature_names=F.columns.tolist(),
					target_feature='species'
				)

				# won't even get here...
				pipe.fit(train)
			except TypeError as t:
				failed = True
			assert failed



			# type error for non-h2o estimators
			failed = False
			try:
				pipe = H2OPipeline([
						('nzv', H2ONearZeroVarianceFilterer()),
						('mc',  H2OMulticollinearityFilterer(threshold=0.9)),
						('est', RandomForestClassifier)
					], 
					feature_names=F.columns.tolist(),
					target_feature='species'
				)

				# won't even get here...
				pipe.fit(train)
			except TypeError as t:
				failed =True
			assert failed
			
		else:
			pass


	def grid():
		# test as_numpy
		assert_fails(_as_numpy, ValueError, F) # fails if not 1 col


		f = F.copy()
		targ = iris.target
		targ = ['a' if x == 0 else 'b' if x == 1 else 'c' for x in targ]
		f['species'] = targ

		# shuffle the rows
		f = f.iloc[np.random.permutation(np.arange(f.shape[0]))]

		# try uploading...
		try:
			frame = new_h2o_frame(f)
		except Exception as e:
			frame = None

		def get_param_grid(est):
			if isinstance(est, (H2ORandomForestEstimator, H2OGradientBoostingEstimator)):
				return {
					'ntrees' : [10,20]
				}
			elif isinstance(est, H2ODeepLearningEstimator):
				return {
					'activation' : ['Tanh','Rectifier']
				}
			else:
				return {
					'standardize' : [True, False]
				}


		# most of this is for extreme coverage to make sure it all works...
		if frame is not None:
			n_folds = 2

			for is_random in [False, True]:
				for estimator in new_estimators():
					for do_pipe in [False, True]:
						for iid in [False, True]:
							for verbose in [2, 3]:
								for scoring in ['accuracy_score', 'bad', None, accuracy_score]:

									# should we shuffle?
									do_shuffle = choice([True, False])

									# just for coverage...
									which_cv = choice([
										n_folds, 
										H2OKFold(n_folds=n_folds, shuffle=do_shuffle)
									])


									# get which module to use
									if is_random:
										grid_module = H2ORandomizedSearchCV
									else:
										grid_module = H2OGridSearchCV


									if not do_pipe:
										# we're just testing the search on actual estimators
										grid = grid_module(estimator=estimator,
											feature_names=F.columns.tolist(), target_feature='species',
											param_grid=get_param_grid(estimator),
											scoring=scoring, iid=iid, verbose=verbose,
											cv=which_cv)
									else:

										# pipify -- the feature names, etc., will be set in the grid
										pipe = H2OPipeline([
												('nzv', H2ONearZeroVarianceFilterer()),
												('est', estimator)
											])

										# determine which params to use
										# we'll just use a NZV filter and tinker with the thresh
										if is_random:
											params = {
												'nzv__threshold' : uniform(1e-6, 0.0025)
											}
										else:
											params = {
												'nzv__threshold' : [1e-6, 1e-8]
											}

										grid = grid_module(pipe, param_grid=params,
											feature_names=F.columns.tolist(), target_feature='species',
											scoring=scoring, iid=iid, verbose=verbose,
											cv=which_cv)


									# if it's a random search CV obj, let's keep it brief
									if is_random:
										grid.n_iter = n_folds

									# sometimes we'll expect it to fail...
									expect_failure = scoring is None or (isinstance(scoring,str) and scoring in ('bad'))
									try:
										# fit the grid
										grid.fit(frame)

										# we expect the exception to be thrown
										# above, so now we set expect_failure to False
										expect_failure = False

										# predict on the grid
										p = grid.predict(frame)

										# score on the frame
										s = grid.score(frame)
									except ValueError as v:
										if expect_failure:
											pass
										else:
											raise


			# can we just fit one with a validation frame for coverage?
			pipe = H2OPipeline([
				('nzv', H2ONearZeroVarianceFilterer()),
				('est', H2OGradientBoostingEstimator())
			])

			hyper = {
				'nzv__threshold' : uniform(1e-6, 0.0025)
			}

			grid = H2ORandomizedSearchCV(pipe, param_grid=hyper,
				feature_names=F.columns.tolist(), target_feature='species',
				scoring='accuracy_score', iid=True, verbose=0, cv=2,
				validation_frame=frame)

			grid.fit(frame)



			# let's assert that this will fail for princompanalysis (or anything without 'predict')
			"""
			pipe = H2OPipeline([
				('nzv', H2ONearZeroVarianceFilterer()),
				('est', H2OPrincipalComponentAnalysisEstimator())
			])

			hyper = {
				'nzv__threshold' : uniform(1e-6, 0.0025)
			}

			grid = H2ORandomizedSearchCV(pipe, param_grid=hyper,
				feature_names=F.columns.tolist(), target_feature='species',
				scoring='accuracy_score', iid=True, verbose=0, cv=2,
				validation_frame=frame)

			assert_fails(grid.fit, ValueError, frame) # should fail due to not having 'predict'
			"""

		else:
			pass


		# let's do one pass with a stratified K fold... but we need to increase our data length, 
		# lest we lose records on the split, which will throw errors...
		f = pd.concat([f,f,f,f,f], axis=0) # times FIVE!

		# shuffle the rows
		f = f.iloc[np.random.permutation(np.arange(f.shape[0]))]

		# try uploading again...
		try:
			frame = new_h2o_frame(f)
		except Exception as e:
			frame = None

		if frame is not None:
			n_folds = 2
			for estimator in new_estimators():
				pipe = H2OPipeline([
					('nzv', H2ONearZeroVarianceFilterer()),
					('est', estimator)
				])

				params = {
					'nzv__threshold' : uniform(1e-6, 0.0025)
				}

				grid = H2ORandomizedSearchCV(pipe, param_grid=params,
					feature_names=F.columns.tolist(), target_feature='species',
					scoring='accuracy_score', n_iter=2, cv=H2OStratifiedKFold(n_folds=3, shuffle=True))

				# do fit
				grid.fit(frame)
		else:
			pass


	def anon_class():
		class H2OAnonClass100(BaseH2OFunctionWrapper):
			__min_version__ = 100.0
			__max_version__ = None

			def __init__(self):
				super(H2OAnonClass100, self).__init__(
					min_version=self.__min_version__,
					max_version=self.__max_version__)

		# assert fails for min version > current version
		assert_fails(H2OAnonClass100, EnvironmentError)



		class H2OAnonClassAny(BaseH2OFunctionWrapper):
			__min_version__ = 'any'
			__max_version__ = 0.1

			def __init__(self):
				super(H2OAnonClassAny, self).__init__(
					min_version=self.__min_version__,
					max_version=self.__max_version__)

		# assert fails for max version < current version
		assert_fails(H2OAnonClassAny, EnvironmentError)



		class H2OAnonClassPass(BaseH2OFunctionWrapper):
			def __init__(self):
				super(H2OAnonClassPass, self).__init__(
					min_version='any',
					max_version=None)

		# assert fails for max version < current version
		h = H2OAnonClassPass()
		assert h.max_version is None
		assert h.min_version == 'any'


	def cv():
		assert check_cv(None).get_n_splits() == 3
		assert_fails(check_cv, ValueError, 'not_a_valid_arg')

		# test the __repr__
		H2OStratifiedKFold().__repr__()

	def from_pandas_h2o():
		if X is not None:
			y = from_pandas(F)
			assert y.shape[0] == X.shape[0]
			assert y.shape[1] == X.shape[1]
			assert all(y.columns[i] == F.columns.tolist()[i] for i in range(y.shape[1]))
		else:
			pass

	def from_array_h2o():
		if X is not None:
			y = from_array(F.values, F.columns.tolist())
			assert y.shape[0] == X.shape[0]
			assert y.shape[1] == X.shape[1]
			assert all(y.columns[i] == F.columns.tolist()[i] for i in range(y.shape[1]))
		else:
			pass

	def split_tsts():
		# testing val_y
		assert_fails(_val_y, TypeError, 1)
		assert _val_y(None) is None
		assert isinstance(_val_y(unicode('asdf')), str)

		# testing _validate_shuffle_split_init
		assert_fails(_validate_shuffle_split_init, ValueError, **{'test_size':None,'train_size':None})     # can't both be None
		assert_fails(_validate_shuffle_split_init, ValueError, **{'test_size':1.1, 'train_size':0.0})      # if float, can't be over 1.
		assert_fails(_validate_shuffle_split_init, ValueError, **{'test_size':'string','train_size':None}) # if not float, must be int
		assert_fails(_validate_shuffle_split_init, ValueError, **{'train_size':1.1, 'test_size':0.0})      # if float, can't be over 1.
		assert_fails(_validate_shuffle_split_init, ValueError, **{'train_size':'string', 'test_size':None})# if not float, must be int
		assert_fails(_validate_shuffle_split_init, ValueError, **{'test_size':0.6, 'train_size':0.5})      # if both float, must not exceed 1.

		# testing _validate_shuffle_split
		assert_fails(_validate_shuffle_split, ValueError, **{'n_samples':10, 'test_size':11, 'train_size':0}) # test_size can't exceed n_samples
		assert_fails(_validate_shuffle_split, ValueError, **{'n_samples':10, 'train_size':11, 'test_size':0}) # train_size can't exceed n_samples
		assert_fails(_validate_shuffle_split, ValueError, **{'n_samples':10, 'train_size':5, 'test_size':6})  # sum can't exceed n_samples

		# test with train as None
		n_train, n_test = _validate_shuffle_split(n_samples=100, test_size=30, train_size=None)
		assert n_train==70

		# test with test as None
		n_train, n_test = _validate_shuffle_split(n_samples=100, test_size=None, train_size=70)
		assert n_test==30

		# test what works:
		n_train, n_test = _validate_shuffle_split(n_samples=10, test_size=3, train_size=7)
		assert n_train == 7
		assert n_test == 3

		n_train, n_test = _validate_shuffle_split(n_samples=10, test_size=0.3, train_size=0.7)
		assert n_train == 7
		assert n_test == 3

		# now actually get the train, test splits...
		# let's do one pass with a stratified K fold... but we need to increase our data length, 
		# lest we lose records on the split, which will throw errors...
		f = F.copy()
		targ = iris.target
		targ = ['a' if x == 0 else 'b' if x == 1 else 'c' for x in targ]
		f['species'] = targ

		# Times FIVE!
		f = pd.concat([f,f,f,f,f], axis=0) # times FIVE!

		# shuffle the rows
		f = f.iloc[np.random.permutation(np.arange(f.shape[0]))]

		# try uploading again...
		try:
			frame = new_h2o_frame(f)
		except Exception as e:
			frame = None

		if frame is not None:
			for stratified in [None, 'species']:
				X_train, X_test = h2o_train_test_split(frame, test_size=0.3, train_size=0.7, stratify=stratified)
				a, b = (X_train.shape[0] + X_test.shape[0]), frame.shape[0]
				assert a==b, '%d != %d' % (a, b) 
		else:
			pass

	def act_search():
		boston = load_boston()
		X_bost = pd.DataFrame.from_records(data=boston.data, columns=boston.feature_names)
		X_bost['target'] = boston.target

		# need to make up some exposure features
		X_bost['expo'] = [1+np.random.rand() for i in range(X_bost.shape[0])]
		X_bost['loss'] = [1+np.random.rand() for i in range(X_bost.shape[0])]

		# do split
		X_train, X_test = train_test_split(X_bost, train_size=0.7)

		# now upload to cloud...
		try:
			train = from_pandas(X_train)
			test = from_pandas(X_test)
		except Exception as e:
			train = None
			test = None


		if all([x is not None for x in (train, test)]):
			# define a pipeline
			pipe = H2OPipeline([
					('nzv', H2ONearZeroVarianceFilterer(na_warn=False)),
					('mcf', H2OMulticollinearityFilterer(na_warn=False)),
					('gbm', H2OGradientBoostingEstimator())
				])

			# define our hyperparams
			hyper_params = {
				'nzv__threshold'  : uniform(1e-8, 1e-2), #[1e-8, 1e-6, 1e-4, 1e-2],
				'mcf__threshold'  : uniform(0.85, 0.15),
				'gbm__ntrees'     : randint(25, 100),
				'gbm__max_depth'  : randint(2, 8),
				'gbm__min_rows'	  : randint(8, 25)
			}

			# define our grid search
			rand_state=42
			search = H2OActuarialRandomizedSearchCV(
									estimator=pipe,
									param_grid=hyper_params,
									exposure_feature='expo',
									loss_feature='loss',
									random_state=rand_state,
									feature_names=boston.feature_names,
									target_feature='target',
									scoring='lift',
									validation_frame=test,
									cv=H2OKFold(n_folds=2, shuffle=True, random_state=rand_state),
									verbose=1, # don't want to go overkill
									n_iter=1
								)

			search.fit(train)

			# can we plot in the actual tests?
			try:
				search.plot(timestep='number_of_trees', metric='MSE')
			except Exception as e:
				pass # naive for now, but not even sure this can be done in nosetests...

			# report:
			report = search.report_scores()
			assert report.shape[0] == 1

		else:
			pass


	def sparse():
		f = F.copy()
		f['sparse'] = np.zeros(f.shape[0])

		# try uploading...
		try:
			frame = new_h2o_frame(f)
		except Exception as e:
			frame = None

		if frame is not None:
			filterer = H2OSparseFeatureDropper()
			filterer.fit(frame)
			assert 'sparse' in filterer.drop_

			# assert fails for over 1.0 or under 0.0
			assert_fails(H2OSparseFeatureDropper(threshold=0.0).fit, ValueError, frame)
			assert_fails(H2OSparseFeatureDropper(threshold=1.0).fit, ValueError, frame)

		else:
			pass


	# run them
	multicollinearity()
	nzv()
	pipeline()
	grid()
	anon_class()
	cv()
	from_pandas_h2o()
	from_array_h2o()
	split_tsts()
	act_search()
	sparse()
