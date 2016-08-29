from __future__ import division, print_function, absolute_import
from abc import ABCMeta, abstractmethod
import warnings
import pandas as pd
import numpy as np
from sklearn.externals import six
from sklearn.base import BaseEstimator, TransformerMixin, is_classifier
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import Imputer
from sklearn.ensemble import BaggingRegressor, BaggingClassifier
from ..base import SelectiveMixin
from ..utils import is_entirely_numeric, get_numeric, validate_is_pd, is_numeric


__all__ = [
	'BaggedImputer',
	'BaggedCategoricalImputer',
	'SelectiveImputer'
]


def _validate_all_numeric(X):
	# need to check on numeric nature of the cols
	if not is_entirely_numeric(X):
		raise ValueError('provided columns must be of only numeric columns')


def _col_mode(col):
	vals = col.value_counts()
	return vals.index[0] if not np.isnan(vals.index[0]) else vals.index[1]


###############################################################################
class _BaseImputer(SelectiveMixin, BaseEstimator, TransformerMixin):
	"""A base class for all imputers"""
	__def_fill__ = -999999

	def __init__(self, cols=None, as_df=True, def_fill=None):
		self.cols = cols
		self.as_df = as_df
		self.fill_ = _BaseImputer.__def_fill__ if def_fill is None else def_fill



def _val_values(vals):
	if not all([
		(is_numeric(i) or \
			(isinstance(i, (str,unicode))) and \
			i in ('mode', 'mean', 'median')) \
		for i in vals
	]):
		raise TypeError('All values in self.fill must be numeric or in ("mode", "mean", "median"). '
						'Got: %s' % ', '.join(vals))


class SelectiveImputer(_BaseImputer):
	"""A more customizable form on sklearn's Imputer class. This class
	can handle more than mean, median or most common... it will also take
	numeric values. Moreover, it will take a vector of strategies or values
	with which to impute corresponding columns.
	
	Parameters
	----------
	cols : array_like, optional (default=None)
		the features to impute

	as_df : boolean , optional (default=True)
		whether to return a dataframe

	def_fill : int, optional (default=None)
		the fill to use for missing values in the training matrix 
		when fitting a SelectiveClassifier. If None, will default to 'mean'
	"""

	def __init__(self, cols=None, as_df=True, def_fill='mean'):
		super(SelectiveImputer, self).__init__(cols, as_df, def_fill)

	def fit(self, X, y=None):
		"""Fit the CategoricalImputer and return the 
		transformed matrix or frame.

		Parameters
		----------
		X : pandas DataFrame
			The frame to fit

		y : None, passthrough for pipeline
		"""

		# check on state of X and cols
		X, self.cols = validate_is_pd(X, self.cols)
		cols = self.cols if not self.cols is None else X.columns.values

		# validate the fill, do fit
		fill = self.fill_
		if isinstance(fill, (str, unicode)):
			if not str(fill) in ('mode', 'mean', 'median'):
				raise TypeError('self.fill must be either "mode", "mean", "median", None, '
								'a number, or an iterable. Got %s' % fill)

			if fill == 'mode':
				# for each column to impute, we go through and get the value counts
				# of each, sorting by the max...
				self.modes_ = dict(zip(cols, X[cols].apply(lambda x: _col_mode(x))))

			elif fill == 'median':
				self.modes_ = dict(zip(cols, X[cols].apply(lambda x: np.nanmedian(x.values))))

			else:
				self.modes_ = dict(zip(cols, X[cols].apply(lambda x: np.nanmean(x.values))))


		# if the fill is an iterable, we have to get a bit more stringent on our validation
		elif hasattr(fill, '__iter__'):

			# if fill is a dictionary
			if isinstance(fill, dict):
				# if it's a dict, we can assume that these are the cols...
				cols, fill = zip(*fill.items())
				self.cols = cols # we reset self.cols in this case!!!


			# we need to get the length of the iterable,
			# make sure it matches the len of cols
			if not len(fill) == len(cols):
				raise ValueError('len of fill does not match that of cols')

			# make sure they're all ints
			_val_values(fill)
			d = {}
			for ind, c in enumerate(cols):
				f = fill[ind]

				if is_numeric(f):
					d[c] = f
				else:
					the_col = X[c]
					if f == 'mode':
						d[c] = _col_mode(the_col)
					elif f == 'median':
						d[c] = np.nanmedian(the_col.values)
					else:
						d[c] = np.nanmean(the_col.values)


			self.modes_ = d

		else:
			if not is_numeric(fill):
				raise TypeError('self.fill must be either "mode", "mean", "median", None, '
								'a number, or an iterable. Got %s' % str(fill))

			# either the fill is an int, or it's something the user provided...
			# if it's not an int or float, we'll let it go and not catch it because
			# the it's their fault they were dumb.
			self.modes_ = fill

		return self

	def transform(self, X):
		"""Transform a dataframe given the fit imputer.

		Parameters
		----------
		X : pandas DataFrame
			The frame to fit

		y : None, passthrough for pipeline
		"""

		check_is_fitted(self, 'modes_')
		# check on state of X and cols
		X, _ = validate_is_pd(X, self.cols)
		cols = self.cols if not self.cols is None else X.columns.values

		# get the fills
		modes = self.modes_

		# if it's a single int, easy:
		if isinstance(modes, int):
			X[cols] = X[cols].fillna(modes)
		else:
			# it's a dict
			for nm in cols:
				X[nm] = X[nm].fillna(modes[nm])

		return X if self.as_df else X.as_matrix()


class _BaseBaggedImputer(_BaseImputer):
	def __init__(self, cols=None, base_estimator=None, n_estimators=10, 
		max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=True,
		oob_score=False, n_jobs=1, random_state=None, verbose=0, as_df=True, 
		def_fill=None, is_classification=False):

		super(_BaseBaggedImputer, self).__init__(cols=cols, as_df=as_df, def_fill=def_fill)

		# set self attributes
		self.base_estimator = base_estimator
		self.n_estimators = n_estimators
		self.max_samples = max_samples
		self.max_features = max_features
		self.bootstrap = bootstrap
		self.bootstrap_features = bootstrap_features
		self.oob_score = oob_score
		self.n_jobs = n_jobs
		self.random_state = random_state
		self.verbose = verbose
		self.is_classification = is_classification

	def fit(self, X, y=None):
		"""Fit the BaggedImputer.

		Parameters
		----------
		X : pandas DataFrame
			The frame to fit

		y : None, passthrough for pipeline
		"""

		self.fit_transform(X, y)
		return self

	def fit_transform(self, X, y=None):
		"""Fit the BaggedImputer and return the 
		transformed matrix or frame.

		Parameters
		----------
		X : pandas DataFrame
			The frame to fit

		y : None, passthrough for pipeline
		"""

		# check on state of X and cols
		X, self.cols = validate_is_pd(X, self.cols)
		cols = self.cols if not self.cols is None else X.columns.values

		# subset, validate
		# we have to validate that all of the columns we're going to impute
		# are numeric (this could be float, or int...).
		_validate_all_numeric(X[cols])

		# we need to get all of the numerics out of X, because these are
		# the features we'll be modeling on.
		numeric_cols = get_numeric(X)
		numerics = X[numeric_cols]

		# if is_classification and our estimator is NOT, then we need to raise
		if self.base_estimator is not None:
			if self.is_classification and not is_classifier(self.base_estimator):
				raise TypeError('self.is_classification=True, '
								'but base_estimator is not a classifier')

		# set which estimator type to fit:
		_model = BaggingRegressor if not self.is_classification else BaggingClassifier

		# if there's only one numeric, we know at this point it's the one
		# we're imputing. In that case, there's too few cols on which to model
		if numerics.shape[1] == 1:
			raise ValueError('too few numeric columns on which to model')


		# the core algorithm:
		# - for each col to impute
		#   - subset to all numeric columns except the col to impute
		#   - retain only the complete observations, separate the missing observations
		#   - build a bagging regressor model to predict for observations with missing values
		#   - fill in missing values in a copy of the dataframe

		models = {}
		for col in cols:
			x = numerics.copy()              # get copy of numerics for this model iteration
			y_missing = pd.isnull(x[col])    # boolean vector of which are missing in the current y
			y = x.pop(col)                   # pop off the y vector from the matrix

			# if y_missing is all of the rows, we need to bail
			if y_missing.sum() == x.shape[0]:
				raise ValueError('%s has all missing values, cannot train model' % col)

			# at this point we've identified which y values we need to predict, however, we still
			# need to prep our x matrix... There are a few corner cases we need to account for:
			# 
			# 1. there are no complete rows in the X matrix
			#   - we can eliminate some columns to model on in this case, but there's no silver bullet
			# 2. the cols selected for model building are missing in the rows needed to impute.
			#   - this is a hard solution that requires even more NA imputation...
			# 
			# the most "catch-all" solution is going to be to fill all missing values with some val, say -999999

			x = x.fillna(self.fill_)
			X_train = x[~y_missing]          # the rows that don't correspond to missing y values
			X_test  = x[y_missing]           # the rows to "predict" on
			y_train = y[~y_missing]          # the training y vector

			# define the model
			model = _model(
				base_estimator=self.base_estimator,
				n_estimators=self.n_estimators,
				max_samples=self.max_samples,
				max_features=self.max_features,
				bootstrap=self.bootstrap,
				bootstrap_features=self.bootstrap_features,
				oob_score=self.oob_score,
				n_jobs=self.n_jobs,
				random_state=self.random_state,
				verbose=self.verbose)

			# fit the model
			model.fit(X_train, y_train)

			# predict on the missing values, stash the model and the features used to train it
			if X_test.shape[0] != 0: # only do this step if there are actually any missing
				y_pred = model.predict(X_test)
				X.loc[y_missing, col] = y_pred # fill the y vector missing slots and reassign back to X

			models[col] = {
				'model'         : model,
				'feature_names' : X_train.columns.values
			}


		# assign the model dict to self -- this is the "fit" portion
		self.models_ = models
		return X if self.as_df else X.as_matrix()

	def transform(self, X, y=None):
		"""Transform a dataframe given the fit imputer.

		Parameters
		----------
		X : pandas DataFrame
			The frame to fit

		y : None, passthrough for pipeline
		"""

		check_is_fitted(self, 'models_')
		# check on state of X and cols
		X, _ = validate_is_pd(X, self.cols)

		# perform the transformations for missing vals
		models = self.models_
		for col, kv in six.iteritems(models):
			features, model = kv['feature_names'], kv['model']
			y = X[col] # the y we're predicting

			# this will throw a key error if one of the features isn't there
			X_test = X[features] # we need another copy

			# if col is in the features, there's something wrong internally
			assert not col in features, 'predictive column should not be in fit features (%s)' % col

			# since this is a copy, we can add the missing vals where needed
			X_test = X_test.fillna(self.fill_)

			# generate predictions, subset where y was null
			y_null = pd.isnull(y)
			pred_y = model.predict(X_test.loc[y_null])

			# fill where necessary:
			if y_null.sum() > 0:
				y[y_null] = pred_y # fill where null
				X[col] = y # set back to X

		return X if self.as_df else X.as_matrix()



class BaggedCategoricalImputer(_BaseBaggedImputer):
	"""Performs imputation on select columns by using BaggingRegressors
	on the provided columns.

	cols : array_like, optional (default=None)
		the features to impute

	base_estimator : object or None, optional (default=None)
		The base estimator to fit on random subsets of the dataset. 
		If None, then the base estimator is a decision tree.

	n_estimators : int, optional (default=10)
		The number of base estimators in the ensemble.

	max_samples : int or float, optional (default=1.0)
		The number of samples to draw from X to train each base estimator.
		If int, then draw max_samples samples.
		If float, then draw max_samples * X.shape[0] samples.

	max_features : int or float, optional (default=1.0)
		The number of features to draw from X to train each base estimator.
		If int, then draw max_features features.
		If float, then draw max_features * X.shape[1] features.

	bootstrap : boolean, optional (default=True)
		Whether samples are drawn with replacement.

	bootstrap_features : boolean, optional (default=False)
		Whether features are drawn with replacement.

	oob_score : bool, optional (default=False)
		Whether to use out-of-bag samples to estimate the generalization error.

	n_jobs : int, optional (default=1)
		The number of jobs to run in parallel for both fit and predict. If -1, 
		then the number of jobs is set to the number of cores.
	
	random_state : int, RandomState instance or None, optional (default=None)
		If int, random_state is the seed used by the random number generator; If 
		RandomState instance, random_state is the random number generator; If None, 
		the random number generator is the RandomState instance used by np.random.
	
	verbose : int, optional (default=0)
		Controls the verbosity of the building process.

	as_df : boolean , optional (default=True)
		whether to return a dataframe

	def_fill : int, optional (default=None)
		the fill to use for missing values in the training matrix 
		when fitting a BaggingClassifier. If None, will default to -999999
	"""
	def __init__(self, cols=None, base_estimator=None, n_estimators=10, 
		max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=True,
		oob_score=False, n_jobs=1, random_state=None, verbose=0, as_df=True, def_fill=None):

		# categorical imputer needs to be classification
		super(BaggedCategoricalImputer, self).__init__(
			cols=cols, as_df=as_df, def_fill=def_fill, 
			base_estimator=base_estimator, n_estimators=n_estimators, 
			max_samples=max_samples, max_features=max_features, bootstrap=bootstrap, 
			bootstrap_features=bootstrap_features, oob_score=oob_score, 
			n_jobs=n_jobs, random_state=random_state, verbose=verbose, 
			is_classification=True)




class BaggedImputer(_BaseBaggedImputer):
	"""Performs imputation on select columns by using BaggingRegressors
	on the provided columns.

	cols : array_like, optional (default=None)
		the features to impute

	base_estimator : object or None, optional (default=None)
		The base estimator to fit on random subsets of the dataset. 
		If None, then the base estimator is a decision tree.

	n_estimators : int, optional (default=10)
		The number of base estimators in the ensemble.

	max_samples : int or float, optional (default=1.0)
		The number of samples to draw from X to train each base estimator.
		If int, then draw max_samples samples.
		If float, then draw max_samples * X.shape[0] samples.

	max_features : int or float, optional (default=1.0)
		The number of features to draw from X to train each base estimator.
		If int, then draw max_features features.
		If float, then draw max_features * X.shape[1] features.

	bootstrap : boolean, optional (default=True)
		Whether samples are drawn with replacement.

	bootstrap_features : boolean, optional (default=False)
		Whether features are drawn with replacement.

	oob_score : bool, optional (default=False)
		Whether to use out-of-bag samples to estimate the generalization error.

	n_jobs : int, optional (default=1)
		The number of jobs to run in parallel for both fit and predict. If -1, 
		then the number of jobs is set to the number of cores.
	
	random_state : int, RandomState instance or None, optional (default=None)
		If int, random_state is the seed used by the random number generator; If 
		RandomState instance, random_state is the random number generator; If None, 
		the random number generator is the RandomState instance used by np.random.
	
	verbose : int, optional (default=0)
		Controls the verbosity of the building process.

	as_df : boolean , optional (default=True)
		whether to return a dataframe

	def_fill : int, optional (default=None)
		the fill to use for missing values in the training matrix 
		when fitting a BaggingRegressor. If None, will default to -999999
	"""

	def __init__(self, cols=None, base_estimator=None, n_estimators=10, 
		max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=True,
		oob_score=False, n_jobs=1, random_state=None, verbose=0, as_df=True, def_fill=None):

		# invoke super constructor
		super(BaggedImputer, self).__init__(
			cols=cols, as_df=as_df, def_fill=def_fill, 
			base_estimator=base_estimator, n_estimators=n_estimators, 
			max_samples=max_samples, max_features=max_features, bootstrap=bootstrap, 
			bootstrap_features=bootstrap_features, oob_score=oob_score, 
			n_jobs=n_jobs, random_state=random_state, verbose=verbose, 
			is_classification=False)

		