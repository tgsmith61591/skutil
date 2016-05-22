import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from ..base import SelectiveMixin
from ..utils import validate_is_pd


__all__ = [
	'FeatureSelector',
	'MulticollinearityFilterer',
	'NearZeroVarianceFilterer'
]


###############################################################################
class FeatureSelector(BaseEstimator, TransformerMixin, SelectiveMixin):
	"""A very simple class to be used at the beginning of a Pipeline that will
	only propagte the given features throughout the remainder of the pipe

	Parameters
	----------
	cols : array_like (string)
		The features to select
	"""

	def __init__(self, cols=None):
		self.cols_ = cols

	def fit(self, X, y = None):
		validate_is_pd(X)

		## If cols is None, then apply to all by default
		if not self.cols_:
			self.cols_ = X.columns.tolist()

		return self

	def transform(self, X, y = None):
		validate_is_pd(X)

		if not self.cols_:
			raise ValueError('model has not been fit')

		return X[self.cols_]


###############################################################################
class MulticollinearityFilterer(BaseEstimator, TransformerMixin, SelectiveMixin):
	"""Filter out features with a correlation greater than the provided threshold.
	When a pair of correlated features is identified, the mean absolute correlation (MAC)
	of each feature is considered, and the feature with the highsest MAC is discarded.

	Parameters
	----------
	cols : array_like, string
		The columns used to generate the correlation matrix

	threshold : float, default 0.85
		The threshold above which to filter correlated features

	method : str, one of ['pearson','kendall','spearman'], default 'pearson'
		The method used to compute the correlation

	as_df : boolean, default True
		Whether to return a pandas DataFrame

	Attributes
	----------
	cols_ : the cols used to compute the correlation matrix

	drop_ : list, string
		The columns to drop

	as_df : boolean
		Whether or not to return a dataframe

	"""

	def __init__(self, cols=None, threshold=0.85, method='pearson', as_df=True):
		self.cols_ = cols
		self.threshold = threshold
		self.method = method
		self.as_df = as_df


	def fit(self, X, y = None):
		"""Fit the multicollinearity filterer.

		Parameters
		----------
		X : pandas DataFrame
			The frame to fit

		y : None, passthrough for pipeline
		"""

		self.fit_transform(X, y)
		return self


	def fit_transform(self, X, y = None):
		"""Fit the multicollinearity filterer and
		return the filtered frame.

		Parameters
		----------
		X : pandas DataFrame
			The frame to fit

		y : None, passthrough for pipeline
		"""

		validate_is_pd(X)

		## copy X -- don't need to validate method, numpy will
		X = X.copy()

		## If cols is None, then apply to all by default
		if not self.cols_:
			self.cols_ = X.columns.tolist()
		if len(self.cols_) < 2:
			raise ValueError('too few features')

		## init drops list
		drops = []

		## Generate correlation matrix
		c = X[self.cols_].corr(method=self.method).apply(lambda x: np.abs(x))

		## Iterate over each feature
		finished = False
		while not finished:

			# Whenever there's a break, this loop will start over
			for i,nm in enumerate(c.columns):
				this_col = c[nm].drop(nm).sort_values()

				# check if last value is over thresh
				if this_col[-1] < self.threshold or this_col.shape[0] == 1:
					if i == c.columns.shape[0] - 1:
						finished = True

					# control passes to next column name or end if finished
					continue

				# gets the current col, and drops the same row, sorts asc and gets other col
				other_col_nm = this_col.index[-1]
				that_col = c[other_col_nm].drop(other_col_nm)

				# get the mean absolute correlations of each
				mn_1, mn_2 = this_col.mean(), that_col.mean()
				drop_nm = nm if mn_1 > mn_2 else other_col_nm

				# drop the bad col, row
				c.drop(drop_nm, axis=1, inplace=True)
				c.drop(drop_nm, axis=0, inplace=True)

				# add the bad col to drops
				drops.append(drop_nm)

				# if we get here, we have to break so will start over
				break

			# if not finished, restarts loop, otherwise will exit loop

		# Assign attributes, return
		self.drop_ = drops
		dropped = X.drop(drops, axis=1)

		return dropped if self.as_df else dropped.as_matrix()


	def transform(self, X, y = None):
		"""Drops the highly-correlated features from the new
		input frame.

		Parameters
		----------
		X : pandas DataFrame
			The frame to transform

		y : None, passthrough for pipeline
		"""
		validate_is_pd(X)
		check_is_fitted(self, 'drop_')

		dropped = X.drop(self.drop_, axis=1)
		return dropped if self.as_df else dropped.as_matrix()


###############################################################################
class NearZeroVarianceFilterer(BaseEstimator, TransformerMixin, SelectiveMixin):
	"""Identify and remove any features that have a variance below
	a certain threshold.

	Parameters
	----------
	cols : array_like, string
		The columns to evaluate for potential drops

	threshold : float, default 1e-6
		The threshold below which to declare "zero variance"

	as_df : boolean, default True
		Whether to return a pandas DataFrame
	"""

	def __init__(self, cols=None, threshold=1e-6, as_df=True):
		self.cols_ = cols
		self.threshold = threshold
		self.as_df = as_df

	def fit(self, X, y = None):
		validate_is_pd(X)

		## If cols is None, then apply to all by default
		if not self.cols_:
			self.cols_ = X.columns.tolist()

		srs = X[self.cols_].apply(lambda x: np.var(x) < self.threshold)
		drops = X[self.cols_].columns[srs]

		if drops.shape[0] == 0:
			self.drop_ = None
		else:
			self.drop_ = drops

		return self

	def transform(self, X, y = None):
		validate_is_pd(X)
		check_is_fitted(self, 'drop_')

		x = X.copy()
		if self.drop_ is None:
			return x if self.as_df else x.as_matrix()
		else:
			x.drop(self.drop_, axis=1, inplace=True)
			return x if self.as_df else x.as_matrix()


