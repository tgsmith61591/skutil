from __future__ import division, print_function
import warnings
import pandas as pd
import numpy as np
import abc
from numpy.random import choice
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.neighbors import NearestNeighbors
from ..base import *
from ..utils import *


__all__ = [
	'OversamplingClassBalancer',
	'SMOTEClassBalancer',
	'UndersamplingClassBalancer'
]


def _validate_x_y_ratio(X, y, ratio):
	"""Validates the following, given that X is
	already a validated pandas DataFrame:

	1. That y is a string
	2. That the number of classes does not exceed __max_classes__
	   as defined by the _BaseBalancer class
	3. That the number of classes is at least 2
	4. That ratio is a float that falls between 0.0 (exclusive) and
	   1.0 (inclusive)

	Return
	------
	(cts, n_classes), a tuple with the sorted class value_counts and the number of classes
	"""
	mc = _BaseBalancer.__max_classes__

	# validate y
	if (not y) or (not isinstance(y, str)):
		raise ValueError('y must be a column name')

	# validate is < max classes
	cts = X[y].value_counts().sort_values()
	n_classes = cts.shape[0]
	if n_classes > mc:
		raise ValueError('class balancing can only handle <= %i classes, but got %i' % (mc, n_classes))
	elif n_classes < 2:
		raise ValueError('class balancing requires at least 2 classes')

	# validate ratio, if the current ratio is >= the ratio, it's "balanced enough"
	if not isinstance(ratio, float) or ratio <= 0 or ratio > 1:
		raise ValueError('ratio should be a float between 0.0 and 1.0, but got %s' % str(ratio))

	return cts, n_classes



###############################################################################
class _BaseBalancer:
	"""A super class for all balancer classes. Balancers are not like TransformerMixins 
	or BaseEstimators, and do not implement fit or predict. This is because Balancers
	are ONLY applied to training data.
	"""
	# the max classes handled by class balancers
	__max_classes__ = 20
	__metaclass__ = abc.ABCMeta

	def __init__(self, ratio=0.2, y=None, as_df=True):
		self.ratio=ratio
		self.y_ = y
		self.as_df = as_df

	@abc.abstractmethod
	def balance(self, X):
		return NotImplemented


class OversamplingClassBalancer(_BaseBalancer):
	"""Oversample the minority classes until they are represented
	at the target proportion to the majority class.

	Parameters
	----------
	y : str, def None
		The name of the response column. The response column must be
		biclass, no more or less.

	ratio : float, def 0.2
		The target ratio of the minority records to the majority records. If the
		existing ratio is >= the provided ratio, the return value will merely be
		a copy of the input matrix, otherwise SMOTE will impute records until the
		target ratio is reached.

	as_df : bool, optional (default=True)
		Whether to return a dataframe
	"""

	def __init__(self, y=None, ratio=0.2, as_df=True):
		super(OversamplingClassBalancer, self).__init__(ratio=ratio, y=y, as_df=as_df)

	def balance(self, X):
		"""Apply the oversampling balance operation. Oversamples
		the minority class to the provided ratio of minority
		class : majority class
		
		Parameters
		----------
		X : pandas DF, shape [n_samples, n_features]
			The data used for estimating the lambdas
		"""
		# check on state of X
		X, _ = validate_is_pd(X, None) # there are no cols, and we don't want warnings
		mc = _BaseBalancer.__max_classes__

		# since we rely on indexing X, we need to reset indices
		# in case X is the result of a slice and they're out of order.
		X.index = np.arange(0,X.shape[0])
		ratio = self.ratio
		cts, n_classes = _validate_x_y_ratio(X, self.y_, ratio)


		# get the maj class
		majority = cts.index[-1]
		n_required = np.maximum(1, int(ratio * cts[majority]))
		for minority in cts.index:
			if minority == majority:
				break

			min_ct = cts[minority]
			if min_ct == 1:
				warnings.warn('class %s only has one observation' % str(minority), SamplingWarning)

			current_ratio = min_ct / cts[majority]	
			if current_ratio >= ratio:
				continue # if ratio is already met, continue

			n_samples = n_required - min_ct # the difference in the current present and the number we need
			# the np maximum can cause weirdness
			if n_samples <= 0:
				continue # move onto next class

			minority_recs = X[X[self.y_] == minority]
			idcs = choice(minority_recs.index, n_samples, replace=True)
			pts = X.iloc[idcs]

			# append to X
			X = pd.concat([X, pts])

		# return the combined frame
		return X if self.as_df else X.as_matrix()



###############################################################################
class SMOTEClassBalancer(_BaseBalancer):
	"""Transform a matrix with the SMOTE (Synthetic Minority Oversampling TEchnique)
	method.

	Parameters
	----------
	y : str, def None
		The name of the response column. The response column must be
		biclass, no more or less.

	k : int, def 3
		The number of neighbors to use in the nearest neighbors model

	ratio : float, def 0.2
		The target ratio of the minority records to the majority records. If the
		existing ratio is >= the provided ratio, the return value will merely be
		a copy of the input matrix, otherwise SMOTE will impute records until the
		target ratio is reached.

	as_df : bool, optional (default=True)
		Whether to return a dataframe
	"""

	def __init__(self, y=None, ratio=0.2, k=3, as_df=True):
		super(SMOTEClassBalancer, self).__init__(ratio=ratio, y=y, as_df=as_df)
		self.k = k

	def balance(self, X):
		"""Apply the SMOTE balancing operation. Oversamples
		the minority class to the provided ratio of minority
		class : majority class by interpolating points between
		each sampled point's k-nearest neighbors.
		
		Parameters
		----------
		X : pandas DF, shape [n_samples, n_features]
			The data used for estimating the lambdas
		"""
		# check on state of X
		X, _ = validate_is_pd(X, None, assert_all_finite=True) # there are no cols, and we don't want warnings

		# since we rely on indexing X, we need to reset indices
		# in case X is the result of a slice and they're out of order.
		X.index = np.arange(0,X.shape[0])
		ratio = self.ratio
		cts, n_classes = _validate_x_y_ratio(X, self.y_, ratio)
		

		# get the maj class
		majority = cts.index[-1]
		n_required = np.maximum(1, int(ratio * cts[majority]))
		for minority in cts.index:
			if minority == majority:
				break

			min_ct = cts[minority]
			if min_ct == 1:
				raise ValueError('cannot perform SMOTE on only one observation (class=%s)' % str(minority))

			current_ratio = min_ct / cts[majority]	
			if current_ratio >= ratio:
				continue # if ratio is already met, continue

			n_samples = n_required - min_ct # the difference in the current present and the number we need
			# the np maximum can cause weirdness
			if n_samples <= 0:
				continue # move onto next class


			# don't need to validate K, neighbors will
			# randomly select n_samples points from the minority records
			minority_recs = X[X[self.y_] == minority]
			replace = n_samples > minority_recs.shape[0] # may have to replace if required num > num available
			idcs = choice(minority_recs.index, n_samples, replace=replace)
			pts = X.iloc[idcs].drop([self.y_], axis=1)

			# Fit the neighbors model on the random points
			nn = NearestNeighbors(n_neighbors=self.k).fit(pts)

			# do imputation
			synthetics_pts = []
			for neighbors in nn.kneighbors()[1]: # go over indices
				mn = pts.iloc[neighbors].mean()

				# add the minority target, and the mean record
				synthetics_pts.append(mn.tolist())

			# append the minority target to the frame
			syn_frame = pd.DataFrame.from_records(data=synthetics_pts, columns=pts.columns)
			syn_frame[self.y_] = np.array([minority] * syn_frame.shape[0])

			# reorder the columns
			syn_frame = syn_frame[X.columns]

			# append to X
			X = pd.concat([X, syn_frame])

		# return the combined frame
		return X if self.as_df else X.as_matrix()


###############################################################################
class UndersamplingClassBalancer(_BaseBalancer):
	"""Undersample the majority class until it is represented
	at the target proportion to the most-represented minority class.
	For example, give the follow pd.Series:

	0  150
	1  30
	2  10

	and the ratio 0.5, the majority class (0) will be undersampled until
	the second most-populous class (1) is represented at a ratio of 0.5:

	0  60
	1  30
	2  10

	Parameters
	----------
	y : str, def None
		The name of the response column. The response column must be
		biclass, no more or less.

	ratio : float, def 0.2
		The target ratio of the minority records to the majority records. If the
		existing ratio is >= the provided ratio, the return value will merely be
		a copy of the input matrix, otherwise SMOTE will impute records until the
		target ratio is reached.

	as_df : bool, optional (default=True)
		Whether to return a dataframe
	"""

	def __init__(self, y=None, ratio=0.2, as_df=True):
		super(UndersamplingClassBalancer, self).__init__(ratio=ratio, y=y, as_df=as_df)

	def balance(self, X):
		"""Apply the undersampling balance operation. Undersamples
		the majority class to the provided ratio over the second-most-
		populous class label.
		
		Parameters
		----------
		X : pandas DF, shape [n_samples, n_features]
			The data used for estimating the lambdas
		"""
		# check on state of X
		X, _ = validate_is_pd(X, None) # there are no cols, and we don't want warnings
		mc = _BaseBalancer.__max_classes__

		# since we rely on indexing X, we need to reset indices
		# in case X is the result of a slice and they're out of order.
		X.index = np.arange(0,X.shape[0])
		ratio = self.ratio
		cts, n_classes = _validate_x_y_ratio(X, self.y_, ratio)


		# get the maj class
		majority = cts.index[-1]
		next_most= cts.index[-2] # the next-most-populous class label
		n_required = int((1/ratio) * cts[next_most]) # i.e., if ratio == 0.5 and next_most == 30, n_required = 60

		# check the exit condition (that majority class <= n_required)
		if cts[majority] <= n_required:
			return X

		# if not returned early, drop some indices
		majority_recs = X[X[self.y_] == majority]
		idcs = choice(majority_recs.index, n_required, replace=False)

		# get the rows that were not included in the keep sample
		x_drop_rows = majority_recs.drop(idcs, axis=0).index

		# now the only rows remaining in x_drop_rows are the ones
		# that were not selected in the random choice.
		# drop all those rows (from the copy)
		dropped = X.drop(x_drop_rows, axis=0)

		return dropped if self.as_df else dropped.as_matrix()
