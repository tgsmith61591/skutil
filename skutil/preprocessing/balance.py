from __future__ import division, print_function
import abc, warnings
import pandas as pd
import numpy as np
from numpy.random import choice
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.neighbors import NearestNeighbors
from ..base import SelectiveMixin
from ..utils import *


__all__ = [
	'OversamplingClassBalancer',
	'SMOTEClassBalancer'
]


def _validate_x_y_ratio(X, y, ratio):
	mc = BalancerMixin.__max_classes__

	# validate y
	if (not y) or (not isinstance(y, str)):
		raise ValueError('y must be a column name')

	# validate is < max classes
	cts = X[y].value_counts().sort_values()
	n_classes = cts.shape[0]
	if n_classes > mc:
		raise RuntimeError('class balancing can only handle <= %i classes, but got %i' % (mc, n_classes))

	# validate ratio, if the current ratio is >= the ratio, it's "balanced enough"
	if not isinstance(ratio, float) or ratio <= 0 or ratio > 1:
		raise ValueError('ratio should be a float between 0.0 and 1.0, but got %s' % str(ratio))

	return cts, n_classes


###############################################################################
class SamplingWarning(UserWarning):
	"""Custom warning used to notify the user that sub-optimal sampling behavior
	has occurred. For instance, performing oversampling on a minority class with only
	one instance will cause this warning to be thrown.
	"""

class BalancerMixin:
	"""A mixin class for all balancer classes.
	Balancers are not like TransformerMixins or
	BaseEstimators, and do not implement fit or predict.
	"""
	# the max classes handled by class balancers
	__max_classes__ = 20
	__metaclass__ = abc.ABCMeta

	@abc.abstractmethod
	def balance(self, X):
		return


###############################################################################
class OversamplingClassBalancer(BalancerMixin):
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
	"""

	def __init__(self, y=None, ratio=0.2):
		self.y_ = y
		self.ratio = ratio

	def balance(self, X):
		"""Apply the oversampling balance operation. Oversamples
		the minority class to the provided ratio of minority
		class : majority class
		
		Parameters
		----------
		X : pandas DF, shape [n_samples, n_features]
			The data used for estimating the lambdas
		"""
		validate_is_pd(X)
		X = X.copy()
		mc = BalancerMixin.__max_classes__

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
		return X



###############################################################################
class SMOTEClassBalancer(BalancerMixin):
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
	"""

	def __init__(self, y=None, ratio=0.2, k=3):
		self.y_ = y
		self.k = k
		self.ratio = ratio

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
		validate_is_pd(X)
		X = X.copy()

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
		return X



