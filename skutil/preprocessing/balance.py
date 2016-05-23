from __future__ import division, print_function
import pandas as pd
import numpy as np
from numpy.random import choice
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.neighbors import NearestNeighbors
from ..base import SelectiveMixin
from ..utils import *


__all__ = [
	'SMOTEStratifiedBalancer'
]


###############################################################################
class SMOTEStratifiedBalancer:
	"""Transform a matrix with the SMOTE (Synthetic Minority Oversampling TEchnique)
	method.

	Parameters
	----------
	y : str, def None
		The name of the response column. The response column must be
		biclass, no more or less.

	k : int, def 3
		The number of neighbors to use in the nearest neighbors model

	ratio : float, def 0.5
		The target ratio of the minority records to the majority records. If the
		existing ratio is >= the provided ratio, the return value will merely be
		a copy of the input matrix, otherwise SMOTE will impute records until the
		target ratio is reached.

	"""

	def __init__(self, y=None, k=3, ratio=0.5):
		self.y_ = y
		self.k = k
		self.ratio = ratio

	def transform(self, X):
		"""Apply the SMOTE balancing operation. Oversamples
		the minority class to the provided ratio of minority
		class : majority class
        
        Parameters
        ----------
        X : pandas DF, shape [n_samples, n_features]
            The data used for estimating the lambdas
        """
		validate_is_pd(X)

		# validate y
		if (not self.y_) or (not isinstance(self.y_, str)):
			raise ValueError('y must be a column name')

		# validate is two class
		cts = X[self.y_].value_counts().sort_values()
		n_classes = cts.shape[0]

		# ensure only two class
		if not n_classes == 2:
			raise RuntimeError('n_classes should equal 2, but got %i' % n_classes)

		# get the two classes, check on the current ratio
		minority, majority = cts.index[0], cts.index[1]
		current_ratio = cts[minority] / cts[majority]
		

		# validate ratio, if the current ratio is >= the ratio, it's "balanced enough"
		ratio = self.ratio
		if not isinstance(ratio, float) or ratio <= 0 or ratio > 1:
			raise ValueError('ratio should be a float between 0.0 and 1.0, but got %s' % str(ratio))
		if current_ratio >= ratio:
			return X.copy() # return a copy


		n_required = np.maximum(1, int(ratio * cts[majority]))
		n_samples = n_required - cts[minority] # the difference in the current present and the number we need
		
		# the np maximum can cause weirdness
		if n_samples <= 0:
			return X.copy() 


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
		combined = pd.concat([X, syn_frame])

		# return the combined frame
		return combined



