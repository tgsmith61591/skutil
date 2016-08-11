from __future__ import division, print_function
import numpy as np
from sklearn.externals import six
from sklearn.utils.validation import check_is_fitted
from numpy.linalg.linalg import LinAlgError
from skutil.odr import QRDecomposition
from .base import _BaseFeatureSelector
from .select import _validate_cols
from ..utils import flatten_all, validate_is_pd


__all__ = [
	'LinearCombinationFilterer'
]


###############################################################################
class LinearCombinationFilterer(_BaseFeatureSelector):
	"""Resolve linear combinations in a numeric matrix. The QR decomposition is 
	used to determine whether the matrix is full rank, and then identify the sets
	of columns that are involved in the dependencies. This class is adapted from
	the implementation in the R package, caret.

	Parameters
	----------
	cols : array_like (string)
		The features to select

	as_df : boolean, optional (True default)
		Whether to return a dataframe
	"""

	def __init__(self, cols=None, as_df=True):
		super(LinearCombinationFilterer, self).__init__(cols=cols, as_df=as_df)

	def fit(self, X, y=None):
		"""Fit the linear combination filterer.

		Parameters
		----------
		X : pandas DataFrame
			The frame to fit

		y : None, passthrough for pipeline
		"""
		self.fit_transform(X, y)
		return self

	def fit_transform(self, X, y=None):
		"""Fit the multicollinearity filterer and
		return the filtered frame.

		Parameters
		----------
		X : pandas DataFrame
			The frame to fit

		y : None, passthrough for pipeline
		"""

		# check on state of X and cols
		X, self.cols = validate_is_pd(X, self.cols, assert_all_finite=True)
		_validate_cols(self.cols)

		# init drops list
		drops = []

		# Generate sub matrix for qr decomposition
		cols = [n for n in (self.cols if not self.cols is None else X.columns)] # get a copy of the cols
		x = X[cols].as_matrix()
		cols = np.array(cols) # so we can do boolean indexing

		# do subroutines
		lc_list = _enumLC(QRDecomposition(x))

		if not lc_list is None:
			while lc_list is not None:
				# we want the first index in each of the keys in the dict
				bad = np.array([p for p in set([v[0] for _, v in six.iteritems(lc_list)])])

				# get the corresponding bad names
				bad_nms = cols[bad]
				drops.extend(bad_nms)

				# update our X, and then our cols
				x = np.delete(x, bad, axis=1)
				cols = np.delete(cols, bad)

				# keep removing linear dependencies until it resolves
				lc_list = _enumLC(QRDecomposition(x))

				# will break when lc_list returns None

		# Assign attributes, return
		self.drop = [p for p in set(drops)] # a list from the a set of the drops
		dropped = X.drop(self.drop, axis=1)

		return dropped if self.as_df else dropped.as_matrix()

	def transform(self, X, y = None):
		"""Drops the linear combination features from the new
		input frame.

		Parameters
		----------
		X : pandas DataFrame
			The frame to transform

		y : None, passthrough for pipeline
		"""
		check_is_fitted(self, 'drop')
		# check on state of X and cols
		X, _ = validate_is_pd(X, self.cols)

		dropped = X.drop(self.drop, axis=1)
		return dropped if self.as_df else dropped.as_matrix()
		

def _enumLC(decomp):
	"""Perform a single iteration of linear combo scoping.

	Parameters
	----------
	qr_decomp : a QRDecomposition object
		The QR decomposition of the matrix
	"""
	qr = decomp.qr # the decomposition matrix

	# extract the R matrix
	R = decomp.get_R()         # the R matrix
	n_features = R.shape[1]    # number of columns in R
	is_zero = n_features == 0  # whether there are no features
	rank = decomp.get_rank()   # the rank of the original matrix, or num of independent cols

	if not (rank == n_features):
		pivot = decomp.pivot        # the pivot vector
		X = R[:rank, :rank]         # extract the independent cols
		Y = R[:rank, rank:]#+1?     # extract the dependent columns

		new_qr = QRDecomposition(X) # factor the independent columns
		b = new_qr.get_coef(Y)      # get regression coefficients of dependent cols

		# if b is None, then there were no dependent columns
		if b is not None:
			b[np.abs(b) < 1e-6] = 0 # zap small values
			
			# will return a dict of {dim : list of bad idcs}
			d = {}
			row_idcs = np.arange(b.shape[0])
			for i in range(Y.shape[1]): # should only ever be 1, right?
				nested = [ 
							pivot[rank+i],
							pivot[row_idcs[b[:,i] != 0]]
						 ]
				d[i] = flatten_all(nested)

			return d

	# if we get here, there are no linear combos to discover
	return None
