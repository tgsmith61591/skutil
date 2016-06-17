from __future__ import division, print_function
from .base import _BaseFeatureSelector
from scipy.linalg import qr
from numpy.linalg import matrix_rank as mr
from numpy.linalg.linalg import LinAlgError
from ..odr import dqrcf


__all__ = [
	#'LinearCombinationFilterer'
]


###############################################################################
class _LinearCombinationFilterer(_BaseFeatureSelector):
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
		X, self.cols = validate_is_pd(X, self.cols)
		_validate_cols(self.cols)

		# init drops list
		drops = []

		# Generate sub matrix for qr decomposition
		cols = self.cols if not self.cols is None else X.columns
		x = X[cols].as_matrix()

		# do subroutines
		lc_list = _enumLC(qr(x, pivoting=True))

		if not lc_list is None:
			while True:
				# keep removing linear dependencies until it resolves
				drops.extend(lc_list)
				#lc_list = _enumLC() ## FIXME

				if not lc_list:
					break

		# Assign attributes, return
		self.drop = drops
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
	decomp : a QR decomposition object
		The QR decomposition of the matrix
	"""
	QR = decomp[1] # position 1 in the tuple

	# extract the R matrix
	R = _qr_R(QR)              # the R matrix
	n_features = R.shape[1]    # number of columns in R
	is_zero = n_features == 0  # whether there are no features

	try:
		rank = mr(R)           # the rank of the matrix, or num of independent cols
	except LinAlgError as lae: # if is empty, will get this error
		rank = 0

	if not (is_zero or rank == n_features):
		pivot = decomp[2]           # the pivot vector
		xprime = R[:rank, :rank]    # extract the independent cols
		yprime = R[:rank, rank+1:]  # extract the dependent columns

		b = qr(xprime)              # factor the independent columns
		b = _qr_coef(b, yprime)     # get regression coefficients of dependent cols

		# if b is None, then there were no dependent columns
		if b is not None:
			b[np.abs(b) < 1e-6] = 0 # zap small values

			## FIXME

	# if we get here, there are no linear combos to discover
	return None

def _qr_R(qr_dec):
	"""Extract the R matrix from a QR decomposition"""
	min_dim = min(qr_dec.shape)
	return qr_dec[:min_dim+1,:]

def _qr_coef(qr_dec, y):
	"""Extract coefficients from a QR decomposition"""
	QR = qr_dec[1]
	n, p = QR.shape
	nx, ny = y.shape

	if n * p > 2147483647:
		raise ValueError('too many elements for LINPACK')

	# ensure dims -- shouldn't be necessary, as this is an internal method
	#if not nx == n:
	#	raise ValueError('qr and y must have same number of rows')

	try:
		k = mr(QR)             # the rank of the matrix, or num of independent cols
	except LinAlgError as lae: # if is empty, will get this error
		k = 0

	# can't go any further
	if 0 == p or 0 == ny or 0 == k:
		return None

	# get ix vector
	if p > n:
		ix = np.ones(n + (p - n)) * np.nan
		ix[:n] = np.arange(n) # i.e., array([0,1,2,nan,nan,nan])
	else:
		ix = np.arange(n)

	# get z
	#z = dqrcf(QR, n, k, )

