from __future__ import print_function, division, absolute_import
import warnings
import numpy as np
import pandas as pd
import abc

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.externals import six

import h2o
from h2o.frame import H2OFrame

from ..feature_selection import filter_collinearity
from ..utils import is_numeric
from .base import (NAWarning, 
				   BaseH2OTransformer, 
				   _check_is_frame, 
				   _retain_features,
				   _frame_from_x_y)


__all__ = [
	'BaseH2OFeatureSelector',
	'H2OMulticollinearityFilterer',
	'H2ONearZeroVarianceFilterer',
	'H2OSparseFeatureDropper'
]


def _validate_use(X, use, na_warn):
	"""For H2OMulticollinearityFilterer and H2ONearZeroVarianceFilterer,
	validate that our 'use' arg is appropriate given the presence of NA
	values in the H2OFrame.

	Parameters
	----------
	X : H2OFrame
		The frame to evaluate. Since this is an internal method,
		no validation is done to ensure it is, in fact, an H2OFrame

	use : str, one of ('complete.obs', 'all.obs', 'everything')
		The 'use' argument passed to the transformer

	na_warn : bool
		Whether to warn if there are NAs present in the frame. If there are,
		and na_warn is set to False, the function will use the provided use,
		however, if na_warn is True and there are NA values present it will
		raise a warning and use 'complete.obs'

	Returns
	-------
	use
	"""

	# validate use
	_valid_use = ['complete.obs','all.obs','everything']
	if not use in _valid_use:
		raise ValueError('expected one of (%s) but got %s' % (', '.join(_valid_use), use))

	# check on NAs
	if na_warn:
		nasum = X.isna().sum()
		if nasum > 0:
			warnings.warn('%i NA value(s) in frame; using "complete.obs"' % nasum)
		 	use = 'complete.obs'

	return use




class BaseH2OFeatureSelector(BaseH2OTransformer):
	"""Base class for all H2O selectors.

	Parameters
	----------
	target_feature : str (default None)
		The name of the target feature (is excluded from the fit)

	min_version : str, float (default 'any')
		The minimum version of h2o that is compatible with the transformer

	max_version : str, float (default None)
		The maximum version of h2o that is compatible with the transformer
	"""
	def __init__(self, feature_names=None, target_feature=None, min_version='any', max_version=None):
		super(BaseH2OFeatureSelector, self).__init__(feature_names=feature_names,
												 target_feature=target_feature,
												 min_version=min_version,
												 max_version=max_version)
			
	def transform(self, X):
		# validate state, frame
		check_is_fitted(self, 'drop_')
		X = _check_is_frame(X)
		return X[_retain_features(X, self.drop_)]



class H2OSparseFeatureDropper(BaseH2OFeatureSelector):
	"""Retains features that are less sparse (NA) than
	the provided threshold.

	Parameters
	----------
	feature_names : array_like (string)
		The features from which to drop

	target_feature : str (default None)
		The name of the target feature (is excluded from the fit)

	threshold : float (default=0.5)
		The threshold of sparsity above which to drop

	as_df : boolean, optional (True default)
		Whether to return a dataframe
	"""

	__min_version__ = '3.8.2.9'
	__max_version__ = None

	def __init__(self, feature_names=None, target_feature=None, threshold=0.5):
		super(H2OSparseFeatureDropper, self).__init__(feature_names=feature_names,
													  target_feature=target_feature,
													  min_version=self.__min_version__,
													  max_version=self.__max_version__)

		self.threshold = threshold

	def fit(self, X):
		"""Fit the sparsity filterer.

		Parameters
		----------
		X : H2OFrame
			The frame to fit
		"""
		frame, thresh = _check_is_frame(X), self.threshold
		frame = _frame_from_x_y(frame, self.feature_names, self.target_feature)

		# validate the threshold
		if not (is_numeric(thresh) and (0.0 <= thresh < 1.0)):
			raise ValueError('thresh must be a float between '
							 '0 (inclusive) and 1. Got %s' % str(thresh))

		df = (frame.isna().apply(lambda x: x.sum()) / frame.shape[0]).as_data_frame(use_pandas=True)
		df.columns = frame.columns
		ser = df.T[0]

		self.drop_ = [str(x) for x in ser.index[ser > thresh]]
		return self



class H2OMulticollinearityFilterer(BaseH2OFeatureSelector):
	"""Filter out features with a correlation greater than the provided threshold.
	When a pair of correlated features is identified, the mean absolute correlation (MAC)
	of each feature is considered, and the feature with the highsest MAC is discarded.

	Parameters
	----------
	target_feature : str (default None)
		The name of the target feature (is excluded from the fit)
	
	threshold : float, default 0.85
		The threshold above which to filter correlated features
		
	na_warn : bool (default True)
		Whether to warn if any NAs are present

	na_rm : bool (default False)
		Whether to remove NA values

	use : str (default "complete.obs"), one of {'complete.obs','all.obs','everything'}
		A string indicating how to handle missing values.

	Attributes
	----------
	drop_ : list, string
		The columns to drop

	mean_abs_correlations_ : list, float
		The corresponding mean absolute correlations of each drop_ name
	"""
	
	__min_version__ = '3.8.2.9'
	__max_version__ = None
	
	def __init__(self, feature_names=None, target_feature=None, threshold=0.85, 
				 na_warn=True, na_rm=False, use='complete.obs'):

		super(H2OMulticollinearityFilterer, self).__init__(feature_names=feature_names,
														   target_feature=target_feature, 
														   min_version=self.__min_version__,
														   max_version=self.__max_version__)
		self.threshold = threshold
		self.na_warn = na_warn
		self.na_rm = na_rm
		self.use = use
		
		
	def fit(self, X):
		"""Fit the multicollinearity filterer.

		Parameters
		----------
		X : H2OFrame
			The frame to fit
		"""

		self.fit_transform(X)
		return self
	
	
	def fit_transform(self, X):
		"""Fit the multicollinearity filterer and
		return the transformed H2OFrame, X.

		Parameters
		----------
		X : H2OFrame
			The frame to fit
		"""
		frame, thresh = _check_is_frame(X), self.threshold
		frame = _frame_from_x_y(frame, self.feature_names, self.target_feature)

		# validate use, check NAs
		use = _validate_use(frame, self.use, self.na_warn)
		
		## Generate absolute correlation matrix
		c = frame.cor(use=use, na_rm=self.na_rm).abs().as_data_frame(use_pandas=True)
		c.columns = frame.columns # set the cols to the same names
		c.index = frame.columns
		
		## get drops list
		self.drop_, self.mean_abs_correlations_, self.correlations_ = filter_collinearity(c, self.threshold)
		return self.transform(X)
		


class H2ONearZeroVarianceFilterer(BaseH2OFeatureSelector):
	"""Identify and remove any features that have a variance below
	a certain threshold.

	Parameters
	----------
	target_feature : str (default None)
		The name of the target feature (is excluded from the fit)

	threshold : float, default 1e-6
		The threshold below which to declare "zero variance"
		
	na_warn : bool (default True)
		Whether to warn if any NAs are present

	na_rm : bool (default False)
		Whether to remove NA values

	use : str (default "complete.obs"), one of {'complete.obs','all.obs','everything'}
		A string indicating how to handle missing values.

	Attributes
	----------
	drop : list, string
		The columns to drop
	"""
	
	__min_version__ = '3.8.2.9'
	__max_version__ = None
	
	def __init__(self, feature_names=None, target_feature=None, threshold=1e-6, 
				 na_warn=True, na_rm=False, use='complete.obs'):

		super(H2ONearZeroVarianceFilterer, self).__init__(feature_names=feature_names,
														  target_feature=target_feature, 
														  min_version=self.__min_version__,
														  max_version=self.__max_version__)
		self.threshold = threshold
		self.na_warn = na_warn
		self.na_rm = na_rm
		self.use = use

	def fit(self, X):
		"""Fit the near zero variance filterer,
		return the transformed X frame.

		Parameters
		----------
		X : H2OFrame
			The frame to fit
		"""
		self.fit_transform(X)
		return self

	def fit_transform(self, X):
		"""Fit the near zero variance filterer.

		Parameters
		----------
		X : H2OFrame
			The frame to fit
		"""
		frame, thresh = _check_is_frame(X), self.threshold
		frame = _frame_from_x_y(frame, self.feature_names, self.target_feature)

		# validate use, check NAs
		use = _validate_use(frame, self.use, self.na_warn)

		cols = frame.columns
		variances = [frame[n].var(use=use, na_rm=self.na_rm) for n in cols]
		var_mask = np.asarray(variances) < thresh
		
		self.drop_ = [str(n) for n in np.asarray(cols)[var_mask]] # make them strings
		self.vars_ = dict(zip(self.drop_, np.asarray(variances)[var_mask]))

		return self.transform(X)
			
		