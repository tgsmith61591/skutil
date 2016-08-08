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
from .base import (NAWarning, 
				   BaseH2OTransformer, 
				   _check_is_frame, 
				   _retain_features)


__all__ = [
	'H2OMulticollinearityFilterer',
	'H2ONearZeroVarianceFilterer'
]


def _validate_use(X, use, na_warn):
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


class H2OMulticollinearityFilterer(BaseH2OTransformer):
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
	drop : list, string
		The columns to drop
	"""
	
	__min_version__ = '3.8.3'
	__max_version__ = None
	
	def __init__(self, target_feature=None, threshold=0.85, 
				 na_warn=True, na_rm=False, use='complete.obs'):

		super(H2OMulticollinearityFilterer, self).__init__(target_feature=target_feature, 
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

		y : None, passthrough for pipeline
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
		
		# if there's a target feature, let's strip it out for now...
		if self.target_feature is not None:
			frame = frame[[x for x in frame.columns if not x == self.target_feature]] # make list

		# validate use, check NAs
		use = _validate_use(frame, self.use, self.na_warn)
		
		## Generate absolute correlation matrix
		c = frame.cor(use=use, na_rm=self.na_rm).abs().as_data_frame(use_pandas=True)
		c.columns = frame.columns # set the cols to the same names
		c.index = frame.columns
		
		## get drops list
		self.drop_ = filter_collinearity(c, self.threshold)
		return self.transform(X)
		


class H2ONearZeroVarianceFilterer(BaseH2OTransformer):
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
	
	__min_version__ = '3.8.3'
	__max_version__ = None
	
	def __init__(self, target_feature=None, threshold=1e-6, 
				 na_warn=True, na_rm=False, use='complete.obs'):

		super(H2ONearZeroVarianceFilterer, self).__init__(target_feature, 
														   self.__min_version__,
														   self.__max_version__)
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
		
		# if there's a target feature, let's strip it out for now...
		if self.target_feature:
			frame = frame[[x for x in frame.columns if not x == self.target_feature]] # make list

		# validate use, check NAs
		use = _validate_use(frame, self.use, self.na_warn)

		cols = frame.columns
		self.drop_ = [str(n) for n in cols if (frame[n].var(use=use, na_rm=self.na_rm) < thresh)]
		return self.transform(X)
			
		