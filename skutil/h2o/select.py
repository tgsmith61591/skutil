from __future__ import print_function, division
import warnings
import numpy as np
import pandas as pd
import abc
from pkg_resources import parse_version

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.externals import six

import h2o
from h2o.frame import H2OFrame

from ..utils import is_numeric
from ..feature_selection import filter_collinearity
from .base import NAWarning


__all__ = [
	'BaseH2OTransformer',
	'H2OMulticollinearityFilterer',
	'H2ONearZeroVarianceFilterer'
]


def _check_is_frame(X):
	"""Returns X if X is a frame else throws a TypeError"""
	if not isinstance(X, H2OFrame):
		raise TypeError('expected H2OFrame but got %s' % type(X))
	return X

def _retain_features(X, exclude):
	"""Returns the features to retain"""
	return [x for x in X.columns if not x in exclude]

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



class BaseH2OTransformer(BaseEstimator, TransformerMixin):
	"""Base class for all H2OTransformers.

	Parameters
	----------
	target_feature : str (default None)
		The name of the target feature (is excluded from the fit)

	min_version : str, float (default 'any')
		The minimum version of h2o that is compatible with the transformer

	max_version : str, float (default None)
		The maximum version of h2o that is compatible with the transformer
	"""
	
	@abc.abstractmethod
	def __init__(self, target_feature=None, min_version='any', max_version=None):
		self.target_feature = target_feature
		
		# validate min version
		h2ov = h2o.__version__
				
		# won't enter this block if passed at 'any'
		if is_numeric(min_version): # then int or float
			min_version = str(min_version)
		
		if isinstance(min_version, str):
			if min_version == 'any':
				pass # anything goes
			else:
				if parse_version(h2ov) < parse_version(min_version):
					raise EnvironmentError('your h2o version (%s) '
										   'does not meet the minimum ' 
										   'requirement for this transformer (%s)'
										   % (h2ov, str(min_version)))
		
		else:
			raise ValueError('min_version must be a float, '
							 'a string in the form of "X.x" '
							 'or "any", but got %s' % type(min_version))



		# validate max version
		if not max_version:
			pass
	   	elif is_numeric(max_version):
	   		max_version = str(max_version)

	   	if isinstance(max_version, str):
	   		if parse_version(h2ov) > parse_version(max_version):
	   			raise EnvironmentError('your h2o version (%s) '
									   'exceeds the maximum permitted ' 
									   'version for this transformer (%s)'
									   % (h2ov, str(max_version)))
	   	elif not max_version is None: # remember we allow None
	   		raise ValueError('max_version must be a float, '
							 'a string in the form of "X.x" '
							 'or None, but got %s' % type(max_version))


	   	# test connection, warn where needed
		try:
			g = h2o.frames() # returns a dict of frames
		except (EnvironmentError, ValueError) as v:
			warnings.warn('h2o has not been started; '
						  'initializing an H2O transformer without '
						  'a connection will not cause any issues, '
						  'but it will throw a ValueError if the '
						  'H2O cloud is not started prior to fitting')
			
	def transform(self, X):
		# validate state, frame
		check_is_fitted(self, 'drop_')
		X = _check_is_frame(X)

		retain = _retain_features(X, self.drop_)
		return X[retain]


	@property
	def max_version(self):
		try:
			mv = self.__max_version__
			return mv if not mv else str(mv)
		except NameError as n:
			return None
	

	@property
	def min_version(self):
		try:
			return str(self.__min_version__)
		except NameError as n:
			return 'any'


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

		super(H2OMulticollinearityFilterer, self).__init__(target_feature, 
														   self.__min_version__,
														   self.__max_version__)
		self.threshold = threshold
		self.na_warn = na_warn
		self.na_rm = na_rm
		self.use = use
		
		
	def fit(self, X, y=None):
		"""Fit the multicollinearity filterer.

		Parameters
		----------
		X : H2OFrame
			The frame to fit

		y : None, passthrough for pipeline
		"""

		self.fit_transform(X, y)
		return self
	
	
	def fit_transform(self, X, y=None):
		"""Fit the multicollinearity filterer and
		return the transformed H2OFrame, X.

		Parameters
		----------
		X : H2OFrame
			The frame to fit

		y : None, passthrough for pipeline
		"""
		frame, thresh = _check_is_frame(X), self.threshold
		
		# if there's a target feature, let's strip it out for now...
		if self.target_feature:
			X_nms = [x for x in frame.columns if not x == self.target_feature] # make list
			frame = frame[X_nms]

		# validate use, check NAs
		use = _validate_use(frame, self.use, self.na_warn)
		
		## Generate absolute correlation matrix
		c = frame.cor(use=use, na_rm=self.na_rm).abs().as_data_frame(use_pandas=True)
		c.columns = frame.columns # set the cols to the same names
		c.index = [x for x in frame.columns] # set the index to the same names
		
		## get drops list
		self.drop_ = filter_collinearity(c, self.threshold)
		retain = _retain_features(X, self.drop_) # pass original frame

		return frame[retain]
		


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

	def fit(self, X, y = None):
		"""Fit the near zero variance filterer.

		Parameters
		----------
		X : H2OFrame
			The frame to fit

		y : None, passthrough for pipeline
		"""
		frame, thresh = _check_is_frame(X), self.threshold
		
		# if there's a target feature, let's strip it out for now...
		if self.target_feature:
			X_nms = [x for x in frame.columns if not x == self.target_feature] # make list
			frame = frame[X_nms]

		# validate use, check NAs
		use = _validate_use(frame, self.use, self.na_warn)

		cols = frame.columns
		self.drop_ = [str(n) for n in cols if (frame[n].var(use=use, na_rm=self.na_rm) < thresh)]

		return self
			
		