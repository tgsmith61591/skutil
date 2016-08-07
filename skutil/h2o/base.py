from __future__ import print_function, division
from sklearn.base import BaseEstimator, TransformerMixin
import abc
import h2o
import warnings
from sklearn.externals import six
from pkg_resources import parse_version
from ..utils import is_numeric


__all__ = [
	'NAWarning',
	'BaseH2OFunctionWrapper',
	'BaseH2OTransformer'
]


class NAWarning(UserWarning):
	"""Custom warning used to notify user that an NA exists
	within an h2o frame (h2o can handle NA values)
	"""


def _check_is_frame(X):
	"""Returns X if X is a frame else throws a TypeError"""
	if not isinstance(X, H2OFrame):
		raise TypeError('expected H2OFrame but got %s' % type(X))
	return X

def _retain_features(X, exclude):
	"""Returns the features to retain"""
	return [x for x in X.columns if not x in exclude]



class BaseH2OFunctionWrapper(BaseEstimator):
	"""Base class for all H2O estimators or functions.

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
	


class BaseH2OTransformer(BaseH2OFunctionWrapper, TransformerMixin):
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
		super(BaseH2OTransformer, self).__init__(target_feature=target_feature,
												 min_version=min_version,
												 max_version=max_version)
			
	def transform(self, X):
		# validate state, frame
		check_is_fitted(self, 'drop_')
		X = _check_is_frame(X)

		retain = _retain_features(X, self.drop_)
		return X[retain]