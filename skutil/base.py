import abc

__all__ = [
	'BalancerMixin',
	'ModuleImportWarning',
	'SamplingWarning',
	'SelectiveMixin',
	'SelectiveWarning'
]

###############################################################################
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

class ModuleImportWarning(UserWarning):
	"""Custom warning used to notify user a non-critical import failed, and to
	suggest the installation of the module for optimal results.
	"""

class SamplingWarning(UserWarning):
	"""Custom warning used to notify the user that sub-optimal sampling behavior
	has occurred. For instance, performing oversampling on a minority class with only
	one instance will cause this warning to be thrown.
	"""

class SelectiveWarning(UserWarning):
	"""Custom warning used to notify user when a structure implementing SelectiveMixin
	operates improperly. A common usecase is when the fit method receives a non-DataFrame
	X, and no cols.
	"""

class SelectiveMixin:
	"""A mixin class that all selective transformers
	should implement. Returns the columns used to transform on.

	Attributes
	----------
	cols : array_like
		The columns transformed
	"""

	def get_features(self):
		return self.cols

	def set_features(self, cols=None):
		self.cols = cols