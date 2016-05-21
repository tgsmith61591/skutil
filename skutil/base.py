
__all__ = [
	'SelectiveMixin'
]

class SelectiveMixin:
	"""A mixin class that all selective transformers
	should implement. Returns the columns used to transform on.

	Attributes
	----------
	cols_ : array_like
		The columns transformed
	"""

	def get_features(self):
		return self.cols_

	def set_features(self, cols=None):
		self.cols_ = cols