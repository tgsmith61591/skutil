from __future__ import print_function, division
import warnings

__all__ = [
	'NAWarning'
]

class NAWarning(UserWarning):
	"""Custom warning used to notify user that an NA exists
	within an h2o frame (h2o can handle NA values)
	"""