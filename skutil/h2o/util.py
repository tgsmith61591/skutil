from __future__ import print_function, division, absolute_import
import numpy as np
import h2o
import pandas as pd
from ..utils import validate_is_pd
from h2o.frame import H2OFrame
from sklearn.utils.validation import check_array


__all__ = [
	'from_array',
	'from_pandas'
]


def from_pandas(X):
	"""A simple wrapper for H2OFrame.from_python. This takes
	a pandas dataframe and returns an H2OFrame with all the 
	default args (generally enough) plus named columns.

	Parameters
	----------
	X : pd.DataFrame
		The dataframe to convert.

	Returns
	-------
	H2OFrame
	"""
	pd, _ = validate_is_pd(X, None)

	# if h2o hasn't started, we'll let this fail through
	return H2OFrame.from_python(X, header=1, column_names=X.columns.tolist())

def from_array(X, column_names=None):
	"""A simple wrapper for H2OFrame.from_python. This takes a
	numpy array (or 2d array) and returns an H2OFrame with all 
	the default args.

	Parameters
	----------
	X : ndarray
		The array to convert.

	column_names : list, tuple (default=None)
		the names to use for your columns

	Returns
	-------
	H2OFrame
	"""
	X = check_array(X, force_all_finite=False)
	return from_pandas(pd.DataFrame.from_records(data=X, columns=column_names))
