from __future__ import print_function, division, absolute_import
import numpy as np
import h2o
import pandas as pd
from ..utils import validate_is_pd
from h2o.frame import H2OFrame


__all__ = [
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