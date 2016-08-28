from __future__ import print_function, division, absolute_import
from .base import BaseH2OTransformer, _frame_from_x_y, _check_is_frame
from ..utils import is_numeric, flatten_all
import numpy as np
import numbers
from sklearn.utils.validation import check_is_fitted

__all__ = [
	'H2OSelectiveImputer'
]


def _flatten_one(x):
	"""There is a bug in some versions of h2o
	where a scalar is not returned by mean, but
	a list is. This will determine the proper 
	type for each item in the vec.
	"""
	return x[0] if hasattr(x, '__iter__') else x


class _H2OBaseImputer(BaseH2OTransformer):
	"""A base class for all H2O imputers"""
	__def_fill__ = -999999

	def __init__(self, feature_names=None, target_feature=None, min_version='any', max_version=None, def_fill=None):
		super(_H2OBaseImputer, self).__init__(feature_names=feature_names,
											  target_feature=target_feature, 
											  min_version=min_version,
											  max_version=max_version)
		self.fill_ = self.__def_fill__ if def_fill is None else def_fill


class H2OSelectiveImputer(_H2OBaseImputer):

	__min_version__ = '3.8.2.9'
	__max_version__ = None

	def __init__(self, feature_names=None, target_feature=None, def_fill='mean'):
		super(H2OSelectiveImputer, self).__init__(feature_names=feature_names,
												  target_feature=target_feature,
												  def_fill=def_fill)

	def fit(self, X):
		frame = _check_is_frame(X)
		frame = _frame_from_x_y(frame, self.feature_names, self.target_feature)

		# at this point, the entirety of frame can be operated on...
		cols = [str(u) for u in frame.columns] # convert to string...

		# validate the fill, do fit
		fill = self.fill_
		if isinstance(fill, (str, unicode)):
			if not str(fill) in ('mode', 'mean', 'median'):
				raise TypeError('self.fill must be either "mode", "mean", "median", None, '
								'a number, or an iterable. Got %s' % fill)

			if fill == 'mode':
				# for each column to impute, we go through and get the value counts
				# of each, sorting by the max...
				raise NotImplementedError('h2o has not yet implemented "mode" functionality')

			elif fill == 'median':
				self.fill_val_ = dict(zip(cols, flatten_all([X[c].median(na_rm=True) for c in cols])))

			else:
				self.fill_val_ = dict(zip(cols, flatten_all([X[c].mean(na_rm=True) for c in cols])))


		elif hasattr(fill, '__iter__'):
			# we need to get the length of the iterable,
			# make sure it matches the len of cols
			if not len(fill) == len(cols):
				raise ValueError('len of fill does not match that of cols')

			# make sure they're all ints
			if not all([
					(is_numeric(i) or \
						(isinstance(i, (str,unicode))) and \
						i in ('mode', 'mean', 'median')) \
					for i in fill
				]):

				raise TypeError('All values in self.fill must be numeric or in ("mode", "mean", "median"). '
								'Got: %s' % ', '.join(fill))

			d = {}
			for ind, c in enumerate(cols):
				f = fill[ind]

				if is_numeric(f): # if we fill with a single value...
					d[c] = f
				else:
					the_col = X[c]
					if f == 'mode':
						raise NotImplementedError('h2o has not yet implemented "mode" functionality')
						# d[c] = _col_mode(the_col)
					elif f == 'median':
						d[c] = _flatten_one(the_col.median(na_rm=True))
					else:
						d[c] = _flatten_one(the_col.mean(na_rm=True))


			self.fill_val_ = d

		else:
			if not is_numeric(fill):
				raise TypeError('self.fill must be either "mode", "mean", "median", None, '
								'a number, or an iterable. Got %s' % str(fill))

			# either the fill is an int, or it's something the user provided...
			# if it's not an int or float, we'll let it go and not catch it because
			# the it's their fault they were dumb.
			self.fill_val_ = fill

		return self


	def transform(self, X):
		"""Transform an H2OFrame given the fit imputer.

		Parameters
		----------
		X : pandas DataFrame
			The frame to fit

		y : None, passthrough for pipeline
		"""

		check_is_fitted(self, 'fill_val_')
		X = _check_is_frame(X)

		# get the fills
		fill_val = self.fill_val_

		# we get the subset frame just to retrieve the column names. We affect
		# X in place anyways, so no use using the slice...
		frame = _frame_from_x_y(X, self.feature_names, self.target_feature)
		cols  = [str(u) for u in frame.columns] # the cols we'll ultimately impute
		X_columns = [str(u) for u in X.columns] # used for index lookup

		# get the frame of NAs
		na_frame = frame.isna()
		na_frame.columns = cols

		#iter over cols
		is_int = isinstance(fill_val, int) # is it an int?
		for _, col in enumerate(cols):
			if not is_int and not col in fill_val: # then it's a dict and this col doesn't exist in it...
				continue

			# get the column index
			col_idx = X_columns.index(col)

			# if it's a single int, easy, otherwise query dict
			col_imp_value = fill_val if is_int else fill_val[col]

			# unfortunately, since we can't boolean index the
			# h2oframe, we have to convert pandas
			the_na_col = na_frame[col].as_data_frame(use_pandas=True)[col]
			na_mask_idcs = the_na_col.index[the_na_col == 1].tolist()

			for na_row in na_mask_idcs:
				X[na_row, col_idx] = col_imp_value


		# this is going to impact it in place...
		return X
