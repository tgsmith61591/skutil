from __future__ import division, print_function, absolute_import
from abc import ABCMeta, abstractmethod
import warnings
import time
import numbers
import numpy as np
import pandas as pd

import h2o
from h2o.frame import H2OFrame
from h2o import H2OEstimator
from .base import _check_is_frame
from ..base import overrides

from sklearn.externals import six
from sklearn.base import _pprint
from sklearn.utils.fixes import signature
from sklearn.utils import check_random_state


__all__ = [
	'check_cv',
	'H2OKFold'
]




def _build_repr(self):
	# XXX This is copied from sklearn.BaseEstimator's get_params
	cls = self.__class__
	init = getattr(cls.__init__, 'deprecated_original', cls.__init__)

	init_signature = signature(init)

	if init is object.__init__:
		args = []
	else:
		args = sorted([p.name for p in init_signature.parameters.values()
						if p.name != 'self' and p.kind != p.VAR_KEYWORD])

	class_name = self.__class__.__name__
	params = dict()
	for key in args:
		warnings.simplefilter("always", DeprecationWarning)
		try:
			with warnings.catch_warnings(record=True) as w:
				value = getattr(self, key, None)
			if len(w) and w[0].category == DeprecationWarning:
				continue
		finally:
			warnings.filters.pop(0)
		params[key] = value

	return '%s(%s)' % (class_name, _pprint(params, offset=len(class_name)))


def check_cv(cv=3):
	if cv is None:
		cv = 3

	if isinstance(cv, numbers.Integral):
		return H2OKFold(cv)

	if not isinstance(cv, H2OBaseCrossValidator):
		raise ValueError('expected int or instance of '
						 'H2OBaseCrossValidator but got %s'
						 % type(cv))

	return cv


class H2OBaseCrossValidator(six.with_metaclass(ABCMeta)):

	def __init__(self):
		pass

	def split(self, frame):
		"""Generate indices to split data into training and test.

		Parameters
		----------
		frame : H2OFrame
			The h2o frame to split

		Returns
		-------
		train : ndarray
			The training set indices for the split

		test : ndarray
			The testing set indices for that split
		"""

		_check_is_frame(frame)
		indices = np.arange(frame.shape[0])
		for test_index in self._iter_test_masks(frame):
			train_index = indices[np.logical_not(test_index)]
			test_index = indices[test_index]
			yield train_index, test_index

	def _iter_test_masks(self, frame):
		"""Generates boolean masks corresponding to the tests set."""
		for test_index in self._iter_test_indices(frame):
			test_mask = np.zeros(frame.shape[0], dtype=np.bool)
			test_mask[test_index] = True
			yield test_mask

	def _iter_test_indices(self, frame):
		raise NotImplementedError

	@abstractmethod
	def get_n_splits(self, frame):
		pass

	def __repr__(self):
		return _build_repr(self)


class _H2OBaseKFold(six.with_metaclass(ABCMeta, H2OBaseCrossValidator)):
	"""Base class for KFold and Stratified KFold"""

	@abstractmethod
	def __init__(self, n_folds, shuffle, random_state):
		if not isinstance(n_folds, numbers.Integral):
			raise ValueError('n_folds must be of Integral type. '
							 '%s of type %s was passed' % (n_folds, type(n_folds)))

		n_folds = int(n_folds)
		if n_folds <= 1:
			raise ValueError('k-fold cross-validation requires at least one '
							 'train/test split by setting n_folds=2 or more')

		if not isinstance(shuffle, bool):
			raise TypeError('shuffle must be True or False')

		self.n_folds = n_folds
		self.shuffle = shuffle
		self.random_state = random_state

	@overrides(H2OBaseCrossValidator)
	def split(self, frame):
		_check_is_frame(frame)
		n_obs = frame.shape[0]

		if self.n_folds > n_obs:
			raise ValueError('Cannot have n_folds greater than n_obs')

		for train, test in super(H2OBaseCrossValidator, self).split(frame):
			yield train, test

	@overrides(H2OBaseCrossValidator)
	def get_n_splits(self):
		return self.n_folds


class H2OKFold(_H2OBaseKFold):
	"""K-folds cross-validator for an H2OFrame"""

	def __init__(self, n_folds=3, shuffle=False, random_state=None):
		super(H2OKFold, self).__init__(n_folds, shuffle, random_state)

	@overrides(_H2OBaseKFold)
	def _iter_test_indices(self, frame):
		n_obs = frame.shape[0]
		indices = np.arange(n_obs)
		if self.shuffle:
			check_random_state(self.random_state).shuffle(indices)

		n_folds = self.n_folds
		fold_sizes = (n_obs // n_folds) * np.ones(n_folds, dtype=np.int)
		fold_sizes[:n_obs % n_folds] += 1
		current = 0
		for fold_size in fold_sizes:
			start, stop = current, current + fold_size
			yield indices[start:stop]
			current = stop


