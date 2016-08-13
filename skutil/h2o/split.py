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
from sklearn.utils.fixes import signature, bincount
from sklearn.utils import check_random_state


__all__ = [
	'check_cv',
	'H2OKFold',
	'H2OStratifiedKFold'
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

def _val_y(y):
	if isinstance(y, (str, unicode)):
		return str(y)
	elif y is None:
		return y
	raise TypeError('y must be a string. Got %s' % y)


class H2OBaseCrossValidator(six.with_metaclass(ABCMeta)):

	def __init__(self):
		pass

	def split(self, frame, y=None):
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
		for test_index in self._iter_test_masks(frame, y):
			train_index = indices[np.logical_not(test_index)]
			test_index = indices[test_index]

			# h2o can't handle anything but lists...
			yield list(train_index), list(test_index)

	def _iter_test_masks(self, frame, y=None):
		"""Generates boolean masks corresponding to the tests set."""
		for test_index in self._iter_test_indices(frame, y):
			test_mask = np.zeros(frame.shape[0], dtype=np.bool)
			test_mask[test_index] = True
			yield test_mask

	def _iter_test_indices(self, frame, y=None):
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

		if not shuffle in [True, False]:
			raise TypeError('shuffle must be True or False. Got %s (type=%s)'
				% (str(shuffle), type(shuffle)))

		self.n_folds = n_folds
		self.shuffle = shuffle
		self.random_state = random_state

	@overrides(H2OBaseCrossValidator)
	def split(self, frame, y=None):
		_check_is_frame(frame)
		n_obs = frame.shape[0]

		if self.n_folds > n_obs:
			raise ValueError('Cannot have n_folds greater than n_obs')

		for train, test in super(_H2OBaseKFold, self).split(frame, y):
			yield train, test

	@overrides(H2OBaseCrossValidator)
	def get_n_splits(self):
		return self.n_folds


class H2OKFold(_H2OBaseKFold):
	"""K-folds cross-validator for an H2OFrame"""

	def __init__(self, n_folds=3, shuffle=False, random_state=None):
		super(H2OKFold, self).__init__(n_folds, shuffle, random_state)

	@overrides(_H2OBaseKFold)
	def _iter_test_indices(self, frame, y=None):
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


class H2OStratifiedKFold(_H2OBaseKFold):

	def __init__(self, n_folds=3, shuffle=False, random_state=None):
		super(H2OStratifiedKFold, self).__init__(n_folds, shuffle, random_state)


	def split(self, frame, y):
		return super(H2OStratifiedKFold, self).split(frame, y)


	def _iter_test_masks(self, frame, y):
		test_folds = self._make_test_folds(frame, y)
		for i in range(self.n_folds):
			yield test_folds == i


	def _make_test_folds(self, frame, y):
		if self.shuffle:
			rng = check_random_state(self.random_state)
		else:
			rng = self.random_state

		# validate that it's a string
		y = _val_y(y) # gets a string back or None
		if y is None:
			raise ValueError('H2OStratifiedKFold requires a target name (got None)')
		
		target = frame[y].as_data_frame(use_pandas=True)[y].values
		n_samples = target.shape[0]
		unique_y, y_inversed = np.unique(target, return_inverse=True)
		y_counts = bincount(y_inversed)
		min_labels = np.min(y_counts)

		if np.all(self.n_folds > y_counts):
			raise ValueError(('All the n_labels for individual classes'
							  ' are less than %d folds.' 
							  % (self.n_folds)), Warning)
		if self.n_folds > min_labels:
			warnings.warn(('The least populated class in y has only %d'
						   ' members, which is too few. The minimum'
						   ' number of labels for any class cannot'
						   ' be less than n_folds=%d.' 
						   % (min_labels, self.n_folds)), Warning)

		# NOTE FROM SKLEARN:

		# pre-assign each sample to a test fold index using individual KFold
		# splitting strategies for each class so as to respect the balance of
		# classes
		# NOTE: Passing the data corresponding to ith class say X[y==class_i]
		# will break when the data is not 100% stratifiable for all classes.
		# So we pass np.zeroes(max(c, n_folds)) as data to the KFold
		per_cls_cvs = [
			H2OKFold(self.n_folds, shuffle=self.shuffle,
				random_state=rng).split(np.zeros(max(count, self.n_folds)))
			for count in y_counts
		]

		test_folds = np.zeros(n_samples, dtype=np.int)
		for test_fold_indices, per_cls_splits in enumerate(zip(*per_cls_cvs)):
			for cls, (_, test_split) in zip(unique_y, per_cls_splits):
				cls_test_folds = test_folds[y==cls]

				# the test split can be too big because we used
				# KFold(...).split(X[:max(c, n_folds)]) when data is not 100%
				# stratifiable for all the classes
				# (we use a warning instead of raising an exception)
				# If this is the case, let's trim it:
				test_split = test_split[test_split < len(cls_test_folds)]
				cls_test_folds[test_split] = test_fold_indices
				test_folds[y == cls] = cls_test_folds

		return test_folds


