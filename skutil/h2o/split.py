from __future__ import division, print_function, absolute_import
from abc import ABCMeta, abstractmethod
import warnings
import time
import numbers
import numpy as np
import pandas as pd

import h2o
from h2o.frame import H2OFrame
try:
	from h2o import H2OEstimator
except ImportError as e:
	from h2o.estimators.estimator_base import H2OEstimator

from .base import _check_is_frame
from ..base import overrides

from sklearn.externals import six
from sklearn.base import _pprint
from sklearn.utils.fixes import signature, bincount
from sklearn.utils import check_random_state
from math import ceil, floor

try:
	from sklearn.model_selection import KFold
	SK18 = True
except ImportError as e:
	from sklearn.cross_validation import KFold
	SK18 = False


__all__ = [
	'check_cv',
	'h2o_train_test_split',
	'H2OKFold',
	'H2OShuffleSplit',
	'H2OStratifiedKFold',
	'H2OStratifiedShuffleSplit'
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


def h2o_train_test_split(frame, test_size=None, train_size=None, random_state=None, stratify=None):
	"""Splits an H2OFrame into random train and test subsets

	Parameters
	----------
	frame : H2OFrame
		The h2o frame to split

	test_size : float, int, or None (default=None)
		If float, should be between 0.0 and 1.0 and represent the
		proportion of the dataset to include in the test split. If
		int, represents the absolute number of test samples. If None,
		the value is automatically set to the complement of the train size.
		If train size is also None, test size is set to 0.25

	train_size : float, int, or None (default=None)
		If float, should be between 0.0 and 1.0 and represent the
		proportion of the dataset to include in the train split. If
		int, represents the absolute number of train samples. If None,
		the value is automatically set to the complement of the test size.

	random_state : int or RandomState
		Pseudo-random number generator state used for random sampling.

	stratify : str or None (default=None)
		The name of the target on which to stratify the sampling
	"""
	_check_is_frame(frame)
	if test_size is None and train_size is None:
		test_size = 0.25

	if stratify is not None:
		CVClass = H2OStratifiedShuffleSplit
	else:
		CVClass = H2OShuffleSplit

	cv = CVClass(n_splits=2,
				 test_size=test_size,
				 train_size=train_size,
				 random_state=random_state)

	# for the h2o one, we only need iter 0
	tr_te_tuples = [(tr,te) for tr,te in cv.split(frame, stratify)][0]
	train, test = list(tr_te_tuples[0]), list(tr_te_tuples[1])
	out = (
		frame[train, :], 
		frame[test,  :]
	)

	return out



# Avoid a pb with nosetests...
h2o_train_test_split.__test__ = False




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


def _validate_shuffle_split_init(test_size, train_size):
	"""Validation helper to check the test_size and train_size at init"""
	if test_size is None and train_size is None:
		raise ValueError('test_size and train_size can not both be None')

	if test_size is not None:
		if np.asarray(test_size).dtype.kind == 'f':
			if test_size >= 1.:
				raise ValueError(
					'test_size=%f should be smaller '
					'than 1.0 or be an integer' % test_size)
		elif np.asarray(test_size).dtype.kind != 'i':
			raise ValueError('Invalid value for test_size: %r' % test_size)

	if train_size is not None:
		if np.asarray(train_size).dtype.kind == 'f':
			if train_size >= 1.:
				raise ValueError(
					'train_size=%f should be smaller '
					'than 1.0 or be an integer' % test_size)
			elif (np.asarray(test_size).dtype.kind == 'f' and
					(train_size + test_size) > 1.):
				raise ValueError('The sum of test_size and train_size = %f'
					'should be smaller than 1.0. Reduce test_size '
					'and/or train_size.' % (train_size + test_size))
		elif np.asarray(train_size).dtype.kind != 'i':
			raise ValueError('Invalid value for train_size: %r' % train_size)


def _validate_shuffle_split(n_samples, test_size, train_size):
	if (test_size is not None and np.asarray(test_size).dtype.kind == 'i'
			and test_size >= n_samples):
		raise ValueError('test_size=%d should be smaller '
			'than the number of samples %d' % (test_size, n_samples))

	if (train_size is not None and np.asarray(train_size).dtype.kind == 'i'
			and train_size >= n_samples):
		raise ValueError('train_size=%d should be smaller '
			'than the number of samples %d' % (train_size, n_samples))

	if np.asarray(test_size).dtype.kind == 'f':
		n_test = ceil(test_size * n_samples)
	elif np.asarray(test_size).dtype.kind == 'i':
		n_test = float(test_size)

	if train_size is None:
		n_train = n_samples - n_test
	elif np.asarray(train_size).dtype.kind == 'f':
		n_train = floor(train_size * n_samples)
	else:
		n_train = float(train_size)

	if test_size is None:
		n_test = n_samples - n_train

	if n_train + n_test > n_samples:
		raise ValueError('The sum of train_size and test_size=%d, '
			'should be smaller than the number of '
			'samples %d. Reduce test_size and/or '
			'train_size.' % (n_train+n_test, n_samples))

	return int(n_train), int(n_test)


class H2OBaseShuffleSplit(six.with_metaclass(ABCMeta)):
	"""Base class for H2OShuffleSplit and H2OStratifiedShuffleSplit"""

	def __init__(self, n_splits=2, test_size=0.1, train_size=None, random_state=None):
		_validate_shuffle_split_init(test_size, train_size)
		self.n_splits = n_splits
		self.test_size = test_size
		self.train_size = train_size
		self.random_state= random_state

	def split(self, frame, y=None):
		for train, test in self._iter_indices(frame, y):
			yield train, test

	@abstractmethod
	def _iter_indices(self, frame, y):
		pass

	def get_n_splits(self, frame=None, y=None):
		return self.n_splits

	def __repr__(self):
		return _build_repr(self)


class H2OShuffleSplit(H2OBaseShuffleSplit):
	def _iter_indices(self, frame, y=None):
		n_samples = frame.shape[0]
		n_train, n_test = _validate_shuffle_split(n_samples, self.test_size, self.train_size)

		rng = check_random_state(self.random_state)
		for i in range(self.n_splits):
			permutation = rng.permutation(n_samples)
			ind_test = permutation[:n_test]
			ind_train = permutation[n_test:(n_test + n_train)]
			yield ind_train, ind_test


class H2OStratifiedShuffleSplit(H2OBaseShuffleSplit):
	def __init__(self, n_splits=2, test_size=0.1, train_size=None, random_state=None):
		super(H2OStratifiedShuffleSplit, self).__init__(
			n_splits=n_splits, 
			test_size=test_size, 
			train_size=train_size, 
			random_state=random_state)

	def _iter_indices(self, frame, y):
		n_samples = frame.shape[0]
		n_train, n_test = _validate_shuffle_split(n_samples, 
			self.test_size, self.train_size)

		# need to validate y...
		y = _val_y(y)
		target = np.asarray(frame[y].as_data_frame(use_pandas=True)[y].tolist())

		classes, y_indices = np.unique(target, return_inverse=True)
		n_classes = classes.shape[0]

		class_counts = bincount(y_indices)
		if np.min(class_counts) < 2:
			raise ValueError('The least populated class in y has only 1 '
				'member, which is too few. The minimum number of labels '
				'for any class cannot be less than 2.')

		if n_train < n_classes:
			raise ValueError('The train_size=%d should be greater than or '
				'equal to the number of classes=%d' % (n_train, n_classes))

		if n_test < n_classes:
			raise ValueError('The test_size=%d should be greater than or '
				'equal to the number of classes=%d' % (n_test, n_classes))

		rng = check_random_state(self.random_state)
		p_i = class_counts / float(n_samples)
		n_i = np.round(n_train * p_i).astype(int)
		t_i = np.minimum(class_counts - n_i, np.round(n_test * p_i).astype(int))

		for _ in range(self.n_splits):
			train = []
			test = []

			for i, class_i in enumerate(classes):
				permutation = rng.permutation(class_counts[i])
				perm_indices_class_i = np.where((target==class_i))[0][permutation]

				train.extend(perm_indices_class_i[:n_i[i]])
				test.extend(perm_indices_class_i[n_i[i]:n_i[i] + t_i[i]])

			# Might end up here with less samples in train and test than we asked
			# for, due to rounding errors.
			if len(train) + len(test) < n_train + n_test:
				missing_indices = np.where(bincount(train + test, minlength=len(target)) == 0)[0]
				missing_indices = rng.permutation(missing_indices)
				n_missing_train = n_train - len(train)
				n_missing_test  = n_test - len(test)

				if n_missing_train > 0:
					train.extend(missing_indices[:n_missing_train])
				if n_missing_test > 0:
					test.extend(missing_indices[-n_missing_test:])

			train = rng.permutation(train)
			test = rng.permutation(test)

			yield train, test

	def split(self, frame, y):
		return super(H2OStratifiedShuffleSplit, self).split(frame, y)



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
		# So we pass np.zeroes(max(c, n_folds)) as data to the KFold. 

		# Remember, however that we might be using the old-fold KFold which doesn't
		# have a split method...
		if SK18:
			per_cls_cvs = [
				KFold(self.n_folds, # using sklearn's KFold here
					shuffle=self.shuffle,
					random_state=rng).split(np.zeros(max(count, self.n_folds)))
				for count in y_counts
			]
		else:
			per_cls_cvs = [
				KFold(max(count, self.n_folds), # using sklearn's KFold here
					self.n_folds, 
					shuffle=self.shuffle,
					random_state=rng)
				for count in y_counts
			]


		test_folds = np.zeros(n_samples, dtype=np.int)
		for test_fold_indices, per_cls_splits in enumerate(zip(*per_cls_cvs)):
			for cls, (_, test_split) in zip(unique_y, per_cls_splits):
				cls_test_folds = test_folds[target == cls]

				# the test split can be too big because we used
				# KFold(...).split(X[:max(c, n_folds)]) when data is not 100%
				# stratifiable for all the classes
				# (we use a warning instead of raising an exception)
				# If this is the case, let's trim it:
				test_split = test_split[test_split < len(cls_test_folds)]
				cls_test_folds[test_split] = test_fold_indices
				test_folds[target == cls] = cls_test_folds

		return test_folds


