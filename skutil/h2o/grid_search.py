from __future__ import division, print_function, absolute_import
from abc import ABCMeta, abstractmethod
import warnings
import time
import numpy as np
import pandas as pd

import h2o
from h2o.frame import H2OFrame
try:
	from h2o import H2OEstimator
except ImportError as e:
	from h2o.estimators.estimator_base import H2OEstimator

from .pipeline import H2OPipeline
from .base import _check_is_frame, BaseH2OFunctionWrapper, validate_x_y, VizMixin
from ..base import overrides
from ..utils import is_numeric, report_grid_score_detail
from ..grid_search import _CVScoreTuple, _check_param_grid
from ..metrics import ActStatisticalReport
from .split import check_cv
from .split import *

from sklearn.preprocessing import LabelEncoder
from sklearn.externals.joblib import logger
from sklearn.base import clone, MetaEstimatorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.externals import six
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.metrics import (accuracy_score,
							 explained_variance_score,
							 f1_score,
							 log_loss,
							 mean_absolute_error,
							 mean_squared_error,
							 median_absolute_error,
							 precision_score,
							 r2_score,
							 recall_score)

# >= sklearn 0.18
try:
	from sklearn.model_selection import ParameterSampler, ParameterGrid
	SK18 = True
except ImportError as i:
	from sklearn.grid_search import ParameterSampler, ParameterGrid
	SK18 = False


__all__ = [
	'H2OGridSearchCV',
	'H2ORandomizedSearchCV',
	'H2OActuarialRandomizedSearchCV'
]


SCORERS = {
	'accuracy_score' : accuracy_score,
	'explained_variance_score' : explained_variance_score,
	'f1_score' : f1_score,
	'log_loss' : log_loss,
	'mean_absolute_error' : mean_absolute_error,
	'mean_squared_error' : mean_squared_error,
	'median_absolute_error' : median_absolute_error,
	'precision_score' : precision_score,
	'r2_score' : r2_score,
	'recall_score' : recall_score
}




"""These parameters are ones h2o stores
that we don't necessarily want to clone.
"""
PARM_IGNORE = set([
	'model_id',
	'fold_column',
	'fold_assignment',
	'keep_cross_validation_predictions',
	'offset_column',
	'checkpoint',
	'training_frame',
	'validation_frame',
	'response_column',
	'ignored_columns',
	'max_confusion_matrix_size',
	'score_each_iteration',
	'histogram_type',
	'col_sample_rate',
	'stopping_metric',
	'weights_column',
	'stopping_rounds',
	'col_sample_rate_change_per_level',
	'max_hit_ratio_k',
	'nbins_cats',
	'class_sampling_factors',
	'ignore_const_cols',
	'keep_cross_validation_fold_assignment'
])


def _as_numpy(_1d_h2o_frame):
	"""Takes a single column h2o frame and
	converts it into a numpy array
	"""
	n_cols = _1d_h2o_frame.shape[1]
	if not n_cols == 1:
		raise ValueError('expected single column frame, got %d' % n_cols)
	
	nm = str(_1d_h2o_frame.columns[0])
	return _1d_h2o_frame[nm].as_data_frame(use_pandas=True)[nm].values


def _clone_h2o_obj(estimator, ignore=False, **kwargs):
	# do initial clone
	est = clone(estimator)

	# set kwargs:
	if kwargs:
		for k,v in six.iteritems(kwargs):
			setattr(est, k, v)

	# check on h2o estimator
	if isinstance(estimator, H2OPipeline):
		# the last step from the original estimator
		e = estimator.steps[-1][1]
		if isinstance(e, H2OEstimator):
			last_step = est.steps[-1][1]

			# so it's the last step
			for k,v in six.iteritems(e._parms):
				k = str(k) # h2o likes unicode...

				# likewise, if the v is unicode, let's make it a string.
				if isinstance(v, unicode):
					v = str(v)

				#if (not k in PARM_IGNORE) and (not v is None):
				#	e._parms[k] = v
				last_step._parms[k] = v

		else:
			# otherwise it's an BaseH2OFunctionWrapper
			pass

	return est


def _score(estimator, frame, target_feature, scorer, parms, is_regression, **kwargs):
	# this is a bottleneck:
	y_truth = _as_numpy(frame[target_feature])

	# gen predictions...
	pred = _as_numpy(estimator.predict(frame)['predict'])
	# pred = estimator.predict(frame).as_data_frame(use_pandas=True)['predict'] #old...

	if not is_regression:
		# there's a very real chance that the truth or predictions are enums,
		# as h2o is capable of handling these... we need to explicitly make the
		# predictions and target numeric.
		encoder = LabelEncoder()

		try:
			y_truth = encoder.fit_transform(y_truth)
			pred = encoder.transform(pred)
		except ValueError as v:
			raise ValueError('y contains new labels. '
							 'Seen: %s\n, New:%s' % (
							 	str(encoder.classes_), 
							 	str(set(pred))))

	# pop all of the kwargs into the parms
	for k,v in six.iteritems(kwargs):
		# we could warn, but parms is affected in place, so we won't...
		#if k in parms:
		#	warnings.warn('parm %s already exists in score parameters, but is contained in kwargs' % (k))
		parms[k] = v

	return scorer(y_truth, pred, **parms)


def _fit_and_score(estimator, frame, feature_names, target_feature,
				   scorer, parameters, verbose, scoring_params,
				   train, test, is_regression, act_args,
				   cv_fold, iteration):
	
	if verbose > 1:
		if parameters is None:
			msg = ''
		else:
			msg = 'Target: %s; %s' % (target_feature, ', '.join('%s=%s' % (k,v)
									 for k, v in parameters.items()))
		print("[CV (iter %i, fold %i)] %s %s" % (iteration, cv_fold, msg, (64 - len(msg)) * '.'))

	# set the params for this estimator -- also set feature_names, target_feature
	if not isinstance(estimator, (H2OEstimator, BaseH2OFunctionWrapper)):
		raise TypeError('estimator must be either an H2OEstimator '
						'or a BaseH2OFunctionWrapper but got %s'
						% type(estimator))


	# h2o doesn't currently re-order rows... and sometimes will
	# complain for some reason. We need to sort our train/test idcs
	train = sorted(train)
	test = sorted(test)


	# generate split
	train_frame = frame[train, :]
	test_frame = frame[test, :]

	# if xtra args, it's act...
	if act_args is not None:
		kwargs = {
			'expo' : _as_numpy(test_frame[act_args['expo']]),
			'loss' : _as_numpy(test_frame[act_args['loss']]),
			'prem' : _as_numpy(test_frame[act_args['prem']]) if (act_args['prem'] is not None) else None
		}
	else:
		kwargs = {}

	start_time = time.time()


	#it's probably a pipeline
	is_h2o_est = isinstance(estimator, H2OEstimator)
	if not is_h2o_est: 
		estimator.set_params(**parameters)

		# the name setting should be taken care of pre-clone...
		# setattr(estimator, 'feature_names', feature_names)
		# setattr(estimator, 'target_feature',target_feature)

		# do fit
		estimator.fit(train_frame)
	else: # it's just an H2OEstimator
		# parm_dict = {}
		for k, v in six.iteritems(parameters):
			if '__' in k:
				raise ValueError('only one estimator passed to grid search, '
								 'but multiple named parameters passed: %s' % k)

			# {parm_name : v}
			estimator._parms[k] = v

		# do train
		estimator.train(training_frame=train_frame, x=feature_names, y=target_feature)


	# score model
	test_score = _score(estimator, test_frame, target_feature, scorer, scoring_params, is_regression, **kwargs)

	# h2o is verbose.. if we are too, print a new line:
	if verbose > 1:
		print() # new line

	scoring_time = time.time() - start_time

	if verbose > 2:
		msg += ', score=%f' % test_score
	if verbose > 1:
		end_msg = '%s -%s' % (msg, logger.short_format_time(scoring_time))
		print('[CV (iter %i, fold %i)] %s %s' % (iteration, cv_fold, (64 - len(end_msg)) * '.', end_msg))
		print() # new line
		print() # new line

	return [test_score, len(test), estimator, parameters]


class BaseH2OSearchCV(BaseH2OFunctionWrapper, VizMixin):
	"""Base for all H2O grid searches"""

	__min_version__ = '3.8.2.9'
	__max_version__ = None
	
	@abstractmethod
	def __init__(self, estimator, feature_names,
				 target_feature, scoring=None, 
				 n_jobs=1, scoring_params=None, 
				 cv=5, verbose=0, iid=True,
				 validation_frame=None):

		super(BaseH2OSearchCV, self).__init__(target_feature=target_feature,
											  min_version=self.__min_version__,
											  max_version=self.__max_version__)

		self.estimator = estimator
		self.feature_names = feature_names
		self.scoring = scoring
		self.n_jobs = n_jobs
		self.scoring_params = scoring_params if not scoring_params is None else {}
		self.cv = cv
		self.verbose = verbose
		self.iid = iid
		self.validation_frame = validation_frame

	def _fit(self, X, parameter_iterable):
		"""Actual fitting,  performing the search over parameters."""
		X = _check_is_frame(X) # if it's a frame, will be turned into a matrix

		estimator = self.estimator

		# we need to require scoring...
		scoring = self.scoring
		if scoring is None:
			raise ValueError('require string or callable for scoring')
		elif isinstance(scoring, str):
			if not scoring in SCORERS:
				raise ValueError('Scoring must be one of (%s) or a callable. '
								 'Got %s' % (', '.join(SCORERS.keys()), scoring))
			self.scorer_ = SCORERS[scoring]
		# else we'll let it fail through if it's a bad callable
		else:
			self.scorer_ = scoring

		# validate CV
		cv = check_cv(self.cv)

		# make list of strings
		self.feature_names, self.target_feature = validate_x_y(self.feature_names, self.target_feature)
		nms = {
			'feature_names' : self.feature_names,
			'target_feature': self.target_feature
		}

		# do first clone, remember to set the names...
		base_estimator = _clone_h2o_obj(self.estimator, **nms)
		self.is_regression_ = (not X[self.target_feature].isfactor()[0])


		# check estimator... needs to have the 'predict' attr
		#tmp_est = base_estimator
		#if isinstance(tmp_est, H2OPipeline):
		#	tmp_est = tmp_est._final_estimator
		#if not hasattr(tmp_est, 'predict')
		#	raise ValueError('base_estimator must implement "predict"')


		# the addition of the actuarial search necessitates some hackiness.
		# if we have the attr 'extra_args_' then we know it's an actuarial search
		xtra = self.extra_args_ if hasattr(self, 'extra_args_') else None

		# do fits, scores
		out = [
			_fit_and_score(estimator=_clone_h2o_obj(base_estimator),
						   frame=X, feature_names=self.feature_names,
						   target_feature=self.target_feature,
						   scorer=self.scorer_, parameters=params,
						   verbose=self.verbose, scoring_params=self.scoring_params,
						   train=train, test=test, is_regression=self.is_regression_,
						   act_args=xtra, cv_fold=cv_fold, iteration=iteration)
			for iteration,params in enumerate(parameter_iterable)
			for cv_fold,(train,test) in enumerate(cv.split(X, self.target_feature))
		]

		# Out is a list of quad: score, n_test_samples, estimator, parameters
		n_fits = len(out)
		n_folds = cv.get_n_splits()

		# if a validation frame was passed, user might want to see how it scores
		# on each model, so we'll do that here...
		if self.validation_frame is not None:
			score_validation = True
			self.validation_scores = []

			if xtra is not None:
				self.val_score_report_ = ActStatisticalReport(
					n_folds=n_folds, 
					n_iter= n_fits//n_folds,
					iid=self.iid)

				# set scoring function
				val_scorer = self.val_score_report_._score

				kwargs = {
					'expo' : _as_numpy(self.validation_frame[xtra['expo']]),
					'loss' : _as_numpy(self.validation_frame[xtra['loss']]),
					'prem' : _as_numpy(self.validation_frame[xtra['prem']]) if (xtra['prem'] is not None) else None
				}
			else:
				kwargs = {}
				val_scorer = self.scorer_
		else:
			score_validation = False


		# do scoring
		scores = list()
		grid_scores = list()
		for grid_start in range(0, n_fits, n_folds):
			n_test_samples = 0
			score = 0
			all_scores = []

			# iterate over OUT
			for this_score, this_n_test_samples, this_estimator, parameters in \
					out[grid_start:grid_start + n_folds]:
				all_scores.append(this_score)

				if self.iid:
					this_score *= this_n_test_samples
					n_test_samples += this_n_test_samples
				score += this_score

				# score validation set if necessary
				if score_validation:
					val_score = _score(this_estimator, self.validation_frame, 
						self.target_feature, val_scorer, self.scoring_params, 
						self.is_regression_, **kwargs)

					# if it's actuarial scorer, handles the iid condition internally...
					self.validation_scores.append(val_score)

			if self.iid:
				score /= float(n_test_samples)
			else:
				score /= float(n_folds)

			scores.append((score, parameters))
			# TODO: shall we also store the test_fold_sizes?
			grid_scores.append(_CVScoreTuple(
				parameters,
				score,
				np.array(all_scores)))
		# Store the computed scores
		self.grid_scores_ = grid_scores

		# Find the best parameters by comparing on the mean validation score:
		# note that `sorted` is deterministic in the way it breaks ties
		best = sorted(grid_scores, key=lambda x: x.mean_validation_score,
					  reverse=True)[0]
		self.best_params_ = best.parameters
		self.best_score_ = best.mean_validation_score

		# fit the best estimator using the entire dataset
		# clone first to work around broken estimators
		best_estimator = _clone_h2o_obj(base_estimator, **nms)


		# if verbose alert user we're at the end...
		if self.verbose > 1:
			msg = 'Target: %s; %s' % (self.target_feature, ', '.join('%s=%s' % (k,v)
									 for k, v in six.iteritems(best.parameters) ))
			print("[BEST] %s %s" % (msg, (64 - len(msg)) * '.'))


		# set params -- remember h2o gets funky with this...
		if isinstance(best_estimator, H2OEstimator):
			for k,v in six.iteritems(best.parameters):
				best_estimator._parms[k] = v
			best_estimator.train(training_frame=X, x=self.feature_names, y=self.target_feature)
		else:
			best_estimator.set_params(**best.parameters)
			best_estimator.fit(X)

		self.best_estimator_ = best_estimator

		return self

	def score(self, frame):
		check_is_fitted(self, 'best_estimator_')
		return _score(self.best_estimator_, frame, self.target_feature, self.scorer_, self.scoring_params, self.is_regression_)

	def predict(self, frame):
		check_is_fitted(self, 'best_estimator_')

		if not hasattr(self, 'predict'):
			return NotImplemented

		frame = _check_is_frame(frame)
		return self.best_estimator_.predict(frame)

	@overrides(VizMixin)
	def plot(self, timestep, metric):
		check_is_fitted(self, 'best_estimator_')

		# then it's a pipeline:
		if hasattr(self.best_estimator_, 'plot'):
			self.best_estimator_.plot(timestep=timestep, metric=metric)
		else:
			# should be an H2OEstimator
			self.best_estimator_._plot(timestep=timestep, metric=metric)
	


class H2OGridSearchCV(BaseH2OSearchCV):

	def __init__(self, estimator, param_grid, 
				 feature_names, target_feature, 
				 scoring=None, n_jobs=1, 
				 scoring_params=None, cv=5, 
				 verbose=0, iid=True,
				 validation_frame=None):

		super(H2OGridSearchCV, self).__init__(
				estimator=estimator,
				feature_names=feature_names,
				target_feature=target_feature,
				scoring=scoring, n_jobs=n_jobs,
				scoring_params=scoring_params,
				cv=cv, verbose=verbose,
				iid=iid, validation_frame=validation_frame
			)

		self.param_grid = param_grid
		_check_param_grid(param_grid)

	def fit(self, frame):
		return self._fit(frame, ParameterGrid(self.param_grid))


class H2ORandomizedSearchCV(BaseH2OSearchCV):

	def __init__(self, estimator, param_grid,
				 feature_names, target_feature, 
				 n_iter=10, random_state=None,
				 scoring=None, n_jobs=1, 
				 scoring_params=None, cv=5, 
				 verbose=0, iid=True,
				 validation_frame=None):

		super(H2ORandomizedSearchCV, self).__init__(
				estimator=estimator,
				feature_names=feature_names,
				target_feature=target_feature,
				scoring=scoring, n_jobs=n_jobs,
				scoring_params=scoring_params,
				cv=cv, verbose=verbose,
				iid=iid, validation_frame=validation_frame
			)

		self.param_grid = param_grid
		self.n_iter = n_iter
		self.random_state = random_state

	def fit(self, frame):
		sampled_params = ParameterSampler(self.param_grid,
										  self.n_iter,
										  random_state=self.random_state)

		return self._fit(frame, sampled_params)




def _val_exp_loss_prem(x,y,z):
	if not all([isinstance(i, (str, unicode)) for i in (x,y)]):
		raise TypeError('exposure and loss must be strings or unicode')

	if not z is None:
		if not isinstance(z, (str, unicode)):
			raise TypeError('premium must be None or string or unicode')

	return str(x), str(y), str(z) if z is not None else z


class H2OActuarialRandomizedSearchCV(H2ORandomizedSearchCV):
	"""A grid search that scores based on actuarial metrics
	(See skutil.metrics.ActStatisticalReport)
	"""

	def __init__(self, estimator, param_grid,
				 feature_names, target_feature, 
				 exposure_feature, loss_feature,
				 premium_feature=None, n_iter=10, 
				 random_state=None, scoring='lift', 
				 n_jobs=1, scoring_params=None, cv=5, 
				 verbose=0, iid=True, #n_groups=10,
				 validation_frame=None):

		super(H2OActuarialRandomizedSearchCV, self).__init__(
				estimator=estimator,
				param_grid=param_grid,
				feature_names=feature_names,
				target_feature=target_feature,
				n_iter=n_iter, random_state=random_state,
				scoring=scoring, n_jobs=n_jobs,
				scoring_params=scoring_params,
				cv=cv, verbose=verbose,
				iid=iid, validation_frame=validation_frame
			)

		#self.n_groups = 10
		self.exposure_feature = exposure_feature
		self.loss_feature = loss_feature
		self.premium_feature = premium_feature

		# our score method will ALWAYS be the same
		self.score_report_ = ActStatisticalReport(
			score_by=scoring, 
			n_folds=check_cv(cv).get_n_splits(), 
			n_iter=n_iter,
			iid=iid)

		self.scoring = self.score_report_._score ## callable -- resets scoring


	def fit(self, frame):
		sampled_params = ParameterSampler(self.param_grid,
										  self.n_iter,
										  random_state=self.random_state)

		exp, loss, prem = _val_exp_loss_prem(self.exposure_feature, self.loss_feature, self.premium_feature)
		self.extra_args_ = {
			'expo':exp,
			'loss':loss,
			'prem':prem
		}

		# do fit
		return self._fit(frame, sampled_params)

	def report_scores(self):
		"""Get the actuarial report"""
		check_is_fitted(self, 'best_estimator_')
		report_res = self.score_report_.as_data_frame()
		n_obs, _ = report_res.shape

		# Need to cbind the parameters... we don't care about ["score", "std"]
		rdf = report_grid_score_detail(self, charts=False).drop(["score", "std"], axis=1)
		assert n_obs == rdf.shape[0], 'Internal error: %d!=%d'%(n_obs, rdf.shape[0])

		# change the names in the dataframe...
		report_res.columns = ['train_%s'%x for x in report_res.columns.values]

		# cbind...
		rdf = pd.concat([rdf, report_res], axis=1)

		#if we scored on the validation set, also need to get the val score struct
		if hasattr(self, 'val_score_report_'):
			val_res_df = self.val_score_report_.as_data_frame()
			assert n_obs == val_res_df.shape[0], 'Internal error: %d!=%d'%(n_obs, val_res_df.shape[0])
			val_res_df.columns = ['val_%s'%x for x in val_res_df.columns.values]

			# cbind
			rdf = pd.concat([rdf, val_res_df], axis=1)

		rdf.index = ['Iter_%i' %i for i in range(self.n_iter)]
		return rdf


