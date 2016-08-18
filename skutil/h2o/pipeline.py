from __future__ import print_function, division, absolute_import
import h2o
from h2o.frame import H2OFrame
try:
	from h2o import H2OEstimator
except ImportError as e:
	from h2o.estimators.estimator_base import H2OEstimator

import warnings
import numpy as np

from sklearn.externals import six
from .base import BaseH2OTransformer, BaseH2OFunctionWrapper, validate_x_y
from ..base import overrides

from sklearn.utils import tosequence
from sklearn.externals import six
from sklearn.base import BaseEstimator
from sklearn.utils.metaestimators import if_delegate_has_method


__all__ = [
	'H2OPipeline'
]


class H2OPipeline(BaseH2OFunctionWrapper):
	"""Create a sklearn-esque pipeline of H2O steps finished with an H2OEstimator.

	Parameters
	----------
	steps : list
		A list of named tuples wherein element 1 of each tuple is
		an instance of a BaseH2OTransformer or an H2OEstimator.

	feature_names : iterable (default None)
		The names of features on which to fit the pipeline

	target_feature : str (default None)
		The name of the target feature
	"""

	__min_version__ = '3.8.3'
	__max_version__ = None
	
	def __init__(self, steps, feature_names=None, target_feature=None):
		super(H2OPipeline, self).__init__(target_feature=target_feature,
										  min_version=self.__min_version__,
										  max_version=self.__max_version__)

		# assign to attribute
		self.feature_names = feature_names
		
		names, estimators = zip(*steps)
		if len(dict(steps)) != len(steps):
			raise ValueError("Provided step names are not unique: %s"
							 % (names,))

		# shallow copy of steps
		self.steps = tosequence(steps)
		transforms = estimators[:-1]
		estimator = estimators[-1]

		for t in transforms:
			if (not isinstance(t, BaseH2OTransformer)):
				raise TypeError("All intermediate steps of the chain should "
								"be instances of BaseH2OTransformer"
								" '%s' (type %s) isn't)" % (t, type(t)))

		if not isinstance(estimator, H2OEstimator):
			raise TypeError("Last step of chain should implement train "
							"'%s' (type %s) doesn't)"
							% (estimator, type(estimator)))
		
	"""@property
	def _estimator_type(self):
		return self.steps[-1][1]._estimator_type
	"""
		
	@property
	def named_steps(self):
		return dict(self.steps)

	@property
	def _final_estimator(self):
		return self.steps[-1][1]
	
	def _pre_transform(self, frame=None, **fit_params):
		fit_params_steps = dict((step, {}) for step, _ in self.steps)
		for pname, pval in six.iteritems(fit_params):
			step, param = pname.split('__', 1)
			fit_params_steps[step][param] = pval
			
		frameT = frame
		for name, transform in self.steps[:-1]:
			# for each transformer in the steps sequence, we need
			# to ensure the target_feature has been set... we do
			# this in the fit method and not the init because we've
			# now validated the y/target_feature. Also this way if
			# target_feature is ever changed, this will be updated...
			transform.target_feature = self.target_feature
			
			if hasattr(transform, "fit_transform"):
				frameT = transform.fit_transform(frameT, **fit_params_steps[name])
			else:
				frameT = transform.fit(frameT, **fit_params_steps[name]).transform(frameT)
					
		# this will have y and exclude re-combined in the matrix
		return frameT, fit_params_steps[self.steps[-1][0]]
		
	def _reset(self):
		"""Each individual step should handle its own
		state resets, but this method will reset any Pipeline
		state variables.
		"""
		if hasattr(self, 'training_cols_'):
			del self.training_cols_
		
	def fit(self, frame=None):
		"""Fit all the transforms one after the other and transform the
		data, then fit the transformed data using the final estimator.
		
		Parameters
		----------
		frame : h2o Frame
			Training data. Must fulfill input requirements of first step of the
			pipeline.
		"""
		self._reset() # reset if needed
		x, y = validate_x_y(self.feature_names, self.target_feature)
		self.target_feature = y # reset to the cleaned one, if necessary...
		
		# First, if there are any columns in the frame that are not in x, y drop them
		xy = [p for p in x]
		if y is not None:
			xy.append(y)
		
		# retain only XY
		frame = frame[xy]
		
		# get the fit
		Xt, fit_params = self._pre_transform(frame, **{})
		
		# get the names of the matrix that are not Y
		training_cols = Xt.columns
		self.training_cols_ = [n for n in training_cols if not (n==y)]
			
		if not training_cols:
			raise ValueError('no columns retained after fit!')
		
		self.steps[-1][1].train(training_frame=Xt, x=self.training_cols_, y=y, **fit_params)
		return self


	@overrides(BaseEstimator)
	def set_params(self, **params):
		"""Set the parameters for this pipeline. Will revalidate the
		steps in the estimator prior to setting the parameters.

		Returns
		-------
		self
		"""
		if not params:
			return self

		# create dict of {step_name : {param_name : val}}
		parm_dict = {}
		for k, v in six.iteritems(params):
			key, val = k.split('__')
			if not key in parm_dict:
				parm_dict[key] = {}

			# step_name : {parm_name : v}
			parm_dict[key][val] = v

		# go through steps, now (first the transforms).
		for name, transform in self.steps[:-1]:
			step_params = parm_dict.get(name, None)
			if step_params is None:
				continue

			for parm, value in six.iteritems(step_params):
				setattr(transform, parm, value)

		# finally, set the h2o estimator params (if needed). 
		est_name, last_step = self.steps[-1]
		if est_name in parm_dict:
			for parm, value in six.iteritems(parm_dict[est_name]):
				try:
					last_step._parms[parm] = value
				except Exception as e:
					raise ValueError('Invalid parameter for %s: %s'
									 % (parm, last_step.__name__))

		return self

		
	@if_delegate_has_method(delegate='_final_estimator')
	def predict(self, frame):
		"""Applies transforms to the data, and the predict method of the
		final estimator. Valid only if the final estimator implements
		predict.
		
		Parameters
		----------
		frame : an h2o Frame
			Data to predict on. Must fulfill input requirements of first step
			of the pipeline.
		"""
		Xt = frame
		for name, transform in self.steps[:-1]:
			Xt = transform.transform(Xt)
			
		return self.steps[-1][-1].predict(Xt)