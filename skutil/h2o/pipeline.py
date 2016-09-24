from __future__ import print_function, division, absolute_import
import h2o
from h2o.frame import H2OFrame
try:
    from h2o import H2OEstimator
except ImportError as e:
    from h2o.estimators.estimator_base import H2OEstimator

import os
import warnings
import numpy as np

from sklearn.externals import six
from .base import (BaseH2OTransformer, BaseH2OFunctionWrapper, 
                   validate_x_y, VizMixin, _frame_from_x_y)
from ..base import overrides

from sklearn.utils import tosequence
from sklearn.externals import six
from sklearn.base import BaseEstimator
from sklearn.utils.metaestimators import if_delegate_has_method

try:
    import cPickle as pickle
except ImportError as ie:
    import pickle


__all__ = [
    'H2OPipeline'
]


class H2OPipeline(BaseH2OFunctionWrapper, VizMixin):
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

    _min_version = '3.8.2.9'
    _max_version = None
    
    def __init__(self, steps, feature_names=None, target_feature=None):
        super(H2OPipeline, self).__init__(target_feature=target_feature,
                                          min_version=self._min_version,
                                          max_version=self._max_version)

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

        if not isinstance(estimator, (H2OEstimator, BaseH2OTransformer)):
            raise TypeError("Last step of chain should be an H2OEstimator or BaseH2OTransformer, "
                            "not of type %s" % type(estimator))
        
    @property
    def named_steps(self):
        return dict(self.steps)

    @property
    def _final_estimator(self):
        return self.steps[-1][1]
    
    def _pre_transform(self, frame=None):
        frameT = frame

        # we have to set the feature names at each stage to be
        # the remaining feature names (not the target though)
        next_feature_names = self.feature_names
        for name, transform in self.steps[:-1]:
            # for each transformer in the steps sequence, we need
            # to ensure the target_feature has been set... we do
            # this in the fit method and not the init because we've
            # now validated the y/target_feature. Also this way if
            # target_feature is ever changed, this will be updated...
            transform.target_feature = self.target_feature
            transform.feature_names = next_feature_names
            
            if hasattr(transform, "fit_transform"):
                frameT = transform.fit_transform(frameT)
            else:
                frameT = transform.fit(frameT).transform(frameT)

            # now reset the next_feature_names to be the remaining names...
            next_feature_names = [str(nm) for nm in frameT.columns if not (nm==self.target_feature)]
            if not next_feature_names or len(next_feature_names) < 1:
                raise ValueError('no columns retained after fit!')
                    
        # this will have y re-combined in the matrix
        return frameT, next_feature_names
        
    def _reset(self):
        """Each individual step should handle its own
        state resets, but this method will reset any Pipeline
        state variables.
        """
        if hasattr(self, 'training_cols_'):
            del self.training_cols_
        
    def fit(self, frame):
        """Fit all the transforms one after the other and transform the
        data, then fit the transformed data using the final estimator.
        
        Parameters
        ----------
        frame : h2o Frame
            Training data. Must fulfill input requirements of first step of the
            pipeline.
        """
        self._reset() # reset if needed
        x, y = validate_x_y(frame, self.feature_names, self.target_feature)
        self.target_feature = y # reset to the cleaned one, if necessary...
        
        # First, if there are any columns in the frame that are not in x, y drop them
        # we need to reappend y to make sure it doesn't get dropped out by the
        # frame_from_x_y method
        xy = [p for p in x]
        if y is not None:
            xy.append(y)
        
        # retain only XY
        frame = frame[xy]
        
        # get the fit
        Xt, self.training_cols_ = self._pre_transform(frame)
        
        # if the last step is not an h2o estimator, we need to do things differently...
        if isinstance(self.steps[-1][1], H2OEstimator):
            self.steps[-1][1].train(training_frame=Xt, x=self.training_cols_, y=y)
        else:
            _est = self.steps[-1][1]

            # set the instance members
            _est.feature_names = self.training_cols_
            _est.target_feature = y

            # do the fit
            _est.fit(Xt)

        return self


    @overrides(VizMixin)
    def plot(self, timestep, metric):
        # should be confident final step is an H2OEstimator
        self._final_estimator._plot(timestep=timestep, metric=metric)


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
            if isinstance(last_step, H2OEstimator):
                for parm, value in six.iteritems(parm_dict[est_name]):
                    try:
                        last_step._parms[parm] = value
                    except Exception as e:
                        raise ValueError('Invalid parameter for %s: %s'
                                         % (parm, last_step.__name__))

            # if it's not an H2OEstimator, but a BaseH2OTransformer,
            # we gotta do things a bit differently...
            else:
                # we're already confident it's in the parm_dict
                step_params = parm_dict[est_name]
                for parm, value in six.iteritems(step_params):
                    setattr(last_step, parm, value)


        return self


    @staticmethod
    def load(location):
        with open(location) as f:
            model = pickle.load(f)

        if not isinstance(model, H2OPipeline):
            raise TypeError('expected H2OPipeline, got %s' % type(model))

        # if the pipe didn't end in an h2o estimator, we don't need to
        # do the following IO segment...
        ends_in_h2o = hasattr(model, 'model_loc_')
        if ends_in_h2o:
            # read the model portion, delete the model path
            ex = None
            for pth in [model.model_loc_, 'hdfs://%s'%model.model_loc_]:
                try:
                    the_h2o_model = h2o.load_model(pth)
                except Exception as e:
                    if ex is None:
                        ex = e
                    else:
                        # only throws if fails twice
                        raise ex        

            model.steps[-1] = (model.est_name_, the_h2o_model)
            del model.model_loc_
            del model.est_name_

        return model


    def _save_internal(self, **kwargs):
        loc = kwargs.pop('location')
        model_loc = kwargs.pop('model_location')

        # first, save the estimator... if it's there
        ends_in_h2o = isinstance(self._final_estimator, H2OEstimator)
        if ends_in_h2o:
            force = kwargs.pop('force', False)
            self.model_loc_ = h2o.save_model(model=self._final_estimator, path=model_loc, force=force)

            # set the _final_estimator to None just for pickling
            self.est_name_ = self.steps[-1][0]
            self.steps[-1] = None

        # now save the rest of things...
        with open(loc, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)


        
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


    @if_delegate_has_method(delegate='_final_estimator')
    def transform(self, frame):
        """Applies transforms to the data. Valid only if the 
        final estimator implements predict.
        
        Parameters
        ----------
        frame : an h2o Frame
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.
        """
        Xt = frame
        for name, transform in self.steps:
            Xt = transform.transform(Xt)
            
        return Xt
