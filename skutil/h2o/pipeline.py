from __future__ import print_function, division, absolute_import

import h2o

try:
    from h2o import H2OEstimator
except ImportError as e:
    from h2o.estimators.estimator_base import H2OEstimator

from .base import (BaseH2OTransformer, BaseH2OFunctionWrapper,
                   validate_x_y, validate_x, VizMixin)
from skutil.base import overrides

from sklearn.utils import tosequence
from sklearn.externals import six
from sklearn.base import BaseEstimator
from ..utils.metaestimators import if_delegate_has_method, if_delegate_isinstance
from ..utils import flatten_all

try:
    import cPickle as pickle
except ImportError as ie:
    import pickle

__all__ = [
    'H2OPipeline'
]


def _union_exclusions(a, b):
    """Take the exlusion features from two preprocessors
    and create a union set or None.
    """
    if (not a) and (not b):
        return None
    if not a:
        return b
    if not b:
        return a

    return flatten_all([a, b])


class H2OPipeline(BaseH2OFunctionWrapper, VizMixin):
    """Create a sklearn-esque pipeline of H2O steps finished with an 
    optional `H2OEstimator`. Note that as of version 0.1.0, the behavior
    of the H2OPipeline has slightly changed, given the inclusion of the
    ``exclude_from_ppc`` and ``exclude_from_fit`` parameters.

    The pipeline, at the core, is comprised of a list of length-two tuples
    in the form of `('name', SomeH2OTransformer())`, punctuated with an
    optional `H2OEstimator` as the final step. The pipeline will procedurally 
    fit each stage, transforming the training data prior to fitting the next stage.
    When predicting or transforming new (test) data, each stage calls either
    `transform` or `predict` at the respective step.

    **On the topic of exclusions and `feature_names`:**

    Prior to version 0.1.0, `H2OTransformer`s did not take the keyword `exclude_features`.
    Its addition necessitated two new keywords in the `H2OPipeline`, and a slight change
    in behavior of ``feature_names``:

        * ``exclude_from_ppc`` - If set in the `H2OPipeline` constructor, these features
                                 will be universally omitted from every preprocessing stage.
                                 Since `exclude_features` can be set individually in each
                                 separate transformer, in the case that `exclude_features` has
                                 been explicitly set, the exclusions in that respective stage
                                 will include the union of ``exclude_from_ppc`` and 
                                 `exclude_features`.

        * ``exclude_from_fit`` - If set in the `H2OPipeline` constructor, these features
                                 will be omitted from the ``training_cols_`` fit attribute,
                                 which are the columns passed to the final stage in the pipeline.

        * ``feature_names`` - The former behavior of the `H2OPipeline` only used ``feature_names``
                              in the fit of the first transformer, passing the remaining columns to
                              the next transformer as the ``feature_names`` parameter. The new
                              behavior is more discriminating in the case of explicitly-set attributes.
                              In the case where a transformer's ``feature_names`` parameter has been
                              explicitly set, *only those names* will be used in the fit. This is useful
                              in cases where someone may only want to, for instance, drop one of two
                              multicollinear features using the `H2OMulticollinearityFilterer` rather than
                              fitting against the entire dataset. It also adheres to the now expected
                              behavior of the exclusion parameters.

    Parameters
    ----------

    steps : list
        A list of named tuples wherein element 1 of each tuple is
        an instance of a BaseH2OTransformer or an H2OEstimator.

    feature_names : iterable (default=None)
        The names of features on which to fit the first transformer 
        in the pipeline. The next transformer will be fit with
        ``feature_names`` as the result-set columns from the previous
        transformer, minus any exclusions or target features.

    target_feature : str (default=None)
        The name of the target feature

    exclude_from_ppc : iterable, optional (default=None)
        Any names to be excluded from any preprocessor fits.
        Since the ``exclude_features`` can be set in respective
        steps in each preprocessor, these features will be considered
        as global exclusions and will be appended to any individually
        set exclusion features.

    exclude_from_fit : iterable, optional (default=None)
        Any names to be excluded from the final model fit
    """

    _min_version = '3.8.2.9'
    _max_version = None

    def __init__(self, steps, feature_names=None, target_feature=None,
                 exclude_from_ppc=None, exclude_from_fit=None):
        super(H2OPipeline, self).__init__(target_feature=target_feature,
                                          min_version=self._min_version,
                                          max_version=self._max_version)

        # assign to attribute
        self.feature_names = feature_names

        # if we have any to exclude...
        self.exclude_from_ppc = validate_x(exclude_from_ppc)
        self.exclude_from_fit = validate_x(exclude_from_fit)

        names, estimators = zip(*steps)
        if len(dict(steps)) != len(steps):
            raise ValueError("Provided step names are not unique: %s"
                             % (names,))

        # shallow copy of steps
        self.steps = tosequence(steps)
        transforms = estimators[:-1]
        estimator = estimators[-1]

        for t in transforms:
            if not isinstance(t, BaseH2OTransformer):
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
        frame_t = frame

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

            # if the feature names are explicitly set in this estimator,
            # we won't set them to the `next_feature_names`, however,
            # if the names are *not* explicitly set, we will set the 
            # estimator's `feature_names` to the `next_feature_names`
            # variable set...
            if transform.feature_names is None:
                transform.feature_names = next_feature_names

            # now set the exclude_features if they exist
            transform.exclude_features = _union_exclusions(self.exclude_from_ppc,
                                                           transform.exclude_features)

            if hasattr(transform, "fit_transform"):
                frame_t = transform.fit_transform(frame_t)
            else:
                frame_t = transform.fit(frame_t).transform(frame_t)

            # now reset the next_feature_names to be the remaining names...
            next_feature_names = [str(nm) for nm in frame_t.columns if not (nm == self.target_feature)]
            if not next_feature_names or len(next_feature_names) < 1:
                raise ValueError('no columns retained after fit!')

        # this will have y re-combined in the matrix
        return frame_t, next_feature_names


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

        Returns
        -------

        self
        """
        self._reset()  # reset if needed

        # reset to the cleaned ones, if necessary...
        self.feature_names, self.target_feature = validate_x_y(frame, self.feature_names, self.target_feature)

        # get the fit
        Xt, training_cols_ = self._pre_transform(frame)

        # if there are any exclude names, remove them from training_cols_, then assign to self
        if self.exclude_from_fit:
            training_cols_ = [i for i in training_cols_ if i not in self.exclude_from_fit]
        self.training_cols_ = training_cols_

        # if the last step is not an h2o estimator, we need to do things differently...
        if isinstance(self.steps[-1][1], H2OEstimator):
            self.steps[-1][1].train(training_frame=Xt, x=self.training_cols_, y=self.target_feature)
        else:
            _est = self.steps[-1][1]

            # set the instance members
            _est.feature_names = self.training_cols_
            _est.target_feature = self.target_feature

            # do the fit
            _est.fit(Xt)

        return self


    @overrides(VizMixin)
    @if_delegate_has_method(delegate='_final_estimator', method='_plot')
    def plot(self, timestep, metric):
        """If the ``_final_estimator`` is an ``H2OEstimator``,
        this method is injected at runtime. This method generates
        a plot of the estimator's scoring over a provided timestep.

        Parameters
        ----------

        timestep : string
            Might be one of ('epochs', 'number_of_trees') and
            provides the x-axis of the plot.

        metric : string
            Might be one of ('MSE', 'logloss') and provides
            the y-axis of the plot.
        """
        self._final_estimator._plot(timestep=timestep, metric=metric)


    @overrides(BaseEstimator)
    def set_params(self, **params):
        """Set the parameters for this pipeline. Will revalidate the
        steps in the estimator prior to setting the parameters. Parameters
        is a **kwargs-style dictionary whose keys should be prefixed by the
        name of the step targeted and a double underscore:

            >>> pipeline = H2OPipeline([
            ...     ('mcf', H2OMulticollinearityFilterer()),
            ...     ('rf',  H2ORandomForestEstimator())
            ... ])
            >>> pipe.set_params(**{
            ...     'rf__ntrees':     100, 
            ...     'mcf__threshold': 0.75
            ... })

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
            if key not in parm_dict:
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
            for pth in [model.model_loc_, 'hdfs://%s' % model.model_loc_]:
                try:
                    the_h2o_model = h2o.load_model(pth)
                except Exception as e:
                    if ex is None:
                        ex = e
                    else:
                        # only throws if fails twice
                        raise ex

            model.steps[-1] = (model.est_name_, the_h2o_model)

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

            # let's keep a pointer to the last step, so
            # after the pickling we can reassign it to retain state
            last_step_ = self.steps[-1]
            self.steps[-1] = None

        # now save the rest of things...
        with open(loc, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

        # after pickle, we can add the last_step_ back in.
        # this allows re-use/re-predict after saving to disk
        if ends_in_h2o:
            self.steps[-1] = last_step_


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


    @if_delegate_has_method(delegate='_final_estimator', method='predict')
    def fit_predict(self, frame):
        """Fit all the transforms one after the other and transform the
        data, then fit the transformed data using the final estimator. Finally,
        either predict on the final step.
        
        Parameters
        ----------

        frame : h2o Frame
            Training data. Must fulfill input requirements of first step of the
            pipeline.
        """
        return self.fit(frame).predict(frame)


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


    @if_delegate_has_method(delegate='_final_estimator', method='transform')
    def fit_transform(self, frame):
        """Fit all the transforms one after the other and transform the
        data, then fit the transformed data using the final estimator. Finally,
        either transform on the final step.
        
        Parameters
        ----------

        frame : h2o Frame
            Training data. Must fulfill input requirements of first step of the
            pipeline.
        """
        return self.fit(frame).transform(frame)


    @if_delegate_has_method(delegate='_final_estimator')
    def varimp(self, use_pandas=True):
        """Get the variable importance, if the final
        estimator implements such a function.

        Parameters
        ----------

        use_pandas : bool, optional (default=True)
            Whether to return a pandas dataframe
        """
        return self._final_estimator.varimp(use_pandas=use_pandas)


    @if_delegate_isinstance(delegate='_final_estimator', instance_type=H2OEstimator)
    def download_pojo(self, path="", get_jar=True):
        """This method is injected at runtime if the ``_final_estimator``
        is an instance of an ``H2OEstimator``. This method downloads the POJO
        from a fit estimator.

        Parameters
        ----------

        path : string, optional (default="")
            Path to folder in which to save the POJO.

        get_jar : bool, optional (default=True)
            Whether to get the jar from the POJO.

        Returns
        -------

        None or string
            Returns None if ``path`` is "" else, the filepath
            where the POJO was saved.
        """
        return h2o.download_pojo(self._final_estimator, path=path, get_jar=get_jar)
