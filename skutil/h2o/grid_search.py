"""Sklearn-esque grid searches for H2O frames"""
# Author : Taylor Smith, originally adapted from sklearn for use with H2O datastructures
# License: BSD

from __future__ import division, print_function, absolute_import

import time
from abc import abstractmethod

import h2o
import numpy as np
import pandas as pd
from h2o.frame import H2OFrame

try:
    from h2o import H2OEstimator
except ImportError as e:
    from h2o.estimators.estimator_base import H2OEstimator

from .pipeline import H2OPipeline
from .frame import _check_is_1d_frame
from .base import check_frame, BaseH2OFunctionWrapper, validate_x_y, VizMixin
from skutil.base import overrides
from ..utils import report_grid_score_detail
from ..utils.metaestimators import if_delegate_has_method, if_delegate_isinstance
from skutil.grid_search import _CVScoreTuple, _check_param_grid
from ..metrics import GainsStatisticalReport
from .split import *
from .metrics import (h2o_accuracy_score,
                      h2o_f1_score,
                      h2o_mean_absolute_error,
                      h2o_mean_squared_error,
                      h2o_median_absolute_error,
                      h2o_precision_score,
                      h2o_recall_score,
                      h2o_r2_score,
                      make_h2o_scorer)

from sklearn.externals.joblib import logger
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted
from sklearn.externals import six
from h2o.estimators import (H2ODeepLearningEstimator,
                            H2OGradientBoostingEstimator,
                            H2OGeneralizedLinearEstimator,
                            H2ONaiveBayesEstimator,
                            H2ORandomForestEstimator)

try:
    import cPickle as pickle
except ImportError as ie:
    import pickle

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
    'H2OGainsRandomizedSearchCV'
]

SCORERS = {
    'accuracy_score': h2o_accuracy_score,
    'f1_score': h2o_f1_score,
    # 'log_loss'             :,
    'mean_absolute_error': h2o_mean_absolute_error,
    'mean_squared_error': h2o_mean_squared_error,
    'median_absolute_error': h2o_median_absolute_error,
    'precision_score': h2o_precision_score,
    'r2_score': h2o_r2_score,
    'recall_score': h2o_recall_score
}

"""These parameters are ones h2o stores
that we don't necessarily want to clone.
"""
PARM_IGNORE = {
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
}


def _as_numpy(_1d_h2o_frame):
    """Takes a single column h2o frame and
    converts it into a numpy array
    """
    f = _check_is_1d_frame(_1d_h2o_frame)

    nm = str(f.columns[0])
    return f[nm].as_data_frame(use_pandas=True)[nm].values


def _kv_str(k, v):
    k = str(k)  # h2o likes unicode...
    # likewise, if the v is unicode, let's make it a string.
    v = v if not isinstance(v, unicode) else str(v)
    return k, v


def _clone_h2o_obj(estimator, ignore=False, **kwargs):
    # do initial clone
    est = clone(estimator)

    # set kwargs:
    if kwargs:
        for k, v in six.iteritems(kwargs):
            setattr(est, k, v)

    # check on h2o estimator
    if isinstance(estimator, H2OPipeline):
        # the last step from the original estimator
        e = estimator.steps[-1][1]
        if isinstance(e, H2OEstimator):
            last_step = est.steps[-1][1]

            # so it's the last step
            for k, v in six.iteritems(e._parms):
                k, v = _kv_str(k, v)

                # if (not k in PARM_IGNORE) and (not v is None):
                #   e._parms[k] = v
                last_step._parms[k] = v

                # otherwise it's an BaseH2OFunctionWrapper
    return est


def _new_base_estimator(est, clonable_kwargs):
    """When the grid searches are pickled, the estimator
    has to be dropped out. When we load it back in, we have
    to reinstate a new one, since the fit is predicated on
    being able to clone a base estimator, we've got to have
    an estimator to clone and fit.

    Parameters
    ----------

    est : str
        The type of model to build

    Returns
    -------

    estimator : H2OEstimator
        The cloned base estimator
    """
    est_map = {
        'dl': H2ODeepLearningEstimator,
        'gbm': H2OGradientBoostingEstimator,
        'glm': H2OGeneralizedLinearEstimator,
        # 'glrm': H2OGeneralizedLowRankEstimator,
        # 'km'  : H2OKMeansEstimator,
        'nb': H2ONaiveBayesEstimator,
        'rf': H2ORandomForestEstimator
    }

    estimator = est_map[est]()  # initialize the new ones
    for k, v in six.iteritems(clonable_kwargs):
        k, v = _kv_str(k, v)
        estimator._parms[k] = v

    return estimator


def _get_estimator_string(estimator):
    """Looks up the estimator string in the reverse
    dictionary. This way we can regenerate the base 
    estimator. This is kind of a hack...

    Parameters
    ----------

    estimator : H2OEstimator
        The estimator
    """
    if isinstance(estimator, H2ODeepLearningEstimator):
        return 'dl'
    elif isinstance(estimator, H2OGradientBoostingEstimator):
        return 'gbm'
    elif isinstance(estimator, H2OGeneralizedLinearEstimator):
        return 'glm'
    # elif isinstance(estimator, H2OGeneralizedLowRankEstimator):
    #    return 'glrm'
    # elif isinstance(estimator, H2OKMeansEstimator):
    #    return 'km'
    elif isinstance(estimator, H2ONaiveBayesEstimator):
        return 'nb'
    elif isinstance(estimator, H2ORandomForestEstimator):
        return 'rf'
    else:
        raise TypeError('unknown type for gridsearch: %s'
                        % type(estimator))


def _score(estimator, frame, target_feature, scorer, is_regression, **kwargs):
    y_truth = frame[target_feature]

    # gen predictions...
    pred = estimator.predict(frame)['predict']

    # it's calling and h2o scorer at this point
    return scorer.score(y_truth, pred, **kwargs)


def _fit_and_score(estimator, frame, feature_names, target_feature,
                   scorer, parameters, verbose, scoring_params,
                   train, test, is_regression, act_args,
                   cv_fold, iteration):
    """Fits the current fold on the current parameters.

        Parameters
        ----------

        estimator : H2OPipeline or H2OEstimator
            The estimator to fit

        frame : H2OFrame
            The training frame

        feature_names : iterable (str)
            The feature names on which to train

        target_feature : str
            The name of the target feature

        scorer : H2OScorer
            The scoring function

        parameters : dict
            The parameters to set in the estimator clone

        verbose : int
            The level of verbosity

        scoring_params : dict
            The parameters to pass as kwargs to the scoring function

        train : iterable
            The train fold indices

        test : iterable
            The test fold indices

        is_regression : bool
            Whether we are fitting a continuous target

        act_args : dict
            GainsStatisticalReport args if called from a 
            H2OGainsRandomizedSearchCV

        cv_fold : int
            The fold number for reporting

        iteration : int
            The iteration number for reporting

        Returns
        -------

        out : list, shape=(4,)
            test_score : float
                The score produced by the ``_score`` method
                on the test fold of the training set.

            len(test) : int
                The number of samples included in the
                test fold of the training set. Used later
                for IID normalizing of test scores.

            estimator : ``H2OEstimator`` or ``H2OPipeline``
                The fit pipeline or estimator. Used for later
                scoring on the validation set.

            parameters : dict
                The parameters used to fit this estimator.
    """
    if parameters is None:
        parameters = {}

    if verbose > 1:
        if not parameters:
            msg = ''
        else:
            msg = 'Target: %s; %s' % (target_feature, ', '.join('%s=%s' % (k, v) for k, v in parameters.items()))
        print("[CV (iter %i, fold %i)] %s %s" % (iteration, cv_fold, msg, (64 - len(msg)) * '.'))

    # h2o doesn't currently re-order rows... and sometimes will
    # complain for some reason. We need to sort our train/test idcs
    train = sorted(train)
    test = sorted(test)

    # if act_args, then it's a gains search. We just need to slice
    # our existing numpy arrays
    if act_args is not None:
        kwargs = {
            'expo': act_args['expo'][test],
            'loss': act_args['loss'][test],
            'prem': act_args['prem'][test] if act_args['prem'] is not None else None
        }
    else:
        kwargs = scoring_params

    # generate split
    train_frame = frame[train, :]
    test_frame = frame[test, :]

    start_time = time.time()

    # it's probably a pipeline
    is_h2o_est = isinstance(estimator, H2OEstimator)
    if not is_h2o_est:
        estimator.set_params(**parameters)

        # the name setting should be taken care of pre-clone...
        # setattr(estimator, 'feature_names', feature_names)
        # setattr(estimator, 'target_feature',target_feature)

        # do fit
        estimator.fit(train_frame)
    else:  # it's just an H2OEstimator
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
    test_score = _score(estimator, test_frame, target_feature, scorer, is_regression, **kwargs)

    # h2o is verbose.. if we are too, print a new line:
    if verbose > 1:
        print()  # new line

    scoring_time = time.time() - start_time

    if verbose > 2:
        msg += ', score=%f' % test_score
    if verbose > 1:
        end_msg = '%s -%s' % (msg, logger.short_format_time(scoring_time))
        print('[CV (iter %i, fold %i)] %s %s' % (iteration, cv_fold, (64 - len(end_msg)) * '.', end_msg))
        print()  # new line
        print()  # new line

    out = [test_score, len(test), estimator, parameters]
    return out


class BaseH2OSearchCV(BaseH2OFunctionWrapper, VizMixin):
    """Base for all H2O grid searches"""

    _min_version = '3.8.2.9'
    _max_version = None

    @abstractmethod
    def __init__(self, estimator, feature_names,
                 target_feature, scoring=None,
                 scoring_params=None,
                 cv=5, verbose=0, iid=True,
                 validation_frame=None,
                 minimize='bias'):

        super(BaseH2OSearchCV, self).__init__(target_feature=target_feature,
                                              min_version=self._min_version,
                                              max_version=self._max_version)

        self.estimator = estimator
        self.feature_names = feature_names
        self.scoring = scoring
        self.scoring_params = scoring_params if scoring_params else {}
        self.cv = cv
        self.verbose = verbose
        self.iid = iid
        self.validation_frame = validation_frame
        self.minimize = minimize


    def _fit(self, X, parameter_iterable):
        """Actual fitting,  performing the search over parameters."""
        X = check_frame(X, copy=True)  # copy because who knows what people do inside of score...

        self.feature_names, self.target_feature = validate_x_y(X, self.feature_names, self.target_feature)
        self.is_regression_ = (not X[self.target_feature].isfactor()[0])

        # local scope
        minimize = self.minimize
        estimator = self.estimator

        # ensure minimize is in {'bias':'variance'}
        min_permitted = ['bias', 'variance']
        if minimize not in min_permitted:
            raise ValueError('minimize must be one of %s, but got %s' % (', '.join(min_permitted), str(minimize)))

        # validate the estimator... for grid search, we ONLY ALLOW the last step
        # of the grid search estimator to be an H2OEstimator. That means pipelines
        # that don't end in an estimator are invalid. If it's not a pipeline, it must
        # be an h2oestimator
        if isinstance(estimator, H2OPipeline):
            if not isinstance(estimator._final_estimator, H2OEstimator):
                raise TypeError('if estimator is H2OPipeline, its _final_estimator must '
                                'be of type H2OEstimator. Got %s' % type(estimator._final_estimator))
        elif not isinstance(estimator, H2OEstimator):
            raise TypeError('estimator must be an H2OPipeline or an H2OEstimator. Got %s' % type(estimator))

        # the addition of the gains search necessitates some hackiness.
        # if we have the attr 'extra_args_' then we know it's an gains search
        xtra = self.extra_args_ if hasattr(self, 'extra_args_') else None  # np arrays themselves
        xtra_nms = self.extra_names_ if hasattr(self,
                                                'extra_names_') else None  # the names of the prem,exp,loss features

        # we need to require scoring...
        scoring = self.scoring
        if hasattr(self,
                   'scoring_class_') or xtra is not None:  # this is a gains search, and we don't need to h2o-ize it
            pass
        else:
            if scoring is None:
                # set defaults
                if self.is_regression_:
                    scoring = 'r2_score'
                else:
                    scoring = 'accuracy_score'

            # make strs into scoring functions
            if isinstance(scoring, str):
                if scoring not in SCORERS:
                    raise ValueError('Scoring must be one of (%s) or a callable. '
                                     'Got %s' % (', '.join(SCORERS.keys()), scoring))

                scoring = SCORERS[scoring]
            # make it a scorer
            if hasattr(scoring, '__call__'):
                self.scoring_class_ = make_h2o_scorer(scoring, X[self.target_feature])
            else:  # should be impossible to get here
                raise TypeError('expected string or callable for scorer, but got %s' % type(self.scoring))

        # validate CV
        cv = check_cv(self.cv)

        # clone estimator
        nms = {
            'feature_names': self.feature_names,
            'target_feature': self.target_feature
        }

        # do first clone, remember to set the names...
        base_estimator = _clone_h2o_obj(self.estimator, **nms)

        # do fits, scores
        out = [
            _fit_and_score(estimator=_clone_h2o_obj(base_estimator),
                           frame=X, feature_names=self.feature_names,
                           target_feature=self.target_feature,
                           scorer=self.scoring_class_, parameters=params,
                           verbose=self.verbose, scoring_params=self.scoring_params,
                           train=train, test=test, is_regression=self.is_regression_,
                           act_args=xtra, cv_fold=cv_fold, iteration=iteration)
            for iteration, params in enumerate(parameter_iterable)
            for cv_fold, (train, test) in enumerate(cv.split(X, self.target_feature))
            ]

        # Out is a list of quad: score, n_test_samples, estimator, parameters
        n_fits = len(out)
        n_folds = cv.get_n_splits()

        # if a validation frame was passed, user might want to see how it scores
        # on each model, so we'll do that here...
        if self.validation_frame is not None:
            score_validation = True
            self.validation_scores = []

            if xtra_nms is not None:
                self.val_score_report_ = GainsStatisticalReport(
                    n_folds=n_folds,
                    n_iter=n_fits // n_folds,
                    iid=self.iid)

                # set scoring function
                val_scorer = self.val_score_report_

                kwargs = {
                    'expo': _as_numpy(self.validation_frame[xtra_nms['expo']]),
                    'loss': _as_numpy(self.validation_frame[xtra_nms['loss']]),
                    'prem': _as_numpy(self.validation_frame[xtra_nms['prem']]) if (
                    xtra_nms['prem'] is not None) else None
                }
            else:
                kwargs = self.scoring_params
                val_scorer = self.scoring_class_
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
                                       self.target_feature, val_scorer,
                                       self.is_regression_, **kwargs)

                    # if it's gains scorer, handles the iid condition internally...
                    self.validation_scores.append(val_score)

            if self.iid:
                score /= float(n_test_samples)
            else:
                score /= float(n_folds)

            scores.append((score, parameters))
            grid_scores.append(_CVScoreTuple(
                parameters,
                score,
                np.array(all_scores)))

        # Store the computed scores
        self.grid_scores_ = grid_scores

        # Find the best parameters by comparing on the mean validation score:
        # note that `sorted` is deterministic in the way it breaks ties
        is_bias = minimize == 'bias'
        # else == variance
        the_key = (lambda x: x.mean_validation_score) if is_bias else (lambda x: x.cv_validation_scores.std())
        best = sorted(grid_scores, key=the_key, reverse=is_bias)[0]

        self.best_params_ = best.parameters
        self.best_score_ = best.mean_validation_score

        # fit the best estimator using the entire dataset
        # clone first to work around broken estimators
        best_estimator = _clone_h2o_obj(base_estimator, **nms)

        # if verbose alert user we're at the end...
        if self.verbose > 1:
            msg = 'Target: %s; %s' % (self.target_feature, ', '.join('%s=%s' % (k, v)
                                                                     for k, v in six.iteritems(best.parameters)))
            print("\nFitting best hyperparameters across all folds")
            print("[BEST] %s %s" % (msg, (64 - len(msg)) * '.'))

        # set params -- remember h2o gets funky with this...
        if isinstance(best_estimator, H2OEstimator):
            for k, v in six.iteritems(best.parameters):
                best_estimator._parms[k] = v
            best_estimator.train(training_frame=X, x=self.feature_names, y=self.target_feature)
        else:
            best_estimator.set_params(**best.parameters)
            best_estimator.fit(X)

        # Set the best estimator
        self.best_estimator_ = best_estimator

        return self


    def score(self, frame):
        """After the grid search is fit, generates and scores 
        the predictions of the ``best_estimator_``.

        Parameters
        ----------

        frame : H2OFrame
            The test frame on which to predict

        Returns
        -------

        scor : float
            The score of the test predictions
        """
        check_is_fitted(self, 'best_estimator_')
        frame = check_frame(frame, copy=True) # copy because who knows what people do inside of score...
        scor = _score(self.best_estimator_, frame, self.target_feature,
                      self.scoring_class_, self.is_regression_,
                      **self.scoring_params)
        return scor


    @if_delegate_has_method(delegate='best_estimator_')
    def predict(self, frame):
        """After the grid search is fit, generates predictions 
        on the test frame using the ``best_estimator_``.

        Parameters
        ----------

        frame : H2OFrame
            The test frame on which to predict

        Returns
        -------

        p : H2OFrame
            The test predictions
        """
        frame = check_frame(frame, copy=False) # don't copy because predict doesn't need it
        p = self.best_estimator_.predict(frame)
        return p


    def fit_predict(self, frame):
        """First, fits the grid search and then generates predictions 
        on the training frame using the ``best_estimator_``.

        Parameters
        ----------

        frame : H2OFrame
            The training frame on which to predict

        Returns
        -------

        p : H2OFrame
            The training predictions
        """
        p = self.fit(frame).predict(frame)
        return p


    @if_delegate_isinstance(delegate='best_estimator_', instance_type=(H2OEstimator, H2OPipeline))
    def download_pojo(self, path="", get_jar=True):
        """This method is injected at runtime if the ``best_estimator_``
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
        is_h2o = isinstance(self.best_estimator_, H2OEstimator)
        if is_h2o:
            return h2o.download_pojo(self.best_estimator_, path=path, get_jar=get_jar)
        else:
            return self.best_estimator_.download_pojo(path=path, get_jar=get_jar)


    @overrides(VizMixin)
    def plot(self, timestep, metric):
        """Plot an H2OEstimator's performance over a
        given ``timestep`` (x-axis) against a provided 
        ``metric`` (y-axis).

        Parameters
        ----------

        timestep : str
            A timestep as defined in the H2O API. One of
            ("AUTO", "duration", "number_of_trees").

        metric : str
            The performance metric to evaluate. One of
            ("log_likelihood", "objective", "MSE", "AUTO")
        """
        check_is_fitted(self, 'best_estimator_')

        # then it's a pipeline:
        if hasattr(self.best_estimator_, 'plot'):
            self.best_estimator_.plot(timestep=timestep, metric=metric)
        else:
            # should be an H2OEstimator
            self.best_estimator_._plot(timestep=timestep, metric=metric)


    @staticmethod
    def load(location):
        """Loads a persisted state of an instance of BaseH2OSearchCV
        from disk. This method will handle loading H2OEstimator models separately 
        and outside of the constraints of the pickle package. 

        Note that this is a static method and should be called accordingly:

            >>> search = BaseH2OSearchCV.load('path/to/h2o/search.pkl') # GOOD!

        Also note that since BaseH2OSearchCV will contain an H2OEstimator, it's
        ``load`` functionality differs from that of its superclass, BaseH2OFunctionWrapper
        and will not function properly if called at the highest level of abstraction:

            >>> search = BaseH2OFunctionWrapper.load('path/to/h2o/search.pkl') # BAD!

        Furthermore, trying to load a different type of BaseH2OFunctionWrapper from
        this method will raise a TypeError:

            >>> mcf = BaseH2OSearchCV.load('path/to/some/other/transformer.pkl') # BAD!

        Parameters
        ----------

        location : str
            The location where the persisted BaseH2OSearchCV model resides.

        Returns
        -------

        model : BaseH2OSearchCV
            The unpickled instance of the BaseH2OSearchCV model
        """
        with open(location) as f:
            model = pickle.load(f)

        if not isinstance(model, BaseH2OSearchCV):
            raise TypeError('expected BaseH2OSearchCV, got %s' % type(model))

        # read the model portion, delete the model path
        ex = None
        the_h2o_est = None
        for pth in [model.model_loc_, 'hdfs://%s' % model.model_loc_]:
            try:
                the_h2o_est = h2o.load_model(pth)
            except Exception as e:
                if ex is None:
                    ex = e
                else:
                    # only throws if fails twice
                    raise ex

                    # break if successfully loaded
            if the_h2o_est is not None:
                break

        # if self.estimator is None, then it's simply the H2OEstimator,
        # otherwise it's going to be the H2OPipeline
        if model.best_estimator_ is None:
            model.best_estimator_ = the_h2o_est
            model.estimator = _new_base_estimator(model.est_type_, model.base_estimator_parms_)
        else:
            model.best_estimator_.steps[-1] = (model.est_name_, the_h2o_est)
            model.estimator.steps[-1] = (
                model.est_name_, _new_base_estimator(model.est_type_, model.base_estimator_parms_))

        return model


    def _save_internal(self, **kwargs):
        check_is_fitted(self, 'best_estimator_')
        best_estimator = self.best_estimator_
        estimator = self.estimator

        # where we'll save things
        loc = kwargs.pop('location')
        model_loc = kwargs.pop('model_location')

        # need to save the h2o est before anything else. Note that since
        # we verify pre-fit that the _final_estimator is of type H2OEstimator,
        # we can assume nothing has changed internally...
        is_pipe = False
        if isinstance(best_estimator, H2OPipeline):
            self.est_name_ = best_estimator.steps[-1][0]  # don't need to duplicate--can use for base

            the_h2o_est = best_estimator._final_estimator
            the_base_est = estimator._final_estimator

            is_pipe = True
        else:
            # otherwise it's the H2OEstimator
            the_h2o_est = best_estimator
            the_base_est = estimator

        # get the key that will map to the new H2OEstimator
        self.est_type_ = _get_estimator_string(the_base_est)

        # first, save the best estimator's H2O piece...
        force = kwargs.pop('force', False)
        self.model_loc_ = h2o.save_model(model=the_h2o_est, path=model_loc, force=force)

        # set to none for pickling, and then restore state for scoring
        if is_pipe:
            last_step_ = best_estimator.steps[-1]
            best_estimator.steps[-1] = None

            base_last_step_ = estimator.steps[-1]
            estimator.steps[-1] = None
            self.base_estimator_parms_ = base_last_step_[1]._parms  # it's a tuple...
        else:
            last_step_ = self.best_estimator_
            base_last_step_ = self.estimator
            self.best_estimator_ = None
            self.estimator = None
            self.base_estimator_parms_ = base_last_step_._parms

            # now save the rest of things...
        with open(loc, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

        # restore state for re-use
        if is_pipe:
            best_estimator.steps[-1] = last_step_
            estimator.steps[-1] = base_last_step_
        else:
            self.best_estimator_ = last_step_
            self.estimator = base_last_step_


    @if_delegate_has_method(delegate='best_estimator_')
    def varimp(self, use_pandas=True):
        """Get the variable importance, if the final
        estimator implements such a function.

        Parameters
        ----------

        use_pandas : bool, optional (default=True)
            Whether to return a pandas dataframe
        """
        return self.best_estimator_.varimp(use_pandas=use_pandas)


class H2OGridSearchCV(BaseH2OSearchCV):
    """An exhaustive grid search that will fit models across the
    entire hyperparameter grid provided.

    Parameters
    ----------

    estimator : H2OPipeline or H2OEstimator
        The estimator to fit.

    param_grid : dict
        The hyper parameter grid over which to search.

    feature_names : iterable (str)
        The list of feature names on which to fit

    target_feature : str
        The name of the target

    scoring : str, optional (default='lift')
        A valid scoring metric, i.e., "accuracy_score". See
        ``skutil.h2o.grid_search.SCORERS`` for a comprehensive list.

    scoring_params : dict, optional (default=None)
        Any kwargs to be passed to the scoring function for
        scoring at each iteration.

    cv : int or H2OCrossValidator, optional (default=5)
        The number of folds to be fit for cross validation.

    verbose : int, optional (default=0)
        The level of verbosity. 1,2 or greater.

    iid : bool, optional (default=True)
        Whether to consider each fold as IID. The fold scores
        are normalized at the end by the number of observations
        in each fold.

    validation_frame : H2OFrame, optional (default=None)
        Whether to score on the full validation frame at the
        end of all of the model fits. Note that this will NOT be
        used in the actual model selection process.

    minimize : str, optional (default='bias')
        How the search selects the best model to fit on the entire dataset.
        One of {'bias','variance'}. The default behavior is 'bias', which is
        also the default behavior of sklearn. This will select the set of
        hyper parameters which maximizes the cross validation score mean.
        Alternatively, 'variance' will select the model which minimizes
        the standard deviations between cross validation scores.
    """

    def __init__(self, estimator, param_grid,
                 feature_names, target_feature,
                 scoring=None, scoring_params=None,
                 cv=5, verbose=0, iid=True,
                 validation_frame=None,
                 minimize='bias'):
        super(H2OGridSearchCV, self).__init__(
            estimator=estimator,
            feature_names=feature_names,
            target_feature=target_feature,
            scoring=scoring, scoring_params=scoring_params,
            cv=cv, verbose=verbose,
            iid=iid, validation_frame=validation_frame,
            minimize=minimize
        )

        self.param_grid = param_grid
        _check_param_grid(param_grid)

    def fit(self, frame):
        return self._fit(frame, ParameterGrid(self.param_grid))


class H2ORandomizedSearchCV(BaseH2OSearchCV):
    """A grid search that operates over a random sub-hyperparameter space
    at each iteration.

    Parameters
    ----------

    estimator : H2OPipeline or H2OEstimator
        The estimator to fit.

    param_grid : dict
        The hyper parameter grid over which to search.

    feature_names : iterable (str)
        The list of feature names on which to fit

    target_feature : str
        The name of the target

    n_iter : int, optional (default=10)
        The number of iterations to fit. Note that 
        ``n_iter * cv.get_n_splits`` will be fit. If there
        are 10 folds and 10 iterations, 100 models (plus
        one) will be fit.

    random_state : int, optional (default=None)
        The random state for the search

    scoring : str, optional (default='lift')
        A valid scoring metric, i.e., "accuracy_score". See
        ``skutil.h2o.grid_search.SCORERS`` for a comprehensive list.

    scoring_params : dict, optional (default=None)
        Any kwargs to be passed to the scoring function for
        scoring at each iteration.

    cv : int or H2OCrossValidator, optional (default=5)
        The number of folds to be fit for cross validation.
        Note that ``n_iter * cv.get_n_splits`` will be fit. If there
        are 10 folds and 10 iterations, 100 models (plus
        one) will be fit.

    verbose : int, optional (default=0)
        The level of verbosity. 1,2 or greater.

    iid : bool, optional (default=True)
        Whether to consider each fold as IID. The fold scores
        are normalized at the end by the number of observations
        in each fold.

    validation_frame : H2OFrame, optional (default=None)
        Whether to score on the full validation frame at the
        end of all of the model fits. Note that this will NOT be
        used in the actual model selection process.

    minimize : str, optional (default='bias')
        How the search selects the best model to fit on the entire dataset.
        One of {'bias','variance'}. The default behavior is 'bias', which is
        also the default behavior of sklearn. This will select the set of
        hyper parameters which maximizes the cross validation score mean.
        Alternatively, 'variance' will select the model which minimizes
        the standard deviations between cross validation scores.
    """

    def __init__(self, estimator, param_grid,
                 feature_names, target_feature,
                 n_iter=10, random_state=None,
                 scoring=None, scoring_params=None,
                 cv=5, verbose=0, iid=True,
                 validation_frame=None,
                 minimize='bias'):
        super(H2ORandomizedSearchCV, self).__init__(
            estimator=estimator,
            feature_names=feature_names,
            target_feature=target_feature,
            scoring=scoring, scoring_params=scoring_params,
            cv=cv, verbose=verbose,
            iid=iid, validation_frame=validation_frame,
            minimize=minimize
        )

        self.param_grid = param_grid
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, frame):
        sampled_params = ParameterSampler(self.param_grid,
                                          self.n_iter,
                                          random_state=self.random_state)

        return self._fit(frame, sampled_params)


def _val_exp_loss_prem(x, y, z):
    """Takes three strings (or unicode) and cleans them
    for indexing an H2OFrame.

    Parameters
    ----------

    x : str
        exp name

    y : str
        loss name

    z : str
        premium name

    Returns
    -------

    out : tuple
        exp : str
            The name of the exp feature (``x``) 

        loss : str
            The name of the loss feature (``y``)

        prem : str or None
            The name of the prem feature (``z``)
    """
    if not all([isinstance(i, (str, unicode)) for i in (x, y)]):
        raise TypeError('exposure and loss must be strings or unicode')

    if z is not None:
        if not isinstance(z, (str, unicode)):
            raise TypeError('premium must be None or string or unicode')

    out = (str(x), str(y), str(z) if z is not None else z)
    return out


class H2OGainsRandomizedSearchCV(H2ORandomizedSearchCV):
    """A grid search that scores based on actuarial metrics
    (See ``skutil.metrics.GainsStatisticalReport``). This is a more
    customized form of grid search, and must use a gains metric
    provided by the ``GainsStatisticalReport``.

    Parameters
    ----------

    estimator : H2OPipeline or H2OEstimator
        The estimator to fit.

    param_grid : dict
        The hyper parameter grid over which to search.

    feature_names : iterable (str)
        The list of feature names on which to fit

    target_feature : str
        The name of the target

    exposure_feature : str
        The name of the exposure feature

    loss_feature : str
        The name of the loss feature

    premium_feature : str
        The name of the premium feature

    n_iter : int, optional (default=10)
        The number of iterations to fit. Note that 
        ``n_iter * cv.get_n_splits`` will be fit. If there
        are 10 folds and 10 iterations, 100 models (plus
        one) will be fit.

    random_state : int, optional (default=None)
        The random state for the search

    scoring : str, optional (default='lift')
        One of {'lift','gini'} or other valid GainsStatisticalReport
        scoring metrics.

    scoring_params : dict, optional (default=None)
        Any kwargs to be passed to the scoring function for
        scoring at each iteration.

    cv : int or H2OCrossValidator, optional (default=5)
        The number of folds to be fit for cross validation.
        Note that ``n_iter * cv.get_n_splits`` will be fit. If there
        are 10 folds and 10 iterations, 100 models (plus
        one) will be fit.

    verbose : int, optional (default=0)
        The level of verbosity. 1,2 or greater.

    iid : bool, optional (default=True)
        Whether to consider each fold as IID. The fold scores
        are normalized at the end by the number of observations
        in each fold.

    validation_frame : H2OFrame, optional (default=None)
        Whether to score on the full validation frame at the
        end of all of the model fits. Note that this will NOT be
        used in the actual model selection process.

    minimize : str, optional (default='bias')
        How the search selects the best model to fit on the entire dataset.
        One of {'bias','variance'}. The default behavior is 'bias', which is
        also the default behavior of sklearn. This will select the set of
        hyper parameters which maximizes the cross validation score mean.
        Alternatively, 'variance' will select the model which minimizes
        the standard deviations between cross validation scores.

    error_score : float, optional (default=np.nan)
        The default score to use in the case of a pd.qcuts ValueError
        (when there are non-unique bin edges)

    error_behavior : str, optional (default='warn')
        How to handle the pd.qcut ValueError. One of {'warn','raise','ignore'}
    """

    def __init__(self, estimator, param_grid,
                 feature_names, target_feature,
                 exposure_feature, loss_feature,
                 premium_feature=None, n_iter=10,
                 random_state=None, scoring='lift',
                 scoring_params=None, cv=5,
                 verbose=0, iid=True,  # n_groups=10,
                 validation_frame=None, minimize='bias',
                 error_score=np.nan, error_behavior='warn'):
        super(H2OGainsRandomizedSearchCV, self).__init__(
            estimator=estimator,
            param_grid=param_grid,
            feature_names=feature_names,
            target_feature=target_feature,
            n_iter=n_iter, random_state=random_state,
            scoring=scoring, scoring_params=scoring_params,
            cv=cv, verbose=verbose,
            iid=iid, validation_frame=validation_frame,
            minimize=minimize
        )

        # self.n_groups = 10
        self.exposure_feature = exposure_feature
        self.loss_feature = loss_feature
        self.premium_feature = premium_feature

        # for re-fitting, we need these kwargs saved
        self.grsttngs_ = {
            'score_by': scoring,
            'n_folds': check_cv(cv).get_n_splits(),
            'n_iter': n_iter,
            'iid': iid,
            'error_score': error_score,
            'error_behavior': error_behavior
        }

        # the scoring_class_ (set in ``fit``) will do the scoring
        self.scoring = None

    def fit(self, frame):
        sampled_params = ParameterSampler(self.param_grid,
                                          self.n_iter,
                                          random_state=self.random_state)

        # set our score class
        self.scoring_class_ = GainsStatisticalReport(**self.grsttngs_)

        # we can do this once to avoid many as_data_frame operations
        exp, loss, prem = _val_exp_loss_prem(self.exposure_feature, self.loss_feature, self.premium_feature)
        self.extra_args_ = {
            'expo': _as_numpy(frame[exp]),
            'loss': _as_numpy(frame[loss]),
            'prem': _as_numpy(frame[prem]) if prem is not None else None
        }

        # for validation set
        self.extra_names_ = {
            'expo': exp,
            'loss': loss,
            'prem': prem
        }

        # do fit
        the_fit = self._fit(frame, sampled_params)

        # clear extra_args_, because they might take lots of mem
        # we can do this because a re-fit will re-assign them anyways.
        # don't delete the extra_names_ though, because they're used in
        # scoring the incoming frame.
        del self.extra_args_

        return the_fit

    def report_scores(self):
        """Create a dataframe report for the fitting and scoring of the
        gains search. Will report lift, gini and any other relevant metrics.
        If a validation set was included, will also report validation scores.

        Returns
        -------

        rdf : pd.DataFrame
            The grid search report
        """
        check_is_fitted(self, 'best_estimator_')
        report_res = self.scoring_class_.as_data_frame()
        n_obs, _ = report_res.shape

        # Need to cbind the parameters... we don't care about ["score", "std"]
        rdf = report_grid_score_detail(self, charts=False).drop(["score", "std"], axis=1)
        assert n_obs == rdf.shape[0], 'Internal error: %d!=%d' % (n_obs, rdf.shape[0])

        # change the names in the dataframe...
        report_res.columns = ['train_%s' % x for x in report_res.columns.values]

        # cbind...
        rdf = pd.concat([rdf, report_res], axis=1)

        # if we scored on the validation set, also need to get the val score struct
        if hasattr(self, 'val_score_report_'):
            val_res_df = self.val_score_report_.as_data_frame()
            assert n_obs == val_res_df.shape[0], 'Internal error: %d!=%d' % (n_obs, val_res_df.shape[0])
            val_res_df.columns = ['val_%s' % x for x in val_res_df.columns.values]

            # cbind
            rdf = pd.concat([rdf, val_res_df], axis=1)

        rdf.index = ['Iter_%i' % i for i in range(self.n_iter)]
        return rdf

    @overrides(BaseH2OSearchCV)
    def score(self, frame):
        """Predict and score on a new frame. Note that this method
        will not store performance metrics in the report that ``report_score``
        generates.

        Parameters
        ----------

        frame : H2OFrame
            The test frame on which to predict and score performance.

        Returns
        -------

        scor : float
            The score on the testing frame
        """
        check_is_fitted(self, 'best_estimator_')
        e, l, p = self.extra_names_['expo'], self.extra_names_['loss'], self.extra_names_['prem']

        kwargs = {
            'expo': frame[e],
            'loss': frame[l],
            'prem': frame[p] if p is not None else None
        }

        y_truth = frame[self.target_feature]
        pred = self.best_estimator_.predict(frame)['predict']
        scor = self.scoring_class_.score_no_store(y_truth, pred, **kwargs)

        return scor
