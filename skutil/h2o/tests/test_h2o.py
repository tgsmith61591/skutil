from __future__ import print_function, division
import warnings
import numpy as np
import pandas as pd
import time
import os

import h2o
from h2o.frame import H2OFrame
from h2o.estimators import (H2ORandomForestEstimator,
                            H2OGeneralizedLinearEstimator,
                            H2OGradientBoostingEstimator,
                            H2ODeepLearningEstimator,
                            H2ONaiveBayesEstimator)

from skutil.h2o import from_pandas, from_array
from skutil.h2o.base import *
from skutil.h2o.encode import *
from skutil.h2o.select import *
from skutil.h2o.pipeline import *
from skutil.h2o.grid_search import *
from skutil.h2o.base import BaseH2OFunctionWrapper
from skutil.h2o.one_way_fs import h2o_f_classif
from skutil.preprocessing.balance import _pd_frame_to_np
from skutil.h2o.util import (h2o_frame_memory_estimate, h2o_corr_plot, h2o_bincount,
                             load_iris_h2o, load_breast_cancer_h2o, load_boston_h2o,
                             shuffle_h2o_frame)
from skutil.h2o.grid_search import _as_numpy
from skutil.h2o.metrics import *
from skutil.h2o.grid_search import _val_exp_loss_prem
from skutil.utils import load_iris_df, load_breast_cancer_df, shuffle_dataframe, df_memory_estimate, load_boston_df
from skutil.utils.tests.utils import assert_fails
from skutil.feature_selection import NearZeroVarianceFilterer
from skutil.h2o.split import (check_cv, H2OKFold,
                              H2OStratifiedKFold, h2o_train_test_split,
                              _validate_shuffle_split_init, _validate_shuffle_split,
                              _val_y, H2OBaseCrossValidator, H2OStratifiedShuffleSplit)
from skutil.h2o.balance import H2OUndersamplingClassBalancer, H2OOversamplingClassBalancer
from skutil.h2o.transform import H2OSelectiveImputer, H2OInteractionTermTransformer, H2OSelectiveScaler, H2OLabelEncoder
from skutil.utils import flatten_all
from skutil.h2o.frame import is_integer, is_float
from skutil.h2o.pipeline import _union_exclusions
from skutil.h2o.select import _validate_use

from sklearn.datasets import load_iris, load_boston
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint, uniform

from numpy.random import choice
from numpy.testing import (assert_array_equal, assert_almost_equal, assert_array_almost_equal)

# for split
try:
    from sklearn.model_selection import train_test_split
except ImportError as i:
    from sklearn.cross_validation import train_test_split


def new_h2o_frame(X):
    Y = H2OFrame.from_python(X, header=1,
                             column_names=X.columns.tolist())
    return Y


def new_estimators():
    """Returns a tuple of newly initialized estimators to test all of them
    with the skutil framework. This ensures it will work with all the estimators...
    """
    return (
        H2ORandomForestEstimator(ntrees=5),
        # H2OGeneralizedLinearEstimator(family='multinomial'),
        H2OGradientBoostingEstimator(distribution='multinomial', ntrees=5),
        H2ODeepLearningEstimator(distribution='multinomial', epochs=1, hidden=[10, 10])
    )


def test_h2o_no_conn_needed():
    # make an anonymous class that extends base h2o
    class AnonH2O(BaseH2OFunctionWrapper):
        def __init__(self, min_version, max_version):
            super(AnonH2O, self).__init__(target_feature=None,
                                          min_version=min_version,
                                          max_version=max_version)

    class SomeObj(object):
        def __init__(self):
            super(SomeObj, self).__init__()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        vs = [
            {'min_version': None, 'max_version': None},
            {'min_version': '3.8.2.9', 'max_version': SomeObj}
        ]

        for kwargs in vs:
            assert_fails(AnonH2O, ValueError, **kwargs)

    # test NotImplemented on anonymous VizMixin
    class AnonViz(object, VizMixin):
        def plot(self, timestep, metric):
            return super(AnonViz, self).plot(timestep, metric)

    assert AnonViz().plot(None, None) is NotImplemented

    # test _val_exp_loss_prem
    assert_fails(_val_exp_loss_prem, TypeError, 1, 2, 3)  # they should be strings or unicode
    assert_fails(_val_exp_loss_prem, TypeError, '1', '2', 3)  # z should also be a string

    # test union_exclusion in pipeline
    a, b = ['a','b'], ['a','c']
    assert not _union_exclusions(None, None)
    assert _union_exclusions(None, b) == b
    assert _union_exclusions(a, None) == a
    assert _union_exclusions(a, b) == flatten_all([a, b])

    # test _validate_use
    assert_fails(_validate_use, ValueError, None, 'blah', False)


# if we can't start an h2o instance, let's just pass all these tests
def test_h2o_with_conn():
    iris = load_iris()
    F = pd.DataFrame.from_records(data=iris.data, columns=iris.feature_names)
    # F = load_iris_df(include_tgt=False)
    X = None

    try:
        h2o.init()
        # h2o.init(ip='localhost', port=54321) # this might throw a warning

        # sleep before trying this:
        time.sleep(10)
        X = new_h2o_frame(F)
    except Exception as e:
        # raise #for debugging

        # if we can't start on localhost, try default (127.0.0.1)
        # we'll try it a few times, since H2O is SUPER finnicky
        max_tries = 3
        count = 0

        while (count < max_tries) and (X is None):
            try:
                h2o.init()
                X = new_h2o_frame(F)
            except Exception as e:
                count += 1

        if X is None:
            warnings.warn('could not successfully start H2O instance, tried %d times' % max_tries, UserWarning)

    def catch_warning_assert_thrown(fun, kwargs):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            ret = fun(**kwargs)
            # assert len(w) > 0 if X is None else True, 'expected warning to be thrown'
            return ret

    def valid_use():
        if X is not None:
            df = pd.DataFrame.from_records(data=[[1,'NA'], [2,'NA'], [3, 3]],
                                           columns=['a','b'])

            try:
                dfh = new_h2o_frame(df)
            except Exception as e:
                dfh = None
                return

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                assert _validate_use(dfh, 'all.obs', True) == 'complete.obs'
                assert len(w) > 0

    def feature_dropper_coverage():
        if X is not None:
            dropper = H2OFeatureDropper(feature_names=None)
            y = dropper.fit_transform(X)

            # assert nothing dropped
            assert len(y.columns) == len(X.columns)

            # try feature dropper with non-iterable
            dropper = H2OFeatureDropper(feature_names=1)
            assert_fails(dropper.fit, ValueError, X)

    def multicollinearity():
        # one way or another, we can initialize it
        filterer = catch_warning_assert_thrown(H2OMulticollinearityFilterer, {'threshold': 0.6})
        assert not filterer.max_version

        if X is not None:
            x = filterer.fit_transform(X)
            assert x.shape[1] == 2

            # test some exceptions...
            assert_fails(filterer.fit, TypeError, F)

            # test with a target feature
            tgt = 'sepal length (cm)'
            new_filterer = catch_warning_assert_thrown(H2OMulticollinearityFilterer,
                                                       {'threshold': 0.6, 'target_feature': tgt})
            x = new_filterer.fit_transform(X)

            # h2o throws weird error because it override __contains__, so use the comprehension
            assert tgt in [c for c in x.columns], 'target feature was accidentally dropped...'

            # assert save/load works
            filterer.fit(X)
            the_path = 'mcf.pkl'
            filterer.save(the_path, warn_if_exists=False)
            assert os.path.exists(the_path)

            # load and transform...
            filterer = H2OMulticollinearityFilterer.load(the_path)
            x = filterer.transform(X)
            assert x.shape[1] == 2

        else:
            pass

    def nzv():
        filterer = catch_warning_assert_thrown(H2ONearZeroVarianceFilterer, {'threshold': 1e-8})
        assert not filterer.max_version

        # let's add a zero var feature to F
        f = F.copy()
        f['zerovar'] = np.zeros(F.shape[0])

        try:
            Y = new_h2o_frame(f)
        except Exception as e:
            Y = None

        if Y is not None:
            y = filterer.fit_transform(Y)
            assert len(filterer.drop_) == 1
            assert y.shape[1] == 4
        else:
            pass

        # test with a target feature
        if X is not None:
            tgt = 'sepal length (cm)'
            new_filterer = catch_warning_assert_thrown(H2ONearZeroVarianceFilterer,
                                                       {'threshold': 1e-8, 'target_feature': tgt})
            y = new_filterer.fit_transform(Y)

            assert len(new_filterer.drop_) == 1
            assert y.shape[1] == 4

            # h2o throws weird error because it override __contains__, so use the comprehension
            assert tgt in [c for c in y.columns], 'target feature was accidentally dropped...'

        else:
            pass

    def pipeline():
        f = F.copy()
        targ = iris.target
        targ = ['a' if x == 0 else 'b' if x == 1 else 'c' for x in targ]

        # do split
        X_train, X_test, y_train, y_test = train_test_split(f, targ, train_size=0.7)

        # add the y into the matrix for h2o's sake -- pandas will throw a warning here...
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore")
            X_train['species'] = y_train
            X_test['species'] = y_test

        try:
            train = new_h2o_frame(X_train)
            test = new_h2o_frame(X_test)
        except Exception as e:
            train = None
            test = None

        if train is not None:
            for estimator in new_estimators():
                # define pipe
                pipe = H2OPipeline([
                    ('nzv', H2ONearZeroVarianceFilterer()),
                    ('mc', H2OMulticollinearityFilterer(threshold=0.9)),
                    ('est', estimator)
                ],
                    feature_names=F.columns.tolist(),
                    target_feature='species'
                )

                # fit pipe...
                pipe.fit(train)

                # refit for _reset coverage...
                pipe.fit(train)

                # try predicting
                pipe.predict(test)

                # coverage:
                fe = pipe._final_estimator
                ns = pipe.named_steps

            # test with all transformers and no estimator
            pipe = H2OPipeline([
                ('nzv', H2ONearZeroVarianceFilterer()),
                ('mc', H2OMulticollinearityFilterer(threshold=0.9))
            ],
                feature_names=F.columns.tolist(),
                target_feature='species'
            )

            X_transformed = pipe.fit(train).transform(train)

            # test with all transformers and no estimator -- but this time, we
            # are testing that we can set the params even when the last step
            # is not an H2OEstimator
            pipe = H2OPipeline([
                ('nzv', H2ONearZeroVarianceFilterer()),
                ('mc', H2OMulticollinearityFilterer())
            ],
                feature_names=F.columns.tolist(),
                target_feature='species'
            )

            # here's where we test...
            pipe.set_params(**{'mc__threshold': 0.9})
            assert pipe._final_estimator.threshold == 0.9

            # test some failures -- first, Y
            for y in [None, 1]:
                pipe = H2OPipeline([
                    ('nzv', H2ONearZeroVarianceFilterer()),
                    ('mc', H2OMulticollinearityFilterer(threshold=0.9)),
                    ('est', H2OGradientBoostingEstimator(distribution='multinomial', ntrees=5))
                ],
                    feature_names=F.columns.tolist(),
                    target_feature=y
                )

                excepted = False
                try:
                    pipe.fit(train)
                except (TypeError, ValueError, EnvironmentError) as e:
                    excepted = True
                assert excepted, 'expected failure for y=%s' % str(y)

            # now X
            pipe = H2OPipeline([
                ('nzv', H2ONearZeroVarianceFilterer()),
                ('mc', H2OMulticollinearityFilterer(threshold=0.9)),
                ('est', H2OGradientBoostingEstimator(distribution='multinomial', ntrees=5))
            ],
                feature_names=1,
                target_feature='species'
            )

            assert_fails(pipe.fit, TypeError, train)

            # now non-unique names
            failed = False
            try:
                pipe = H2OPipeline([
                    ('nzv', H2ONearZeroVarianceFilterer()),
                    ('mc', H2OMulticollinearityFilterer(threshold=0.9)),
                    ('mc', H2OGradientBoostingEstimator(distribution='multinomial', ntrees=5))
                ],
                    feature_names=F.columns.tolist(),
                    target_feature='species'
                )

                # won't even get here...
                pipe.fit(train)
            except ValueError as v:
                failed = True
            assert failed

            # fails for non-h2o transformers
            failed = False
            try:
                pipe = H2OPipeline([
                    ('nzv', NearZeroVarianceFilterer()),
                    ('mc', H2OMulticollinearityFilterer(threshold=0.9)),
                    ('est', H2OGradientBoostingEstimator(distribution='multinomial', ntrees=5))
                ],
                    feature_names=F.columns.tolist(),
                    target_feature='species'
                )

                # won't even get here...
                # pipe.fit(train)
            except TypeError as t:
                failed = True
            assert failed

            # type error for non-h2o estimators
            failed = False
            try:
                pipe = H2OPipeline([
                    ('nzv', H2ONearZeroVarianceFilterer()),
                    ('mc', H2OMulticollinearityFilterer(threshold=0.9)),
                    ('est', RandomForestClassifier())
                ],
                    feature_names=F.columns.tolist(),
                    target_feature='species'
                )

                # won't even get here...
                # pipe.fit(train)
            except TypeError as t:
                failed = True
            assert failed

            # test with such stringent thresholds that no features are retained
            pipe = H2OPipeline([
                ('mcf', H2OMulticollinearityFilterer(threshold=0.1)),  # will retain one
                ('nzv', H2ONearZeroVarianceFilterer(threshold=100)),  # super high thresh
                ('est', H2ORandomForestEstimator())
            ], feature_names=F.columns.tolist(), target_feature='species'
            )

            # this will fail because no cols are retained
            assert_fails(pipe.fit, ValueError, train)

            # test with exclusions
            pipe = H2OPipeline([
                ('nzv', H2ONearZeroVarianceFilterer()),
                ('mc', H2OMulticollinearityFilterer(threshold=0.9)),
                ('est', H2ORandomForestEstimator())
            ],
                feature_names=F.columns.tolist(),
                target_feature='species',
                exclude_from_fit=['sepal width (cm)'] # will not be included in the final fit
            )

            # fit pipe, predict...
            pipe.fit_predict(train)

            # if we set params to None, assert it just does nothing but return itself
            pipe.set_params(**{})
            assert pipe.exclude_from_fit == ['sepal width (cm)']
        else:
            pass

    def grid():
        # test as_numpy
        assert_fails(_as_numpy, (ValueError, TypeError, AssertionError), F)  # fails because not H2OFrame

        f = F.copy()
        targ = iris.target
        names = f.columns.values
        TGT_NAME = 'species'
        targ = ['a' if x == 0 else 'b' if x == 1 else 'c' for x in targ]
        f[TGT_NAME] = targ

        # shuffle the rows
        f = f.iloc[np.random.permutation(np.arange(f.shape[0]))]

        # try uploading...
        try:
            frame = new_h2o_frame(f)
        except Exception as e:
            frame = None

        def get_param_grid(est):
            if isinstance(est, (H2ORandomForestEstimator, H2OGradientBoostingEstimator)):
                return {
                    'ntrees': [3, 5]
                }
            elif isinstance(est, H2ODeepLearningEstimator):
                return {
                    'activation': ['Tanh', 'Rectifier']
                }
            else:
                return {
                    'standardize': [True, False]
                }

        # most of this is for extreme coverage to make sure it all works...
        if frame is not None:
            n_folds = 2

            # first, let's assert things don't work with all transformers and no estimator
            pipe = H2OPipeline([
                ('nzv', H2ONearZeroVarianceFilterer()),
                ('mc', H2OMulticollinearityFilterer(threshold=0.9))
            ],
                feature_names=names,
                target_feature=TGT_NAME
            )

            hyp = {
                'nzv__threshold': [0.5, 0.6]
            }

            grd = H2ORandomizedSearchCV(estimator=pipe,
                                        feature_names=names, target_feature=TGT_NAME,
                                        param_grid=hyp, scoring='accuracy_score', cv=2, n_iter=1)

            # it will fail in the fit method because there's no estimator at the end...
            assert_fails(grd.fit, TypeError, frame)

            # test on all the types of classification metrics...
            mtrcs = [
                h2o_accuracy_score,
                h2o_f1_score,
                h2o_precision_score,
                h2o_recall_score,
                None,
                'bad'
            ]

            pipe = H2OPipeline([
                ('nzv', H2ONearZeroVarianceFilterer()),
                ('est', H2ORandomForestEstimator())
            ])

            hyp = {
                'nzv__threshold': [0.5, 0.6]
            }

            for mtc in mtrcs:
                vbs = choice([1, 2])
                iid = choice([True, False])
                kwargs = {} if mtc in (h2o_accuracy_score, None, 'bad') else {'average': 'micro'}
                grd = H2ORandomizedSearchCV(estimator=pipe,
                                            feature_names=names, target_feature=TGT_NAME,
                                            param_grid=hyp, scoring=mtc, cv=2, n_iter=1,
                                            scoring_params=kwargs, verbose=vbs, iid=iid)

                if mtc == 'bad':
                    assert_fails(grd.fit, ValueError, frame)
                else:
                    # should pass
                    grd.fit(frame)

            for is_random in [False, True]:
                for estimator in new_estimators():
                    for do_pipe in [False, True]:
                        for iid in [True]:  # just do True for now since we already hit false above
                            for verbose in [3]:  # just do 3 for now since we already hit 1 or 2 above
                                for scoring in ['accuracy_score']:

                                    # should we shuffle?
                                    do_shuffle = choice([True, False])
                                    minimize = choice(['bias', 'variance'])

                                    # just for coverage...
                                    which_cv = choice([
                                        n_folds,
                                        H2OKFold(n_folds=n_folds, shuffle=do_shuffle)
                                    ])

                                    # get which module to use
                                    if is_random:
                                        grid_module = H2ORandomizedSearchCV
                                    else:
                                        grid_module = H2OGridSearchCV

                                    if not do_pipe:
                                        # we're just testing the search on actual estimators
                                        grid = grid_module(estimator=estimator,
                                                           feature_names=F.columns.tolist(), target_feature='species',
                                                           param_grid=get_param_grid(estimator),
                                                           scoring=scoring, iid=iid, verbose=verbose,
                                                           cv=which_cv, minimize=minimize)
                                    else:

                                        # pipify -- the feature names, etc., will be set in the grid
                                        pipe = H2OPipeline([
                                            ('nzv', H2ONearZeroVarianceFilterer()),
                                            ('est', estimator)
                                        ])

                                        # determine which params to use
                                        # we'll just use a NZV filter and tinker with the thresh
                                        if is_random:
                                            params = {
                                                'nzv__threshold': uniform(1e-6, 0.0025)
                                            }
                                        else:
                                            params = {
                                                'nzv__threshold': [1e-6, 1e-8]
                                            }

                                        grid = grid_module(pipe, param_grid=params,
                                                           feature_names=F.columns.tolist(), target_feature='species',
                                                           scoring=scoring, iid=iid, verbose=verbose,
                                                           cv=which_cv, minimize=minimize)

                                    # if it's a random search CV obj, let's keep it brief
                                    if is_random:
                                        grid.n_iter = n_folds

                                    # sometimes we'll expect it to fail...
                                    expect_failure = scoring is None or (
                                    isinstance(scoring, str) and scoring in ('bad'))
                                    try:
                                        # fit the grid
                                        grid.fit(frame)

                                        # we expect the exception to be thrown
                                        # above, so now we set expect_failure to False
                                        expect_failure = False

                                        # predict on the grid
                                        p = grid.predict(frame)

                                        # score on the frame
                                        s = grid.score(frame)
                                    except ValueError as v:
                                        if expect_failure:
                                            pass
                                        else:
                                            raise

                                    # try varimp
                                    if hasattr(grid, 'varimp'):
                                        grid.varimp()

            # can we just fit one with a validation frame for coverage?
            pipe = H2OPipeline([
                ('nzv', H2ONearZeroVarianceFilterer()),
                ('est', H2OGradientBoostingEstimator(ntrees=5))
            ])

            hyper = {
                'nzv__threshold': uniform(1e-6, 0.0025)
            }

            grid = H2ORandomizedSearchCV(pipe, param_grid=hyper,
                                         feature_names=F.columns.tolist(), target_feature='species',
                                         scoring='accuracy_score', iid=True, verbose=0, cv=2,
                                         validation_frame=frame)

            grid.fit(frame)

            # also, can we make it fail for non-permitted minimizer?
            grid = H2ORandomizedSearchCV(pipe, param_grid=hyper,
                                         feature_names=F.columns.tolist(), target_feature='species',
                                         scoring='accuracy_score', iid=True, verbose=0, cv=2,
                                         validation_frame=frame, minimize='bad_string')

            assert_fails(grid.fit, ValueError, frame)  # fails because of the minimize value


        else:
            pass

        # let's do one pass with a stratified K fold... but we need to increase our data length,
        # lest we lose records on the split, which will throw errors...
        f = pd.concat([f, f, f, f, f], axis=0)  # times FIVE!

        # shuffle the rows
        f = f.iloc[np.random.permutation(np.arange(f.shape[0]))]

        # try uploading again...
        try:
            frame = new_h2o_frame(f)
        except Exception as e:
            frame = None

        if frame is not None:
            n_folds = 2
            for estimator in new_estimators():
                pipe = H2OPipeline([
                    ('nzv', H2ONearZeroVarianceFilterer()),
                    ('est', estimator)
                ])

                params = {
                    'nzv__threshold': uniform(1e-6, 0.0025)
                }

                grid = H2ORandomizedSearchCV(pipe, param_grid=params,
                                             feature_names=F.columns.tolist(), target_feature='species',
                                             scoring='accuracy_score', n_iter=2,
                                             cv=H2OStratifiedKFold(n_folds=3, shuffle=True))

                # do fit
                grid.fit(frame)
        else:
            pass

    def anon_class():
        class H2OAnonClass100(BaseH2OFunctionWrapper):
            _min_version = 100.0
            _max_version = None

            def __init__(self):
                super(H2OAnonClass100, self).__init__(
                    min_version=self._min_version,
                    max_version=self._max_version)

        # assert fails for min version > current version
        assert_fails(H2OAnonClass100, EnvironmentError)

        class H2OAnonClassAny(BaseH2OFunctionWrapper):
            _min_version = 'any'
            _max_version = 0.1

            def __init__(self):
                super(H2OAnonClassAny, self).__init__(
                    min_version=self._min_version,
                    max_version=self._max_version)

        # assert fails for max version < current version
        assert_fails(H2OAnonClassAny, EnvironmentError)

        class H2OAnonClassPass(BaseH2OFunctionWrapper):
            def __init__(self):
                super(H2OAnonClassPass, self).__init__(
                    min_version='any',
                    max_version=None)

        # assert fails for max version < current version
        h = H2OAnonClassPass()
        assert h.max_version is None
        assert h.min_version == 'any'

    def cv():
        assert check_cv(None).get_n_splits() == 3
        assert_fails(check_cv, ValueError, 'not_a_valid_arg')

        f = F.copy()
        targ = iris.target
        targ = ['a' if x == 0 else 'b' if x == 1 else 'c' for x in targ]
        f['species'] = targ
        f = shuffle_dataframe(pd.concat([f, f, f, f, f], axis=0))  # times FIVE!

        try:
            Y = from_pandas(f)
        except Exception as e:
            Y = None

        if Y is not None:
            # test the splits
            def do_split(gen, *args):
                return [x for x in gen(*args)]

            # test stratifed with shuffle on/off
            for shuffle in [True, False]:
                for y in ['species', None]:
                    strat = H2OStratifiedKFold(shuffle=shuffle)

                    # test the __repr__
                    strat.__repr__()

                    # only fails if y is None
                    try:
                        do_split(strat.split, Y, y)
                    except ValueError as v:
                        if y is None:
                            pass
                        else:
                            raise

            # test kfold with too many or too few folds
            for n_folds in [1, 'blah']:
                assert_fails(H2OKFold, ValueError, **{'n_folds': n_folds})

            # assert not logical shuffle fails
            assert_fails(H2OKFold, TypeError, **{'shuffle': 'sure'})

            # assert split with n_splits > n_obs fails
            failed = False
            try:
                do_split(H2OKFold(n_folds=Y.shape[0] + 1).split, Y)
            except ValueError:
                failed = True
            assert failed

        # we'll try the stratified split on the BC dataset
        try:
            B = new_h2o_frame(load_breast_cancer_df(tgt_name='target'))
        except Exception as e:
            B = None

        if B is not None:
            # can we force this weird stratified behavior where
            # n_train and n_train don't add up to enough rows?
            splitter = H2OStratifiedShuffleSplit(test_size=0.2473)
            splits = [x for x in splitter.split(B, 'target')]  # it's a list of tuples

    def split_tsts():
        # testing val_y
        assert_fails(_val_y, TypeError, 1)
        assert _val_y(None) is None
        assert isinstance(_val_y(unicode('asdf')), str)

        # testing _validate_shuffle_split_init
        assert_fails(_validate_shuffle_split_init, ValueError,
                     **{'test_size': None, 'train_size': None})  # can't both be None
        assert_fails(_validate_shuffle_split_init, ValueError,
                     **{'test_size': 1.1, 'train_size': 0.0})  # if float, can't be over 1.
        assert_fails(_validate_shuffle_split_init, ValueError,
                     **{'test_size': 'string', 'train_size': None})  # if not float, must be int
        assert_fails(_validate_shuffle_split_init, ValueError,
                     **{'train_size': 1.1, 'test_size': 0.0})  # if float, can't be over 1.
        assert_fails(_validate_shuffle_split_init, ValueError,
                     **{'train_size': 'string', 'test_size': None})  # if not float, must be int
        assert_fails(_validate_shuffle_split_init, ValueError,
                     **{'test_size': 0.6, 'train_size': 0.5})  # if both float, must not exceed 1.

        # testing _validate_shuffle_split
        assert_fails(_validate_shuffle_split, ValueError,
                     **{'n_samples': 10, 'test_size': 11, 'train_size': 0})  # test_size can't exceed n_samples
        assert_fails(_validate_shuffle_split, ValueError,
                     **{'n_samples': 10, 'train_size': 11, 'test_size': 0})  # train_size can't exceed n_samples
        assert_fails(_validate_shuffle_split, ValueError,
                     **{'n_samples': 10, 'train_size': 5, 'test_size': 6})  # sum can't exceed n_samples

        # test with train as None
        n_train, n_test = _validate_shuffle_split(n_samples=100, test_size=30, train_size=None)
        assert n_train == 70

        # test with test as None
        n_train, n_test = _validate_shuffle_split(n_samples=100, test_size=None, train_size=70)
        assert n_test == 30

        # test what works:
        n_train, n_test = _validate_shuffle_split(n_samples=10, test_size=3, train_size=7)
        assert n_train == 7
        assert n_test == 3

        n_train, n_test = _validate_shuffle_split(n_samples=10, test_size=0.3, train_size=0.7)
        assert n_train == 7
        assert n_test == 3

        # now actually get the train, test splits...
        # let's do one pass with a stratified K fold... but we need to increase our data length, 
        # lest we lose records on the split, which will throw errors...
        f = F.copy()
        targ = iris.target
        targ = ['a' if x == 0 else 'b' if x == 1 else 'c' for x in targ]
        f['species'] = targ

        # Times FIVE!
        f = pd.concat([f, f, f, f, f], axis=0)  # times FIVE!

        # shuffle the rows
        f = f.iloc[np.random.permutation(np.arange(f.shape[0]))]

        # try uploading again...
        try:
            frame = new_h2o_frame(f)
        except Exception as e:
            frame = None

        if frame is not None:
            for stratified in [None, 'species']:
                X_train, X_test = h2o_train_test_split(frame, test_size=0.3, train_size=0.7, stratify=stratified)
                a, b = (X_train.shape[0] + X_test.shape[0]), frame.shape[0]
                assert a == b, '%d != %d' % (a, b)

                # do split with train/test sizes = None, assert shapes
            X_train, X_test = h2o_train_test_split(frame)
            assert X_test.shape[0] in (187, 188)  # 25%
        else:
            pass

    def act_search():
        tgt = 'target'
        X_bost = load_boston_df(shuffle=True, include_tgt=True, tgt_name=tgt)
        names = X_bost.columns[:-1]

        X_bost['expo'] = [1 + np.random.rand() for i in range(X_bost.shape[0])]
        X_bost['loss'] = [1 + np.random.rand() for i in range(X_bost.shape[0])]
        X_train, X_test = train_test_split(X_bost, train_size=0.7)

        # now upload to cloud...
        try:
            train = from_pandas(X_train)
            test = from_pandas(X_test)
        except Exception as e:
            train = None
            test = None

        if all([x is not None for x in (train, test)]):
            pipe = H2OPipeline([
                ('rf', H2ORandomForestEstimator(seed=42))
            ])

            hyper = {
                'rf__ntrees': [10, 15]
            }

            search = H2OGainsRandomizedSearchCV(
                estimator=pipe,
                param_grid=hyper,
                exposure_feature='expo',
                loss_feature='loss',
                feature_names=names,
                target_feature=tgt,
                cv=2,
                n_iter=2)

            search.fit(train)
            search.score(test)

            # report:
            report = search.report_scores()

        else:
            pass

    def sparse():
        f = F.copy()
        f['sparse'] = ['NA' for i in range(f.shape[0])]

        # try uploading...
        try:
            frame = new_h2o_frame(f)
        except Exception as e:
            frame = None

        if frame is not None:
            filterer = H2OSparseFeatureDropper()
            filterer.fit(frame)
            assert 'sparse' in filterer.drop_

            # assert fails for over 1.0 or under 0.0
            assert_fails(H2OSparseFeatureDropper(threshold=-0.1).fit, ValueError, frame)
            assert_fails(H2OSparseFeatureDropper(threshold=1.0).fit, ValueError, frame)

        else:
            pass

    def impute():
        f = F.copy()
        choices = [choice(range(f.shape[0]), 15) for i in range(f.shape[1])]  # which indices will be NA?
        for i, v in enumerate(choices):
            f.iloc[v, i] = 'NA'

        def _basic_scenario(X, fill):
            imputer = H2OSelectiveImputer(def_fill=fill)
            imputer.fit_transform(X)
            na_cnt = sum(X.nacnt())
            assert not na_cnt, 'expected no NAs, but found %d' % na_cnt

        def scenario_1(X):
            """Assert functions with 'mean'"""
            _basic_scenario(X, 'mean')

        def scenario_2(X):
            """Assert functions with 'median'"""
            _basic_scenario(X, 'median')

        def scenario_3(X):
            """Assert fails (for now) with 'mode' -- unimplemented"""
            assert_fails(_basic_scenario, NotImplementedError, X, 'mode')

        def scenario_4(X):
            """Assert fails with unknown string arg"""
            assert_fails(_basic_scenario, TypeError, X, 'bad_string')

        def scenario_5(X):
            """Assert functions with list of imputes"""
            _basic_scenario(X, ['mean', 1.5, 'median', 'median'])

        def scenario_6(X):
            """Assert fails with list with unknown string args"""
            assert_fails(_basic_scenario, TypeError, X, ['bad_string', 1.5, 'median', 'median'])

        def scenario_7(X):
            """Assert fails with 'mode' in the list -- unimplemented"""
            assert_fails(_basic_scenario, NotImplementedError, X, ['mode', 1.5, 'median', 'median'])

        def scenario_8(X):
            """Assert fails with len != ncols"""
            assert_fails(_basic_scenario, ValueError, X, ['mean', 1.5, 2.0])

        def scenario_9(X):
            """Assert fails with random object as def_fill"""

            # arbitrary object:
            class Anon1(object):
                def __init__(self):
                    pass

                def __str__(self):
                    return 'Type: Anon1'

            assert_fails(_basic_scenario, TypeError, X, Anon1)

        def scenario_10(X):
            """Assert functions feature_names arg"""
            imputer = H2OSelectiveImputer(feature_names=X.columns[:3], def_fill='median')
            imputer.fit_transform(X)
            na_cnt = sum(X.nacnt())
            assert na_cnt > 0, 'expected some NAs, but found none'

        def scenario_11(X):
            """Assert functions with target_feature arg"""
            imputer = H2OSelectiveImputer(target_feature=X.columns[3], def_fill=[1, 1, 1])
            imputer.fit_transform(X)
            na_cnt = sum(X.nacnt())
            assert na_cnt > 0, 'expected some NAs, but found none'

        def scenario_12(X):
            """Assert functions with both feature_names and target_feature arg"""
            imputer = H2OSelectiveImputer(feature_names=X.columns[:3], target_feature=X.columns[3], def_fill=[2, 2, 2])
            imputer.fit_transform(X)
            na_cnt = sum(X.nacnt())
            assert na_cnt > 0, 'expected some NAs, but found none'

        def scenario_13(X):
            """Assert functions with list of imputes"""
            cols = [str(u) for u in X.columns]
            vals = ('mean', 1.5, 'median', 'median')
            fill = dict(zip(cols, vals))
            _basic_scenario(X, fill)

        # these are our test scenarios:
        scenarios = [
            scenario_1, scenario_2, scenario_3, scenario_4,
            scenario_5, scenario_6, scenario_7, scenario_8,
            scenario_9, scenario_10, scenario_11, scenario_12,
            scenario_13
        ]

        # since the imputer works in place, we have to do this for each scenario...
        for scenario in scenarios:
            try:
                M = new_h2o_frame(f.copy())
            except Exception as e:
                M = None

            if M is not None:
                scenario(M)
            else:
                pass

    def persist():
        f = F.copy()
        targ = iris.target
        targ = ['a' if x == 0 else 'b' if x == 1 else 'c' for x in targ]
        f['species'] = targ
        f = shuffle_dataframe(pd.concat([f, f, f, f, f], axis=0))  # times FIVE!

        try:
            Y = from_pandas(f)
        except Exception as e:
            Y = None

        if Y is not None:
            # test on H2OPipeline
            pipe = H2OPipeline([
                ('mcf', H2OMulticollinearityFilterer(threshold=0.65)),
                ('est', H2OGradientBoostingEstimator(ntrees=5))
            ],
                feature_names=Y.columns[:4],
                target_feature=Y.columns[4]
            )

            # fit and save
            pipe = pipe.fit(Y)
            the_path = 'pipe.pkl'
            pipe.save(the_path, warn_if_exists=False)
            assert os.path.exists(the_path)

            # assert we can predict with it even
            # when it's not reloaded yet (fix issue #7)
            pipe.predict(Y)

            # load and predict
            pipe = H2OPipeline.load(the_path)
            pred = pipe.predict(Y)

            # assert we can load and predict again (that attributes were not deleted)
            pipe = H2OPipeline.load(the_path)
            pred = pipe.predict(Y)

            # test one pipeline with all transformers and no estimator
            pipe = H2OPipeline([
                ('nzv', H2ONearZeroVarianceFilterer()),
                ('mc', H2OMulticollinearityFilterer(threshold=0.9))
            ],
                feature_names=Y.columns[:4],
                target_feature=Y.columns[4]
            )

            # fit and save
            pipe = pipe.fit(Y)
            the_path = 'pipe_2.pkl'
            pipe.save(the_path, warn_if_exists=False)
            assert os.path.exists(the_path)

            # assert we can predict with it even
            # when it's not reloaded yet (fix issue #7)
            pipe.transform(Y)

            # load and transform
            pipe = H2OPipeline.load(the_path)
            pred = pipe.transform(Y)

            # assert we can load and transform again (that attributes were not deleted)
            pipe = H2OPipeline.load(the_path)
            pred = pipe.transform(Y)

            # assert that after load, we can refit again if we want
            pipe.fit(Y)

            # test on grid with different types of estimators (keep em small)
            for i, HE in enumerate([H2OGradientBoostingEstimator(ntrees=5),
                                    H2ODeepLearningEstimator(epochs=1, hidden=[5, 5]),
                                    H2ORandomForestEstimator(ntrees=5),
                                    H2OGeneralizedLinearEstimator(family='multinomial'),
                                    H2ONaiveBayesEstimator()]):
                pipe2 = H2OPipeline([
                    ('mcf', H2OMulticollinearityFilterer(threshold=0.65)),
                    ('est', HE)
                ])

                hyp = {
                    'mcf__threshold': [0.80, 0.85, 0.90, 0.95]
                }

                grid = H2ORandomizedSearchCV(estimator=pipe2,
                                             param_grid=hyp,
                                             n_iter=1, cv=2,
                                             feature_names=Y.columns[:4],
                                             target_feature=Y.columns[4],
                                             scoring='accuracy_score')
                grid = grid.fit(Y)
                the_path = 'grid_%i.pkl' % i
                grid.save(the_path, warn_if_exists=False)
                assert os.path.exists(the_path)

                grid = H2ORandomizedSearchCV.load(the_path)
                grid.predict(Y)

                # assert we can load again (that attributes weren't deleted)
                grid = H2ORandomizedSearchCV.load(the_path)
                grid.predict(Y)

                # now assert that after load, we can fit again...
                grid.fit(Y)

            # can we perform the same thing for a grid search with no pipeline?
            est = H2OGradientBoostingEstimator(ntrees=5)
            hyp = {
                'ntrees': [5, 10]
            }
            grid = H2ORandomizedSearchCV(estimator=est,
                                         param_grid=hyp,
                                         n_iter=1, cv=2,
                                         feature_names=Y.columns[:4],
                                         target_feature=Y.columns[4],
                                         scoring='accuracy_score')

            grid = grid.fit(Y)
            the_path = 'grid_%i.pkl' % (i + 1)
            grid.save(the_path, warn_if_exists=False)
            assert os.path.exists(the_path)

            grid = H2ORandomizedSearchCV.load(the_path)
            grid.predict(Y)

            # assert we can load again (that attributes weren't deleted)
            grid = H2ORandomizedSearchCV.load(the_path)
            grid.predict(Y)

            # now assert that after load, we can fit again...
            grid.fit_predict(Y)

        else:
            pass

    def mem_est():
        if X is not None:
            h2o_frame_memory_estimate(X)

        else:
            pass

    def corr():
        if X is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                assert_fails(h2o_corr_plot, ValueError, **{'X': X, 'plot_type': 'bad_type'})

        else:
            pass

    def interactions():
        x_dict = {
            'a': [0, 0, 0, 1],
            'b': [1, 0, 0, 1],
            'c': [0, 1, 0, 1],
            'd': [1, 1, 1, 0]
        }

        X_pd = pd.DataFrame.from_dict(x_dict)[['a', 'b', 'c', 'd']]  # ordering

        try:
            frame = H2OFrame.from_python(X_pd, column_names=X_pd.columns.tolist())[1:, :]
        except Exception as e:
            frame = None

        if frame is not None:
            # try with no cols arg
            trans = H2OInteractionTermTransformer()
            X_trans = trans.fit_transform(frame)
            expected_names = ['a', 'b', 'c', 'd', 'a_b_I', 'a_c_I', 'a_d_I', 'b_c_I', 'b_d_I', 'c_d_I']
            assert all([str(i) == str(j) for i, j in zip(X_trans.columns, expected_names)])  # assert col names equal
            assert_array_equal(X_trans.as_data_frame(use_pandas=True).as_matrix(), np.array([
                [0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 1, 1, 0, 1, 0, 0]
            ]))

            # try with a custom function...
            def cust_add(a, b):
                return a + b

            trans = H2OInteractionTermTransformer(interaction_function=cust_add)
            X_trans = trans.fit_transform(frame).as_data_frame(use_pandas=True).as_matrix()
            assert_array_equal(X_trans, np.array([
                [0, 1, 0, 1, 1, 0, 1, 1, 2, 1],
                [0, 0, 1, 1, 0, 1, 1, 1, 1, 2],
                [0, 0, 0, 1, 0, 0, 1, 0, 1, 1],
                [1, 1, 1, 0, 2, 2, 1, 2, 1, 1]
            ]))

            # assert fails with a non-function arg
            assert_fails(H2OInteractionTermTransformer(interaction_function='a').fit, TypeError, frame)

            # test with just two cols
            # try with no cols arg
            trans = H2OInteractionTermTransformer(feature_names=['a', 'b'])
            X_trans = trans.fit_transform(frame)
            expected_names = ['a', 'b', 'c', 'd', 'a_b_I']
            assert all([i == j for i, j in zip(X_trans.columns, expected_names)])  # assert col names equal
            assert_array_equal(X_trans.as_data_frame(use_pandas=True).as_matrix(), np.array([
                [0, 1, 0, 1, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 1, 0],
                [1, 1, 1, 0, 1]
            ]))

            # test with just interaction terms returns
            trans = H2OInteractionTermTransformer(feature_names=['a', 'b'], only_return_interactions=True)
            X_trans = trans.fit_transform(frame)
            expected_names = sorted(['a', 'b', 'a_b_I'])
            actual_names = sorted([str(u) for u in X_trans.columns])
            assert all([expected_names[i] == actual_names[i] for i in range(len(expected_names))])

        else:
            pass

    def balance():
        if X is not None:
            # test that we can turn a frame's first col into a np array
            x = _pd_frame_to_np(X)  # just gets back the first col...
            assert isinstance(x, np.ndarray)

            # upload to cloud with the target
            f = F.copy()
            f['species'] = iris.target

            try:
                Y = from_pandas(f)
            except Exception as e:
                Y = None

            if Y is not None:
                # assert undersampling the balance changes nothing:
                b = H2OUndersamplingClassBalancer(target_feature='species').balance(Y)
                assert b.shape[0] == Y.shape[0]

                # do a real undersample
                x = Y[:60, :]  # 50 zeros, 10 ones
                b = H2OUndersamplingClassBalancer(target_feature='species', ratio=0.5).balance(x).as_data_frame(
                    use_pandas=True)
                assert b.shape[0] == 30
                cts = b.species.value_counts()
                assert cts[0] == 20
                assert cts[1] == 10

                # assert oversampling works
                y = Y[:105, :]
                d = H2OOversamplingClassBalancer(target_feature='species', ratio=1.0).balance(y).as_data_frame(
                    use_pandas=True)
                assert d.shape[0] == 150

                cts = d.species.value_counts()
                assert cts[0] == 50
                assert cts[1] == 50
                assert cts[2] == 50

        else:
            pass

    def encode():
        P = load_iris_df()
        P['letter'] = ['a' if x == 0 else 'b' if x == 1 else 'c' for x in P.Species]
        P.loc[0, 'letter'] = 'NA'

        try:
            Y = from_pandas(P)
        except Exception as e:
            Y = None

        if Y is not None:
            # default encoder
            encoder = H2OSafeOneHotEncoder(feature_names=['letter'])
            y = encoder.fit_transform(Y)

            assert not 'letter' in y.columns
            assert all([i in y.columns for i in ('letter.nan', 'letter.a', 'letter.b', 'letter.c')])

            # no drop encoder
            encoder = H2OSafeOneHotEncoder(feature_names=['letter'], drop_after_encoded=False)
            y = encoder.fit_transform(Y)

            assert 'letter' in y.columns
            assert y[['letter.a', 'letter.b', 'letter.c']].sum() == 149

        else:
            pass

    def feature_dropper():
        if X is not None:
            dropper = H2OFeatureDropper(feature_names=[X.columns[0]])
            Y = dropper.fit_transform(X)
            assert Y.shape[1] == X.shape[1] - 1
            assert not X.columns[0] in Y.columns

        else:
            pass

    def scale():
        def almost_eq(x, y, eps=1e-8):
            return abs(x - y) < eps

        if X is not None:
            scaler = H2OSelectiveScaler()
            trans = scaler.fit_transform(X)

            # assert mean zero
            means = flatten_all(trans.mean())
            assert all([almost_eq(x, 0) for x in means])

            # assert std one
            sds = flatten_all(trans.sd())
            assert all([almost_eq(x, 1) for x in sds])

            # assert X not affected
            X_means = flatten_all(X.mean())
            assert not all([almost_eq(x, 0) for x in X_means])
        else:
            pass

    def metrics():
        irs = F.copy()
        irs['species'] = iris.target
        irs['letters'] = ['a' if i == 0 else 'b' if i == 1 else 'c' for i in iris.target]
        irs['arbitrary'] = [3 for i in range(irs.shape[0])]

        try:
            Y = new_h2o_frame(irs)
        except Exception as e:
            Y = None

        if Y is not None:
            assert h2o_accuracy_score(Y['species'], Y['species']) == 1.0
            assert h2o_accuracy_score(Y['letters'], Y['letters']) == 1.0
            assert h2o_accuracy_score(Y['species'], Y['arbitrary']) == 0.0

            # test making the scorer
            accuracy_scorer = make_h2o_scorer(h2o_accuracy_score, Y['species'])
            assert accuracy_scorer.score(Y['species'], Y['species']) == 1.0

            # Test MAE, MSE
            reg_target = Y['sepal length (cm)']
            shifted_down = reg_target - 1

            # test on shifted down
            assert h2o_mean_absolute_error(reg_target, shifted_down) == 1.0
            assert h2o_median_absolute_error(reg_target, shifted_down) == 1.0
            assert h2o_mean_squared_error(reg_target, shifted_down) == 1.0

            # test on same
            assert h2o_mean_absolute_error(reg_target, reg_target) == 0.0
            assert h2o_median_absolute_error(reg_target, reg_target) == 0.0
            assert h2o_mean_squared_error(reg_target, reg_target) == 0.0

            # test R^2 on the same
            assert h2o_r2_score(reg_target, reg_target) == 1.0

            # test errors
            assert_fails(h2o_mean_squared_error, ValueError, Y['species'], Y['species'])
            assert_fails(h2o_accuracy_score, ValueError, reg_target, reg_target)
            assert_fails(make_h2o_scorer, TypeError, 'a', Y['species'])  # 'a' is not callable

        else:
            pass

    def encoder():
        irs = F.copy()
        irs['species'] = iris.target
        irs['target'] = [5 if i == 0 else 6 if i == 1 else 7 for i in iris.target]

        try:
            Y = new_h2o_frame(irs)
        except Exception as e:
            Y = None

        if Y is not None:
            encoder = H2OLabelEncoder()
            trans = encoder.fit_transform(Y['target'])

            assert (Y['species'] == trans).sum() == Y.shape[0]
            assert (Y['target'] == trans).sum() == 0  # assert not changed in place
        else:
            pass

    def bincount():
        col = pd.DataFrame(pd.Series([1, 1, 1, 3, 5], name='a'))
        neg = pd.DataFrame(pd.Series([-1, 1, 1, 3, 5], name='a'))
        flo = pd.DataFrame(pd.Series([1.1, 1.1, 1.1, 3.1, 5.1], name='a'))
        wt1 = [1, 1, 1, 1, 1]
        wt2 = [1, 0.5, 1, 1, 1]
        wp1 = pd.DataFrame(pd.Series(wt1, name='a'))
        wp2 = pd.DataFrame(pd.Series(wt2, name='a'))

        try:
            C = new_h2o_frame(col)
            N = new_h2o_frame(neg)
            F = new_h2o_frame(flo)
            W1 = new_h2o_frame(wp1)
            W2 = new_h2o_frame(wp2)
        except Exception as e:
            C = None
            W1 = None
            W2 = None

        if not any([i is None for i in (C, W1, W2)]):
            assert_array_equal(h2o_bincount(C), np.array([0, 3, 0, 1, 0, 1]))
            assert_array_equal(h2o_bincount(C, weights=wt1), np.array([0., 3., 0., 1., 0., 1.]))
            assert_array_equal(h2o_bincount(C, weights=wt2), np.array([0., 2.5, 0., 1., 0., 1.]))
            assert_array_equal(h2o_bincount(C, weights=wt1, minlength=7), np.array([0., 3., 0., 1., 0., 1., 0.]))
            assert_array_equal(h2o_bincount(C, weights=wt2, minlength=7), np.array([0., 2.5, 0., 1., 0., 1., 0.]))

            assert_array_equal(h2o_bincount(C), np.array([0, 3, 0, 1, 0, 1]))
            assert_array_equal(h2o_bincount(C, weights=W1), np.array([0., 3., 0., 1., 0., 1.]))
            assert_array_equal(h2o_bincount(C, weights=W2), np.array([0., 2.5, 0., 1., 0., 1.]))
            assert_array_equal(h2o_bincount(C, weights=W1, minlength=7), np.array([0., 3., 0., 1., 0., 1., 0.]))
            assert_array_equal(h2o_bincount(C, weights=W2, minlength=7), np.array([0., 2.5, 0., 1., 0., 1., 0.]))

            # test failures
            assert_fails(h2o_bincount, TypeError, col)
            assert_fails(h2o_bincount, ValueError, C, [0, 0, 0, 0])  # fail for dim mismatch
            assert_fails(h2o_bincount, ValueError, C, wt1, -1)  # negative minlength
            assert_fails(h2o_bincount, ValueError, N)  # one of them is negative
            assert_fails(h2o_bincount, ValueError, F)  # they're floats
        else:
            pass

    def load_frames():
        if X is not None:
            # all of these assertions pass locally, but not on travis
            # for some strange reason... 

            irs = load_iris_h2o(shuffle=True, include_tgt=True)
            assert irs.shape[1] == 5
            # assert irs.isfactor()[-1]

            bc = load_breast_cancer_h2o(shuffle=True, include_tgt=True)
            # assert bc.isfactor()[-1]

            bo = load_boston_h2o(shuffle=True, include_tgt=True)
            # assert not bo.isfactor()[-1]
        else:
            pass

    def isinteger_isfloat():
        irs = F.copy()
        irs['species'] = iris.target
        irs['letters'] = ['a' if i == 0 else 'b' if i == 1 else 'c' for i in iris.target]

        try:
            I = from_pandas(irs)
        except Exception as e:
            I = None

        if I is not None:
            assert is_integer(I['species'])
            assert is_float(I['sepal width (cm)'])
            assert not is_integer(I['letters'])
            assert not is_float(I['letters'])
        else:
            pass

    def shuffle():
        if X is not None:
            shuffle_h2o_frame(X)

    def f_test():
        if X is not None:
            try:
                t = load_iris_h2o()
            except Exception as e:
                return

            f,p = h2o_f_classif(t, t.columns[:-1], t.columns[-1])
            assert_array_almost_equal(f, np.array([119.26450218,47.3644614,1179.0343277,959.32440573]))
            assert_array_almost_equal(f, np.array([1.66966919e-31,1.32791652e-16,3.05197580e-91,4.37695696e-85]))

    # run the tests -- put new or commonly failing tests
    # up front as smoke tests. i.e., act, persist and grid
    persist()
    act_search()
    grid()
    encoder()
    bincount()
    metrics()
    multicollinearity()
    nzv()
    pipeline()
    anon_class()
    cv()
    split_tsts()
    sparse()
    impute()
    mem_est()
    corr()
    interactions()
    balance()
    encode()
    feature_dropper()
    scale()
    load_frames()
    isinteger_isfloat()
    shuffle()
    valid_use()
    feature_dropper_coverage()
