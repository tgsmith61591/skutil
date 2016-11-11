from __future__ import division, absolute_import, print_function
import sklearn
from .base import overrides
from .utils.fixes import (_validate_X, _validate_y,
                          _check_param_grid, _as_numpy, _CVScoreTuple)

__all__ = [
    'GridSearchCV',
    'RandomizedSearchCV'
]

# deprecation in sklearn 0.18
if sklearn.__version__ >= '0.18':
    import sklearn.model_selection as ms


    class GridSearchCV(ms.GridSearchCV):
        """Exhaustive search over specified parameter values for an estimator.
        This class is a skutil fix of the sklearn 0.18 GridSearchCV module, and allows
        use with SelectiveMixins and other skutil classes that don't interact so kindly
        with other sklearn 0.18 structures (i.e. when ``as_df`` is True in many transformers,
        predicting on a column vector from a pd.DataFrame will cause issues in sklearn).

        Parameters
        ----------

        estimator : estimator object.
            This is assumed to implement the scikit-learn estimator interface.
            Either estimator needs to provide a ``score`` function,
            or ``scoring`` must be passed.

        param_grid : dict or list of dictionaries
            Dictionary with parameters names (string) as keys and lists of
            parameter settings to try as values, or a list of such
            dictionaries, in which case the grids spanned by each dictionary
            in the list are explored. This enables searching over any sequence
            of parameter settings.

        scoring : string, callable or None, default=None
            A string (see model evaluation documentation) or
            a scorer callable object / function with signature
            ``scorer(estimator, X, y)``.
            If ``None``, the ``score`` method of the estimator is used.

        fit_params : dict, optional
            Parameters to pass to the fit method.

        n_jobs : int, default=1
            Number of jobs to run in parallel.

        pre_dispatch : int, or string, optional
            Controls the number of jobs that get dispatched during parallel
            execution. Reducing this number can be useful to avoid an
            explosion of memory consumption when more jobs get dispatched
            than CPUs can process. This parameter can be:
                - None, in which case all the jobs are immediately
                  created and spawned. Use this for lightweight and
                  fast-running jobs, to avoid delays due to on-demand
                  spawning of the jobs
                - An int, giving the exact number of total jobs that are
                  spawned
                - A string, giving an expression as a function of n_jobs,
                  as in '2*n_jobs'

        iid : boolean, default=True
            If True, the data is assumed to be identically distributed across
            the folds, and the loss minimized is the total loss per sample,
            and not the mean loss across the folds.

        cv : int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:
              - None, to use the default 3-fold cross validation,
              - integer, to specify the number of folds in a ``(Stratified)KFold``,
              - An object to be used as a cross-validation generator.
              - An iterable yielding train, test splits.
            For integer/None inputs, if the estimator is a classifier and ``y`` is
            either binary or multiclass, ``StratifiedKFold`` is used. In all
            other cases, ``KFold`` is used.

        refit : boolean, default=True
            Refit the best estimator with the entire dataset.
            If "False", it is impossible to make predictions using
            this GridSearchCV instance after fitting.

        verbose : integer
            Controls the verbosity: the higher, the more messages.

        error_score : 'raise' (default) or numeric
            Value to assign to the score if an error occurs in estimator fitting.
            If set to 'raise', the error is raised. If a numeric value is given,
            FitFailedWarning is raised. This parameter does not affect the refit
            step, which will always raise the error.

        return_train_score : boolean, default=True
            If ``'False'``, the ``cv_results_`` attribute will not include training
            scores.

        
        Attributes
        ----------

        cv_results_ : dict of numpy (masked) ndarrays
            A dict with keys as column headers and values as columns, that can be
            imported into a pandas ``DataFrame``.

            For instance, the following table

            +------------+-----------+------------+-----------------+---+---------+
            |param_kernel|param_gamma|param_degree|split0_test_score|...|rank_....|
            +============+===========+============+=================+===+=========+
            |  'poly'    |     --    |      2     |        0.8      |...|    2    |
            +------------+-----------+------------+-----------------+---+---------+
            |  'poly'    |     --    |      3     |        0.7      |...|    4    |
            +------------+-----------+------------+-----------------+---+---------+
            |  'rbf'     |     0.1   |     --     |        0.8      |...|    3    |
            +------------+-----------+------------+-----------------+---+---------+
            |  'rbf'     |     0.2   |     --     |        0.9      |...|    1    |
            +------------+-----------+------------+-----------------+---+---------+

            will be represented by a ``cv_results_`` dict of:

                {
                    'param_kernel': masked_array(data = ['poly', 'poly', 'rbf', 'rbf'],
                                                 mask = [False False False False]...)
                    'param_gamma': masked_array(data = [-- -- 0.1 0.2],
                                                mask = [ True  True False False]...),
                    'param_degree': masked_array(data = [2.0 3.0 -- --],
                                                 mask = [False False  True  True]...),
                    'split0_test_score'  : [0.8, 0.7, 0.8, 0.9],
                    'split1_test_score'  : [0.82, 0.5, 0.7, 0.78],
                    'mean_test_score'    : [0.81, 0.60, 0.75, 0.82],
                    'std_test_score'     : [0.02, 0.01, 0.03, 0.03],
                    'rank_test_score'    : [2, 4, 3, 1],
                    'split0_train_score' : [0.8, 0.9, 0.7],
                    'split1_train_score' : [0.82, 0.5, 0.7],
                    'mean_train_score'   : [0.81, 0.7, 0.7],
                    'std_train_score'    : [0.03, 0.03, 0.04],
                    'mean_fit_time'      : [0.73, 0.63, 0.43, 0.49],
                    'std_fit_time'       : [0.01, 0.02, 0.01, 0.01],
                    'mean_score_time'    : [0.007, 0.06, 0.04, 0.04],
                    'std_score_time'     : [0.001, 0.002, 0.003, 0.005],
                    'params'             : [{'kernel': 'poly', 'degree': 2}, ...],
                }

            NOTE that the key ``'params'`` is used to store a list of parameter
            settings dict for all the parameter candidates. The ``mean_fit_time``, 
            ``std_fit_time``, ``mean_score_time`` and ``std_score_time`` are all in seconds.

        best_estimator_ : estimator
            Estimator that was chosen by the search, i.e. estimator
            which gave highest score (or smallest loss if specified)
            on the left out data. Not available if refit=False.

        best_score_ : float
            Score of best_estimator on the left out data.

        best_params_ : dict
            Parameter setting that gave the best results on the hold out data.

        best_index_ : int
            The index (of the ``cv_results_`` arrays) which corresponds to the best
            candidate parameter setting.
            The dict at ``search.cv_results_['params'][search.best_index_]`` gives
            the parameter setting for the best model, that gives the highest
            mean score (``search.best_score_``).

        scorer_ : function
            Scorer function used on the held out data to choose the best
            parameters for the model.

        n_splits_ : int
            The number of cross-validation splits (folds/iterations).
        """
        
        @overrides(ms.GridSearchCV)
        def fit(self, X, y=None, groups=None):
            """Run fit with all sets of parameters.

            Parameters
            ----------

            X : array-like, shape=(n_samples, n_features)
                Training vector, where n_samples is the number of samples and
                n_features is the number of features.

            y : array-like, shape=(n_samples,) or (n_samples, n_output), optional (default=None)
                Target relative to X for classification or regression;
                None for unsupervised learning.

            groups : array-like, shape=(n_samples,), optional (default=None)
                Group labels for the samples used while splitting the dataset into
                train/test set.
            """
            return super(GridSearchCV, self).fit(X, _as_numpy(y), groups)


    class RandomizedSearchCV(ms.RandomizedSearchCV):
        """Randomized search on hyper parameters.
        This class is a skutil fix of the sklearn 0.18 RandomizedSearchCV module, and allows
        use with SelectiveMixins and other skutil classes that don't interact so kindly
        with other sklearn 0.18 structures (i.e. when ``as_df`` is True in many transformers,
        predicting on a column vector from a pd.DataFrame will cause issues in sklearn).

        Parameters
        ----------

        estimator : estimator object.
            A object of that type is instantiated for each grid point.
            This is assumed to implement the scikit-learn estimator interface.
            Either estimator needs to provide a ``score`` function,
            or ``scoring`` must be passed.

        param_distributions : dict
            Dictionary with parameters names (string) as keys and distributions
            or lists of parameters to try. Distributions must provide a ``rvs``
            method for sampling (such as those from scipy.stats.distributions).
            If a list is given, it is sampled uniformly.

        n_iter : int, default=10
            Number of parameter settings that are sampled. n_iter trades
            off runtime vs quality of the solution.

        scoring : string, callable or None, default=None
            A string (see model evaluation documentation) or
            a scorer callable object / function with signature
            ``scorer(estimator, X, y)``.
            If ``None``, the ``score`` method of the estimator is used.

        fit_params : dict, optional
            Parameters to pass to the fit method.

        n_jobs : int, default=1
            Number of jobs to run in parallel.

        pre_dispatch : int, or string, optional
            Controls the number of jobs that get dispatched during parallel
            execution. Reducing this number can be useful to avoid an
            explosion of memory consumption when more jobs get dispatched
            than CPUs can process. This parameter can be:
                - None, in which case all the jobs are immediately
                  created and spawned. Use this for lightweight and
                  fast-running jobs, to avoid delays due to on-demand
                  spawning of the jobs
                - An int, giving the exact number of total jobs that are
                  spawned
                - A string, giving an expression as a function of n_jobs,
                  as in '2*n_jobs'

        iid : boolean, default=True
            If True, the data is assumed to be identically distributed across
            the folds, and the loss minimized is the total loss per sample,
            and not the mean loss across the folds.

        cv : int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:
              - None, to use the default 3-fold cross validation,
              - integer, to specify the number of folds in a ``(Stratified)KFold``,
              - An object to be used as a cross-validation generator.
              - An iterable yielding train, test splits.
            For integer/None inputs, if the estimator is a classifier and ``y`` is
            either binary or multiclass, ``StratifiedKFold`` is used. In all
            other cases, ``KFold`` is used.

        refit : boolean, default=True
            Refit the best estimator with the entire dataset.
            If "False", it is impossible to make predictions using
            this RandomizedSearchCV instance after fitting.

        verbose : integer
            Controls the verbosity: the higher, the more messages.

        random_state : int or RandomState
            Pseudo random number generator state used for random uniform sampling
            from lists of possible values instead of scipy.stats distributions.

        error_score : 'raise' (default) or numeric
            Value to assign to the score if an error occurs in estimator fitting.
            If set to 'raise', the error is raised. If a numeric value is given,
            FitFailedWarning is raised. This parameter does not affect the refit
            step, which will always raise the error.

        return_train_score : boolean, default=True
            If ``'False'``, the ``cv_results_`` attribute will not include training
            scores.


        Attributes
        ----------

        cv_results_ : dict of numpy (masked) ndarrays
            A dict with keys as column headers and values as columns, that can be
            imported into a pandas ``DataFrame``.

            For instance the following table:

            +--------------+-------------+-------------------+---+---------------+
            | param_kernel | param_gamma | split0_test_score |...|rank_test_score|
            +==============+=============+===================+===+===============+
            |    'rbf'     |     0.1     |        0.8        |...|       2       |
            +--------------+-------------+-------------------+---+---------------+
            |    'rbf'     |     0.2     |        0.9        |...|       1       |
            +--------------+-------------+-------------------+---+---------------+
            |    'rbf'     |     0.3     |        0.7        |...|       1       |
            +--------------+-------------+-------------------+---+---------------+

            will be represented by a ``cv_results_`` dict of:

                {
                    'param_kernel' : masked_array(data = ['rbf', 'rbf', 'rbf'],
                                                  mask = False),
                    'param_gamma'  : masked_array(data = [0.1 0.2 0.3], mask = False),
                    'split0_test_score'  : [0.8, 0.9, 0.7],
                    'split1_test_score'  : [0.82, 0.5, 0.7],
                    'mean_test_score'    : [0.81, 0.7, 0.7],
                    'std_test_score'     : [0.02, 0.2, 0.],
                    'rank_test_score'    : [3, 1, 1],
                    'split0_train_score' : [0.8, 0.9, 0.7],
                    'split1_train_score' : [0.82, 0.5, 0.7],
                    'mean_train_score'   : [0.81, 0.7, 0.7],
                    'std_train_score'    : [0.03, 0.03, 0.04],
                    'mean_fit_time'      : [0.73, 0.63, 0.43, 0.49],
                    'std_fit_time'       : [0.01, 0.02, 0.01, 0.01],
                    'mean_score_time'    : [0.007, 0.06, 0.04, 0.04],
                    'std_score_time'     : [0.001, 0.002, 0.003, 0.005],
                    'params' : [{'kernel' : 'rbf', 'gamma' : 0.1}, ...],
                }

            NOTE that the key ``'params'`` is used to store a list of parameter
            settings dict for all the parameter candidates. The ``mean_fit_time``, 
            ``std_fit_time``, ``mean_score_time`` and ``std_score_time`` are all in seconds.

        best_estimator_ : estimator
            Estimator that was chosen by the search, i.e. estimator
            which gave highest score (or smallest loss if specified)
            on the left out data. Not available if refit=False.

        best_score_ : float
            Score of best_estimator on the left out data.

        best_params_ : dict
            Parameter setting that gave the best results on the hold out data.

        best_index_ : int
            The index (of the ``cv_results_`` arrays) which corresponds to the best
            candidate parameter setting.
            The dict at ``search.cv_results_['params'][search.best_index_]`` gives
            the parameter setting for the best model, that gives the highest
            mean score (``search.best_score_``).

        scorer_ : function
            Scorer function used on the held out data to choose the best
            parameters for the model.

        n_splits_ : int
            The number of cross-validation splits (folds/iterations).
        """

        @overrides(ms.RandomizedSearchCV)
        def fit(self, X, y=None, groups=None):
            """Run fit on the estimator with randomly drawn parameters.

            Parameters
            ----------

            X : array-like, shape=(n_samples, n_features)
                Training vector, where n_samples is the number of samples and
                n_features is the number of features.

            y : array-like, shape=(n_samples,) or (n_samples, n_output), optional (default=None)
                Target relative to X for classification or regression;
                None for unsupervised learning.

            groups : array-like, shape=(n_samples,), optional (default=None)
                Group labels for the samples used while splitting the dataset into
                train/test set.
            """
            return super(RandomizedSearchCV, self).fit(X, _as_numpy(y), groups)
else:
    """
    sklearn deprecates the GridSearch and cross validation API we know and
    love in 0.18, thus, we only define these methods if we're using < 0.18.
    Otherwise, we'll use their default. These are defined in skutil.utils.fixes
    """
    from .utils import fixes


    class GridSearchCV(fixes._SK17GridSearchCV):
        """Exhaustive search over specified parameter values for an estimator.
        This class is a skutil fix of the sklearn 0.17 GridSearchCV module, and allows
        use with SelectiveMixins and other skutil classes that don't interact so kindly
        with other sklearn 0.17 structures.

        Parameters
        ----------

        estimator : estimator object.
            A object of that type is instantiated for each grid point.
            This is assumed to implement the scikit-learn estimator interface.
            Either estimator needs to provide a ``score`` function,
            or ``scoring`` must be passed.

        param_grid : dict or list of dictionaries
            Dictionary with parameters names (string) as keys and lists of
            parameter settings to try as values, or a list of such
            dictionaries, in which case the grids spanned by each dictionary
            in the list are explored. This enables searching over any sequence
            of parameter settings.

        scoring : string, callable or None, default=None
            A string (see model evaluation documentation) or
            a scorer callable object / function with signature
            ``scorer(estimator, X, y)``.
            If ``None``, the ``score`` method of the estimator is used.

        fit_params : dict, optional
            Parameters to pass to the fit method.

        n_jobs : int, default=1
            Number of jobs to run in parallel.

        pre_dispatch : int, or string, optional
            Controls the number of jobs that get dispatched during parallel
            execution. Reducing this number can be useful to avoid an
            explosion of memory consumption when more jobs get dispatched
            than CPUs can process. This parameter can be:
                - None, in which case all the jobs are immediately
                  created and spawned. Use this for lightweight and
                  fast-running jobs, to avoid delays due to on-demand
                  spawning of the jobs
                - An int, giving the exact number of total jobs that are
                  spawned
                - A string, giving an expression as a function of n_jobs,
                  as in '2*n_jobs'

        iid : boolean, default=True
            If True, the data is assumed to be identically distributed across
            the folds, and the loss minimized is the total loss per sample,
            and not the mean loss across the folds.

        cv : int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:
            - None, to use the default 3-fold cross-validation,
            - integer, to specify the number of folds.
            - An object to be used as a cross-validation generator.
            - An iterable yielding train/test splits.
            For integer/None inputs, if the estimator is a classifier and ``y`` is
            either binary or multiclass, `sklearn.model_selection.StratifiedKFold` is used. 
            In all other cases, `sklearn.model_selection.KFold` is used.

        refit : boolean, default=True
            Refit the best estimator with the entire dataset.
            If "False", it is impossible to make predictions using
            this GridSearchCV instance after fitting.

        verbose : integer
            Controls the verbosity: the higher, the more messages.

        error_score : 'raise' (default) or numeric
            Value to assign to the score if an error occurs in estimator fitting.
            If set to 'raise', the error is raised. If a numeric value is given,
            FitFailedWarning is raised. This parameter does not affect the refit
            step, which will always raise the error.


        Attributes
        ----------

        grid_scores_ : list of named tuples
            Contains scores for all parameter combinations in param_grid.
            Each entry corresponds to one parameter setting.
            Each named tuple has the attributes:
                * ``parameters``, a dict of parameter settings
                * ``mean_validation_score``, the mean score over the
                  cross-validation folds
                * ``cv_validation_scores``, the list of scores for each fold

        best_estimator_ : estimator
            Estimator that was chosen by the search, i.e. estimator
            which gave highest score (or smallest loss if specified)
            on the left out data. Not available if refit=False.

        best_score_ : float
            Score of best_estimator on the left out data.

        best_params_ : dict
            Parameter setting that gave the best results on the hold out data.

        scorer_ : function
            Scorer function used on the held out data to choose the best
            parameters for the model.
        """
        pass


    class RandomizedSearchCV(fixes._SK17RandomizedSearchCV):
        """Randomized search on hyper parameters. This class is a skutil fix of the sklearn 
        0.17 RandomizedSearchCV module, and allows use with SelectiveMixins and other skutil 
        classes that don't interact so kindly with other sklearn 0.17 structures.

        Parameters
        ----------

        estimator : estimator object.
            A object of that type is instantiated for each grid point.
            This is assumed to implement the scikit-learn estimator interface.
            Either estimator needs to provide a ``score`` function,
            or ``scoring`` must be passed.

        param_distributions : dict
            Dictionary with parameters names (string) as keys and distributions
            or lists of parameters to try. Distributions must provide a ``rvs``
            method for sampling (such as those from scipy.stats.distributions).
            If a list is given, it is sampled uniformly.

        n_iter : int, default=10
            Number of parameter settings that are sampled. n_iter trades
            off runtime vs quality of the solution.

        scoring : string, callable or None, default=None
            A string (see model evaluation documentation) or
            a scorer callable object / function with signature
            ``scorer(estimator, X, y)``.
            If ``None``, the ``score`` method of the estimator is used.

        fit_params : dict, optional
            Parameters to pass to the fit method.

        n_jobs : int, default=1
            Number of jobs to run in parallel.

        pre_dispatch : int, or string, optional
            Controls the number of jobs that get dispatched during parallel
            execution. Reducing this number can be useful to avoid an
            explosion of memory consumption when more jobs get dispatched
            than CPUs can process. This parameter can be:
                - None, in which case all the jobs are immediately
                  created and spawned. Use this for lightweight and
                  fast-running jobs, to avoid delays due to on-demand
                  spawning of the jobs
                - An int, giving the exact number of total jobs that are
                  spawned
                - A string, giving an expression as a function of n_jobs,
                  as in '2*n_jobs'

        iid : boolean, default=True
            If True, the data is assumed to be identically distributed across
            the folds, and the loss minimized is the total loss per sample,
            and not the mean loss across the folds.

        cv : int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:
            - None, to use the default 3-fold cross-validation,
            - integer, to specify the number of folds.
            - An object to be used as a cross-validation generator.
            - An iterable yielding train/test splits.
            For integer/None inputs, if the estimator is a classifier and ``y`` is
            either binary or multiclass, `sklearn.model_selection.StratifiedKFold` is used. 
            In all other cases, `sklearn.model_selection.KFold` is used.

        refit : boolean, default=True
            Refit the best estimator with the entire dataset.
            If "False", it is impossible to make predictions using
            this RandomizedSearchCV instance after fitting.

        verbose : integer
            Controls the verbosity: the higher, the more messages.

        random_state : int or RandomState
            Pseudo random number generator state used for random uniform sampling
            from lists of possible values instead of scipy.stats distributions.

        error_score : 'raise' (default) or numeric
            Value to assign to the score if an error occurs in estimator fitting.
            If set to 'raise', the error is raised. If a numeric value is given,
            FitFailedWarning is raised. This parameter does not affect the refit
            step, which will always raise the error.


        Attributes
        ----------

        grid_scores_ : list of named tuples
            Contains scores for all parameter combinations in param_grid.
            Each entry corresponds to one parameter setting.
            Each named tuple has the attributes:
                * ``parameters``, a dict of parameter settings
                * ``mean_validation_score``, the mean score over the
                  cross-validation folds
                * ``cv_validation_scores``, the list of scores for each fold

        best_estimator_ : estimator
            Estimator that was chosen by the search, i.e. estimator
            which gave highest score (or smallest loss if specified)
            on the left out data. Not available if refit=False.

        best_score_ : float
            Score of best_estimator on the left out data.

        best_params_ : dict
            Parameter setting that gave the best results on the hold out data.
        """
        pass
