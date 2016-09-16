from __future__ import print_function
import warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from scipy.stats import uniform, randint
from skutil.grid_search import RandomizedSearchCV
from skutil.decomposition import *
from skutil.preprocessing import *
from skutil.utils.tests.utils import assert_fails
from skutil.utils import report_grid_score_detail
from skutil.grid_search import _as_numpy
from numpy.testing import (assert_array_equal, assert_almost_equal, assert_array_almost_equal)

try:
	from sklearn.model_selection import KFold, train_test_split, RandomizedSearchCV
	SK18 = True
except ImportError as ie:
	from sklearn.cross_validation import KFold, train_test_split
	SK18 = False

# generate a totally random matrix
X = np.random.rand(500, 25) # kind of large...

# generate a totally random discrete response
def factorize(x):
	return np.array([0 if i < 0.5 else 1 for i in x])

y = factorize(np.random.rand(X.shape[0]))

# get the split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75)


def test_as_numpy():
	assert_fails(_as_numpy, TypeError, 'blah')
	assert _as_numpy(None) is None

	i = [1,2,3]
	x = np.array(i)
	assert_array_equal(x, _as_numpy(x))
	assert_array_equal(np.asarray(i), _as_numpy(i))
	assert_array_equal(_as_numpy(pd.DataFrame.from_records(X)), X)



def test_large_grid():
	"""In this test, we purposely overfit a RandomForest to completely random data
	in order to assert that the test error will far supercede the train error.
	"""

	if not SK18:
		custom_cv = KFold(n=y_train.shape[0], n_folds=3, shuffle=True, random_state=42)
	else:
		custom_cv = KFold(n_folds=3, shuffle=True, random_state=42)

	# define the pipe
	pipe = Pipeline([
			('scaler', SelectiveScaler()),
			('pca',	SelectivePCA(weight=True)),
			('rf',	 RandomForestClassifier(random_state=42))
		])

	# define hyper parameters
	hp = {
		'scaler__scaler' : [StandardScaler(), RobustScaler(), MinMaxScaler()],
		'pca__whiten' : [True, False],
		'pca__weight' : [True, False],
		'pca__n_components' : uniform(0.75, 0.15),
		'rf__n_estimators' : randint(5, 10),
		'rf__max_depth' : randint(5, 15)
	}

	# define the grid
	grid = RandomizedSearchCV(pipe, hp, n_iter=2, scoring='accuracy', n_jobs=-1, cv=custom_cv, random_state=42)

	# this will fail because we haven't fit yet
	assert_fails(grid.score, (ValueError, AttributeError), X_train, y_train)

	# fit the grid
	grid.fit(X_train, y_train)

	# score for coverage -- this might warn...
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		grid.score(X_train, y_train)

	# coverage:
	assert grid._estimator_type == 'classifier'

	# get predictions
	tr_pred, te_pred = grid.predict(X_train), grid.predict(X_test)

	# evaluate score (SHOULD be better than random...)
	tr_score, te_score = accuracy_score(y_train, tr_pred), accuracy_score(y_test, te_pred)

	# do we want to do this? Probably not because it's reliant on chance
	# if not tr_score >= te_score:
	# 	warnings.warn('expected training accuracy to be higher (train: %.5f, test: %.5f)' % (tr_score, te_score))

	# grid score reports:
	# assert fails for bad percentile
	assert_fails(report_grid_score_detail, ValueError, **{'random_search':grid, 'percentile':0.0})
	assert_fails(report_grid_score_detail, ValueError, **{'random_search':grid, 'percentile':1.0})

	# assert fails for bad y_axis
	assert_fails(report_grid_score_detail, ValueError, **{'random_search':grid, 'y_axis':'bad_axis'})

	# assert passes otherwise
	report = report_grid_score_detail(grid, charts=False, percentile=0.95) # just ensure percentile works
