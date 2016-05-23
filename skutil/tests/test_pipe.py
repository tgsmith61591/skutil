import numpy as np
from numpy.testing import (assert_array_equal, assert_almost_equal, assert_array_almost_equal)
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split, KFold
from sklearn.grid_search import RandomizedSearchCV
from skutil.preprocessing import *
from skutil.decomposition import *
from skutil.feature_selection import *
from skutil.utils import report_grid_score_detail
from scipy.stats import randint, uniform
import pandas as pd


## Def data for testing
iris = load_iris()
X = pd.DataFrame(data=iris.data, columns=iris.feature_names)

def test_pipeline_basic():
	pipe = Pipeline([
			('selector', FeatureRetainer(cols=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)'])),
			('scaler', SelectiveScaler()),
			('model', RandomForestClassifier())
		])

	pipe.fit(X, iris.target)


def test_pipeline_complex():
	pipe = Pipeline([
			('selector', FeatureRetainer(cols=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)'])),
			('scaler', SelectiveScaler()),
			('boxcox', BoxCoxTransformer()),
			('pca', SelectivePCA()),
			('svd', SelectiveTruncatedSVD()),
			('model', RandomForestClassifier())
		])

	pipe.fit(X, iris.target)

def test_grid():
	# get our train/test
	X_train, X_test, y_train, y_test = train_test_split(X, iris.target, train_size=0.75, random_state=42)

	# default CV does not shuffle, so we define our own
	custom_cv = KFold(n=y_train.shape[0], n_folds=5, shuffle=True, random_state=42)

	# build a pipeline
	pipe = Pipeline([
        ('collinearity', MulticollinearityFilterer(threshold=0.85)),
        ('scaler'      , SelectiveScaler()),
        ('boxcox'      , BoxCoxTransformer()),
        ('pca'         , PCA(n_components=0.9)),
        ('model'       , RandomForestClassifier(n_jobs=-1))
    ])

    # let's define a set of hyper-parameters over which to search
	hp = {
	    'collinearity__threshold' : uniform(loc=.8, scale=.15),
	    'collinearity__method'    : ['pearson','kendall','spearman'],
	    'scaler__scaler'          : [StandardScaler(), RobustScaler()],
	    'pca__n_components'       : uniform(loc=.75, scale=.2),
	    'pca__whiten'             : [True, False],
	    'model__n_estimators'     : randint(5,100),
	    'model__max_depth'        : randint(2,25),
	    'model__min_samples_leaf' : randint(1,15),
	    'model__max_features'     : uniform(loc=.5, scale=.5),
	    'model__max_leaf_nodes'   : randint(10,75)
	}

	# define the gridsearch
	search = RandomizedSearchCV(pipe, hp,
	                            n_iter=3, # just to test it even works
	                            scoring='accuracy',
	                            cv=custom_cv,
	                            random_state=42)

	# fit the search
	search.fit(X_train, y_train)

	# test the report
	the_report = report_grid_score_detail(search, charts=False)
	# do nothing with it

