from __future__ import print_function, absolute_import, division
import numpy as np
import pandas as pd
import warnings
from numpy.testing import (assert_array_equal, assert_almost_equal, assert_array_almost_equal)
from sklearn.datasets import load_iris
from skutil.utils import *
from skutil.utils.util import __min_log__, __max_exp__
from skutil.base import suppress_warnings
from skutil.utils.tests.utils import assert_fails


## Def data for testing
iris = load_iris()
X = load_iris_df(False)
X_no_targ = X.copy()

# ensure things work with a categorical feature
X['target'] = ['A' if x == 1 else 'B' if x == 2 else 'C' for x in iris.target]

# exact copy of second col
X['perfect'] = X[[1]]

def _check_equal(L1, L2):
    return len(L1) == len(L2) and sorted(L1) == sorted(L2)

def test_suppress():
    @suppress_warnings
    def raise_warning():
        warnings.warn('warning', UserWarning)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        raise_warning() # should be caught
        assert len(w) == 0, 'expected no warning to be thrown'

def test_safe_log_exp():
    assert log(0) == __min_log__
    assert exp(1000000) == __max_exp__

    l_res = log([1,2,3])
    e_res = exp([1,2,3])
    assert_array_almost_equal(l_res, np.array([ 0. , 0.69314718, 1.09861229]))
    assert_array_almost_equal(e_res, np.array([  2.71828183, 7.3890561 , 20.08553692]))

    assert isinstance(l_res, np.ndarray)
    assert isinstance(e_res, np.ndarray)

    # try something with no __iter__ attr
    assert_fails(log, ValueError, 'A')
    assert_fails(exp, ValueError, 'A')


def test_pd_stats():
    Y = load_iris_df()

    # add a float copy of species
    Y['species_float'] = Y.Species.astype('float')

    # add an object col
    Y['species_factor'] = ['a' if i == 0 else 'b' if i == 1 else 'c' for i in Y.Species]

    # test with all
    stats = pd_stats(Y, col_type='all')
    assert all([nm in stats.columns for nm in Y.columns])
    assert stats['species_float']['dtype'].startswith('int') # we assert it's considered an int

    # test with numerics
    stats = pd_stats(Y, col_type='numeric')
    assert not 'species_factor' in stats.columns
    assert stats.shape[1] == (Y.shape[1]-1)

    # test with object
    stats = pd_stats(Y, col_type='object')
    assert 'species_factor' in stats.columns
    assert stats.shape[1] == 1

    # add feature with one value, assert the ratio of min : max is NA string...
    Y['constant'] = np.zeros(Y.shape[0])
    stats = pd_stats(Y, col_type='all')
    assert all([nm in stats.columns for nm in Y.columns])
    assert stats['constant']['dtype'].startswith('int') # we assert it's considered an int
    assert stats.loc['min_max_class_ratio']['constant'] == '--'

    # test with bad col_type
    assert_fails(pd_stats, ValueError, Y, 'bad_type')



def test_corr():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        assert_fails(corr_plot, ValueError, **{'X':X_no_targ, 'plot_type':'bad_type'})

        pass
        # we'll lose coverage, but it'll save the windows from tying things up...
        #corr_plot(X_no_targ)
        #corr_plot(X_no_targ, kde=True, n_levels=3)

def test_bytes():
    # assert works for DF
    df_memory_estimate(X_no_targ)
    # assert fails for bad str
    assert_fails(df_memory_estimate, ValueError, **{'X':X_no_targ, 'unit':'pb'})


def test_flatten():
    a = [[[],3,4],['1','a'],[[[1]]],1,2]
    b = flatten_all(a)
    assert _check_equal(b, [3,4,'1','a',1,1,2])

def test_is_entirely_numeric():
    x = pd.DataFrame.from_records(data=iris.data)
    assert is_entirely_numeric(x)
    assert not is_entirely_numeric(X)

def test_is_numeric():
    assert is_numeric(1)
    assert is_numeric(1.)
    assert is_numeric(1L)
    assert is_numeric(np.int(1.0))
    assert is_numeric(np.float(1))
    assert is_numeric(1e-12)
    assert not is_numeric('a')


def test_get_numeric():
    a = pd.Series(['a','b','b'])
    b = pd.DataFrame(a)

    assert len(get_numeric(b)) == 0, 'expected empty'

def test_validate_on_non_df():
    x = iris.data
    validate_is_pd(x, None)

    # it will try to create a DF out of a String
    assert_fails(validate_is_pd, TypeError, 'asdf', 'asdf')

    # try on list of list and no cols
    x = [[1,2,3],[4,5,6],[7,8,9]]
    validate_is_pd(x, None)

def test_conf_matrix():
    a = [0,1,0,1,1]
    b = [0,1,1,1,0]

    df, ser = report_confusion_matrix(a, b)
    assert df.iloc[0,0] == 1
    assert df.iloc[0,1] == 1
    assert df.iloc[1,0] == 1
    assert df.iloc[1,1] == 2
    assert_almost_equal(ser['True Pos. Rate'], 0.666666666667)
    assert_almost_equal(ser['Diagnostic odds ratio'], 2.00000)

    # assert false yields None on series
    df, ser = report_confusion_matrix(a, b, False)
    assert ser is None

    # assert fails with > 2 classes
    a[0] = 2
    assert_fails(report_confusion_matrix, ValueError, a, b)

def test_load_iris_df():
    assert 'target' in load_iris_df(True, 'target').columns.values

