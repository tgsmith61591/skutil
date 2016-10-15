from __future__ import division, absolute_import, print_function
import sklearn
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
        """Had to wrap GridSearchCV in order to allow
        fitting a series as Y.
        """
        def fit(self, X, y=None):
            super(GridSearchCV, self).fit(X, _as_numpy(y))

    class RandomizedSearchCV(ms.RandomizedSearchCV):
        """Had to wrap RandomizedSearchCV in order to allow
        fitting a series as Y.
        """
        def fit(self, X, y=None):
            super(RandomizedSearchCV, self).fit(X, _as_numpy(y))
else:
    """
    sklearn deprecates the GridSearch and cross validation API we know and
    love in 0.18, thus, we only define these methods if we're using < 0.18.
    Otherwise, we'll use their default. These are defined in skutil.utils.fixes
    """
    from .utils import fixes

    class GridSearchCV(fixes._SK17GridSearchCV):
        pass

    class RandomizedSearchCV(fixes._SK17RandomizedSearchCV):
        pass
