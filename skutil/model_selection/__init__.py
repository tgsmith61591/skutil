"""
In scikit-learn 0.18, sklearn.grid_search was deprecated. Since
skutil handles the deprecation issues in skutil.utils.fixes, the
skutil.model_selection module merely provides the same import
functionality as sklearn 0.18, so sklearn users can seamlessly
migrate to skutil for grid_search imports.
"""

from skutil.grid_search import *

__all__ = [
    'GridSearchCV',
    'RandomizedSearchCV'
]
