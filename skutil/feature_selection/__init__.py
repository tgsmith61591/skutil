"""
skutil.feature_selection provides a mechanism by which you can provide an
array of columns and subsequently drop columns that are deemed worthy of dropping
via the fit method within the _BaseFeatureSelector class.
The LinearCombinationFilter class is used to remove linear combinations of features.
All public classes within select.py extend the _BaseFeatureSelector class.
"""

from .select import *
from .combos import *

__all__ = [s for s in dir() if not s.startswith('_')]