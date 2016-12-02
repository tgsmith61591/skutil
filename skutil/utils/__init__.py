"""
skutil.utils provides common utilitarian functionality for the skutil library.
skutil.utils.fixes adds adaptations for bridging the scikit-learn 0.17 to 0.18 behavior.
skutil.utils.metaestimators adapts scikit-learns metaestimator for more specific use of skutil.
"""

from .fixes import *
from .util import *
from .tests.utils import assert_fails

__all__ = [s for s in dir() if not s.startswith('_')]
