"""
===========================
Linear models
===========================

Provides sklearn-esque linear models
"""

from .poisson import *

__all__ = [s for s in dir() if not s.startswith("_")] ## Remove hiddens