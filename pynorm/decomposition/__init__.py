"""
===========================
Statistical transformations
===========================

Provides sklearn-esque decompositions
"""

from .decompose import *

__all__ = [s for s in dir() if not s.startswith("_")] ## Remove hiddens
