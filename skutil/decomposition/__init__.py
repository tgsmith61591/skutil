"""
skutil.decomposition provides sklearn decompositions
(`PCA`, `TruncatedSVD`) within the skutil API, i.e., 
allowing such transformers to operate on a select subset
of columns rather than the entire matrix.
"""

from .decompose import *

__all__ = [s for s in dir() if not s.startswith("_")]  # Remove hiddens
