"""
skutil.metrics houses the pairwise kernel matrix functionality that
is built using Cython which behaves similar to scikit-learns pairwise
behavior.
"""
from .pairwise import *
from ._act import *

# __all__ = [s for s in dir() if not s.startswith("_")]  # Remove hiddens
