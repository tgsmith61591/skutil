"""
skutil.odr is a python port of R's QR Decomposition backend (legacy Fortran subroutines)
"""

from .dqrutl import *

__all__ = [s for s in dir() if not s.startswith("_")]  # Remove hiddens
