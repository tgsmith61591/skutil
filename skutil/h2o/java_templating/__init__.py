"""
Java main class templating for deploying POJOs easily.
"""
from .templater import *

__all__ = [s for s in dir() if not s.startswith("_")]  # Remove hiddens