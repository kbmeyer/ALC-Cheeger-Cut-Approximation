# clustering/__init__.py
"""
Clustering adapters and spectral embedding utilities.
"""

from .alc import ALCAdapter
from .kmeans import KMeansAdapter
from .spectral import spectral_embed

__all__ = ["ALCAdapter", "KMeansAdapter", "spectral_embed"]
