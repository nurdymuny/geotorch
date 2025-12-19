"""
GeoTorch Storage: O(1) Geometric Retrieval
==========================================

Implements the Davis-Wilson framework for constant-time similarity search
via topological binning on Riemannian manifolds.

Key insight: Γ(x) ≠ Γ(y) ⟹ d(x,y) ≥ κ
(Different bins = geometrically distinguishable)

Based on geodesic_storage.py patterns adapted for GeoTorch manifolds.
"""

from .cache import DavisCache, SpatialHash
from .storage import GeoStorage, StorageItem
from .binning import (
    curvature_binning,
    spatial_binning,
    lsh_binning,
    morton_encode,
)

__all__ = [
    # Core
    'DavisCache',
    'GeoStorage',
    'StorageItem',
    'SpatialHash',
    # Binning methods
    'curvature_binning',
    'spatial_binning', 
    'lsh_binning',
    'morton_encode',
]
