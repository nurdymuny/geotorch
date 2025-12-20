"""GeoTorch: Manifold-Native Deep Learning Framework."""

from .manifold import Manifold
from .manifolds import Euclidean, Sphere, Hyperbolic
from .tensor import ManifoldTensor, TangentTensor

# Storage (Phase 4)
from . import storage
from .storage import DavisCache, GeoStorage

# Neural Network (Phase 3 + 4)
from . import nn
from .nn import (
    ManifoldParameter,
    GeoCachedAttention,
    FastGeoCachedAttention,
    GeoKVCache,
)

__version__ = "0.2.0"

__all__ = [
    # Core
    'Manifold',
    'Euclidean',
    'Sphere',
    'Hyperbolic',
    'ManifoldTensor',
    'TangentTensor',
    # Storage
    'storage',
    'DavisCache',
    'GeoStorage',
    # Neural Network
    'nn',
    'ManifoldParameter',
    'GeoCachedAttention',
    'FastGeoCachedAttention',
    'GeoKVCache',
]
