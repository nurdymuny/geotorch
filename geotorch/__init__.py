"""GeoTorch: Manifold-Native Deep Learning Framework."""

from .manifold import Manifold
from .manifolds import Euclidean, Sphere, Hyperbolic
from .tensor import ManifoldTensor, TangentTensor

__version__ = "0.1.0"

__all__ = [
    'Manifold',
    'Euclidean',
    'Sphere',
    'Hyperbolic',
    'ManifoldTensor',
    'TangentTensor',
]
