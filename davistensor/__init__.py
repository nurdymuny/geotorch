"""DavisTensor: A geometry-native tensor library where tensors know their own geometry.

DavisTensor is a from-scratch tensor library where Riemannian geometry is not bolted on,
but built into the DNA. Every tensor knows what manifold it lives on. Every operation
respects curvature. Every gradient is automatically a tangent vector.
"""

__version__ = '0.1.0'

# Types
from .tensor import ManifoldTensor, TangentTensor, Scalar

# Manifolds
from .manifolds import Manifold, Euclidean

# Factory functions
from .tensor import randn, origin, tangent_randn, tangent_zeros

# Core (for advanced users)
from .core import TensorCore, Storage, Device, DType, GeometricType
from .core import zeros, ones, randn as core_randn, rand, tensor, from_numpy

__all__ = [
    # Version
    '__version__',
    
    # Types
    'ManifoldTensor',
    'TangentTensor',
    'Scalar',
    
    # Manifolds
    'Manifold',
    'Euclidean',
    
    # Factory functions
    'randn',
    'origin',
    'tangent_randn',
    'tangent_zeros',
    
    # Core
    'TensorCore',
    'Storage',
    'Device',
    'DType',
    'GeometricType',
    'zeros',
    'ones',
    'core_randn',
    'rand',
    'tensor',
    'from_numpy',
]
