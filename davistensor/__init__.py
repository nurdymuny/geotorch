"""
DavisTensor: A Geometry-Native Tensor Library
==============================================

Where tensors know their own geometry.

>>> import davistensor as dt
>>> x = dt.randn(64, manifold=dt.Hyperbolic(64))
>>> y = dt.randn(64, manifold=dt.Hyperbolic(64))
>>> d = x.distance(y)  # Geodesic distance
>>> v = x.log(y)       # Tangent vector from x to y
>>> z = x + v          # exp(x, v) = y
"""

__version__ = "0.2.0"

# Core types
from .core.storage import (
    TensorCore,
    Storage,
    Device,
    DeviceType,
    DType,
    CPU,
    float32,
    float64,
    int32,
    int64,
    GeometricType,
)

# Core factory functions (raw)
from .core.storage import (
    zeros as _zeros,
    ones as _ones,
    randn as _randn,
    rand as _rand,
    tensor as _tensor,
    from_numpy,
)

# Manifolds
from .manifolds.base import Manifold, Euclidean
from .manifolds.hyperbolic import Hyperbolic
from .manifolds.sphere import Sphere
from .manifolds.spd import SPD
from .manifolds.product import (
    ProductManifold,
    HyperbolicSphere,
    HyperbolicEuclidean,
    MultiHyperbolic,
    MultiSphere,
)

# Type-safe tensors
from .tensor import (
    ManifoldTensor,
    TangentTensor,
    Scalar,
    randn,
    origin,
    tangent_randn,
    tangent_zeros,
)

# Autograd
from .autograd import (
    GradientTape,
    backward,
    GradFn,
    SavedContext,
)

# Neural network layers
from . import nn


# Convenience: raw tensor creation (for Euclidean / non-geometric use)
def zeros(*shape, dtype=float32, device=CPU, requires_grad=False):
    """Create tensor of zeros."""
    return _zeros(*shape, dtype=dtype, device=device, requires_grad=requires_grad)


def ones(*shape, dtype=float32, device=CPU, requires_grad=False):
    """Create tensor of ones."""
    return _ones(*shape, dtype=dtype, device=device, requires_grad=requires_grad)


def tensor(data, dtype=None, device=CPU, requires_grad=False):
    """Create tensor from data."""
    return _tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)


__all__ = [
    # Version
    "__version__",
    
    # Core types
    "TensorCore",
    "Storage", 
    "Device",
    "DeviceType",
    "DType",
    "CPU",
    "float32",
    "float64",
    "int32",
    "int64",
    "GeometricType",
    
    # Factory functions
    "zeros",
    "ones",
    "tensor",
    "from_numpy",
    "randn",
    "origin",
    "tangent_randn",
    "tangent_zeros",
    
    # Manifolds
    "Manifold",
    "Euclidean",
    "Hyperbolic",
    "Sphere",
    "SPD",
    "ProductManifold",
    "HyperbolicSphere",
    "HyperbolicEuclidean",
    "MultiHyperbolic",
    "MultiSphere",
    
    # Type-safe tensors
    "ManifoldTensor",
    "TangentTensor",
    "Scalar",
    
    # Autograd
    "GradientTape",
    "backward",
    "GradFn",
    "SavedContext",
    
    # Neural network module
    "nn",
]
