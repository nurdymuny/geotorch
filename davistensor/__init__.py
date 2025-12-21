"""
DavisTensor: A Geometry-Native Tensor Library
==============================================

DavisTensor is a from-scratch tensor library where Riemannian geometry is built into the DNA.
Every tensor knows what manifold it lives on. Every operation respects curvature.

Example:
    >>> import davistensor as dt
    >>> x = dt.randn(10, manifold=dt.Euclidean(10))
    >>> v = dt.tangent_randn(x)
    >>> y = x + v  # Exponential map!
    >>> d = x.distance(y)  # Geodesic distance
"""

__version__ = "0.1.0"

# Core types
from .tensor import ManifoldTensor, TangentTensor, Scalar

# Manifolds
from .manifolds import Manifold, Euclidean

# Low-level core (for advanced users)
from .core import (
    TensorCore,
    Storage,
    Device,
    DType,
    GeometricType,
    float32,
    float64,
)

import numpy as np
from typing import Optional, Union
from .core.storage import CPU, _create_tensor


def randn(*shape, manifold: Manifold, dtype: Optional[DType] = None, device: Optional[Device] = None) -> ManifoldTensor:
    """
    Create random point(s) on a manifold.
    
    Args:
        *shape: Batch dimensions (manifold dimension added automatically)
        manifold: The Riemannian manifold
        dtype: Data type (default: float32)
        device: Device (default: CPU)
    
    Returns:
        ManifoldTensor with random point(s)
    
    Example:
        >>> x = dt.randn(5, 10, manifold=dt.Euclidean(64))  # Shape (5, 10, 64)
    """
    from .core.storage import float32
    
    if dtype is None:
        dtype = float32
    if device is None:
        device = CPU
    
    core = manifold.random_point(*shape, dtype=dtype, device=device)
    return ManifoldTensor(core, manifold=manifold)


def origin(*shape, manifold: Manifold, dtype: Optional[DType] = None, device: Optional[Device] = None) -> ManifoldTensor:
    """
    Create origin/identity point(s) on a manifold.
    
    Args:
        *shape: Batch dimensions
        manifold: The Riemannian manifold
        dtype: Data type (default: float32)
        device: Device (default: CPU)
    
    Returns:
        ManifoldTensor at origin
    
    Example:
        >>> o = dt.origin(manifold=dt.Euclidean(10))
    """
    from .core.storage import float32
    
    if dtype is None:
        dtype = float32
    if device is None:
        device = CPU
    
    core = manifold.origin(*shape, dtype=dtype, device=device)
    return ManifoldTensor(core, manifold=manifold)


def tangent_randn(base_point: ManifoldTensor, scale: float = 1.0) -> TangentTensor:
    """
    Create random tangent vector at a base point.
    
    Args:
        base_point: Point on manifold
        scale: Scale factor for random vector
    
    Returns:
        Random tangent vector at base_point
    
    Example:
        >>> x = dt.randn(manifold=dt.Euclidean(10))
        >>> v = dt.tangent_randn(x)
    """
    # For Euclidean space, tangent space is the whole space
    shape = base_point.shape
    data = np.random.randn(*shape).astype(base_point.dtype.numpy_dtype) * scale
    
    return TangentTensor(data, base_point=base_point, manifold=base_point.manifold)


def tangent_zeros(base_point: ManifoldTensor) -> TangentTensor:
    """
    Create zero tangent vector at a base point.
    
    Args:
        base_point: Point on manifold
    
    Returns:
        Zero tangent vector at base_point
    
    Example:
        >>> x = dt.randn(manifold=dt.Euclidean(10))
        >>> v = dt.tangent_zeros(x)
    """
    shape = base_point.shape
    data = np.zeros(shape, dtype=base_point.dtype.numpy_dtype)
    
    return TangentTensor(data, base_point=base_point, manifold=base_point.manifold)


__all__ = [
    # Version
    "__version__",
    
    # Main classes
    "ManifoldTensor",
    "TangentTensor",
    "Scalar",
    
    # Manifolds
    "Manifold",
    "Euclidean",
    
    # Factory functions
    "randn",
    "origin",
    "tangent_randn",
    "tangent_zeros",
    
    # Core types (advanced)
    "TensorCore",
    "Storage",
    "Device",
    "DType",
    "GeometricType",
    "float32",
    "float64",
]
