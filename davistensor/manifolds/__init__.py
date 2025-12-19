"""
DavisTensor Manifolds Module
============================

Riemannian manifold implementations.
"""

from .base import (
    Manifold,
    Euclidean,
    test_euclidean,
)

from .hyperbolic import (
    Hyperbolic,
    test_hyperbolic,
)

from .sphere import (
    Sphere,
    test_sphere,
)

from .spd import (
    SPD,
    test_spd,
)

from .product import (
    ProductManifold,
    HyperbolicSphere,
    HyperbolicEuclidean,
    MultiHyperbolic,
    MultiSphere,
    test_product,
)

__all__ = [
    # Base
    "Manifold",
    "Euclidean",
    "test_euclidean",
    
    # Hyperbolic
    "Hyperbolic",
    "test_hyperbolic",
    
    # Sphere
    "Sphere",
    "test_sphere",
    
    # SPD
    "SPD",
    "test_spd",
    
    # Product
    "ProductManifold",
    "HyperbolicSphere",
    "HyperbolicEuclidean",
    "MultiHyperbolic",
    "MultiSphere",
    "test_product",
]
