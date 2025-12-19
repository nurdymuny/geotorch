"""GeoTorch: Manifold-Native Deep Learning Framework.

A Riemannian deep learning framework that extends PyTorch with native support
for manifold-valued parameters, geodesic optimization, and geometric operations.
"""

__version__ = "0.1.0"

from .manifold import Manifold
from .tensor import ManifoldTensor, TangentTensor
from .manifolds.euclidean import Euclidean
from .manifolds.sphere import Sphere
from .manifolds.hyperbolic import Hyperbolic

__all__ = [
    "Manifold",
    "ManifoldTensor",
    "TangentTensor",
    "Euclidean",
    "Sphere",
    "Hyperbolic",
]
