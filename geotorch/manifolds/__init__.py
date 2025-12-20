"""Manifold implementations."""

from .euclidean import Euclidean
from .sphere import Sphere
from .hyperbolic import Hyperbolic
from .spd import SPD, LogEuclideanSPD, SPDTransform, SPDBiMap, SPDReLU, SPDLogEig
from .davis import DavisManifold, DavisMetricLearner
from .product import (
    ProductManifold,
    HyperbolicSphere,
    HyperbolicEuclidean,
    SphereEuclidean,
    MultiHyperbolic,
    MultiSphere,
    ProductEmbedding,
    ProductLinear
)

__all__ = [
    # Basic manifolds
    'Euclidean',
    'Sphere',
    'Hyperbolic',
    # SPD manifold
    'SPD',
    'LogEuclideanSPD',
    'SPDTransform',
    'SPDBiMap',
    'SPDReLU',
    'SPDLogEig',
    # Learned metric
    'DavisManifold',
    'DavisMetricLearner',
    # Product manifolds
    'ProductManifold',
    'HyperbolicSphere',
    'HyperbolicEuclidean',
    'SphereEuclidean',
    'MultiHyperbolic',
    'MultiSphere',
    'ProductEmbedding',
    'ProductLinear',
]
