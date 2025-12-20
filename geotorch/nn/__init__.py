"""Neural network components for GeoTorch."""

from .parameter import ManifoldParameter
from .layers import (
    ManifoldLinear,
    GeodesicEmbedding,
    FrechetMean,
    GeometricAttention,
    MultiHeadGeometricAttention,
    GeodesicLayer,
    HyperbolicLinear,
    HyperbolicEmbedding,
    SphericalLinear,
    SphericalEmbedding,
)
from .attention import GeoCachedAttention, FastGeoCachedAttention
from .kv_cache import GeoKVCache, StreamingGeoKVCache

__all__ = [
    # Parameter
    'ManifoldParameter',
    # Layers
    'ManifoldLinear',
    'GeodesicEmbedding',
    'FrechetMean',
    'GeometricAttention',
    'MultiHeadGeometricAttention',
    'GeodesicLayer',
    # Convenience wrappers
    'HyperbolicLinear',
    'HyperbolicEmbedding',
    'SphericalLinear',
    'SphericalEmbedding',
    # Cached Attention (Phase 4)
    'GeoCachedAttention',
    'FastGeoCachedAttention',
    # KV-Cache (Phase 4)
    'GeoKVCache',
    'StreamingGeoKVCache',
]
