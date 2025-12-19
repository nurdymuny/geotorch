"""
DavisTensor Neural Network Module
==================================

Geometry-aware neural network layers.

Key insight: Standard neural network operations (linear, attention, pooling)
have natural Riemannian generalizations.

- Linear: maps through tangent space
- Attention: uses geodesic distance instead of dot product
- Pooling: Fréchet mean instead of arithmetic mean
- Normalization: Riemannian centering and scaling
"""

from .module import Module, Parameter, ManifoldParameter
from .linear import Linear, GeodesicLinear, ManifoldMLR
from .embedding import Embedding, ManifoldEmbedding
from .pooling import MeanPool, FrechetMeanPool
from .attention import GeometricAttention
from .normalization import ManifoldBatchNorm
from .activation import ReLU, TangentReLU
from .container import Sequential

__all__ = [
    # Base
    'Module',
    'Parameter',
    'ManifoldParameter',
    
    # Linear
    'Linear',
    'GeodesicLinear',
    'ManifoldMLR',
    
    # Embedding
    'Embedding',
    'ManifoldEmbedding',
    
    # Pooling
    'MeanPool',
    'FrechetMeanPool',
    
    # Attention
    'GeometricAttention',
    
    # Normalization
    'ManifoldBatchNorm',
    
    # Activation
    'ReLU',
    'TangentReLU',
    
    # Container
    'Sequential',
]
