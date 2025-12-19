"""
DavisTensor Autograd Module
===========================

Geometry-aware automatic differentiation.

THE KEY INSIGHT:
    Gradients are TANGENT VECTORS, not ambient vectors.
    
    In standard autograd (PyTorch):
        grad = ∂L/∂x ∈ R^n (ambient space)
        
    In Riemannian autograd (DavisTensor):
        grad = Riemannian gradient ∈ T_x M (tangent space at x)

This requires:
1. Automatic projection to tangent space
2. Parallel transport when combining gradients
3. Metric-aware preconditioning (optional: natural gradient)
"""

from .engine import GradientTape, backward
from .grad_fn import (
    GradFn,
    SavedContext,
    AddBackward,
    MulBackward,
    MatMulBackward,
    SumBackward,
    ExpBackward,
    LogBackward,
)
from .geometric_grad import (
    ManifoldExpBackward,
    ManifoldLogBackward,
    ManifoldDistanceBackward,
    check_gradients,
    natural_gradient,
)

__all__ = [
    # Engine
    'GradientTape',
    'backward',
    
    # Basic grad functions
    'GradFn',
    'SavedContext',
    'AddBackward',
    'MulBackward',
    'MatMulBackward',
    'SumBackward',
    'ExpBackward',
    'LogBackward',
    
    # Geometric grad functions
    'ManifoldExpBackward',
    'ManifoldLogBackward',
    'ManifoldDistanceBackward',
    'check_gradients',
    'natural_gradient',
]
