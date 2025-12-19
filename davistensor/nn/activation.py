"""
DavisTensor NN - Activation Functions
======================================

Standard and manifold-aware activations.
"""

from __future__ import annotations
import numpy as np

from .module import Module


class ReLU(Module):
    """Standard ReLU activation."""
    
    def forward(self, x):
        from ..core.storage import tensor
        return tensor(np.maximum(x.numpy(), 0))


class TangentReLU(Module):
    """
    ReLU in tangent space.
    
    For manifold points:
    1. Map to tangent space at origin
    2. Apply ReLU
    3. Map back to manifold
    """
    
    def __init__(self, manifold):
        super().__init__()
        self.manifold = manifold
    
    def forward(self, x):
        from ..core.storage import tensor
        
        x_np = x.numpy()
        origin = self.manifold.origin()
        
        # To tangent space
        if x_np.ndim == 1:
            v = self.manifold.log(origin, x_np)
            v_relu = np.maximum(v, 0)
            y = self.manifold.exp(origin, v_relu)
        else:
            # Batched
            vs = np.stack([self.manifold.log(origin, xi) for xi in x_np])
            vs_relu = np.maximum(vs, 0)
            y = np.stack([self.manifold.exp(origin, vi) for vi in vs_relu])
        
        result = tensor(y)
        result.manifold = self.manifold
        return result
    
    def __repr__(self) -> str:
        return f"TangentReLU({self.manifold})"


class Sigmoid(Module):
    """Sigmoid activation."""
    
    def forward(self, x):
        from ..core.storage import tensor
        return tensor(1 / (1 + np.exp(-x.numpy())))


class Tanh(Module):
    """Tanh activation."""
    
    def forward(self, x):
        from ..core.storage import tensor
        return tensor(np.tanh(x.numpy()))


class Softmax(Module):
    """Softmax activation."""
    
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        from ..core.storage import tensor
        
        x_np = x.numpy()
        exp_x = np.exp(x_np - x_np.max(axis=self.dim, keepdims=True))
        return tensor(exp_x / exp_x.sum(axis=self.dim, keepdims=True))


class GELU(Module):
    """Gaussian Error Linear Unit."""
    
    def forward(self, x):
        from ..core.storage import tensor
        x_np = x.numpy()
        return tensor(0.5 * x_np * (1 + np.tanh(np.sqrt(2 / np.pi) * (x_np + 0.044715 * x_np**3))))


class LeakyReLU(Module):
    """Leaky ReLU activation."""
    
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope
    
    def forward(self, x):
        from ..core.storage import tensor
        x_np = x.numpy()
        return tensor(np.where(x_np > 0, x_np, self.negative_slope * x_np))


class ELU(Module):
    """Exponential Linear Unit."""
    
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x):
        from ..core.storage import tensor
        x_np = x.numpy()
        return tensor(np.where(x_np > 0, x_np, self.alpha * (np.exp(x_np) - 1)))
