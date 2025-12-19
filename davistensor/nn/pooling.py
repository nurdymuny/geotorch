"""
DavisTensor NN - Pooling Layers
================================

Standard and geometric pooling operations.
"""

from __future__ import annotations
from typing import Optional
import numpy as np

from .module import Module


class MeanPool(Module):
    """
    Standard arithmetic mean pooling.
    """
    
    def __init__(self, dim: int = -2):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        from ..core.storage import tensor
        return tensor(x.numpy().mean(axis=self.dim))


class FrechetMeanPool(Module):
    """
    Pooling via Fréchet mean on a manifold.
    
    Instead of arithmetic mean, computes the Riemannian center of mass:
        μ = argmin_p Σ_i w_i d(p, x_i)²
    
    This is the natural generalization of mean pooling to manifolds.
    """
    
    def __init__(self, manifold, dim: int = -2, n_iters: int = 10):
        super().__init__()
        self.manifold = manifold
        self.dim = dim
        self.n_iters = n_iters
    
    def forward(
        self, 
        x, 
        weights: Optional[np.ndarray] = None
    ):
        """
        Args:
            x: Points to pool, shape (..., N, ambient_dim) where N is pool dim
            weights: Optional weights, shape (..., N)
        
        Returns:
            Pooled point, shape (..., ambient_dim)
        """
        from ..core.storage import tensor
        
        x_np = x.numpy()
        
        # Move pooling dim to position -2 if needed
        if self.dim != -2:
            x_np = np.moveaxis(x_np, self.dim, -2)
        
        batch_shape = x_np.shape[:-2]
        N = x_np.shape[-2]
        
        if weights is None:
            weights = np.ones((*batch_shape, N)) / N
        else:
            weights = weights / weights.sum(axis=-1, keepdims=True)
        
        # Compute Fréchet mean via gradient descent
        def compute_mean(points, w):
            """Compute single Fréchet mean."""
            # Initialize with first point
            mean = points[0].copy()
            
            for _ in range(self.n_iters):
                # Weighted sum of log maps
                tangent_sum = np.zeros_like(mean)
                for i in range(len(points)):
                    v = self.manifold.log(mean, points[i])
                    tangent_sum += w[i] * v
                
                # Check convergence
                if self.manifold.norm(mean, tangent_sum) < 1e-6:
                    break
                
                # Update
                mean = self.manifold.exp(mean, tangent_sum)
            
            return mean
        
        # Handle batching
        if len(batch_shape) == 0:
            result = compute_mean(x_np, weights)
        else:
            flat_x = x_np.reshape(-1, N, x_np.shape[-1])
            flat_w = weights.reshape(-1, N)
            
            means = []
            for b in range(flat_x.shape[0]):
                means.append(compute_mean(flat_x[b], flat_w[b]))
            
            result = np.stack(means).reshape(*batch_shape, -1)
        
        out = tensor(result)
        out.manifold = self.manifold
        return out
    
    def __repr__(self) -> str:
        return f"FrechetMeanPool({self.manifold}, dim={self.dim})"


class MaxPool(Module):
    """
    Max pooling (Euclidean).
    """
    
    def __init__(self, dim: int = -2):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        from ..core.storage import tensor
        return tensor(x.numpy().max(axis=self.dim))


class SumPool(Module):
    """
    Sum pooling.
    """
    
    def __init__(self, dim: int = -2):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        from ..core.storage import tensor
        return tensor(x.numpy().sum(axis=self.dim))
