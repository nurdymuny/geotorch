"""
DavisTensor NN - Attention Layers
==================================

Geometric attention mechanisms.
"""

from __future__ import annotations
import numpy as np

from .module import Module


class GeometricAttention(Module):
    """
    Attention using geodesic distances instead of dot products.
    
    Standard attention: score = q·k / √d
    Geometric attention: score = -d(q, k)² / τ
    
    Where d is geodesic distance and τ is temperature.
    """
    
    def __init__(
        self,
        manifold,
        temperature: float = 1.0
    ):
        super().__init__()
        self.manifold = manifold
        self.temperature = temperature
    
    def forward(
        self,
        query,    # (B, N, D)
        key,      # (B, M, D)
        value     # (B, M, V)
    ):
        """
        Compute geometric attention.
        
        Returns:
            Weighted values, shape (B, N, V)
        """
        from ..core.storage import tensor
        
        q = query.numpy()
        k = key.numpy()
        v = value.numpy()
        
        B, N, D = q.shape
        M = k.shape[1]
        V = v.shape[-1]
        
        # Compute pairwise distances
        scores = np.zeros((B, N, M))
        for b in range(B):
            for i in range(N):
                for j in range(M):
                    d = self.manifold.distance(q[b, i], k[b, j])
                    scores[b, i, j] = -d ** 2 / self.temperature
        
        # Softmax
        scores_exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = scores_exp / scores_exp.sum(axis=-1, keepdims=True)
        
        # Weighted sum of values
        output = attn @ v  # (B, N, V)
        
        return tensor(output)
    
    def __repr__(self) -> str:
        return f"GeometricAttention({self.manifold}, τ={self.temperature})"


class DotProductAttention(Module):
    """
    Standard scaled dot-product attention.
    
    score = q·k / √d
    """
    
    def __init__(self, scale: float = None):
        super().__init__()
        self.scale = scale
    
    def forward(self, query, key, value):
        from ..core.storage import tensor
        
        q = query.numpy()
        k = key.numpy()
        v = value.numpy()
        
        d = q.shape[-1]
        scale = self.scale or np.sqrt(d)
        
        # Scaled dot product
        scores = (q @ np.swapaxes(k, -2, -1)) / scale
        
        # Softmax
        scores_exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = scores_exp / scores_exp.sum(axis=-1, keepdims=True)
        
        # Weighted sum
        output = attn @ v
        
        return tensor(output)


class HyperbolicAttention(Module):
    """
    Attention specifically designed for hyperbolic space.
    
    Uses hyperbolic distance and Möbius operations.
    """
    
    def __init__(self, manifold, temperature: float = 1.0):
        super().__init__()
        self.manifold = manifold
        self.temperature = temperature
    
    def forward(self, query, key, value):
        # For now, just use the generic GeometricAttention
        # Could be optimized with vectorized hyperbolic operations
        return GeometricAttention(
            self.manifold, 
            self.temperature
        ).forward(query, key, value)
