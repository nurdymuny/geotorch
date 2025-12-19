"""
DavisTensor NN - Normalization Layers
======================================

Manifold-aware normalization.
"""

from __future__ import annotations
import numpy as np

from .module import Module, Parameter


class ManifoldBatchNorm(Module):
    """
    Batch normalization on a manifold.
    
    1. Center: compute Fréchet mean, map to tangent space
    2. Scale: normalize variance in tangent space
    3. Shift/scale: learnable parameters in tangent space
    4. Map back to manifold
    """
    
    def __init__(
        self,
        manifold,
        momentum: float = 0.1,
        eps: float = 1e-5
    ):
        super().__init__()
        self.manifold = manifold
        self.momentum = momentum
        self.eps = eps
        
        # Running statistics
        self._running_mean = manifold.origin()
        
        # Learnable scale and bias (in tangent space)
        self.scale = Parameter(np.ones(manifold.ambient_dim))
        self.bias = Parameter(np.zeros(manifold.ambient_dim))
    
    def forward(self, x):
        """
        Args:
            x: Points on manifold, shape (B, ambient_dim)
        
        Returns:
            Normalized points, shape (B, ambient_dim)
        """
        from ..core.storage import tensor
        
        x_np = x.numpy()
        B = x_np.shape[0]
        
        if self._training:
            # Compute batch Fréchet mean
            weights = np.ones(B) / B
            mean = self._running_mean.copy()
            for _ in range(5):
                tangent_sum = np.zeros_like(mean)
                for i in range(B):
                    tangent_sum += self.manifold.log(mean, x_np[i]) / B
                mean = self.manifold.exp(mean, tangent_sum)
            
            # Update running mean (geodesic moving average)
            self._running_mean = self.manifold.geodesic(
                self._running_mean, mean, self.momentum
            )
        else:
            mean = self._running_mean
        
        # Map to tangent space at mean
        tangents = np.stack([self.manifold.log(mean, x_np[i]) for i in range(B)])
        
        # Normalize in tangent space
        if self._training:
            std = tangents.std(axis=0) + self.eps
        else:
            std = 1.0  # Use fixed scale at test time
        
        normalized = tangents / std
        
        # Apply learnable scale and bias
        transformed = normalized * self.scale.numpy() + self.bias.numpy()
        
        # Map back to manifold
        output = np.stack([self.manifold.exp(mean, transformed[i]) for i in range(B)])
        
        result = tensor(output)
        result.manifold = self.manifold
        return result
    
    def __repr__(self) -> str:
        return f"ManifoldBatchNorm({self.manifold})"


class LayerNorm(Module):
    """
    Standard layer normalization (Euclidean).
    """
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        self.scale = Parameter(np.ones(normalized_shape))
        self.bias = Parameter(np.zeros(normalized_shape))
    
    def forward(self, x):
        from ..core.storage import tensor
        
        x_np = x.numpy()
        
        # Normalize over last dimension
        mean = x_np.mean(axis=-1, keepdims=True)
        var = x_np.var(axis=-1, keepdims=True)
        
        normalized = (x_np - mean) / np.sqrt(var + self.eps)
        output = normalized * self.scale.numpy() + self.bias.numpy()
        
        return tensor(output, requires_grad=x.requires_grad)
    
    def __repr__(self) -> str:
        return f"LayerNorm({self.normalized_shape})"
