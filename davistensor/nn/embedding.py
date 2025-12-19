"""
DavisTensor NN - Embedding Layers
==================================

Standard and manifold embedding tables.
"""

from __future__ import annotations
import numpy as np

from .module import Module, Parameter, ManifoldParameter


class Embedding(Module):
    """
    Standard Euclidean embedding table.
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Initialize with normal distribution
        self.weight = Parameter(np.random.randn(num_embeddings, embedding_dim) * 0.01)
    
    def forward(self, indices: np.ndarray):
        """
        Args:
            indices: Integer indices, any shape
        
        Returns:
            Embeddings, shape (*indices.shape, embedding_dim)
        """
        from ..core.storage import tensor
        
        embeddings = self.weight.numpy()[indices]
        return tensor(embeddings)
    
    def __repr__(self) -> str:
        return f"Embedding({self.num_embeddings}, {self.embedding_dim})"


class ManifoldEmbedding(Module):
    """
    Embedding table where embeddings live on a manifold.
    
    Unlike standard Embedding which returns Euclidean vectors,
    this returns points on a Riemannian manifold.
    """
    
    def __init__(
        self,
        num_embeddings: int,
        manifold,
        scale: float = 0.01
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.manifold = manifold
        
        # Initialize as random points on manifold
        initial = manifold.random_point(num_embeddings) * scale
        self.weight = ManifoldParameter(initial, manifold)
    
    def forward(self, indices):
        """
        Args:
            indices: Integer indices (np.ndarray or TensorCore), any shape
        
        Returns:
            Manifold points, shape (*indices.shape, ambient_dim)
        """
        from ..core.storage import tensor, TensorCore
        
        # Convert to numpy integer array if needed
        if isinstance(indices, TensorCore):
            idx = indices.numpy().astype(int)
        elif isinstance(indices, np.ndarray):
            idx = indices.astype(int)
        else:
            idx = np.array(indices, dtype=int)
        
        embeddings = self.weight.numpy()[idx]
        result = tensor(embeddings)
        result.manifold = self.manifold
        return result
    
    def __repr__(self) -> str:
        return f"ManifoldEmbedding({self.num_embeddings}, {self.manifold})"
