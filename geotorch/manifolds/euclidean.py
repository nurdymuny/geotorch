"""Euclidean space manifold."""

import torch
from torch import Tensor
from ..manifold import Manifold


class Euclidean(Manifold):
    """Euclidean space ℝⁿ.
    
    This is a flat manifold where all geometric operations reduce to
    standard linear operations. It's included for compatibility with
    PyTorch's standard optimization.
    
    Args:
        n: Dimension of the Euclidean space
    """
    
    def __init__(self, n: int):
        self._dim = n
    
    @property
    def dim(self) -> int:
        """Intrinsic dimension (same as ambient dimension for Euclidean space)."""
        return self._dim
    
    def exp(self, p: Tensor, v: Tensor) -> Tensor:
        """Exponential map in Euclidean space is just addition.
        
        Args:
            p: Point in ℝⁿ
            v: Tangent vector (any vector in ℝⁿ)
        
        Returns:
            p + v
        """
        return p + v
    
    def log(self, p: Tensor, q: Tensor) -> Tensor:
        """Logarithmic map in Euclidean space is just subtraction.
        
        Args:
            p: Base point in ℝⁿ
            q: Target point in ℝⁿ
        
        Returns:
            q - p
        """
        return q - p
    
    def parallel_transport(self, v: Tensor, p: Tensor, q: Tensor) -> Tensor:
        """Parallel transport in Euclidean space is identity.
        
        In flat space, parallel transport doesn't change the vector.
        
        Args:
            v: Tangent vector at p
            p: Source point
            q: Destination point
        
        Returns:
            v (unchanged)
        """
        return v
    
    def distance(self, p: Tensor, q: Tensor) -> Tensor:
        """Euclidean distance between points.
        
        Args:
            p: First point
            q: Second point
        
        Returns:
            ||p - q||
        """
        return torch.linalg.norm(p - q, dim=-1)
    
    def project(self, x: Tensor) -> Tensor:
        """Projection onto Euclidean space is identity.
        
        Args:
            x: Point in ambient space (already in ℝⁿ)
        
        Returns:
            x (unchanged)
        """
        return x
    
    def project_tangent(self, p: Tensor, v: Tensor) -> Tensor:
        """Tangent space projection in Euclidean space is identity.
        
        Args:
            p: Point on manifold
            v: Vector in ambient space
        
        Returns:
            v (unchanged)
        """
        return v
    
    def random_point(self, *shape, device=None, dtype=None) -> Tensor:
        """Generate random point(s) from standard normal distribution.
        
        Args:
            *shape: Shape of the output (batch dimensions)
            device: PyTorch device
            dtype: PyTorch dtype
        
        Returns:
            Random point(s) in ℝⁿ
        """
        full_shape = shape + (self._dim,)
        return torch.randn(full_shape, device=device, dtype=dtype)
