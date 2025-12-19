"""Euclidean space manifold."""

import torch
from torch import Tensor
from ..manifold import Manifold


class Euclidean(Manifold):
    """
    Euclidean space R^n with flat metric.
    
    This is the trivial manifold where:
    - exp_p(v) = p + v
    - log_p(q) = q - p
    - distance(p, q) = ||q - p||
    
    Useful for PyTorch compatibility and as a baseline.
    
    Args:
        n: Dimension of the space
    """
    
    def __init__(self, n: int):
        self.n = n
    
    @property
    def dim(self) -> int:
        """Intrinsic dimension of the manifold."""
        return self.n
    
    def exp(self, p: Tensor, v: Tensor) -> Tensor:
        """
        Exponential map: move from p along geodesic with velocity v.
        
        For Euclidean space, this is simply addition: exp_p(v) = p + v
        
        Args:
            p: Point on manifold, shape (..., n)
            v: Tangent vector at p, shape (..., n)
        
        Returns:
            Point on manifold after geodesic flow
        """
        return p + v
    
    def log(self, p: Tensor, q: Tensor) -> Tensor:
        """
        Logarithmic map: tangent vector at p pointing toward q.
        
        For Euclidean space, this is simply subtraction: log_p(q) = q - p
        
        Args:
            p: Base point on manifold
            q: Target point on manifold
        
        Returns:
            Tangent vector v such that exp(p, v) = q
        """
        return q - p
    
    def parallel_transport(self, v: Tensor, p: Tensor, q: Tensor) -> Tensor:
        """
        Parallel transport tangent vector v from T_pM to T_qM.
        
        In Euclidean space, tangent spaces are all identified, so parallel
        transport is the identity: PT(v) = v
        
        Args:
            v: Tangent vector at p
            p: Source point
            q: Destination point
        
        Returns:
            Tangent vector at q with same geometric properties as v
        """
        return v
    
    def distance(self, p: Tensor, q: Tensor) -> Tensor:
        """
        Geodesic distance between points.
        
        For Euclidean space, this is the standard Euclidean norm: ||q - p||
        
        Args:
            p: First point, shape (..., n)
            q: Second point, shape (..., n)
        
        Returns:
            Distance, shape (...)
        """
        return torch.linalg.norm(q - p, dim=-1)
    
    def project(self, x: Tensor) -> Tensor:
        """
        Project ambient space point onto manifold.
        
        For Euclidean space, every point is already on the manifold.
        
        Args:
            x: Point in ambient space
        
        Returns:
            Closest point on manifold (identity)
        """
        return x
    
    def project_tangent(self, p: Tensor, v: Tensor) -> Tensor:
        """
        Project ambient vector onto tangent space at p.
        
        For Euclidean space, tangent space is the entire space.
        
        Args:
            p: Point on manifold
            v: Vector in ambient space
        
        Returns:
            Component of v in T_pM (identity)
        """
        return v
    
    def random_point(self, *shape, device=None, dtype=None) -> Tensor:
        """
        Generate random point(s) on manifold.
        
        Samples from standard normal distribution.
        
        Args:
            *shape: Shape of points to generate (excluding final dimension n)
            device: Device to create tensor on
            dtype: Data type of tensor
        
        Returns:
            Random point(s) on manifold
        """
        if not shape:
            shape = ()
        return torch.randn(*shape, self.n, device=device, dtype=dtype)
    
    def random_tangent(self, p: Tensor) -> Tensor:
        """
        Generate random tangent vector at p.
        
        For Euclidean space, samples from standard normal distribution.
        
        Args:
            p: Point on manifold
        
        Returns:
            Random tangent vector at p
        """
        return torch.randn_like(p)
