"""Base abstract class for Riemannian manifolds."""

from abc import ABC, abstractmethod
import torch
from torch import Tensor


class Manifold(ABC):
    """
    Abstract base class for Riemannian manifolds.
    
    A Riemannian manifold (M, g) is a smooth space equipped with a metric tensor g
    that defines distances, angles, and geodesics (shortest paths).
    """
    
    @property
    @abstractmethod
    def dim(self) -> int:
        """Intrinsic dimension of the manifold."""
        ...
    
    @abstractmethod
    def exp(self, p: Tensor, v: Tensor) -> Tensor:
        """
        Exponential map: move from p along geodesic with velocity v.
        
        Args:
            p: Point on manifold, shape (..., ambient_dim)
            v: Tangent vector at p, shape (..., ambient_dim)
        
        Returns:
            Point on manifold after geodesic flow
        """
        ...
    
    @abstractmethod
    def log(self, p: Tensor, q: Tensor) -> Tensor:
        """
        Logarithmic map: tangent vector at p pointing toward q.
        
        Args:
            p: Base point on manifold
            q: Target point on manifold
        
        Returns:
            Tangent vector v such that exp(p, v) = q
            
        Raises:
            ValueError: If q is at or beyond the cut locus of p
        """
        ...
    
    @abstractmethod
    def parallel_transport(self, v: Tensor, p: Tensor, q: Tensor) -> Tensor:
        """
        Parallel transport tangent vector v from T_pM to T_qM.
        
        Args:
            v: Tangent vector at p
            p: Source point
            q: Destination point
        
        Returns:
            Tangent vector at q with same geometric properties as v
        """
        ...
    
    @abstractmethod
    def distance(self, p: Tensor, q: Tensor) -> Tensor:
        """
        Geodesic distance between points.
        
        Args:
            p: First point, shape (..., ambient_dim)
            q: Second point, shape (..., ambient_dim)
        
        Returns:
            Distance, shape (...)
        """
        ...
    
    @abstractmethod
    def project(self, x: Tensor) -> Tensor:
        """
        Project ambient space point onto manifold.
        
        Args:
            x: Point in ambient space
        
        Returns:
            Closest point on manifold
        """
        ...
    
    @abstractmethod
    def project_tangent(self, p: Tensor, v: Tensor) -> Tensor:
        """
        Project ambient vector onto tangent space at p.
        
        Args:
            p: Point on manifold
            v: Vector in ambient space
        
        Returns:
            Component of v in T_pM
        """
        ...
    
    def norm(self, p: Tensor, v: Tensor) -> Tensor:
        """
        Riemannian norm of tangent vector v at point p.
        
        Args:
            p: Point on manifold
            v: Tangent vector at p
        
        Returns:
            ||v||_p = sqrt(g_p(v, v)) where g is the Riemannian metric
        """
        # Default for embedded submanifolds with induced metric
        return torch.linalg.norm(v, dim=-1)
    
    def in_domain(self, p: Tensor, q: Tensor) -> Tensor:
        """
        Check if log_p(q) is well-defined (q not at cut locus of p).
        
        Args:
            p: Base point
            q: Target point
            
        Returns:
            Boolean tensor, True if log_p(q) is defined
        """
        # Default: always in domain (override for compact manifolds)
        return torch.ones(p.shape[:-1], dtype=torch.bool, device=p.device)
    
    def geodesic(self, p: Tensor, q: Tensor, t: float) -> Tensor:
        """
        Point along geodesic from p to q at parameter t.
        
        Args:
            p: Start point
            q: End point
            t: Parameter in [0, 1]
        
        Returns:
            Point γ(t) where γ(0)=p, γ(1)=q
        """
        v = self.log(p, q)
        return self.exp(p, t * v)
    
    @abstractmethod
    def random_point(self, *shape, device=None, dtype=None) -> Tensor:
        """Generate random point(s) on manifold."""
        ...
    
    @abstractmethod
    def random_tangent(self, p: Tensor) -> Tensor:
        """Generate random tangent vector at p."""
        ...
