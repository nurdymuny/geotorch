"""Base manifold class for GeoTorch."""

from abc import ABC, abstractmethod
import torch
from torch import Tensor


class Manifold(ABC):
    """Abstract base class for Riemannian manifolds.
    
    This class defines the interface that all manifold implementations must follow.
    It provides the core geometric operations needed for Riemannian optimization.
    """
    
    @property
    @abstractmethod
    def dim(self) -> int:
        """Intrinsic dimension of the manifold.
        
        Returns:
            The intrinsic dimension (number of degrees of freedom).
        """
        pass
    
    @abstractmethod
    def exp(self, p: Tensor, v: Tensor) -> Tensor:
        """Exponential map: move from p along geodesic with velocity v.
        
        The exponential map exp_p(v) gives the point on the manifold reached by
        following the geodesic starting at p with initial velocity v for unit time.
        
        Args:
            p: Point on manifold, shape (..., ambient_dim)
            v: Tangent vector at p, shape (..., ambient_dim)
        
        Returns:
            Point on manifold after geodesic flow
        """
        pass
    
    @abstractmethod
    def log(self, p: Tensor, q: Tensor) -> Tensor:
        """Logarithmic map: tangent vector at p pointing toward q.
        
        The logarithmic map log_p(q) gives the tangent vector v at p such that
        exp_p(v) = q. This is the inverse of the exponential map.
        
        Args:
            p: Base point on manifold
            q: Target point on manifold
        
        Returns:
            Tangent vector v such that exp(p, v) = q
        """
        pass
    
    @abstractmethod
    def parallel_transport(self, v: Tensor, p: Tensor, q: Tensor) -> Tensor:
        """Parallel transport tangent vector v from T_pM to T_qM.
        
        Parallel transport moves a tangent vector along a geodesic while
        preserving its geometric properties (direction and norm).
        
        Args:
            v: Tangent vector at p
            p: Source point
            q: Destination point
        
        Returns:
            Tangent vector at q with same geometric properties as v
        """
        pass
    
    @abstractmethod
    def distance(self, p: Tensor, q: Tensor) -> Tensor:
        """Geodesic distance between points.
        
        Computes the length of the shortest geodesic connecting p and q.
        
        Args:
            p: First point, shape (..., ambient_dim)
            q: Second point, shape (..., ambient_dim)
        
        Returns:
            Distance, shape (...)
        """
        pass
    
    @abstractmethod
    def project(self, x: Tensor) -> Tensor:
        """Project ambient space point onto manifold.
        
        Finds the closest point on the manifold to x (in ambient space).
        
        Args:
            x: Point in ambient space
        
        Returns:
            Closest point on manifold
        """
        pass
    
    @abstractmethod
    def project_tangent(self, p: Tensor, v: Tensor) -> Tensor:
        """Project ambient vector onto tangent space at p.
        
        Projects an ambient space vector onto the tangent space T_pM.
        
        Args:
            p: Point on manifold
            v: Vector in ambient space
        
        Returns:
            Component of v in T_pM
        """
        pass
    
    def norm(self, p: Tensor, v: Tensor) -> Tensor:
        """Riemannian norm of tangent vector v at point p.
        
        Computes ||v||_p = sqrt(g_p(v, v)) where g_p is the metric tensor.
        For manifolds embedded in Euclidean space, this defaults to the
        Euclidean norm.
        
        Args:
            p: Point on manifold
            v: Tangent vector at p
        
        Returns:
            Norm of v at p
        """
        return torch.linalg.norm(v, dim=-1)
    
    def in_domain(self, p: Tensor, q: Tensor) -> bool:
        """Check if log_p(q) is well-defined (q not at cut locus).
        
        The logarithmic map may not be well-defined everywhere. For example,
        on the sphere, the cut locus of p is the antipodal point -p.
        
        Args:
            p: Base point
            q: Target point
        
        Returns:
            True if log_p(q) is well-defined, False otherwise
        """
        return True  # Default: log is well-defined everywhere
    
    def geodesic(self, p: Tensor, q: Tensor, t: float) -> Tensor:
        """Point along geodesic from p to q at parameter t.
        
        Computes the point γ(t) on the geodesic where γ(0) = p and γ(1) = q.
        
        Args:
            p: Start point
            q: End point
            t: Parameter in [0, 1]
        
        Returns:
            Point γ(t) along the geodesic
        """
        v = self.log(p, q)
        return self.exp(p, t * v)
    
    def random_point(self, *shape, device=None, dtype=None) -> Tensor:
        """Generate random point(s) on manifold.
        
        Args:
            *shape: Shape of the output (batch dimensions)
            device: PyTorch device
            dtype: PyTorch dtype
        
        Returns:
            Random point(s) on the manifold
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement random_point")
    
    def random_tangent(self, p: Tensor) -> Tensor:
        """Generate random tangent vector at p.
        
        Args:
            p: Base point on manifold
        
        Returns:
            Random tangent vector at p
        """
        v = torch.randn_like(p)
        return self.project_tangent(p, v)
