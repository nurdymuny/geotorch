"""Sphere manifold."""

import torch
from torch import Tensor
from ..manifold import Manifold


class Sphere(Manifold):
    """Unit sphere S^{n-1} embedded in ℝⁿ.
    
    The sphere consists of all points x in ℝⁿ such that ||x|| = 1.
    Note: Sphere(n) creates the (n-1)-dimensional sphere S^{n-1} embedded in ℝⁿ.
    
    Geodesics are great circles, and the exponential/logarithmic maps have
    closed-form expressions using trigonometric functions.
    
    Args:
        n: Ambient dimension (creates S^{n-1} sphere)
    
    Examples:
        >>> S = Sphere(3)  # Creates S^2 (2-sphere) in R^3
        >>> p = S.random_point()
        >>> assert torch.allclose(torch.norm(p), torch.tensor(1.0))
    """
    
    def __init__(self, n: int):
        if n < 2:
            raise ValueError(f"Sphere requires n >= 2, got {n}")
        self._ambient_dim = n
        self._dim = n - 1
        self._eps = 1e-7  # For numerical stability
    
    @property
    def dim(self) -> int:
        """Intrinsic dimension of the sphere (n-1 for S^{n-1})."""
        return self._dim
    
    def exp(self, p: Tensor, v: Tensor) -> Tensor:
        """Exponential map on the sphere.
        
        Uses the closed-form formula:
            exp_p(v) = cos(||v||) * p + sin(||v||) / ||v|| * v
        
        Args:
            p: Point on sphere (unit vector)
            v: Tangent vector at p (orthogonal to p)
        
        Returns:
            Point on sphere after moving along geodesic
        """
        # Ensure inputs are plain tensors (not subclasses)
        p = torch.as_tensor(p)
        v = torch.as_tensor(v)
        
        norm_v = torch.linalg.norm(v, dim=-1, keepdim=True)
        
        # Handle zero vector case
        norm_v = torch.clamp(norm_v, min=self._eps)
        
        # Compute exponential map
        result = torch.cos(norm_v) * p + torch.sin(norm_v) / norm_v * v
        
        # Ensure result is on sphere (numerical stability)
        return result / torch.linalg.norm(result, dim=-1, keepdim=True)
    
    def log(self, p: Tensor, q: Tensor) -> Tensor:
        """Logarithmic map on the sphere.
        
        Uses the closed-form formula:
            log_p(q) = θ * (q - cos(θ)*p) / sin(θ)
        where θ = arccos(clamp(p·q, -1, 1))
        
        Args:
            p: Base point on sphere
            q: Target point on sphere
        
        Returns:
            Tangent vector at p pointing toward q
        """
        # Ensure inputs are plain tensors (not subclasses)
        p = torch.as_tensor(p)
        q = torch.as_tensor(q)
        
        # Compute angle between p and q
        dot_pq = torch.sum(p * q, dim=-1, keepdim=True)
        dot_pq = torch.clamp(dot_pq, -1.0 + self._eps, 1.0 - self._eps)
        theta = torch.acos(dot_pq)
        
        # Handle case where p ≈ q (theta ≈ 0)
        sin_theta = torch.sin(theta)
        sin_theta = torch.clamp(sin_theta, min=self._eps)
        
        # Compute log map
        result = theta / sin_theta * (q - dot_pq * p)
        
        return result
    
    def parallel_transport(self, v: Tensor, p: Tensor, q: Tensor) -> Tensor:
        """Parallel transport on the sphere.
        
        Uses the formula for parallel transport along a geodesic:
            PT_p->q(v) = v - ((v·p) / (1 + p·q)) * (p + q)
        
        Args:
            v: Tangent vector at p
            p: Source point
            q: Destination point
        
        Returns:
            Parallel transported vector at q
        """
        # Ensure inputs are plain tensors (not subclasses)
        v = torch.as_tensor(v)
        p = torch.as_tensor(p)
        q = torch.as_tensor(q)
        
        dot_pq = torch.sum(p * q, dim=-1, keepdim=True)
        dot_vp = torch.sum(v * p, dim=-1, keepdim=True)
        
        # Avoid division by zero when p ≈ -q (antipodal points)
        denom = 1.0 + dot_pq
        denom = torch.where(torch.abs(denom) < self._eps, 
                           torch.ones_like(denom) * self._eps, 
                           denom)
        
        result = v - (dot_vp / denom) * (p + q)
        
        return result
    
    def distance(self, p: Tensor, q: Tensor) -> Tensor:
        """Geodesic distance on the sphere.
        
        The distance is the angle between the two unit vectors:
            d(p, q) = arccos(clamp(p·q, -1, 1))
        
        Args:
            p: First point on sphere
            q: Second point on sphere
        
        Returns:
            Geodesic distance (angle in radians)
        """
        dot_pq = torch.sum(p * q, dim=-1)
        dot_pq = torch.clamp(dot_pq, -1.0 + self._eps, 1.0 - self._eps)
        return torch.acos(dot_pq)
    
    def project(self, x: Tensor) -> Tensor:
        """Project point onto sphere by normalization.
        
        Args:
            x: Point in ambient space ℝⁿ
        
        Returns:
            x / ||x|| (normalized to unit sphere)
        """
        norm_x = torch.linalg.norm(x, dim=-1, keepdim=True)
        norm_x = torch.clamp(norm_x, min=self._eps)
        return x / norm_x
    
    def project_tangent(self, p: Tensor, v: Tensor) -> Tensor:
        """Project vector onto tangent space at p.
        
        The tangent space at p consists of all vectors orthogonal to p:
            proj_TpS(v) = v - (v·p) * p
        
        Args:
            p: Point on sphere
            v: Vector in ambient space
        
        Returns:
            Component of v orthogonal to p
        """
        dot_vp = torch.sum(v * p, dim=-1, keepdim=True)
        return v - dot_vp * p
    
    def in_domain(self, p: Tensor, q: Tensor) -> bool:
        """Check if log_p(q) is well-defined.
        
        The logarithmic map is not well-defined at antipodal points
        (the cut locus). Returns False when q ≈ -p.
        
        Args:
            p: Base point
            q: Target point
        
        Returns:
            True if q is not antipodal to p
        """
        dot_pq = torch.sum(p * q, dim=-1)
        # Check if points are approximately antipodal (dot product ≈ -1)
        return torch.all(dot_pq > -1.0 + self._eps).item()
    
    def random_point(self, *shape, device=None, dtype=None) -> Tensor:
        """Generate random point(s) uniformly on the sphere.
        
        Uses the standard method of normalizing Gaussian random vectors.
        
        Args:
            *shape: Shape of the output (batch dimensions)
            device: PyTorch device
            dtype: PyTorch dtype
        
        Returns:
            Random point(s) on the unit sphere
        """
        full_shape = shape + (self._ambient_dim,)
        x = torch.randn(full_shape, device=device, dtype=dtype)
        return self.project(x)
