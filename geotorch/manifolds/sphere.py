"""Sphere manifold S^{n-1} embedded in R^n."""

import torch
from torch import Tensor
from ..manifold import Manifold


class Sphere(Manifold):
    """
    Unit sphere S^{n-1} embedded in R^n.
    
    Sphere(n) creates the set of unit vectors in R^n, which forms the 
    (n-1)-dimensional sphere S^{n-1}.
    
    Example:
        >>> S = Sphere(64)  # S^63 embedded in R^64
        >>> p = S.random_point()
        >>> print(p.shape)  # torch.Size([64])
        >>> print(torch.norm(p))  # tensor(1.0)
        >>> print(S.dim)  # 63 (intrinsic dimension)
    
    Closed-form geodesic formulas:
        - exp_p(v) = cos(||v||) * p + sin(||v||) * v / ||v||
        - log_p(q) = θ * (q - cos(θ)*p) / sin(θ), where θ = arccos(p·q)
        - distance(p, q) = arccos(p·q)
    
    Cut locus: Antipodal points (q = -p). log_p(-p) is undefined.
    
    Args:
        n: Ambient dimension (points are in R^n, manifold is S^{n-1})
    """
    
    def __init__(self, n: int):
        self.n = n
        self.eps = 1e-7  # For numerical stability
    
    @property
    def dim(self) -> int:
        """Intrinsic dimension of the manifold."""
        return self.n - 1
    
    def exp(self, p: Tensor, v: Tensor) -> Tensor:
        """
        Exponential map: move from p along geodesic with velocity v.
        
        Formula: exp_p(v) = cos(||v||) * p + sin(||v||) * v / ||v||
        
        Args:
            p: Point on manifold, shape (..., n)
            v: Tangent vector at p, shape (..., n)
        
        Returns:
            Point on manifold after geodesic flow
        """
        v_norm = torch.linalg.norm(v, dim=-1, keepdim=True)
        
        # Handle small velocities for numerical stability
        mask = v_norm > self.eps
        
        result = torch.zeros_like(p)
        result[..., :] = p  # Default: exp_p(0) = p
        
        if mask.any():
            v_norm_safe = v_norm[mask]
            p_masked = p[mask.squeeze(-1)]
            v_masked = v[mask.squeeze(-1)]
            
            cos_norm = torch.cos(v_norm_safe)
            sin_norm = torch.sin(v_norm_safe)
            
            result[mask.squeeze(-1)] = cos_norm * p_masked + sin_norm * v_masked / v_norm_safe
        
        return result
    
    def log(self, p: Tensor, q: Tensor) -> Tensor:
        """
        Logarithmic map: tangent vector at p pointing toward q.
        
        Formula: log_p(q) = θ * (q - cos(θ)*p) / sin(θ), where θ = arccos(p·q)
        
        Args:
            p: Base point on manifold
            q: Target point on manifold
        
        Returns:
            Tangent vector v such that exp(p, v) = q
            
        Raises:
            ValueError: If q is at or beyond the cut locus of p
        """
        # Compute dot product, clamped to [-1, 1] for numerical stability
        dot = torch.sum(p * q, dim=-1, keepdim=True)
        dot = torch.clamp(dot, -1.0 + self.eps, 1.0 - self.eps)
        
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)
        
        # Check for cut locus (antipodal points)
        if not self.in_domain(p, q).all():
            raise ValueError("Target point is at or beyond the cut locus (antipodal point)")
        
        # Handle small angles for numerical stability
        mask = sin_theta.abs() > self.eps
        
        result = torch.zeros_like(p)
        
        if mask.any():
            theta_safe = theta[mask]
            sin_theta_safe = sin_theta[mask]
            p_masked = p[mask.squeeze(-1)]
            q_masked = q[mask.squeeze(-1)]
            dot_masked = dot[mask]
            
            result[mask.squeeze(-1)] = theta_safe * (q_masked - dot_masked * p_masked) / sin_theta_safe
        
        return result
    
    def parallel_transport(self, v: Tensor, p: Tensor, q: Tensor) -> Tensor:
        """
        Parallel transport tangent vector v from T_pM to T_qM.
        
        Uses the formula for parallel transport along geodesics on the sphere.
        
        Args:
            v: Tangent vector at p
            p: Source point
            q: Destination point
        
        Returns:
            Tangent vector at q with same geometric properties as v
        """
        # Compute log_p(q)
        w = self.log(p, q)
        w_norm = torch.linalg.norm(w, dim=-1, keepdim=True)
        
        # Handle case where p = q
        mask = w_norm > self.eps
        
        result = torch.zeros_like(v)
        result[..., :] = v  # Default: PT(v) = v when p = q
        
        if mask.any():
            w_norm_safe = w_norm[mask]
            p_masked = p[mask.squeeze(-1)]
            q_masked = q[mask.squeeze(-1)]
            v_masked = v[mask.squeeze(-1)]
            w_masked = w[mask.squeeze(-1)]
            
            # Parallel transport formula for sphere
            cos_norm = torch.cos(w_norm_safe)
            sin_norm = torch.sin(w_norm_safe)
            
            w_unit = w_masked / w_norm_safe
            v_parallel = torch.sum(v_masked * w_unit, dim=-1, keepdim=True) * w_unit
            v_perp = v_masked - v_parallel
            
            result[mask.squeeze(-1)] = (
                -sin_norm * torch.sum(v_masked * w_unit, dim=-1, keepdim=True) * p_masked
                + cos_norm * v_perp
                + torch.sum(v_masked * w_unit, dim=-1, keepdim=True) * q_masked
            )
        
        return result
    
    def distance(self, p: Tensor, q: Tensor) -> Tensor:
        """
        Geodesic distance between points.
        
        Formula: distance(p, q) = arccos(p·q)
        
        Args:
            p: First point, shape (..., n)
            q: Second point, shape (..., n)
        
        Returns:
            Distance, shape (...)
        """
        dot = torch.sum(p * q, dim=-1)
        dot = torch.clamp(dot, -1.0 + self.eps, 1.0 - self.eps)
        return torch.acos(dot)
    
    def project(self, x: Tensor) -> Tensor:
        """
        Project ambient space point onto manifold.
        
        Projects by normalizing to unit length.
        
        Args:
            x: Point in ambient space
        
        Returns:
            Closest point on manifold (normalized)
        """
        x_norm = torch.linalg.norm(x, dim=-1, keepdim=True)
        return x / torch.clamp(x_norm, min=self.eps)
    
    def project_tangent(self, p: Tensor, v: Tensor) -> Tensor:
        """
        Project ambient vector onto tangent space at p.
        
        The tangent space at p is orthogonal to p, so we subtract the
        radial component: v - (v·p)p
        
        Args:
            p: Point on manifold
            v: Vector in ambient space
        
        Returns:
            Component of v in T_pM
        """
        dot = torch.sum(v * p, dim=-1, keepdim=True)
        return v - dot * p
    
    def in_domain(self, p: Tensor, q: Tensor) -> Tensor:
        """
        Check if log_p(q) is well-defined (q not at cut locus of p).
        
        The cut locus consists of antipodal points where p·q ≈ -1.
        
        Args:
            p: Base point
            q: Target point
            
        Returns:
            Boolean tensor, True if log_p(q) is defined
        """
        dot = torch.sum(p * q, dim=-1)
        # Not at cut locus if dot > -1 + eps
        return dot > -1.0 + self.eps
    
    def random_point(self, *shape, device=None, dtype=None) -> Tensor:
        """
        Generate random point(s) on manifold.
        
        Samples from standard normal distribution and normalizes.
        
        Args:
            *shape: Shape of points to generate (excluding final dimension n)
            device: Device to create tensor on
            dtype: Data type of tensor
        
        Returns:
            Random point(s) on manifold
        """
        if not shape:
            shape = ()
        x = torch.randn(*shape, self.n, device=device, dtype=dtype)
        return self.project(x)
    
    def random_tangent(self, p: Tensor) -> Tensor:
        """
        Generate random tangent vector at p.
        
        Samples from standard normal and projects to tangent space.
        
        Args:
            p: Point on manifold
        
        Returns:
            Random tangent vector at p
        """
        v = torch.randn_like(p)
        return self.project_tangent(p, v)
