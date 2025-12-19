"""Hyperbolic space H^{n-1} with constant negative curvature."""

import torch
from torch import Tensor
from ..manifold import Manifold


class Hyperbolic(Manifold):
    """
    Hyperbolic space H^{n-1} with constant negative curvature.
    
    Supports two models:
    - 'poincare': Poincaré ball model (open unit ball in R^n)
    - 'hyperboloid': Hyperboloid model (upper sheet in Minkowski space)
    
    Hyperbolic(n) creates H^{n-1} represented with n-dimensional vectors.
    
    Example:
        >>> H = Hyperbolic(64, model='poincare')  # H^63
        >>> p = H.random_point()
        >>> print(p.shape)  # torch.Size([64])
        >>> print(torch.norm(p) < 1)  # True (inside unit ball)
        >>> print(H.dim)  # 63 (intrinsic dimension)
    
    Poincaré ball formulas (curvature c = -1):
        - Uses Möbius addition for exp/log
        - distance(p, q) = arcosh(1 + 2||p-q||² / ((1-||p||²)(1-||q||²)))
    
    No cut locus (log is always defined).
    
    Args:
        n: Ambient dimension (points are in R^n, manifold is H^{n-1})
        model: 'poincare' (default) or 'hyperboloid'
        curvature: Negative curvature parameter (default: -1.0)
    """
    
    def __init__(self, n: int, model: str = 'poincare', curvature: float = -1.0):
        if model not in ['poincare', 'hyperboloid']:
            raise ValueError(f"Model must be 'poincare' or 'hyperboloid', got {model}")
        if curvature >= 0:
            raise ValueError(f"Curvature must be negative, got {curvature}")
        
        self.n = n
        self.model = model
        self.curvature = curvature
        self.eps = 1e-7
    
    @property
    def dim(self) -> int:
        """Intrinsic dimension of the manifold."""
        return self.n - 1
    
    def _lambda_x(self, x: Tensor) -> Tensor:
        """Conformal factor for Poincaré ball."""
        x_sqnorm = torch.sum(x * x, dim=-1, keepdim=True)
        return 2.0 / (1.0 - x_sqnorm).clamp(min=self.eps)
    
    def _mobius_add(self, x: Tensor, y: Tensor) -> Tensor:
        """Möbius addition in Poincaré ball."""
        x_sqnorm = torch.sum(x * x, dim=-1, keepdim=True)
        y_sqnorm = torch.sum(y * y, dim=-1, keepdim=True)
        xy_dot = torch.sum(x * y, dim=-1, keepdim=True)
        
        numerator = (1.0 + 2.0 * xy_dot + y_sqnorm) * x + (1.0 - x_sqnorm) * y
        denominator = (1.0 + 2.0 * xy_dot + x_sqnorm * y_sqnorm).clamp(min=self.eps)
        
        return numerator / denominator
    
    def _mobius_scalar_mult(self, r: Tensor, x: Tensor) -> Tensor:
        """Möbius scalar multiplication in Poincaré ball."""
        x_norm = torch.linalg.norm(x, dim=-1, keepdim=True)
        
        # Handle zero vector
        if (x_norm < self.eps).all():
            return x
        
        # r ⊗ x = tanh(r * arctanh(||x||)) * x / ||x||
        x_norm_clamped = torch.clamp(x_norm, max=1.0 - self.eps)
        arctanh_norm = torch.arctanh(x_norm_clamped)
        
        new_norm = torch.tanh(r * arctanh_norm)
        return new_norm * x / x_norm.clamp(min=self.eps)
    
    def exp(self, p: Tensor, v: Tensor) -> Tensor:
        """
        Exponential map: move from p along geodesic with velocity v.
        
        For Poincaré ball: exp_p(v) = p ⊕ (tanh(λ_p||v||/2) * v / ||v||)
        where ⊕ is Möbius addition and λ_p is the conformal factor.
        
        Args:
            p: Point on manifold, shape (..., n)
            v: Tangent vector at p, shape (..., n)
        
        Returns:
            Point on manifold after geodesic flow
        """
        if self.model != 'poincare':
            raise NotImplementedError("Only Poincaré model is implemented")
        
        v_norm = torch.linalg.norm(v, dim=-1, keepdim=True)
        
        # Handle zero velocity
        if (v_norm < self.eps).all():
            return p
        
        lambda_p = self._lambda_x(p)
        
        # Compute direction and scaled norm
        v_normalized = v / v_norm.clamp(min=self.eps)
        scaled_norm = torch.tanh(lambda_p * v_norm / 2.0)
        
        # Exponential map: p ⊕ (scaled_norm * v_normalized)
        direction = scaled_norm * v_normalized
        return self._mobius_add(p, direction)
    
    def log(self, p: Tensor, q: Tensor) -> Tensor:
        """
        Logarithmic map: tangent vector at p pointing toward q.
        
        For Poincaré ball: log_p(q) = (2/λ_p) * arctanh(||(-p) ⊕ q||) * ((-p) ⊕ q) / ||(-p) ⊕ q||
        
        Args:
            p: Base point on manifold
            q: Target point on manifold
        
        Returns:
            Tangent vector v such that exp(p, v) = q
        """
        if self.model != 'poincare':
            raise NotImplementedError("Only Poincaré model is implemented")
        
        # Compute -p ⊕ q
        minus_p = -p
        diff = self._mobius_add(minus_p, q)
        diff_norm = torch.linalg.norm(diff, dim=-1, keepdim=True)
        
        # Handle same point
        if (diff_norm < self.eps).all():
            return torch.zeros_like(p)
        
        lambda_p = self._lambda_x(p)
        
        # log_p(q) = (2/λ_p) * arctanh(||diff||) * diff / ||diff||
        diff_norm_clamped = torch.clamp(diff_norm, max=1.0 - self.eps)
        arctanh_norm = torch.arctanh(diff_norm_clamped)
        
        return (2.0 / lambda_p) * arctanh_norm * diff / diff_norm.clamp(min=self.eps)
    
    def parallel_transport(self, v: Tensor, p: Tensor, q: Tensor) -> Tensor:
        """
        Parallel transport tangent vector v from T_pM to T_qM.
        
        Uses the gyrovector transport formula for Poincaré ball.
        
        Args:
            v: Tangent vector at p
            p: Source point
            q: Destination point
        
        Returns:
            Tangent vector at q with same geometric properties as v
        """
        if self.model != 'poincare':
            raise NotImplementedError("Only Poincaré model is implemented")
        
        # Compute log_p(q) for the direction
        direction = self.log(p, q)
        direction_norm = torch.linalg.norm(direction, dim=-1, keepdim=True)
        
        # Handle p = q case
        if (direction_norm < self.eps).all():
            return v
        
        lambda_p = self._lambda_x(p)
        lambda_q = self._lambda_x(q)
        
        # Gyrovector parallel transport
        # PT(v) = (λ_p / λ_q) * v
        return (lambda_p / lambda_q) * v
    
    def distance(self, p: Tensor, q: Tensor) -> Tensor:
        """
        Geodesic distance between points.
        
        For Poincaré ball:
        distance(p, q) = arcosh(1 + 2||p-q||² / ((1-||p||²)(1-||q||²)))
        
        Args:
            p: First point, shape (..., n)
            q: Second point, shape (..., n)
        
        Returns:
            Distance, shape (...)
        """
        if self.model != 'poincare':
            raise NotImplementedError("Only Poincaré model is implemented")
        
        diff_sqnorm = torch.sum((p - q) ** 2, dim=-1)
        p_sqnorm = torch.sum(p * p, dim=-1)
        q_sqnorm = torch.sum(q * q, dim=-1)
        
        denominator = ((1.0 - p_sqnorm) * (1.0 - q_sqnorm)).clamp(min=self.eps)
        
        # arcosh argument must be >= 1
        arcosh_arg = 1.0 + 2.0 * diff_sqnorm / denominator
        arcosh_arg = torch.clamp(arcosh_arg, min=1.0)
        
        return torch.acosh(arcosh_arg)
    
    def project(self, x: Tensor) -> Tensor:
        """
        Project ambient space point onto manifold.
        
        For Poincaré ball, projects to inside the unit ball.
        
        Args:
            x: Point in ambient space
        
        Returns:
            Closest point on manifold
        """
        if self.model != 'poincare':
            raise NotImplementedError("Only Poincaré model is implemented")
        
        x_norm = torch.linalg.norm(x, dim=-1, keepdim=True)
        
        # If already inside ball (with margin), return as is
        # Otherwise, project to boundary with margin
        max_norm = 1.0 - self.eps
        scale = torch.where(
            x_norm < max_norm,
            torch.ones_like(x_norm),
            max_norm / x_norm.clamp(min=self.eps)
        )
        
        return scale * x
    
    def project_tangent(self, p: Tensor, v: Tensor) -> Tensor:
        """
        Project ambient vector onto tangent space at p.
        
        For Poincaré ball with the canonical metric, the tangent space
        is the entire ambient space R^n, so this is the identity.
        
        Args:
            p: Point on manifold
            v: Vector in ambient space
        
        Returns:
            Component of v in T_pM
        """
        # In Poincaré ball, tangent space is all of R^n
        return v
    
    def in_domain(self, p: Tensor, q: Tensor) -> Tensor:
        """
        Check if log_p(q) is well-defined (q not at cut locus of p).
        
        Hyperbolic space has no cut locus, so log is always defined.
        
        Args:
            p: Base point
            q: Target point
            
        Returns:
            Boolean tensor, True if log_p(q) is defined (always True)
        """
        return torch.ones(p.shape[:-1], dtype=torch.bool, device=p.device)
    
    def random_point(self, *shape, device=None, dtype=None) -> Tensor:
        """
        Generate random point(s) on manifold.
        
        Samples uniformly from the Poincaré ball.
        
        Args:
            *shape: Shape of points to generate (excluding final dimension n)
            device: Device to create tensor on
            dtype: Data type of tensor
        
        Returns:
            Random point(s) on manifold
        """
        if self.model != 'poincare':
            raise NotImplementedError("Only Poincaré model is implemented")
        
        if not shape:
            shape = ()
        
        # Sample from normal distribution and project
        x = torch.randn(*shape, self.n, device=device, dtype=dtype)
        
        # Scale to be inside unit ball
        # Use uniform radius sampling for better distribution
        x_norm = torch.linalg.norm(x, dim=-1, keepdim=True)
        
        # Sample radius uniformly in [0, 1)
        radius = torch.rand(*shape, 1, device=device, dtype=dtype)
        radius = torch.pow(radius, 1.0 / self.n)  # Correct for volume in n dimensions
        radius = radius * (1.0 - self.eps)  # Stay away from boundary
        
        return radius * x / x_norm.clamp(min=self.eps)
    
    def random_tangent(self, p: Tensor) -> Tensor:
        """
        Generate random tangent vector at p.
        
        For Poincaré ball, samples from standard normal distribution.
        
        Args:
            p: Point on manifold
        
        Returns:
            Random tangent vector at p
        """
        return torch.randn_like(p)
