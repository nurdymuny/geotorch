"""Hyperbolic space manifold."""

import torch
from torch import Tensor
from ..manifold import Manifold


class Hyperbolic(Manifold):
    """Hyperbolic space H^{n-1} with constant negative curvature.
    
    Supports two models:
    - Poincaré ball: Open unit ball in ℝⁿ with conformal metric
    - Hyperboloid: Upper sheet of hyperboloid in Minkowski space
    
    Note: Hyperbolic(n) creates (n-1)-dimensional hyperbolic space H^{n-1}.
    For Poincaré ball, points live in ℝⁿ with ||x|| < 1.
    
    Args:
        n: Dimension parameter (creates H^{n-1})
        model: Either 'poincare' (default) or 'hyperboloid'
        curvature: Negative curvature constant (default: -1.0)
    
    Examples:
        >>> H = Hyperbolic(64, model='poincare')  # H^63 in Poincaré ball
        >>> p = H.random_point()
        >>> assert torch.norm(p) < 1.0  # Inside unit ball
    """
    
    def __init__(self, n: int, model: str = 'poincare', curvature: float = -1.0):
        if n < 2:
            raise ValueError(f"Hyperbolic requires n >= 2, got {n}")
        if model not in ['poincare', 'hyperboloid']:
            raise ValueError(f"Model must be 'poincare' or 'hyperboloid', got {model}")
        if curvature >= 0:
            raise ValueError(f"Curvature must be negative, got {curvature}")
        
        self._ambient_dim = n
        self._dim = n - 1
        self.model = model
        self.c = abs(curvature)  # Use positive c, handle sign in formulas
        self._eps = 1e-7
    
    @property
    def dim(self) -> int:
        """Intrinsic dimension of hyperbolic space (n-1 for H^{n-1})."""
        return self._dim
    
    def exp(self, p: Tensor, v: Tensor) -> Tensor:
        """Exponential map in hyperbolic space.
        
        Args:
            p: Point in hyperbolic space
            v: Tangent vector at p
        
        Returns:
            Point after moving along geodesic
        """
        if self.model == 'poincare':
            return self._exp_poincare(p, v)
        else:
            return self._exp_hyperboloid(p, v)
    
    def log(self, p: Tensor, q: Tensor) -> Tensor:
        """Logarithmic map in hyperbolic space.
        
        Args:
            p: Base point
            q: Target point
        
        Returns:
            Tangent vector at p pointing toward q
        """
        if self.model == 'poincare':
            return self._log_poincare(p, q)
        else:
            return self._log_hyperboloid(p, q)
    
    def parallel_transport(self, v: Tensor, p: Tensor, q: Tensor) -> Tensor:
        """Parallel transport in hyperbolic space.
        
        Args:
            v: Tangent vector at p
            p: Source point
            q: Destination point
        
        Returns:
            Parallel transported vector at q
        """
        if self.model == 'poincare':
            return self._parallel_transport_poincare(v, p, q)
        else:
            return self._parallel_transport_hyperboloid(v, p, q)
    
    def distance(self, p: Tensor, q: Tensor) -> Tensor:
        """Geodesic distance in hyperbolic space.
        
        Args:
            p: First point
            q: Second point
        
        Returns:
            Geodesic distance
        """
        if self.model == 'poincare':
            return self._distance_poincare(p, q)
        else:
            return self._distance_hyperboloid(p, q)
    
    def project(self, x: Tensor) -> Tensor:
        """Project point onto hyperbolic space.
        
        Args:
            x: Point in ambient space
        
        Returns:
            Projected point
        """
        if self.model == 'poincare':
            # Project onto open unit ball
            norm_x = torch.linalg.norm(x, dim=-1, keepdim=True)
            # Scale to be inside ball with small margin
            scale = torch.where(norm_x >= 1.0, 
                              (1.0 - self._eps) / (norm_x + self._eps),
                              torch.ones_like(norm_x))
            return x * scale
        else:
            # Project onto hyperboloid
            return self._project_hyperboloid(x)
    
    def project_tangent(self, p: Tensor, v: Tensor) -> Tensor:
        """Project vector onto tangent space.
        
        Args:
            p: Point on manifold
            v: Vector in ambient space
        
        Returns:
            Tangent vector at p
        """
        if self.model == 'poincare':
            # In Poincaré model, tangent space is just ℝⁿ (with metric)
            return v
        else:
            # In hyperboloid model, tangent space is orthogonal to p (Minkowski)
            return self._project_tangent_hyperboloid(p, v)
    
    def random_point(self, *shape, device=None, dtype=None) -> Tensor:
        """Generate random point(s) in hyperbolic space.
        
        Args:
            *shape: Shape of the output (batch dimensions)
            device: PyTorch device
            dtype: PyTorch dtype
        
        Returns:
            Random point(s)
        """
        full_shape = shape + (self._ambient_dim,)
        
        if self.model == 'poincare':
            # Generate random point in Poincaré ball
            x = torch.randn(full_shape, device=device, dtype=dtype)
            # Map to inside unit ball
            r = torch.rand(shape + (1,), device=device, dtype=dtype)
            r = r ** (1.0 / self._ambient_dim)  # Uniform in ball
            r = r * (1.0 - self._eps)  # Keep away from boundary
            norm_x = torch.linalg.norm(x, dim=-1, keepdim=True)
            return r * x / (norm_x + self._eps)
        else:
            # Generate random point on hyperboloid
            x = torch.randn(full_shape, device=device, dtype=dtype)
            return self._project_hyperboloid(x)
    
    # Poincaré ball implementations
    
    def _exp_poincare(self, p: Tensor, v: Tensor) -> Tensor:
        """Exponential map in Poincaré ball model using Möbius addition."""
        norm_v = torch.linalg.norm(v, dim=-1, keepdim=True)
        norm_v = torch.clamp(norm_v, min=self._eps)
        
        sqrt_c = torch.sqrt(torch.tensor(self.c))
        
        # Compute Möbius scalar multiplication
        factor = torch.tanh(sqrt_c * norm_v / 2) / (sqrt_c * norm_v)
        scaled_v = factor * v
        
        # Möbius addition: p ⊕ scaled_v
        return self._mobius_add(p, scaled_v)
    
    def _log_poincare(self, p: Tensor, q: Tensor) -> Tensor:
        """Logarithmic map in Poincaré ball model."""
        # Möbius subtraction: q ⊖ p = -p ⊕ q
        neg_p_add_q = self._mobius_add(-p, q)
        
        norm_diff = torch.linalg.norm(neg_p_add_q, dim=-1, keepdim=True)
        norm_diff = torch.clamp(norm_diff, min=self._eps)
        
        sqrt_c = torch.sqrt(torch.tensor(self.c))
        lambda_p = self._lambda(p)
        
        # Compute log
        factor = 2.0 / (sqrt_c * lambda_p * norm_diff)
        factor = factor * torch.atanh(sqrt_c * norm_diff)
        
        return factor * neg_p_add_q
    
    def _distance_poincare(self, p: Tensor, q: Tensor) -> Tensor:
        """Distance in Poincaré ball model."""
        sqrt_c = torch.sqrt(torch.tensor(self.c))
        
        diff_norm_sq = torch.sum((p - q) ** 2, dim=-1)
        norm_p_sq = torch.sum(p ** 2, dim=-1)
        norm_q_sq = torch.sum(q ** 2, dim=-1)
        
        # Avoid numerical issues
        norm_p_sq = torch.clamp(norm_p_sq, max=1.0 - self._eps)
        norm_q_sq = torch.clamp(norm_q_sq, max=1.0 - self._eps)
        
        numerator = 2.0 * diff_norm_sq
        denominator = (1.0 - norm_p_sq) * (1.0 - norm_q_sq)
        denominator = torch.clamp(denominator, min=self._eps)
        
        arg = 1.0 + numerator / denominator
        arg = torch.clamp(arg, min=1.0 + self._eps)
        
        return torch.acosh(arg) / sqrt_c
    
    def _parallel_transport_poincare(self, v: Tensor, p: Tensor, q: Tensor) -> Tensor:
        """Parallel transport in Poincaré ball model."""
        # Use gyration-based formula
        neg_p_add_q = self._mobius_add(-p, q)
        
        lambda_p = self._lambda(p)
        lambda_q = self._lambda(q)
        
        # Simplified parallel transport (preserves norm)
        factor = lambda_p / lambda_q
        
        # Apply gyration (simplified version)
        return factor * v
    
    def _mobius_add(self, x: Tensor, y: Tensor) -> Tensor:
        """Möbius addition: x ⊕ y in Poincaré ball."""
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        norm_x_sq = torch.sum(x ** 2, dim=-1, keepdim=True)
        norm_y_sq = torch.sum(y ** 2, dim=-1, keepdim=True)
        
        norm_x_sq = torch.clamp(norm_x_sq, max=1.0 - self._eps)
        norm_y_sq = torch.clamp(norm_y_sq, max=1.0 - self._eps)
        
        numerator = (1.0 + 2.0 * self.c * xy + self.c * norm_y_sq) * x + (1.0 - self.c * norm_x_sq) * y
        denominator = 1.0 + 2.0 * self.c * xy + self.c ** 2 * norm_x_sq * norm_y_sq
        denominator = torch.clamp(denominator, min=self._eps)
        
        result = numerator / denominator
        
        # Ensure result is in the ball
        norm_result = torch.linalg.norm(result, dim=-1, keepdim=True)
        scale = torch.where(norm_result >= 1.0,
                          (1.0 - self._eps) / (norm_result + self._eps),
                          torch.ones_like(norm_result))
        return result * scale
    
    def _lambda(self, x: Tensor) -> Tensor:
        """Conformal factor in Poincaré ball."""
        norm_x_sq = torch.sum(x ** 2, dim=-1, keepdim=True)
        norm_x_sq = torch.clamp(norm_x_sq, max=1.0 - self._eps)
        return 2.0 / (1.0 - self.c * norm_x_sq)
    
    # Hyperboloid model implementations (simplified)
    
    def _exp_hyperboloid(self, p: Tensor, v: Tensor) -> Tensor:
        """Exponential map in hyperboloid model."""
        # Simplified implementation
        norm_v = self._minkowski_norm(v)
        norm_v = torch.clamp(torch.abs(norm_v), min=self._eps)
        
        sqrt_c = torch.sqrt(torch.tensor(self.c))
        
        result = torch.cosh(sqrt_c * norm_v) * p + torch.sinh(sqrt_c * norm_v) / norm_v * v
        return self._project_hyperboloid(result)
    
    def _log_hyperboloid(self, p: Tensor, q: Tensor) -> Tensor:
        """Logarithmic map in hyperboloid model."""
        dot = self._minkowski_dot(p, q)
        dot = torch.clamp(dot, min=1.0 + self._eps)
        
        sqrt_c = torch.sqrt(torch.tensor(self.c))
        alpha = torch.acosh(dot) / sqrt_c
        
        v = q - dot * p
        norm_v = self._minkowski_norm(v)
        norm_v = torch.clamp(torch.abs(norm_v), min=self._eps)
        
        return alpha / norm_v * v
    
    def _distance_hyperboloid(self, p: Tensor, q: Tensor) -> Tensor:
        """Distance in hyperboloid model."""
        dot = self._minkowski_dot(p, q)
        dot = torch.clamp(dot, min=1.0 + self._eps)
        
        sqrt_c = torch.sqrt(torch.tensor(self.c))
        return torch.acosh(dot) / sqrt_c
    
    def _parallel_transport_hyperboloid(self, v: Tensor, p: Tensor, q: Tensor) -> Tensor:
        """Parallel transport in hyperboloid model."""
        # Simplified: project v onto tangent space at q
        return self._project_tangent_hyperboloid(q, v)
    
    def _project_hyperboloid(self, x: Tensor) -> Tensor:
        """Project onto hyperboloid upper sheet."""
        # x[:-1] are spatial components, x[-1] is time component
        spatial = x[..., :-1]
        sqrt_c = torch.sqrt(torch.tensor(self.c))
        
        # Compute time component: x_n = sqrt(1/c + ||x||^2)
        norm_sq = torch.sum(spatial ** 2, dim=-1, keepdim=True)
        time = torch.sqrt(1.0 / self.c + norm_sq)
        
        return torch.cat([spatial, time], dim=-1)
    
    def _project_tangent_hyperboloid(self, p: Tensor, v: Tensor) -> Tensor:
        """Project onto tangent space in hyperboloid model."""
        # Tangent space: Minkowski orthogonal to p
        dot = self._minkowski_dot(p, v)
        return v - dot * p
    
    def _minkowski_dot(self, x: Tensor, y: Tensor) -> Tensor:
        """Minkowski inner product: -x_n*y_n + sum(x_i*y_i)."""
        spatial_dot = torch.sum(x[..., :-1] * y[..., :-1], dim=-1, keepdim=True)
        time_dot = x[..., -1:] * y[..., -1:]
        return -time_dot + spatial_dot
    
    def _minkowski_norm(self, x: Tensor) -> Tensor:
        """Minkowski norm."""
        return torch.sqrt(torch.abs(self._minkowski_dot(x, x)))
