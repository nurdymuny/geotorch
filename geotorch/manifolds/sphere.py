import torch
from .manifold import Manifold
from typing import Optional


class Sphere(Manifold):
    """
    Sphere manifold S^{n-1} = {x ∈ ℝ^n : ||x|| = 1}
    
    The sphere manifold represents unit vectors in n-dimensional space.
    """
    
    def __init__(self, size: tuple):
        """
        Initialize the Sphere manifold.
        
        Args:
            size: The size of the tensor. The last dimension represents
                  the ambient space dimension n.
        """
        super().__init__(size)
        self.radius = 1.0
    
    @property
    def dim(self) -> int:
        """
        Returns the dimension of the manifold.
        For S^{n-1}, the dimension is n-1.
        """
        return self.size[-1] - 1
    
    def projx(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project a point onto the sphere by normalizing it.
        
        Args:
            x: Input tensor of shape (..., n)
            
        Returns:
            Projected point on the sphere
        """
        return x / x.norm(dim=-1, keepdim=True)
    
    def proju(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Project a vector onto the tangent space at point x.
        
        For the sphere, the tangent space at x is the orthogonal complement
        of x, so we project by removing the component parallel to x:
        proj_x(v) = v - <v, x>x
        
        Args:
            x: Point on the manifold of shape (..., n)
            v: Vector to project of shape (..., n)
            
        Returns:
            Projected vector in the tangent space at x
        """
        # Compute <v, x> along the last dimension
        dot_product = (v * x).sum(dim=-1, keepdim=True)
        # Project: v - <v, x>x
        return v - dot_product * x
    
    def inner(self, x: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Compute the Riemannian inner product between two tangent vectors.
        
        For the sphere, the Riemannian metric is inherited from the Euclidean metric:
        <v, w>_x = <v, w>
        
        Args:
            x: Point on the manifold of shape (..., n)
            v: First tangent vector of shape (..., n)
            w: Second tangent vector of shape (..., n)
            
        Returns:
            Inner product scalar(s)
        """
        return (v * w).sum(dim=-1)
    
    def norm(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute the norm of a tangent vector.
        
        Args:
            x: Point on the manifold of shape (..., n)
            v: Tangent vector of shape (..., n)
            
        Returns:
            Norm of the tangent vector
        """
        return v.norm(dim=-1)
    
    def exp(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Exponential map: maps a tangent vector to a point on the manifold.
        
        For the sphere, the exponential map is:
        exp_x(v) = cos(||v||)x + sin(||v||)(v/||v||)
        
        When ||v|| = 0, exp_x(v) = x
        
        Args:
            x: Point on the manifold of shape (..., n)
            v: Tangent vector at x of shape (..., n)
            
        Returns:
            Point on the manifold reached by following the geodesic
        """
        # Strip subclasses from input tensors
        x = x.data if isinstance(x, torch.Tensor) else torch.as_tensor(x)
        v = v.data if isinstance(v, torch.Tensor) else torch.as_tensor(v)
        
        # Compute norm of v
        v_norm = v.norm(dim=-1, keepdim=True)
        
        # Handle the case where v_norm is zero
        # Use a small epsilon to avoid division by zero
        eps = 1e-15
        
        # Compute the exponential map
        # When v_norm is very small, exp_x(v) ≈ x
        cos_v_norm = torch.cos(v_norm)
        sin_v_norm = torch.sin(v_norm)
        
        # Avoid division by zero: when v_norm is small, sin(v_norm)/v_norm ≈ 1
        sin_v_norm_over_v_norm = torch.where(
            v_norm > eps,
            sin_v_norm / v_norm,
            torch.ones_like(v_norm)
        )
        
        result = cos_v_norm * x + sin_v_norm_over_v_norm * v
        
        return result
    
    def log(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic map: maps a point on the manifold to a tangent vector.
        
        For the sphere, the logarithmic map is the inverse of exp:
        log_x(y) = (θ / sin(θ)) * (y - cos(θ)x)
        where θ = arccos(<x, y>)
        
        When y = x (θ = 0), log_x(y) = 0
        
        Args:
            x: Base point on the manifold of shape (..., n)
            y: Target point on the manifold of shape (..., n)
            
        Returns:
            Tangent vector at x pointing towards y
        """
        # Strip subclasses from input tensors
        x = x.data if isinstance(x, torch.Tensor) else torch.as_tensor(x)
        y = y.data if isinstance(y, torch.Tensor) else torch.as_tensor(y)
        
        # Compute <x, y>
        dot_xy = (x * y).sum(dim=-1, keepdim=True)
        
        # Clamp to avoid numerical issues with arccos
        dot_xy = torch.clamp(dot_xy, -1.0 + 1e-7, 1.0 - 1e-7)
        
        # Compute angle θ
        theta = torch.acos(dot_xy)
        
        # Handle the case where theta is zero (x ≈ y)
        eps = 1e-15
        sin_theta = torch.sin(theta)
        
        # Compute coefficient: θ / sin(θ)
        # When θ is small, θ / sin(θ) ≈ 1
        coeff = torch.where(
            theta > eps,
            theta / sin_theta,
            torch.ones_like(theta)
        )
        
        # Compute the tangent vector
        result = coeff * (y - dot_xy * x)
        
        return result
    
    def distance(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Compute the geodesic distance between two points on the sphere.
        
        The geodesic distance on the sphere is:
        d(p, q) = arccos(<p, q>)
        
        Args:
            p: First point on the manifold of shape (..., n)
            q: Second point on the manifold of shape (..., n)
            
        Returns:
            Geodesic distance between p and q
        """
        # Strip subclasses from input tensors
        p = p.data if isinstance(p, torch.Tensor) else torch.as_tensor(p)
        q = q.data if isinstance(q, torch.Tensor) else torch.as_tensor(q)
        
        # Compute <p, q>
        dot_pq = (p * q).sum(dim=-1)
        
        # Clamp to avoid numerical issues with arccos
        dot_pq = torch.clamp(dot_pq, -1.0 + 1e-7, 1.0 - 1e-7)
        
        # Return the geodesic distance
        return torch.acos(dot_pq)
    
    def parallel_transport(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        v: torch.Tensor
    ) -> torch.Tensor:
        """
        Parallel transport a tangent vector from x to y along the geodesic.
        
        For the sphere, parallel transport along the geodesic from x to y is:
        P_{x→y}(v) = v - [(x + y) / (1 + <x, y>)] <(x + y), v>
        
        When x = y, P_{x→y}(v) = v
        
        Args:
            x: Starting point on the manifold of shape (..., n)
            y: Ending point on the manifold of shape (..., n)
            v: Tangent vector at x of shape (..., n)
            
        Returns:
            Parallel transported vector in the tangent space at y
        """
        # Strip subclasses from input tensors
        x = x.data if isinstance(x, torch.Tensor) else torch.as_tensor(x)
        y = y.data if isinstance(y, torch.Tensor) else torch.as_tensor(y)
        v = v.data if isinstance(v, torch.Tensor) else torch.as_tensor(v)
        
        # Compute <x, y>
        dot_xy = (x * y).sum(dim=-1, keepdim=True)
        
        # Compute x + y
        x_plus_y = x + y
        
        # Handle the case where x ≈ -y (antipodal points)
        eps = 1e-15
        denom = 1.0 + dot_xy
        
        # When x and y are not antipodal
        # Compute <x + y, v>
        dot_xy_v = (x_plus_y * v).sum(dim=-1, keepdim=True)
        
        # Compute the parallel transport
        # P_{x→y}(v) = v - [(x + y) / (1 + <x, y>)] <(x + y), v>
        result = torch.where(
            torch.abs(denom) > eps,
            v - (x_plus_y / denom) * dot_xy_v,
            v  # When x ≈ -y, fall back to v (this case is degenerate)
        )
        
        return result
    
    def random_point(self, *batch_dims: int, dtype=None, device=None) -> torch.Tensor:
        """
        Generate random point(s) on the sphere.
        
        Args:
            *batch_dims: Additional batch dimensions
            dtype: Data type for the tensor
            device: Device for the tensor
            
        Returns:
            Random point(s) on the sphere
        """
        shape = batch_dims + self.size
        # Generate random points from a standard normal distribution
        x = torch.randn(shape, dtype=dtype, device=device)
        # Project onto the sphere
        return self.projx(x)
    
    def random_tangent(
        self,
        x: torch.Tensor,
        *batch_dims: int,
        dtype=None,
        device=None
    ) -> torch.Tensor:
        """
        Generate random tangent vector(s) at point x.
        
        Args:
            x: Point on the manifold
            *batch_dims: Additional batch dimensions
            dtype: Data type for the tensor
            device: Device for the tensor
            
        Returns:
            Random tangent vector(s) at x
        """
        if dtype is None:
            dtype = x.dtype
        if device is None:
            device = x.device
            
        shape = batch_dims + self.size if batch_dims else x.shape
        # Generate random vectors
        v = torch.randn(shape, dtype=dtype, device=device)
        # Project onto the tangent space
        return self.proju(x, v)
