"""Base manifold classes and Euclidean implementation."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np


class Manifold(ABC):
    """Abstract base class for Riemannian manifolds."""
    
    def __init__(self, dim: int):
        """Initialize manifold.
        
        Args:
            dim: Intrinsic dimension of the manifold
        """
        self._dim = dim
    
    @property
    def dim(self) -> int:
        """Intrinsic dimension of the manifold."""
        return self._dim
    
    @property
    def ambient_dim(self) -> int:
        """Dimension of ambient space (may equal dim for embedded manifolds)."""
        return self._dim
    
    @property
    def curvature_type(self) -> str:
        """Type of curvature: 'constant', 'variable', or 'learned'."""
        return 'variable'
    
    # Point operations
    
    @abstractmethod
    def random_point(self, *shape, dtype=None) -> np.ndarray:
        """Sample random point on manifold.
        
        Args:
            *shape: Batch shape
            dtype: Data type
            
        Returns:
            Random point(s) on manifold
        """
        pass
    
    @abstractmethod
    def origin(self, *shape, dtype=None) -> np.ndarray:
        """Canonical origin/identity point.
        
        Args:
            *shape: Batch shape
            dtype: Data type
            
        Returns:
            Origin point(s)
        """
        pass
    
    @abstractmethod
    def check_point(self, x: np.ndarray, atol: float = 1e-6) -> bool:
        """Verify x is on the manifold (within tolerance).
        
        Args:
            x: Point to check
            atol: Absolute tolerance
            
        Returns:
            True if point is on manifold
        """
        pass
    
    @abstractmethod
    def project_point(self, x: np.ndarray) -> np.ndarray:
        """Project ambient point onto manifold.
        
        Args:
            x: Point in ambient space
            
        Returns:
            Projected point on manifold
        """
        pass
    
    # Tangent space operations
    
    @abstractmethod
    def check_tangent(self, x: np.ndarray, v: np.ndarray, atol: float = 1e-6) -> bool:
        """Verify v is in tangent space at x.
        
        Args:
            x: Base point on manifold
            v: Vector to check
            atol: Absolute tolerance
            
        Returns:
            True if v is in tangent space at x
        """
        pass
    
    @abstractmethod
    def project_tangent(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Project ambient vector to tangent space at x.
        
        Args:
            x: Base point on manifold
            v: Vector in ambient space
            
        Returns:
            Projected vector in tangent space at x
        """
        pass
    
    @abstractmethod
    def random_tangent(self, x: np.ndarray) -> np.ndarray:
        """Sample random tangent vector at x.
        
        Args:
            x: Base point on manifold
            
        Returns:
            Random tangent vector at x
        """
        pass
    
    def zero_tangent(self, x: np.ndarray) -> np.ndarray:
        """Zero tangent vector at x.
        
        Args:
            x: Base point on manifold
            
        Returns:
            Zero tangent vector at x
        """
        return np.zeros_like(x)
    
    # Metric operations
    
    @abstractmethod
    def metric(self, x: np.ndarray) -> np.ndarray:
        """Metric tensor g_ij(x) at point x.
        
        Args:
            x: Point on manifold with shape (..., dim)
            
        Returns:
            Metric tensor with shape (..., dim, dim)
        """
        pass
    
    @abstractmethod
    def inner(self, x: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Inner product ⟨u, v⟩_x using metric at x.
        
        Args:
            x: Base point on manifold
            u: First tangent vector
            v: Second tangent vector
            
        Returns:
            Inner product scalar(s)
        """
        pass
    
    def norm(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Norm ||v||_x = sqrt(⟨v, v⟩_x).
        
        Args:
            x: Base point on manifold
            v: Tangent vector at x
            
        Returns:
            Norm of v
        """
        return np.sqrt(self.inner(x, v, v))
    
    # Exponential and logarithm maps
    
    @abstractmethod
    def exp(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Exponential map: exp_x(v).
        
        Move from x in direction v for unit time along geodesic.
        
        Args:
            x: Base point on manifold
            v: Tangent vector at x
            
        Returns:
            Point on manifold
        """
        pass
    
    @abstractmethod
    def log(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Logarithm map: log_x(y).
        
        Tangent vector at x pointing toward y.
        
        Args:
            x: Base point on manifold
            y: Target point on manifold
            
        Returns:
            Tangent vector at x
        """
        pass
    
    def distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Geodesic distance d(x, y) = ||log_x(y)||_x.
        
        Args:
            x: First point on manifold
            y: Second point on manifold
            
        Returns:
            Geodesic distance
        """
        v = self.log(x, y)
        return self.norm(x, v)
    
    def geodesic(self, x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        """Point at fraction t ∈ [0,1] along geodesic from x to y.
        
        Args:
            x: Starting point on manifold
            y: Ending point on manifold
            t: Interpolation parameter (0 = x, 1 = y)
            
        Returns:
            Point on geodesic
        """
        v = self.log(x, y)
        return self.exp(x, t * v)
    
    # Parallel transport
    
    @abstractmethod
    def parallel_transport(self, x: np.ndarray, y: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Parallel transport v from T_x M to T_y M along geodesic.
        
        Args:
            x: Source point on manifold
            y: Target point on manifold
            v: Tangent vector at x
            
        Returns:
            Transported tangent vector at y
        """
        pass
    
    # Curvature (optional, default implementations)
    
    def sectional_curvature(self, x: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Sectional curvature K(u, v) of plane spanned by u, v at x.
        
        Args:
            x: Base point on manifold
            u: First tangent vector
            v: Second tangent vector
            
        Returns:
            Sectional curvature
        """
        raise NotImplementedError("Sectional curvature not implemented for this manifold")
    
    def scalar_curvature(self, x: np.ndarray) -> np.ndarray:
        """Scalar curvature R(x) = trace of Ricci tensor.
        
        Args:
            x: Point on manifold
            
        Returns:
            Scalar curvature
        """
        raise NotImplementedError("Scalar curvature not implemented for this manifold")


class Euclidean(Manifold):
    """Flat Euclidean space R^n.
    
    - Zero curvature
    - exp_x(v) = x + v
    - log_x(y) = y - x
    - Parallel transport is identity
    """
    
    def __init__(self, dim: int):
        """Initialize Euclidean space.
        
        Args:
            dim: Dimension of the space
        """
        super().__init__(dim)
    
    @property
    def curvature_type(self) -> str:
        """Euclidean space has constant zero curvature."""
        return 'constant'
    
    def random_point(self, *shape, dtype=None) -> np.ndarray:
        """Sample random point (standard normal).
        
        Args:
            *shape: Batch shape
            dtype: Data type
            
        Returns:
            Random point(s) with shape (*shape, dim)
        """
        if dtype is None:
            dtype = np.float64
        full_shape = shape + (self.dim,)
        return np.random.randn(*full_shape).astype(dtype)
    
    def origin(self, *shape, dtype=None) -> np.ndarray:
        """Zero vector as origin.
        
        Args:
            *shape: Batch shape
            dtype: Data type
            
        Returns:
            Zero vector(s) with shape (*shape, dim)
        """
        if dtype is None:
            dtype = np.float64
        full_shape = shape + (self.dim,)
        return np.zeros(full_shape, dtype=dtype)
    
    def check_point(self, x: np.ndarray, atol: float = 1e-6) -> bool:
        """All points are valid in Euclidean space.
        
        Args:
            x: Point to check
            atol: Absolute tolerance (unused)
            
        Returns:
            Always True
        """
        # Check shape
        if x.shape[-1] != self.dim:
            return False
        return True
    
    def project_point(self, x: np.ndarray) -> np.ndarray:
        """Identity projection (all points valid).
        
        Args:
            x: Point in ambient space
            
        Returns:
            Same point
        """
        return x
    
    def check_tangent(self, x: np.ndarray, v: np.ndarray, atol: float = 1e-6) -> bool:
        """All vectors are valid tangent vectors in Euclidean space.
        
        Args:
            x: Base point (unused)
            v: Vector to check
            atol: Absolute tolerance (unused)
            
        Returns:
            Always True
        """
        # Check shape
        if v.shape[-1] != self.dim:
            return False
        return True
    
    def project_tangent(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Identity projection (all vectors valid).
        
        Args:
            x: Base point (unused)
            v: Vector in ambient space
            
        Returns:
            Same vector
        """
        return v
    
    def random_tangent(self, x: np.ndarray) -> np.ndarray:
        """Sample random tangent vector (standard normal).
        
        Args:
            x: Base point
            
        Returns:
            Random tangent vector with same shape as x
        """
        return np.random.randn(*x.shape).astype(x.dtype)
    
    def metric(self, x: np.ndarray) -> np.ndarray:
        """Identity metric tensor.
        
        Args:
            x: Point on manifold with shape (..., dim)
            
        Returns:
            Identity matrix with shape (..., dim, dim)
        """
        batch_shape = x.shape[:-1]
        metric = np.eye(self.dim, dtype=x.dtype)
        # Broadcast to batch shape
        for _ in batch_shape:
            metric = np.expand_dims(metric, 0)
        metric = np.broadcast_to(metric, batch_shape + (self.dim, self.dim))
        return metric
    
    def inner(self, x: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Euclidean inner product ⟨u, v⟩ = u·v.
        
        Args:
            x: Base point (unused)
            u: First vector
            v: Second vector
            
        Returns:
            Inner product
        """
        return np.sum(u * v, axis=-1)
    
    def exp(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Exponential map: exp_x(v) = x + v.
        
        Args:
            x: Base point
            v: Tangent vector
            
        Returns:
            x + v
        """
        return x + v
    
    def log(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Logarithm map: log_x(y) = y - x.
        
        Args:
            x: Base point
            y: Target point
            
        Returns:
            y - x
        """
        return y - x
    
    def parallel_transport(self, x: np.ndarray, y: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Parallel transport is identity in Euclidean space.
        
        Args:
            x: Source point (unused)
            y: Target point (unused)
            v: Vector to transport
            
        Returns:
            Same vector v
        """
        return v
    
    def sectional_curvature(self, x: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Sectional curvature is zero in Euclidean space.
        
        Args:
            x: Base point
            u: First tangent vector
            v: Second tangent vector
            
        Returns:
            Zero
        """
        batch_shape = x.shape[:-1]
        return np.zeros(batch_shape, dtype=x.dtype)
    
    def scalar_curvature(self, x: np.ndarray) -> np.ndarray:
        """Scalar curvature is zero in Euclidean space.
        
        Args:
            x: Point on manifold
            
        Returns:
            Zero
        """
        batch_shape = x.shape[:-1]
        return np.zeros(batch_shape, dtype=x.dtype)
    
    def __repr__(self):
        return f'Euclidean({self.dim})'
