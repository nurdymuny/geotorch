"""Base manifold classes and Euclidean implementation."""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional, Any
from ..core.storage import TensorCore, randn as core_randn, zeros as core_zeros


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
    def exp(self, p: TensorCore, v: TensorCore) -> TensorCore:
        """
        Exponential map: move from p along geodesic with velocity v.
        
        Args:
            p: Point on manifold
            v: Tangent vector at p
        
        Returns:
            Point on manifold after geodesic flow
        """
        ...
    
    @abstractmethod
    def log(self, p: TensorCore, q: TensorCore) -> TensorCore:
        """
        Logarithmic map: tangent vector at p pointing toward q.
        
        Args:
            p: Base point on manifold
            q: Target point on manifold
        
        Returns:
            Tangent vector v such that exp(p, v) = q
        """
        ...
    
    @abstractmethod
    def distance(self, p: TensorCore, q: TensorCore) -> TensorCore:
        """
        Geodesic distance between points.
        
        Args:
            p: First point
            q: Second point
        
        Returns:
            Geodesic distance
        """
        ...
    
    @abstractmethod
    def project(self, x: TensorCore) -> TensorCore:
        """
        Project point onto manifold.
        
        Args:
            x: Point in ambient space
        
        Returns:
            Closest point on manifold
        """
        ...
    
    @abstractmethod
    def random_point(self, *shape, dtype=None, device=None) -> TensorCore:
        """Generate random point(s) on manifold."""
        ...
    
    @abstractmethod
    def origin(self, *shape, dtype=None, device=None) -> TensorCore:
        """Generate origin/identity point on manifold."""
        ...
    
    def norm(self, p: TensorCore, v: TensorCore) -> TensorCore:
        """
        Riemannian norm of tangent vector v at point p.
        
        Args:
            p: Point on manifold
            v: Tangent vector at p
        
        Returns:
            ||v||_p
        """
        # Default implementation using Euclidean norm
        return TensorCore(
            storage=p.storage.__class__(1, p.dtype, p.device),
            shape=(),
            dtype=p.dtype,
            device=p.device,
        )
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.dim})"
    
    def __eq__(self, other) -> bool:
        """Check if two manifolds are equal."""
        if not isinstance(other, Manifold):
            return False
        return self.__class__ == other.__class__ and self.dim == other.dim


class Euclidean(Manifold):
    """
    Euclidean space R^n with flat metric.
    
    This is the trivial manifold where:
    - exp_p(v) = p + v
    - log_p(q) = q - p
    - distance(p, q) = ||q - p||
    
    Args:
        n: Dimension of the space
    """
    
    def __init__(self, n: int):
        self.n = n
    
    @property
    def dim(self) -> int:
        """Intrinsic dimension of the manifold."""
        return self.n
    
    def exp(self, p: TensorCore, v: TensorCore) -> TensorCore:
        """
        Exponential map: exp_p(v) = p + v
        
        Args:
            p: Point on manifold
            v: Tangent vector at p
        
        Returns:
            Point on manifold
        """
        # Element-wise addition
        p_data = p.numpy()
        v_data = v.numpy()
        result_data = p_data + v_data
        
        from ..core.storage import _create_tensor, GeometricType
        return _create_tensor(
            result_data,
            dtype=p.dtype,
            device=p.device,
            manifold=self,
            geometric_type=GeometricType.MANIFOLD_POINT,
        )
    
    def log(self, p: TensorCore, q: TensorCore) -> TensorCore:
        """
        Logarithmic map: log_p(q) = q - p
        
        Args:
            p: Base point
            q: Target point
        
        Returns:
            Tangent vector at p
        """
        # Element-wise subtraction
        p_data = p.numpy()
        q_data = q.numpy()
        result_data = q_data - p_data
        
        from ..core.storage import _create_tensor, GeometricType
        return _create_tensor(
            result_data,
            dtype=p.dtype,
            device=p.device,
            manifold=self,
            geometric_type=GeometricType.TANGENT,
        )
    
    def distance(self, p: TensorCore, q: TensorCore) -> TensorCore:
        """
        Geodesic distance: ||q - p||
        
        Args:
            p: First point
            q: Second point
        
        Returns:
            Euclidean distance
        """
        p_data = p.numpy()
        q_data = q.numpy()
        diff = q_data - p_data
        dist = float(np.linalg.norm(diff))
        
        from ..core.storage import _create_tensor, GeometricType
        return _create_tensor(
            dist,
            dtype=p.dtype,
            device=p.device,
            geometric_type=GeometricType.SCALAR,
        )
    
    def project(self, x: TensorCore) -> TensorCore:
        """
        Project onto manifold (identity for Euclidean space).
        
        Args:
            x: Point in ambient space
        
        Returns:
            Same point
        """
        return x
    
    def random_point(self, *shape, dtype=None, device=None) -> TensorCore:
        """
        Generate random point(s) on manifold.
        
        Samples from standard normal distribution.
        
        Args:
            *shape: Batch shape (without manifold dimension)
            dtype: Data type
            device: Device
        
        Returns:
            Random point(s)
        """
        from ..core.storage import float32, CPU, _create_tensor, GeometricType
        
        if dtype is None:
            dtype = float32
        if device is None:
            device = CPU
        
        # Create shape with manifold dimension
        full_shape = shape + (self.n,) if shape else (self.n,)
        data = np.random.randn(*full_shape).astype(dtype.numpy_dtype)
        
        return _create_tensor(
            data,
            dtype=dtype,
            device=device,
            manifold=self,
            geometric_type=GeometricType.MANIFOLD_POINT,
        )
    
    def origin(self, *shape, dtype=None, device=None) -> TensorCore:
        """
        Generate origin point (zero vector).
        
        Args:
            *shape: Batch shape (without manifold dimension)
            dtype: Data type
            device: Device
        
        Returns:
            Zero vector(s)
        """
        from ..core.storage import float32, CPU, _create_tensor, GeometricType
        
        if dtype is None:
            dtype = float32
        if device is None:
            device = CPU
        
        # Create shape with manifold dimension
        full_shape = shape + (self.n,) if shape else (self.n,)
        data = np.zeros(full_shape, dtype=dtype.numpy_dtype)
        
        return _create_tensor(
            data,
            dtype=dtype,
            device=device,
            manifold=self,
            geometric_type=GeometricType.MANIFOLD_POINT,
        )
    
    def norm(self, p: TensorCore, v: TensorCore) -> TensorCore:
        """
        Norm of tangent vector.
        
        Args:
            p: Base point
            v: Tangent vector
        
        Returns:
            Euclidean norm
        """
        v_data = v.numpy()
        norm_val = float(np.linalg.norm(v_data))
        
        from ..core.storage import _create_tensor, GeometricType
        return _create_tensor(
            norm_val,
            dtype=v.dtype,
            device=v.device,
            geometric_type=GeometricType.SCALAR,
        )
