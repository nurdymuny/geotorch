"""
DavisTensor Manifolds: Base Class and Euclidean
================================================

The geometry layer - defines the mathematical structure
of Riemannian manifolds.

GPU-ready: Uses array_api for NumPy/CuPy compatibility.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, Optional, TYPE_CHECKING, Any
import numpy as np
import math

from ..core.array_api import get_array_module, to_numpy, to_device, gpu_available

if TYPE_CHECKING:
    from ..core.storage import TensorCore


# =============================================================================
# Manifold Base Class
# =============================================================================

class Manifold(ABC):
    """
    Abstract base class for Riemannian manifolds.
    
    A manifold defines:
    - The space where points live
    - The metric tensor (how to measure distances/angles)
    - Geodesics (shortest paths)
    - Exponential and logarithm maps
    - Parallel transport
    - Curvature
    
    All operations work on raw numpy arrays for the core implementation.
    The TensorCore and ManifoldTensor wrappers handle the user interface.
    """
    
    @property
    @abstractmethod
    def dim(self) -> int:
        """Intrinsic dimension of the manifold."""
        ...
    
    @property
    def ambient_dim(self) -> int:
        """
        Dimension of ambient space.
        
        For embedded manifolds (Sphere in R^{n+1}), this differs from dim.
        Default: same as intrinsic dimension.
        """
        return self.dim
    
    @property
    def curvature_type(self) -> str:
        """
        Type of curvature: 'flat', 'constant', 'variable', or 'learned'.
        
        Used by compiler for optimization.
        """
        return 'variable'
    
    @property
    def name(self) -> str:
        """Human-readable name."""
        return self.__class__.__name__
    
    # =========================================================================
    # Point Operations
    # =========================================================================
    
    @abstractmethod
    def random_point(self, *shape: int, device: str = 'cpu') -> Any:
        """
        Sample random point(s) on the manifold.
        
        Args:
            shape: Batch dimensions. Final array has shape (*shape, ambient_dim)
            device: 'cpu' or 'cuda' - which device to create on
        
        Returns:
            Points on the manifold
        """
        ...
    
    @abstractmethod
    def origin(self, *shape: int, device: str = 'cpu') -> Any:
        """
        Canonical origin/identity point.
        
        For Euclidean: zero vector
        For Hyperbolic: center of Poincaré ball
        For Sphere: north pole
        For SPD: identity matrix
        
        Args:
            device: 'cpu' or 'cuda'
        """
        ...
    
    @abstractmethod
    def check_point(self, x: Any, atol: float = 1e-5) -> bool:
        """
        Check if x is on the manifold (within tolerance).
        """
        ...
    
    @abstractmethod
    def project_point(self, x: Any) -> Any:
        """
        Project point from ambient space onto manifold.
        """
        ...
    
    # =========================================================================
    # Tangent Space Operations
    # =========================================================================
    
    @abstractmethod
    def check_tangent(self, x: Any, v: Any, atol: float = 1e-5) -> bool:
        """
        Check if v is in tangent space T_x M.
        """
        ...
    
    @abstractmethod
    def project_tangent(self, x: Any, v: Any) -> Any:
        """
        Project v onto tangent space T_x M.
        """
        ...
    
    def random_tangent(self, x: Any, device: str = 'cpu') -> Any:
        """
        Sample random tangent vector at x.
        
        Default: random vector projected to tangent space.
        """
        from ..core.array_api import randn
        v = randn(*x.shape, device=device)
        return self.project_tangent(x, v)
    
    def zero_tangent(self, x: Any) -> Any:
        """Zero tangent vector at x."""
        xp = get_array_module(x)
        return xp.zeros_like(x)
    
    # =========================================================================
    # Metric Operations
    # =========================================================================
    
    @abstractmethod
    def metric(self, x: Any) -> Any:
        """
        Metric tensor g_ij(x) at point x.
        
        Args:
            x: Point(s) on manifold, shape (..., ambient_dim)
        
        Returns:
            Metric tensor, shape (..., dim, dim)
        """
        ...
    
    @abstractmethod
    def inner(self, x: Any, u: Any, v: Any) -> Any:
        """
        Riemannian inner product ⟨u, v⟩_x.
        
        Args:
            x: Base point, shape (..., ambient_dim)
            u, v: Tangent vectors at x, shape (..., ambient_dim)
        
        Returns:
            Inner product, shape (...)
        """
        ...
    
    def norm(self, x: Any, v: Any) -> Any:
        """
        Riemannian norm ||v||_x = sqrt(⟨v, v⟩_x).
        """
        xp = get_array_module(x)
        return xp.sqrt(xp.maximum(self.inner(x, v, v), 0))
    
    # =========================================================================
    # Exponential and Logarithm Maps
    # =========================================================================
    
    @abstractmethod
    def exp(self, x: Any, v: Any) -> Any:
        """
        Exponential map: exp_x(v).
        
        Move from x along geodesic in direction v for unit time.
        
        Args:
            x: Base point, shape (..., ambient_dim)
            v: Tangent vector at x, shape (..., ambient_dim)
        
        Returns:
            Point on manifold, shape (..., ambient_dim)
        """
        ...
    
    @abstractmethod
    def log(self, x: Any, y: Any) -> Any:
        """
        Logarithm map: log_x(y).
        
        Tangent vector at x pointing toward y.
        
        Args:
            x: Base point, shape (..., ambient_dim)
            y: Target point, shape (..., ambient_dim)
        
        Returns:
            Tangent vector at x, shape (..., ambient_dim)
        """
        ...
    
    def distance(self, x: Any, y: Any) -> Any:
        """
        Geodesic distance d(x, y).
        
        Default: ||log_x(y)||_x
        """
        v = self.log(x, y)
        return self.norm(x, v)
    
    def geodesic(self, x: Any, y: Any, t: float) -> Any:
        """
        Point at fraction t ∈ [0, 1] along geodesic from x to y.
        
        γ(t) = exp_x(t * log_x(y))
        """
        v = self.log(x, y)
        return self.exp(x, t * v)
    
    # =========================================================================
    # Parallel Transport
    # =========================================================================
    
    @abstractmethod
    def parallel_transport(
        self, 
        x: Any, 
        y: Any, 
        v: Any
    ) -> Any:
        """
        Parallel transport v from T_x M to T_y M along geodesic.
        
        Args:
            x: Source point
            y: Target point
            v: Tangent vector at x
        
        Returns:
            Tangent vector at y
        """
        ...
    
    # =========================================================================
    # Curvature (optional - not all manifolds implement)
    # =========================================================================
    
    def sectional_curvature(
        self, 
        x: Any, 
        u: Any, 
        v: Any
    ) -> Any:
        """
        Sectional curvature K(u, v) of plane spanned by u, v at x.
        """
        raise NotImplementedError(f"{self.name} does not implement sectional_curvature")
    
    def scalar_curvature(self, x: Any) -> Any:
        """
        Scalar curvature R(x).
        """
        raise NotImplementedError(f"{self.name} does not implement scalar_curvature")
    
    # =========================================================================
    # String representation
    # =========================================================================
    
    def __repr__(self) -> str:
        return f"{self.name}({self.dim})"


# =============================================================================
# Euclidean Manifold
# =============================================================================

class Euclidean(Manifold):
    """
    Flat Euclidean space R^n.
    
    This is the "trivial" manifold where:
    - Curvature is zero everywhere
    - exp_x(v) = x + v
    - log_x(y) = y - x
    - Parallel transport is identity
    - Metric is identity matrix
    
    Useful for:
    - Testing (baseline geometry)
    - Components of product manifolds
    - Standard deep learning
    """
    
    def __init__(self, n: int):
        """
        Args:
            n: Dimension of Euclidean space
        """
        self._dim = n
    
    @property
    def dim(self) -> int:
        return self._dim
    
    @property
    def curvature_type(self) -> str:
        return 'flat'
    
    # =========================================================================
    # Point Operations
    # =========================================================================
    
    def random_point(self, *shape: int, device: str = 'cpu') -> Any:
        from ..core.array_api import randn
        return randn(*shape, self._dim, device=device)
    
    def origin(self, *shape: int, device: str = 'cpu') -> Any:
        from ..core.array_api import zeros
        if shape:
            return zeros(*shape, self._dim, device=device)
        return zeros(self._dim, device=device)
    
    def check_point(self, x: Any, atol: float = 1e-5) -> bool:
        # All finite vectors are valid points
        xp = get_array_module(x)
        return bool(xp.all(xp.isfinite(x)))
    
    def project_point(self, x: Any) -> Any:
        # Identity - all points are on R^n
        return x
    
    # =========================================================================
    # Tangent Space (= R^n at every point)
    # =========================================================================
    
    def check_tangent(self, x: Any, v: Any, atol: float = 1e-5) -> bool:
        # All vectors are valid tangent vectors
        xp = get_array_module(v)
        return bool(xp.all(xp.isfinite(v)))
    
    def project_tangent(self, x: Any, v: Any) -> Any:
        # Identity
        return v
    
    # =========================================================================
    # Metric (Identity)
    # =========================================================================
    
    def metric(self, x: Any) -> Any:
        xp = get_array_module(x)
        batch_shape = x.shape[:-1]
        eye = xp.eye(self._dim)
        # Broadcast to batch shape
        return xp.broadcast_to(eye, (*batch_shape, self._dim, self._dim)).copy()
    
    def inner(self, x: Any, u: Any, v: Any) -> Any:
        # Standard Euclidean inner product
        xp = get_array_module(u)
        return xp.sum(u * v, axis=-1)
    
    def norm(self, x: Any, v: Any) -> Any:
        xp = get_array_module(v)
        return xp.linalg.norm(v, axis=-1)
    
    # =========================================================================
    # Exp / Log (trivial)
    # =========================================================================
    
    def exp(self, x: Any, v: Any) -> Any:
        return x + v
    
    def log(self, x: Any, y: Any) -> Any:
        return y - x
    
    def distance(self, x: Any, y: Any) -> Any:
        xp = get_array_module(x)
        return xp.linalg.norm(y - x, axis=-1)
    
    def geodesic(self, x: Any, y: Any, t: float) -> Any:
        return x + t * (y - x)
    
    # =========================================================================
    # Parallel Transport (identity)
    # =========================================================================
    
    def parallel_transport(
        self, 
        x: Any, 
        y: Any, 
        v: Any
    ) -> Any:
        return v
    
    # =========================================================================
    # Curvature (zero)
    # =========================================================================
    
    def sectional_curvature(
        self, 
        x: Any, 
        u: Any, 
        v: Any
    ) -> Any:
        xp = get_array_module(x)
        return xp.zeros(x.shape[:-1])
    
    def scalar_curvature(self, x: Any) -> Any:
        xp = get_array_module(x)
        return xp.zeros(x.shape[:-1])


# =============================================================================
# Tests
# =============================================================================

def test_euclidean():
    """Test Euclidean manifold."""
    print("=" * 60)
    print("Testing Euclidean Manifold")
    print("=" * 60)
    
    E = Euclidean(5)
    print(f"Manifold: {E}")
    
    # Test random point
    print("\n1. Random point")
    x = E.random_point(10)
    print(f"   Shape: {x.shape}")
    assert x.shape == (10, 5)
    assert E.check_point(x)
    print("   ✅ PASS")
    
    # Test origin
    print("\n2. Origin")
    o = E.origin()
    print(f"   Shape: {o.shape}, norm: {np.linalg.norm(o)}")
    assert o.shape == (5,)
    assert np.allclose(o, 0)
    print("   ✅ PASS")
    
    # Test metric
    print("\n3. Metric tensor")
    x = E.random_point()
    G = E.metric(x)
    print(f"   Metric shape: {G.shape}")
    assert np.allclose(G, np.eye(5))
    print("   ✅ PASS")
    
    # Test inner product
    print("\n4. Inner product")
    u = np.array([1., 0., 0., 0., 0.])
    v = np.array([1., 1., 0., 0., 0.])
    ip = E.inner(x, u, v)
    print(f"   ⟨[1,0,0,0,0], [1,1,0,0,0]⟩ = {ip}")
    assert ip == 1.0
    print("   ✅ PASS")
    
    # Test exp/log inverse
    print("\n5. Exp/Log inverse")
    x = E.random_point()
    y = E.random_point()
    v = E.log(x, y)
    y_recovered = E.exp(x, v)
    error = np.max(np.abs(y - y_recovered))
    print(f"   Max error: {error:.2e}")
    assert error < 1e-10
    print("   ✅ PASS")
    
    # Test distance
    print("\n6. Distance")
    x = np.zeros(5)
    y = np.array([3., 4., 0., 0., 0.])
    d = E.distance(x, y)
    print(f"   d([0,...], [3,4,0,...]) = {d}")
    assert d == 5.0
    print("   ✅ PASS")
    
    # Test geodesic
    print("\n7. Geodesic")
    x = np.zeros(5)
    y = np.ones(5) * 2
    mid = E.geodesic(x, y, 0.5)
    expected = np.ones(5)
    error = np.max(np.abs(mid - expected))
    print(f"   Midpoint: {mid[:3]}... (expected [1,1,1,...])")
    assert error < 1e-10
    print("   ✅ PASS")
    
    # Test parallel transport (identity)
    print("\n8. Parallel transport")
    x = E.random_point()
    y = E.random_point()
    v = E.random_tangent(x)
    v_transported = E.parallel_transport(x, y, v)
    assert np.allclose(v, v_transported)
    print("   Transport is identity (as expected for flat space)")
    print("   ✅ PASS")
    
    # Test batched operations
    print("\n9. Batched operations")
    x = E.random_point(32)  # Batch of 32
    y = E.random_point(32)
    d = E.distance(x, y)
    print(f"   Distances shape: {d.shape}")
    assert d.shape == (32,)
    print("   ✅ PASS")
    
    print("\n" + "=" * 60)
    print("All Euclidean tests passed! ✅")
    print("=" * 60)


if __name__ == "__main__":
    test_euclidean()
