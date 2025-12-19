"""
DavisTensor Manifolds: Sphere
==============================

The n-dimensional sphere S^n embedded in R^{n+1}.

The sphere is ideal for data with directional/angular structure:
- Unit vectors (embeddings that care about direction, not magnitude)
- Points on Earth (lat/lon)
- Protein structure angles
- Color hue (on a circle)

GPU-ready: Uses array_api for NumPy/CuPy compatibility.
"""

from __future__ import annotations
import numpy as np
import math
from typing import Tuple, Optional, Any

from .base import Manifold
from ..core.array_api import get_array_module, to_numpy, to_device


class Sphere(Manifold):
    """
    The n-sphere S^n = {x ∈ R^{n+1} : ||x|| = 1}.
    
    Constant positive curvature K = 1 (or 1/r² for radius r sphere).
    
    The tangent space T_x S^n = {v ∈ R^{n+1} : ⟨x, v⟩ = 0}
    consists of vectors orthogonal to the base point.
    
    Args:
        n: Dimension of sphere (lives in R^{n+1})
        
    Example:
        >>> S = Sphere(2)  # 2-sphere (surface of ball in R³)
        >>> x = S.random_point()
        >>> print(np.linalg.norm(x))  # ≈ 1.0
    """
    
    def __init__(self, n: int):
        """
        Args:
            n: Intrinsic dimension. S^n lives in R^{n+1}.
               - Sphere(1) = circle in R²
               - Sphere(2) = sphere in R³
        """
        self._dim = n
        self._eps = 1e-10
    
    @property
    def dim(self) -> int:
        return self._dim
    
    @property
    def ambient_dim(self) -> int:
        """Sphere S^n lives in R^{n+1}."""
        return self._dim + 1
    
    @property
    def curvature_type(self) -> str:
        return 'constant'
    
    def __repr__(self) -> str:
        return f"Sphere({self._dim})"
    
    @property
    def name(self) -> str:
        return self.__repr__()
    
    # =========================================================================
    # Point Operations
    # =========================================================================
    
    def random_point(self, *shape: int, device: str = 'cpu') -> Any:
        """Random point uniformly distributed on sphere."""
        from ..core import array_api as xp
        
        if shape:
            full_shape = (*shape, self.ambient_dim)
        else:
            full_shape = (self.ambient_dim,)
        
        # Random direction in ambient space, then normalize
        x = xp.randn(*full_shape, device=device)
        arr = get_array_module(x)
        return x / arr.linalg.norm(x, axis=-1, keepdims=True)
    
    def origin(self, *shape: int, device: str = 'cpu') -> Any:
        """
        Canonical origin: north pole (0, 0, ..., 0, 1).
        """
        from ..core import array_api as xp
        
        if shape:
            result = xp.zeros((*shape, self.ambient_dim), device=device)
            result[..., -1] = 1.0
            return result
        else:
            result = xp.zeros((self.ambient_dim,), device=device)
            result[-1] = 1.0
            return result
    
    def check_point(self, x: Any, atol: float = 1e-5) -> bool:
        """Check if x is on the unit sphere."""
        xp = get_array_module(x)
        norm = xp.linalg.norm(x, axis=-1)
        return bool(xp.all(xp.abs(norm - 1.0) < atol) and xp.all(xp.isfinite(x)))
    
    def project_point(self, x: Any) -> Any:
        """Project to unit sphere by normalizing."""
        xp = get_array_module(x)
        norm = xp.linalg.norm(x, axis=-1, keepdims=True)
        return x / xp.maximum(norm, self._eps)
    
    # =========================================================================
    # Tangent Space
    # =========================================================================
    
    def check_tangent(self, x: Any, v: Any, atol: float = 1e-5) -> bool:
        """Check if v ⊥ x (tangent to sphere at x)."""
        xp = get_array_module(x)
        dot = xp.sum(x * v, axis=-1)
        return bool(xp.all(xp.abs(dot) < atol))
    
    def project_tangent(self, x: Any, v: Any) -> Any:
        """
        Project v to tangent space T_x S^n.
        
        v_tangent = v - ⟨v, x⟩ x
        """
        xp = get_array_module(x)
        dot = xp.sum(x * v, axis=-1, keepdims=True)
        return v - dot * x
    
    def random_tangent(self, x: Any) -> Any:
        """Random unit tangent vector at x."""
        xp = get_array_module(x)
        from ..core import array_api
        device = array_api.get_device(x)
        v = array_api.randn(*x.shape, device=device)
        v = self.project_tangent(x, v)
        norm = xp.linalg.norm(v, axis=-1, keepdims=True)
        return v / xp.maximum(norm, self._eps)
    
    # =========================================================================
    # Metric (induced from R^{n+1})
    # =========================================================================
    
    def metric(self, x: Any) -> Any:
        """
        Metric tensor at x.
        
        The metric is induced from the ambient Euclidean space,
        restricted to the tangent space. In ambient coordinates,
        g = I - x x^T (projection to tangent space).
        """
        xp = get_array_module(x)
        batch_shape = x.shape[:-1]
        n = self.ambient_dim
        
        # g = I - x ⊗ x
        eye = xp.eye(n)
        x_outer = x[..., :, xp.newaxis] * x[..., xp.newaxis, :]  # (..., n, n)
        
        return xp.broadcast_to(eye, (*batch_shape, n, n)) - x_outer
    
    def inner(self, x: Any, u: Any, v: Any) -> Any:
        """
        Riemannian inner product ⟨u, v⟩_x.
        
        For the sphere with induced metric, this is just Euclidean inner product
        (assuming u, v are already tangent).
        """
        xp = get_array_module(x)
        return xp.sum(u * v, axis=-1)
    
    def norm(self, x: Any, v: Any) -> Any:
        """Riemannian norm (= Euclidean norm for tangent vectors)."""
        xp = get_array_module(x)
        return xp.linalg.norm(v, axis=-1)
    
    # =========================================================================
    # Exponential Map
    # =========================================================================
    
    def exp(self, x: Any, v: Any) -> Any:
        """
        Exponential map on sphere.
        
        exp_x(v) = cos(||v||) x + sin(||v||) v/||v||
        
        This traces out a great circle starting at x with velocity v.
        """
        xp = get_array_module(x)
        v_norm = xp.linalg.norm(v, axis=-1, keepdims=True)
        
        # Handle zero tangent vector
        is_small = v_norm < self._eps
        v_norm_safe = xp.where(is_small, 1.0, v_norm)
        v_unit = v / v_norm_safe
        
        cos_t = xp.cos(v_norm)
        sin_t = xp.sin(v_norm)
        
        result = cos_t * x + sin_t * v_unit
        
        # For very small v, exp_x(v) ≈ x
        result = xp.where(is_small, x, result)
        
        # Ensure on sphere
        return self.project_point(result)
    
    # =========================================================================
    # Logarithm Map
    # =========================================================================
    
    def log(self, x: Any, y: Any) -> Any:
        """
        Logarithm map on sphere.
        
        log_x(y) = d(x,y) * (y - ⟨x,y⟩x) / ||y - ⟨x,y⟩x||
        
        Returns tangent vector at x pointing toward y.
        """
        xp = get_array_module(x)
        
        # Project y to tangent space at x
        dot = xp.sum(x * y, axis=-1, keepdims=True)
        dot = xp.clip(dot, -1.0, 1.0)  # Numerical stability
        
        v = y - dot * x  # Tangent component
        v_norm = xp.linalg.norm(v, axis=-1, keepdims=True)
        
        # Distance = arccos(⟨x,y⟩)
        dist = xp.arccos(xp.clip(dot, -1.0, 1.0))
        
        # Handle case when x ≈ y (v_norm ≈ 0)
        is_small = v_norm < self._eps
        v_norm_safe = xp.where(is_small, 1.0, v_norm)
        
        result = dist * v / v_norm_safe
        result = xp.where(is_small, xp.zeros_like(v), result)
        
        return result
    
    # =========================================================================
    # Distance
    # =========================================================================
    
    def distance(self, x: Any, y: Any) -> Any:
        """
        Geodesic distance = arccos(⟨x, y⟩).
        """
        xp = get_array_module(x)
        dot = xp.sum(x * y, axis=-1)
        dot = xp.clip(dot, -1.0, 1.0)
        return xp.arccos(dot)
    
    def geodesic(self, x: Any, y: Any, t: float) -> Any:
        """
        Point at fraction t along geodesic (great circle).
        
        Uses spherical linear interpolation (slerp).
        """
        v = self.log(x, y)
        return self.exp(x, t * v)
    
    # =========================================================================
    # Parallel Transport
    # =========================================================================
    
    def parallel_transport(
        self, 
        x: Any, 
        y: Any, 
        v: Any
    ) -> Any:
        """
        Parallel transport v from T_x to T_y along geodesic.
        
        For sphere, parallel transport along geodesic from x to y:
            P_{x→y}(v) = v - (⟨log_x(y), v⟩ / d²) * (log_x(y) + log_y(x))
        
        where d = d(x, y).
        """
        xp = get_array_module(x)
        
        # Get tangent vectors
        log_xy = self.log(x, y)  # At x, pointing to y
        log_yx = self.log(y, x)  # At y, pointing to x
        
        d = self.distance(x, y)
        d_sq = d * d
        
        # Handle case when x ≈ y
        is_small = d_sq < self._eps * self._eps
        d_sq_safe = xp.where(is_small[..., xp.newaxis], 1.0, d_sq[..., xp.newaxis])
        
        # Coefficient
        coef = xp.sum(log_xy * v, axis=-1, keepdims=True) / d_sq_safe
        
        # Transport formula
        result = v - coef * (log_xy + log_yx)
        
        # When x ≈ y, parallel transport is identity
        result = xp.where(is_small[..., xp.newaxis], v, result)
        
        # Project to tangent space at y (numerical cleanup)
        return self.project_tangent(y, result)
    
    # =========================================================================
    # Curvature
    # =========================================================================
    
    def sectional_curvature(
        self, 
        x: Any, 
        u: Any, 
        v: Any
    ) -> Any:
        """Constant sectional curvature K = 1."""
        xp = get_array_module(x)
        batch_shape = x.shape[:-1] if x.ndim > 1 else ()
        return xp.ones(batch_shape if batch_shape else 1)
    
    def scalar_curvature(self, x: Any) -> Any:
        """Scalar curvature R = n(n-1) for unit sphere."""
        xp = get_array_module(x)
        n = self._dim
        batch_shape = x.shape[:-1] if x.ndim > 1 else ()
        return xp.full(batch_shape if batch_shape else 1, n * (n - 1))


# =============================================================================
# Tests
# =============================================================================

def test_sphere():
    """Test Sphere manifold."""
    print("=" * 60)
    print("Testing Sphere Manifold")
    print("=" * 60)
    
    S = Sphere(2)  # 2-sphere in R³
    print(f"Manifold: {S}")
    print(f"Intrinsic dim: {S.dim}, Ambient dim: {S.ambient_dim}")
    
    # Test random point
    print("\n1. Random point")
    x = S.random_point()
    norm = np.linalg.norm(x)
    print(f"   Shape: {x.shape}, ||x|| = {norm:.6f}")
    assert x.shape == (3,)
    assert abs(norm - 1.0) < 1e-10
    assert S.check_point(x)
    print("   ✅ PASS")
    
    # Test origin (north pole)
    print("\n2. Origin (north pole)")
    o = S.origin()
    print(f"   Origin: {o}")
    assert np.allclose(o, [0, 0, 1])
    print("   ✅ PASS")
    
    # Test projection
    print("\n3. Point projection")
    p = np.array([3., 4., 0.])
    p_proj = S.project_point(p)
    print(f"   [3,4,0] → {p_proj}")
    assert abs(np.linalg.norm(p_proj) - 1.0) < 1e-10
    assert np.allclose(p_proj, [0.6, 0.8, 0.0])
    print("   ✅ PASS")
    
    # Test tangent space
    print("\n4. Tangent space")
    x = np.array([0., 0., 1.])  # North pole
    v = np.array([1., 0., 0.])  # Tangent (perpendicular to x)
    assert S.check_tangent(x, v)
    
    w = np.array([1., 0., 1.])  # NOT tangent
    assert not S.check_tangent(x, w)
    
    w_proj = S.project_tangent(x, w)
    print(f"   [1,0,1] projected: {w_proj}")
    assert S.check_tangent(x, w_proj)
    assert np.allclose(w_proj, [1., 0., 0.])
    print("   ✅ PASS")
    
    # Test exp at north pole
    print("\n5. Exponential map at north pole")
    x = np.array([0., 0., 1.])  # North pole
    v = np.array([np.pi/2, 0., 0.])  # Move 90° toward equator
    y = S.exp(x, v)
    print(f"   exp([0,0,1], [π/2,0,0]) = {y}")
    # Should end up on equator
    assert abs(y[2]) < 1e-10  # z ≈ 0
    assert abs(np.linalg.norm(y) - 1.0) < 1e-10
    print("   ✅ PASS")
    
    # Test log (inverse of exp)
    print("\n6. Log/Exp inverse")
    x = S.random_point()
    y = S.random_point()
    v = S.log(x, y)
    y_recovered = S.exp(x, v)
    error = np.max(np.abs(y - y_recovered))
    print(f"   Max error: {error:.2e}")
    assert error < 1e-6  # Relaxed for numerical precision
    print("   ✅ PASS")
    
    # Test distance
    print("\n7. Distance")
    # North pole to equator should be π/2
    x = np.array([0., 0., 1.])  # North pole
    y = np.array([1., 0., 0.])  # Equator
    d = S.distance(x, y)
    print(f"   d(north pole, equator) = {d:.6f}")
    print(f"   Expected π/2 = {np.pi/2:.6f}")
    assert abs(d - np.pi/2) < 1e-10
    
    # Antipodal points should be π apart
    y = np.array([0., 0., -1.])  # South pole
    d = S.distance(x, y)
    print(f"   d(north pole, south pole) = {d:.6f}")
    print(f"   Expected π = {np.pi:.6f}")
    assert abs(d - np.pi) < 1e-10
    print("   ✅ PASS")
    
    # Test distance symmetry
    print("\n8. Distance symmetry")
    x = S.random_point()
    y = S.random_point()
    d_xy = S.distance(x, y)
    d_yx = S.distance(y, x)
    print(f"   d(x,y) = {d_xy:.6f}")
    print(f"   d(y,x) = {d_yx:.6f}")
    assert abs(d_xy - d_yx) < 1e-10
    print("   ✅ PASS")
    
    # Test geodesic
    print("\n9. Geodesic (great circle)")
    x = np.array([0., 0., 1.])  # North pole
    y = np.array([1., 0., 0.])  # Equator
    
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        p = S.geodesic(x, y, t)
        assert abs(np.linalg.norm(p) - 1.0) < 1e-10  # On sphere
        print(f"   t={t:.2f}: {p}")
    
    mid = S.geodesic(x, y, 0.5)
    d_to_mid = S.distance(x, mid)
    d_total = S.distance(x, y)
    print(f"   Distance to midpoint: {d_to_mid:.4f} (expected {d_total/2:.4f})")
    assert abs(d_to_mid - d_total/2) < 1e-6
    print("   ✅ PASS")
    
    # Test parallel transport
    print("\n10. Parallel transport")
    x = np.array([0., 0., 1.])  # North pole
    y = np.array([1., 0., 0.])  # Equator
    v = np.array([1., 0., 0.])  # Tangent at north pole
    
    v_transported = S.parallel_transport(x, y, v)
    print(f"   Transport [1,0,0] from north pole to equator")
    print(f"   Result: {v_transported}")
    
    # Should be tangent at y
    assert S.check_tangent(y, v_transported, atol=1e-6)
    # Norm should be preserved
    assert abs(np.linalg.norm(v_transported) - np.linalg.norm(v)) < 1e-6
    print("   ✅ PASS")
    
    # Test batched operations
    print("\n11. Batched operations")
    X = S.random_point(32)
    Y = S.random_point(32)
    D = S.distance(X, Y)
    print(f"   Batch shape: {D.shape}")
    assert D.shape == (32,)
    print(f"   Mean distance: {D.mean():.4f}")
    print("   ✅ PASS")
    
    # Test triangle inequality
    print("\n12. Triangle inequality")
    x = S.random_point()
    y = S.random_point()
    z = S.random_point()
    d_xy = S.distance(x, y)
    d_yz = S.distance(y, z)
    d_xz = S.distance(x, z)
    print(f"   d(x,y) = {d_xy:.4f}")
    print(f"   d(y,z) = {d_yz:.4f}")
    print(f"   d(x,z) = {d_xz:.4f}")
    print(f"   d(x,y) + d(y,z) = {d_xy + d_yz:.4f}")
    assert d_xz <= d_xy + d_yz + 1e-10
    print("   ✅ PASS")
    
    print("\n" + "=" * 60)
    print("All Sphere tests passed! ✅")
    print("=" * 60)


if __name__ == "__main__":
    test_sphere()
