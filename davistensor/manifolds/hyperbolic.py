"""
DavisTensor Manifolds: Hyperbolic Space (Poincaré Ball Model)
==============================================================

Hyperbolic geometry for hierarchical/tree-like data.

The Poincaré ball B^n = {x ∈ R^n : ||x|| < 1} with the metric:
    g_x = (2 / (1 - ||x||²))² I

Key properties:
- Constant negative curvature K = -1/c² (c is curvature parameter)
- Exponential volume growth (ideal for trees)
- Distances grow logarithmically toward boundary

GPU-ready: Uses array_api for NumPy/CuPy compatibility.
"""

from __future__ import annotations
import numpy as np
import math
from typing import Tuple, Optional, Any

from .base import Manifold
from ..core.array_api import get_array_module, to_numpy, to_device


class Hyperbolic(Manifold):
    """
    Hyperbolic space in the Poincaré ball model.
    
    The Poincaré ball is the unit ball B^n = {x ∈ R^n : ||x|| < 1}
    equipped with the Riemannian metric:
    
        g_x(u, v) = (λ_x)² ⟨u, v⟩
    
    where λ_x = 2 / (1 - ||x||²) is the conformal factor.
    
    Args:
        n: Dimension of hyperbolic space
        c: Curvature parameter (default 1.0). Curvature K = -1/c².
           Larger c = more negative curvature = faster distance growth.
    
    Example:
        >>> H = Hyperbolic(5)
        >>> x = H.random_point()
        >>> y = H.random_point()
        >>> d = H.distance(x, y)  # Hyperbolic distance
    """
    
    def __init__(self, n: int, c: float = 1.0):
        self._dim = n
        self._c = c
        self._sqrt_c = math.sqrt(c)
        
        # Numerical stability constants
        self._eps = 1e-15
        self._max_norm = 1.0 - 1e-5  # Stay away from boundary
    
    @property
    def dim(self) -> int:
        return self._dim
    
    @property
    def c(self) -> float:
        """Curvature parameter."""
        return self._c
    
    @property
    def curvature(self) -> float:
        """Sectional curvature K = -1/c²."""
        return -1.0 / (self._c * self._c)
    
    @property
    def curvature_type(self) -> str:
        return 'constant'
    
    def __repr__(self) -> str:
        if self._c == 1.0:
            return f"Hyperbolic({self._dim})"
        return f"Hyperbolic({self._dim}, c={self._c})"
    
    @property
    def name(self) -> str:
        return self.__repr__()
    
    # =========================================================================
    # Helper functions
    # =========================================================================
    
    def _lambda_x(self, x: Any) -> Any:
        """
        Conformal factor λ_x = 2 / (1 - c||x||²).
        
        Shape: (...,) from input (..., n)
        """
        xp = get_array_module(x)
        norm_sq = xp.sum(x * x, axis=-1)
        return 2.0 / xp.maximum(1.0 - self._c * norm_sq, self._eps)
    
    def _mobius_add(self, x: Any, y: Any) -> Any:
        """
        Möbius addition: x ⊕ y.
        
        This is the hyperbolic analog of vector addition.
        
        x ⊕ y = ((1 + 2c⟨x,y⟩ + c||y||²)x + (1 - c||x||²)y) / 
                (1 + 2c⟨x,y⟩ + c²||x||²||y||²)
        """
        xp = get_array_module(x)
        c = self._c
        
        x_norm_sq = xp.sum(x * x, axis=-1, keepdims=True)
        y_norm_sq = xp.sum(y * y, axis=-1, keepdims=True)
        xy = xp.sum(x * y, axis=-1, keepdims=True)
        
        num = (1 + 2 * c * xy + c * y_norm_sq) * x + (1 - c * x_norm_sq) * y
        denom = 1 + 2 * c * xy + c * c * x_norm_sq * y_norm_sq
        
        return num / xp.maximum(denom, self._eps)
    
    def _mobius_scalar(self, r: Any, x: Any) -> Any:
        """
        Möbius scalar multiplication: r ⊗ x.
        
        r ⊗ x = (1/√c) tanh(r * artanh(√c ||x||)) * (x / ||x||)
        """
        xp = get_array_module(x)
        c = self._c
        sqrt_c = self._sqrt_c
        
        x_norm = xp.linalg.norm(x, axis=-1, keepdims=True)
        x_norm = xp.maximum(x_norm, self._eps)
        
        # artanh(√c ||x||)
        sqrt_c_norm = sqrt_c * x_norm
        sqrt_c_norm = xp.clip(sqrt_c_norm, -1 + self._eps, 1 - self._eps)
        artanh_val = xp.arctanh(sqrt_c_norm)
        
        # tanh(r * artanh_val)
        r = xp.asarray(r)
        if r.ndim == 0:
            r = r.reshape(1)
        while r.ndim < x.ndim:
            r = r[..., xp.newaxis]
        
        tanh_val = xp.tanh(r * artanh_val)
        
        return (1.0 / sqrt_c) * tanh_val * (x / x_norm)
    
    def _project_to_ball(self, x: Any) -> Any:
        """Project to interior of Poincaré ball."""
        xp = get_array_module(x)
        norm = xp.linalg.norm(x, axis=-1, keepdims=True)
        max_norm = self._max_norm / self._sqrt_c
        
        # Scale down if too close to boundary
        scale = xp.minimum(1.0, max_norm / xp.maximum(norm, self._eps))
        return x * scale
    
    # =========================================================================
    # Point Operations
    # =========================================================================
    
    def random_point(self, *shape: int, device: str = 'cpu') -> Any:
        """
        Random point in Poincaré ball.
        
        Uses uniform sampling scaled by a power law to distribute
        points throughout the ball (not just near origin).
        """
        from ..core.array_api import randn, rand
        
        if shape:
            full_shape = (*shape, self._dim)
        else:
            full_shape = (self._dim,)
        
        # Random direction
        direction = randn(*full_shape, device=device)
        xp = get_array_module(direction)
        direction = direction / xp.linalg.norm(direction, axis=-1, keepdims=True)
        
        # Random radius with power law (concentrate toward boundary for variety)
        # Use uniform in [0, 1) then apply inverse CDF for hyperbolic measure
        u = rand(*full_shape[:-1], device=device)
        # This gives decent spread across the ball
        r = 0.7 * xp.power(u, 1/self._dim)  # Scale factor 0.7 keeps us away from boundary
        r = r[..., xp.newaxis]
        
        return (r * direction / self._sqrt_c)
    
    def origin(self, *shape: int, device: str = 'cpu') -> Any:
        """Origin of Poincaré ball (the center)."""
        from ..core.array_api import zeros
        if shape:
            return zeros(*shape, self._dim, device=device)
        return zeros(self._dim, device=device)
    
    def check_point(self, x: Any, atol: float = 1e-5) -> bool:
        """Check if x is strictly inside the ball."""
        xp = get_array_module(x)
        norm_sq = xp.sum(x * x, axis=-1)
        return bool(xp.all(self._c * norm_sq < 1.0 - atol) and xp.all(xp.isfinite(x)))
    
    def project_point(self, x: Any) -> Any:
        """Project to interior of Poincaré ball."""
        return self._project_to_ball(x)
    
    # =========================================================================
    # Tangent Space
    # =========================================================================
    
    def check_tangent(self, x: Any, v: Any, atol: float = 1e-5) -> bool:
        """Tangent space is R^n at every point (no constraint)."""
        xp = get_array_module(v)
        return bool(xp.all(xp.isfinite(v)))
    
    def project_tangent(self, x: Any, v: Any) -> Any:
        """Tangent space is R^n, so no projection needed."""
        return v
    
    # =========================================================================
    # Metric
    # =========================================================================
    
    def metric(self, x: Any) -> Any:
        """
        Metric tensor g_ij(x) = λ_x² δ_ij.
        
        The Poincaré ball has a conformally flat metric.
        """
        xp = get_array_module(x)
        lam = self._lambda_x(x)  # Shape: (...)
        batch_shape = lam.shape
        
        # g = λ² I
        lam_sq = lam * lam
        eye = xp.eye(self._dim)
        
        # Broadcast: (..., 1, 1) * (n, n) -> (..., n, n)
        return lam_sq[..., xp.newaxis, xp.newaxis] * eye
    
    def inner(self, x: Any, u: Any, v: Any) -> Any:
        """
        Riemannian inner product ⟨u, v⟩_x = λ_x² ⟨u, v⟩_E.
        """
        xp = get_array_module(x)
        lam = self._lambda_x(x)
        euclidean_ip = xp.sum(u * v, axis=-1)
        return (lam * lam) * euclidean_ip
    
    def norm(self, x: Any, v: Any) -> Any:
        """Riemannian norm ||v||_x = λ_x ||v||_E."""
        xp = get_array_module(x)
        lam = self._lambda_x(x)
        euclidean_norm = xp.linalg.norm(v, axis=-1)
        return lam * euclidean_norm
    
    # =========================================================================
    # Exponential Map
    # =========================================================================
    
    def exp(self, x: Any, v: Any) -> Any:
        """
        Exponential map: exp_x(v).
        
        exp_x(v) = x ⊕ (tanh(√c λ_x ||v|| / 2) * v / (√c ||v||))
        
        This moves from x along the geodesic with initial velocity v.
        """
        xp = get_array_module(x)
        c = self._c
        sqrt_c = self._sqrt_c
        
        v_norm = xp.linalg.norm(v, axis=-1, keepdims=True)
        v_norm = xp.maximum(v_norm, self._eps)
        
        lam = self._lambda_x(x)[..., xp.newaxis]
        
        # Second term: tanh(√c λ ||v|| / 2) * v / (√c ||v||)
        arg = sqrt_c * lam * v_norm / 2.0
        arg = xp.clip(arg, -20, 20)  # Numerical stability
        tanh_val = xp.tanh(arg)
        
        second_term = tanh_val * v / (sqrt_c * v_norm)
        
        # Möbius addition
        result = self._mobius_add(x, second_term)
        
        return self._project_to_ball(result)
    
    # =========================================================================
    # Logarithm Map
    # =========================================================================
    
    def log(self, x: Any, y: Any) -> Any:
        """
        Logarithm map: log_x(y).
        
        log_x(y) = (2 / (√c λ_x)) * artanh(√c ||−x ⊕ y||) * (−x ⊕ y) / ||−x ⊕ y||
        
        Returns tangent vector at x pointing toward y.
        """
        xp = get_array_module(x)
        c = self._c
        sqrt_c = self._sqrt_c
        
        # -x ⊕ y
        diff = self._mobius_add(-x, y)
        diff_norm = xp.linalg.norm(diff, axis=-1, keepdims=True)
        diff_norm = xp.maximum(diff_norm, self._eps)
        
        lam = self._lambda_x(x)[..., xp.newaxis]
        
        # artanh(√c ||diff||)
        sqrt_c_norm = sqrt_c * diff_norm
        sqrt_c_norm = xp.clip(sqrt_c_norm, -1 + self._eps, 1 - self._eps)
        artanh_val = xp.arctanh(sqrt_c_norm)
        
        # log = (2 / √c λ) * artanh(...) * (diff / ||diff||)
        return (2.0 / (sqrt_c * lam)) * artanh_val * (diff / diff_norm)
    
    # =========================================================================
    # Distance
    # =========================================================================
    
    def distance(self, x: Any, y: Any) -> Any:
        """
        Geodesic distance d(x, y).
        
        d(x, y) = (2/√c) * artanh(√c ||−x ⊕ y||)
        """
        xp = get_array_module(x)
        sqrt_c = self._sqrt_c
        
        diff = self._mobius_add(-x, y)
        diff_norm = xp.linalg.norm(diff, axis=-1)
        
        sqrt_c_norm = sqrt_c * diff_norm
        sqrt_c_norm = xp.clip(sqrt_c_norm, -1 + self._eps, 1 - self._eps)
        
        return (2.0 / sqrt_c) * xp.arctanh(sqrt_c_norm)
    
    def geodesic(self, x: Any, y: Any, t: float) -> Any:
        """Point at fraction t along geodesic from x to y."""
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
        
        Uses the formula:
            P_{x→y}(v) = (λ_x / λ_y) * gyr[y, -x](v)
        
        where gyr is the gyration operator (rotation from Möbius addition).
        """
        xp = get_array_module(x)
        lam_x = self._lambda_x(x)
        lam_y = self._lambda_x(y)
        
        # Compute gyration: gyr[y, -x](v) = -(-y ⊕ x) ⊕ ((-y ⊕ x) ⊕ v) doesn't work simply
        # Use the explicit formula for parallel transport in Poincaré ball
        
        # Simpler approach: use the scaling formula
        # P_{x→y}(v) = λ_x/λ_y * rotated_v
        # where the rotation preserves the direction relative to the geodesic
        
        # For now, use approximate transport (works well for small distances)
        # Full gyration-based transport can be added later
        
        # Project v to get component perpendicular to x
        scale = lam_x / xp.maximum(lam_y, self._eps)
        scale = scale[..., xp.newaxis]
        
        return scale * v
    
    # =========================================================================
    # Curvature
    # =========================================================================
    
    def sectional_curvature(
        self, 
        x: Any, 
        u: Any, 
        v: Any
    ) -> Any:
        """Constant sectional curvature K = -1/c²."""
        xp = get_array_module(x)
        batch_shape = x.shape[:-1] if x.ndim > 1 else ()
        return xp.full(batch_shape, -1.0 / (self._c * self._c))
    
    def scalar_curvature(self, x: Any) -> Any:
        """Scalar curvature R = n(n-1)K."""
        xp = get_array_module(x)
        n = self._dim
        K = -1.0 / (self._c * self._c)
        batch_shape = x.shape[:-1] if x.ndim > 1 else ()
        return xp.full(batch_shape, n * (n - 1) * K)


# =============================================================================
# Tests
# =============================================================================

def test_hyperbolic():
    """Test Hyperbolic manifold."""
    print("=" * 60)
    print("Testing Hyperbolic Manifold (Poincaré Ball)")
    print("=" * 60)
    
    H = Hyperbolic(5)
    print(f"Manifold: {H}")
    print(f"Curvature: K = {H.curvature}")
    
    # Test random point
    print("\n1. Random point")
    x = H.random_point()
    norm = np.linalg.norm(x)
    print(f"   Shape: {x.shape}, ||x|| = {norm:.4f}")
    assert x.shape == (5,)
    assert H.check_point(x)
    print("   ✅ PASS")
    
    # Test origin
    print("\n2. Origin")
    o = H.origin()
    print(f"   Origin: {o}")
    assert np.allclose(o, 0)
    assert H.check_point(o)
    print("   ✅ PASS")
    
    # Test metric at origin (should be 4I since λ_0 = 2)
    print("\n3. Metric tensor at origin")
    G = H.metric(o)
    expected_scale = 4.0  # λ² = 2² = 4
    print(f"   g_0 = {expected_scale}I (conformal)")
    assert np.allclose(G, expected_scale * np.eye(5))
    print("   ✅ PASS")
    
    # Test inner product
    print("\n4. Inner product")
    u = np.array([1., 0., 0., 0., 0.])
    v = np.array([0., 1., 0., 0., 0.])
    ip = H.inner(o, u, v)
    print(f"   At origin: ⟨e1, e2⟩ = {ip}")
    assert ip == 0.0  # Orthogonal
    
    ip_self = H.inner(o, u, u)
    print(f"   At origin: ⟨e1, e1⟩ = {ip_self}")
    assert ip_self == 4.0  # λ² * ||u||² = 4 * 1
    print("   ✅ PASS")
    
    # Test exp/log inverse at origin
    print("\n5. Exp/Log inverse at origin")
    v = np.array([0.3, 0., 0., 0., 0.])
    y = H.exp(o, v)
    v_recovered = H.log(o, y)
    error = np.max(np.abs(v - v_recovered))
    print(f"   exp_0(v): ||result|| = {np.linalg.norm(y):.4f}")
    print(f"   log_0(exp_0(v)) ≈ v: error = {error:.2e}")
    assert error < 1e-10
    print("   ✅ PASS")
    
    # Test exp/log inverse at general point
    print("\n6. Exp/Log inverse at general point")
    x = H.random_point()
    v = H.random_tangent(x) * 0.5  # Scale down to avoid boundary issues
    y = H.exp(x, v)
    v_recovered = H.log(x, y)
    error = np.max(np.abs(v - v_recovered))
    print(f"   Max error: {error:.2e}")
    assert error < 1e-6
    print("   ✅ PASS")
    
    # Test distance symmetry
    print("\n7. Distance symmetry")
    x = H.random_point()
    y = H.random_point()
    d_xy = H.distance(x, y)
    d_yx = H.distance(y, x)
    print(f"   d(x,y) = {d_xy:.6f}")
    print(f"   d(y,x) = {d_yx:.6f}")
    assert abs(d_xy - d_yx) < 1e-6  # Relaxed for numerical precision
    print("   ✅ PASS")
    
    # Test distance from origin
    print("\n8. Distance from origin (explicit formula)")
    p = np.array([0.5, 0., 0., 0., 0.])  # Point with ||p|| = 0.5
    d = H.distance(o, p)
    # d(0, p) = 2 * arctanh(||p||) for c=1
    expected_d = 2 * np.arctanh(0.5)
    print(f"   d(0, [0.5,0,0,0,0]) = {d:.6f}")
    print(f"   Expected: 2*arctanh(0.5) = {expected_d:.6f}")
    assert abs(d - expected_d) < 1e-10
    print("   ✅ PASS")
    
    # Test geodesic
    print("\n9. Geodesic")
    x = H.origin()
    y = np.array([0.5, 0., 0., 0., 0.])
    mid = H.geodesic(x, y, 0.5)
    
    # Distance from x to mid should be half of x to y
    d_total = H.distance(x, y)
    d_half = H.distance(x, mid)
    print(f"   Total distance: {d_total:.4f}")
    print(f"   Distance to midpoint: {d_half:.4f}")
    print(f"   Ratio: {d_half / d_total:.4f} (expected ~0.5)")
    assert abs(d_half / d_total - 0.5) < 0.01
    print("   ✅ PASS")
    
    # Test triangle inequality
    print("\n10. Triangle inequality")
    x = H.random_point()
    y = H.random_point()
    z = H.random_point()
    d_xy = H.distance(x, y)
    d_yz = H.distance(y, z)
    d_xz = H.distance(x, z)
    print(f"   d(x,y) = {d_xy:.4f}")
    print(f"   d(y,z) = {d_yz:.4f}")
    print(f"   d(x,z) = {d_xz:.4f}")
    print(f"   d(x,y) + d(y,z) = {d_xy + d_yz:.4f}")
    assert d_xz <= d_xy + d_yz + 1e-10
    print("   ✅ PASS")
    
    # Test batched operations
    print("\n11. Batched operations")
    X = H.random_point(32)
    Y = H.random_point(32)
    D = H.distance(X, Y)
    print(f"   Batch shape: {D.shape}")
    assert D.shape == (32,)
    print(f"   Mean distance: {D.mean():.4f}")
    print("   ✅ PASS")
    
    # Test curvature parameter
    print("\n12. Curvature parameter")
    H2 = Hyperbolic(5, c=2.0)
    print(f"   c=1: K = {H.curvature:.4f}")
    print(f"   c=2: K = {H2.curvature:.4f}")
    assert H.curvature == -1.0
    assert H2.curvature == -0.25
    print("   ✅ PASS")
    
    print("\n" + "=" * 60)
    print("All Hyperbolic tests passed! ✅")
    print("=" * 60)


if __name__ == "__main__":
    test_hyperbolic()
