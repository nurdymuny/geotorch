"""
DavisTensor Product Manifold Specification
==========================================

Cartesian product of manifolds: M₁ × M₂ × ... × Mₖ

Key insight: Different aspects of data need different geometries.
- Hierarchical structure → Hyperbolic
- Directional similarity → Sphere  
- Continuous attributes → Euclidean
- Covariance structure → SPD

Product manifolds let you combine them!

IMPLEMENTATION SPEC
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from .base import Manifold


class ProductManifold(Manifold):
    """
    Cartesian product of Riemannian manifolds.
    
    M = M₁ × M₂ × ... × Mₖ
    
    Properties:
    - Points: (x₁, x₂, ..., xₖ) where xᵢ ∈ Mᵢ
    - Metric: Block diagonal (independent components)
    - Distance: d² = d₁² + d₂² + ... + dₖ²
    - Exp/Log: Component-wise
    
    Storage format:
    - Points stored as concatenated vectors
    - Slices tracked by component dimensions
    
    Parameters
    ----------
    manifolds : List[Manifold]
        Component manifolds
    """
    
    def __init__(self, manifolds: List[Manifold]):
        if len(manifolds) < 2:
            raise ValueError("ProductManifold requires at least 2 manifolds")
        
        self._manifolds = manifolds
        self._n_components = len(manifolds)
        
        # Compute dimensions and slice indices
        self._dims = [m.dim for m in manifolds]
        self._ambient_dims = [m.ambient_dim for m in manifolds]
        
        # Slice boundaries for extracting components
        self._slices = []
        start = 0
        for dim in self._ambient_dims:
            self._slices.append((start, start + dim))
            start += dim
    
    @property
    def manifolds(self) -> List[Manifold]:
        """Component manifolds."""
        return self._manifolds
    
    @property
    def n_components(self) -> int:
        """Number of component manifolds."""
        return self._n_components
    
    @property
    def dim(self) -> int:
        """Total intrinsic dimension."""
        return sum(self._dims)
    
    @property
    def ambient_dim(self) -> int:
        """Total ambient dimension."""
        return sum(self._ambient_dims)
    
    @property
    def curvature_type(self) -> str:
        # Variable unless all components are flat
        if all(m.curvature_type == 'flat' for m in self._manifolds):
            return 'flat'
        return 'variable'
    
    @property
    def name(self) -> str:
        names = [m.name for m in self._manifolds]
        return " × ".join(names)
    
    # =========================================================================
    # Component Access
    # =========================================================================
    
    def get_component(self, x: np.ndarray, index: int) -> np.ndarray:
        """
        Extract component from product point.
        
        Parameters
        ----------
        x : np.ndarray
            Point on product manifold, shape (..., ambient_dim)
        index : int
            Component index (0 to n_components-1)
        
        Returns
        -------
        Component point, shape (..., component_ambient_dim)
        """
        if index < 0 or index >= self._n_components:
            raise IndexError(f"Component index {index} out of range [0, {self._n_components})")
        
        start, end = self._slices[index]
        return x[..., start:end]
    
    def set_component(self, x: np.ndarray, index: int, value: np.ndarray) -> np.ndarray:
        """
        Set component in product point (returns new array).
        """
        if index < 0 or index >= self._n_components:
            raise IndexError(f"Component index {index} out of range")
        
        result = x.copy()
        start, end = self._slices[index]
        result[..., start:end] = value
        return result
    
    def split(self, x: np.ndarray) -> List[np.ndarray]:
        """
        Split product point into components.
        
        Returns list of component arrays.
        """
        return [self.get_component(x, i) for i in range(self._n_components)]
    
    def combine(self, components: List[np.ndarray]) -> np.ndarray:
        """
        Combine component arrays into product point.
        """
        if len(components) != self._n_components:
            raise ValueError(f"Expected {self._n_components} components, got {len(components)}")
        
        return np.concatenate(components, axis=-1)
    
    # =========================================================================
    # Point Operations
    # =========================================================================
    
    def random_point(self, *shape: int) -> np.ndarray:
        """
        Random point: concatenate random points from each component.
        """
        components = [m.random_point(*shape) for m in self._manifolds]
        return self.combine(components)
    
    def origin(self, *shape: int) -> np.ndarray:
        """
        Origin: concatenate origins from each component.
        """
        components = [m.origin(*shape) for m in self._manifolds]
        return self.combine(components)
    
    def check_point(self, x: np.ndarray, atol: float = 1e-5) -> bool:
        """
        Check if each component is on its manifold.
        """
        components = self.split(x)
        return all(
            m.check_point(c, atol) 
            for m, c in zip(self._manifolds, components)
        )
    
    def project_point(self, x: np.ndarray) -> np.ndarray:
        """
        Project each component onto its manifold.
        """
        components = self.split(x)
        projected = [
            m.project_point(c) 
            for m, c in zip(self._manifolds, components)
        ]
        return self.combine(projected)
    
    # =========================================================================
    # Tangent Space Operations
    # =========================================================================
    
    def check_tangent(self, x: np.ndarray, v: np.ndarray, atol: float = 1e-5) -> bool:
        """
        Check if each component is tangent.
        """
        x_components = self.split(x)
        v_components = self.split(v)
        return all(
            m.check_tangent(xc, vc, atol)
            for m, xc, vc in zip(self._manifolds, x_components, v_components)
        )
    
    def project_tangent(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Project each component to tangent space.
        """
        x_components = self.split(x)
        v_components = self.split(v)
        projected = [
            m.project_tangent(xc, vc)
            for m, xc, vc in zip(self._manifolds, x_components, v_components)
        ]
        return self.combine(projected)
    
    def random_tangent(self, x: np.ndarray) -> np.ndarray:
        """
        Random tangent: concatenate random tangents from each component.
        """
        x_components = self.split(x)
        tangents = [
            m.random_tangent(xc)
            for m, xc in zip(self._manifolds, x_components)
        ]
        return self.combine(tangents)
    
    # =========================================================================
    # Metric Operations (Block Diagonal)
    # =========================================================================
    
    def metric(self, x: np.ndarray) -> np.ndarray:
        """
        Block diagonal metric tensor.
        
        G = diag(G₁, G₂, ..., Gₖ)
        """
        x_components = self.split(x)
        
        # Get component metrics
        component_metrics = [
            m.metric(xc)
            for m, xc in zip(self._manifolds, x_components)
        ]
        
        # Build block diagonal (simplified - assumes single point)
        total_dim = self.dim
        G = np.zeros((total_dim, total_dim))
        
        offset = 0
        for Gc, d in zip(component_metrics, self._dims):
            G[offset:offset+d, offset:offset+d] = Gc
            offset += d
        
        return G
    
    def inner(self, x: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Sum of component inner products.
        
        ⟨u, v⟩_x = Σᵢ ⟨uᵢ, vᵢ⟩_{xᵢ}
        """
        x_components = self.split(x)
        u_components = self.split(u)
        v_components = self.split(v)
        
        total = 0.0
        for m, xc, uc, vc in zip(self._manifolds, x_components, u_components, v_components):
            total = total + m.inner(xc, uc, vc)
        
        return total
    
    # =========================================================================
    # Exponential and Logarithm Maps (Component-wise)
    # =========================================================================
    
    def exp(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Component-wise exponential map.
        
        exp_x(v) = (exp_{x₁}(v₁), ..., exp_{xₖ}(vₖ))
        """
        x_components = self.split(x)
        v_components = self.split(v)
        
        results = [
            m.exp(xc, vc)
            for m, xc, vc in zip(self._manifolds, x_components, v_components)
        ]
        return self.combine(results)
    
    def log(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Component-wise logarithm map.
        
        log_x(y) = (log_{x₁}(y₁), ..., log_{xₖ}(yₖ))
        """
        x_components = self.split(x)
        y_components = self.split(y)
        
        results = [
            m.log(xc, yc)
            for m, xc, yc in zip(self._manifolds, x_components, y_components)
        ]
        return self.combine(results)
    
    def distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Product distance.
        
        d(x, y) = sqrt(d₁² + d₂² + ... + dₖ²)
        """
        x_components = self.split(x)
        y_components = self.split(y)
        
        squared_sum = 0.0
        for m, xc, yc in zip(self._manifolds, x_components, y_components):
            d = m.distance(xc, yc)
            squared_sum = squared_sum + d ** 2
        
        return np.sqrt(squared_sum)
    
    def geodesic(self, x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        """
        Component-wise geodesic.
        """
        x_components = self.split(x)
        y_components = self.split(y)
        
        results = [
            m.geodesic(xc, yc, t)
            for m, xc, yc in zip(self._manifolds, x_components, y_components)
        ]
        return self.combine(results)
    
    # =========================================================================
    # Parallel Transport (Component-wise)
    # =========================================================================
    
    def parallel_transport(
        self,
        x: np.ndarray,
        y: np.ndarray,
        v: np.ndarray
    ) -> np.ndarray:
        """
        Component-wise parallel transport.
        """
        x_components = self.split(x)
        y_components = self.split(y)
        v_components = self.split(v)
        
        results = [
            m.parallel_transport(xc, yc, vc)
            for m, xc, yc, vc in zip(self._manifolds, x_components, y_components, v_components)
        ]
        return self.combine(results)
    
    # =========================================================================
    # Component-wise distance (for analysis)
    # =========================================================================
    
    def component_distances(self, x: np.ndarray, y: np.ndarray) -> List[np.ndarray]:
        """
        Get distance in each component separately.
        
        Useful for analyzing which components contribute to total distance.
        """
        x_components = self.split(x)
        y_components = self.split(y)
        
        return [
            m.distance(xc, yc)
            for m, xc, yc in zip(self._manifolds, x_components, y_components)
        ]


# =============================================================================
# Convenience Constructors
# =============================================================================

def HyperbolicSphere(hyp_dim: int, sphere_dim: int, curvature: float = -1.0) -> ProductManifold:
    """
    Hyperbolic × Sphere product.
    
    Useful for: hierarchy + direction
    """
    from .hyperbolic import Hyperbolic
    from .sphere import Sphere
    
    return ProductManifold([
        Hyperbolic(hyp_dim, curvature=curvature),
        Sphere(sphere_dim)
    ])


def HyperbolicEuclidean(hyp_dim: int, euc_dim: int, curvature: float = -1.0) -> ProductManifold:
    """
    Hyperbolic × Euclidean product.
    
    Useful for: hierarchy + continuous features
    """
    from .hyperbolic import Hyperbolic
    from .base import Euclidean
    
    return ProductManifold([
        Hyperbolic(hyp_dim, curvature=curvature),
        Euclidean(euc_dim)
    ])


def MultiHyperbolic(dim: int, n_copies: int, curvatures: Optional[List[float]] = None) -> ProductManifold:
    """
    Product of multiple hyperbolic spaces.
    
    Useful for: multi-scale hierarchies
    """
    from .hyperbolic import Hyperbolic
    
    if curvatures is None:
        curvatures = [-1.0] * n_copies
    
    return ProductManifold([
        Hyperbolic(dim, curvature=c) for c in curvatures
    ])


def MultiSphere(dim: int, n_copies: int) -> ProductManifold:
    """
    Product of multiple spheres (torus-like).
    """
    from .sphere import Sphere
    
    return ProductManifold([Sphere(dim) for _ in range(n_copies)])


# =============================================================================
# Test Function
# =============================================================================

def test_product():
    """Test ProductManifold."""
    print("=" * 60)
    print("Testing Product Manifold")
    print("=" * 60)
    
    # Import component manifolds
    from .base import Euclidean
    from .hyperbolic import Hyperbolic
    from .sphere import Sphere
    
    # Create product: Hyperbolic(3) × Sphere(2) × Euclidean(4)
    H = Hyperbolic(3)
    S = Sphere(2)
    E = Euclidean(4)
    
    M = ProductManifold([H, S, E])
    
    print(f"Manifold: {M.name}")
    print(f"Components: {M.n_components}")
    print(f"Total dim: {M.dim} (intrinsic)")
    print(f"Total ambient dim: {M.ambient_dim}")
    
    # Test 1: Random point
    print("\n1. Random point")
    x = M.random_point()
    print(f"   Shape: {x.shape}")
    print(f"   Is on manifold: {M.check_point(x)}")
    
    # Check components
    components = M.split(x)
    print(f"   Component shapes: {[c.shape for c in components]}")
    for i, (m, c) in enumerate(zip([H, S, E], components)):
        print(f"   Component {i} ({m.name}): on manifold = {m.check_point(c)}")
    
    assert M.check_point(x)
    print("   ✅ PASS")
    
    # Test 2: Origin
    print("\n2. Origin")
    o = M.origin()
    assert M.check_point(o)
    o_components = M.split(o)
    for i, (m, c) in enumerate(zip([H, S, E], o_components)):
        expected = m.origin()
        assert np.allclose(c, expected)
    print("   All component origins correct")
    print("   ✅ PASS")
    
    # Test 3: Split and combine
    print("\n3. Split and combine")
    x = M.random_point()
    components = M.split(x)
    x_reconstructed = M.combine(components)
    assert np.allclose(x, x_reconstructed)
    print("   combine(split(x)) = x")
    print("   ✅ PASS")
    
    # Test 4: Exp/Log inverse
    print("\n4. Exp/Log inverse")
    x = M.random_point()
    y = M.random_point()
    v = M.log(x, y)
    y_recovered = M.exp(x, v)
    error = np.linalg.norm(y - y_recovered)
    print(f"   exp(x, log(x, y)) ≈ y: error = {error:.2e}")
    assert error < 1e-8
    print("   ✅ PASS")
    
    # Test 5: Distance
    print("\n5. Distance (Pythagorean)")
    x = M.random_point()
    y = M.random_point()
    
    d_total = M.distance(x, y)
    component_dists = M.component_distances(x, y)
    d_pythagorean = np.sqrt(sum(d ** 2 for d in component_dists))
    
    print(f"   Total distance: {d_total:.4f}")
    print(f"   Component distances: {[f'{d:.4f}' for d in component_dists]}")
    print(f"   sqrt(sum of squares): {d_pythagorean:.4f}")
    
    assert np.allclose(d_total, d_pythagorean)
    print("   ✅ PASS")
    
    # Test 6: Distance symmetry
    print("\n6. Distance symmetry")
    d_xy = M.distance(x, y)
    d_yx = M.distance(y, x)
    print(f"   d(x, y) = {d_xy:.6f}")
    print(f"   d(y, x) = {d_yx:.6f}")
    assert np.allclose(d_xy, d_yx)
    print("   ✅ PASS")
    
    # Test 7: Geodesic
    print("\n7. Geodesic")
    x = M.random_point()
    y = M.random_point()
    mid = M.geodesic(x, y, 0.5)
    
    d_total = M.distance(x, y)
    d_to_mid = M.distance(x, mid)
    ratio = d_to_mid / d_total
    
    print(f"   Total distance: {d_total:.4f}")
    print(f"   Distance to midpoint: {d_to_mid:.4f}")
    print(f"   Ratio: {ratio:.4f} (expected ~0.5)")
    
    assert np.allclose(ratio, 0.5, atol=1e-2)
    print("   ✅ PASS")
    
    # Test 8: Parallel transport preserves norm
    print("\n8. Parallel transport")
    x = M.random_point()
    y = M.random_point()
    v = M.random_tangent(x)
    
    v_transported = M.parallel_transport(x, y, v)
    
    norm_x = M.norm(x, v)
    norm_y = M.norm(y, v_transported)
    
    print(f"   ||v||_x = {norm_x:.4f}")
    print(f"   ||v_transported||_y = {norm_y:.4f}")
    
    assert np.allclose(norm_x, norm_y, rtol=1e-2)
    print("   ✅ PASS")
    
    # Test 9: Inner product is sum of component inner products
    print("\n9. Inner product decomposition")
    x = M.random_point()
    u = M.random_tangent(x)
    v = M.random_tangent(x)
    
    ip_total = M.inner(x, u, v)
    
    x_comp = M.split(x)
    u_comp = M.split(u)
    v_comp = M.split(v)
    
    ip_sum = sum(
        m.inner(xc, uc, vc)
        for m, xc, uc, vc in zip([H, S, E], x_comp, u_comp, v_comp)
    )
    
    print(f"   Total inner product: {ip_total:.4f}")
    print(f"   Sum of component inner products: {ip_sum:.4f}")
    
    assert np.allclose(ip_total, ip_sum)
    print("   ✅ PASS")
    
    # Test 10: Batched operations
    print("\n10. Batched operations")
    x_batch = np.stack([M.random_point() for _ in range(32)])
    y_batch = np.stack([M.random_point() for _ in range(32)])
    
    distances = np.array([M.distance(x_batch[i], y_batch[i]) for i in range(32)])
    print(f"   Batch shape: {distances.shape}")
    print(f"   Mean distance: {distances.mean():.4f}")
    print("   ✅ PASS")
    
    print("\n" + "=" * 60)
    print("All Product Manifold tests passed! ✅")
    print("=" * 60)


if __name__ == "__main__":
    test_product()
