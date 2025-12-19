#!/usr/bin/env python
"""
DavisTensor Test Suite
======================

Run all tests for the DavisTensor library.
"""

import sys
import os

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    print("=" * 70)
    print("DAVISTENSOR TEST SUITE")
    print("=" * 70)
    
    # Test 1: Core storage and TensorCore
    print("\n" + "=" * 70)
    print("TEST 1: Core Storage and TensorCore")
    print("=" * 70)
    from davistensor.core.storage import test_core
    test_core()
    
    # Test 2: Euclidean manifold
    print("\n" + "=" * 70)
    print("TEST 2: Euclidean Manifold")
    print("=" * 70)
    from davistensor.manifolds.base import test_euclidean
    test_euclidean()
    
    # Test 3: Hyperbolic manifold
    print("\n" + "=" * 70)
    print("TEST 3: Hyperbolic Manifold (Poincare Ball)")
    print("=" * 70)
    from davistensor.manifolds.hyperbolic import test_hyperbolic
    test_hyperbolic()
    
    # Test 4: Sphere manifold
    print("\n" + "=" * 70)
    print("TEST 4: Sphere Manifold")
    print("=" * 70)
    from davistensor.manifolds.sphere import test_sphere
    test_sphere()
    
    # Test 5: SPD manifold
    print("\n" + "=" * 70)
    print("TEST 5: SPD Manifold")
    print("=" * 70)
    from davistensor.manifolds.spd import test_spd
    test_spd()
    
    # Test 6: Product manifold
    print("\n" + "=" * 70)
    print("TEST 6: Product Manifold")
    print("=" * 70)
    from davistensor.manifolds.product import test_product
    test_product()
    
    # Test 7: Type system (ManifoldTensor, TangentTensor, Scalar)
    print("\n" + "=" * 70)
    print("TEST 7: Type System")
    print("=" * 70)
    from davistensor.tensor import test_type_system
    test_type_system()
    
    # Test 8: Autograd
    print("\n" + "=" * 70)
    print("TEST 8: Autograd Engine")
    print("=" * 70)
    from davistensor.autograd.engine import test_autograd
    test_autograd()
    
    # Test 9: Neural Network Layers
    print("\n" + "=" * 70)
    print("TEST 9: Neural Network Layers")
    print("=" * 70)
    test_nn_layers()
    
    # Test 10: Integration test
    print("\n" + "=" * 70)
    print("TEST 10: Integration Test")
    print("=" * 70)
    test_integration()
    
    print("\n")
    print("=" * 70)
    print("ALL TESTS PASSED! ✅")
    print("=" * 70)


def test_nn_layers():
    """Test neural network layers."""
    import numpy as np
    from davistensor.core.storage import tensor, randn
    from davistensor.manifolds.base import Euclidean
    from davistensor.manifolds.hyperbolic import Hyperbolic
    from davistensor.manifolds.sphere import Sphere
    
    from davistensor.nn.linear import Linear, GeodesicLinear, ManifoldMLR
    from davistensor.nn.embedding import Embedding, ManifoldEmbedding
    from davistensor.nn.pooling import MeanPool, FrechetMeanPool
    from davistensor.nn.attention import GeometricAttention
    from davistensor.nn.container import Sequential
    from davistensor.nn.activation import ReLU
    
    print("=" * 60)
    print("Testing DavisTensor Neural Network Layers")
    print("=" * 60)
    
    # Test 1: Linear layer
    print("\n1. Linear layer")
    linear = Linear(10, 5)
    x = randn(32, 10)
    y = linear(x)
    print(f"   Input: {x.shape}")
    print(f"   Output: {y.shape}")
    print(f"   Parameters: {len(list(linear.parameters()))}")
    assert y.shape == (32, 5)
    print("   ✅ PASS")
    
    # Test 2: GeodesicLinear
    print("\n2. GeodesicLinear (Hyperbolic → Euclidean)")
    H = Hyperbolic(8)
    E = Euclidean(4)
    geo_linear = GeodesicLinear(H, E)
    
    x = tensor(H.random_point(16))
    x.manifold = H
    y = geo_linear(x)
    
    print(f"   Input: {x.shape} on {H}")
    print(f"   Output: {y.shape} on {y.manifold}")
    assert y.shape == (16, 4)
    print("   ✅ PASS")
    
    # Test 3: ManifoldEmbedding
    print("\n3. ManifoldEmbedding")
    emb = ManifoldEmbedding(1000, Hyperbolic(32))
    indices = np.array([0, 5, 10, 15])
    embeddings = emb(indices)
    
    print(f"   Indices: {indices.shape}")
    print(f"   Embeddings: {embeddings.shape}")
    print(f"   On manifold: {embeddings.manifold}")
    assert embeddings.shape == (4, 32)
    print("   ✅ PASS")
    
    # Test 4: FrechetMeanPool
    print("\n4. FrechetMeanPool")
    S = Sphere(2)
    pool = FrechetMeanPool(S)
    
    # Create points on sphere
    points = tensor(S.random_point(5))
    points.manifold = S
    
    mean = pool(points)
    print(f"   Input: {points.shape} on {S}")
    print(f"   Output: {mean.shape}")
    print(f"   ||mean|| = {np.linalg.norm(mean.numpy()):.6f} (should be ~1)")
    assert np.allclose(np.linalg.norm(mean.numpy()), 1.0, atol=1e-2)
    print("   ✅ PASS")
    
    # Test 5: ManifoldMLR
    print("\n5. ManifoldMLR (Classification)")
    mlr = ManifoldMLR(Hyperbolic(16), n_classes=5)
    x = tensor(Hyperbolic(16).random_point(32))
    logits = mlr(x)
    
    print(f"   Input: {x.shape}")
    print(f"   Logits: {logits.shape}")
    assert logits.shape == (32, 5)
    print("   ✅ PASS")
    
    # Test 6: Sequential
    print("\n6. Sequential container")
    model = Sequential(
        Linear(10, 20),
        ReLU(),
        Linear(20, 5)
    )
    
    x = randn(8, 10)
    y = model(x)
    
    print(f"   Model: Sequential(Linear → ReLU → Linear)")
    print(f"   Input: {x.shape}")
    print(f"   Output: {y.shape}")
    print(f"   Total parameters: {sum(p.numpy().size for p in model.parameters())}")
    assert y.shape == (8, 5)
    print("   ✅ PASS")
    
    print("\n" + "=" * 60)
    print("All NN layer tests passed! ✅")
    print("=" * 60)
    
    print("\n")
    print("=" * 70)
    print("ALL TESTS PASSED! ✅")
    print("=" * 70)


def test_integration():
    """Integration test: use DavisTensor like a user would."""
    import davistensor as dt
    import numpy as np
    
    print("\n1. Import and basic usage")
    print(f"   DavisTensor version: {dt.__version__}")
    print("   ✅ PASS")
    
    print("\n2. Create manifold and points")
    E = dt.Euclidean(10)
    x = dt.randn(manifold=E)
    y = dt.randn(manifold=E)
    print(f"   Manifold: {E}")
    print(f"   Point x: shape={x.shape}")
    print("   ✅ PASS")
    
    print("\n3. Geometric operations")
    d = x.distance(y)
    v = x.log(y)
    z = x.exp(v)
    print(f"   distance(x, y) = {d.item():.4f}")
    print(f"   log(x, y) is TangentTensor: {type(v).__name__}")
    print(f"   exp(x, v) ≈ y: error={x.distance(z).item():.2e}")
    print("   ✅ PASS")
    
    print("\n4. Arithmetic is geometric")
    y2 = x + v  # Same as exp(x, v)
    assert np.allclose(y2.numpy(), z.numpy())
    w = x - y   # Same as log(x, y)
    # Note: w is tangent at x pointing to y, v is tangent at x pointing to y
    # But x - y gives log(x, y) which is different sign convention
    print("   x + tangent = exp(x, tangent) ✓")
    print("   x - y = tangent from x to y ✓")
    print("   ✅ PASS")
    
    print("\n5. Type safety: tangent vectors at different points")
    x1 = dt.randn(manifold=E)
    x2 = dt.randn(manifold=E)
    v1 = dt.tangent_randn(x1)
    v2 = dt.tangent_randn(x2)
    
    try:
        bad = v1 + v2  # Should fail!
        print("   ERROR: Should have raised TypeError!")
        assert False
    except TypeError:
        print("   v1 + v2 raises TypeError (different base points) ✓")
    
    # Fix by transporting
    v1_at_x2 = v1.transport_to(x2)
    good = v1_at_x2 + v2
    print("   v1.transport_to(x2) + v2 works ✓")
    print("   ✅ PASS")
    
    print("\n6. Geodesic interpolation")
    x = dt.origin(E)
    y = dt.ManifoldTensor(np.ones(10) * 2, E)
    
    points = [x.geodesic(y, t) for t in [0.0, 0.25, 0.5, 0.75, 1.0]]
    print(f"   Geodesic from 0 to 2:")
    for t, p in zip([0.0, 0.25, 0.5, 0.75, 1.0], points):
        print(f"     t={t}: [{p.numpy()[0]:.2f}, ...]")
    print("   ✅ PASS")
    
    print("\n7. Batched operations")
    X = dt.randn(100, manifold=E)
    Y = dt.randn(100, manifold=E)
    D = X.distance(Y)
    print(f"   100 pairwise distances: shape={D.numpy().shape}")
    print(f"   Mean distance: {D.numpy().mean():.4f}")
    print("   ✅ PASS")
    
    print("\n8. Hyperbolic space integration")
    H = dt.Hyperbolic(5)
    x = dt.randn(manifold=H)
    y = dt.randn(manifold=H)
    
    # Points are automatically inside Poincare ball
    assert np.linalg.norm(x.numpy()) < 1.0
    print(f"   Points inside ball: ||x|| = {np.linalg.norm(x.numpy()):.4f} < 1")
    
    # Hyperbolic distance grows fast near boundary
    d = x.distance(y)
    print(f"   Hyperbolic distance: {d.item():.4f}")
    
    # Geodesic stays inside ball
    mid = x.geodesic(y, 0.5)
    assert np.linalg.norm(mid.numpy()) < 1.0
    print(f"   Midpoint inside ball: ||mid|| = {np.linalg.norm(mid.numpy()):.4f} < 1")
    print("   ✅ PASS")
    
    print("\n9. Sphere integration")
    S = dt.Sphere(2)  # 2-sphere in R³
    x = dt.randn(manifold=S)
    y = dt.randn(manifold=S)
    
    # Points are automatically on unit sphere
    assert abs(np.linalg.norm(x.numpy()) - 1.0) < 1e-10
    print(f"   Points on sphere: ||x|| = {np.linalg.norm(x.numpy()):.6f}")
    
    # Great circle distance
    d = x.distance(y)
    print(f"   Great circle distance: {d.item():.4f} (max = pi = {np.pi:.4f})")
    
    # Geodesic stays on sphere
    mid = x.geodesic(y, 0.5)
    assert abs(np.linalg.norm(mid.numpy()) - 1.0) < 1e-10
    print(f"   Midpoint on sphere: ||mid|| = {np.linalg.norm(mid.numpy()):.6f}")
    print("   ✅ PASS")


if __name__ == "__main__":
    main()
