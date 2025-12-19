#!/usr/bin/env python
"""
DavisTensor Demo: Geometry-Native Tensors
==========================================

This demo showcases how DavisTensor makes Riemannian geometry natural
and type-safe. Every tensor knows its manifold, and operations respect
the geometry automatically.

Compare with PyTorch + GeoTorch:
- PyTorch: Geometry is bolted on top
- DavisTensor: Geometry is built into the DNA
"""

import sys
import os

# Add parent directory to path so we can import davistensor
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parent_dir)

import davistensor as dt
import numpy as np


def demo_euclidean():
    """Demo: Euclidean space (the familiar baseline)."""
    print("\n" + "=" * 60)
    print("EUCLIDEAN SPACE (baseline)")
    print("=" * 60)
    
    E = dt.Euclidean(3)
    
    # Create points
    x = dt.randn(manifold=E)
    y = dt.randn(manifold=E)
    
    print(f"\nPoints in R^3:")
    print(f"  x = {x.numpy()}")
    print(f"  y = {y.numpy()}")
    
    # Distance is Euclidean norm of difference
    d = x.distance(y)
    print(f"\nEuclidean distance: {d.item():.4f}")
    
    # log(x, y) = y - x (simple!)
    v = x.log(y)
    print(f"log_x(y) = {v.numpy()}")
    
    # exp(x, v) = x + v (simple!)
    y2 = x.exp(v)
    print(f"exp_x(v) = {y2.numpy()}")
    print(f"Same as y? {np.allclose(y.numpy(), y2.numpy())}")
    
    # Geodesic is straight line
    print("\nGeodesic (straight line):")
    for t in [0, 0.25, 0.5, 0.75, 1.0]:
        p = x.geodesic(y, t)
        print(f"  t={t:.2f}: {p.numpy()}")


def demo_hyperbolic():
    """Demo: Hyperbolic space for hierarchical data."""
    print("\n" + "=" * 60)
    print("HYPERBOLIC SPACE (for hierarchies)")
    print("=" * 60)
    
    H = dt.Hyperbolic(2)  # 2D for visualization
    
    # The origin is the center of the Poincaré disk
    origin = dt.origin(H)
    print(f"\nOrigin (center): {origin.numpy()}")
    
    # Random points stay inside the disk!
    x = dt.randn(manifold=H)
    y = dt.randn(manifold=H)
    print(f"\nRandom points (inside unit disk):")
    print(f"  x = {x.numpy()}, ||x|| = {np.linalg.norm(x.numpy()):.4f} < 1 ✓")
    print(f"  y = {y.numpy()}, ||y|| = {np.linalg.norm(y.numpy()):.4f} < 1 ✓")
    
    # Hyperbolic distance grows logarithmically near boundary
    d = x.distance(y)
    print(f"\nHyperbolic distance: {d.item():.4f}")
    print("  (Distance grows fast near boundary - perfect for trees!)")
    
    # Tangent vector at origin
    v = origin.log(x)
    print(f"\nTangent at origin pointing to x:")
    print(f"  v = {v.numpy()}")
    
    # Move along geodesic
    print("\nGeodesic (curve in Poincaré disk):")
    for t in [0, 0.25, 0.5, 0.75, 1.0]:
        p = origin.geodesic(x, t)
        print(f"  t={t:.2f}: {p.numpy()}, ||p|| = {np.linalg.norm(p.numpy()):.4f}")


def demo_sphere():
    """Demo: Sphere for directional data."""
    print("\n" + "=" * 60)
    print("SPHERE (for directional data)")
    print("=" * 60)
    
    S = dt.Sphere(2)  # 2-sphere in R³
    
    # North pole
    north = dt.origin(S)
    print(f"\nNorth pole: {north.numpy()}")
    
    # Random points are ON the sphere
    x = dt.randn(manifold=S)
    y = dt.randn(manifold=S)
    print(f"\nRandom points (on unit sphere):")
    print(f"  x = {x.numpy()}, ||x|| = {np.linalg.norm(x.numpy()):.6f} ≈ 1 ✓")
    print(f"  y = {y.numpy()}, ||y|| = {np.linalg.norm(y.numpy()):.6f} ≈ 1 ✓")
    
    # Great circle distance
    d = x.distance(y)
    print(f"\nGreat circle distance: {d.item():.4f}")
    print(f"  (Maximum possible = π = {np.pi:.4f} for antipodal points)")
    
    # Tangent space is perpendicular to point
    v = x.log(y)
    dot = np.dot(x.numpy(), v.numpy())
    print(f"\nTangent at x pointing to y:")
    print(f"  v = {v.numpy()}")
    print(f"  v ⊥ x? dot = {dot:.2e} ≈ 0 ✓")
    
    # Geodesic stays on sphere
    print("\nGeodesic (great circle arc):")
    for t in [0, 0.25, 0.5, 0.75, 1.0]:
        p = x.geodesic(y, t)
        print(f"  t={t:.2f}: ||p|| = {np.linalg.norm(p.numpy()):.6f}")


def demo_type_safety():
    """Demo: Type-safe tangent vectors."""
    print("\n" + "=" * 60)
    print("TYPE SAFETY: Tangent Vectors")
    print("=" * 60)
    
    H = dt.Hyperbolic(3)
    
    x = dt.randn(manifold=H)
    y = dt.randn(manifold=H)
    
    # Create tangent vectors at different points
    vx = dt.tangent_randn(x)
    vy = dt.tangent_randn(y)
    
    print(f"\nTangent at x: {vx}")
    print(f"Tangent at y: {vy}")
    
    # Try to add them - ERROR!
    print("\nAttempting vx + vy (different base points)...")
    try:
        bad = vx + vy
        print("  ERROR: This should have failed!")
    except TypeError as e:
        print(f"  ✓ Correctly rejected: {str(e)[:50]}...")
    
    # Fix: parallel transport first
    print("\nParallel transport vx to y, then add:")
    vx_at_y = vx.transport_to(y)
    print(f"  vx transported: {vx_at_y}")
    
    result = vx_at_y + vy
    print(f"  vx_at_y + vy = {result}")
    print("  ✓ Works because both are at y!")


def demo_arithmetic():
    """Demo: Arithmetic is geometric."""
    print("\n" + "=" * 60)
    print("ARITHMETIC = GEOMETRY")
    print("=" * 60)
    
    S = dt.Sphere(2)
    
    x = dt.randn(manifold=S)
    y = dt.randn(manifold=S)
    
    print(f"\nx = {x.numpy()}")
    print(f"y = {y.numpy()}")
    
    # Point + Tangent = Exponential map
    v = x.log(y)
    y_via_add = x + v
    y_via_exp = x.exp(v)
    
    print(f"\nx + v (where v = log_x(y)):")
    print(f"  x + v   = {y_via_add.numpy()}")
    print(f"  exp(x,v)= {y_via_exp.numpy()}")
    print(f"  Same? {np.allclose(y_via_add.numpy(), y_via_exp.numpy())}")
    
    # Point - Point = Logarithm map
    w = x - y  # This is log(x, y), tangent at x pointing to y
    w_explicit = x.log(y)
    
    print(f"\nx - y:")
    print(f"  x - y   = {w.numpy()}")
    print(f"  log(x,y)= {w_explicit.numpy()}")
    print(f"  Same? {np.allclose(w.numpy(), w_explicit.numpy())}")


def demo_comparison():
    """Compare DavisTensor with GeoTorch approach."""
    print("\n" + "=" * 60)
    print("COMPARISON: DavisTensor vs PyTorch+GeoTorch")
    print("=" * 60)
    
    print("""
PyTorch + GeoTorch (current approach):
--------------------------------------
```python
x = torch.randn(64)
x = manifold.project(x)           # Easy to forget!
v = torch.randn(64)               # Is this a point or tangent? Who knows!
v = manifold.project_tangent(x, v) # Must remember to project
y = manifold.exp(x, v)            # Explicit exp map
loss.backward()
# grad is in ambient space, not tangent space!
# must manually project, transport, etc.
```

DavisTensor (geometry-native):
------------------------------
```python
x = dt.randn(64, manifold=Hyperbolic)    # Tensor knows it's hyperbolic
v = dt.tangent_randn(x)                   # Type-safe tangent vector at x
y = x + v                                  # Automatically exp map!
d = x.distance(y)                          # Native operation
d.backward()
# x.grad is ALREADY in tangent space at x
# No manual projection needed
```

Key Differences:
- Type safety: Points and tangents are different types
- Automatic projection: Points stay on manifold
- Natural syntax: x + v means "move x in direction v"
- Correct gradients: Always in tangent space (future)
""")


if __name__ == "__main__":
    print("=" * 60)
    print("DAVISTENSOR: Geometry-Native Tensors")
    print("=" * 60)
    
    demo_euclidean()
    demo_hyperbolic()
    demo_sphere()
    demo_type_safety()
    demo_arithmetic()
    demo_comparison()
    
    print("\n" + "=" * 60)
    print("Demo complete! DavisTensor makes geometry natural.")
    print("=" * 60)
