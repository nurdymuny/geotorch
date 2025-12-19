#!/usr/bin/env python
"""
DavisTensor: Product Manifolds for Multi-Aspect Embeddings
===========================================================

Many real-world entities have multiple aspects:
- A person: social hierarchy + location + interests
- A product: category tree + price range + features
- A document: topic hierarchy + sentiment + style

Product manifolds let us combine geometries:
- H^n × S^m: Hyperbolic for hierarchy + Sphere for directions
- H^n × R^m: Hyperbolic for hierarchy + Euclidean for features
- H^n × H^m × ...: Multiple independent hierarchies

This demo shows how to work with product manifolds in DavisTensor.
"""

import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parent_dir)

import numpy as np
import davistensor as dt
from davistensor import (
    Euclidean, Hyperbolic, Sphere, 
    ProductManifold, HyperbolicSphere, HyperbolicEuclidean, 
    MultiHyperbolic, MultiSphere
)


def demo_product_basics():
    """Basic product manifold operations."""
    print("\n" + "=" * 60)
    print("PRODUCT MANIFOLD BASICS")
    print("=" * 60)
    
    # Create H² × S² (hyperbolic plane × 2-sphere)
    H = Hyperbolic(2)
    S = Sphere(2)
    product = ProductManifold([H, S])
    
    print(f"\nManifold: {product}")
    print(f"Components: {[m.name for m in product.manifolds]}")
    print(f"Total dimension: {product.dim} (intrinsic)")
    print(f"Ambient dimension: {product.ambient_dim}")
    
    # Random point
    x = product.random_point()
    print(f"\nRandom point shape: {x.shape}")
    
    # Split into components
    x_H, x_S = product.split(x)
    print(f"\nHyperbolic component: {x_H}, ||x_H|| = {np.linalg.norm(x_H):.4f} < 1")
    print(f"Sphere component: {x_S}, ||x_S|| = {np.linalg.norm(x_S):.4f} ≈ 1")
    
    # Verify constraints
    print(f"\nOn hyperbolic? {H.check_point(x_H)}")
    print(f"On sphere? {S.check_point(x_S)}")


def demo_hyperbolic_sphere():
    """HyperbolicSphere for entities with hierarchy + direction."""
    print("\n" + "=" * 60)
    print("HYPERBOLIC × SPHERE: Hierarchy + Direction")
    print("=" * 60)
    
    # H^4 × S^3: 4D hyperbolic for hierarchy, 3-sphere for direction
    HS = HyperbolicSphere(4, 3)
    
    print(f"\nManifold: {HS}")
    print(f"Use case: Entities with hierarchical position AND directional aspect")
    print(f"Example: Documents in topic hierarchy + sentiment direction")
    
    # Sample points
    x = HS.random_point()
    y = HS.random_point()
    
    # Distance combines both components
    d = HS.distance(x, y)
    
    # Get component distances
    x_H, x_S = HS.split(x)
    y_H, y_S = HS.split(y)
    
    H = Hyperbolic(4)
    S = Sphere(3)
    d_H = H.distance(x_H, y_H)
    d_S = S.distance(x_S, y_S)
    
    print(f"\nDistance between random points:")
    print(f"  Total: {d:.4f}")
    print(f"  Hyperbolic component: {d_H:.4f}")
    print(f"  Sphere component: {d_S:.4f}")
    print(f"  Pythagorean: sqrt({d_H:.4f}² + {d_S:.4f}²) = {np.sqrt(d_H**2 + d_S**2):.4f}")


def demo_multi_hyperbolic():
    """Multiple independent hierarchies."""
    print("\n" + "=" * 60)
    print("MULTI-HYPERBOLIC: Independent Hierarchies")
    print("=" * 60)
    
    # H^8 × H^8 × H^8: Three 8D hyperbolic spaces
    MH = MultiHyperbolic(8, 3)
    
    print(f"\nManifold: {MH}")
    print(f"Use case: Knowledge graphs with multiple relation types")
    print(f"Example: Organization hierarchy × Project hierarchy × Skill hierarchy")
    
    # Sample point
    x = MH.random_point()
    components = MH.split(x)
    
    print(f"\nPoint shape: {x.shape}")
    print(f"Number of components: {len(components)}")
    
    for i, comp in enumerate(components):
        norm = np.linalg.norm(comp)
        print(f"  Component {i+1}: ||x|| = {norm:.4f} < 1 ✓")


def demo_knowledge_graph():
    """Simulated knowledge graph embedding."""
    print("\n" + "=" * 60)
    print("KNOWLEDGE GRAPH EMBEDDING")
    print("=" * 60)
    
    # Use HyperbolicEuclidean: hierarchy + features
    # H^8 for organizational hierarchy
    # R^4 for numeric features
    HE = HyperbolicEuclidean(8, 4)
    
    print(f"\nManifold: {HE}")
    print(f"Hyperbolic: organizational hierarchy")
    print(f"Euclidean: numeric attributes (salary, tenure, etc.)")
    
    # Simulate employee embeddings
    np.random.seed(42)
    n_employees = 20
    
    # Create embeddings
    embeddings = HE.random_point(n_employees)
    
    # Assign hierarchical depth (simulate org chart)
    depths = np.random.choice([0, 1, 2, 3], size=n_employees, p=[0.1, 0.2, 0.3, 0.4])
    
    # Modify hyperbolic component based on depth
    # (Deeper in org = further from center in Poincaré ball)
    H = Hyperbolic(8)
    E = Euclidean(4)
    
    for i in range(n_employees):
        h_comp, e_comp = HE.split(embeddings[i])
        
        # Scale radius based on depth
        target_radius = 0.1 + 0.2 * depths[i]  # 0.1, 0.3, 0.5, 0.7
        current_radius = np.linalg.norm(h_comp)
        if current_radius > 1e-6:
            h_comp = h_comp * (target_radius / current_radius)
        
        embeddings[i] = HE.combine([h_comp, e_comp])
    
    print(f"\nEmployees by level:")
    for level in range(4):
        count = np.sum(depths == level)
        mask = depths == level
        avg_norm = np.mean([np.linalg.norm(HE.split(embeddings[i])[0]) 
                           for i in range(n_employees) if mask[i]])
        print(f"  Level {level}: {count} employees, avg hyperbolic radius = {avg_norm:.3f}")
    
    # Find similar employees (close in product space)
    print(f"\nSimilarity analysis (by distance):")
    
    query_idx = 0
    query_level = depths[query_idx]
    distances = np.array([HE.distance(embeddings[query_idx], embeddings[i]) 
                         for i in range(n_employees)])
    
    # Sort by distance
    sorted_idx = np.argsort(distances)
    print(f"\n  Query: Employee {query_idx} (Level {query_level})")
    print(f"  Nearest neighbors:")
    for rank, idx in enumerate(sorted_idx[1:6]):  # Top 5 (excluding self)
        print(f"    {rank+1}. Employee {idx} (Level {depths[idx]}), distance = {distances[idx]:.4f}")


def demo_geodesics_product():
    """Geodesics on product manifolds."""
    print("\n" + "=" * 60)
    print("GEODESICS ON PRODUCT MANIFOLDS")
    print("=" * 60)
    
    HS = HyperbolicSphere(2, 2)
    
    # Two points
    x = HS.random_point()
    y = HS.random_point()
    
    print(f"\nStart: {x}")
    print(f"End:   {y}")
    print(f"\nGeodesic interpolation:")
    
    H = Hyperbolic(2)
    S = Sphere(2)
    
    for t in [0, 0.25, 0.5, 0.75, 1.0]:
        gamma = HS.geodesic(x, y, t)
        h_comp, s_comp = HS.split(gamma)
        
        # Verify constraints maintained
        h_norm = np.linalg.norm(h_comp)
        s_norm = np.linalg.norm(s_comp)
        
        print(f"\n  t={t:.2f}:")
        print(f"    Hyperbolic ||x|| = {h_norm:.4f} < 1 ✓" if h_norm < 1 else f"    ERROR!")
        print(f"    Sphere ||x|| = {s_norm:.6f} ≈ 1 ✓" if abs(s_norm - 1) < 1e-6 else f"    ERROR!")


def demo_parallel_transport_product():
    """Parallel transport on product manifolds."""
    print("\n" + "=" * 60)
    print("PARALLEL TRANSPORT ON PRODUCT MANIFOLDS")
    print("=" * 60)
    
    HS = HyperbolicSphere(3, 2)
    
    x = HS.random_point()
    y = HS.random_point()
    
    # Random tangent vector at x
    v = np.random.randn(HS.ambient_dim) * 0.3
    v = HS.project_tangent(x, v)
    
    print(f"\nBase point x: {x[:3]}... (truncated)")
    print(f"Destination y: {y[:3]}... (truncated)")
    print(f"Tangent v at x: {v[:3]}... (truncated)")
    
    # Transport
    v_transported = HS.parallel_transport(x, y, v)
    
    print(f"\nTransported to y: {v_transported[:3]}... (truncated)")
    
    # Norm should be preserved
    norm_before = HS.norm(x, v)
    norm_after = HS.norm(y, v_transported)
    
    print(f"\n||v||_x = {norm_before:.4f}")
    print(f"||v_transported||_y = {norm_after:.4f}")
    print(f"Norm preserved? {np.isclose(norm_before, norm_after)}")


def main():
    print("=" * 60)
    print("DAVISTENSOR: Product Manifolds for Multi-Aspect Data")
    print("=" * 60)
    
    demo_product_basics()
    demo_hyperbolic_sphere()
    demo_multi_hyperbolic()
    demo_knowledge_graph()
    demo_geodesics_product()
    demo_parallel_transport_product()
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
