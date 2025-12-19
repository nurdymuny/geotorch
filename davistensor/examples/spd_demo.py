#!/usr/bin/env python
"""
DavisTensor: SPD Manifold for Brain Connectivity
=================================================

Symmetric Positive Definite (SPD) matrices arise naturally in:
- Covariance matrices
- Diffusion tensors (DTI)
- Brain connectivity matrices
- Kernel matrices

This demo shows how to work with SPD matrices using DavisTensor's
affine-invariant geometry.

Key insight: The space of SPD matrices is a curved manifold!
Using the affine-invariant metric gives:
- Scale-invariant distances
- Geodesics via matrix exponential/logarithm
- Fréchet mean for averaging covariances
"""

import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parent_dir)

import numpy as np
import davistensor as dt
from davistensor import SPD

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def demo_spd_basics():
    """Basic SPD manifold operations."""
    print("\n" + "=" * 60)
    print("SPD MANIFOLD BASICS")
    print("=" * 60)
    
    n = 3
    spd = SPD(n)
    
    # Create random SPD matrices (covariance matrices)
    P = spd.random_point()
    Q = spd.random_point()
    
    print(f"\nManifold: {spd}")
    print(f"Matrix size: {n}×{n}")
    print(f"Intrinsic dimension: {spd.dim} (n(n+1)/2 = symmetric)")
    
    print(f"\nRandom SPD matrix P:")
    print(P)
    print(f"Eigenvalues (all positive): {np.linalg.eigvalsh(P)}")
    
    # The identity matrix is the "origin"
    I = spd.origin()
    print(f"\nOrigin (identity matrix):")
    print(I)
    
    # Affine-invariant distance
    d = spd.distance(P, Q)
    print(f"\nAffine-invariant distance d(P, Q) = {d:.4f}")
    print("(This distance is invariant under congruence: d(APA', AQA') = d(P,Q))")
    
    # Distance from identity has closed form
    d_from_I = spd.distance(I, P)
    # d(I, P) = ||log(P)||_F = sqrt(sum(log(λ_i)²))
    eigenvalues = np.linalg.eigvalsh(P)
    expected = np.sqrt(np.sum(np.log(eigenvalues) ** 2))
    print(f"\nDistance from identity: {d_from_I:.4f}")
    print(f"Expected (||log(P)||_F): {expected:.4f}")


def demo_geodesics():
    """Geodesics on SPD manifold."""
    print("\n" + "=" * 60)
    print("GEODESICS ON SPD MANIFOLD")
    print("=" * 60)
    
    n = 2
    spd = SPD(n)
    
    # Two SPD matrices
    P = np.array([[2.0, 0.5], [0.5, 1.0]])  # Slightly correlated
    Q = np.array([[1.0, -0.3], [-0.3, 3.0]])  # Different correlation
    
    print(f"\nStart P:")
    print(P)
    print(f"\nEnd Q:")
    print(Q)
    
    print("\nGeodesic path (matrix morphing):")
    for t in [0, 0.25, 0.5, 0.75, 1.0]:
        gamma_t = spd.geodesic(P, Q, t)
        eigvals = np.linalg.eigvalsh(gamma_t)
        print(f"\nt={t:.2f}:")
        print(gamma_t)
        print(f"  Eigenvalues: {eigvals} (always positive!)")


def demo_frechet_mean():
    """Fréchet mean of SPD matrices."""
    print("\n" + "=" * 60)
    print("FRÉCHET MEAN (Geometric Average)")
    print("=" * 60)
    
    n = 3
    spd = SPD(n)
    
    # Generate covariance matrices with different structures
    np.random.seed(42)
    n_samples = 5
    
    covariances = []
    print("\nInput covariance matrices (diagonal elements):")
    for i in range(n_samples):
        C = spd.random_point()
        covariances.append(C)
        print(f"  C{i+1}: diag = {np.diag(C)}")
    
    # Stack into batch
    C_batch = np.stack(covariances)
    
    # Compute Fréchet mean
    C_mean = spd.frechet_mean(C_batch)
    
    print(f"\nFréchet mean (geometric average):")
    print(C_mean)
    print(f"Mean diagonal: {np.diag(C_mean)}")
    
    # Compare with arithmetic mean
    arith_mean = np.mean(C_batch, axis=0)
    print(f"\nArithmetic mean (often not SPD for general matrices):")
    print(arith_mean)
    print(f"Arithmetic mean diagonal: {np.diag(arith_mean)}")
    
    # Verify Fréchet mean minimizes sum of squared distances
    frechet_dist_sq = sum(spd.distance(C_mean, C)**2 for C in covariances)
    print(f"\nSum of squared distances to Fréchet mean: {frechet_dist_sq:.4f}")


def demo_brain_connectivity():
    """Simulate brain connectivity analysis."""
    print("\n" + "=" * 60)
    print("BRAIN CONNECTIVITY ANALYSIS")
    print("=" * 60)
    
    n_regions = 4  # Brain regions
    spd = SPD(n_regions)
    
    # Simulate connectivity matrices for different conditions
    np.random.seed(123)
    
    # Healthy subjects: moderate connectivity
    print("\nSimulating connectivity matrices...")
    healthy = []
    for i in range(5):
        base = np.eye(n_regions) * 2
        base += 0.3 * np.random.randn(n_regions, n_regions)
        C = base @ base.T  # Make SPD
        healthy.append(C)
    
    # Patient subjects: altered connectivity pattern
    patients = []
    for i in range(5):
        base = np.eye(n_regions) * 2
        base[0, 1] = base[1, 0] = 1.5  # Increased connectivity in specific regions
        base += 0.3 * np.random.randn(n_regions, n_regions)
        C = base @ base.T
        patients.append(C)
    
    # Compute group centroids using Fréchet mean
    healthy_batch = np.stack(healthy)
    patient_batch = np.stack(patients)
    
    healthy_centroid = spd.frechet_mean(healthy_batch)
    patient_centroid = spd.frechet_mean(patient_batch)
    
    print("\nHealthy group centroid (diagonal):")
    print(f"  {np.diag(healthy_centroid)}")
    
    print("\nPatient group centroid (diagonal):")
    print(f"  {np.diag(patient_centroid)}")
    
    # Distance between group centroids
    group_distance = spd.distance(healthy_centroid, patient_centroid)
    print(f"\nDistance between group centroids: {group_distance:.4f}")
    
    # Within-group variance (average distance to centroid)
    healthy_var = np.mean([spd.distance(healthy_centroid, C) for C in healthy])
    patient_var = np.mean([spd.distance(patient_centroid, C) for C in patients])
    
    print(f"\nWithin-group average distances:")
    print(f"  Healthy: {healthy_var:.4f}")
    print(f"  Patients: {patient_var:.4f}")
    
    # Simple classification: assign to nearest centroid
    print("\nClassification (nearest centroid):")
    correct = 0
    for i, C in enumerate(healthy):
        d_h = spd.distance(healthy_centroid, C)
        d_p = spd.distance(patient_centroid, C)
        pred = "healthy" if d_h < d_p else "patient"
        actual = "healthy"
        correct += pred == actual
        print(f"  Healthy {i+1}: d_h={d_h:.3f}, d_p={d_p:.3f} -> {pred} {'✓' if pred==actual else '✗'}")
    
    for i, C in enumerate(patients):
        d_h = spd.distance(healthy_centroid, C)
        d_p = spd.distance(patient_centroid, C)
        pred = "healthy" if d_h < d_p else "patient"
        actual = "patient"
        correct += pred == actual
        print(f"  Patient {i+1}: d_h={d_h:.3f}, d_p={d_p:.3f} -> {pred} {'✓' if pred==actual else '✗'}")
    
    print(f"\nAccuracy: {correct}/10 ({100*correct/10:.0f}%)")


def demo_parallel_transport():
    """Parallel transport on SPD manifold."""
    print("\n" + "=" * 60)
    print("PARALLEL TRANSPORT ON SPD")
    print("=" * 60)
    
    n = 2
    spd = SPD(n)
    
    P = np.array([[2.0, 0.5], [0.5, 1.0]])
    Q = np.array([[1.0, 0.2], [0.2, 2.0]])
    
    # A tangent vector at P (symmetric matrix)
    V = np.array([[0.3, 0.1], [0.1, -0.2]])
    
    print(f"\nBase point P:")
    print(P)
    print(f"\nTangent vector V at P:")
    print(V)
    print(f"\nDestination Q:")
    print(Q)
    
    # Transport V from P to Q
    V_transported = spd.parallel_transport(P, Q, V)
    
    print(f"\nV transported to Q:")
    print(V_transported)
    
    # Verify it's still symmetric
    is_symmetric = np.allclose(V_transported, V_transported.T)
    print(f"\nStill symmetric? {is_symmetric}")


def main():
    print("=" * 60)
    print("DAVISTENSOR: SPD Manifold for Covariance Data")
    print("=" * 60)
    
    demo_spd_basics()
    demo_geodesics()
    demo_frechet_mean()
    demo_brain_connectivity()
    demo_parallel_transport()
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
