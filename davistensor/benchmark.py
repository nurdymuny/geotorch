"""
DavisTensor Benchmark Suite
============================

Performance benchmarks for manifold operations and neural network layers.
"""

import time
import numpy as np
from typing import Callable, Dict, List, Tuple

# Import DavisTensor
from davistensor import (
    ManifoldTensor, TangentTensor, Scalar,
    Euclidean, Hyperbolic, Sphere, SPD, ProductManifold,
    HyperbolicSphere, MultiHyperbolic,
)
from davistensor.core.storage import randn, tensor
from davistensor.nn import Linear, GeodesicLinear, ManifoldEmbedding, FrechetMeanPool, Sequential
from davistensor.nn.activation import ReLU


def benchmark(fn: Callable, warmup: int = 3, runs: int = 10) -> Tuple[float, float]:
    """Run benchmark and return (mean_time_ms, std_time_ms)."""
    # Warmup
    for _ in range(warmup):
        fn()
    
    # Timed runs
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    return np.mean(times), np.std(times)


def format_time(mean: float, std: float) -> str:
    """Format time with appropriate units."""
    if mean < 1:
        return f"{mean*1000:.2f} ± {std*1000:.2f} µs"
    elif mean < 1000:
        return f"{mean:.2f} ± {std:.2f} ms"
    else:
        return f"{mean/1000:.2f} ± {std/1000:.2f} s"


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def benchmark_manifold_ops(manifold, name: str, dim: int, batch_size: int = 1000):
    """Benchmark core manifold operations."""
    print_header(f"{name} (dim={dim}, batch={batch_size})")
    
    # Generate test data
    if isinstance(manifold, SPD):
        # SPD needs special handling - smaller batch for matrices
        batch_size = min(batch_size, 100)
        x = manifold.random_point(batch_size)
        y = manifold.random_point(batch_size)
        v = np.random.randn(batch_size, dim, dim)
        v = (v + v.transpose(0, 2, 1)) / 2  # Make symmetric
        v = v * 0.1  # Scale down
    else:
        x = manifold.random_point(batch_size)
        y = manifold.random_point(batch_size)
        v = np.random.randn(batch_size, manifold.ambient_dim) * 0.3
        # Project to tangent space
        if hasattr(manifold, 'project_tangent'):
            v = manifold.project_tangent(x, v)
    
    results = {}
    
    # Distance - vectorized
    def dist_fn():
        return manifold.distance(x, y)
    mean, std = benchmark(dist_fn)
    results['distance'] = (mean, std)
    print(f"  distance (×{batch_size}):     {format_time(mean, std)}")
    
    # Exponential map - vectorized
    def exp_fn():
        return manifold.exp(x, v)
    mean, std = benchmark(exp_fn)
    results['exp'] = (mean, std)
    print(f"  exp (×{batch_size}):          {format_time(mean, std)}")
    
    # Logarithmic map - vectorized
    def log_fn():
        return manifold.log(x, y)
    mean, std = benchmark(log_fn)
    results['log'] = (mean, std)
    print(f"  log (×{batch_size}):          {format_time(mean, std)}")
    
    # Parallel transport - vectorized
    def transport_fn():
        return manifold.parallel_transport(x, y, v)
    mean, std = benchmark(transport_fn)
    results['transport'] = (mean, std)
    print(f"  transport (×{batch_size}):    {format_time(mean, std)}")
    
    # Geodesic - vectorized
    def geodesic_fn():
        return manifold.geodesic(x, y, 0.5)
    mean, std = benchmark(geodesic_fn)
    results['geodesic'] = (mean, std)
    print(f"  geodesic (×{batch_size}):     {format_time(mean, std)}")
    
    # Per-operation throughput
    total_time = sum(r[0] for r in results.values())
    ops_per_sec = (batch_size * len(results)) / (total_time / 1000)
    print(f"  ─────────────────────────────────────")
    print(f"  Throughput: {ops_per_sec:,.0f} ops/sec")
    
    return results


def benchmark_nn_layers():
    """Benchmark neural network layers."""
    print_header("Neural Network Layers")
    
    batch_size = 64
    
    # Linear layer
    print("\n  Linear(256 → 128):")
    linear = Linear(256, 128)
    x_linear = randn(batch_size, 256)
    
    mean, std = benchmark(lambda: linear(x_linear))
    print(f"    forward: {format_time(mean, std)}")
    
    # GeodesicLinear
    print("\n  GeodesicLinear(Hyperbolic(64) → Euclidean(32)):")
    H = Hyperbolic(64)
    E = Euclidean(32)
    geo_linear = GeodesicLinear(H, E)
    x_hyp = H.random_point(batch_size)
    x_hyp_tensor = tensor(x_hyp)
    
    mean, std = benchmark(lambda: geo_linear(x_hyp_tensor))
    print(f"    forward: {format_time(mean, std)}")
    
    # ManifoldEmbedding
    print("\n  ManifoldEmbedding(10000, Hyperbolic(64)):")
    emb = ManifoldEmbedding(10000, Hyperbolic(64))
    indices = np.random.randint(0, 10000, size=(batch_size,))
    
    mean, std = benchmark(lambda: emb(tensor(indices)))
    print(f"    forward: {format_time(mean, std)}")
    
    # FrechetMeanPool
    print("\n  FrechetMeanPool(Sphere(31), 10 points):")
    S = Sphere(31)
    pool = FrechetMeanPool(S)
    x_sphere = tensor(S.random_point(10))
    
    mean, std = benchmark(lambda: pool(x_sphere), runs=5)
    print(f"    forward: {format_time(mean, std)}")
    
    # Sequential MLP
    print("\n  Sequential MLP (256 → 128 → 64):")
    mlp = Sequential(
        Linear(256, 128),
        ReLU(),
        Linear(128, 64),
        ReLU(),
    )
    x_mlp = randn(batch_size, 256)
    
    mean, std = benchmark(lambda: mlp(x_mlp))
    print(f"    forward: {format_time(mean, std)}")


def benchmark_product_manifold():
    """Benchmark product manifold operations."""
    print_header("Product Manifolds")
    
    batch_size = 500
    
    # HyperbolicSphere
    print("\n  HyperbolicSphere(32, 31):")
    HS = HyperbolicSphere(32, 31)
    x = HS.random_point(batch_size)
    y = HS.random_point(batch_size)
    
    mean, std = benchmark(lambda: HS.distance(x, y))
    print(f"    distance (×{batch_size}): {format_time(mean, std)}")
    
    mean, std = benchmark(lambda: HS.log(x, y))
    print(f"    log (×{batch_size}):      {format_time(mean, std)}")
    
    # MultiHyperbolic
    print("\n  MultiHyperbolic(16, 4) = H^16 × H^16 × H^16 × H^16:")
    MH = MultiHyperbolic(16, 4)
    x = MH.random_point(batch_size)
    y = MH.random_point(batch_size)
    
    mean, std = benchmark(lambda: MH.distance(x, y))
    print(f"    distance (×{batch_size}): {format_time(mean, std)}")


def benchmark_spd_special():
    """Benchmark SPD-specific operations."""
    print_header("SPD Special Operations")
    
    n = 8  # Matrix size
    batch_size = 50
    
    spd = SPD(n)
    
    # Generate SPD matrices
    points = [spd.random_point() for _ in range(batch_size)]
    
    # Matrix operations
    print(f"\n  Matrix operations (n={n}):")
    
    P = points[0]
    
    # Square root
    mean, std = benchmark(lambda: spd._sqrtm(P), runs=20)
    print(f"    sqrtm:  {format_time(mean, std)}")
    
    # Matrix log
    mean, std = benchmark(lambda: spd._logm(P), runs=20)
    print(f"    logm:   {format_time(mean, std)}")
    
    # Matrix exp
    V = np.random.randn(n, n)
    V = (V + V.T) / 2 * 0.1
    mean, std = benchmark(lambda: spd._expm(V), runs=20)
    print(f"    expm:   {format_time(mean, std)}")
    
    # Test caching benefit
    print(f"\n  Eigendecomposition caching:")
    spd2 = SPD(n)  # Fresh instance with empty cache
    
    # First pass - cold cache
    def cold_ops():
        for p in points[:10]:
            spd2._sqrtm(p)
            spd2._logm(p)
    
    spd2.clear_cache()
    mean_cold, std_cold = benchmark(cold_ops, warmup=0, runs=5)
    print(f"    cold cache (10 pts):  {format_time(mean_cold, std_cold)}")
    
    # Second pass - warm cache (same matrices)
    def warm_ops():
        for p in points[:10]:
            spd2._sqrtm(p)
            spd2._logm(p)
    
    mean_warm, std_warm = benchmark(warm_ops, warmup=0, runs=5)
    print(f"    warm cache (10 pts):  {format_time(mean_warm, std_warm)}")
    print(f"    cache speedup: {mean_cold/mean_warm:.1f}x")
    print(f"    cache stats: {spd2.cache_info}")
    
    # Fréchet mean
    print(f"\n  Fréchet mean ({batch_size} points, n={n}):")
    mean, std = benchmark(lambda: spd.frechet_mean(np.stack(points)), warmup=1, runs=5)
    print(f"    frechet_mean: {format_time(mean, std)}")


def run_all_benchmarks():
    """Run complete benchmark suite."""
    print("=" * 60)
    print(" DAVISTENSOR BENCHMARK SUITE")
    print("=" * 60)
    print(f" NumPy version: {np.__version__}")
    print(f" Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Core manifolds
    benchmark_manifold_ops(Euclidean(64), "Euclidean", 64)
    benchmark_manifold_ops(Hyperbolic(64), "Hyperbolic (Poincaré Ball)", 64)
    benchmark_manifold_ops(Sphere(63), "Sphere", 64)  # S^63 has ambient dim 64
    benchmark_manifold_ops(SPD(8), "SPD (8×8 matrices)", 8, batch_size=100)
    
    # Product manifolds
    benchmark_product_manifold()
    
    # SPD special
    benchmark_spd_special()
    
    # Neural network layers
    benchmark_nn_layers()
    
    print("\n" + "=" * 60)
    print(" BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_all_benchmarks()
