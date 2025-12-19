#!/usr/bin/env python
"""
DavisTensor GPU vs CPU Benchmark
================================

Compare manifold operation performance on CPU vs GPU.
"""

import time
import numpy as np
from typing import Tuple

from davistensor.manifolds.base import Euclidean
from davistensor.manifolds.sphere import Sphere
from davistensor.manifolds.hyperbolic import Hyperbolic
from davistensor.manifolds.spd import SPD
from davistensor.core.array_api import gpu_available, info, to_device, to_numpy


def timeit(func, n_runs: int = 50, warmup: int = 10) -> Tuple[float, float]:
    """Time a function, return (mean_ms, std_ms)."""
    for _ in range(warmup):
        func()
    
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    return np.mean(times), np.std(times)


def benchmark_manifold_device(manifold, batch: int, device: str) -> dict:
    """Benchmark manifold on specific device."""
    results = {}
    
    # random_point
    mean, std = timeit(lambda: manifold.random_point(batch, device=device))
    results['random_point'] = (mean, std)
    
    # Generate test data
    x = manifold.random_point(batch, device=device)
    y = manifold.random_point(batch, device=device)
    
    # log
    mean, std = timeit(lambda: manifold.log(x, y))
    results['log'] = (mean, std)
    
    # exp
    v = manifold.log(x, y)
    mean, std = timeit(lambda: manifold.exp(x, v))
    results['exp'] = (mean, std)
    
    # distance
    mean, std = timeit(lambda: manifold.distance(x, y))
    results['distance'] = (mean, std)
    
    # parallel_transport
    mean, std = timeit(lambda: manifold.parallel_transport(x, y, v))
    results['transport'] = (mean, std)
    
    return results


def format_speedup(cpu_ms: float, gpu_ms: float) -> str:
    """Format speedup ratio."""
    if gpu_ms > 0:
        speedup = cpu_ms / gpu_ms
        if speedup >= 1:
            return f"{speedup:.1f}x faster"
        else:
            return f"{1/speedup:.1f}x slower"
    return "N/A"


def run_benchmark():
    """Run GPU vs CPU benchmark."""
    print("=" * 70)
    print("DavisTensor GPU vs CPU Benchmark")
    print("=" * 70)
    
    info()
    
    if not gpu_available():
        print("\n❌ GPU not available! Install cupy-cuda11x or cupy-cuda12x")
        return
    
    print("\n✅ GPU detected!")
    print()
    
    # Test configurations
    configs = [
        (Euclidean(256), "Euclidean(256)", 10000),
        (Sphere(255), "Sphere(255)", 10000),
        (Hyperbolic(256), "Hyperbolic(256)", 10000),
        (SPD(16), "SPD(16)", 1000),
    ]
    
    ops = ['random_point', 'exp', 'log', 'distance', 'transport']
    
    for manifold, name, batch in configs:
        print(f"\n{'='*70}")
        print(f"{name} (batch={batch})")
        print("=" * 70)
        
        # CPU benchmark
        print("\nCPU:")
        cpu_results = benchmark_manifold_device(manifold, batch, 'cpu')
        for op in ops:
            mean, std = cpu_results[op]
            print(f"  {op:15}: {mean:8.2f} ± {std:5.2f} ms")
        
        # GPU benchmark
        print("\nGPU (CUDA):")
        gpu_results = benchmark_manifold_device(manifold, batch, 'cuda')
        for op in ops:
            mean, std = gpu_results[op]
            print(f"  {op:15}: {mean:8.2f} ± {std:5.2f} ms")
        
        # Speedup comparison
        print("\nSpeedup (GPU vs CPU):")
        for op in ops:
            cpu_mean = cpu_results[op][0]
            gpu_mean = gpu_results[op][0]
            speedup = format_speedup(cpu_mean, gpu_mean)
            print(f"  {op:15}: {speedup}")
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: GPU Speedup (batch operations)")
    print("=" * 70)
    print(f"{'Manifold':>20} | {'exp':>10} | {'log':>10} | {'distance':>10}")
    print("-" * 70)
    
    for manifold, name, batch in configs:
        cpu = benchmark_manifold_device(manifold, batch, 'cpu')
        gpu = benchmark_manifold_device(manifold, batch, 'cuda')
        
        exp_speedup = cpu['exp'][0] / gpu['exp'][0]
        log_speedup = cpu['log'][0] / gpu['log'][0]
        dist_speedup = cpu['distance'][0] / gpu['distance'][0]
        
        print(f"{name:>20} | {exp_speedup:>8.1f}x | {log_speedup:>8.1f}x | {dist_speedup:>8.1f}x")
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    run_benchmark()
