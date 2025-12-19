"""Benchmark suite for GeoTorch manifold operations and Riemannian optimizers.

This module provides benchmark scripts for evaluating the performance of
GeoTorch's manifold operations and Riemannian optimizers.

Benchmarks include:
- optimizer_step.py: Optimizer step time overhead vs PyTorch baseline
- convergence.py: Convergence speed on synthetic tasks
- hard_benchmark.py: Geometric transformer on deep hierarchies (Euclidean vs Hyperbolic)

Usage:
    python -m benchmarks.run_all
    python -m benchmarks.optimizer_step
    python -m benchmarks.convergence
    python -m benchmarks.hard_benchmark
"""

from . import optimizer_step
from . import convergence
from . import hard_benchmark

__all__ = [
    'run_all_benchmarks',
    'optimizer_step',
    'convergence', 
    'hard_benchmark',
]


def run_all_benchmarks():
    """Run all benchmark scripts."""
    print("=" * 70)
    print("GeoTorch Benchmarks")
    print("=" * 70)
    print()
    
    # Optimizer step benchmark
    print("\n>>> BENCHMARK 1: Optimizer Step Time")
    print("-" * 70)
    optimizer_step.run_benchmark()
    
    # Convergence benchmark
    print("\n>>> BENCHMARK 2: Convergence Speed")
    print("-" * 70)
    convergence.run_benchmark()
    
    # Hard benchmark (takes longer)
    print("\n>>> BENCHMARK 3: Hard Benchmark (Deep Hierarchy)")
    print("-" * 70)
    print("Skipping hard_benchmark (takes ~3 min). Run separately:")
    print("  python -m benchmarks.hard_benchmark")
    print()
    
    print("=" * 70)
    print("BENCHMARK SUITE COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    run_all_benchmarks()
