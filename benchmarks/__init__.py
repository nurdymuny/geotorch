"""Benchmark suite for GeoTorch optimizers.

This module provides benchmark scripts for evaluating the performance of
Riemannian optimizers compared to standard PyTorch optimizers.

Benchmarks include:
- Optimizer step time overhead
- Convergence speed on synthetic tasks
- Memory usage
- Throughput on real datasets

Usage:
    python -m benchmarks.run_all
    python -m benchmarks.optimizer_step
    python -m benchmarks.convergence
"""

__all__ = [
    'run_all_benchmarks',
]


def run_all_benchmarks():
    """Run all benchmark scripts."""
    print("GeoTorch Benchmarks")
    print("=" * 80)
    print()
    print("To be implemented:")
    print("- optimizer_step.py: Measure step time overhead")
    print("- convergence.py: Compare convergence speed")
    print("- memory_usage.py: Track memory consumption")
    print("- throughput.py: Measure training throughput")
    print()
    print("Run individual benchmarks with:")
    print("  python -m benchmarks.optimizer_step")
    print("  python -m benchmarks.convergence")
    print()


if __name__ == '__main__':
    run_all_benchmarks()
