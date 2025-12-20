#!/usr/bin/env python
"""Benchmark optimizer step time.

Compares the overhead of RiemannianSGD and RiemannianAdam against
standard PyTorch optimizers (torch.optim.SGD and torch.optim.Adam).

Metrics:
- Time per optimizer step (ms)
- Overhead percentage vs PyTorch baseline
- Breakdown by operation (gradient, projection, exp map)

Example:
    python -m benchmarks.optimizer_step
"""

import time
import torch
import torch.nn as nn
from geotorch import Sphere
from geotorch.nn import ManifoldParameter
from geotorch.optim import RiemannianSGD, RiemannianAdam


def benchmark_optimizer_step(optimizer_cls, params, n_steps=100):
    """
    Benchmark optimizer step time.
    
    Args:
        optimizer_cls: Optimizer class to benchmark
        params: List of parameters
        n_steps: Number of steps to run
    
    Returns:
        Average step time in milliseconds
    """
    # Create optimizer
    if optimizer_cls.__name__ == 'RiemannianSGD':
        optimizer = optimizer_cls(params, lr=0.01, momentum=0.9)
    elif optimizer_cls.__name__ == 'RiemannianAdam':
        optimizer = optimizer_cls(params, lr=0.001)
    elif optimizer_cls.__name__ == 'SGD':
        optimizer = optimizer_cls(params, lr=0.01, momentum=0.9)
    elif optimizer_cls.__name__ == 'Adam':
        optimizer = optimizer_cls(params, lr=0.001)
    else:
        optimizer = optimizer_cls(params, lr=0.01)
    
    # Warmup
    for _ in range(10):
        optimizer.zero_grad()
        loss = sum((p ** 2).sum() for p in params)
        loss.backward()
        optimizer.step()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(n_steps):
        optimizer.zero_grad()
        loss = sum((p ** 2).sum() for p in params)
        loss.backward()
        optimizer.step()
    end = time.perf_counter()
    
    avg_step_time_ms = (end - start) / n_steps * 1000
    return avg_step_time_ms


def run_benchmark():
    """Run optimizer step benchmarks."""
    print("Optimizer Step Time Benchmark")
    print("=" * 80)
    print()
    
    # Setup
    manifold = Sphere(128)
    n_params = 5
    
    print(f"Configuration:")
    print(f"  Manifold: Sphere(128)")
    print(f"  Number of parameters: {n_params}")
    print(f"  Steps: 100")
    print()
    
    # Benchmark RiemannianSGD
    print("Benchmarking RiemannianSGD...")
    params_rsgd = [ManifoldParameter(manifold.random_point(), manifold) 
                   for _ in range(n_params)]
    rsgd_time = benchmark_optimizer_step(RiemannianSGD, params_rsgd)
    print(f"  Average step time: {rsgd_time:.3f} ms")
    print()
    
    # Benchmark RiemannianAdam
    print("Benchmarking RiemannianAdam...")
    params_radam = [ManifoldParameter(manifold.random_point(), manifold) 
                    for _ in range(n_params)]
    radam_time = benchmark_optimizer_step(RiemannianAdam, params_radam)
    print(f"  Average step time: {radam_time:.3f} ms")
    print()
    
    # Benchmark torch.optim.SGD (baseline)
    print("Benchmarking torch.optim.SGD (baseline)...")
    params_sgd = [nn.Parameter(torch.randn(128)) for _ in range(n_params)]
    sgd_time = benchmark_optimizer_step(torch.optim.SGD, params_sgd)
    print(f"  Average step time: {sgd_time:.3f} ms")
    print()
    
    # Benchmark torch.optim.Adam (baseline)
    print("Benchmarking torch.optim.Adam (baseline)...")
    params_adam = [nn.Parameter(torch.randn(128)) for _ in range(n_params)]
    adam_time = benchmark_optimizer_step(torch.optim.Adam, params_adam)
    print(f"  Average step time: {adam_time:.3f} ms")
    print()
    
    # Summary
    print("Summary")
    print("-" * 80)
    print(f"RiemannianSGD:    {rsgd_time:.3f} ms  "
          f"(+{(rsgd_time/sgd_time - 1)*100:.1f}% vs SGD)")
    print(f"RiemannianAdam:   {radam_time:.3f} ms  "
          f"(+{(radam_time/adam_time - 1)*100:.1f}% vs Adam)")
    print(f"torch.optim.SGD:  {sgd_time:.3f} ms  (baseline)")
    print(f"torch.optim.Adam: {adam_time:.3f} ms  (baseline)")
    print()
    print(f"Target overhead: <50%")
    print()


if __name__ == '__main__':
    run_benchmark()
