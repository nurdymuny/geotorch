#!/usr/bin/env python
"""Benchmark convergence speed.

Compares convergence speed of Riemannian optimizers vs standard optimizers
on synthetic manifold optimization tasks.

Metrics:
- Steps to reach target loss/distance
- Wall-clock time to convergence
- Final distance to target

Example:
    python -m benchmarks.convergence
"""

import time
import torch
from geotorch import Sphere
from geotorch.nn import ManifoldParameter
from geotorch.optim import RiemannianSGD, RiemannianAdam


def benchmark_convergence_sphere(optimizer_cls, lr=0.1, max_steps=500, target_distance=1e-4):
    """
    Benchmark convergence on sphere distance minimization task.
    
    Args:
        optimizer_cls: Optimizer class
        lr: Learning rate
        max_steps: Maximum optimization steps
        target_distance: Target distance to stop
    
    Returns:
        Tuple of (steps_to_converge, wall_time, final_distance)
    """
    manifold = Sphere(128)
    param = ManifoldParameter(manifold.random_point(), manifold)
    target = manifold.random_point()
    
    # Create optimizer
    if 'Adam' in optimizer_cls.__name__:
        optimizer = optimizer_cls([param], lr=lr, betas=(0.9, 0.999))
    else:
        optimizer = optimizer_cls([param], lr=lr, momentum=0.9)
    
    start_time = time.perf_counter()
    
    for step in range(max_steps):
        optimizer.zero_grad()
        loss = ((param - target) ** 2).sum()
        loss.backward()
        optimizer.step()
        
        # Check convergence
        current_distance = ((param - target) ** 2).sum().sqrt().item()
        if current_distance < target_distance:
            break
    
    end_time = time.perf_counter()
    wall_time = end_time - start_time
    final_distance = ((param - target) ** 2).sum().sqrt().item()
    
    return step + 1, wall_time, final_distance


def run_benchmark():
    """Run convergence benchmarks."""
    print("Convergence Speed Benchmark")
    print("=" * 80)
    print()
    
    print("Task: Minimize distance to target on Sphere(128)")
    print("Target distance: 1e-4")
    print("Max steps: 500")
    print()
    
    # Benchmark RiemannianSGD
    print("Benchmarking RiemannianSGD...")
    steps_rsgd, time_rsgd, dist_rsgd = benchmark_convergence_sphere(
        RiemannianSGD, lr=0.1
    )
    print(f"  Steps to converge: {steps_rsgd}")
    print(f"  Wall time: {time_rsgd:.3f} s")
    print(f"  Final distance: {dist_rsgd:.6f}")
    print()
    
    # Benchmark RiemannianAdam
    print("Benchmarking RiemannianAdam...")
    steps_radam, time_radam, dist_radam = benchmark_convergence_sphere(
        RiemannianAdam, lr=0.1
    )
    print(f"  Steps to converge: {steps_radam}")
    print(f"  Wall time: {time_radam:.3f} s")
    print(f"  Final distance: {dist_radam:.6f}")
    print()
    
    # Summary
    print("Summary")
    print("-" * 80)
    print(f"RiemannianSGD:  {steps_rsgd:3d} steps, {time_rsgd:.3f}s, "
          f"distance={dist_rsgd:.6f}")
    print(f"RiemannianAdam: {steps_radam:3d} steps, {time_radam:.3f}s, "
          f"distance={dist_radam:.6f}")
    print()
    
    if dist_rsgd < 1e-4 and dist_radam < 1e-4:
        print("✓ Both optimizers converged to target")
    else:
        print("✗ One or more optimizers did not reach target distance")
    print()


if __name__ == '__main__':
    run_benchmark()
