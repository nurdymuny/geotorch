#!/usr/bin/env python
"""Demo: Using Riemannian Optimizers with GeoTorch

This script demonstrates how to use RiemannianSGD and RiemannianAdam
optimizers to optimize parameters on the sphere manifold.

Example:
    python examples/optimizer_demo.py
"""

import torch
import torch.nn as nn
from geotorch import Sphere
from geotorch.nn import ManifoldParameter
from geotorch.optim import RiemannianSGD, RiemannianAdam


def demo_basic_optimization():
    """Basic example: Minimize distance to target point on sphere."""
    print("=" * 80)
    print("Demo 1: Basic Optimization on Sphere")
    print("=" * 80)
    
    # Create manifold and parameters
    manifold = Sphere(64)
    param = ManifoldParameter(manifold.random_point(), manifold)
    target = torch.randn(64)
    target = target / target.norm()  # Normalize to sphere
    
    # Create optimizer
    optimizer = RiemannianSGD([param], lr=0.1, momentum=0.9)
    
    initial_distance = ((param - target) ** 2).sum().sqrt().item()
    print(f"Initial distance to target: {initial_distance:.4f}")
    
    # Optimize
    for i in range(100):
        optimizer.zero_grad()
        loss = ((param - target) ** 2).sum()
        loss.backward()
        optimizer.step()
        
        if i % 20 == 0:
            distance = ((param - target) ** 2).sum().sqrt().item()
            print(f"Step {i:3d}: distance = {distance:.6f}, param_norm = {param.norm():.6f}")
    
    final_distance = ((param - target) ** 2).sum().sqrt().item()
    print(f"Final distance to target: {final_distance:.6f}")
    print(f"Distance reduction: {(1 - final_distance/initial_distance) * 100:.1f}%")
    print()


if __name__ == '__main__':
    print()
    print("GeoTorch Riemannian Optimizer Demo")
    print()
    
    demo_basic_optimization()
    
    print("=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)
