"""Benchmark: Retraction vs Exp map, Vector transport vs Parallel transport."""

import torch
import time
from geotorch.manifolds import Sphere, Hyperbolic
from geotorch.nn import ManifoldParameter
from geotorch.optim import RiemannianSGD, RiemannianAdam, FusedRiemannianSGD, FusedRiemannianAdam

torch.manual_seed(42)

def benchmark_manifold_ops(manifold, name, n_iters=10000):
    """Benchmark exp vs retract, parallel_transport vs vector_transport."""
    p = manifold.random_point()
    v = manifold.random_tangent(p) * 0.01  # Small step like in optimization
    q = manifold.exp(p, v)
    
    # Warm up
    for _ in range(100):
        _ = manifold.exp(p, v)
        _ = manifold.retract(p, v)
    
    # Benchmark exp
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = manifold.exp(p, v)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    exp_time = time.perf_counter() - start
    
    # Benchmark retract
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = manifold.retract(p, v)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    retract_time = time.perf_counter() - start
    
    # Benchmark parallel_transport
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = manifold.parallel_transport(v, p, q)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    pt_time = time.perf_counter() - start
    
    # Benchmark vector_transport
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = manifold.vector_transport(v, p, q)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    vt_time = time.perf_counter() - start
    
    print(f"\n{name} (dim={manifold.dim + 1}):")
    print(f"  exp():              {exp_time*1000:.2f} ms ({n_iters} iters)")
    print(f"  retract():          {retract_time*1000:.2f} ms ({n_iters} iters)")
    print(f"  -> Speedup:         {exp_time/retract_time:.2f}x")
    print(f"  parallel_transport: {pt_time*1000:.2f} ms ({n_iters} iters)")
    print(f"  vector_transport:   {vt_time*1000:.2f} ms ({n_iters} iters)")
    print(f"  -> Speedup:         {pt_time/vt_time:.2f}x")


def benchmark_optimizer(manifold, optimizer_cls, name, n_steps=100, n_params=5):
    """Benchmark optimizer with use_retraction=True vs False."""
    
    def run_optimization(use_retraction):
        torch.manual_seed(42)
        params = [ManifoldParameter(manifold.random_point(), manifold) for _ in range(n_params)]
        optimizer = optimizer_cls(params, lr=0.01, use_retraction=use_retraction)
        
        # Warm up
        for _ in range(10):
            optimizer.zero_grad()
            loss = sum((p ** 2).sum() for p in params)
            loss.backward()
            optimizer.step()
        
        # Reset
        torch.manual_seed(42)
        params = [ManifoldParameter(manifold.random_point(), manifold) for _ in range(n_params)]
        optimizer = optimizer_cls(params, lr=0.01, use_retraction=use_retraction)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        for _ in range(n_steps):
            optimizer.zero_grad()
            loss = sum((p ** 2).sum() for p in params)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        return time.perf_counter() - start
    
    time_with_retraction = run_optimization(use_retraction=True)
    time_without_retraction = run_optimization(use_retraction=False)
    
    print(f"\n{name} on {manifold.__class__.__name__} ({n_params} params, {n_steps} steps):")
    print(f"  use_retraction=True:  {time_with_retraction*1000:.2f} ms")
    print(f"  use_retraction=False: {time_without_retraction*1000:.2f} ms")
    print(f"  -> Speedup:           {time_without_retraction/time_with_retraction:.2f}x")


def benchmark_vs_pytorch(n_steps=100, n_params=5, dim=128):
    """Compare Riemannian optimizer overhead vs standard PyTorch."""
    
    # PyTorch baseline
    torch.manual_seed(42)
    params = [torch.nn.Parameter(torch.randn(dim)) for _ in range(n_params)]
    optimizer = torch.optim.SGD(params, lr=0.01)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    for _ in range(n_steps):
        optimizer.zero_grad()
        loss = sum((p ** 2).sum() for p in params)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    pytorch_time = time.perf_counter() - start
    
    # Riemannian SGD with retraction
    manifold = Sphere(dim)
    torch.manual_seed(42)
    params = [ManifoldParameter(manifold.random_point(), manifold) for _ in range(n_params)]
    optimizer = RiemannianSGD(params, lr=0.01, use_retraction=True)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    for _ in range(n_steps):
        optimizer.zero_grad()
        loss = sum((p ** 2).sum() for p in params)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    rsgd_time = time.perf_counter() - start
    
    # Fused Riemannian SGD (single param for now)
    torch.manual_seed(42)
    param = manifold.random_point().requires_grad_(True)
    fused_opt = FusedRiemannianSGD(param, 'sphere', lr=0.01)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    for _ in range(n_steps):
        fused_opt.zero_grad()
        loss = (param ** 2).sum()
        loss.backward()
        fused_opt.step()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    fused_time = time.perf_counter() - start
    
    print(f"\nOverhead vs PyTorch SGD ({n_params} params x {dim} dim, {n_steps} steps):")
    print(f"  PyTorch SGD:        {pytorch_time*1000:.2f} ms (baseline)")
    print(f"  RiemannianSGD:      {rsgd_time*1000:.2f} ms  -> {(rsgd_time/pytorch_time - 1)*100:+.1f}% overhead")
    print(f"  FusedRiemannianSGD: {fused_time*1000:.2f} ms  -> {(fused_time/pytorch_time - 1)*100:+.1f}% overhead (1 param)")


def benchmark_fused_optimizers(n_steps=200, dim=128):
    """Benchmark fused vs unfused optimizers."""
    manifold = Sphere(dim)
    
    # --- SGD comparison ---
    # Unfused
    torch.manual_seed(42)
    param = ManifoldParameter(manifold.random_point(), manifold)
    optimizer = RiemannianSGD([param], lr=0.01, momentum=0.9, use_retraction=True)
    
    start = time.perf_counter()
    for _ in range(n_steps):
        optimizer.zero_grad()
        loss = (param ** 2).sum()
        loss.backward()
        optimizer.step()
    unfused_sgd_time = time.perf_counter() - start
    
    # Fused
    torch.manual_seed(42)
    param = manifold.random_point().requires_grad_(True)
    fused_opt = FusedRiemannianSGD(param, 'sphere', lr=0.01, momentum=0.9)
    
    start = time.perf_counter()
    for _ in range(n_steps):
        fused_opt.zero_grad()
        loss = (param ** 2).sum()
        loss.backward()
        fused_opt.step()
    fused_sgd_time = time.perf_counter() - start
    
    # --- Adam comparison ---
    # Unfused
    torch.manual_seed(42)
    param = ManifoldParameter(manifold.random_point(), manifold)
    optimizer = RiemannianAdam([param], lr=0.001, use_retraction=True)
    
    start = time.perf_counter()
    for _ in range(n_steps):
        optimizer.zero_grad()
        loss = (param ** 2).sum()
        loss.backward()
        optimizer.step()
    unfused_adam_time = time.perf_counter() - start
    
    # Fused
    torch.manual_seed(42)
    param = manifold.random_point().requires_grad_(True)
    fused_opt = FusedRiemannianAdam(param, 'sphere', lr=0.001)
    
    start = time.perf_counter()
    for _ in range(n_steps):
        fused_opt.zero_grad()
        loss = (param ** 2).sum()
        loss.backward()
        fused_opt.step()
    fused_adam_time = time.perf_counter() - start
    
    print(f"\n--- Fused vs Unfused ({n_steps} steps, dim={dim}) ---")
    print(f"  RiemannianSGD:       {unfused_sgd_time*1000:.2f} ms")
    print(f"  FusedRiemannianSGD:  {fused_sgd_time*1000:.2f} ms  -> {unfused_sgd_time/fused_sgd_time:.2f}x speedup")
    print(f"  RiemannianAdam:      {unfused_adam_time*1000:.2f} ms")
    print(f"  FusedRiemannianAdam: {fused_adam_time*1000:.2f} ms  -> {unfused_adam_time/fused_adam_time:.2f}x speedup")


if __name__ == "__main__":
    print("=" * 60)
    print("GeoTorch Performance Benchmarks")
    print("=" * 60)
    
    # Manifold operation benchmarks
    print("\n--- Manifold Operations ---")
    benchmark_manifold_ops(Sphere(128), "Sphere")
    benchmark_manifold_ops(Hyperbolic(128), "Hyperbolic")
    
    # Optimizer benchmarks
    print("\n--- Optimizer Benchmarks ---")
    benchmark_optimizer(Sphere(128), RiemannianSGD, "RiemannianSGD")
    benchmark_optimizer(Sphere(128), RiemannianAdam, "RiemannianAdam")
    benchmark_optimizer(Hyperbolic(128), RiemannianSGD, "RiemannianSGD")
    
    # Overhead comparison
    print("\n--- Overhead vs PyTorch ---")
    benchmark_vs_pytorch()
    
    # Fused vs unfused
    print("\n--- Fused JIT Kernels ---")
    benchmark_fused_optimizers()
    
    print("\n" + "=" * 60)
    print("Done!")
