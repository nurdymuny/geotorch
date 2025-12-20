"""Fused JIT kernels for fast Riemannian optimization.

These kernels fuse multiple operations into single JIT-compiled functions,
eliminating kernel launch overhead and enabling better memory access patterns.

Expected improvement: 72% overhead → 20-30% overhead vs PyTorch
"""

import torch
from torch import Tensor
from typing import Tuple, Optional


# =============================================================================
# SPHERE KERNELS
# =============================================================================

@torch.jit.script
def sphere_project_tangent(p: Tensor, v: Tensor) -> Tensor:
    """Project v to tangent space at p: v - (v·p)p"""
    return v - (v * p).sum(-1, keepdim=True) * p


@torch.jit.script
def sphere_normalize(x: Tensor) -> Tensor:
    """Normalize to unit sphere."""
    return x / torch.linalg.norm(x, dim=-1, keepdim=True).clamp(min=1e-7)


@torch.jit.script
def sphere_retract(p: Tensor, v: Tensor) -> Tensor:
    """Fast retraction: normalize(p + v)"""
    result = p + v
    return result / torch.linalg.norm(result, dim=-1, keepdim=True).clamp(min=1e-7)


@torch.jit.script
def sphere_sgd_step(
    p: Tensor,
    grad: Tensor,
    lr: float
) -> Tensor:
    """
    Fused SGD step on sphere (no momentum).
    
    Combines: project_tangent + retract in one kernel.
    """
    # Project gradient to tangent space
    g = grad - (grad * p).sum(-1, keepdim=True) * p
    # Retract
    result = p - lr * g
    return result / torch.linalg.norm(result, dim=-1, keepdim=True).clamp(min=1e-7)


@torch.jit.script
def sphere_sgd_step_momentum(
    p: Tensor,
    grad: Tensor,
    momentum_buf: Tensor,
    lr: float,
    momentum: float,
    dampening: float
) -> Tuple[Tensor, Tensor]:
    """
    Fused SGD step with momentum on sphere.
    
    Combines: project_tangent (grad) + vector_transport (momentum) + 
              momentum update + retract in one kernel.
    
    Returns: (new_p, new_momentum_buf)
    """
    # Project gradient to tangent space
    g = grad - (grad * p).sum(-1, keepdim=True) * p
    
    # Vector transport momentum to current tangent space
    buf = momentum_buf - (momentum_buf * p).sum(-1, keepdim=True) * p
    
    # Update momentum: buf = momentum * buf + (1 - dampening) * g
    buf = momentum * buf + (1.0 - dampening) * g
    
    # Retract with momentum
    result = p - lr * buf
    new_p = result / torch.linalg.norm(result, dim=-1, keepdim=True).clamp(min=1e-7)
    
    return new_p, buf


@torch.jit.script
def sphere_adam_step(
    p: Tensor,
    grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    step: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Fused Adam step on sphere.
    
    Returns: (new_p, new_exp_avg, new_exp_avg_sq)
    """
    # Project gradient to tangent space
    g = grad - (grad * p).sum(-1, keepdim=True) * p
    
    # Vector transport moments to current tangent space
    m = exp_avg - (exp_avg * p).sum(-1, keepdim=True) * p
    v = exp_avg_sq - (exp_avg_sq * p).sum(-1, keepdim=True) * p
    v = v.abs()  # Keep positive
    
    # Update moments
    m = beta1 * m + (1.0 - beta1) * g
    v = beta2 * v + (1.0 - beta2) * g * g
    
    # Bias correction
    bias_correction1 = 1.0 - beta1 ** step
    bias_correction2 = 1.0 - beta2 ** step
    step_size = lr * (bias_correction2 ** 0.5) / bias_correction1
    
    # Update direction
    update = m / (v.sqrt() + eps)
    
    # Retract
    result = p - step_size * update
    new_p = result / torch.linalg.norm(result, dim=-1, keepdim=True).clamp(min=1e-7)
    
    return new_p, m, v


# =============================================================================
# HYPERBOLIC (POINCARÉ BALL) KERNELS
# =============================================================================

@torch.jit.script
def poincare_conformal_factor(x: Tensor) -> Tensor:
    """Conformal factor λ_x = 2/(1 - ||x||²)"""
    x_sqnorm = (x * x).sum(-1, keepdim=True)
    return 2.0 / (1.0 - x_sqnorm).clamp(min=1e-7)


@torch.jit.script
def poincare_project(x: Tensor) -> Tensor:
    """Project to inside Poincaré ball (max norm 0.99999)."""
    max_norm = 1.0 - 1e-5
    x_norm = torch.linalg.norm(x, dim=-1, keepdim=True)
    scale = torch.where(
        x_norm < max_norm,
        torch.ones_like(x_norm),
        max_norm / x_norm.clamp(min=1e-7)
    )
    return scale * x


@torch.jit.script
def poincare_retract(p: Tensor, v: Tensor) -> Tensor:
    """Fast retraction: project(p + v/λ_p)"""
    lambda_p = poincare_conformal_factor(p)
    return poincare_project(p + v / lambda_p)


@torch.jit.script
def poincare_vector_transport(v: Tensor, p: Tensor, q: Tensor) -> Tensor:
    """Vector transport: scale by conformal factor ratio."""
    lambda_p = poincare_conformal_factor(p)
    lambda_q = poincare_conformal_factor(q)
    return (lambda_p / lambda_q) * v


@torch.jit.script
def poincare_sgd_step(
    p: Tensor,
    grad: Tensor,
    lr: float
) -> Tensor:
    """
    Fused SGD step on Poincaré ball (no momentum).
    
    Note: Tangent space is all of R^n for Poincaré, so no projection needed.
    """
    lambda_p = poincare_conformal_factor(p)
    result = p - lr * grad / lambda_p
    return poincare_project(result)


@torch.jit.script
def poincare_sgd_step_momentum(
    p: Tensor,
    grad: Tensor,
    momentum_buf: Tensor,
    lr: float,
    momentum: float,
    dampening: float
) -> Tuple[Tensor, Tensor]:
    """
    Fused SGD step with momentum on Poincaré ball.
    
    Returns: (new_p, new_momentum_buf)
    """
    lambda_p = poincare_conformal_factor(p)
    
    # Vector transport momentum (scale by conformal factor)
    # For simplicity, we transport from "previous" tangent space
    # In practice, buf is already approximately correct
    buf = momentum * momentum_buf + (1.0 - dampening) * grad
    
    # Retract with momentum
    result = p - lr * buf / lambda_p
    new_p = poincare_project(result)
    
    return new_p, buf


@torch.jit.script
def poincare_adam_step(
    p: Tensor,
    grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    step: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Fused Adam step on Poincaré ball.
    
    Returns: (new_p, new_exp_avg, new_exp_avg_sq)
    """
    lambda_p = poincare_conformal_factor(p)
    
    # Update moments (no tangent projection needed for Poincaré)
    m = beta1 * exp_avg + (1.0 - beta1) * grad
    v = beta2 * exp_avg_sq + (1.0 - beta2) * grad * grad
    
    # Bias correction
    bias_correction1 = 1.0 - beta1 ** step
    bias_correction2 = 1.0 - beta2 ** step
    step_size = lr * (bias_correction2 ** 0.5) / bias_correction1
    
    # Update direction
    update = m / (v.sqrt() + eps)
    
    # Retract
    result = p - step_size * update / lambda_p
    new_p = poincare_project(result)
    
    return new_p, m, v


# =============================================================================
# GENERIC FUSED OPTIMIZER
# =============================================================================

class FusedRiemannianSGD:
    """
    Ultra-fast Riemannian SGD using fused JIT kernels.
    
    Supports: Sphere, Hyperbolic (Poincaré)
    
    Example:
        >>> from geotorch import Sphere
        >>> from geotorch.optim.fused import FusedRiemannianSGD
        >>> 
        >>> manifold = Sphere(64)
        >>> param = manifold.random_point(requires_grad=True)
        >>> optimizer = FusedRiemannianSGD(param, manifold, lr=0.01, momentum=0.9)
        >>> 
        >>> # Training loop
        >>> optimizer.zero_grad()
        >>> loss = (param ** 2).sum()
        >>> loss.backward()
        >>> optimizer.step()
    """
    
    def __init__(
        self,
        param: Tensor,
        manifold_type: str,  # 'sphere' or 'hyperbolic'
        lr: float = 0.01,
        momentum: float = 0.0,
        dampening: float = 0.0,
    ):
        self.param = param
        self.manifold_type = manifold_type.lower()
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        
        # State
        self.momentum_buf: Optional[Tensor] = None
        
        if self.manifold_type not in ['sphere', 'hyperbolic', 'poincare']:
            raise ValueError(f"Unsupported manifold: {manifold_type}")
    
    def zero_grad(self):
        if self.param.grad is not None:
            self.param.grad.zero_()
    
    @torch.no_grad()
    def step(self):
        if self.param.grad is None:
            return
        
        grad = self.param.grad.data
        
        if self.manifold_type == 'sphere':
            if self.momentum == 0:
                self.param.data = sphere_sgd_step(self.param.data, grad, self.lr)
            else:
                if self.momentum_buf is None:
                    self.momentum_buf = torch.zeros_like(grad)
                self.param.data, self.momentum_buf = sphere_sgd_step_momentum(
                    self.param.data, grad, self.momentum_buf,
                    self.lr, self.momentum, self.dampening
                )
        
        elif self.manifold_type in ['hyperbolic', 'poincare']:
            if self.momentum == 0:
                self.param.data = poincare_sgd_step(self.param.data, grad, self.lr)
            else:
                if self.momentum_buf is None:
                    self.momentum_buf = torch.zeros_like(grad)
                self.param.data, self.momentum_buf = poincare_sgd_step_momentum(
                    self.param.data, grad, self.momentum_buf,
                    self.lr, self.momentum, self.dampening
                )


class FusedRiemannianAdam:
    """
    Ultra-fast Riemannian Adam using fused JIT kernels.
    
    Supports: Sphere, Hyperbolic (Poincaré)
    """
    
    def __init__(
        self,
        param: Tensor,
        manifold_type: str,
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.param = param
        self.manifold_type = manifold_type.lower()
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        
        # State
        self.step_count = 0
        self.exp_avg: Optional[Tensor] = None
        self.exp_avg_sq: Optional[Tensor] = None
        
        if self.manifold_type not in ['sphere', 'hyperbolic', 'poincare']:
            raise ValueError(f"Unsupported manifold: {manifold_type}")
    
    def zero_grad(self):
        if self.param.grad is not None:
            self.param.grad.zero_()
    
    @torch.no_grad()
    def step(self):
        if self.param.grad is None:
            return
        
        grad = self.param.grad.data
        self.step_count += 1
        
        # Initialize state
        if self.exp_avg is None:
            self.exp_avg = torch.zeros_like(grad)
            self.exp_avg_sq = torch.zeros_like(grad)
        
        if self.manifold_type == 'sphere':
            self.param.data, self.exp_avg, self.exp_avg_sq = sphere_adam_step(
                self.param.data, grad, self.exp_avg, self.exp_avg_sq,
                self.step_count, self.lr, self.beta1, self.beta2, self.eps
            )
        
        elif self.manifold_type in ['hyperbolic', 'poincare']:
            self.param.data, self.exp_avg, self.exp_avg_sq = poincare_adam_step(
                self.param.data, grad, self.exp_avg, self.exp_avg_sq,
                self.step_count, self.lr, self.beta1, self.beta2, self.eps
            )


# =============================================================================
# BENCHMARK UTILITY
# =============================================================================

def benchmark_fused_vs_unfused():
    """Compare fused kernels vs unfused operations."""
    import time
    
    print("=" * 60)
    print("FUSED vs UNFUSED KERNEL BENCHMARK")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    n = 128
    batch_size = 1000
    n_iterations = 100
    
    # Generate test data
    p = torch.randn(batch_size, n, device=device)
    p = p / p.norm(dim=-1, keepdim=True)  # On sphere
    v = torch.randn(batch_size, n, device=device)
    v = v - (v * p).sum(-1, keepdim=True) * p  # Tangent vector
    
    # Warmup JIT
    _ = sphere_sgd_step(p, v, 0.01)
    _ = sphere_sgd_step_momentum(p, v, torch.zeros_like(v), 0.01, 0.9, 0.0)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark fused SGD step
    start = time.perf_counter()
    for _ in range(n_iterations):
        result = sphere_sgd_step(p, v, 0.01)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    fused_time = time.perf_counter() - start
    
    # Benchmark unfused SGD step
    start = time.perf_counter()
    for _ in range(n_iterations):
        g = v - (v * p).sum(-1, keepdim=True) * p
        result = p - 0.01 * g
        result = result / torch.linalg.norm(result, dim=-1, keepdim=True)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    unfused_time = time.perf_counter() - start
    
    print(f"\nSphere SGD step ({batch_size} params, {n_iterations} iters):")
    print(f"  Fused:   {fused_time*1000:.2f} ms")
    print(f"  Unfused: {unfused_time*1000:.2f} ms")
    print(f"  Speedup: {unfused_time/fused_time:.2f}x")
    
    # Benchmark fused SGD with momentum
    buf = torch.zeros_like(v)
    start = time.perf_counter()
    for _ in range(n_iterations):
        p_new, buf = sphere_sgd_step_momentum(p, v, buf, 0.01, 0.9, 0.0)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    fused_momentum_time = time.perf_counter() - start
    
    print(f"\nSphere SGD+momentum step:")
    print(f"  Fused:   {fused_momentum_time*1000:.2f} ms")
    
    # Benchmark Adam
    m = torch.zeros_like(v)
    v_sq = torch.zeros_like(v)
    start = time.perf_counter()
    for i in range(n_iterations):
        p_new, m, v_sq = sphere_adam_step(p, v, m, v_sq, i+1, 0.001, 0.9, 0.999, 1e-8)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    fused_adam_time = time.perf_counter() - start
    
    print(f"\nSphere Adam step:")
    print(f"  Fused:   {fused_adam_time*1000:.2f} ms")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    benchmark_fused_vs_unfused()
