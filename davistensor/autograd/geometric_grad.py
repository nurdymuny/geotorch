"""
DavisTensor Autograd - Geometric Gradient Functions
====================================================

Backward functions for geometric (manifold) operations.
These are the key differentiators from standard autograd.
"""

from __future__ import annotations
from typing import Tuple, Optional, Callable, List
import numpy as np

from .grad_fn import GradFn, SavedContext


class ManifoldExpBackward(GradFn):
    """
    Backward for manifold exponential map: y = exp_x(v)
    
    Given grad_y (gradient w.r.t. y at y), compute:
    - grad_x: gradient w.r.t. base point x (at x)
    - grad_v: gradient w.r.t. tangent vector v (at x)
    
    Key insight: grad_y is at y, but we need grads at x.
    Must parallel transport grad_y from y back to x!
    """
    
    def __init__(self, x, v, y, manifold):
        self._inputs = (x, v)
        self.ctx = SavedContext()
        self.ctx.save_for_backward(
            x=x.numpy(),
            v=v.numpy(),
            y=y.numpy(),
            manifold=manifold
        )
    
    @property
    def inputs(self) -> Tuple:
        return self._inputs
    
    def apply(self, grad_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = self.ctx.tensors['x']
        v = self.ctx.tensors['v']
        y = self.ctx.tensors['y']
        M = self.ctx.manifolds['manifold']
        
        # Transport gradient from y back to x
        grad_y_at_x = M.parallel_transport(y, x, grad_y)
        
        # For exp map, the differential at v=0 is identity
        # For general v, need Jacobi field computation
        # Simplified: assume small v, so grad_v ≈ grad_y_at_x
        
        grad_v = grad_y_at_x
        
        # grad_x is more complex - involves derivative of exp map w.r.t. base point
        # For now, approximate with transported gradient
        grad_x = grad_y_at_x
        
        return grad_x, grad_v


class ManifoldLogBackward(GradFn):
    """
    Backward for manifold logarithm map: v = log_x(y)
    
    Given grad_v (gradient w.r.t. v at x), compute:
    - grad_x: gradient w.r.t. base point x
    - grad_y: gradient w.r.t. target point y
    """
    
    def __init__(self, x, y, manifold):
        self._inputs = (x, y)
        self.ctx = SavedContext()
        self.ctx.save_for_backward(
            x=x.numpy(),
            y=y.numpy(),
            manifold=manifold
        )
    
    @property
    def inputs(self) -> Tuple:
        return self._inputs
    
    def apply(self, grad_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = self.ctx.tensors['x']
        y = self.ctx.tensors['y']
        M = self.ctx.manifolds['manifold']
        
        # Simplified: grad_y needs transport from x to y
        grad_y_at_x = grad_v
        grad_y = M.parallel_transport(x, y, grad_y_at_x)
        
        # grad_x involves negative direction
        grad_x = -grad_v
        
        return grad_x, grad_y


class ManifoldDistanceBackward(GradFn):
    """
    Backward for geodesic distance: d = dist(x, y)
    
    d = ||log_x(y)||_x
    
    Gradients:
        ∂d/∂x = -v/||v|| where v = log_x(y)
        ∂d/∂y = transport(v/||v||, x → y)
    """
    
    def __init__(self, x, y, manifold):
        self._inputs = (x, y)
        self.ctx = SavedContext()
        
        x_np = x.numpy()
        y_np = y.numpy()
        v = manifold.log(x_np, y_np)
        v_norm = manifold.norm(x_np, v)
        
        self.ctx.save_for_backward(
            x=x_np,
            y=y_np,
            v=v,
            v_norm=v_norm,
            manifold=manifold
        )
    
    @property
    def inputs(self) -> Tuple:
        return self._inputs
    
    def apply(self, grad_d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = self.ctx.tensors['x']
        y = self.ctx.tensors['y']
        v = self.ctx.tensors['v']
        v_norm = self.ctx.scalars['v_norm']
        M = self.ctx.manifolds['manifold']
        
        if v_norm < 1e-8:
            # x ≈ y, gradients are zero
            return np.zeros_like(x), np.zeros_like(y)
        
        # Unit tangent vector
        v_unit = v / v_norm
        
        # Gradient w.r.t. x: negative direction
        grad_x = -grad_d * v_unit
        
        # Gradient w.r.t. y: transport to y
        grad_y = grad_d * M.parallel_transport(x, y, v_unit)
        
        return grad_x, grad_y


class ManifoldGeodesicBackward(GradFn):
    """
    Backward for geodesic interpolation: z = geodesic(x, y, t)
    
    z = exp_x(t * log_x(y))
    """
    
    def __init__(self, x, y, t: float, manifold):
        self._inputs = (x, y)
        self.ctx = SavedContext()
        
        x_np = x.numpy()
        y_np = y.numpy()
        
        self.ctx.save_for_backward(
            x=x_np,
            y=y_np,
            t=t,
            manifold=manifold
        )
    
    @property
    def inputs(self) -> Tuple:
        return self._inputs
    
    def apply(self, grad_z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = self.ctx.tensors['x']
        y = self.ctx.tensors['y']
        t = self.ctx.scalars['t']
        M = self.ctx.manifolds['manifold']
        
        z = M.geodesic(x, y, t)
        
        # Transport gradient from z back to x
        grad_z_at_x = M.parallel_transport(z, x, grad_z)
        
        # Approximate gradients
        grad_x = (1 - t) * grad_z_at_x
        
        # Transport to y
        grad_y_at_x = t * grad_z_at_x
        grad_y = M.parallel_transport(x, y, grad_y_at_x)
        
        return grad_x, grad_y


class FrechetMeanBackward(GradFn):
    """
    Backward for Fréchet mean computation.
    
    μ = argmin_p Σ_i w_i d(p, x_i)²
    """
    
    def __init__(self, points, mean, weights, manifold):
        self._inputs = tuple(points) if hasattr(points, '__iter__') else (points,)
        self.ctx = SavedContext()
        self.ctx.save_for_backward(
            mean=mean,
            weights=weights,
            manifold=manifold
        )
    
    @property
    def inputs(self) -> Tuple:
        return self._inputs
    
    def apply(self, grad_mean: np.ndarray) -> Tuple[np.ndarray, ...]:
        mean = self.ctx.tensors['mean']
        weights = self.ctx.tensors['weights']
        M = self.ctx.manifolds['manifold']
        
        n = len(self._inputs)
        grads = []
        
        for i, inp in enumerate(self._inputs):
            x_i = inp.numpy()
            w_i = weights[i] if weights is not None else 1.0 / n
            
            # Approximate: gradient flows through exp map
            # Transport gradient from mean to x_i
            grad_i = w_i * M.parallel_transport(mean, x_i, grad_mean)
            grads.append(grad_i)
        
        return tuple(grads)


# =============================================================================
# Gradient Checking (Numerical Verification)
# =============================================================================

def check_gradients(
    func: Callable,
    inputs: List,
    eps: float = 1e-5,
    atol: float = 1e-4,
    rtol: float = 1e-3
) -> bool:
    """
    Verify gradients numerically.
    
    Compares analytical gradients from autograd with finite differences.
    
    Parameters
    ----------
    func : Callable
        Function to differentiate. Should return scalar.
    inputs : List
        Input tensors
    eps : float
        Finite difference step size
    atol, rtol : float
        Absolute and relative tolerance
    
    Returns
    -------
    True if gradients match, raises AssertionError otherwise
    """
    from ..core.storage import tensor
    from .engine import backward
    
    # Compute analytical gradients
    for inp in inputs:
        inp.requires_grad = True
    
    output = func(*inputs)
    backward(output)
    
    analytical_grads = [inp.grad.numpy() if inp.grad is not None else None for inp in inputs]
    
    # Compute numerical gradients
    for i, inp in enumerate(inputs):
        if analytical_grads[i] is None:
            continue
        
        numerical_grad = np.zeros_like(inp.numpy())
        x_flat = inp.numpy().flatten()
        
        for j in range(len(x_flat)):
            # Forward difference
            original = x_flat[j]
            
            x_flat[j] = original + eps
            inp._storage._data[:] = x_flat
            f_plus = func(*inputs).numpy().item()
            
            x_flat[j] = original - eps
            inp._storage._data[:] = x_flat
            f_minus = func(*inputs).numpy().item()
            
            x_flat[j] = original
            inp._storage._data[:] = x_flat
            
            numerical_grad.flat[j] = (f_plus - f_minus) / (2 * eps)
        
        # Compare
        if not np.allclose(analytical_grads[i], numerical_grad, atol=atol, rtol=rtol):
            print(f"Gradient mismatch for input {i}:")
            print(f"  Analytical: {analytical_grads[i]}")
            print(f"  Numerical:  {numerical_grad}")
            print(f"  Diff:       {analytical_grads[i] - numerical_grad}")
            raise AssertionError(f"Gradient check failed for input {i}")
    
    return True


# =============================================================================
# Natural Gradient (Optional Enhancement)
# =============================================================================

def natural_gradient(
    grad: np.ndarray,
    x: np.ndarray,
    manifold
) -> np.ndarray:
    """
    Convert Riemannian gradient to natural gradient.
    
    Natural gradient = G^{-1} @ gradient
    
    Where G is the metric tensor at x.
    
    This gives the steepest descent direction in the Riemannian metric.
    """
    G = manifold.metric(x)
    G_inv = np.linalg.inv(G)
    
    # For matrix-valued gradients, need to reshape
    grad_flat = grad.flatten()
    natural_flat = G_inv @ grad_flat
    
    return natural_flat.reshape(grad.shape)


def riemannian_gradient(
    euclidean_grad: np.ndarray,
    x: np.ndarray,
    manifold
) -> np.ndarray:
    """
    Convert Euclidean gradient to Riemannian gradient.
    
    Projects the Euclidean gradient to the tangent space and
    applies the inverse metric.
    """
    # Project to tangent space
    tangent_grad = manifold.project_tangent(x, euclidean_grad)
    
    return tangent_grad
