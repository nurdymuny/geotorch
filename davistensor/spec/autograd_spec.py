"""
DavisTensor Autograd Specification
==================================

Geometry-aware automatic differentiation.

THE KEY INSIGHT:
    Gradients are TANGENT VECTORS, not ambient vectors.
    
    In standard autograd (PyTorch):
        grad = ∂L/∂x ∈ R^n (ambient space)
        
    In Riemannian autograd (DavisTensor):
        grad = Riemannian gradient ∈ T_x M (tangent space at x)

This requires:
1. Automatic projection to tangent space
2. Parallel transport when combining gradients
3. Metric-aware preconditioning (optional: natural gradient)

IMPLEMENTATION SPEC
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import weakref

# Forward references
from ..core.storage import TensorCore, GeometricType
from ..manifolds.base import Manifold


# =============================================================================
# Graph Node Types
# =============================================================================

class GradFn(ABC):
    """
    Base class for backward functions.
    
    Each operation defines how gradients flow backward.
    """
    
    @abstractmethod
    def apply(self, grad_output: np.ndarray) -> Tuple[Optional[np.ndarray], ...]:
        """
        Compute gradients w.r.t. inputs given gradient of output.
        
        Parameters
        ----------
        grad_output : np.ndarray
            Gradient of loss w.r.t. this operation's output
        
        Returns
        -------
        Tuple of gradients for each input (None if input doesn't require grad)
        """
        ...
    
    @property
    @abstractmethod
    def inputs(self) -> Tuple['TensorCore', ...]:
        """Input tensors to this operation."""
        ...


@dataclass
class SavedContext:
    """
    Saved tensors and metadata for backward pass.
    """
    tensors: Dict[str, np.ndarray] = field(default_factory=dict)
    manifolds: Dict[str, Manifold] = field(default_factory=dict)
    scalars: Dict[str, Any] = field(default_factory=dict)
    
    def save_for_backward(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                self.tensors[k] = v
            elif isinstance(v, Manifold):
                self.manifolds[k] = v
            else:
                self.scalars[k] = v


# =============================================================================
# Gradient Tape (Records Operations)
# =============================================================================

class GradientTape:
    """
    Records operations for backward pass.
    
    Usage:
        with GradientTape() as tape:
            y = f(x)
            loss = g(y)
        
        grads = tape.gradient(loss, [x])
    
    Or implicit (like PyTorch):
        x.requires_grad = True
        y = f(x)
        loss = g(y)
        loss.backward()
        print(x.grad)
    """
    
    # Global tape for implicit recording
    _default_tape: Optional['GradientTape'] = None
    
    def __init__(self, persistent: bool = False):
        """
        Parameters
        ----------
        persistent : bool
            If True, tape can be used multiple times
        """
        self.persistent = persistent
        self._operations: List[Tuple[GradFn, TensorCore]] = []
        self._watched: Set[int] = set()  # IDs of tensors to track
    
    def __enter__(self) -> 'GradientTape':
        GradientTape._default_tape = self
        return self
    
    def __exit__(self, *args):
        if not self.persistent:
            GradientTape._default_tape = None
    
    def watch(self, tensor: TensorCore):
        """Explicitly watch a tensor for gradients."""
        self._watched.add(id(tensor))
    
    def record(self, grad_fn: GradFn, output: TensorCore):
        """Record an operation."""
        self._operations.append((grad_fn, output))
        output.grad_fn = grad_fn
    
    @staticmethod
    def get_default() -> Optional['GradientTape']:
        return GradientTape._default_tape
    
    def gradient(
        self, 
        target: TensorCore, 
        sources: List[TensorCore],
        output_gradients: Optional[np.ndarray] = None
    ) -> List[Optional[np.ndarray]]:
        """
        Compute gradients of target w.r.t. sources.
        
        Parameters
        ----------
        target : TensorCore
            Scalar loss to differentiate
        sources : List[TensorCore]
            Variables to compute gradients for
        output_gradients : Optional[np.ndarray]
            Initial gradient (defaults to 1 for scalar)
        
        Returns
        -------
        List of gradients, one per source
        """
        return backward(target, sources, output_gradients)


# =============================================================================
# Backward Pass (Reverse-Mode Autodiff)
# =============================================================================

def backward(
    target: TensorCore,
    sources: Optional[List[TensorCore]] = None,
    grad_output: Optional[np.ndarray] = None
) -> Optional[List[np.ndarray]]:
    """
    Compute gradients via reverse-mode autodiff.
    
    This is geometry-aware:
    - Gradients are projected to tangent spaces
    - Parallel transport is applied when needed
    
    Parameters
    ----------
    target : TensorCore
        The output to differentiate (usually scalar loss)
    sources : Optional[List[TensorCore]]
        Specific sources to compute gradients for.
        If None, computes for all tensors with requires_grad=True.
    grad_output : Optional[np.ndarray]
        Initial gradient w.r.t. target. Defaults to 1 for scalars.
    
    Returns
    -------
    If sources provided: list of gradients
    If sources is None: None (gradients stored in .grad attributes)
    """
    
    # Initialize gradient
    if grad_output is None:
        if target.numel == 1:
            grad_output = np.ones(target.shape, dtype=np.float64)
        else:
            raise ValueError("grad_output must be specified for non-scalar outputs")
    
    # Build gradient dictionary: tensor_id -> accumulated gradient
    grads: Dict[int, np.ndarray] = {id(target): grad_output}
    
    # Topological sort (reverse order of computation)
    # For now, simple approach: follow grad_fn chain
    visited: Set[int] = set()
    order: List[Tuple[GradFn, TensorCore]] = []
    
    def visit(tensor: TensorCore):
        if id(tensor) in visited:
            return
        visited.add(id(tensor))
        
        if tensor.grad_fn is not None:
            for inp in tensor.grad_fn.inputs:
                visit(inp)
            order.append((tensor.grad_fn, tensor))
    
    visit(target)
    order.reverse()  # Reverse topological order
    
    # Backward pass
    for grad_fn, output in order:
        if id(output) not in grads:
            continue
        
        grad = grads[id(output)]
        
        # Apply backward function
        input_grads = grad_fn.apply(grad)
        
        # Accumulate gradients to inputs
        for inp, inp_grad in zip(grad_fn.inputs, input_grads):
            if inp_grad is None or not inp.requires_grad:
                continue
            
            # PROJECT TO TANGENT SPACE if input is on manifold
            if inp.manifold is not None:
                inp_grad = _project_gradient(inp, inp_grad)
            
            # Accumulate
            if id(inp) in grads:
                # PARALLEL TRANSPORT if needed
                existing = grads[id(inp)]
                inp_grad = _combine_gradients(inp, existing, inp_grad)
            
            grads[id(inp)] = inp_grad
    
    # Store gradients in .grad attributes
    for tensor_id, grad in grads.items():
        # Find tensor (ugly but works for now)
        for grad_fn, output in order:
            if id(output) == tensor_id:
                output.grad = _create_grad_tensor(output, grad)
                break
            for inp in grad_fn.inputs:
                if id(inp) == tensor_id:
                    inp.grad = _create_grad_tensor(inp, grad)
                    break
    
    # Return specific gradients if requested
    if sources is not None:
        return [grads.get(id(s)) for s in sources]
    
    return None


def _project_gradient(tensor: TensorCore, grad: np.ndarray) -> np.ndarray:
    """
    Project gradient to tangent space.
    
    Standard gradient is in ambient space.
    Riemannian gradient lives in tangent space.
    """
    manifold = tensor.manifold
    if manifold is None:
        return grad
    
    x = tensor.numpy()
    return manifold.project_tangent(x, grad)


def _combine_gradients(tensor: TensorCore, grad1: np.ndarray, grad2: np.ndarray) -> np.ndarray:
    """
    Combine two gradients at the same point.
    
    For Euclidean: simple addition
    For manifolds: both should be in tangent space, so addition works
    """
    return grad1 + grad2


def _create_grad_tensor(original: TensorCore, grad_data: np.ndarray) -> TensorCore:
    """
    Create gradient tensor with appropriate geometric type.
    """
    from ..core.storage import tensor
    
    grad = tensor(grad_data)
    
    if original.manifold is not None:
        grad.manifold = original.manifold
        grad.base_point = original
        grad.geometric_type = GeometricType.TANGENT
    
    return grad


# =============================================================================
# Backward Functions for Basic Operations
# =============================================================================

class AddBackward(GradFn):
    """Backward for addition: z = x + y"""
    
    def __init__(self, x: TensorCore, y: TensorCore):
        self._inputs = (x, y)
        self.ctx = SavedContext()
    
    @property
    def inputs(self) -> Tuple[TensorCore, ...]:
        return self._inputs
    
    def apply(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # ∂L/∂x = ∂L/∂z, ∂L/∂y = ∂L/∂z
        return grad_output, grad_output


class MulBackward(GradFn):
    """Backward for multiplication: z = x * y"""
    
    def __init__(self, x: TensorCore, y: TensorCore):
        self._inputs = (x, y)
        self.ctx = SavedContext()
        self.ctx.save_for_backward(x=x.numpy(), y=y.numpy())
    
    @property
    def inputs(self) -> Tuple[TensorCore, ...]:
        return self._inputs
    
    def apply(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = self.ctx.tensors['x']
        y = self.ctx.tensors['y']
        # ∂L/∂x = ∂L/∂z * y, ∂L/∂y = ∂L/∂z * x
        return grad_output * y, grad_output * x


class MatMulBackward(GradFn):
    """Backward for matrix multiplication: z = x @ y"""
    
    def __init__(self, x: TensorCore, y: TensorCore):
        self._inputs = (x, y)
        self.ctx = SavedContext()
        self.ctx.save_for_backward(x=x.numpy(), y=y.numpy())
    
    @property
    def inputs(self) -> Tuple[TensorCore, ...]:
        return self._inputs
    
    def apply(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = self.ctx.tensors['x']
        y = self.ctx.tensors['y']
        # z = x @ y
        # ∂L/∂x = ∂L/∂z @ y^T
        # ∂L/∂y = x^T @ ∂L/∂z
        grad_x = grad_output @ np.swapaxes(y, -2, -1)
        grad_y = np.swapaxes(x, -2, -1) @ grad_output
        return grad_x, grad_y


class SumBackward(GradFn):
    """Backward for sum: z = sum(x)"""
    
    def __init__(self, x: TensorCore, axis: Optional[int] = None):
        self._inputs = (x,)
        self.ctx = SavedContext()
        self.ctx.save_for_backward(shape=x.shape, axis=axis)
    
    @property
    def inputs(self) -> Tuple[TensorCore, ...]:
        return self._inputs
    
    def apply(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        shape = self.ctx.scalars['shape']
        # Gradient broadcasts back to input shape
        return (np.ones(shape) * grad_output,)


class ExpBackward(GradFn):
    """Backward for exp: z = exp(x)"""
    
    def __init__(self, x: TensorCore):
        self._inputs = (x,)
        self.ctx = SavedContext()
        self.ctx.save_for_backward(exp_x=np.exp(x.numpy()))
    
    @property
    def inputs(self) -> Tuple[TensorCore, ...]:
        return self._inputs
    
    def apply(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        exp_x = self.ctx.tensors['exp_x']
        # ∂exp(x)/∂x = exp(x)
        return (grad_output * exp_x,)


class LogBackward(GradFn):
    """Backward for log: z = log(x)"""
    
    def __init__(self, x: TensorCore):
        self._inputs = (x,)
        self.ctx = SavedContext()
        self.ctx.save_for_backward(x=x.numpy())
    
    @property
    def inputs(self) -> Tuple[TensorCore, ...]:
        return self._inputs
    
    def apply(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        x = self.ctx.tensors['x']
        # ∂log(x)/∂x = 1/x
        return (grad_output / x,)


# =============================================================================
# Backward Functions for Geometric Operations
# =============================================================================

class ManifoldExpBackward(GradFn):
    """
    Backward for manifold exponential map: y = exp_x(v)
    
    Given grad_y (gradient w.r.t. y at y), compute:
    - grad_x: gradient w.r.t. base point x (at x)
    - grad_v: gradient w.r.t. tangent vector v (at x)
    
    Key insight: grad_y is at y, but we need grads at x.
    Must parallel transport grad_y from y back to x!
    """
    
    def __init__(self, x: TensorCore, v: TensorCore, y: TensorCore, manifold: Manifold):
        self._inputs = (x, v)
        self.ctx = SavedContext()
        self.ctx.save_for_backward(
            x=x.numpy(),
            v=v.numpy(),
            y=y.numpy(),
            manifold=manifold
        )
    
    @property
    def inputs(self) -> Tuple[TensorCore, ...]:
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
    
    def __init__(self, x: TensorCore, y: TensorCore, manifold: Manifold):
        self._inputs = (x, y)
        self.ctx = SavedContext()
        self.ctx.save_for_backward(
            x=x.numpy(),
            y=y.numpy(),
            manifold=manifold
        )
    
    @property
    def inputs(self) -> Tuple[TensorCore, ...]:
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
    
    def __init__(self, x: TensorCore, y: TensorCore, manifold: Manifold):
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
    def inputs(self) -> Tuple[TensorCore, ...]:
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


# =============================================================================
# Gradient Checking (Numerical Verification)
# =============================================================================

def check_gradients(
    func: Callable,
    inputs: List[TensorCore],
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
    inputs : List[TensorCore]
        Input tensors
    eps : float
        Finite difference step size
    atol, rtol : float
        Absolute and relative tolerance
    
    Returns
    -------
    True if gradients match, raises AssertionError otherwise
    """
    
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
        
        for idx in np.ndindex(inp.shape):
            # Forward difference
            original = inp[idx]
            
            inp[idx] = original + eps
            f_plus = func(*inputs).numpy().item()
            
            inp[idx] = original - eps
            f_minus = func(*inputs).numpy().item()
            
            inp[idx] = original
            
            numerical_grad[idx] = (f_plus - f_minus) / (2 * eps)
        
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
    manifold: Manifold
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


# =============================================================================
# Test Function
# =============================================================================

def test_autograd():
    """Test autograd functionality."""
    print("=" * 60)
    print("Testing DavisTensor Autograd")
    print("=" * 60)
    
    from ..core.storage import tensor, randn
    from ..manifolds.base import Euclidean
    
    # Test 1: Simple addition
    print("\n1. Addition backward")
    x = randn(3, requires_grad=True)
    y = randn(3, requires_grad=True)
    
    # Manual forward
    z_data = x.numpy() + y.numpy()
    z = tensor(z_data, requires_grad=True)
    z.grad_fn = AddBackward(x, y)
    
    # Backward
    backward(z, grad_output=np.ones(3))
    
    print(f"   x.grad: {x.grad.numpy()}")
    print(f"   y.grad: {y.grad.numpy()}")
    assert np.allclose(x.grad.numpy(), np.ones(3))
    assert np.allclose(y.grad.numpy(), np.ones(3))
    print("   ✅ PASS")
    
    # Test 2: Multiplication backward
    print("\n2. Multiplication backward")
    x = tensor([2.0, 3.0], requires_grad=True)
    y = tensor([4.0, 5.0], requires_grad=True)
    
    z_data = x.numpy() * y.numpy()
    z = tensor(z_data, requires_grad=True)
    z.grad_fn = MulBackward(x, y)
    
    backward(z, grad_output=np.ones(2))
    
    print(f"   x.grad: {x.grad.numpy()} (expected [4, 5])")
    print(f"   y.grad: {y.grad.numpy()} (expected [2, 3])")
    assert np.allclose(x.grad.numpy(), [4.0, 5.0])
    assert np.allclose(y.grad.numpy(), [2.0, 3.0])
    print("   ✅ PASS")
    
    # Test 3: Chain rule (composition)
    print("\n3. Chain rule")
    x = tensor([1.0, 2.0], requires_grad=True)
    
    # y = x * x (square)
    y_data = x.numpy() * x.numpy()
    y = tensor(y_data, requires_grad=True)
    y.grad_fn = MulBackward(x, x)
    
    # z = sum(y)
    z_data = np.sum(y.numpy())
    z = tensor(z_data, requires_grad=True)
    z.grad_fn = SumBackward(y)
    
    backward(z)
    
    print(f"   x.grad: {x.grad.numpy()} (expected [2, 4] = 2x)")
    # d/dx sum(x^2) = 2x
    assert np.allclose(x.grad.numpy(), [2.0, 4.0])
    print("   ✅ PASS")
    
    # Test 4: Euclidean manifold gradients
    print("\n4. Euclidean manifold distance gradient")
    E = Euclidean(3)
    
    x = tensor([0.0, 0.0, 0.0], requires_grad=True)
    x.manifold = E
    y = tensor([3.0, 4.0, 0.0], requires_grad=True)
    y.manifold = E
    
    # d = ||y - x|| = 5
    d_val = E.distance(x.numpy(), y.numpy())
    d = tensor(d_val, requires_grad=True)
    d.grad_fn = ManifoldDistanceBackward(x, y, E)
    
    backward(d)
    
    # Gradient: unit vector from x to y
    expected_grad_x = -np.array([3.0, 4.0, 0.0]) / 5.0
    expected_grad_y = np.array([3.0, 4.0, 0.0]) / 5.0
    
    print(f"   x.grad: {x.grad.numpy()}")
    print(f"   Expected: {expected_grad_x}")
    print(f"   y.grad: {y.grad.numpy()}")
    print(f"   Expected: {expected_grad_y}")
    
    assert np.allclose(x.grad.numpy(), expected_grad_x, atol=1e-6)
    assert np.allclose(y.grad.numpy(), expected_grad_y, atol=1e-6)
    print("   ✅ PASS")
    
    print("\n" + "=" * 60)
    print("All autograd tests passed! ✅")
    print("=" * 60)


if __name__ == "__main__":
    test_autograd()
