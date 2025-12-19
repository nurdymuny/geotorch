"""
DavisTensor Autograd - Engine
==============================

Core autograd engine with gradient tape and backward pass.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Set
import numpy as np

from .grad_fn import GradFn


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
        self._operations: List[Tuple[GradFn, any]] = []
        self._watched: Set[int] = set()  # IDs of tensors to track
    
    def __enter__(self) -> 'GradientTape':
        GradientTape._default_tape = self
        return self
    
    def __exit__(self, *args):
        if not self.persistent:
            GradientTape._default_tape = None
    
    def watch(self, tensor):
        """Explicitly watch a tensor for gradients."""
        self._watched.add(id(tensor))
    
    def record(self, grad_fn: GradFn, output):
        """Record an operation."""
        self._operations.append((grad_fn, output))
        output.grad_fn = grad_fn
    
    @staticmethod
    def get_default() -> Optional['GradientTape']:
        return GradientTape._default_tape
    
    def gradient(
        self, 
        target, 
        sources: List,
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


def backward(
    target,
    sources: Optional[List] = None,
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
    from ..core.storage import tensor, GeometricType
    
    # Initialize gradient
    if grad_output is None:
        if target.numel == 1:
            grad_output = np.ones(target.shape, dtype=np.float64)
        else:
            raise ValueError("grad_output must be specified for non-scalar outputs")
    
    # Build gradient dictionary: tensor_id -> accumulated gradient
    grads: Dict[int, np.ndarray] = {id(target): grad_output}
    
    # Topological sort (reverse order of computation)
    visited: Set[int] = set()
    order: List[Tuple[GradFn, any]] = []
    
    def visit(t):
        if id(t) in visited:
            return
        visited.add(id(t))
        
        if hasattr(t, 'grad_fn') and t.grad_fn is not None:
            for inp in t.grad_fn.inputs:
                visit(inp)
            order.append((t.grad_fn, t))
    
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
            if inp_grad is None:
                continue
            if hasattr(inp, 'requires_grad') and not inp.requires_grad:
                continue
            
            # PROJECT TO TANGENT SPACE if input is on manifold
            if hasattr(inp, 'manifold') and inp.manifold is not None:
                inp_grad = _project_gradient(inp, inp_grad)
            
            # Accumulate
            if id(inp) in grads:
                # PARALLEL TRANSPORT if needed
                existing = grads[id(inp)]
                inp_grad = _combine_gradients(inp, existing, inp_grad)
            
            grads[id(inp)] = inp_grad
    
    # Store gradients in .grad attributes
    for grad_fn, output in order:
        if id(output) in grads:
            output.grad = _create_grad_tensor(output, grads[id(output)])
        for inp in grad_fn.inputs:
            if id(inp) in grads:
                inp.grad = _create_grad_tensor(inp, grads[id(inp)])
    
    # Return specific gradients if requested
    if sources is not None:
        return [grads.get(id(s)) for s in sources]
    
    return None


def _project_gradient(tensor, grad: np.ndarray) -> np.ndarray:
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


def _combine_gradients(tensor, grad1: np.ndarray, grad2: np.ndarray) -> np.ndarray:
    """
    Combine two gradients at the same point.
    
    For Euclidean: simple addition
    For manifolds: both should be in tangent space, so addition works
    """
    return grad1 + grad2


def _create_grad_tensor(original, grad_data: np.ndarray):
    """
    Create gradient tensor with appropriate geometric type.
    """
    from ..core.storage import tensor, GeometricType
    
    grad = tensor(grad_data)
    
    if hasattr(original, 'manifold') and original.manifold is not None:
        grad.manifold = original.manifold
        grad.base_point = original
        grad.geometric_type = GeometricType.TANGENT
    
    return grad


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
    from .grad_fn import AddBackward, MulBackward, SumBackward
    from .geometric_grad import ManifoldDistanceBackward
    
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
    
    # Test 5: Gradient tape context manager
    print("\n5. GradientTape context manager")
    
    with GradientTape() as tape:
        x = tensor([1.0, 2.0, 3.0], requires_grad=True)
        tape.watch(x)
    
    assert GradientTape._default_tape is None  # Should be cleared
    print("   Context manager works correctly")
    print("   ✅ PASS")
    
    print("\n" + "=" * 60)
    print("All autograd tests passed! ✅")
    print("=" * 60)


if __name__ == "__main__":
    test_autograd()
