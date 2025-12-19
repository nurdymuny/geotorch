"""
DavisTensor Type System: ManifoldTensor, TangentTensor, Scalar
===============================================================

Type-safe wrappers that ensure geometric correctness.

- ManifoldTensor: A point on a manifold
- TangentTensor: A tangent vector at a specific point  
- Scalar: Manifold-independent scalar value

The key insight: Tangent vectors at DIFFERENT points cannot be
directly added - they must first be parallel transported.
"""

from __future__ import annotations
from typing import Union, Optional, TYPE_CHECKING, List, Tuple
import numpy as np

from .core.storage import (
    TensorCore, Storage, GeometricType, DType, Device, CPU,
    float32, randn as _randn, zeros as _zeros, ones as _ones, tensor as _tensor
)
from .manifolds.base import Manifold, Euclidean


# =============================================================================
# Scalar: Manifold-independent value
# =============================================================================

class Scalar:
    """
    A scalar value that doesn't live on any manifold.
    
    Results of distance(), norm(), inner() are Scalars.
    Can be freely combined with standard arithmetic.
    """
    
    def __init__(
        self, 
        value: Union[float, np.ndarray, TensorCore],
        requires_grad: bool = False,
        grad_fn: Optional[object] = None
    ):
        if isinstance(value, TensorCore):
            self._data = value
        elif isinstance(value, np.ndarray):
            self._data = _tensor(value, requires_grad=requires_grad)
        else:
            self._data = _tensor(float(value), requires_grad=requires_grad)
        
        self._data.grad_fn = grad_fn
        self._data.geometric_type = GeometricType.SCALAR
    
    @property
    def data(self) -> TensorCore:
        return self._data
    
    @property
    def requires_grad(self) -> bool:
        return self._data.requires_grad
    
    @property
    def grad(self) -> Optional['Scalar']:
        if self._data.grad is not None:
            return Scalar(self._data.grad)
        return None
    
    def item(self) -> float:
        """Convert to Python float."""
        return float(self._data.numpy().flat[0])
    
    def numpy(self) -> np.ndarray:
        return self._data.numpy()
    
    # Arithmetic operations
    def __add__(self, other: Union['Scalar', float]) -> 'Scalar':
        if isinstance(other, Scalar):
            other_val = other.numpy()
        else:
            other_val = other
        return Scalar(self.numpy() + other_val)
    
    def __radd__(self, other: float) -> 'Scalar':
        return self.__add__(other)
    
    def __sub__(self, other: Union['Scalar', float]) -> 'Scalar':
        if isinstance(other, Scalar):
            other_val = other.numpy()
        else:
            other_val = other
        return Scalar(self.numpy() - other_val)
    
    def __rsub__(self, other: float) -> 'Scalar':
        return Scalar(other - self.numpy())
    
    def __mul__(self, other: Union['Scalar', float]) -> 'Scalar':
        if isinstance(other, Scalar):
            other_val = other.numpy()
        else:
            other_val = other
        return Scalar(self.numpy() * other_val)
    
    def __rmul__(self, other: float) -> 'Scalar':
        return self.__mul__(other)
    
    def __truediv__(self, other: Union['Scalar', float]) -> 'Scalar':
        if isinstance(other, Scalar):
            other_val = other.numpy()
        else:
            other_val = other
        return Scalar(self.numpy() / other_val)
    
    def __neg__(self) -> 'Scalar':
        return Scalar(-self.numpy())
    
    def __pow__(self, exp: float) -> 'Scalar':
        return Scalar(self.numpy() ** exp)
    
    def sqrt(self) -> 'Scalar':
        return Scalar(np.sqrt(self.numpy()))
    
    def __repr__(self) -> str:
        return f"Scalar({self.item():.6g})"
    
    def __float__(self) -> float:
        return self.item()
    
    def backward(self):
        """Trigger backward pass (placeholder for now)."""
        # TODO: implement autograd
        pass


# =============================================================================
# TangentTensor: Vector in tangent space at a point
# =============================================================================

class TangentTensor:
    """
    A tangent vector at a specific point on a manifold.
    
    Key property: Tangent vectors can only be added/scaled at the SAME point.
    To move a tangent vector to another point, use parallel_transport.
    
    This prevents the common bug of adding gradients at different points
    without transporting them first.
    """
    
    def __init__(
        self,
        data: Union[np.ndarray, TensorCore],
        base_point: 'ManifoldTensor',
        requires_grad: bool = False
    ):
        """
        Create a tangent vector at base_point.
        
        Args:
            data: The vector data (must be in tangent space!)
            base_point: The ManifoldTensor where this tangent lives
            requires_grad: Whether to track gradients
        """
        self._base_point = base_point
        self._manifold = base_point.manifold
        
        # Convert data to TensorCore
        if isinstance(data, TensorCore):
            self._data = data
        else:
            self._data = _tensor(data, requires_grad=requires_grad)
        
        self._data.manifold = self._manifold
        self._data.base_point = base_point._data
        self._data.geometric_type = GeometricType.TANGENT
        
        # Verify it's in tangent space (optional, can be slow)
        # if not self._manifold.check_tangent(base_point.numpy(), self.numpy()):
        #     raise ValueError("Data is not in tangent space at base_point")
    
    @property
    def manifold(self) -> Manifold:
        return self._manifold
    
    @property
    def base_point(self) -> 'ManifoldTensor':
        return self._base_point
    
    @property
    def data(self) -> TensorCore:
        return self._data
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self._data.shape
    
    @property
    def requires_grad(self) -> bool:
        return self._data.requires_grad
    
    def numpy(self) -> np.ndarray:
        return self._data.numpy()
    
    def clone(self) -> 'TangentTensor':
        """Create a copy."""
        return TangentTensor(
            self._data.clone(),
            self._base_point,
            self.requires_grad
        )
    
    # =========================================================================
    # Vector space operations (only at same base point!)
    # =========================================================================
    
    def _check_same_base(self, other: 'TangentTensor') -> None:
        """Verify two tangent vectors have the same base point."""
        # Check if same manifold
        if self._manifold != other._manifold:
            raise TypeError(
                f"Cannot combine tangent vectors from different manifolds: "
                f"{self._manifold} vs {other._manifold}"
            )
        
        # Check if same base point (approximately)
        if not np.allclose(self._base_point.numpy(), other._base_point.numpy(), atol=1e-6):
            raise TypeError(
                "Cannot add tangent vectors at different points! "
                "Use .transport_to(target) first to parallel transport."
            )
    
    def __add__(self, other: 'TangentTensor') -> 'TangentTensor':
        """Add tangent vectors at the same point."""
        self._check_same_base(other)
        return TangentTensor(
            self.numpy() + other.numpy(),
            self._base_point,
            self.requires_grad or other.requires_grad
        )
    
    def __sub__(self, other: 'TangentTensor') -> 'TangentTensor':
        """Subtract tangent vectors at the same point."""
        self._check_same_base(other)
        return TangentTensor(
            self.numpy() - other.numpy(),
            self._base_point,
            self.requires_grad or other.requires_grad
        )
    
    def __mul__(self, scalar: Union[float, Scalar]) -> 'TangentTensor':
        """Scalar multiplication."""
        if isinstance(scalar, Scalar):
            scalar = scalar.item()
        return TangentTensor(
            self.numpy() * scalar,
            self._base_point,
            self.requires_grad
        )
    
    def __rmul__(self, scalar: Union[float, Scalar]) -> 'TangentTensor':
        return self.__mul__(scalar)
    
    def __neg__(self) -> 'TangentTensor':
        return TangentTensor(-self.numpy(), self._base_point, self.requires_grad)
    
    def __truediv__(self, scalar: Union[float, Scalar]) -> 'TangentTensor':
        if isinstance(scalar, Scalar):
            scalar = scalar.item()
        return TangentTensor(
            self.numpy() / scalar,
            self._base_point,
            self.requires_grad
        )
    
    # =========================================================================
    # Metric operations
    # =========================================================================
    
    def inner(self, other: 'TangentTensor') -> Scalar:
        """
        Inner product ⟨self, other⟩_x using Riemannian metric.
        """
        self._check_same_base(other)
        x = self._base_point.numpy()
        ip = self._manifold.inner(x, self.numpy(), other.numpy())
        return Scalar(ip)
    
    def norm(self) -> Scalar:
        """Riemannian norm ||self||_x."""
        x = self._base_point.numpy()
        n = self._manifold.norm(x, self.numpy())
        return Scalar(n)
    
    def normalize(self) -> 'TangentTensor':
        """Return unit tangent vector."""
        n = self.norm().item()
        if n < 1e-8:
            return self.clone()
        return self / n
    
    # =========================================================================
    # Parallel transport
    # =========================================================================
    
    def transport_to(self, target: 'ManifoldTensor') -> 'TangentTensor':
        """
        Parallel transport this vector to target point.
        
        Args:
            target: The ManifoldTensor to transport to
        
        Returns:
            TangentTensor at target
        """
        if self._manifold != target.manifold:
            raise TypeError("Cannot transport between different manifolds")
        
        transported = self._manifold.parallel_transport(
            self._base_point.numpy(),
            target.numpy(),
            self.numpy()
        )
        return TangentTensor(transported, target, self.requires_grad)
    
    def __repr__(self) -> str:
        data_str = np.array2string(self.numpy(), precision=4, suppress_small=True)
        return f"TangentTensor({data_str}, at={self._base_point.manifold})"


# =============================================================================
# ManifoldTensor: Point on a manifold
# =============================================================================

class ManifoldTensor:
    """
    A tensor that lives on a Riemannian manifold.
    
    Unlike raw tensors, ManifoldTensor:
    - Knows which manifold it belongs to
    - Provides geometric operations (exp, log, distance)
    - Arithmetic is geometric (+ means exp, - means log)
    - Gradients are tangent vectors
    """
    
    def __init__(
        self,
        data: Union[np.ndarray, TensorCore],
        manifold: Manifold,
        requires_grad: bool = False,
        _skip_project: bool = False
    ):
        """
        Create a ManifoldTensor.
        
        Args:
            data: Point data
            manifold: The Riemannian manifold
            requires_grad: Whether to track gradients
            _skip_project: Internal flag to skip projection (use with care)
        """
        self._manifold = manifold
        
        # Convert to TensorCore
        if isinstance(data, TensorCore):
            arr = data.numpy()
        else:
            arr = np.asarray(data)
        
        # Project onto manifold (unless skipped)
        if not _skip_project:
            arr = manifold.project_point(arr)
        
        self._data = _tensor(arr, requires_grad=requires_grad)
        self._data.manifold = manifold
        self._data.geometric_type = GeometricType.MANIFOLD_POINT
    
    @property
    def manifold(self) -> Manifold:
        return self._manifold
    
    @property
    def data(self) -> TensorCore:
        return self._data
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self._data.shape
    
    @property
    def requires_grad(self) -> bool:
        return self._data.requires_grad
    
    @requires_grad.setter
    def requires_grad(self, value: bool):
        self._data.requires_grad = value
    
    @property
    def grad(self) -> Optional[TangentTensor]:
        """Gradient as a tangent vector."""
        if self._data.grad is not None:
            return TangentTensor(self._data.grad, self)
        return None
    
    def numpy(self) -> np.ndarray:
        return self._data.numpy()
    
    def clone(self) -> 'ManifoldTensor':
        """Create a copy."""
        return ManifoldTensor(
            self._data.clone(),
            self._manifold,
            self.requires_grad,
            _skip_project=True  # Already on manifold
        )
    
    # =========================================================================
    # Geometric operations
    # =========================================================================
    
    def exp(self, v: TangentTensor) -> 'ManifoldTensor':
        """
        Exponential map: move from self in direction v.
        
        Args:
            v: Tangent vector at self
        
        Returns:
            New point on manifold
        """
        # Verify v is at self
        if not np.allclose(v.base_point.numpy(), self.numpy(), atol=1e-6):
            raise ValueError("Tangent vector must be at this point")
        
        result = self._manifold.exp(self.numpy(), v.numpy())
        return ManifoldTensor(result, self._manifold, self.requires_grad, _skip_project=True)
    
    def log(self, other: 'ManifoldTensor') -> TangentTensor:
        """
        Logarithm map: tangent vector pointing from self to other.
        
        Args:
            other: Target point on same manifold
        
        Returns:
            Tangent vector at self
        """
        if self._manifold != other._manifold:
            raise TypeError("Points must be on the same manifold")
        
        result = self._manifold.log(self.numpy(), other.numpy())
        return TangentTensor(result, self, self.requires_grad or other.requires_grad)
    
    def distance(self, other: 'ManifoldTensor') -> Scalar:
        """
        Geodesic distance to other point.
        """
        if self._manifold != other._manifold:
            raise TypeError("Points must be on the same manifold")
        
        d = self._manifold.distance(self.numpy(), other.numpy())
        return Scalar(d)
    
    def geodesic(self, other: 'ManifoldTensor', t: float) -> 'ManifoldTensor':
        """
        Point at fraction t along geodesic from self to other.
        
        Args:
            other: Target point
            t: Fraction in [0, 1]. t=0 gives self, t=1 gives other.
        
        Returns:
            Point on geodesic
        """
        if self._manifold != other._manifold:
            raise TypeError("Points must be on the same manifold")
        
        result = self._manifold.geodesic(self.numpy(), other.numpy(), t)
        return ManifoldTensor(result, self._manifold, self.requires_grad or other.requires_grad, _skip_project=True)
    
    def random_tangent(self) -> TangentTensor:
        """Sample random tangent vector at this point."""
        v = self._manifold.random_tangent(self.numpy())
        return TangentTensor(v, self)
    
    def zero_tangent(self) -> TangentTensor:
        """Zero tangent vector at this point."""
        v = self._manifold.zero_tangent(self.numpy())
        return TangentTensor(v, self)
    
    # =========================================================================
    # Arithmetic (geometric interpretation)
    # =========================================================================
    
    def __add__(self, v: TangentTensor) -> 'ManifoldTensor':
        """
        Point + TangentVector = exp(point, vector)
        
        Move from self in direction v.
        """
        if not isinstance(v, TangentTensor):
            raise TypeError(
                f"Can only add TangentTensor to ManifoldTensor, got {type(v)}. "
                "Use .exp(v) for explicit exponential map."
            )
        return self.exp(v)
    
    def __sub__(self, other: 'ManifoldTensor') -> TangentTensor:
        """
        Point - Point = log(self, other)
        
        Returns tangent vector at self pointing to other.
        """
        if not isinstance(other, ManifoldTensor):
            raise TypeError(
                f"Can only subtract ManifoldTensor from ManifoldTensor, got {type(other)}. "
                "Use .log(other) for explicit logarithm map."
            )
        return self.log(other)
    
    # =========================================================================
    # Comparison
    # =========================================================================
    
    def __eq__(self, other: 'ManifoldTensor') -> bool:
        if not isinstance(other, ManifoldTensor):
            return False
        if self._manifold != other._manifold:
            return False
        return np.allclose(self.numpy(), other.numpy())
    
    # =========================================================================
    # String representation
    # =========================================================================
    
    def __repr__(self) -> str:
        data_str = np.array2string(self.numpy(), precision=4, suppress_small=True)
        parts = [f"ManifoldTensor({data_str}"]
        parts.append(f", manifold={self._manifold}")
        if self.requires_grad:
            parts.append(", requires_grad=True")
        parts.append(")")
        return "".join(parts)


# =============================================================================
# Factory functions
# =============================================================================

def randn(*shape: int, manifold: Manifold, requires_grad: bool = False) -> ManifoldTensor:
    """
    Create random point on manifold.
    
    Args:
        shape: Batch dimensions (final dim is manifold.ambient_dim)
        manifold: The Riemannian manifold
        requires_grad: Whether to track gradients
    
    Returns:
        ManifoldTensor with random point(s)
    """
    data = manifold.random_point(*shape) if shape else manifold.random_point()
    return ManifoldTensor(data, manifold, requires_grad, _skip_project=True)


def origin(manifold: Manifold, *shape: int, requires_grad: bool = False) -> ManifoldTensor:
    """
    Create origin/identity point on manifold.
    """
    data = manifold.origin(*shape) if shape else manifold.origin()
    return ManifoldTensor(data, manifold, requires_grad, _skip_project=True)


def tangent_randn(base: ManifoldTensor, requires_grad: bool = False) -> TangentTensor:
    """
    Create random tangent vector at base point.
    """
    v = base.manifold.random_tangent(base.numpy())
    return TangentTensor(v, base, requires_grad)


def tangent_zeros(base: ManifoldTensor, requires_grad: bool = False) -> TangentTensor:
    """
    Create zero tangent vector at base point.
    """
    v = base.manifold.zero_tangent(base.numpy())
    return TangentTensor(v, base, requires_grad)


# =============================================================================
# Tests
# =============================================================================

def test_type_system():
    """Test the type-safe tensor wrappers."""
    print("=" * 60)
    print("Testing DavisTensor Type System")
    print("=" * 60)
    
    E = Euclidean(5)
    
    # Test ManifoldTensor creation
    print("\n1. ManifoldTensor creation")
    x = randn(manifold=E)
    print(f"   Random point: shape={x.shape}")
    print(f"   {x}")
    assert x.manifold == E
    print("   ✅ PASS")
    
    # Test origin
    print("\n2. Origin")
    o = origin(E)
    print(f"   Origin: {o.numpy()}")
    assert np.allclose(o.numpy(), 0)
    print("   ✅ PASS")
    
    # Test TangentTensor
    print("\n3. TangentTensor creation")
    v = tangent_randn(x)
    print(f"   Tangent at x: {v}")
    assert v.base_point == x
    assert v.manifold == E
    print("   ✅ PASS")
    
    # Test tangent addition (same base)
    print("\n4. Tangent addition (same base point)")
    v1 = tangent_randn(x)
    v2 = tangent_randn(x)
    v3 = v1 + v2
    print(f"   v1 + v2 at same point: OK")
    assert np.allclose(v3.numpy(), v1.numpy() + v2.numpy())
    print("   ✅ PASS")
    
    # Test tangent addition error (different base)
    print("\n5. Tangent addition error (different base points)")
    x1 = randn(manifold=E)
    x2 = randn(manifold=E)
    v1 = tangent_randn(x1)
    v2 = tangent_randn(x2)
    try:
        v3 = v1 + v2
        print("   ERROR: Should have raised TypeError!")
        assert False
    except TypeError as e:
        print(f"   Correctly raised: {type(e).__name__}")
        print(f"   Message: {str(e)[:60]}...")
    print("   ✅ PASS")
    
    # Test exp map (point + tangent)
    print("\n6. Exponential map (point + tangent)")
    x = randn(manifold=E)
    v = tangent_randn(x)
    y = x + v  # exp(x, v)
    y_explicit = x.exp(v)
    assert np.allclose(y.numpy(), y_explicit.numpy())
    print(f"   x + v = exp(x, v): OK")
    print("   ✅ PASS")
    
    # Test log map (point - point)
    print("\n7. Logarithm map (point - point)")
    x = randn(manifold=E)
    y = randn(manifold=E)
    v = x.log(y)  # Tangent at x pointing to y
    v_sub = x - y  # Wait, this should be log(x, y) which points FROM x TO y
    # Actually in our definition x - y = x.log(y)
    print(f"   x.log(y) is TangentTensor at x: {type(v).__name__}")
    assert isinstance(v, TangentTensor)
    assert v.base_point == x
    print("   ✅ PASS")
    
    # Test distance
    print("\n8. Distance")
    x = origin(E)
    y_data = np.array([3., 4., 0., 0., 0.])
    y = ManifoldTensor(y_data, E)
    d = x.distance(y)
    print(f"   d(origin, [3,4,0,0,0]) = {d.item()}")
    assert abs(d.item() - 5.0) < 1e-6
    print("   ✅ PASS")
    
    # Test geodesic
    print("\n9. Geodesic")
    x = origin(E)
    y = ManifoldTensor(np.ones(5) * 2, E)
    mid = x.geodesic(y, 0.5)
    print(f"   Midpoint: {mid.numpy()}")
    assert np.allclose(mid.numpy(), np.ones(5))
    print("   ✅ PASS")
    
    # Test Scalar
    print("\n10. Scalar operations")
    s1 = Scalar(3.0)
    s2 = Scalar(4.0)
    s3 = s1 + s2
    s4 = s1 * s2
    print(f"   3 + 4 = {s3.item()}")
    print(f"   3 * 4 = {s4.item()}")
    assert s3.item() == 7.0
    assert s4.item() == 12.0
    print("   ✅ PASS")
    
    # Test tangent norm
    print("\n11. Tangent norm and inner product")
    x = randn(manifold=E)
    v = TangentTensor(np.array([1., 0., 0., 0., 0.]), x)
    n = v.norm()
    print(f"   ||[1,0,0,0,0]|| = {n.item()}")
    assert abs(n.item() - 1.0) < 1e-6
    print("   ✅ PASS")
    
    # Test parallel transport
    print("\n12. Parallel transport")
    x = randn(manifold=E)
    y = randn(manifold=E)
    v = tangent_randn(x)
    v_at_y = v.transport_to(y)
    print(f"   Transported from x to y")
    assert v_at_y.base_point == y
    # For Euclidean, transport is identity
    assert np.allclose(v.numpy(), v_at_y.numpy())
    print("   ✅ PASS")
    
    # Test exp/log inverse
    print("\n13. Exp/Log inverse property")
    x = randn(manifold=E)
    y = randn(manifold=E)
    v = x.log(y)
    y_recovered = x.exp(v)
    error = y.distance(y_recovered).item()  # Compare y and y_recovered!
    print(f"   exp(x, log(x, y)) ≈ y, error = {error:.2e}")
    assert error < 1e-10
    print("   ✅ PASS")
    
    # Test batched ManifoldTensor
    print("\n14. Batched operations")
    X = randn(32, manifold=E)
    Y = randn(32, manifold=E)
    D = X.distance(Y)
    print(f"   Distance shape: {D.numpy().shape}")
    assert D.numpy().shape == (32,)
    print("   ✅ PASS")
    
    print("\n" + "=" * 60)
    print("All type system tests passed! ✅")
    print("=" * 60)


if __name__ == "__main__":
    test_type_system()
