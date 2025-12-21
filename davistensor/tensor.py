"""Type-safe tensor wrappers for DavisTensor."""

from typing import Optional, Union
import numpy as np

from .core.storage import TensorCore, DType, Device, GeometricType
from .core import zeros as core_zeros, randn as core_randn
from .manifolds.base import Manifold


class Scalar:
    """Manifold-independent scalar value."""
    
    def __init__(self, data: Union[float, TensorCore]):
        """Initialize scalar.
        
        Args:
            data: Scalar value or TensorCore
        """
        if isinstance(data, (int, float)):
            self._core = core_zeros(dtype=DType.FLOAT64)
            self._core.storage.data.flat[0] = float(data)
        elif isinstance(data, TensorCore):
            if data.size != 1:
                raise ValueError("Scalar must have exactly one element")
            self._core = data
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
        
        self._core.geometric_type = GeometricType.SCALAR
    
    def item(self) -> float:
        """Convert to Python float."""
        return float(self._core.data.flat[0])
    
    def numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return self._core.numpy()
    
    def sqrt(self) -> 'Scalar':
        """Square root."""
        result = np.sqrt(self._core.numpy())
        return Scalar(core_zeros(dtype=self._core.dtype).storage.data.flat[0] * 0 + result.flat[0])
    
    def backward(self):
        """Backward pass for autograd."""
        raise NotImplementedError("Autograd not yet implemented")
    
    # Arithmetic operations
    
    def __add__(self, other: Union['Scalar', float]) -> 'Scalar':
        """Add scalars."""
        if isinstance(other, (int, float)):
            other = Scalar(other)
        if not isinstance(other, Scalar):
            return NotImplemented
        result = self.item() + other.item()
        return Scalar(result)
    
    def __radd__(self, other: Union[float, int]) -> 'Scalar':
        """Right add."""
        return self.__add__(other)
    
    def __sub__(self, other: Union['Scalar', float]) -> 'Scalar':
        """Subtract scalars."""
        if isinstance(other, (int, float)):
            other = Scalar(other)
        if not isinstance(other, Scalar):
            return NotImplemented
        result = self.item() - other.item()
        return Scalar(result)
    
    def __rsub__(self, other: Union[float, int]) -> 'Scalar':
        """Right subtract."""
        return Scalar(other).__sub__(self)
    
    def __mul__(self, other: Union['Scalar', float]) -> 'Scalar':
        """Multiply scalars."""
        if isinstance(other, (int, float)):
            other = Scalar(other)
        if not isinstance(other, Scalar):
            return NotImplemented
        result = self.item() * other.item()
        return Scalar(result)
    
    def __rmul__(self, other: Union[float, int]) -> 'Scalar':
        """Right multiply."""
        return self.__mul__(other)
    
    def __truediv__(self, other: Union['Scalar', float]) -> 'Scalar':
        """Divide scalars."""
        if isinstance(other, (int, float)):
            other = Scalar(other)
        if not isinstance(other, Scalar):
            return NotImplemented
        result = self.item() / other.item()
        return Scalar(result)
    
    def __rtruediv__(self, other: Union[float, int]) -> 'Scalar':
        """Right divide."""
        return Scalar(other).__truediv__(self)
    
    def __pow__(self, other: Union['Scalar', float]) -> 'Scalar':
        """Power."""
        if isinstance(other, (int, float)):
            other = Scalar(other)
        if not isinstance(other, Scalar):
            return NotImplemented
        result = self.item() ** other.item()
        return Scalar(result)
    
    def __neg__(self) -> 'Scalar':
        """Negate."""
        return Scalar(-self.item())
    
    def __repr__(self):
        return f'Scalar({self.item()})'


class TangentTensor:
    """Vector in tangent space at a specific point on a manifold."""
    
    def __init__(
        self,
        data: TensorCore,
        base_point: 'ManifoldTensor',
        manifold: Optional[Manifold] = None,
    ):
        """Initialize tangent tensor.
        
        Args:
            data: Underlying TensorCore
            base_point: The point where this tangent vector lives
            manifold: The manifold (inferred from base_point if None)
        """
        self._core = data
        self._base_point = base_point
        self._manifold = manifold or base_point.manifold
        
        # Update core metadata
        self._core.manifold = self._manifold
        self._core.base_point = base_point._core
        self._core.geometric_type = GeometricType.TANGENT
    
    @property
    def base_point(self) -> 'ManifoldTensor':
        """The point where this tangent vector lives."""
        return self._base_point
    
    @property
    def manifold(self) -> Manifold:
        """The underlying manifold."""
        return self._manifold
    
    @property
    def shape(self):
        """Shape of the tensor."""
        return self._core.shape
    
    def numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return self._core.numpy()
    
    # Type-safe vector operations
    
    def __add__(self, other: 'TangentTensor') -> 'TangentTensor':
        """Add tangent vectors at the same point.
        
        Raises TypeError if base points differ.
        """
        if not isinstance(other, TangentTensor):
            return NotImplemented
        
        # Check that base points are the same
        if not np.allclose(self._base_point.numpy(), other._base_point.numpy()):
            raise TypeError(
                "Cannot add tangent vectors at different points. "
                "Use transport_to() to move one vector to the other's base point first."
            )
        
        # Add the data
        result_data = self._core.numpy() + other._core.numpy()
        result_core = TensorCore(
            storage=self._core.storage.__class__(result_data, self._core.device),
            shape=result_data.shape,
            dtype=self._core.dtype,
            device=self._core.device,
        )
        
        return TangentTensor(result_core, self._base_point, self._manifold)
    
    def __mul__(self, scalar: Union[float, Scalar]) -> 'TangentTensor':
        """Scalar multiplication."""
        if isinstance(scalar, Scalar):
            scalar = scalar.item()
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        
        result_data = scalar * self._core.numpy()
        result_core = TensorCore(
            storage=self._core.storage.__class__(result_data, self._core.device),
            shape=result_data.shape,
            dtype=self._core.dtype,
            device=self._core.device,
        )
        
        return TangentTensor(result_core, self._base_point, self._manifold)
    
    def __rmul__(self, scalar: Union[float, Scalar]) -> 'TangentTensor':
        """Right scalar multiplication."""
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar: Union[float, Scalar]) -> 'TangentTensor':
        """Scalar division."""
        if isinstance(scalar, Scalar):
            scalar = scalar.item()
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        
        return self.__mul__(1.0 / scalar)
    
    def __neg__(self) -> 'TangentTensor':
        """Negate tangent vector."""
        return self.__mul__(-1.0)
    
    def __sub__(self, other: 'TangentTensor') -> 'TangentTensor':
        """Subtract tangent vectors."""
        return self.__add__(other.__neg__())
    
    def inner(self, other: 'TangentTensor') -> Scalar:
        """Inner product with another tangent vector at the same point."""
        if not isinstance(other, TangentTensor):
            raise TypeError("Can only compute inner product with another TangentTensor")
        
        if not np.allclose(self._base_point.numpy(), other._base_point.numpy()):
            raise TypeError(
                "Cannot compute inner product of tangent vectors at different points"
            )
        
        result = self._manifold.inner(
            self._base_point.numpy(),
            self.numpy(),
            other.numpy()
        )
        return Scalar(float(result))
    
    def norm(self) -> Scalar:
        """Norm of this tangent vector."""
        result = self._manifold.norm(self._base_point.numpy(), self.numpy())
        return Scalar(float(result))
    
    def normalize(self) -> 'TangentTensor':
        """Return normalized version of this tangent vector."""
        n = self.norm().item()
        if n < 1e-10:
            raise ValueError("Cannot normalize zero vector")
        return self / n
    
    def transport_to(self, target: 'ManifoldTensor') -> 'TangentTensor':
        """Parallel transport this vector to another point.
        
        Args:
            target: Target point on the manifold
            
        Returns:
            Tangent vector at target point
        """
        if target.manifold is not self._manifold:
            raise ValueError("Cannot transport between different manifolds")
        
        transported = self._manifold.parallel_transport(
            self._base_point.numpy(),
            target.numpy(),
            self.numpy()
        )
        
        transported_core = TensorCore(
            storage=self._core.storage.__class__(transported, self._core.device),
            shape=transported.shape,
            dtype=self._core.dtype,
            device=self._core.device,
        )
        
        return TangentTensor(transported_core, target, self._manifold)
    
    def __repr__(self):
        return f'TangentTensor(shape={self.shape}, manifold={self._manifold})'


class ManifoldTensor:
    """Point on a Riemannian manifold."""
    
    def __init__(
        self,
        data: TensorCore,
        manifold: Manifold,
        requires_grad: bool = False,
    ):
        """Initialize manifold tensor.
        
        Args:
            data: Underlying TensorCore
            manifold: The manifold this tensor lives on
            requires_grad: Whether to track gradients
        """
        self._core = data
        self._manifold = manifold
        
        # Update core metadata
        self._core.manifold = manifold
        self._core.geometric_type = GeometricType.MANIFOLD_POINT
        self._core.requires_grad = requires_grad
    
    @property
    def manifold(self) -> Manifold:
        """The manifold this tensor lives on."""
        return self._manifold
    
    @property
    def shape(self):
        """Shape of the tensor."""
        return self._core.shape
    
    @property
    def requires_grad(self) -> bool:
        """Whether this tensor requires gradients."""
        return self._core.requires_grad
    
    @property
    def grad(self) -> Optional[TangentTensor]:
        """Gradient (tangent vector at this point)."""
        if self._core.grad is None:
            return None
        return TangentTensor(self._core.grad, self, self._manifold)
    
    def numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return self._core.numpy()
    
    def clone(self) -> 'ManifoldTensor':
        """Create a deep copy."""
        return ManifoldTensor(
            self._core.clone(),
            self._manifold,
            self.requires_grad,
        )
    
    # Geometric operations
    
    def exp(self, v: TangentTensor) -> 'ManifoldTensor':
        """Exponential map: move from self in direction v.
        
        Args:
            v: Tangent vector at self
            
        Returns:
            New point on manifold
        """
        if not isinstance(v, TangentTensor):
            raise TypeError("exp() requires a TangentTensor")
        
        if not np.allclose(v.base_point.numpy(), self.numpy()):
            raise ValueError("Tangent vector must be at this point")
        
        result = self._manifold.exp(self.numpy(), v.numpy())
        result_core = TensorCore(
            storage=self._core.storage.__class__(result, self._core.device),
            shape=result.shape,
            dtype=self._core.dtype,
            device=self._core.device,
        )
        
        return ManifoldTensor(result_core, self._manifold, self.requires_grad)
    
    def log(self, other: 'ManifoldTensor') -> TangentTensor:
        """Logarithm map: tangent vector from self to other.
        
        Args:
            other: Target point on manifold
            
        Returns:
            Tangent vector at self pointing toward other
        """
        if not isinstance(other, ManifoldTensor):
            raise TypeError("log() requires a ManifoldTensor")
        
        if other.manifold is not self._manifold:
            raise ValueError("Points must be on the same manifold")
        
        result = self._manifold.log(self.numpy(), other.numpy())
        result_core = TensorCore(
            storage=self._core.storage.__class__(result, self._core.device),
            shape=result.shape,
            dtype=self._core.dtype,
            device=self._core.device,
        )
        
        return TangentTensor(result_core, self, self._manifold)
    
    def distance(self, other: 'ManifoldTensor') -> Scalar:
        """Geodesic distance to another point.
        
        Args:
            other: Target point on manifold
            
        Returns:
            Geodesic distance
        """
        if not isinstance(other, ManifoldTensor):
            raise TypeError("distance() requires a ManifoldTensor")
        
        if other.manifold is not self._manifold:
            raise ValueError("Points must be on the same manifold")
        
        result = self._manifold.distance(self.numpy(), other.numpy())
        return Scalar(float(result))
    
    def geodesic(self, other: 'ManifoldTensor', t: Union[float, Scalar]) -> 'ManifoldTensor':
        """Point at fraction t along geodesic from self to other.
        
        Args:
            other: Target point on manifold
            t: Interpolation parameter (0 = self, 1 = other)
            
        Returns:
            Point on geodesic
        """
        if not isinstance(other, ManifoldTensor):
            raise TypeError("geodesic() requires a ManifoldTensor")
        
        if other.manifold is not self._manifold:
            raise ValueError("Points must be on the same manifold")
        
        if isinstance(t, Scalar):
            t = t.item()
        
        result = self._manifold.geodesic(self.numpy(), other.numpy(), t)
        result_core = TensorCore(
            storage=self._core.storage.__class__(result, self._core.device),
            shape=result.shape,
            dtype=self._core.dtype,
            device=self._core.device,
        )
        
        return ManifoldTensor(result_core, self._manifold, self.requires_grad)
    
    def random_tangent(self) -> TangentTensor:
        """Generate random tangent vector at this point."""
        v = self._manifold.random_tangent(self.numpy())
        v_core = TensorCore(
            storage=self._core.storage.__class__(v, self._core.device),
            shape=v.shape,
            dtype=self._core.dtype,
            device=self._core.device,
        )
        return TangentTensor(v_core, self, self._manifold)
    
    def zero_tangent(self) -> TangentTensor:
        """Generate zero tangent vector at this point."""
        v = self._manifold.zero_tangent(self.numpy())
        v_core = TensorCore(
            storage=self._core.storage.__class__(v, self._core.device),
            shape=v.shape,
            dtype=self._core.dtype,
            device=self._core.device,
        )
        return TangentTensor(v_core, self, self._manifold)
    
    # Geometric arithmetic
    
    def __add__(self, v: TangentTensor) -> 'ManifoldTensor':
        """Point + TangentVector = exp(point, vector)."""
        if not isinstance(v, TangentTensor):
            return NotImplemented
        return self.exp(v)
    
    def __sub__(self, other: 'ManifoldTensor') -> TangentTensor:
        """Point - Point = log(other, self).
        
        y - x returns log_x(y), a tangent vector at x pointing toward y.
        """
        if not isinstance(other, ManifoldTensor):
            return NotImplemented
        return other.log(self)
    
    def __repr__(self):
        return f'ManifoldTensor(shape={self.shape}, manifold={self._manifold})'


# Factory functions

def randn(*shape, manifold: Manifold, dtype: DType = DType.FLOAT64, requires_grad: bool = False) -> ManifoldTensor:
    """Create random point on manifold.
    
    Args:
        *shape: Batch dimensions (empty for single point)
        manifold: The manifold
        dtype: Data type
        requires_grad: Whether to track gradients
        
    Returns:
        Random point on manifold with shape (*shape, manifold.dim) or (manifold.dim,) if no shape given
    """
    # If shape contains manifold.dim, it's likely a mistake - user probably meant batch dims
    # For now, use shape as batch dimensions
    # Generate random point using manifold's distribution
    point = manifold.random_point(*shape, dtype=dtype.numpy_dtype)
    
    core = TensorCore(
        storage=TensorCore.__new__(TensorCore).__class__.__bases__[0].__new__(
            TensorCore.__new__(TensorCore).__class__.__bases__[0]
        ),
        shape=point.shape,
        dtype=dtype,
    )
    # Simpler approach: use from core module
    from .core.storage import Storage
    storage = Storage(point, Device('cpu'))
    core = TensorCore(
        storage=storage,
        shape=point.shape,
        dtype=dtype,
    )
    
    return ManifoldTensor(core, manifold, requires_grad)


def origin(manifold: Manifold, *shape, dtype: DType = DType.FLOAT64, requires_grad: bool = False) -> ManifoldTensor:
    """Create origin/identity point on manifold.
    
    Args:
        manifold: The manifold
        *shape: Batch dimensions
        dtype: Data type
        requires_grad: Whether to track gradients
        
    Returns:
        Origin point on manifold
    """
    point = manifold.origin(*shape, dtype=dtype.numpy_dtype)
    
    from .core.storage import Storage
    storage = Storage(point, Device('cpu'))
    core = TensorCore(
        storage=storage,
        shape=point.shape,
        dtype=dtype,
    )
    
    return ManifoldTensor(core, manifold, requires_grad)


def tangent_randn(base: ManifoldTensor) -> TangentTensor:
    """Create random tangent vector at base point.
    
    Args:
        base: Base point on manifold
        
    Returns:
        Random tangent vector at base
    """
    return base.random_tangent()


def tangent_zeros(base: ManifoldTensor) -> TangentTensor:
    """Create zero tangent vector at base point.
    
    Args:
        base: Base point on manifold
        
    Returns:
        Zero tangent vector at base
    """
    return base.zero_tangent()
