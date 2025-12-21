"""High-level tensor wrappers for geometric operations."""

from __future__ import annotations
from typing import Optional, Union
import numpy as np
from .core.storage import TensorCore, GeometricType, _create_tensor
from .manifolds.base import Manifold


class ManifoldTensor:
    """
    A tensor that knows it lives on a Riemannian manifold.
    
    Wraps a TensorCore with an associated manifold, enabling
    geometric operations like exp, log, and geodesic distance.
    """
    
    def __init__(self, data: Union[TensorCore, np.ndarray], manifold: Manifold):
        """
        Create a ManifoldTensor.
        
        Args:
            data: Tensor data or numpy array
            manifold: The Riemannian manifold this tensor lives on
        """
        if isinstance(data, TensorCore):
            self._core = data
        else:
            self._core = _create_tensor(
                data,
                manifold=manifold,
                geometric_type=GeometricType.MANIFOLD_POINT,
            )
        
        self.manifold = manifold
        # Ensure core knows about manifold
        self._core.manifold = manifold
        if self._core.geometric_type == GeometricType.EUCLIDEAN:
            self._core.geometric_type = GeometricType.MANIFOLD_POINT
    
    @property
    def shape(self):
        """Shape of the tensor."""
        return self._core.shape
    
    @property
    def dtype(self):
        """Data type of the tensor."""
        return self._core.dtype
    
    @property
    def device(self):
        """Device of the tensor."""
        return self._core.device
    
    def numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return self._core.numpy()
    
    def __repr__(self) -> str:
        data_str = np.array2string(self.numpy(), precision=4, suppress_small=True)
        return f"ManifoldTensor({data_str}, manifold={self.manifold})"
    
    def __add__(self, other: Union['TangentTensor', 'ManifoldTensor']) -> 'ManifoldTensor':
        """
        Addition: Point + TangentVector = exp map.
        
        Args:
            other: TangentTensor at this point
        
        Returns:
            New point on manifold
        """
        if isinstance(other, TangentTensor):
            # Check base point compatibility
            if not np.allclose(other.base_point.numpy(), self.numpy()):
                raise TypeError(
                    "Cannot add tangent vector: base point mismatch. "
                    "The tangent vector must be at the same base point as this manifold tensor."
                )
            return self.exp(other)
        else:
            raise TypeError(f"Cannot add ManifoldTensor and {type(other)}")
    
    def __sub__(self, other: 'ManifoldTensor') -> 'TangentTensor':
        """
        Subtraction: Point - Point = log map.
        
        Args:
            other: Another point on the same manifold
        
        Returns:
            Tangent vector pointing from other to self
        """
        if not isinstance(other, ManifoldTensor):
            raise TypeError(f"Cannot subtract {type(other)} from ManifoldTensor")
        if other.manifold != self.manifold:
            raise ValueError("Points must be on the same manifold")
        
        # log_other(self) gives tangent vector at other pointing to self
        v_core = self.manifold.log(other._core, self._core)
        return TangentTensor(v_core, base_point=other, manifold=self.manifold)
    
    def exp(self, v: 'TangentTensor') -> 'ManifoldTensor':
        """
        Exponential map: move from this point in direction v.
        
        Args:
            v: Tangent vector at this point
        
        Returns:
            New point on manifold
        """
        if not isinstance(v, TangentTensor):
            raise TypeError("exp requires a TangentTensor")
        
        # Verify base point matches
        if not np.allclose(v.base_point.numpy(), self.numpy()):
            raise ValueError("Tangent vector must be at this base point")
        
        result_core = self.manifold.exp(self._core, v._core)
        return ManifoldTensor(result_core, manifold=self.manifold)
    
    def log(self, q: 'ManifoldTensor') -> 'TangentTensor':
        """
        Logarithm map: tangent vector pointing toward q.
        
        Args:
            q: Target point on manifold
        
        Returns:
            Tangent vector at this point
        """
        if not isinstance(q, ManifoldTensor):
            raise TypeError("log requires a ManifoldTensor")
        if q.manifold != self.manifold:
            raise ValueError("Points must be on the same manifold")
        
        v_core = self.manifold.log(self._core, q._core)
        return TangentTensor(v_core, base_point=self, manifold=self.manifold)
    
    def distance(self, q: 'ManifoldTensor') -> 'Scalar':
        """
        Geodesic distance to q.
        
        Args:
            q: Target point on manifold
        
        Returns:
            Geodesic distance as a Scalar
        """
        if not isinstance(q, ManifoldTensor):
            raise TypeError("distance requires a ManifoldTensor")
        if q.manifold != self.manifold:
            raise ValueError("Points must be on the same manifold")
        
        dist_core = self.manifold.distance(self._core, q._core)
        return Scalar(dist_core)


class TangentTensor:
    """
    A tangent vector at a specific point on a manifold.
    
    Type: TangentTensor[M, x] where M is manifold, x is base point.
    
    Key property: tangent vectors can only be added at the SAME point.
    """
    
    def __init__(
        self,
        data: Union[TensorCore, np.ndarray],
        base_point: Union[ManifoldTensor, TensorCore],
        manifold: Manifold,
    ):
        """
        Create a TangentTensor.
        
        Args:
            data: Tensor data or numpy array
            base_point: Base point where this tangent vector lives
            manifold: The Riemannian manifold
        """
        if isinstance(data, TensorCore):
            self._core = data
        else:
            self._core = _create_tensor(
                data,
                manifold=manifold,
                geometric_type=GeometricType.TANGENT,
            )
        
        # Store base point
        if isinstance(base_point, ManifoldTensor):
            self.base_point = base_point
        else:
            # Wrap TensorCore as ManifoldTensor
            self.base_point = ManifoldTensor(base_point, manifold)
        
        self.manifold = manifold
        
        # Ensure core knows about manifold and base point
        self._core.manifold = manifold
        self._core.geometric_type = GeometricType.TANGENT
        self._core.base_point = self.base_point._core
    
    @property
    def shape(self):
        """Shape of the tensor."""
        return self._core.shape
    
    @property
    def dtype(self):
        """Data type of the tensor."""
        return self._core.dtype
    
    @property
    def device(self):
        """Device of the tensor."""
        return self._core.device
    
    def numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return self._core.numpy()
    
    def __repr__(self) -> str:
        data_str = np.array2string(self.numpy(), precision=4, suppress_small=True)
        return f"TangentTensor({data_str}, manifold={self.manifold})"
    
    def __add__(self, other: 'TangentTensor') -> 'TangentTensor':
        """
        Add tangent vectors at the same point.
        
        Args:
            other: Another tangent vector
        
        Returns:
            Sum of tangent vectors
        
        Raises:
            TypeError: If base points don't match
        """
        if not isinstance(other, TangentTensor):
            raise TypeError(f"Cannot add TangentTensor and {type(other)}")
        
        # Check base points match
        if not np.allclose(self.base_point.numpy(), other.base_point.numpy()):
            raise TypeError(
                "Cannot add tangent vectors at different points. "
                "Use parallel_transport first."
            )
        
        # Element-wise addition
        result_data = self.numpy() + other.numpy()
        return TangentTensor(result_data, base_point=self.base_point, manifold=self.manifold)
    
    def __mul__(self, scalar: Union[float, int]) -> 'TangentTensor':
        """
        Scalar multiplication.
        
        Args:
            scalar: Scalar value
        
        Returns:
            Scaled tangent vector
        """
        result_data = self.numpy() * scalar
        return TangentTensor(result_data, base_point=self.base_point, manifold=self.manifold)
    
    def __rmul__(self, scalar: Union[float, int]) -> 'TangentTensor':
        """Reverse scalar multiplication."""
        return self.__mul__(scalar)
    
    def __neg__(self) -> 'TangentTensor':
        """Negate tangent vector."""
        result_data = -self.numpy()
        return TangentTensor(result_data, base_point=self.base_point, manifold=self.manifold)
    
    def norm(self) -> 'Scalar':
        """
        Norm of this tangent vector.
        
        Returns:
            ||self||_p where p is the base point
        """
        norm_core = self.manifold.norm(self.base_point._core, self._core)
        return Scalar(norm_core)


class Scalar:
    """
    A scalar value that doesn't live on any manifold.
    
    Results of distance(), norm(), inner(), etc. are Scalars.
    """
    
    def __init__(self, data: Union[TensorCore, float, int, np.ndarray]):
        """
        Create a Scalar.
        
        Args:
            data: Scalar value
        """
        if isinstance(data, TensorCore):
            self._core = data
        else:
            self._core = _create_tensor(
                data,
                geometric_type=GeometricType.SCALAR,
            )
        
        self._core.geometric_type = GeometricType.SCALAR
    
    def item(self) -> float:
        """Convert to Python float."""
        arr = self._core.numpy()
        if arr.shape == ():
            return float(arr)
        return float(arr.flatten()[0])
    
    def numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return self._core.numpy()
    
    def __repr__(self) -> str:
        return f"Scalar({self.item()})"
    
    def __float__(self) -> float:
        """Convert to float."""
        return self.item()
    
    def __add__(self, other: Union['Scalar', float, int]) -> 'Scalar':
        """Add scalars."""
        if isinstance(other, Scalar):
            other_val = other.item()
        else:
            other_val = float(other)
        return Scalar(self.item() + other_val)
    
    def __radd__(self, other: Union[float, int]) -> 'Scalar':
        """Reverse add."""
        return self.__add__(other)
    
    def __mul__(self, other: Union['Scalar', float, int]) -> 'Scalar':
        """Multiply scalars."""
        if isinstance(other, Scalar):
            other_val = other.item()
        else:
            other_val = float(other)
        return Scalar(self.item() * other_val)
    
    def __rmul__(self, other: Union[float, int]) -> 'Scalar':
        """Reverse multiply."""
        return self.__mul__(other)
    
    def __truediv__(self, other: Union['Scalar', float, int]) -> 'Scalar':
        """Divide scalars."""
        if isinstance(other, Scalar):
            other_val = other.item()
        else:
            other_val = float(other)
        return Scalar(self.item() / other_val)
