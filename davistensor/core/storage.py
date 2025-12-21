"""Core storage layer for DavisTensor - memory management and tensor infrastructure."""

from enum import Enum
from typing import Optional, Tuple, Union
import numpy as np


class DType(Enum):
    """Data type enumeration."""
    FLOAT32 = 'float32'
    FLOAT64 = 'float64'
    INT32 = 'int32'
    INT64 = 'int64'
    
    @property
    def numpy_dtype(self):
        """Convert to numpy dtype."""
        return np.dtype(self.value)
    
    @property
    def itemsize(self):
        """Size in bytes of a single element."""
        return self.numpy_dtype.itemsize


class GeometricType(Enum):
    """Type of geometric object represented by a tensor."""
    SCALAR = 'scalar'
    EUCLIDEAN = 'euclidean'
    MANIFOLD_POINT = 'manifold_point'
    TANGENT = 'tangent'
    COTANGENT = 'cotangent'


class Device:
    """Device abstraction for tensor storage."""
    
    def __init__(self, device_type: str = 'cpu', device_id: int = 0):
        """Initialize device.
        
        Args:
            device_type: 'cpu' or 'cuda'
            device_id: Device ID for CUDA devices
        """
        self.device_type = device_type
        self.device_id = device_id
        
        if device_type not in ('cpu', 'cuda'):
            raise ValueError(f"Unsupported device type: {device_type}")
        
        if device_type == 'cuda':
            raise NotImplementedError("CUDA support not yet implemented")
    
    def __str__(self):
        if self.device_type == 'cpu':
            return 'cpu'
        return f'{self.device_type}:{self.device_id}'
    
    def __repr__(self):
        return f'Device({self.device_type!r}, {self.device_id})'
    
    def __eq__(self, other):
        if not isinstance(other, Device):
            return False
        return self.device_type == other.device_type and self.device_id == other.device_id
    
    def __hash__(self):
        return hash((self.device_type, self.device_id))


class Storage:
    """Raw memory buffer with reference counting."""
    
    def __init__(self, data: np.ndarray, device: Optional[Device] = None):
        """Initialize storage.
        
        Args:
            data: Numpy array containing the data
            device: Device where data is stored
        """
        self.data = data
        self.device = device or Device('cpu')
        self.ref_count = 1
    
    def __del__(self):
        """Clean up when storage is no longer referenced."""
        self.ref_count -= 1
    
    def clone(self) -> 'Storage':
        """Create a copy of this storage."""
        return Storage(self.data.copy(), self.device)


class TensorCore:
    """Core tensor data structure with geometric metadata."""
    
    def __init__(
        self,
        storage: Storage,
        shape: Tuple[int, ...],
        strides: Optional[Tuple[int, ...]] = None,
        offset: int = 0,
        dtype: DType = DType.FLOAT64,
        device: Optional[Device] = None,
        # Geometric metadata
        manifold: Optional[object] = None,
        base_point: Optional['TensorCore'] = None,
        geometric_type: Optional[GeometricType] = None,
        # Autograd metadata
        requires_grad: bool = False,
        grad: Optional['TensorCore'] = None,
        grad_fn: Optional[object] = None,
    ):
        """Initialize TensorCore.
        
        Args:
            storage: Raw memory buffer
            shape: Tensor shape
            strides: Memory layout strides (computed if None)
            offset: Offset into storage
            dtype: Data type
            device: Compute device
            manifold: Manifold this tensor lives on (if any)
            base_point: Base point for tangent vectors
            geometric_type: Type of geometric object
            requires_grad: Whether to track gradients
            grad: Gradient tensor
            grad_fn: Gradient function for autograd
        """
        self.storage = storage
        self.shape = shape
        self.offset = offset
        self.dtype = dtype
        self.device = device or Device('cpu')
        
        # Compute strides if not provided (row-major/C-contiguous)
        if strides is None:
            strides = self._compute_strides(shape)
        self.strides = strides
        
        # Geometric metadata
        self.manifold = manifold
        self.base_point = base_point
        if geometric_type is None:
            geometric_type = self._infer_geometric_type()
        self.geometric_type = geometric_type
        
        # Autograd metadata
        self.requires_grad = requires_grad
        self.grad = grad
        self.grad_fn = grad_fn
    
    @staticmethod
    def _compute_strides(shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute row-major (C-contiguous) strides."""
        if not shape:
            return ()
        strides = [1]
        for dim in reversed(shape[1:]):
            strides.insert(0, strides[0] * dim)
        return tuple(strides)
    
    def _infer_geometric_type(self) -> GeometricType:
        """Infer geometric type from metadata."""
        if self.manifold is None:
            return GeometricType.EUCLIDEAN
        elif self.base_point is not None:
            return GeometricType.TANGENT
        else:
            return GeometricType.MANIFOLD_POINT
    
    @property
    def data(self) -> np.ndarray:
        """Get the underlying numpy array."""
        return self.storage.data
    
    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.shape)
    
    @property
    def size(self) -> int:
        """Total number of elements."""
        if not self.shape:
            return 1
        result = 1
        for dim in self.shape:
            result *= dim
        return result
    
    def numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return self.data.reshape(self.shape)
    
    def clone(self) -> 'TensorCore':
        """Create a deep copy of this tensor."""
        return TensorCore(
            storage=self.storage.clone(),
            shape=self.shape,
            strides=self.strides,
            offset=self.offset,
            dtype=self.dtype,
            device=self.device,
            manifold=self.manifold,
            base_point=self.base_point.clone() if self.base_point is not None else None,
            geometric_type=self.geometric_type,
            requires_grad=self.requires_grad,
            grad=None,  # Don't copy gradients
            grad_fn=None,
        )
    
    def reshape(self, *new_shape: int) -> 'TensorCore':
        """Reshape tensor (creates view if possible)."""
        if len(new_shape) == 1 and isinstance(new_shape[0], (tuple, list)):
            new_shape = tuple(new_shape[0])
        
        # Check size matches
        new_size = 1
        for dim in new_shape:
            new_size *= dim
        if new_size != self.size:
            raise ValueError(f"Cannot reshape tensor of size {self.size} to shape {new_shape}")
        
        # For now, always create a copy
        # TODO: Implement view-based reshaping when strides allow
        new_data = self.numpy().reshape(new_shape)
        new_storage = Storage(new_data, self.device)
        
        return TensorCore(
            storage=new_storage,
            shape=new_shape,
            dtype=self.dtype,
            device=self.device,
            manifold=self.manifold,
            base_point=self.base_point,
            geometric_type=self.geometric_type,
            requires_grad=self.requires_grad,
        )
    
    def __getitem__(self, key):
        """Index into tensor."""
        result = self.numpy()[key]
        if isinstance(result, np.ndarray):
            new_storage = Storage(result.copy(), self.device)
            return TensorCore(
                storage=new_storage,
                shape=result.shape,
                dtype=self.dtype,
                device=self.device,
                manifold=self.manifold,
                base_point=self.base_point,
                geometric_type=self.geometric_type,
                requires_grad=self.requires_grad,
            )
        else:
            # Scalar indexing
            return result
    
    def __repr__(self):
        return f'TensorCore(shape={self.shape}, dtype={self.dtype.value}, geometric_type={self.geometric_type.value})'


# Factory functions

def zeros(*shape: int, dtype: DType = DType.FLOAT64, device: Optional[Device] = None) -> TensorCore:
    """Create a tensor filled with zeros.
    
    Args:
        *shape: Tensor dimensions
        dtype: Data type
        device: Compute device
        
    Returns:
        TensorCore filled with zeros
    """
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    
    device = device or Device('cpu')
    data = np.zeros(shape, dtype=dtype.numpy_dtype)
    storage = Storage(data, device)
    
    return TensorCore(
        storage=storage,
        shape=shape,
        dtype=dtype,
        device=device,
    )


def ones(*shape: int, dtype: DType = DType.FLOAT64, device: Optional[Device] = None) -> TensorCore:
    """Create a tensor filled with ones.
    
    Args:
        *shape: Tensor dimensions
        dtype: Data type
        device: Compute device
        
    Returns:
        TensorCore filled with ones
    """
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    
    device = device or Device('cpu')
    data = np.ones(shape, dtype=dtype.numpy_dtype)
    storage = Storage(data, device)
    
    return TensorCore(
        storage=storage,
        shape=shape,
        dtype=dtype,
        device=device,
    )


def randn(*shape: int, dtype: DType = DType.FLOAT64, device: Optional[Device] = None) -> TensorCore:
    """Create a tensor filled with random normal values.
    
    Args:
        *shape: Tensor dimensions
        dtype: Data type
        device: Compute device
        
    Returns:
        TensorCore with random normal values
    """
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    
    device = device or Device('cpu')
    data = np.random.randn(*shape).astype(dtype.numpy_dtype)
    storage = Storage(data, device)
    
    return TensorCore(
        storage=storage,
        shape=shape,
        dtype=dtype,
        device=device,
    )


def rand(*shape: int, dtype: DType = DType.FLOAT64, device: Optional[Device] = None) -> TensorCore:
    """Create a tensor filled with random uniform values in [0, 1).
    
    Args:
        *shape: Tensor dimensions
        dtype: Data type
        device: Compute device
        
    Returns:
        TensorCore with random uniform values
    """
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    
    device = device or Device('cpu')
    data = np.random.rand(*shape).astype(dtype.numpy_dtype)
    storage = Storage(data, device)
    
    return TensorCore(
        storage=storage,
        shape=shape,
        dtype=dtype,
        device=device,
    )


def tensor(data, dtype: DType = DType.FLOAT64, device: Optional[Device] = None) -> TensorCore:
    """Create a tensor from array-like data.
    
    Args:
        data: Array-like data (list, numpy array, etc.)
        dtype: Data type
        device: Compute device
        
    Returns:
        TensorCore containing the data
    """
    device = device or Device('cpu')
    
    if isinstance(data, np.ndarray):
        np_data = data.astype(dtype.numpy_dtype)
    else:
        np_data = np.array(data, dtype=dtype.numpy_dtype)
    
    storage = Storage(np_data, device)
    
    return TensorCore(
        storage=storage,
        shape=np_data.shape,
        dtype=dtype,
        device=device,
    )


def from_numpy(array: np.ndarray, dtype: Optional[DType] = None, device: Optional[Device] = None) -> TensorCore:
    """Create a tensor from a numpy array.
    
    Args:
        array: Numpy array
        dtype: Data type (inferred from array if None)
        device: Compute device
        
    Returns:
        TensorCore containing the array data
    """
    device = device or Device('cpu')
    
    if dtype is None:
        # Infer dtype from numpy array
        if array.dtype == np.float32:
            dtype = DType.FLOAT32
        elif array.dtype == np.float64:
            dtype = DType.FLOAT64
        elif array.dtype == np.int32:
            dtype = DType.INT32
        elif array.dtype == np.int64:
            dtype = DType.INT64
        else:
            # Default to float64
            dtype = DType.FLOAT64
            array = array.astype(np.float64)
    else:
        array = array.astype(dtype.numpy_dtype)
    
    storage = Storage(array.copy(), device)
    
    return TensorCore(
        storage=storage,
        shape=array.shape,
        dtype=dtype,
        device=device,
    )
