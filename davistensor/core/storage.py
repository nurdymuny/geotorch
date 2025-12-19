"""
DavisTensor Core: Storage and TensorCore
=========================================

The foundation layer - raw memory management and tensor data structures.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, Union, List, Callable, Any
from enum import Enum
from dataclasses import dataclass
import math


# =============================================================================
# Device Abstraction
# =============================================================================

class DeviceType(Enum):
    CPU = "cpu"
    CUDA = "cuda"


@dataclass
class Device:
    """Represents a compute device."""
    type: DeviceType
    index: int = 0
    
    def __repr__(self) -> str:
        if self.type == DeviceType.CPU:
            return "cpu"
        return f"cuda:{self.index}"
    
    @staticmethod
    def cpu() -> 'Device':
        return Device(DeviceType.CPU, 0)
    
    @staticmethod
    def cuda(index: int = 0) -> 'Device':
        return Device(DeviceType.CUDA, index)


CPU = Device.cpu()


# =============================================================================
# Data Types
# =============================================================================

class DType(Enum):
    FLOAT32 = ("float32", np.float32, 4)
    FLOAT64 = ("float64", np.float64, 8)
    INT32 = ("int32", np.int32, 4)
    INT64 = ("int64", np.int64, 8)
    BOOL = ("bool", np.bool_, 1)
    
    def __init__(self, name: str, numpy_dtype, size: int):
        self._name = name
        self.numpy_dtype = numpy_dtype
        self.itemsize = size
    
    def __repr__(self) -> str:
        return f"dt.{self._name}"


float32 = DType.FLOAT32
float64 = DType.FLOAT64
int32 = DType.INT32
int64 = DType.INT64


# =============================================================================
# Storage: Raw Memory Buffer
# =============================================================================

class Storage:
    """
    Raw memory buffer backing tensor data.
    
    For CPU: wraps a numpy array
    For CUDA: would wrap device pointer (future)
    
    Reference counted for memory management.
    """
    
    def __init__(
        self, 
        size: int, 
        dtype: DType = float32,
        device: Device = CPU,
        data: Optional[np.ndarray] = None
    ):
        self.size = size
        self.dtype = dtype
        self.device = device
        self._ref_count = 1
        
        if data is not None:
            self._data = data.astype(dtype.numpy_dtype).flatten()
            if len(self._data) < size:
                # Pad if needed
                self._data = np.concatenate([
                    self._data, 
                    np.zeros(size - len(self._data), dtype=dtype.numpy_dtype)
                ])
        else:
            self._data = np.zeros(size, dtype=dtype.numpy_dtype)
    
    def __getitem__(self, idx: int) -> Any:
        return self._data[idx]
    
    def __setitem__(self, idx: int, value: Any):
        self._data[idx] = value
    
    def clone(self) -> 'Storage':
        """Create a copy of this storage."""
        return Storage(
            self.size, 
            self.dtype, 
            self.device,
            self._data.copy()
        )
    
    @property
    def data_ptr(self) -> int:
        """Memory address of data (for debugging)."""
        return self._data.ctypes.data
    
    def numpy(self) -> np.ndarray:
        """Get underlying numpy array."""
        return self._data


# =============================================================================
# Geometric Type Enum
# =============================================================================

class GeometricType(Enum):
    """What kind of geometric object is this tensor?"""
    SCALAR = "scalar"           # Manifold-independent scalar
    EUCLIDEAN = "euclidean"     # Standard Euclidean tensor
    MANIFOLD_POINT = "point"    # Point on a manifold
    TANGENT = "tangent"         # Tangent vector at a point
    COTANGENT = "cotangent"     # Cotangent (gradient) at a point


# =============================================================================
# TensorCore: The Fundamental Data Structure
# =============================================================================

class TensorCore:
    """
    Core tensor data structure.
    
    Contains:
    - Storage (raw data)
    - Shape and strides
    - Geometric metadata (manifold, base point, etc.)
    - Autograd metadata (grad_fn, requires_grad, etc.)
    
    This is the internal representation. Users interact with
    ManifoldTensor, TangentTensor, or Tensor wrappers.
    """
    
    def __init__(
        self,
        storage: Storage,
        shape: Tuple[int, ...],
        strides: Optional[Tuple[int, ...]] = None,
        offset: int = 0,
        dtype: DType = float32,
        device: Device = CPU,
        requires_grad: bool = False,
        # Geometric metadata
        manifold: Optional[Any] = None,  # Will be Manifold type
        base_point: Optional['TensorCore'] = None,  # For tangent vectors
        geometric_type: GeometricType = GeometricType.EUCLIDEAN,
    ):
        self.storage = storage
        self.shape = tuple(shape)
        self.offset = offset
        self.dtype = dtype
        self.device = device
        
        # Compute strides if not provided (row-major / C-contiguous)
        if strides is None:
            self.strides = self._compute_strides(shape)
        else:
            self.strides = tuple(strides)
        
        # Geometric metadata
        self.manifold = manifold
        self.base_point = base_point
        self.geometric_type = geometric_type
        
        # Autograd metadata
        self.requires_grad = requires_grad
        self.grad: Optional['TensorCore'] = None
        self.grad_fn: Optional[Any] = None  # Will be GradFn type
        self._version = 0  # For detecting in-place modifications
    
    @staticmethod
    def _compute_strides(shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute row-major strides for shape."""
        if len(shape) == 0:
            return ()
        
        strides = [1]
        for dim in reversed(shape[1:]):
            strides.append(strides[-1] * dim)
        return tuple(reversed(strides))
    
    @property
    def ndim(self) -> int:
        return len(self.shape)
    
    @property
    def numel(self) -> int:
        """Total number of elements."""
        if len(self.shape) == 0:
            return 1
        result = 1
        for s in self.shape:
            result *= s
        return result
    
    @property
    def is_contiguous(self) -> bool:
        """Check if tensor is contiguous in memory."""
        return self.strides == self._compute_strides(self.shape)
    
    def _flat_index(self, indices: Tuple[int, ...]) -> int:
        """Convert multi-dimensional index to flat storage index."""
        idx = self.offset
        for i, (index, stride) in enumerate(zip(indices, self.strides)):
            if index < 0:
                index = self.shape[i] + index
            if index < 0 or index >= self.shape[i]:
                raise IndexError(f"Index {index} out of bounds for dimension {i} with size {self.shape[i]}")
            idx += index * stride
        return idx
    
    def __getitem__(self, indices) -> Union['TensorCore', float]:
        """Get element or slice."""
        # Handle single index
        if isinstance(indices, int):
            indices = (indices,)
        
        # Handle slice (simplified - just integer indexing for now)
        if isinstance(indices, tuple) and all(isinstance(i, int) for i in indices):
            if len(indices) == self.ndim:
                # Single element access
                flat_idx = self._flat_index(indices)
                return float(self.storage[flat_idx])
            else:
                # Partial indexing - return view
                # TODO: implement properly
                raise NotImplementedError("Partial indexing not yet implemented")
        
        raise NotImplementedError("Advanced indexing not yet implemented")
    
    def __setitem__(self, indices, value):
        """Set element."""
        if isinstance(indices, int):
            indices = (indices,)
        
        if isinstance(indices, tuple) and all(isinstance(i, int) for i in indices):
            if len(indices) == self.ndim:
                flat_idx = self._flat_index(indices)
                self.storage[flat_idx] = value
                self._version += 1
                return
        
        raise NotImplementedError("Advanced indexing not yet implemented")
    
    def clone(self) -> 'TensorCore':
        """Create a deep copy."""
        return TensorCore(
            storage=self.storage.clone(),
            shape=self.shape,
            strides=self.strides,
            offset=self.offset,
            dtype=self.dtype,
            device=self.device,
            requires_grad=self.requires_grad,
            manifold=self.manifold,
            base_point=self.base_point,
            geometric_type=self.geometric_type,
        )
    
    def contiguous(self) -> 'TensorCore':
        """Return contiguous tensor (copy if necessary)."""
        if self.is_contiguous:
            return self
        
        # Create new contiguous storage
        new_storage = Storage(self.numel, self.dtype, self.device)
        
        # Copy data
        for idx in np.ndindex(self.shape):
            flat_src = self._flat_index(idx)
            flat_dst = sum(i * s for i, s in zip(idx, self._compute_strides(self.shape)))
            new_storage[flat_dst] = self.storage[flat_src]
        
        return TensorCore(
            storage=new_storage,
            shape=self.shape,
            dtype=self.dtype,
            device=self.device,
            requires_grad=self.requires_grad,
            manifold=self.manifold,
            base_point=self.base_point,
            geometric_type=self.geometric_type,
        )
    
    def numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        # Get contiguous data
        tc = self.contiguous()
        data = tc.storage.numpy()[tc.offset:tc.offset + tc.numel]
        return data.reshape(tc.shape)
    
    def view(self, *new_shape: int) -> 'TensorCore':
        """Reshape tensor (must be contiguous)."""
        if not self.is_contiguous:
            raise RuntimeError("view requires contiguous tensor")
        
        # Handle -1 dimension
        new_shape = list(new_shape)
        neg_idx = None
        known_numel = 1
        for i, s in enumerate(new_shape):
            if s == -1:
                if neg_idx is not None:
                    raise ValueError("Only one dimension can be -1")
                neg_idx = i
            else:
                known_numel *= s
        
        if neg_idx is not None:
            new_shape[neg_idx] = self.numel // known_numel
        
        if math.prod(new_shape) != self.numel:
            raise ValueError(f"Cannot reshape {self.shape} to {tuple(new_shape)}")
        
        return TensorCore(
            storage=self.storage,
            shape=tuple(new_shape),
            offset=self.offset,
            dtype=self.dtype,
            device=self.device,
            requires_grad=self.requires_grad,
            manifold=self.manifold,
            base_point=self.base_point,
            geometric_type=self.geometric_type,
        )
    
    def __repr__(self) -> str:
        data_str = np.array2string(self.numpy(), precision=4, suppress_small=True)
        
        parts = [f"tensor({data_str}"]
        
        if self.manifold is not None:
            parts.append(f", manifold={self.manifold}")
        if self.geometric_type != GeometricType.EUCLIDEAN:
            parts.append(f", type={self.geometric_type.value}")
        if self.requires_grad:
            parts.append(", requires_grad=True")
        if self.grad_fn is not None:
            parts.append(f", grad_fn=<{self.grad_fn.__class__.__name__}>")
        
        parts.append(")")
        return "".join(parts)


# =============================================================================
# Factory Functions
# =============================================================================

def _create_tensor(
    data: Union[np.ndarray, List, float, int],
    dtype: Optional[DType] = None,
    device: Device = CPU,
    requires_grad: bool = False,
    manifold: Optional[Any] = None,
    geometric_type: GeometricType = GeometricType.EUCLIDEAN,
) -> TensorCore:
    """Create TensorCore from data."""
    
    # Convert to numpy
    if isinstance(data, (int, float)):
        arr = np.array([data])
        shape = ()
    elif isinstance(data, list):
        arr = np.array(data)
        shape = arr.shape
    elif isinstance(data, np.ndarray):
        arr = data
        shape = arr.shape
    else:
        raise TypeError(f"Cannot create tensor from {type(data)}")
    
    # Infer dtype
    if dtype is None:
        if arr.dtype in (np.float32, np.float64):
            dtype = float32 if arr.dtype == np.float32 else float64
        elif arr.dtype in (np.int32, np.int64):
            dtype = int32 if arr.dtype == np.int32 else int64
        else:
            dtype = float32
    
    # Create storage
    storage = Storage(arr.size, dtype, device, arr.flatten())
    
    return TensorCore(
        storage=storage,
        shape=shape,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
        manifold=manifold,
        geometric_type=geometric_type,
    )


def zeros(*shape: int, dtype: DType = float32, device: Device = CPU, requires_grad: bool = False) -> TensorCore:
    """Create tensor of zeros."""
    numel = math.prod(shape) if shape else 1
    storage = Storage(numel, dtype, device)
    return TensorCore(
        storage=storage,
        shape=shape if shape else (),
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


def ones(*shape: int, dtype: DType = float32, device: Device = CPU, requires_grad: bool = False) -> TensorCore:
    """Create tensor of ones."""
    numel = math.prod(shape) if shape else 1
    data = np.ones(numel, dtype=dtype.numpy_dtype)
    storage = Storage(numel, dtype, device, data)
    return TensorCore(
        storage=storage,
        shape=shape if shape else (),
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


def randn(*shape: int, dtype: DType = float32, device: Device = CPU, requires_grad: bool = False) -> TensorCore:
    """Create tensor with random normal values."""
    numel = math.prod(shape) if shape else 1
    data = np.random.randn(numel).astype(dtype.numpy_dtype)
    storage = Storage(numel, dtype, device, data)
    return TensorCore(
        storage=storage,
        shape=shape if shape else (),
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


def rand(*shape: int, dtype: DType = float32, device: Device = CPU, requires_grad: bool = False) -> TensorCore:
    """Create tensor with random uniform [0, 1) values."""
    numel = math.prod(shape) if shape else 1
    data = np.random.rand(numel).astype(dtype.numpy_dtype)
    storage = Storage(numel, dtype, device, data)
    return TensorCore(
        storage=storage,
        shape=shape if shape else (),
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


def tensor(data, dtype: Optional[DType] = None, device: Device = CPU, requires_grad: bool = False) -> TensorCore:
    """Create tensor from data."""
    return _create_tensor(data, dtype, device, requires_grad)


def from_numpy(arr: np.ndarray, requires_grad: bool = False) -> TensorCore:
    """Create tensor from numpy array."""
    return _create_tensor(arr, requires_grad=requires_grad)


# =============================================================================
# Tests
# =============================================================================

def test_core():
    """Test core functionality."""
    print("=" * 60)
    print("Testing DavisTensor Core")
    print("=" * 60)
    
    # Test tensor creation
    print("\n1. Tensor creation")
    t = randn(3, 4)
    print(f"   Shape: {t.shape}")
    print(f"   Strides: {t.strides}")
    print(f"   Numel: {t.numel}")
    print(f"   Contiguous: {t.is_contiguous}")
    assert t.shape == (3, 4)
    assert t.numel == 12
    assert t.is_contiguous
    print("   ✅ PASS")
    
    # Test indexing
    print("\n2. Indexing")
    t[0, 0] = 42.0
    assert t[0, 0] == 42.0
    print(f"   t[0, 0] = {t[0, 0]}")
    print("   ✅ PASS")
    
    # Test numpy conversion
    print("\n3. Numpy conversion")
    arr = t.numpy()
    print(f"   Numpy shape: {arr.shape}")
    assert arr.shape == t.shape
    assert arr[0, 0] == 42.0
    print("   ✅ PASS")
    
    # Test view
    print("\n4. View/reshape")
    t2 = t.view(12)
    assert t2.shape == (12,)
    t3 = t.view(2, 6)
    assert t3.shape == (2, 6)
    t4 = t.view(-1)
    assert t4.shape == (12,)
    print(f"   view(12): {t2.shape}")
    print(f"   view(2, 6): {t3.shape}")
    print(f"   view(-1): {t4.shape}")
    print("   ✅ PASS")
    
    # Test clone
    print("\n5. Clone")
    t_clone = t.clone()
    t_clone[0, 0] = 0.0
    assert t[0, 0] == 42.0  # Original unchanged
    print("   Clone is independent")
    print("   ✅ PASS")
    
    # Test factory functions
    print("\n6. Factory functions")
    z = zeros(2, 3)
    o = ones(2, 3)
    r = randn(2, 3)
    print(f"   zeros(2,3): sum={z.numpy().sum()}")
    print(f"   ones(2,3): sum={o.numpy().sum()}")
    print(f"   randn(2,3): mean={r.numpy().mean():.3f}")
    assert z.numpy().sum() == 0
    assert o.numpy().sum() == 6
    print("   ✅ PASS")
    
    # Test tensor from data
    print("\n7. tensor() from data")
    t_list = tensor([[1, 2], [3, 4]])
    t_scalar = tensor(3.14)
    print(f"   From list: {t_list.shape}")
    print(f"   From scalar: {t_scalar.shape}, value={t_scalar.numpy()}")
    assert t_list.shape == (2, 2)
    assert t_scalar.shape == ()
    print("   ✅ PASS")
    
    print("\n" + "=" * 60)
    print("All core tests passed! ✅")
    print("=" * 60)


if __name__ == "__main__":
    test_core()
