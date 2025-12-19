"""
DavisTensor Core Module
=======================

Low-level storage, tensor core, and data types.
GPU-ready via CuPy (install cupy-cuda12x for GPU support).
"""

from .storage import (
    # Device
    Device,
    DeviceType,
    CPU,
    
    # Data types
    DType,
    float32,
    float64,
    int32,
    int64,
    
    # Storage
    Storage,
    
    # Geometric types
    GeometricType,
    
    # TensorCore
    TensorCore,
    
    # Factory functions
    zeros,
    ones,
    randn,
    rand,
    tensor,
    from_numpy,
    
    # Tests
    test_core,
)

# GPU-ready array API
from . import array_api
from .array_api import (
    gpu_available,
    cupy_available,
    get_array_module,
    to_device,
    to_numpy,
    get_device,
    set_default_device,
)

__all__ = [
    "Device",
    "DeviceType", 
    "CPU",
    "DType",
    "float32",
    "float64",
    "int32",
    "int64",
    "Storage",
    "GeometricType",
    "TensorCore",
    "zeros",
    "ones",
    "randn",
    "rand",
    "tensor",
    "from_numpy",
    "test_core",
    # GPU-ready
    "array_api",
    "gpu_available",
    "cupy_available",
    "get_array_module",
    "to_device",
    "to_numpy",
    "get_device",
    "set_default_device",
]
