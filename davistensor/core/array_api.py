"""
DavisTensor Array API - GPU-Ready Abstraction
==============================================

Unified array interface that works with:
- NumPy (CPU) - current default
- CuPy (CUDA GPU) - drop-in when ready

The key insight: CuPy has nearly identical API to NumPy.
We just swap the import and everything works.

Usage:
    from davistensor.core.array_api import xp, get_array_module
    
    # xp is either numpy or cupy depending on device
    x = xp.randn(64)
    y = xp.zeros((3, 4))
    
    # Or get module for specific array
    xp = get_array_module(some_array)
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union
import numpy as np

# =============================================================================
# GPU Availability Detection
# =============================================================================

_GPU_AVAILABLE = False
_CUPY_AVAILABLE = False

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
    _GPU_AVAILABLE = cp.cuda.is_available()
except ImportError:
    cp = None

def gpu_available() -> bool:
    """Check if GPU is available."""
    return _GPU_AVAILABLE

def cupy_available() -> bool:
    """Check if CuPy is installed."""
    return _CUPY_AVAILABLE


# =============================================================================
# Array Module Selection
# =============================================================================

# Default to NumPy
xp = np

def set_default_device(device: str):
    """
    Set the default array module based on device.
    
    Args:
        device: 'cpu' or 'cuda' or 'cuda:N'
    """
    global xp
    
    if device == 'cpu':
        xp = np
    elif device.startswith('cuda'):
        if not _CUPY_AVAILABLE:
            raise RuntimeError(
                "CuPy not installed. Install with: pip install cupy-cuda12x\n"
                "See: https://docs.cupy.dev/en/stable/install.html"
            )
        if not _GPU_AVAILABLE:
            raise RuntimeError("CUDA GPU not available")
        xp = cp
        # Parse device index if specified (cuda:0, cuda:1, etc.)
        if ':' in device:
            idx = int(device.split(':')[1])
            cp.cuda.Device(idx).use()


def get_array_module(arr: Any):
    """
    Get the array module (numpy or cupy) for an array.
    
    This is the key to writing device-agnostic code:
        xp = get_array_module(x)
        y = xp.zeros_like(x)  # Same device as x
    """
    if _CUPY_AVAILABLE:
        return cp.get_array_module(arr)
    return np


def to_device(arr: Any, device: str) -> Any:
    """
    Move array to specified device.
    
    Args:
        arr: numpy or cupy array
        device: 'cpu' or 'cuda' or 'cuda:N'
    
    Returns:
        Array on the specified device
    """
    if device == 'cpu':
        if _CUPY_AVAILABLE and isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
        return arr
    
    elif device.startswith('cuda'):
        if not _CUPY_AVAILABLE:
            raise RuntimeError("CuPy not installed for GPU support")
        
        # Parse device index
        idx = 0
        if ':' in device:
            idx = int(device.split(':')[1])
        
        with cp.cuda.Device(idx):
            if isinstance(arr, np.ndarray):
                return cp.asarray(arr)
            elif isinstance(arr, cp.ndarray):
                # Already on GPU, maybe different device
                return cp.asarray(arr)
            else:
                return cp.asarray(arr)
    
    raise ValueError(f"Unknown device: {device}")


def to_numpy(arr: Any) -> np.ndarray:
    """Convert any array to numpy (CPU)."""
    if isinstance(arr, np.ndarray):
        return arr
    if _CUPY_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def get_device(arr: Any) -> str:
    """Get device string for an array."""
    if _CUPY_AVAILABLE and isinstance(arr, cp.ndarray):
        return f"cuda:{arr.device.id}"
    return "cpu"


# =============================================================================
# Device-Aware Factory Functions
# =============================================================================

def zeros(shape: Tuple[int, ...], dtype=np.float32, device: str = 'cpu') -> Any:
    """Create array of zeros on specified device."""
    if device == 'cpu':
        return np.zeros(shape, dtype=dtype)
    else:
        if not _CUPY_AVAILABLE:
            raise RuntimeError("CuPy not installed")
        idx = int(device.split(':')[1]) if ':' in device else 0
        with cp.cuda.Device(idx):
            return cp.zeros(shape, dtype=dtype)


def ones(shape: Tuple[int, ...], dtype=np.float32, device: str = 'cpu') -> Any:
    """Create array of ones on specified device."""
    if device == 'cpu':
        return np.ones(shape, dtype=dtype)
    else:
        if not _CUPY_AVAILABLE:
            raise RuntimeError("CuPy not installed")
        idx = int(device.split(':')[1]) if ':' in device else 0
        with cp.cuda.Device(idx):
            return cp.ones(shape, dtype=dtype)


def randn(*shape, dtype=np.float32, device: str = 'cpu') -> Any:
    """Create array with random normal values on specified device."""
    if device == 'cpu':
        return np.random.randn(*shape).astype(dtype)
    else:
        if not _CUPY_AVAILABLE:
            raise RuntimeError("CuPy not installed")
        idx = int(device.split(':')[1]) if ':' in device else 0
        with cp.cuda.Device(idx):
            return cp.random.randn(*shape, dtype=dtype)


def rand(*shape, dtype=np.float32, device: str = 'cpu') -> Any:
    """Create array with random uniform [0,1) values on specified device."""
    if device == 'cpu':
        if len(shape) == 0:
            return np.array(np.random.rand(), dtype=dtype)
        return np.random.rand(*shape).astype(dtype)
    else:
        if not _CUPY_AVAILABLE:
            raise RuntimeError("CuPy not installed")
        idx = int(device.split(':')[1]) if ':' in device else 0
        with cp.cuda.Device(idx):
            if len(shape) == 0:
                return cp.array(cp.random.rand(), dtype=dtype)
            return cp.random.rand(*shape, dtype=dtype)


def eye(n: int, dtype=np.float32, device: str = 'cpu') -> Any:
    """Create identity matrix on specified device."""
    if device == 'cpu':
        return np.eye(n, dtype=dtype)
    else:
        if not _CUPY_AVAILABLE:
            raise RuntimeError("CuPy not installed")
        idx = int(device.split(':')[1]) if ':' in device else 0
        with cp.cuda.Device(idx):
            return cp.eye(n, dtype=dtype)


def from_numpy(arr: np.ndarray, device: str = 'cpu') -> Any:
    """Create array from numpy on specified device."""
    return to_device(arr, device)


# =============================================================================
# Linear Algebra (GPU-accelerated when available)
# =============================================================================

def matmul(a: Any, b: Any) -> Any:
    """Matrix multiplication."""
    xp = get_array_module(a)
    return xp.matmul(a, b)


def dot(a: Any, b: Any) -> Any:
    """Dot product."""
    xp = get_array_module(a)
    return xp.dot(a.flatten(), b.flatten())


def norm(a: Any, axis: Optional[int] = None, keepdims: bool = False) -> Any:
    """L2 norm."""
    xp = get_array_module(a)
    return xp.linalg.norm(a, axis=axis, keepdims=keepdims)


def eigh(a: Any) -> Tuple[Any, Any]:
    """Eigendecomposition of symmetric matrix."""
    xp = get_array_module(a)
    return xp.linalg.eigh(a)


def solve(a: Any, b: Any) -> Any:
    """Solve linear system Ax = b."""
    xp = get_array_module(a)
    return xp.linalg.solve(a, b)


def inv(a: Any) -> Any:
    """Matrix inverse."""
    xp = get_array_module(a)
    return xp.linalg.inv(a)


def cholesky(a: Any) -> Any:
    """Cholesky decomposition."""
    xp = get_array_module(a)
    return xp.linalg.cholesky(a)


# =============================================================================
# Math Functions
# =============================================================================

def sqrt(a: Any) -> Any:
    xp = get_array_module(a)
    return xp.sqrt(a)

def exp(a: Any) -> Any:
    xp = get_array_module(a)
    return xp.exp(a)

def log(a: Any) -> Any:
    xp = get_array_module(a)
    return xp.log(a)

def sin(a: Any) -> Any:
    xp = get_array_module(a)
    return xp.sin(a)

def cos(a: Any) -> Any:
    xp = get_array_module(a)
    return xp.cos(a)

def tanh(a: Any) -> Any:
    xp = get_array_module(a)
    return xp.tanh(a)

def sinh(a: Any) -> Any:
    xp = get_array_module(a)
    return xp.sinh(a)

def cosh(a: Any) -> Any:
    xp = get_array_module(a)
    return xp.cosh(a)

def arcsinh(a: Any) -> Any:
    xp = get_array_module(a)
    return xp.arcsinh(a)

def arccosh(a: Any) -> Any:
    xp = get_array_module(a)
    return xp.arccosh(a)

def arctanh(a: Any) -> Any:
    xp = get_array_module(a)
    return xp.arctanh(a)

def clip(a: Any, a_min: Optional[float], a_max: Optional[float]) -> Any:
    xp = get_array_module(a)
    return xp.clip(a, a_min, a_max)


# =============================================================================
# Matrix Functions (for SPD manifold)
# =============================================================================

def expm(a: Any) -> Any:
    """Matrix exponential."""
    xp = get_array_module(a)
    if xp == np:
        from scipy.linalg import expm as scipy_expm
        return scipy_expm(a)
    else:
        # CuPy: use eigendecomposition
        eigenvalues, eigenvectors = xp.linalg.eigh(a)
        exp_eigenvalues = xp.exp(eigenvalues)
        return eigenvectors @ xp.diag(exp_eigenvalues) @ eigenvectors.T


def logm(a: Any) -> Any:
    """Matrix logarithm."""
    xp = get_array_module(a)
    if xp == np:
        from scipy.linalg import logm as scipy_logm
        return scipy_logm(a)
    else:
        # CuPy: use eigendecomposition
        eigenvalues, eigenvectors = xp.linalg.eigh(a)
        log_eigenvalues = xp.log(eigenvalues)
        return eigenvectors @ xp.diag(log_eigenvalues) @ eigenvectors.T


def sqrtm(a: Any) -> Any:
    """Matrix square root."""
    xp = get_array_module(a)
    if xp == np:
        from scipy.linalg import sqrtm as scipy_sqrtm
        return scipy_sqrtm(a)
    else:
        # CuPy: use eigendecomposition
        eigenvalues, eigenvectors = xp.linalg.eigh(a)
        sqrt_eigenvalues = xp.sqrt(eigenvalues)
        return eigenvectors @ xp.diag(sqrt_eigenvalues) @ eigenvectors.T


# =============================================================================
# Info
# =============================================================================

def info() -> str:
    """Print backend info."""
    lines = [
        "DavisTensor Array API",
        "=" * 40,
        f"NumPy version: {np.__version__}",
        f"CuPy available: {_CUPY_AVAILABLE}",
        f"GPU available: {_GPU_AVAILABLE}",
    ]
    if _CUPY_AVAILABLE:
        lines.append(f"CuPy version: {cp.__version__}")
        if _GPU_AVAILABLE:
            lines.append(f"CUDA device: {cp.cuda.runtime.getDeviceCount()} GPU(s)")
    return "\n".join(lines)


if __name__ == "__main__":
    print(info())
    print()
    
    # Test CPU
    x = randn(3, 4, device='cpu')
    print(f"CPU array shape: {x.shape}, device: {get_device(x)}")
    
    # Test GPU if available
    if gpu_available():
        y = randn(3, 4, device='cuda')
        print(f"GPU array shape: {y.shape}, device: {get_device(y)}")
        
        # Test move
        y_cpu = to_device(y, 'cpu')
        print(f"Moved to CPU: {get_device(y_cpu)}")
