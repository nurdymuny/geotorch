# DavisTensor

A geometry-native tensor library where tensors know their own geometry.

## Overview

DavisTensor is a from-scratch tensor library where Riemannian geometry is built into the DNA, not bolted on. Every tensor knows what manifold it lives on. Every operation respects curvature. Every gradient is automatically a tangent vector.

## Quick Start

```python
import davistensor as dt

# Create a Euclidean manifold
E = dt.Euclidean(64)

# Create points on the manifold
x = dt.randn(manifold=E)
y = dt.randn(manifold=E)

# Geometric operations are natural
d = x.distance(y)              # Geodesic distance
v = x.log(y)                   # Tangent vector from x to y
z = x.exp(v)                   # Move from x along v

# Arithmetic is geometric
w = x + v                      # exp(x, v)
v2 = y - x                     # log(x, y)

# Type-safe tangent vectors
vx = x.random_tangent()
vy = y.random_tangent()
# vx + vy  # TypeError: different base points!
vy_transported = vx.transport_to(y)
vz = vy + vy_transported       # OK: both at y
```

## Features

- **Type Safety**: `ManifoldTensor`, `TangentTensor`, and `Scalar` types prevent geometric errors
- **Automatic Type Checking**: Adding tangent vectors at different points raises `TypeError`
- **Geometric Arithmetic**: `point + tangent = exp`, `point - point = log`
- **NumPy Backend**: Pure NumPy implementation (Phase 1)

## Installation

From the repository root:

```bash
pip install -e davistensor/
```

Or use it directly from the parent geotorch package:

```python
import davistensor as dt
```

## API Reference

### Types

- `ManifoldTensor` - Point on a Riemannian manifold
- `TangentTensor` - Vector in tangent space at a specific point
- `Scalar` - Manifold-independent scalar value

### Manifolds

- `Euclidean(dim)` - Flat Euclidean space ℝⁿ

### Factory Functions

- `randn(manifold=...)` - Random point on manifold
- `origin(manifold)` - Origin/identity point
- `tangent_randn(base)` - Random tangent vector at base point
- `tangent_zeros(base)` - Zero tangent vector at base point

### Core (Advanced)

- `TensorCore` - Underlying tensor data structure
- `Storage` - Raw memory buffer
- `Device` - Device abstraction (CPU/CUDA)
- `DType` - Data type enumeration
- `GeometricType` - Geometric type enumeration

## Testing

```bash
# From repository root
python run_tests.py

# Or with pytest directly
pytest tests/test_davistensor.py -v
```

## Roadmap

- **Phase 1 (Current)**: Core foundation with Euclidean manifold ✅
- **Phase 2**: Sphere and Hyperbolic manifolds
- **Phase 3**: Geometric autograd engine
- **Phase 4**: Neural network layers
- **Phase 5**: GPU backend and compiler optimizations

## Design Philosophy

See [DAVISTENSOR_SPEC.md](../DAVISTENSOR_SPEC.md) for the complete specification and design philosophy.

## License

Apache License 2.0
