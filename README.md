# GeoTorch: Manifold-Native Deep Learning Framework

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

GeoTorch is a Riemannian deep learning framework that extends PyTorch with native support for manifold-valued parameters, geodesic optimization, and geometric operations.

---

## 🆕 DavisTensor: Geometry-Native Tensor Library

**New in v0.2.0:** DavisTensor Phase 1 - A from-scratch tensor library where tensors know their own geometry.

### What is DavisTensor?

DavisTensor is a geometry-native tensor library where Riemannian geometry is built into the DNA, not bolted on. Every tensor knows what manifold it lives on. Every operation respects curvature. Every gradient is automatically a tangent vector.

### Vision

> **"What if tensors knew their own geometry?"**

Unlike PyTorch where manifolds are constraints to enforce, DavisTensor treats manifolds as the **natural habitat** of data.

### Quick Start with DavisTensor

```python
import davistensor as dt

# Create a Euclidean manifold
E = dt.Euclidean(64)

# Create points on the manifold
x = dt.randn(manifold=E)
y = dt.randn(manifold=E)

# Geometric operations are natural
d = x.distance(y)              # Geodesic distance (returns Scalar)
v = x.log(y)                   # Tangent vector from x to y
z = x.exp(v)                   # Move from x along v (same as y)

# Arithmetic is geometric
w = x + v                      # Point + Tangent = exp(x, v)
v2 = y - x                     # Point - Point = log(x, y)

# Type-safe tangent vectors
vx = x.random_tangent()        # Tangent at x
vy = y.random_tangent()        # Tangent at y
# vx + vy  # TypeError: different base points!
vy_transported = vx.transport_to(y)  # Parallel transport
vz = vy + vy_transported       # Now we can add them
```

### Key Features

- **Type Safety**: `ManifoldTensor`, `TangentTensor`, and `Scalar` prevent geometric errors
- **Automatic Transport**: Tangent vectors can't be added at different points without explicit transport
- **Geometric Arithmetic**: `point + tangent = exp(point, tangent)`, `point - point = log`
- **Pure NumPy Backend**: No PyTorch dependency (Phase 1)

### DavisTensor Architecture

```
davistensor/
├── __init__.py          # Public API
├── core/
│   ├── storage.py       # TensorCore, Storage, Device, DType
│   └── __init__.py
├── manifolds/
│   ├── base.py          # Manifold ABC, Euclidean
│   └── __init__.py
└── tensor.py            # ManifoldTensor, TangentTensor, Scalar
```

### Running DavisTensor Tests

```bash
# Run DavisTensor-specific tests
python run_tests.py

# Or use pytest directly
pytest tests/test_davistensor.py -v
```

### DavisTensor Roadmap

- **Phase 1 (Current)**: Core foundation with Euclidean manifold ✅
- **Phase 2**: Sphere and Hyperbolic manifolds
- **Phase 3**: Geometric autograd engine
- **Phase 4**: Neural network layers
- **Phase 5**: GPU backend and compiler optimizations

For detailed DavisTensor specification, see [DAVISTENSOR_SPEC.md](DAVISTENSOR_SPEC.md).

---

## GeoTorch Installation

### From Source

```bash
git clone https://github.com/nurdymuny/geotorch.git
cd geotorch
pip install -e .
```

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- NumPy ≥ 1.24

### Development Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import torch
from geotorch import Sphere, ManifoldTensor

# Create a sphere manifold
S = Sphere(64)  # S^63 (63-dimensional sphere in R^64)

# Generate random points on the sphere
p = S.random_point()
q = S.random_point()

# Compute geodesic distance
distance = S.distance(p, q)

# Logarithmic map: tangent vector from p to q
v = S.log(p, q)

# Exponential map: move along geodesic
new_point = S.exp(p, v)

# Use ManifoldTensor for convenient operations
p_tensor = ManifoldTensor(p, S)
q_tensor = ManifoldTensor(q, S)
distance = p_tensor.distance(q_tensor)
```

## Supported Manifolds

- **Euclidean**: Flat space ℝⁿ
- **Sphere**: Unit sphere S^{n-1} ⊂ ℝⁿ
- **Hyperbolic**: Hyperbolic space H^{n-1} (Poincaré ball and hyperboloid models)

## Testing

Run the test suite:

```bash
pytest tests/
```

## Documentation

For detailed documentation, see [spec.md](spec.md).

## License

Apache License 2.0. See LICENSE file for details.

## Citation

If you use GeoTorch in your research, please cite:

```bibtex
@software{geotorch2025,
  author = {Davis, Bee Rosa},
  title = {GeoTorch: Manifold-Native Deep Learning Framework},
  year = {2025},
  url = {https://github.com/nurdymuny/geotorch},
  note = {Riemannian deep learning with geodesic optimization and O(1) retrieval}
}
```