# DavisTensor: A Geometry-Native Tensor Library

**Version 0.1.0 - Phase 1 Foundation**

> *"What if tensors knew their own geometry?"*

DavisTensor is a from-scratch tensor library where Riemannian geometry is not bolted on, but **built into the DNA**. Every tensor knows what manifold it lives on. Every operation respects curvature.

## Features

### Phase 1 (Current)

- ✅ **Type-safe tangent vectors**: Adding TangentTensors at different base points raises TypeError
- ✅ **Geometric arithmetic**: `point + tangent` = exp map, `point - point` = log map  
- ✅ **Manifold-aware tensors**: Every ManifoldTensor knows its manifold
- ✅ **Euclidean baseline**: Fully working Euclidean manifold for testing

## Installation

```bash
# From the repository root
pip install -e .
```

## Quick Start

```python
import davistensor as dt

# Create points on a Euclidean manifold
E = dt.Euclidean(10)
x = dt.randn(manifold=E)
y = dt.randn(manifold=E)

# Geometric operations are natural
d = x.distance(y)              # Geodesic distance (Scalar)
v = x.log(y)                   # Tangent vector from x to y (TangentTensor)
z = x.exp(v)                   # Should equal y (ManifoldTensor)

# Arithmetic is geometric
w = x + v                      # exp(x, v) - move x in direction v
v2 = y - x                     # log(x, y) - tangent from x to y
```

## Core Concepts

### ManifoldTensor

A tensor that lives on a Riemannian manifold:

```python
x = dt.randn(manifold=dt.Euclidean(64))
print(x.manifold)  # Euclidean(64)
print(x.shape)     # (64,)
```

### TangentTensor

A tangent vector at a specific point on a manifold:

```python
x = dt.randn(manifold=dt.Euclidean(10))
v = dt.tangent_randn(x)

# v knows its base point
print(v.base_point)  # x

# Type safety: can only add tangent vectors at same point
y = dt.randn(manifold=dt.Euclidean(10))
w = dt.tangent_randn(y)

v + w  # TypeError! Different base points
```

### Scalar

A scalar value independent of any manifold:

```python
x = dt.randn(manifold=dt.Euclidean(10))
y = dt.randn(manifold=dt.Euclidean(10))

d = x.distance(y)  # Scalar
print(d.item())    # 4.23...
```

## Geometric Arithmetic

DavisTensor makes geometric operations natural through operator overloading:

```python
# Point + Tangent = Exponential Map
x = dt.randn(manifold=E)
v = dt.tangent_randn(x)
y = x + v  # Same as x.exp(v)

# Point - Point = Logarithmic Map  
x = dt.randn(manifold=E)
y = dt.randn(manifold=E)
v = y - x  # Same as x.log(y), tangent vector at x pointing to y
```

## Factory Functions

```python
# Create random point on manifold
x = dt.randn(5, 10, manifold=dt.Euclidean(64))  # Shape: (5, 10, 64)

# Create origin point
o = dt.origin(manifold=dt.Euclidean(10))  # Zero vector

# Create random tangent vector at a point
x = dt.randn(manifold=dt.Euclidean(10))
v = dt.tangent_randn(x)  # Random tangent at x

# Create zero tangent vector
v_zero = dt.tangent_zeros(x)  # Zero tangent at x
```

## Architecture

```
davistensor/
├── __init__.py          # Public API
├── tensor.py            # ManifoldTensor, TangentTensor, Scalar
├── core/
│   ├── __init__.py
│   └── storage.py       # TensorCore, Storage, low-level ops
└── manifolds/
    ├── __init__.py
    └── base.py          # Manifold ABC, Euclidean
```

## Testing

```bash
# Run all tests
python run_tests.py

# Or with pytest directly
pytest tests/test_davistensor.py -v
```

## Comparison with PyTorch

| Feature | PyTorch | DavisTensor |
|---------|---------|-------------|
| Geometry | Bolted on via libraries | Native and built-in |
| Type safety | Points and tangents both `Tensor` | Separate `ManifoldTensor` and `TangentTensor` |
| Arithmetic | `x + v` undefined geometrically | `x + v` = exponential map |
| Gradients | In ambient space | In tangent space (future) |

## What's Next?

### Phase 2: More Manifolds
- Sphere (S^n)
- Hyperbolic space (H^n)
- Product manifolds

### Phase 3: Autograd
- Geometry-aware gradients
- Automatic tangent space projection
- Riemannian optimizers

### Phase 4: GPU Support
- CUDA kernels for geometric ops
- Efficient parallel transport

## Citation

```bibtex
@software{davistensor2025,
  title={DavisTensor: A Geometry-Native Tensor Library},
  author={Davis, Bee Rosa},
  year={2025},
  version={0.1.0},
  note={Phase 1: Foundation}
}
```

## License

Apache License 2.0

---

**DavisTensor**: Where geometry is not a constraint to enforce, but the natural language of computation.
