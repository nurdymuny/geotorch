# DavisTensor

**A geometry-native tensor library where tensors know their own geometry.**

```python
import davistensor as dt

# Tensors know their manifold
x = dt.randn(64, manifold=dt.Hyperbolic(64))
y = dt.randn(64, manifold=dt.Hyperbolic(64))

# Geometric operations are natural
d = x.distance(y)       # Geodesic distance
v = x.log(y)            # Tangent vector from x to y
z = x + v               # Same as exp(x, v) = y

# Type-safe: tangent vectors can't be added at different points!
v1 = dt.tangent_randn(x)
v2 = dt.tangent_randn(y)
v3 = v1 + v2            # TypeError! Different base points!

# Fix by parallel transport
v1_at_y = v1.transport_to(y)
v3 = v1_at_y + v2       # Now it works!
```

## Why DavisTensor?

| PyTorch + GeoOpt | DavisTensor |
|------------------|-------------|
| Geometry bolted on | Geometry built in |
| Points and tangents both `Tensor` | Type-safe distinction |
| Manual projection | Automatic |
| Ambient gradients | Tangent gradients |

## Installation

```bash
pip install davistensor
```

Or from source:
```bash
git clone https://github.com/nurdymuny/davistensor
cd davistensor
pip install -e .
```

## Quick Start

### 1. Create Points on Manifolds

```python
import davistensor as dt

# Euclidean space (flat)
E = dt.Euclidean(10)
x = dt.randn(manifold=E)

# Hyperbolic space (coming soon)
# H = dt.Hyperbolic(10)
# y = dt.randn(manifold=H)
```

### 2. Geometric Operations

```python
x = dt.randn(manifold=E)
y = dt.randn(manifold=E)

# Distance
d = x.distance(y)

# Logarithm map (tangent vector from x pointing to y)
v = x.log(y)

# Exponential map (move from x in direction v)
z = x.exp(v)  # z ≈ y

# Geodesic interpolation
mid = x.geodesic(y, 0.5)  # Midpoint
```

### 3. Type-Safe Tangent Vectors

```python
x = dt.randn(manifold=E)
y = dt.randn(manifold=E)

# Create tangent vectors
v1 = dt.tangent_randn(x)  # At x
v2 = dt.tangent_randn(y)  # At y

# Can add at same point
v3 = v1 + dt.tangent_randn(x)  # OK

# Cannot add at different points!
v4 = v1 + v2  # TypeError!

# Must transport first
v1_at_y = v1.transport_to(y)
v4 = v1_at_y + v2  # OK
```

### 4. Arithmetic is Geometric

```python
# Point + Tangent = Exponential map
y = x + v  # Same as x.exp(v)

# Point - Point = Logarithm map
v = x - y  # Same as x.log(y)
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  User API: dt.randn() / x.exp(v) / x.distance(y)       │
├─────────────────────────────────────────────────────────┤
│  Type System: ManifoldTensor / TangentTensor / Scalar  │
├─────────────────────────────────────────────────────────┤
│  Manifolds: Euclidean / Hyperbolic / Sphere / SPD      │
├─────────────────────────────────────────────────────────┤
│  Autograd: Tangent-space gradients / Parallel transport│
├─────────────────────────────────────────────────────────┤
│  Core: TensorCore / Storage / Device                   │
└─────────────────────────────────────────────────────────┘
```

## Roadmap

- [x] **Phase 1**: Core foundation (Storage, TensorCore, Euclidean)
- [ ] **Phase 2**: Geometric autograd (tangent-space gradients)
- [ ] **Phase 3**: More manifolds (Hyperbolic, Sphere, SPD)
- [ ] **Phase 4**: GPU backend (CUDA kernels)
- [ ] **Phase 5**: Neural network layers
- [ ] **Phase 6**: Compiler optimizations

## Key Concepts

### ManifoldTensor

A point on a Riemannian manifold:

```python
x = dt.randn(manifold=dt.Euclidean(10))
x.manifold   # The manifold it lives on
x.shape      # Tensor shape
x.distance(y)  # Geodesic distance
x.log(y)     # Tangent vector to y
x.exp(v)     # Move in direction v
```

### TangentTensor

A vector in the tangent space at a specific point:

```python
v = dt.tangent_randn(x)
v.base_point  # The ManifoldTensor where v lives
v.manifold    # Same as v.base_point.manifold
v.norm()      # Riemannian norm ||v||_x
v.transport_to(y)  # Parallel transport to y
```

### Scalar

A manifold-independent value:

```python
d = x.distance(y)  # Returns Scalar
n = v.norm()       # Returns Scalar
d.item()           # Python float
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{davistensor2025,
  title = {DavisTensor: A Geometry-Native Tensor Library},
  author = {Davis-Wilson Research},
  year = {2025},
  url = {https://github.com/nurdymuny/davistensor}
}
```
