# DavisTensor: Implementation Roadmap

## Current Status ✅

| Component | Status | Tests |
|-----------|--------|-------|
| `core/storage.py` | ✅ Done | Passing |
| `manifolds/base.py` (Euclidean) | ✅ Done | Passing |
| `manifolds/hyperbolic.py` | ✅ Done | Passing |
| `manifolds/sphere.py` | ✅ Done | Passing |
| `tensor.py` (ManifoldTensor, TangentTensor, Scalar) | ✅ Done | Passing |

## Next Up: Specs Ready to Implement

### 1. SPD Manifold (`specs/spd_spec.py`)

**What it is:** Symmetric Positive Definite matrices with affine-invariant metric.

**Key operations:**
```python
# Matrix operations
_sqrtm(P)      # P^{1/2} via eigendecomposition
_logm(P)       # Matrix logarithm
_expm(A)       # Matrix exponential
_powm(P, t)    # P^t

# Manifold operations
exp(P, V)      # exp_P(V) = P^{1/2} exp(P^{-1/2} V P^{-1/2}) P^{1/2}
log(P, Q)      # log_P(Q) = P^{1/2} log(P^{-1/2} Q P^{-1/2}) P^{1/2}
distance(P, Q) # ||log(P^{-1/2} Q P^{-1/2})||_F
frechet_mean() # Iterative Karcher mean
```

**Use cases:** Brain connectivity, covariance matrices, diffusion tensors

**Effort:** ~2-3 hours

---

### 2. Product Manifold (`specs/product_spec.py`)

**What it is:** Cartesian product M₁ × M₂ × ... × Mₖ

**Key operations:**
```python
split(x)       # Split into components
combine(parts) # Combine into product point
exp(x, v)      # Component-wise exp
log(x, y)      # Component-wise log
distance(x, y) # sqrt(d₁² + d₂² + ... + dₖ²)
```

**Convenience constructors:**
```python
HyperbolicSphere(hyp_dim, sphere_dim)   # Hierarchy + direction
HyperbolicEuclidean(hyp_dim, euc_dim)   # Hierarchy + features
MultiHyperbolic(dim, n_copies)          # Multi-scale hierarchies
```

**Use cases:** Knowledge graphs, multi-aspect embeddings

**Effort:** ~1-2 hours

---

### 3. Autograd Engine (`specs/autograd_spec.py`)

**What it is:** Geometry-aware automatic differentiation

**Key insight:** Gradients are TANGENT VECTORS, not ambient vectors

**Components:**
```python
# Core
GradFn           # Base class for backward functions
GradientTape     # Records operations
backward()       # Reverse-mode autodiff

# Basic backward functions
AddBackward      # z = x + y
MulBackward      # z = x * y
MatMulBackward   # z = x @ y
SumBackward      # z = sum(x)
ExpBackward      # z = exp(x)
LogBackward      # z = log(x)

# Geometric backward functions (THE HARD PART)
ManifoldExpBackward      # y = exp_x(v) - includes parallel transport
ManifoldLogBackward      # v = log_x(y)
ManifoldDistanceBackward # d = dist(x, y)
```

**Key features:**
- Automatic projection to tangent space
- Parallel transport when combining gradients
- Gradient checking for verification

**Effort:** ~4-6 hours (this is the hard one)

---

### 4. Neural Network Layers (`specs/layers_spec.py`)

**What it is:** Geometry-aware neural network building blocks

**Core classes:**
```python
# Base
Parameter           # Learnable parameter
ManifoldParameter   # Parameter constrained to manifold
Module              # Base class (like nn.Module)

# Linear layers
Linear              # Standard y = Wx + b
GeodesicLinear      # M₁ → M₂ via tangent space
ManifoldMLR         # Classification using geodesic distances

# Embeddings
Embedding           # Standard lookup table
ManifoldEmbedding   # Embeddings on manifold

# Pooling
MeanPool            # Arithmetic mean
FrechetMeanPool     # Riemannian center of mass

# Attention
GeometricAttention  # Distance-based attention

# Normalization
ManifoldBatchNorm   # BatchNorm on manifold

# Activations
ReLU                # Standard
TangentReLU         # ReLU in tangent space

# Container
Sequential          # Chain of layers
```

**Effort:** ~3-4 hours

---

## Recommended Implementation Order

```
1. SPD Manifold (straightforward, uses eigendecomposition)
   ↓
2. Product Manifold (straightforward, uses existing manifolds)
   ↓
3. Autograd - Basic ops (Add, Mul, Sum, etc.)
   ↓
4. Autograd - Geometric ops (Exp, Log, Distance backward)
   ↓
5. Layers - Parameter, Module, Linear
   ↓
6. Layers - GeodesicLinear, ManifoldEmbedding
   ↓
7. Layers - FrechetMeanPool, GeometricAttention
   ↓
8. Integration tests
```

## File Structure After Implementation

```
davistensor/
├── davistensor/
│   ├── __init__.py
│   ├── tensor.py                 # ✅ ManifoldTensor, TangentTensor, Scalar
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   └── storage.py            # ✅ TensorCore, Storage
│   │
│   ├── manifolds/
│   │   ├── __init__.py
│   │   ├── base.py               # ✅ Manifold ABC, Euclidean
│   │   ├── hyperbolic.py         # ✅ Poincaré ball
│   │   ├── sphere.py             # ✅ n-sphere
│   │   ├── spd.py                # 📋 SPD matrices
│   │   └── product.py            # 📋 Product manifolds
│   │
│   ├── autograd/
│   │   ├── __init__.py
│   │   ├── engine.py             # 📋 GradientTape, backward()
│   │   ├── grad_fn.py            # 📋 Basic backward functions
│   │   └── geometric_grad.py     # 📋 Manifold backward functions
│   │
│   └── nn/
│       ├── __init__.py
│       ├── module.py             # 📋 Module, Parameter
│       ├── linear.py             # 📋 Linear, GeodesicLinear
│       ├── embedding.py          # 📋 Embedding, ManifoldEmbedding
│       ├── pooling.py            # 📋 MeanPool, FrechetMeanPool
│       ├── attention.py          # 📋 GeometricAttention
│       ├── normalization.py      # 📋 ManifoldBatchNorm
│       └── activation.py         # 📋 ReLU, TangentReLU
│
├── specs/                        # Implementation specs
│   ├── spd_spec.py              # ✅ Created
│   ├── product_spec.py          # ✅ Created
│   ├── autograd_spec.py         # ✅ Created
│   └── layers_spec.py           # ✅ Created
│
└── tests/
    └── run_tests.py             # ✅ Test runner
```

## Quick Reference: Copy-Paste Commands

```bash
# Run all tests
python -c "from davistensor.manifolds.base import test_euclidean; test_euclidean()"
python -c "from davistensor.manifolds.hyperbolic import test_hyperbolic; test_hyperbolic()"
python -c "from davistensor.manifolds.sphere import test_sphere; test_sphere()"

# After implementing SPD:
python -c "from davistensor.manifolds.spd import test_spd; test_spd()"

# After implementing Product:
python -c "from davistensor.manifolds.product import test_product; test_product()"

# After implementing autograd:
python -c "from davistensor.autograd.engine import test_autograd; test_autograd()"

# After implementing layers:
python -c "from davistensor.nn.layers import test_layers; test_layers()"
```

## The Big Picture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Your Application                              │
│                  (Knowledge Graphs, Brain Imaging, etc.)            │
├─────────────────────────────────────────────────────────────────────┤
│                        davistensor.nn                                │
│      GeodesicLinear │ ManifoldEmbedding │ FrechetMeanPool           │
├─────────────────────────────────────────────────────────────────────┤
│                      davistensor.autograd                            │
│            Geometry-aware gradients + parallel transport             │
├─────────────────────────────────────────────────────────────────────┤
│                      davistensor.tensor                              │
│        ManifoldTensor │ TangentTensor │ Scalar (type-safe)          │
├─────────────────────────────────────────────────────────────────────┤
│                     davistensor.manifolds                            │
│     Euclidean │ Hyperbolic │ Sphere │ SPD │ Product                 │
├─────────────────────────────────────────────────────────────────────┤
│                       davistensor.core                               │
│              TensorCore │ Storage │ Device                          │
└─────────────────────────────────────────────────────────────────────┘
```

**No PyTorch. No external dependencies except numpy. Geometry is native.**
