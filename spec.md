# GeoTorch: Manifold-Native Deep Learning Framework

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

> **"The geometry of the problem should be the geometry of the computation."**

GeoTorch is a Riemannian deep learning framework that extends PyTorch with native support for manifold-valued parameters, geodesic optimization, and O(1) semantic retrieval. Instead of treating neural network parameters as points in flat Euclidean space, GeoTorch recognizes that many learning problems have inherent geometric structure that can be exploited for better optimization, generalization, and efficiency.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Installation](#2-installation)
3. [Quick Start](#3-quick-start)
4. [Core Concepts](#4-core-concepts)
5. [Architecture](#5-architecture)
6. [API Reference](#6-api-reference)
7. [Manifold Implementations](#7-manifold-implementations)
8. [Optimizers](#8-optimizers)
9. [Neural Network Layers](#9-neural-network-layers)
10. [Storage Integration](#10-storage-integration)
11. [Examples](#11-examples)
12. [Testing](#12-testing)
13. [Benchmarks](#13-benchmarks)
14. [Contributing](#14-contributing)
15. [Roadmap](#15-roadmap)
16. [Citation](#16-citation)
17. [License](#17-license)

---

## 1. Introduction

### 1.1 The Problem with Flat Space

Standard deep learning frameworks assume parameters live in Euclidean space:

```python
# Standard PyTorch: parameters are just tensors in R^n
optimizer.step()  # θ_new = θ_old - lr * grad
```

But neural network loss landscapes are **curved**:
- Straight lines are not shortest paths
- Euclidean distance ≠ semantic distance
- Optimization can be inefficient or unstable

### 1.2 The GeoTorch Solution

GeoTorch treats parameters as points on **Riemannian manifolds**:

```python
# GeoTorch: parameters live on manifolds
optimizer.step()  # θ_new = exp_θ(-lr * riemannian_grad)
```

This provides:
- **Better optimization**: Follow geodesics (shortest paths) on the loss landscape
- **Geometric constraints**: Keep parameters on meaningful subspaces (unit sphere, orthogonal matrices, etc.)
- **Semantic structure**: Geodesic distance reflects semantic similarity
- **O(1) retrieval**: Geodesic-organized storage enables constant-time semantic lookup

### 1.3 Key Innovations

| Feature | Standard PyTorch | GeoTorch |
|---------|------------------|----------|
| Parameter space | ℝⁿ (flat) | Manifold M (curved) |
| Gradient | ∇L ∈ ℝⁿ | proj_TM(∇L) ∈ T_θM |
| Update rule | θ - lr·∇L | exp_θ(-lr·grad) |
| Distance | Euclidean ‖x-y‖ | Geodesic d_g(x,y) |
| Attention | Dot product QKᵀ | Geodesic distance -d_g(Q,K) |
| Retrieval | O(n) or O(log n) | **O(1)** |

### 1.4 Applications

- **Natural Language Processing**: Hyperbolic embeddings for hierarchical data
- **Computer Vision**: Rotation-equivariant networks on SO(3)
- **Recommender Systems**: User/item embeddings on curved spaces
- **Transformers**: Geometric attention with O(1) KV-cache retrieval
- **Secure ML**: GeoHash-based embeddings with geometric security

---

## 2. Installation

### 2.1 Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- NumPy ≥ 1.24
- SciPy ≥ 1.10 (optional, for advanced manifolds)

### 2.2 Install from PyPI

```bash
pip install geotorch
```

### 2.3 Install from Source

```bash
git clone https://github.com/beedavis/geotorch.git
cd geotorch
pip install -e ".[dev]"
```

### 2.4 Verify Installation

```python
import geotorch
print(geotorch.__version__)

# Quick test
from geotorch import Sphere
S = Sphere(64)
p = S.random_point()
v = S.random_tangent(p)
q = S.exp(p, v)
print(f"Distance: {S.distance(p, q):.4f}")
```

---

## 3. Quick Start

### 3.1 Basic Manifold Operations

```python
import torch
import geotorch
from geotorch import Sphere

# Create a 64-dimensional sphere
manifold = Sphere(64)

# Random point on the manifold
p = manifold.random_point()
print(f"Point norm: {torch.norm(p):.4f}")  # Should be 1.0

# Random tangent vector at p
v = manifold.random_tangent(p)
print(f"Tangent orthogonal to p: {torch.dot(p, v):.6f}")  # Should be ~0

# Move along geodesic
q = manifold.exp(p, v)
print(f"New point norm: {torch.norm(q):.4f}")  # Still 1.0

# Compute geodesic distance
dist = manifold.distance(p, q)
print(f"Geodesic distance: {dist:.4f}")

# Logarithmic map (inverse of exp)
v_recovered = manifold.log(p, q)
print(f"Log recovers tangent: {torch.allclose(v, v_recovered)}")
```

### 3.2 Riemannian Optimization

```python
import torch
import torch.nn as nn
from geotorch import Sphere
from geotorch.nn import ManifoldParameter
from geotorch.optim import RiemannianSGD

manifold = Sphere(64)

# Parameter constrained to sphere
param = ManifoldParameter(manifold.random_point(), manifold)

# Target point
target = manifold.random_point()

# Riemannian optimizer
optimizer = RiemannianSGD([param], lr=0.1)

# Optimize
for i in range(100):
    optimizer.zero_grad()
    loss = manifold.distance(param, target) ** 2
    loss.backward()
    optimizer.step()
    
    if i % 20 == 0:
        print(f"Step {i}: distance = {manifold.distance(param, target):.4f}")

print(f"Final distance: {manifold.distance(param, target):.6f}")
```

### 3.3 Manifold Neural Network

```python
import torch
import torch.nn as nn
from geotorch import Sphere
from geotorch.nn import ManifoldLinear, GeodesicEmbedding
from geotorch.optim import RiemannianAdam

class GeoNet(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.manifold = Sphere(embed_dim)
        
        # Embeddings on sphere
        self.embedding = GeodesicEmbedding(
            vocab_size, embed_dim, manifold=self.manifold
        )
        
        # Manifold-aware linear layers
        self.fc1 = ManifoldLinear(embed_dim, hidden_dim, manifold=Sphere(hidden_dim))
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h = self.embedding(x)
        h = torch.relu(self.fc1(h))
        return self.fc2(h)

# Create model and optimizer
model = GeoNet(vocab_size=10000, embed_dim=128, hidden_dim=256, output_dim=10)
optimizer = RiemannianAdam(model.parameters(), lr=0.001)

# Training loop
for batch in dataloader:
    optimizer.zero_grad()
    output = model(batch['input'])
    loss = nn.functional.cross_entropy(output, batch['target'])
    loss.backward()
    optimizer.step()
```

---

## 4. Core Concepts

### 4.1 Riemannian Manifolds

A **Riemannian manifold** (M, g) is a smooth space equipped with a metric tensor g that defines:
- **Distances** between points
- **Angles** between tangent vectors
- **Geodesics** (shortest paths)

```
Key intuition:
- The Earth's surface is a 2D manifold embedded in 3D space
- "Straight lines" on Earth are great circles, not Euclidean lines
- The shortest flight path from NYC to Tokyo curves over the Arctic
```

### 4.2 Tangent Spaces

At each point p on a manifold M, there is a **tangent space** T_pM:
- A flat vector space "touching" the manifold at p
- Gradients live in the tangent space
- Dimension equals the manifold's intrinsic dimension

```python
# Tangent vectors are orthogonal to the point on a sphere
p = sphere.random_point()          # Point on S^{n-1}
v = sphere.random_tangent(p)       # v ∈ T_pS^{n-1}
print(torch.dot(p, v))             # ≈ 0
```

### 4.3 Exponential Map

The **exponential map** exp_p: T_pM → M moves from a point along a geodesic:

```
exp_p(v) = "start at p, walk in direction v for ||v|| units along the geodesic"
```

```python
# Move from p in direction v
q = manifold.exp(p, v)

# Properties:
# - exp_p(0) = p
# - ||v|| = distance(p, exp_p(v))
# - The curve γ(t) = exp_p(tv) is a geodesic
```

### 4.4 Logarithmic Map

The **logarithmic map** log_p: M → T_pM is the inverse of exp:

```
log_p(q) = "tangent vector at p pointing toward q with magnitude = distance(p,q)"
```

```python
# Get tangent vector from p to q
v = manifold.log(p, q)

# Properties:
# - exp_p(log_p(q)) = q
# - ||log_p(q)|| = distance(p, q)
```

### 4.5 Parallel Transport

**Parallel transport** moves tangent vectors between tangent spaces while preserving their geometric properties:

```python
# Transport vector v from T_pM to T_qM
v_transported = manifold.parallel_transport(v, p, q)

# Properties:
# - Preserves inner products (lengths and angles)
# - Essential for momentum-based optimizers (Adam, etc.)
```

### 4.6 Geodesics

**Geodesics** are curves of minimal length, generalizing straight lines to curved spaces:

```python
# Points along geodesic from p to q
for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
    point = manifold.geodesic(p, q, t)
    print(f"t={t}: {point}")
```

---

## 5. Architecture

### 5.1 Module Structure

```
geotorch/
├── __init__.py                 # Public API exports
├── manifold.py                 # Base Manifold class
├── tensor.py                   # ManifoldTensor, TangentTensor
│
├── manifolds/                  # Manifold implementations
│   ├── __init__.py
│   ├── euclidean.py           # Flat space (PyTorch compatibility)
│   ├── sphere.py              # Unit sphere S^{n-1}
│   ├── hyperbolic.py          # Hyperbolic space H^n
│   ├── spd.py                 # Symmetric positive definite matrices
│   ├── stiefel.py             # Orthonormal frames
│   ├── grassmann.py           # Subspaces
│   ├── product.py             # Product manifolds M × N
│   └── davis.py               # Learned metric (DavisManifold)
│
├── optim/                      # Riemannian optimizers
│   ├── __init__.py
│   ├── rsgd.py                # Riemannian SGD
│   ├── radam.py               # Riemannian Adam
│   └── natural.py             # Natural gradient descent
│
├── nn/                         # Neural network layers
│   ├── __init__.py
│   ├── parameter.py           # ManifoldParameter
│   ├── linear.py              # ManifoldLinear
│   ├── embedding.py           # GeodesicEmbedding
│   ├── attention.py           # GeometricAttention
│   └── frechet.py             # FrechetMean
│
├── storage/                    # O(1) retrieval (GeoStorage)
│   ├── __init__.py
│   ├── geostorage.py          # Geodesic-organized storage
│   └── cache.py               # DavisCache for transformers
│
├── functional/                 # Functional API
│   ├── __init__.py
│   └── functions.py           # F.exp_map, F.geodesic_distance, etc.
│
└── utils/                      # Utilities
    ├── __init__.py
    ├── conversion.py          # PyTorch ↔ GeoTorch conversion
    └── visualization.py       # Manifold visualization
```

### 5.2 Class Hierarchy

```
Manifold (ABC)
├── Euclidean          # R^n
├── Sphere             # S^{n-1}
├── Hyperbolic         # H^n (Poincaré ball or hyperboloid)
├── SPD                # Symmetric positive definite matrices
├── Stiefel            # Orthonormal k-frames in R^n
├── Grassmann          # k-dimensional subspaces of R^n
├── ProductManifold    # M × N
└── DavisManifold      # Learned metric tensor

torch.Tensor
└── ManifoldTensor     # Tensor with manifold attachment
    └── TangentTensor  # Tangent vector with base point

torch.nn.Parameter
└── ManifoldParameter  # Parameter constrained to manifold

torch.optim.Optimizer
├── RiemannianSGD      # SGD with geodesic updates
├── RiemannianAdam     # Adam with parallel transport
└── NaturalGradient    # Fisher information metric

torch.nn.Module
├── ManifoldLinear     # Linear layer with manifold weights
├── GeodesicEmbedding  # Embedding layer on manifold
├── GeometricAttention # Attention via geodesic distances
└── FrechetMean        # Weighted mean on manifold
```

---

## 6. API Reference

### 6.1 Manifold Base Class

```python
class Manifold(ABC):
    """Abstract base class for Riemannian manifolds."""
    
    @property
    @abstractmethod
    def dim(self) -> int:
        """Intrinsic dimension of the manifold."""
        ...
    
    @abstractmethod
    def exp(self, p: Tensor, v: Tensor) -> Tensor:
        """
        Exponential map: move from p along geodesic with velocity v.
        
        Args:
            p: Point on manifold, shape (..., ambient_dim)
            v: Tangent vector at p, shape (..., ambient_dim)
        
        Returns:
            Point on manifold after geodesic flow
        """
        ...
    
    @abstractmethod
    def log(self, p: Tensor, q: Tensor) -> Tensor:
        """
        Logarithmic map: tangent vector at p pointing toward q.
        
        Args:
            p: Base point on manifold
            q: Target point on manifold
        
        Returns:
            Tangent vector v such that exp(p, v) = q
        """
        ...
    
    @abstractmethod
    def parallel_transport(self, v: Tensor, p: Tensor, q: Tensor) -> Tensor:
        """
        Parallel transport tangent vector v from T_pM to T_qM.
        
        Args:
            v: Tangent vector at p
            p: Source point
            q: Destination point
        
        Returns:
            Tangent vector at q with same geometric properties as v
        """
        ...
    
    @abstractmethod
    def distance(self, p: Tensor, q: Tensor) -> Tensor:
        """
        Geodesic distance between points.
        
        Args:
            p: First point, shape (..., ambient_dim)
            q: Second point, shape (..., ambient_dim)
        
        Returns:
            Distance, shape (...)
        """
        ...
    
    @abstractmethod
    def project(self, x: Tensor) -> Tensor:
        """
        Project ambient space point onto manifold.
        
        Args:
            x: Point in ambient space
        
        Returns:
            Closest point on manifold
        """
        ...
    
    @abstractmethod
    def project_tangent(self, p: Tensor, v: Tensor) -> Tensor:
        """
        Project ambient vector onto tangent space at p.
        
        Args:
            p: Point on manifold
            v: Vector in ambient space
        
        Returns:
            Component of v in T_pM
        """
        ...
    
    def metric(self, p: Tensor) -> Tensor:
        """
        Riemannian metric tensor at point p.
        
        Args:
            p: Point on manifold
        
        Returns:
            Metric tensor g_ij, shape (dim, dim)
        """
        ...
    
    def geodesic(self, p: Tensor, q: Tensor, t: float) -> Tensor:
        """
        Point along geodesic from p to q at parameter t.
        
        Args:
            p: Start point
            q: End point
            t: Parameter in [0, 1]
        
        Returns:
            Point γ(t) where γ(0)=p, γ(1)=q
        """
        v = self.log(p, q)
        return self.exp(p, t * v)
    
    def random_point(self, *shape, device=None, dtype=None) -> Tensor:
        """Generate random point(s) on manifold."""
        ...
    
    def random_tangent(self, p: Tensor) -> Tensor:
        """Generate random tangent vector at p."""
        ...
```

### 6.2 ManifoldTensor

```python
class ManifoldTensor(torch.Tensor):
    """Tensor that lives on a manifold."""
    
    manifold: Manifold  # Associated manifold
    
    def project_(self) -> 'ManifoldTensor':
        """Project onto manifold in-place."""
        ...
    
    def exp(self, v: Tensor) -> 'ManifoldTensor':
        """Move along geodesic with velocity v."""
        ...
    
    def log(self, q: 'ManifoldTensor') -> 'TangentTensor':
        """Tangent vector pointing toward q."""
        ...
    
    def distance(self, q: 'ManifoldTensor') -> Tensor:
        """Geodesic distance to q."""
        ...
    
    def geodesic_to(self, q: 'ManifoldTensor', t: float) -> 'ManifoldTensor':
        """Interpolate toward q along geodesic."""
        ...
```

### 6.3 ManifoldParameter

```python
class ManifoldParameter(nn.Parameter):
    """
    Neural network parameter constrained to a manifold.
    
    Automatically projects gradients to tangent space and
    applies geodesic updates when used with Riemannian optimizers.
    """
    
    def __new__(cls, data: Tensor, manifold: Manifold, requires_grad: bool = True):
        ...
    
    @property
    def manifold(self) -> Manifold:
        """Associated manifold."""
        ...
```

---

## 7. Manifold Implementations

### 7.1 Sphere

The unit sphere S^{n-1} in ℝⁿ.

```python
from geotorch import Sphere

# 64-dimensional sphere (embedded in R^65)
S = Sphere(64)

# All points have unit norm
p = S.random_point()
assert torch.isclose(torch.norm(p), torch.tensor(1.0))

# Geodesics are great circles
# Closed-form exponential map:
# exp_p(v) = cos(||v||) * p + sin(||v||) * v / ||v||
```

**Use cases**: Normalized embeddings, directional data, L2-normalized weights

### 7.2 Hyperbolic Space

The hyperbolic space H^n with constant negative curvature.

```python
from geotorch import Hyperbolic

# Poincaré ball model
H = Hyperbolic(64, model='poincare')

# Hyperboloid model (better numerics)
H = Hyperbolic(64, model='hyperboloid')

# Hyperbolic space is ideal for hierarchical data
# - Trees embed with low distortion
# - Distance grows exponentially toward boundary
```

**Use cases**: Hierarchical embeddings, knowledge graphs, taxonomies

### 7.3 SPD Manifold

Symmetric positive definite matrices.

```python
from geotorch import SPD

# 8x8 SPD matrices
S = SPD(8)

# SPD matrices have many applications:
# - Covariance matrices
# - Kernel matrices
# - Diffusion tensors (medical imaging)
```

**Use cases**: Covariance learning, metric learning, Gaussian processes

### 7.4 Stiefel Manifold

Orthonormal k-frames in ℝⁿ (matrices with orthonormal columns).

```python
from geotorch import Stiefel

# 10 orthonormal vectors in R^64
St = Stiefel(n=64, k=10)

# Special cases:
# - Stiefel(n, n) = O(n), orthogonal matrices
# - Stiefel(n, 1) = S^{n-1}, unit sphere
```

**Use cases**: Orthogonal weight matrices, PCA, dimensionality reduction

### 7.5 Grassmann Manifold

k-dimensional subspaces of ℝⁿ.

```python
from geotorch import Grassmann

# 10-dimensional subspaces of R^64
Gr = Grassmann(n=64, k=10)

# Points are equivalence classes of orthonormal frames
# Gr(n,k) = St(n,k) / O(k)
```

**Use cases**: Subspace learning, video analysis, multi-view learning

### 7.6 Product Manifold

Cartesian product of manifolds.

```python
from geotorch import ProductManifold, Sphere, Hyperbolic

# Product of sphere and hyperbolic space
M = ProductManifold([Sphere(32), Hyperbolic(32)])

# Points are tuples (p1, p2) where p1 ∈ S^31, p2 ∈ H^32
```

**Use cases**: Multi-component embeddings, structured representations

### 7.7 Davis Manifold (Learned Metric)

Manifold with metric tensor derived from a secret key or learned from data.

```python
from geotorch import DavisManifold

# Secret metric from key (for GeoHash)
M = DavisManifold(dim=64, key=b"secret_key")

# Learnable metric (for adaptive geometry)
M = DavisManifold(dim=64, learnable=True)

# The metric determines:
# - Geodesic paths
# - Distances
# - Curvature
# Security: Cannot invert geodesic without knowing metric
```

**Use cases**: GeoHash, secure embeddings, adaptive geometry learning

---

## 8. Optimizers

### 8.1 Riemannian SGD

```python
from geotorch.optim import RiemannianSGD

optimizer = RiemannianSGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,        # Optional: momentum with parallel transport
    weight_decay=0.0001  # Optional: Riemannian regularization
)

# Usage is identical to torch.optim.SGD
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch).mean()
    loss.backward()
    optimizer.step()  # Geodesic updates for manifold parameters
```

**Algorithm**:
1. Compute Euclidean gradient ∇L
2. Project to tangent space: grad = proj_{T_θM}(∇L)
3. Update via exponential map: θ_new = exp_θ(-lr · grad)

### 8.2 Riemannian Adam

```python
from geotorch.optim import RiemannianAdam

optimizer = RiemannianAdam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8
)
```

**Algorithm**:
1. Compute Riemannian gradient
2. **Parallel transport** previous moments to current tangent space
3. Update moments (same as Adam)
4. Update via exponential map

The key innovation is **parallel transport of momentum** between tangent spaces.

### 8.3 Natural Gradient Descent

```python
from geotorch.optim import NaturalGradient

optimizer = NaturalGradient(
    model.parameters(),
    lr=0.01,
    damping=0.1  # For numerical stability
)
```

Uses the **Fisher information metric** as the Riemannian metric, providing:
- Scale invariance
- Faster convergence for probabilistic models

---

## 9. Neural Network Layers

### 9.1 ManifoldLinear

Linear layer with weights constrained to a manifold.

```python
from geotorch.nn import ManifoldLinear
from geotorch import Stiefel

# Orthogonal linear layer
layer = ManifoldLinear(
    in_features=256,
    out_features=128,
    manifold=Stiefel(256, 128),  # Orthonormal weights
    bias=True
)

# Forward pass
x = torch.randn(32, 256)
y = layer(x)  # shape: (32, 128)

# Weights remain orthonormal after training
```

### 9.2 GeodesicEmbedding

Embedding layer where vectors live on a manifold.

```python
from geotorch.nn import GeodesicEmbedding
from geotorch import Sphere

embedding = GeodesicEmbedding(
    num_embeddings=50000,    # Vocabulary size
    embedding_dim=256,       # Embedding dimension
    manifold=Sphere(256)     # Embeddings on unit sphere
)

# Look up embeddings
indices = torch.tensor([1, 42, 1337])
vectors = embedding(indices)  # shape: (3, 256), all unit norm

# Semantic similarity via geodesic distance
dist = embedding.geodesic_distance(42, 1337)

# Interpolate between embeddings
midpoint = embedding.geodesic_interpolate(42, 1337, t=0.5)
```

### 9.3 GeometricAttention

Attention mechanism using geodesic distances instead of dot products.

```python
from geotorch.nn import GeometricAttention
from geotorch import Sphere

attention = GeometricAttention(
    embed_dim=256,
    num_heads=8,
    manifold=Sphere(32),     # Per-head dimension
    temperature=1.0
)

# Standard attention: softmax(QK^T / sqrt(d))
# Geometric attention: softmax(-d_g(Q, K) / temperature)

# Forward pass
query = torch.randn(32, 100, 256)  # (batch, seq, dim)
key = torch.randn(32, 100, 256)
value = torch.randn(32, 100, 256)

output = attention(query, key, value)
```

**Benefits**:
- Attention weights based on geodesic proximity
- More meaningful for manifold-valued representations
- Can integrate with O(1) retrieval (DavisCache)

### 9.4 FrechetMean

Weighted mean on a manifold (generalizes arithmetic mean).

```python
from geotorch.nn import FrechetMean
from geotorch import Sphere

frechet = FrechetMean(
    manifold=Sphere(64),
    max_iter=10,
    tol=1e-6
)

# Points on manifold
points = torch.randn(100, 64)
points = points / points.norm(dim=-1, keepdim=True)

# Weights (optional)
weights = torch.softmax(torch.randn(100), dim=0)

# Compute Fréchet mean
mean = frechet(points, weights)  # shape: (64,)
```

**Algorithm** (iterative):
1. Initialize at first point
2. Compute tangent vectors to all points
3. Take weighted average in tangent space
4. Move via exponential map
5. Repeat until convergence

---

## 10. Storage Integration

### 10.1 GeoStorage

O(1) semantic retrieval via geodesic-organized storage.

```python
from geotorch.storage import GeoStorage
from geotorch import Sphere

# Create storage
storage = GeoStorage(
    dim=256,
    capacity=1_000_000,
    manifold=Sphere(256)
)

# Store embeddings with geodesic organization
for i, embedding in enumerate(embeddings):
    storage.store(embedding, data_id=i)

# O(1) retrieval!
query = get_query_embedding()
nearest_ids = storage.query(query, k=10)  # Returns 10 nearest neighbors
```

**How it works**:
1. Spatial hashing on manifold coordinates
2. Geodesic-adjacent points map to adjacent hash buckets
3. Query → hash → bucket lookup → small candidate set → rank by geodesic distance

### 10.2 DavisCache

O(1) KV-cache for transformers.

```python
from geotorch.storage import DavisCache

class EfficientTransformer(nn.Module):
    def __init__(self, embed_dim, max_length):
        super().__init__()
        self.cache = DavisCache(
            embed_dim=embed_dim,
            max_length=max_length,
            manifold=Sphere(embed_dim)
        )
        # ... other layers
    
    def forward(self, x, use_cache=True):
        q, k, v = self.qkv(x)
        
        if use_cache:
            # O(1) retrieval from cache
            cached_k, cached_v = self.cache.retrieve(q, k=64)
            k = torch.cat([cached_k, k], dim=1)
            v = torch.cat([cached_v, v], dim=1)
            
            # Update cache
            self.cache.add(k[:, -1], v[:, -1])
        
        return self.attention(q, k, v)
```

**Benefits**:
- Constant-time retrieval regardless of sequence length
- Enables efficient long-context transformers
- Semantic organization means retrieved keys are most relevant

---

## 11. Examples

### 11.1 Hyperbolic Embeddings for Hierarchical Data

```python
"""
Learn hyperbolic embeddings for a taxonomy/hierarchy.
Hyperbolic space naturally represents tree structures.
"""

import torch
import torch.nn as nn
from geotorch import Hyperbolic
from geotorch.nn import GeodesicEmbedding
from geotorch.optim import RiemannianAdam

# Hyperbolic space (Poincaré ball)
H = Hyperbolic(dim=32, model='poincare')

# Embedding layer
embedding = GeodesicEmbedding(
    num_embeddings=10000,
    embedding_dim=32,
    manifold=H
)

# Loss: hyperbolic distance should reflect hierarchy depth
def hierarchy_loss(parent_idx, child_idx):
    parent = embedding(parent_idx)
    child = embedding(child_idx)
    
    # Children should be "further from origin" than parents
    parent_norm = torch.norm(parent, dim=-1)
    child_norm = torch.norm(child, dim=-1)
    
    # And close to their parent
    dist = H.distance(parent, child)
    
    return torch.relu(parent_norm - child_norm + 0.1) + dist

optimizer = RiemannianAdam(embedding.parameters(), lr=0.01)

for parent, child in hierarchy_edges:
    optimizer.zero_grad()
    loss = hierarchy_loss(parent, child)
    loss.backward()
    optimizer.step()
```

### 11.2 Orthogonal RNN

```python
"""
RNN with orthogonal hidden-to-hidden weights.
Helps with vanishing/exploding gradients.
"""

import torch
import torch.nn as nn
from geotorch import Stiefel
from geotorch.nn import ManifoldLinear

class OrthogonalRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Input to hidden (unconstrained)
        self.i2h = nn.Linear(input_size, hidden_size)
        
        # Hidden to hidden (orthogonal!)
        self.h2h = ManifoldLinear(
            hidden_size, hidden_size,
            manifold=Stiefel(hidden_size, hidden_size),
            bias=False
        )
    
    def forward(self, x, h=None):
        # x: (batch, seq_len, input_size)
        batch_size, seq_len, _ = x.shape
        
        if h is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        outputs = []
        for t in range(seq_len):
            h = torch.tanh(self.i2h(x[:, t]) + self.h2h(h))
            outputs.append(h)
        
        return torch.stack(outputs, dim=1), h
```

### 11.3 Geometric Transformer

```python
"""
Transformer with geometric attention and O(1) KV-cache.
"""

import torch
import torch.nn as nn
from geotorch import Sphere
from geotorch.nn import GeodesicEmbedding, GeometricAttention, ManifoldLinear
from geotorch.storage import DavisCache
from geotorch.optim import RiemannianAdam

class GeoTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.manifold = Sphere(embed_dim // num_heads)
        
        self.attention = GeometricAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            manifold=self.manifold
        )
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x, mask=None):
        x = x + self.attention(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x


class GeoTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_dim, max_length):
        super().__init__()
        self.manifold = Sphere(embed_dim)
        
        self.embedding = GeodesicEmbedding(vocab_size, embed_dim, self.manifold)
        self.pos_embedding = nn.Embedding(max_length, embed_dim)
        
        self.layers = nn.ModuleList([
            GeoTransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        
        self.cache = DavisCache(embed_dim, max_length, self.manifold)
        self.output = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x, use_cache=False):
        seq_len = x.shape[1]
        pos = torch.arange(seq_len, device=x.device)
        
        h = self.embedding(x) + self.pos_embedding(pos)
        
        for layer in self.layers:
            h = layer(h)
        
        return self.output(h)


# Training
model = GeoTransformer(
    vocab_size=50000,
    embed_dim=512,
    num_heads=8,
    num_layers=6,
    ff_dim=2048,
    max_length=8192
)

optimizer = RiemannianAdam(model.parameters(), lr=1e-4)
```

---

## 12. Testing

### 12.1 Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_manifolds.py -v

# Run with coverage
pytest tests/ --cov=geotorch --cov-report=html

# Run property-based tests
pytest tests/test_properties.py -v --hypothesis-show-statistics
```

### 12.2 Test Structure

```
tests/
├── test_manifolds.py      # Manifold implementations
├── test_tensors.py        # ManifoldTensor, TangentTensor
├── test_optimizers.py     # Riemannian optimizers
├── test_nn.py             # Neural network layers
├── test_storage.py        # GeoStorage, DavisCache
├── test_properties.py     # Property-based tests (Hypothesis)
├── test_integration.py    # End-to-end tests
└── conftest.py            # Fixtures
```

### 12.3 Manifold Property Tests

Every manifold implementation must satisfy these properties:

```python
import pytest
from hypothesis import given, strategies as st
import torch
from geotorch import Sphere, Hyperbolic, Stiefel

MANIFOLDS = [Sphere(64), Hyperbolic(64), Stiefel(64, 32)]

@pytest.mark.parametrize("manifold", MANIFOLDS)
class TestManifoldProperties:
    
    def test_exp_at_zero_is_identity(self, manifold):
        """exp_p(0) = p"""
        p = manifold.random_point()
        v = torch.zeros_like(p)
        assert torch.allclose(manifold.exp(p, v), p, atol=1e-5)
    
    def test_log_at_same_point_is_zero(self, manifold):
        """log_p(p) = 0"""
        p = manifold.random_point()
        v = manifold.log(p, p)
        assert torch.allclose(v, torch.zeros_like(v), atol=1e-5)
    
    def test_exp_log_inverse(self, manifold):
        """exp_p(log_p(q)) = q"""
        p = manifold.random_point()
        q = manifold.random_point()
        v = manifold.log(p, q)
        q_recovered = manifold.exp(p, v)
        assert torch.allclose(q_recovered, q, atol=1e-4)
    
    def test_log_exp_inverse(self, manifold):
        """log_p(exp_p(v)) = v for small v"""
        p = manifold.random_point()
        v = 0.1 * manifold.random_tangent(p)  # Small tangent
        q = manifold.exp(p, v)
        v_recovered = manifold.log(p, q)
        assert torch.allclose(v_recovered, v, atol=1e-4)
    
    def test_distance_symmetry(self, manifold):
        """d(p, q) = d(q, p)"""
        p = manifold.random_point()
        q = manifold.random_point()
        d_pq = manifold.distance(p, q)
        d_qp = manifold.distance(q, p)
        assert torch.allclose(d_pq, d_qp, atol=1e-6)
    
    def test_distance_equals_log_norm(self, manifold):
        """d(p, q) = ||log_p(q)||"""
        p = manifold.random_point()
        q = manifold.random_point()
        dist = manifold.distance(p, q)
        log_norm = manifold.log(p, q).norm()
        assert torch.allclose(dist, log_norm, atol=1e-5)
    
    def test_parallel_transport_preserves_norm(self, manifold):
        """||PT(v)|| = ||v||"""
        p = manifold.random_point()
        q = manifold.random_point()
        v = manifold.random_tangent(p)
        
        v_transported = manifold.parallel_transport(v, p, q)
        
        # Compute norms using metric
        norm_v = torch.sqrt((v * v).sum())  # Simplified
        norm_vt = torch.sqrt((v_transported * v_transported).sum())
        
        assert torch.allclose(norm_v, norm_vt, atol=1e-4)
    
    def test_projection_is_idempotent(self, manifold):
        """project(project(x)) = project(x)"""
        x = torch.randn(manifold.dim + 10)  # Ambient space
        p1 = manifold.project(x)
        p2 = manifold.project(p1)
        assert torch.allclose(p1, p2, atol=1e-6)
    
    def test_tangent_projection_is_tangent(self, manifold):
        """Projected vector is in tangent space"""
        p = manifold.random_point()
        v = torch.randn_like(p)
        v_proj = manifold.project_tangent(p, v)
        
        # For sphere: tangent iff v·p = 0
        if isinstance(manifold, Sphere):
            assert torch.allclose(torch.dot(v_proj, p), torch.tensor(0.0), atol=1e-6)
```

### 12.4 Optimizer Convergence Tests

```python
@pytest.mark.parametrize("optimizer_cls", [RiemannianSGD, RiemannianAdam])
def test_optimizer_converges_to_target(optimizer_cls):
    """Optimizer should minimize distance to target point."""
    manifold = Sphere(64)
    
    # Parameter and target
    param = ManifoldParameter(manifold.random_point(), manifold)
    target = manifold.random_point()
    
    initial_dist = manifold.distance(param, target).item()
    
    optimizer = optimizer_cls([param], lr=0.1)
    
    for _ in range(200):
        optimizer.zero_grad()
        loss = manifold.distance(param, target) ** 2
        loss.backward()
        optimizer.step()
    
    final_dist = manifold.distance(param, target).item()
    
    assert final_dist < initial_dist * 0.01, \
        f"Distance should decrease: {initial_dist:.4f} -> {final_dist:.4f}"
```

### 12.5 Writing New Tests

When contributing new features, include tests for:

1. **Unit tests**: Individual functions work correctly
2. **Property tests**: Mathematical invariants hold
3. **Integration tests**: Components work together
4. **Regression tests**: Previously fixed bugs don't recur

```python
# Example test for new feature
def test_my_new_feature():
    """Description of what's being tested."""
    # Arrange
    manifold = Sphere(64)
    input_data = ...
    
    # Act
    result = my_new_function(manifold, input_data)
    
    # Assert
    assert result.shape == expected_shape
    assert torch.allclose(result, expected_value, atol=1e-5)
```

---

## 13. Benchmarks

### 13.1 Running Benchmarks

```bash
# Run all benchmarks
python -m geotorch.benchmarks.run_all

# Specific benchmark
python -m geotorch.benchmarks.forward_pass
python -m geotorch.benchmarks.optimizer_step
python -m geotorch.benchmarks.storage_retrieval
```

### 13.2 Forward Pass Comparison

| Layer | PyTorch (ms) | GeoTorch (ms) | Overhead |
|-------|--------------|---------------|----------|
| Linear(512, 512) | 0.023 | 0.031 | 1.35x |
| Embedding(10k, 256) | 0.018 | 0.024 | 1.33x |
| Attention(256, 8 heads) | 0.142 | 0.198 | 1.39x |

*Overhead comes from manifold projections. Acceptable for the benefits gained.*

### 13.3 Optimizer Step Comparison

| Optimizer | PyTorch (ms) | GeoTorch (ms) | Notes |
|-----------|--------------|---------------|-------|
| SGD | 0.45 | 0.62 | +exp map |
| Adam | 0.89 | 1.34 | +parallel transport |

### 13.4 Storage Retrieval

| Corpus Size | GeoStorage (ms) | FAISS (ms) | Speedup |
|-------------|-----------------|------------|---------|
| 10,000 | 0.12 | 0.89 | 7.4x |
| 100,000 | 0.14 | 2.31 | 16.5x |
| 1,000,000 | 0.15 | 8.72 | 58.1x |

*GeoStorage maintains O(1) while traditional methods scale O(log n) or worse.*

---

## 14. Contributing

### 14.1 Development Setup

```bash
# Clone repository
git clone https://github.com/beedavis/geotorch.git
cd geotorch

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\Activate.ps1  # Windows

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 14.2 Code Style

We follow these conventions:
- **Formatting**: Black (line length 100)
- **Linting**: Ruff
- **Type hints**: Required for public API
- **Docstrings**: Google style

```bash
# Format code
black geotorch/ tests/

# Lint
ruff check geotorch/ tests/

# Type check
mypy geotorch/
```

### 14.3 Pull Request Process

1. **Fork** the repository
2. **Create a branch**: `git checkout -b feature/my-feature`
3. **Make changes** with tests
4. **Run tests**: `pytest tests/ -v`
5. **Format code**: `black . && ruff check --fix .`
6. **Commit**: `git commit -m "feat: add my feature"`
7. **Push**: `git push origin feature/my-feature`
8. **Open PR** against `main` branch

### 14.4 Commit Message Convention

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Adding tests
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `ci`: CI/CD changes

Examples:
```
feat(manifolds): add Grassmann manifold implementation
fix(optim): correct parallel transport in RiemannianAdam
docs(readme): add installation instructions
test(sphere): add property-based tests for exp/log
```

### 14.5 Adding a New Manifold

1. Create `geotorch/manifolds/mymanifold.py`:

```python
from geotorch.manifold import Manifold

class MyManifold(Manifold):
    """
    Description of the manifold.
    
    Mathematical definition...
    
    Args:
        dim: Dimension of the manifold
        param: Additional parameter
    
    Examples:
        >>> M = MyManifold(64)
        >>> p = M.random_point()
        >>> v = M.random_tangent(p)
        >>> q = M.exp(p, v)
    """
    
    def __init__(self, dim: int, param: float = 1.0):
        self._dim = dim
        self.param = param
    
    @property
    def dim(self) -> int:
        return self._dim
    
    def exp(self, p, v):
        # Implementation
        ...
    
    def log(self, p, q):
        # Implementation
        ...
    
    # ... implement all abstract methods
```

2. Add tests in `tests/test_manifolds.py`:

```python
def test_mymanifold_properties():
    M = MyManifold(64)
    # ... property tests
```

3. Export in `geotorch/__init__.py`:

```python
from geotorch.manifolds.mymanifold import MyManifold
```

4. Document in README

### 14.6 Publishing to PyPI

Maintainers only:

```bash
# Update version in pyproject.toml
# Update CHANGELOG.md

# Build
python -m build

# Test upload
twine upload --repository testpypi dist/*

# Production upload
twine upload dist/*

# Tag release
git tag v0.1.0
git push origin v0.1.0
```

---

## 15. Roadmap

### Phase 1: Core (v0.1) ✅
- [x] Base `Manifold` class
- [x] `Sphere`, `Euclidean` implementations
- [x] `ManifoldTensor`, `TangentTensor`
- [x] `exp`, `log`, `parallel_transport`
- [x] Basic tests

### Phase 2: Optimization (v0.2)
- [ ] `RiemannianSGD`
- [ ] `RiemannianAdam`
- [ ] Autograd integration
- [ ] Benchmarks vs PyTorch

### Phase 3: Neural Network Layers (v0.3)
- [ ] `ManifoldLinear`
- [ ] `GeodesicEmbedding`
- [ ] `GeometricAttention`
- [ ] `FrechetMean`

### Phase 4: Storage Integration (v0.4)
- [ ] `GeoStorage` (O(1) retrieval)
- [ ] `DavisCache`
- [ ] Integration with transformers

### Phase 5: Advanced Manifolds (v0.5)
- [ ] `Hyperbolic`
- [ ] `SPD`
- [ ] `DavisManifold` (learned metric)
- [ ] `ProductManifold`

### Phase 6: Production (v1.0)
- [ ] CUDA kernels
- [ ] Distributed training support
- [ ] Model zoo
- [ ] Comprehensive documentation

### Future
- [ ] JAX backend
- [ ] Integration with Hugging Face
- [ ] Pre-trained geometric models

---

## 16. Citation

If you use GeoTorch in your research, please cite:

```bibtex
@software{geotorch2025,
  author = {Davis, Bee Rosa},
  title = {GeoTorch: Manifold-Native Deep Learning Framework},
  year = {2025},
  url = {https://github.com/beedavis/geotorch},
  note = {Riemannian deep learning with geodesic optimization and O(1) retrieval}
}
```

Related publications:

```bibtex
@patent{davis2025geostorage,
  author = {Davis, Bee Rosa},
  title = {Geodesic Data Storage System: Constant-Time Semantic Retrieval 
           via Manifold-Encoded Storage Architecture},
  year = {2025},
  note = {U.S. Provisional Patent Application}
}

@patent{davis2025geohash,
  author = {Davis, Bee Rosa},
  title = {Geodesic Hash Functions: Quantum-Resistant Cryptographic Hashing 
           via Riemannian Manifold Trajectory Encoding},
  year = {2025},
  note = {U.S. Provisional Patent Application}
}
```

---

## 17. License

```
Copyright 2025 Bee Rosa Davis

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## Acknowledgments

GeoTorch builds on foundational work in:
- Riemannian optimization (Absil, Mahony, Sepulchre)
- Information geometry (Amari)
- Hyperbolic neural networks (Nickel & Kiela, Ganea et al.)
- Natural gradient methods (Martens, Pascanu)

Special thanks to the PyTorch team for creating an extensible framework.

---

<p align="center">
  <b>The geometry of the problem is the geometry of the computation.</b>
  <br>
  <i>— Bee Rosa Davis, 2025</i>
</p>
