# DavisTensor: A Geometry-Native Tensor Library

## Vision Statement

> **"What if tensors knew their own geometry?"**

DavisTensor is a from-scratch tensor library where Riemannian geometry is not bolted on, but **built into the DNA**. Every tensor knows what manifold it lives on. Every operation respects curvature. Every gradient is automatically a tangent vector, parallel transported correctly.

PyTorch treats manifolds as a constraint to enforce. DavisTensor treats manifolds as the **natural habitat** of data.

---

## Why Build From Scratch?

### The Problem with "Geometry on Top"

```python
# PyTorch + GeoTorch (current approach)
x = torch.randn(64)
x = manifold.project(x)           # Easy to forget!
v = torch.randn(64)               # Is this a point or tangent vector? Who knows!
v = manifold.project_tangent(x, v) # Must remember to project tangent too
y = manifold.exp(x, v)            # Explicit exp map
loss.backward()
# grad is in ambient space, not tangent space!
# must manually project, transport, etc.
```

**Problems:**
1. **Type confusion**: Points and tangent vectors are both `torch.Tensor`
2. **Manual projection**: Forget once and your optimization diverges
3. **Ambient gradients**: Autograd doesn't know about manifolds
4. **No parallel transport**: Gradients at different points can't be compared
5. **Geometry is invisible**: The compiler/JIT can't optimize geometric operations

### The DavisTensor Solution

```python
# DavisTensor (proposed)
x = dt.randn(64, manifold=Hyperbolic)    # Tensor knows it's hyperbolic
v = dt.tangent(x, data=torch.randn(64))  # Type-safe tangent vector at x
y = x + v                                 # Automatically exp map!
d = x.distance(y)                         # Native operation
d.backward()
# x.grad is ALREADY in tangent space at x
# No manual projection needed
# Parallel transport handled automatically
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DavisTensor Stack                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                         User API Layer                              │ │
│  │  dt.randn() / dt.zeros() / dt.tensor() / dt.tangent()              │ │
│  │  Tensor.exp() / .log() / .distance() / .parallel_transport()       │ │
│  │  dt.nn.GeodesicLinear / dt.nn.ManifoldConv / dt.nn.FrechetPool    │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                    │                                     │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                      Type System Layer                              │ │
│  │  ManifoldTensor[M, P]  - Point on manifold M with properties P     │ │
│  │  TangentTensor[M, x]   - Tangent vector at point x on M            │ │
│  │  CotangentTensor[M, x] - Cotangent (gradient) at x                 │ │
│  │  Scalar                - Manifold-independent scalar               │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                    │                                     │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    Geometry Engine Layer                            │ │
│  │  Manifold Registry - Hyperbolic, Sphere, SPD, Product, Learned     │ │
│  │  Metric Tensor - g_ij(x) computed lazily or eagerly                │ │
│  │  Connection - Christoffel symbols, parallel transport              │ │
│  │  Curvature - Riemann tensor, sectional curvature                   │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                    │                                     │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                     Autograd Engine Layer                           │ │
│  │  RiemannianTape - Records ops with geometric context               │ │
│  │  TangentBackward - Gradients live in tangent space                 │ │
│  │  ParallelTransport - Automatic transport during backprop           │ │
│  │  NaturalGradient - Optional: use metric for preconditioning        │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                    │                                     │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                      Compiler Layer                                 │ │
│  │  GeometricIR - Intermediate representation with geometry types     │ │
│  │  ManifoldFusion - Fuse exp(log(x)) → identity                      │ │
│  │  CurvatureSpecialization - Constant curvature fast paths           │ │
│  │  TangentElimination - Avoid materializing tangent vectors          │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                    │                                     │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                      Backend Layer                                  │ │
│  │  CPU Kernels - SIMD-optimized geometric primitives                 │ │
│  │  CUDA Kernels - GPU implementations                                │ │
│  │  Metal/ROCm - Apple/AMD support                                    │ │
│  │  Custom Hardware - TPU, Graphcore, etc. (future)                   │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                    │                                     │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                      Memory Layer                                   │ │
│  │  Storage - Raw memory buffers                                      │ │
│  │  Allocator - Device-specific allocation                            │ │
│  │  TensorCore - Underlying data + strides + geometry metadata        │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Part 1: Type System

### Core Types

The type system is the **foundation**. It ensures geometric correctness at compile time (or runtime with clear errors).

```python
from davistensor import dt
from davistensor.typing import ManifoldTensor, TangentTensor, Scalar

# ============================================================================
# 1. ManifoldTensor - A point on a manifold
# ============================================================================

class ManifoldTensor:
    """
    A tensor that knows it lives on a Riemannian manifold.
    
    Type: ManifoldTensor[M] where M is the manifold type
    
    Attributes:
        data: The underlying numerical data
        manifold: The Riemannian manifold this tensor lives on
        shape: Tensor shape (batch dimensions + manifold point dimension)
        dtype: Data type (float32, float64, etc.)
        device: Compute device (cpu, cuda:0, etc.)
        requires_grad: Whether to track gradients
        
    The tensor is ALWAYS on the manifold. No projection needed.
    Invalid states are prevented by construction.
    """
    
    # Creation
    @staticmethod
    def randn(*shape, manifold: Manifold) -> ManifoldTensor[M]:
        """Random point on manifold (uses manifold's distribution)."""
        ...
    
    @staticmethod
    def zeros(*shape, manifold: Manifold) -> ManifoldTensor[M]:
        """Origin/identity point on manifold."""
        ...
    
    @staticmethod
    def from_ambient(data: RawTensor, manifold: Manifold) -> ManifoldTensor[M]:
        """Project ambient space data onto manifold."""
        ...
    
    # Geometric operations (return ManifoldTensor)
    def exp(self, v: TangentTensor) -> ManifoldTensor[M]:
        """Exponential map: move from self in direction v."""
        ...
    
    def geodesic(self, other: ManifoldTensor[M], t: Scalar) -> ManifoldTensor[M]:
        """Point at fraction t along geodesic from self to other."""
        ...
    
    # Operations that return TangentTensor
    def log(self, other: ManifoldTensor[M]) -> TangentTensor[M, self]:
        """Logarithm map: tangent vector pointing from self to other."""
        ...
    
    # Operations that return Scalar
    def distance(self, other: ManifoldTensor[M]) -> Scalar:
        """Geodesic distance."""
        ...
    
    def norm(self, v: TangentTensor) -> Scalar:
        """Norm of tangent vector using Riemannian metric."""
        ...
    
    # Arithmetic (geometric interpretation)
    def __add__(self, v: TangentTensor) -> ManifoldTensor[M]:
        """Point + TangentVector = exp(point, vector)"""
        return self.exp(v)
    
    def __sub__(self, other: ManifoldTensor[M]) -> TangentTensor[M, self]:
        """Point - Point = log(self, other)"""
        return self.log(other)
    
    # Gradient access
    @property
    def grad(self) -> TangentTensor[M, self]:
        """Gradient lives in tangent space at self."""
        ...


# ============================================================================
# 2. TangentTensor - A vector in the tangent space at a point
# ============================================================================

class TangentTensor:
    """
    A tangent vector at a specific point on a manifold.
    
    Type: TangentTensor[M, x] where M is manifold, x is base point
    
    Key property: tangent vectors can only be added/scaled at the SAME point.
    To move a tangent vector to another point, use parallel transport.
    
    This prevents the common bug of adding gradients at different points
    without transporting them first.
    """
    
    @property
    def base_point(self) -> ManifoldTensor[M]:
        """The point where this tangent vector lives."""
        ...
    
    @property
    def manifold(self) -> Manifold:
        """The underlying manifold."""
        ...
    
    # Vector space operations (only at same base point!)
    def __add__(self, other: TangentTensor[M, x]) -> TangentTensor[M, x]:
        """Add tangent vectors at the same point."""
        if self.base_point != other.base_point:
            raise GeometryError("Cannot add tangent vectors at different points. "
                              "Use parallel_transport first.")
        ...
    
    def __mul__(self, scalar: Scalar) -> TangentTensor[M, x]:
        """Scalar multiplication."""
        ...
    
    def __neg__(self) -> TangentTensor[M, x]:
        """Negate tangent vector."""
        ...
    
    # Inner product using metric
    def inner(self, other: TangentTensor[M, x]) -> Scalar:
        """⟨self, other⟩_x using Riemannian metric at x."""
        ...
    
    def norm(self) -> Scalar:
        """||self||_x = sqrt(⟨self, self⟩_x)"""
        ...
    
    # Transport to another point
    def transport_to(self, target: ManifoldTensor[M]) -> TangentTensor[M, target]:
        """Parallel transport this vector to target point."""
        ...


# ============================================================================
# 3. Scalar - Manifold-independent value
# ============================================================================

class Scalar:
    """
    A scalar value that doesn't live on any manifold.
    
    Results of distance(), norm(), inner(), etc. are Scalars.
    Scalars can be freely combined with standard arithmetic.
    """
    
    def __add__(self, other: Scalar) -> Scalar: ...
    def __mul__(self, other: Scalar) -> Scalar: ...
    def __truediv__(self, other: Scalar) -> Scalar: ...
    def backward(self): ...  # Standard backprop
    
    def item(self) -> float:
        """Convert to Python float."""
        ...


# ============================================================================
# 4. Type Safety Examples
# ============================================================================

# CORRECT: Adding tangent vectors at same point
x = dt.randn(64, manifold=Hyperbolic)
v1 = dt.tangent_randn(x)
v2 = dt.tangent_randn(x)
v3 = v1 + v2  # OK: same base point

# ERROR: Adding tangent vectors at different points
x = dt.randn(64, manifold=Hyperbolic)
y = dt.randn(64, manifold=Hyperbolic)
vx = dt.tangent_randn(x)
vy = dt.tangent_randn(y)
vz = vx + vy  # COMPILE ERROR: different base points!

# CORRECT: Transport first, then add
vx_at_y = vx.transport_to(y)
vz = vx_at_y + vy  # OK: both at y now

# CORRECT: Point + tangent = new point (exp map)
x = dt.randn(64, manifold=Hyperbolic)
v = dt.tangent_randn(x)
y = x + v  # Calls x.exp(v), returns ManifoldTensor

# CORRECT: Point - point = tangent vector (log map)
x = dt.randn(64, manifold=Hyperbolic)
y = dt.randn(64, manifold=Hyperbolic)
v = y - x  # Calls x.log(y), returns TangentTensor at x

# ERROR: Mixing manifolds
x_hyp = dt.randn(64, manifold=Hyperbolic)
y_sph = dt.randn(64, manifold=Sphere)
d = x_hyp.distance(y_sph)  # COMPILE ERROR: manifold mismatch!
```

### Manifold Type Hierarchy

```python
from abc import ABC, abstractmethod
from typing import TypeVar, Generic

M = TypeVar('M', bound='Manifold')

class Manifold(ABC):
    """
    Abstract base class for all Riemannian manifolds.
    
    A manifold defines:
    - The space points live in
    - The metric tensor (how to measure distances/angles)
    - Geodesics (shortest paths)
    - Exponential/logarithm maps
    - Parallel transport
    - Curvature
    """
    
    # -------------------------------------------------------------------------
    # Core Properties
    # -------------------------------------------------------------------------
    
    @property
    @abstractmethod
    def dim(self) -> int:
        """Intrinsic dimension of the manifold."""
        ...
    
    @property
    @abstractmethod
    def ambient_dim(self) -> int:
        """Dimension of ambient space (may equal dim for embedded manifolds)."""
        ...
    
    @property
    def curvature_type(self) -> str:
        """'constant', 'variable', or 'learned'"""
        return 'variable'
    
    # -------------------------------------------------------------------------
    # Point Operations
    # -------------------------------------------------------------------------
    
    @abstractmethod
    def random_point(self, *shape, device, dtype) -> RawTensor:
        """Sample random point on manifold."""
        ...
    
    @abstractmethod
    def origin(self, *shape, device, dtype) -> RawTensor:
        """Canonical origin/identity point."""
        ...
    
    @abstractmethod
    def check_point(self, x: RawTensor) -> bool:
        """Verify x is on the manifold (within tolerance)."""
        ...
    
    @abstractmethod
    def project_point(self, x: RawTensor) -> RawTensor:
        """Project ambient point onto manifold."""
        ...
    
    # -------------------------------------------------------------------------
    # Tangent Space Operations
    # -------------------------------------------------------------------------
    
    @abstractmethod
    def check_tangent(self, x: RawTensor, v: RawTensor) -> bool:
        """Verify v is in tangent space at x."""
        ...
    
    @abstractmethod
    def project_tangent(self, x: RawTensor, v: RawTensor) -> RawTensor:
        """Project ambient vector to tangent space at x."""
        ...
    
    @abstractmethod
    def random_tangent(self, x: RawTensor) -> RawTensor:
        """Sample random tangent vector at x."""
        ...
    
    # -------------------------------------------------------------------------
    # Metric Operations
    # -------------------------------------------------------------------------
    
    @abstractmethod
    def metric(self, x: RawTensor) -> RawTensor:
        """
        Metric tensor g_ij(x) at point x.
        
        Returns: (..., dim, dim) tensor
        """
        ...
    
    @abstractmethod
    def inner(self, x: RawTensor, u: RawTensor, v: RawTensor) -> RawTensor:
        """
        Inner product ⟨u, v⟩_x using metric at x.
        
        Returns: (...,) scalar tensor
        """
        ...
    
    def norm(self, x: RawTensor, v: RawTensor) -> RawTensor:
        """||v||_x = sqrt(⟨v, v⟩_x)"""
        return self.inner(x, v, v).sqrt()
    
    # -------------------------------------------------------------------------
    # Exponential and Logarithm Maps
    # -------------------------------------------------------------------------
    
    @abstractmethod
    def exp(self, x: RawTensor, v: RawTensor) -> RawTensor:
        """
        Exponential map: exp_x(v).
        
        Move from x in direction v for unit time along geodesic.
        """
        ...
    
    @abstractmethod
    def log(self, x: RawTensor, y: RawTensor) -> RawTensor:
        """
        Logarithm map: log_x(y).
        
        Tangent vector at x pointing toward y.
        """
        ...
    
    def distance(self, x: RawTensor, y: RawTensor) -> RawTensor:
        """Geodesic distance d(x, y) = ||log_x(y)||_x"""
        v = self.log(x, y)
        return self.norm(x, v)
    
    def geodesic(self, x: RawTensor, y: RawTensor, t: float) -> RawTensor:
        """Point at fraction t ∈ [0,1] along geodesic from x to y."""
        v = self.log(x, y)
        return self.exp(x, t * v)
    
    # -------------------------------------------------------------------------
    # Parallel Transport
    # -------------------------------------------------------------------------
    
    @abstractmethod
    def parallel_transport(
        self, 
        x: RawTensor, 
        y: RawTensor, 
        v: RawTensor
    ) -> RawTensor:
        """
        Parallel transport v from T_x M to T_y M along geodesic.
        """
        ...
    
    # -------------------------------------------------------------------------
    # Curvature
    # -------------------------------------------------------------------------
    
    def christoffel(self, x: RawTensor) -> RawTensor:
        """
        Christoffel symbols Γ^k_ij(x).
        
        Returns: (..., dim, dim, dim) tensor
        """
        ...
    
    def riemann(self, x: RawTensor) -> RawTensor:
        """
        Riemann curvature tensor R^l_ijk(x).
        
        Returns: (..., dim, dim, dim, dim) tensor
        """
        ...
    
    def sectional_curvature(self, x: RawTensor, u: RawTensor, v: RawTensor) -> RawTensor:
        """
        Sectional curvature K(u, v) of plane spanned by u, v at x.
        """
        ...
    
    def scalar_curvature(self, x: RawTensor) -> RawTensor:
        """
        Scalar curvature R(x) = trace of Ricci tensor.
        """
        ...


# ============================================================================
# Built-in Manifolds
# ============================================================================

class Euclidean(Manifold):
    """
    Flat Euclidean space R^n.
    
    - Zero curvature
    - exp_x(v) = x + v
    - log_x(y) = y - x  
    - Parallel transport is identity
    """
    ...

class Sphere(Manifold):
    """
    n-dimensional sphere S^n embedded in R^{n+1}.
    
    - Constant positive curvature 1/r²
    - Points: ||x|| = r
    - Tangent space: {v : x·v = 0}
    """
    ...

class Hyperbolic(Manifold):
    """
    Hyperbolic space H^n (Poincaré ball model).
    
    - Constant negative curvature -1/r²
    - Points: ||x|| < r (inside the ball)
    - Exponential volume growth
    """
    ...

class SPD(Manifold):
    """
    Symmetric Positive Definite matrices SPD(n).
    
    - Variable curvature (depends on eigenvalues)
    - Affine-invariant metric
    - Geodesics via matrix exponential/logarithm
    """
    ...

class ProductManifold(Manifold):
    """
    Cartesian product M₁ × M₂ × ... × M_k.
    
    - Curvature varies by component
    - Operations are component-wise
    """
    ...

class LearnedManifold(Manifold):
    """
    Manifold with learned metric (DavisManifold).
    
    - Neural network defines metric tensor
    - Curvature adapts to data
    - Christoffel symbols via autodiff
    """
    ...

class Stiefel(Manifold):
    """
    Stiefel manifold St(n, k) = orthonormal k-frames in R^n.
    
    - {X ∈ R^{n×k} : X^T X = I_k}
    - Used for orthogonal constraints
    """
    ...

class Grassmann(Manifold):
    """
    Grassmann manifold Gr(n, k) = k-dimensional subspaces of R^n.
    
    - Quotient of Stiefel by O(k)
    - Used for subspace learning
    """
    ...
```

---

## Part 2: Autograd Engine

The autograd engine is **geometry-aware**. Gradients are tangent vectors. Backpropagation includes parallel transport.

### Design Principles

1. **Gradients are tangent vectors**: `x.grad` has type `TangentTensor[M, x]`
2. **Automatic projection**: No need to manually project to tangent space
3. **Automatic transport**: When combining gradients from different points
4. **Natural gradient (optional)**: Use metric tensor for preconditioning

```python
# ============================================================================
# Geometry-Aware Autograd
# ============================================================================

class RiemannianTape:
    """
    Records operations for backward pass with geometric context.
    
    Unlike standard autograd which only tracks tensor dependencies,
    RiemannianTape also tracks:
    - Which manifold each tensor lives on
    - The base point for tangent vectors
    - Parallel transport paths for gradients
    """
    
    def __init__(self):
        self.operations = []
        self.transport_graph = TransportGraph()
    
    def record(self, op: Operation, inputs: List[Tensor], output: Tensor):
        """Record an operation with geometric metadata."""
        self.operations.append(GeometricOp(
            op=op,
            inputs=inputs,
            output=output,
            manifolds=[t.manifold for t in inputs if hasattr(t, 'manifold')],
            base_points=[t.base_point for t in inputs if hasattr(t, 'base_point')]
        ))
    
    def backward(self, loss: Scalar):
        """
        Backward pass with geometric correctness.
        
        For each parameter x on manifold M:
        1. Compute gradient in ambient space
        2. Project to tangent space T_x M
        3. If gradient was transported from another point, include transport
        """
        ...


class RiemannianGrad:
    """
    Represents a Riemannian gradient (tangent vector).
    
    Key difference from Euclidean gradient:
    - Lives in tangent space T_x M, not ambient space
    - Has a base point
    - Must be transported before combining with grads at other points
    """
    
    def __init__(self, data: RawTensor, base_point: ManifoldTensor, manifold: Manifold):
        self.data = data
        self.base_point = base_point
        self.manifold = manifold
        
        # Verify it's actually in tangent space
        assert manifold.check_tangent(base_point.data, data)
    
    def transport_to(self, new_base: ManifoldTensor) -> 'RiemannianGrad':
        """Parallel transport this gradient to a new base point."""
        transported = self.manifold.parallel_transport(
            self.base_point.data,
            new_base.data,
            self.data
        )
        return RiemannianGrad(transported, new_base, self.manifold)


# ============================================================================
# Backward Rules for Geometric Operations
# ============================================================================

class ExpBackward:
    """
    Backward pass for exponential map: y = exp_x(v)
    
    Given grad_y (gradient w.r.t. y), compute:
    - grad_x: gradient w.r.t. base point x
    - grad_v: gradient w.r.t. tangent vector v
    
    Key insight: grad_y is at y, but we need grads at x.
    Must parallel transport grad_y from y back to x.
    """
    
    @staticmethod
    def apply(ctx, grad_y: TangentTensor) -> Tuple[TangentTensor, TangentTensor]:
        x, v = ctx.saved_tensors
        y = ctx.output
        manifold = ctx.manifold
        
        # Transport gradient from T_y M back to T_x M
        grad_y_at_x = manifold.parallel_transport(y, x, grad_y)
        
        # Differential of exp map
        d_exp = manifold.exp_differential(x, v)  # Linear map T_x → T_y
        d_exp_adj = d_exp.adjoint()              # Adjoint: T_y → T_x
        
        grad_v = d_exp_adj(grad_y_at_x)
        grad_x = compute_base_point_gradient(x, v, grad_y_at_x, manifold)
        
        return grad_x, grad_v


class LogBackward:
    """
    Backward pass for logarithm map: v = log_x(y)
    
    Given grad_v (gradient w.r.t. v, a tangent vector at x),
    compute gradients w.r.t. x and y.
    """
    
    @staticmethod
    def apply(ctx, grad_v: TangentTensor) -> Tuple[TangentTensor, TangentTensor]:
        x, y = ctx.saved_tensors
        manifold = ctx.manifold
        
        # Differential of log map
        d_log_x, d_log_y = manifold.log_differential(x, y)
        
        grad_x = d_log_x.adjoint()(grad_v)
        
        # grad_y needs to be transported from x to y
        grad_y_at_x = d_log_y.adjoint()(grad_v)
        grad_y = manifold.parallel_transport(x, y, grad_y_at_x)
        
        return grad_x, grad_y


class DistanceBackward:
    """
    Backward pass for distance: d = distance(x, y)
    
    d = ||log_x(y)||_x
    
    Gradient: ∂d/∂x = -v/||v|| where v = log_x(y)
              ∂d/∂y = parallel_transport(v/||v||, x → y)
    """
    
    @staticmethod
    def apply(ctx, grad_d: Scalar) -> Tuple[TangentTensor, TangentTensor]:
        x, y = ctx.saved_tensors
        manifold = ctx.manifold
        
        v = manifold.log(x, y)
        v_norm = manifold.norm(x, v)
        
        # Unit tangent vector
        v_unit = v / v_norm
        
        grad_x = -grad_d * v_unit  # TangentTensor at x
        grad_y = grad_d * manifold.parallel_transport(x, y, v_unit)  # at y
        
        return grad_x, grad_y


# ============================================================================
# Natural Gradient (Optional)
# ============================================================================

class NaturalGradient:
    """
    Natural gradient uses the inverse metric as preconditioner.
    
    Standard gradient: g
    Natural gradient: G^{-1} g where G is the metric tensor
    
    This is equivalent to steepest descent in the Riemannian metric,
    which is more natural for curved spaces.
    
    References:
    - Amari, "Natural Gradient Works Efficiently in Learning" (1998)
    """
    
    @staticmethod
    def compute(x: ManifoldTensor, grad: TangentTensor) -> TangentTensor:
        G = x.manifold.metric(x.data)
        G_inv = torch.linalg.inv(G)
        
        natural_grad = G_inv @ grad.data
        return TangentTensor(natural_grad, x)
```

### Gradient Flow Example

```python
# Forward pass
x = dt.randn(64, manifold=Hyperbolic, requires_grad=True)
y = dt.randn(64, manifold=Hyperbolic, requires_grad=True)
v = dt.tangent_randn(x, requires_grad=True)

# Operations
z = x.exp(v)           # z = exp_x(v)
d = z.distance(y)      # d = d(z, y)

# Backward pass
d.backward()

# Gradients are TANGENT VECTORS at their respective base points
print(x.grad)  # TangentTensor at x (∂d/∂x)
print(y.grad)  # TangentTensor at y (∂d/∂y)
print(v.grad)  # TangentTensor at x (∂d/∂v, since v is at x)

# The backward pass automatically:
# 1. Computed ∂d/∂z at z
# 2. Transported ∂d/∂z from z back to x (for x.grad)
# 3. Used chain rule through exp map
# 4. All gradients are correctly in tangent spaces
```

---

## Part 3: Operations and Kernels

### Operation Categories

```python
# ============================================================================
# 1. Manifold-Intrinsic Operations (preserve manifold)
# ============================================================================

# These operations take manifold points and return manifold points
ManifoldTensor.exp(v: TangentTensor) -> ManifoldTensor
ManifoldTensor.geodesic(other: ManifoldTensor, t: Scalar) -> ManifoldTensor
ManifoldTensor.frechet_mean(points: List[ManifoldTensor], weights: Tensor) -> ManifoldTensor

# Batched operations
dt.geodesic_interpolate(x: ManifoldTensor, y: ManifoldTensor, t: Tensor) -> ManifoldTensor
dt.frechet_mean(points: ManifoldTensor, weights: Tensor, dim: int) -> ManifoldTensor


# ============================================================================
# 2. Manifold → Tangent Operations (produce tangent vectors)
# ============================================================================

ManifoldTensor.log(other: ManifoldTensor) -> TangentTensor
ManifoldTensor.random_tangent() -> TangentTensor
ManifoldTensor.zero_tangent() -> TangentTensor


# ============================================================================
# 3. Manifold → Scalar Operations (produce scalars)
# ============================================================================

ManifoldTensor.distance(other: ManifoldTensor) -> Scalar
ManifoldTensor.inner(u: TangentTensor, v: TangentTensor) -> Scalar
TangentTensor.norm() -> Scalar

# Batched
dt.pairwise_distance(x: ManifoldTensor, y: ManifoldTensor) -> Scalar  # (N, M)


# ============================================================================
# 4. Tangent Space Operations (linear algebra in tangent space)
# ============================================================================

TangentTensor + TangentTensor -> TangentTensor  # (same base point!)
Scalar * TangentTensor -> TangentTensor
TangentTensor.inner(other: TangentTensor) -> Scalar
TangentTensor.transport_to(new_base: ManifoldTensor) -> TangentTensor


# ============================================================================
# 5. Cross-Manifold Operations (via projection)
# ============================================================================

# Project between manifolds
dt.project(x: ManifoldTensor[M1], target: Manifold[M2]) -> ManifoldTensor[M2]

# Product manifold split/combine
ProductManifold.split(x: ManifoldTensor) -> Tuple[ManifoldTensor, ...]
ProductManifold.combine(*components: ManifoldTensor) -> ManifoldTensor
```

### Kernel Implementations

```c++
// ============================================================================
// C++/CUDA Kernels for Geometric Primitives
// ============================================================================

namespace davistensor {
namespace kernels {

// ---------------------------------------------------------------------------
// Hyperbolic Space (Poincaré Ball)
// ---------------------------------------------------------------------------

template <typename scalar_t>
__global__ void hyperbolic_exp_kernel(
    const scalar_t* __restrict__ x,      // Base points: (N, D)
    const scalar_t* __restrict__ v,      // Tangent vectors: (N, D)
    scalar_t* __restrict__ y,            // Output points: (N, D)
    const scalar_t c,                    // Curvature parameter
    const int N,
    const int D
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    const scalar_t* x_i = x + idx * D;
    const scalar_t* v_i = v + idx * D;
    scalar_t* y_i = y + idx * D;
    
    // Compute ||v|| using Poincaré metric
    scalar_t x_norm_sq = 0;
    scalar_t v_norm_sq = 0;
    for (int d = 0; d < D; d++) {
        x_norm_sq += x_i[d] * x_i[d];
        v_norm_sq += v_i[d] * v_i[d];
    }
    
    scalar_t lambda_x = 2.0 / (1.0 - c * x_norm_sq);
    scalar_t v_norm = sqrt(v_norm_sq) * lambda_x;
    
    if (v_norm < 1e-7) {
        // Zero tangent vector: return x
        for (int d = 0; d < D; d++) {
            y_i[d] = x_i[d];
        }
        return;
    }
    
    // Möbius addition for exp map
    // y = x ⊕_c (tanh(sqrt(c) * ||v|| / 2) * v / (sqrt(c) * ||v||))
    scalar_t sqrt_c = sqrt(c);
    scalar_t t = tanh(sqrt_c * v_norm / lambda_x / 2.0);
    
    // ... (full Möbius addition implementation)
}


template <typename scalar_t>
__global__ void hyperbolic_distance_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ y,
    scalar_t* __restrict__ dist,
    const scalar_t c,
    const int N,
    const int D
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    // d(x, y) = (2/√c) arctanh(√c ||−x ⊕ y||)
    // ... (implementation)
}


template <typename scalar_t>
__global__ void hyperbolic_parallel_transport_kernel(
    const scalar_t* __restrict__ x,      // Source point
    const scalar_t* __restrict__ y,      // Target point
    const scalar_t* __restrict__ v,      // Vector at x
    scalar_t* __restrict__ v_transported, // Vector at y
    const scalar_t c,
    const int N,
    const int D
) {
    // Parallel transport along geodesic from x to y
    // Uses the formula for Poincaré ball
    // ... (implementation)
}


// ---------------------------------------------------------------------------
// Sphere
// ---------------------------------------------------------------------------

template <typename scalar_t>
__global__ void sphere_exp_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ y,
    const int N,
    const int D
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    const scalar_t* x_i = x + idx * D;
    const scalar_t* v_i = v + idx * D;
    scalar_t* y_i = y + idx * D;
    
    // exp_x(v) = cos(||v||) * x + sin(||v||) * v / ||v||
    scalar_t v_norm = 0;
    for (int d = 0; d < D; d++) {
        v_norm += v_i[d] * v_i[d];
    }
    v_norm = sqrt(v_norm);
    
    if (v_norm < 1e-7) {
        for (int d = 0; d < D; d++) {
            y_i[d] = x_i[d];
        }
        return;
    }
    
    scalar_t cos_t = cos(v_norm);
    scalar_t sin_t = sin(v_norm);
    scalar_t sin_t_over_norm = sin_t / v_norm;
    
    for (int d = 0; d < D; d++) {
        y_i[d] = cos_t * x_i[d] + sin_t_over_norm * v_i[d];
    }
}


// ---------------------------------------------------------------------------
// SPD Matrices (using cuSOLVER for eigendecomposition)
// ---------------------------------------------------------------------------

template <typename scalar_t>
void spd_exp_kernel(
    const scalar_t* P,      // Base point (N, D, D)
    const scalar_t* V,      // Tangent vector (N, D, D)
    scalar_t* Q,            // Output (N, D, D)
    const int N,
    const int D,
    cudaStream_t stream
) {
    // exp_P(V) = P^{1/2} exp(P^{-1/2} V P^{-1/2}) P^{1/2}
    //
    // Steps:
    // 1. Eigendecompose P = U Λ U^T
    // 2. Compute P^{1/2} = U Λ^{1/2} U^T
    // 3. Compute P^{-1/2} = U Λ^{-1/2} U^T
    // 4. Compute M = P^{-1/2} V P^{-1/2}
    // 5. Eigendecompose M = W Σ W^T
    // 6. Compute exp(M) = W exp(Σ) W^T
    // 7. Compute Q = P^{1/2} exp(M) P^{1/2}
    
    // Use cuSOLVER for batched eigendecomposition
    // ...
}


// ---------------------------------------------------------------------------
// Parallel Transport (generic, uses Schild's ladder if no closed form)
// ---------------------------------------------------------------------------

template <typename Manifold>
__global__ void schilds_ladder_kernel(
    const scalar_t* x,
    const scalar_t* y,
    const scalar_t* v,
    scalar_t* v_transported,
    const int N,
    const int D,
    const int n_steps
) {
    // Schild's ladder: approximate parallel transport via geodesic parallelograms
    // Works for any manifold, but slower than closed-form when available
    
    // For each step:
    // 1. m = midpoint of geodesic x → (x + v)
    // 2. m' = point such that y is midpoint of m → m'
    // 3. x ← midpoint, v ← direction to m'
    // ...
}

}  // namespace kernels
}  // namespace davistensor
```

---

## Part 4: Compiler and Optimization

### Geometric IR (Intermediate Representation)

```python
# ============================================================================
# Geometric IR: Operations Annotated with Manifold Types
# ============================================================================

class GeometricIR:
    """
    Intermediate representation that preserves geometric types.
    
    Unlike standard IR which loses type information, GeometricIR tracks:
    - Which manifold each value lives on
    - Whether it's a point, tangent, or scalar
    - Curvature information for optimization
    """
    
    class Op:
        inputs: List[Value]
        outputs: List[Value]
        manifolds: List[Manifold]
        
    class Value:
        name: str
        dtype: DType
        shape: Tuple[int, ...]
        geometric_type: GeometricType  # Point, Tangent, Scalar
        manifold: Optional[Manifold]
        base_point: Optional[Value]  # For tangent vectors


# ============================================================================
# Optimization Passes
# ============================================================================

class ManifoldFusion(OptimizationPass):
    """
    Fuse redundant geometric operations.
    
    Examples:
    - exp(x, log(x, y)) → y
    - log(x, exp(x, v)) → v
    - project(project(x)) → project(x)
    - transport(transport(v, x, y), y, x) → v (if same geodesic)
    """
    
    def apply(self, ir: GeometricIR) -> GeometricIR:
        # Pattern matching and replacement
        patterns = [
            Pattern("exp($x, log($x, $y))", replacement="$y"),
            Pattern("log($x, exp($x, $v))", replacement="$v"),
            Pattern("project(project($x))", replacement="project($x)"),
        ]
        return ir.apply_patterns(patterns)


class CurvatureSpecialization(OptimizationPass):
    """
    Specialize operations for constant curvature manifolds.
    
    Constant curvature (Sphere, Hyperbolic, Euclidean) have:
    - Closed-form exp, log, distance, parallel transport
    - No need to compute Christoffel symbols
    - Much faster than generic Riemannian ops
    """
    
    def apply(self, ir: GeometricIR) -> GeometricIR:
        for op in ir.ops:
            if op.manifold.curvature_type == 'constant':
                # Replace generic op with specialized version
                if isinstance(op, GenericExp):
                    if isinstance(op.manifold, Hyperbolic):
                        ir.replace(op, HyperbolicExpSpecialized(op))
                    elif isinstance(op.manifold, Sphere):
                        ir.replace(op, SphereExpSpecialized(op))
        return ir


class TangentVectorElimination(OptimizationPass):
    """
    Avoid materializing intermediate tangent vectors when possible.
    
    Example:
        v = log(x, y)
        d = norm(x, v)
    
    Can be fused to:
        d = distance(x, y)  # Avoids allocating v
    """
    
    def apply(self, ir: GeometricIR) -> GeometricIR:
        # Fuse log + norm → distance
        # Fuse log + exp → geodesic
        # etc.
        ...


class ParallelTransportCSE(OptimizationPass):
    """
    Common subexpression elimination for parallel transport.
    
    If we transport multiple vectors along the same geodesic,
    we can compute the transport operator once and reuse it.
    """
    
    def apply(self, ir: GeometricIR) -> GeometricIR:
        # Group transport ops by (source, target)
        # Compute transport operator once, apply to all vectors
        ...


class GeometricConstantFolding(OptimizationPass):
    """
    Fold constants with geometric knowledge.
    
    Examples:
    - distance(x, x) → 0
    - exp(x, zero_tangent) → x
    - geodesic(x, y, 0) → x
    - geodesic(x, y, 1) → y
    """
    ...


class MetricCaching(OptimizationPass):
    """
    Cache metric tensor computations.
    
    For learned manifolds (DavisManifold), metric tensor G(x) is expensive.
    If multiple operations use the same base point, compute G(x) once.
    """
    
    def apply(self, ir: GeometricIR) -> GeometricIR:
        # Find all ops that need metric at same point
        # Insert metric computation once, share result
        ...
```

### JIT Compilation

```python
# ============================================================================
# JIT Compiler for Geometric Computations
# ============================================================================

@dt.jit
def geodesic_mean(points: ManifoldTensor, weights: Tensor) -> ManifoldTensor:
    """
    Compute weighted Fréchet mean.
    
    The JIT compiler will:
    1. Parse the function into GeometricIR
    2. Apply optimization passes
    3. Generate specialized kernels
    4. Cache compiled code
    """
    # Karcher mean algorithm
    mean = points[0]
    for _ in range(10):
        tangent_sum = dt.zeros_tangent(mean)
        for i in range(len(points)):
            v = mean.log(points[i])  # Tangent vector at mean
            tangent_sum = tangent_sum + weights[i] * v
        mean = mean.exp(tangent_sum)
    return mean


# What the JIT produces:
# 1. Recognizes constant manifold → uses specialized kernels
# 2. Fuses log + weighted sum into single kernel
# 3. Avoids allocating intermediate tangent vectors
# 4. Parallelizes over points
# 5. Caches the compiled kernel
```

---

## Part 5: Neural Network Layers

```python
import davistensor as dt
import davistensor.nn as dtn

# ============================================================================
# Core Layers
# ============================================================================

class GeodesicLinear(dtn.Module):
    """
    Linear layer between manifolds via tangent space.
    
    Maps: M₁ → M₂
    
    1. log_origin(x) → tangent vector
    2. Linear transform in tangent space
    3. exp_origin → point on M₂
    
    Learnable: weight matrix W
    """
    
    def __init__(
        self,
        in_manifold: Manifold,
        out_manifold: Manifold,
        bias: bool = True
    ):
        super().__init__()
        self.in_manifold = in_manifold
        self.out_manifold = out_manifold
        
        self.weight = dtn.Parameter(
            dt.randn(out_manifold.dim, in_manifold.dim)
        )
        if bias:
            # Bias is a tangent vector at origin
            self.bias = dtn.Parameter(
                dt.zeros(out_manifold.dim)
            )
    
    def forward(self, x: ManifoldTensor) -> ManifoldTensor:
        # To tangent space at origin of input manifold
        v = x.manifold.log(x.manifold.origin(), x)
        
        # Linear transform
        v_out = v @ self.weight.T
        if self.bias is not None:
            v_out = v_out + self.bias
        
        # To output manifold
        return self.out_manifold.exp(self.out_manifold.origin(), v_out)


class ManifoldEmbedding(dtn.Module):
    """
    Embedding table where embeddings live on a manifold.
    
    Unlike nn.Embedding which returns Euclidean vectors,
    this returns ManifoldTensor objects.
    """
    
    def __init__(
        self,
        num_embeddings: int,
        manifold: Manifold,
        scale: float = 0.01
    ):
        super().__init__()
        self.manifold = manifold
        self.embeddings = dtn.ManifoldParameter(
            dt.randn(num_embeddings, manifold=manifold) * scale
        )
    
    def forward(self, indices: Tensor) -> ManifoldTensor:
        return self.embeddings[indices]


class FrechetMeanPool(dtn.Module):
    """
    Pooling via Fréchet mean instead of arithmetic mean.
    
    For a set of points on a manifold, computes the point
    that minimizes sum of squared geodesic distances.
    """
    
    def __init__(self, n_iters: int = 5):
        super().__init__()
        self.n_iters = n_iters
    
    def forward(
        self,
        x: ManifoldTensor,  # (B, N, D) - N points to pool
        weights: Optional[Tensor] = None
    ) -> ManifoldTensor:  # (B, D)
        return dt.frechet_mean(x, weights, dim=1, n_iters=self.n_iters)


class GeometricAttention(dtn.Module):
    """
    Attention using geodesic distances instead of dot products.
    
    score(q, k) = -d(q, k)² instead of q·k
    
    Respects manifold geometry for keys and queries.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        manifold: Manifold,
        dropout: float = 0.0
    ):
        super().__init__()
        self.manifold = manifold
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = GeodesicLinear(manifold, manifold)
        self.k_proj = GeodesicLinear(manifold, manifold)
        self.v_proj = dtn.Linear(embed_dim, embed_dim)  # Values stay Euclidean
        
    def forward(
        self,
        query: ManifoldTensor,  # (B, N, D)
        key: ManifoldTensor,    # (B, M, D)
        value: Tensor           # (B, M, D)
    ) -> Tensor:
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Compute pairwise geodesic distances
        # d[i,j] = distance(q[i], k[j])
        d = dt.pairwise_distance(q, k)  # (B, N, M)
        
        # Attention scores (negative squared distance)
        scores = -d.pow(2) / math.sqrt(self.head_dim)
        
        # Softmax and weighted sum
        attn = dt.softmax(scores, dim=-1)
        return attn @ v


class ManifoldBatchNorm(dtn.Module):
    """
    Batch normalization on a manifold.
    
    Centers data around Fréchet mean, scales in tangent space.
    """
    
    def __init__(self, manifold: Manifold, momentum: float = 0.1):
        super().__init__()
        self.manifold = manifold
        self.momentum = momentum
        
        self.running_mean = dtn.Buffer(manifold.origin())
        self.scale = dtn.Parameter(dt.ones(manifold.dim))
    
    def forward(self, x: ManifoldTensor) -> ManifoldTensor:
        if self.training:
            # Compute batch Fréchet mean
            mean = dt.frechet_mean(x, dim=0)
            
            # Update running mean via geodesic interpolation
            self.running_mean = self.running_mean.geodesic(
                mean, self.momentum
            )
        else:
            mean = self.running_mean
        
        # Center: log map to tangent space at mean
        centered = mean.log(x)  # Tangent vectors at mean
        
        # Scale in tangent space
        scaled = centered * self.scale
        
        # Project back to manifold
        return mean.exp(scaled)


class ManifoldResidual(dtn.Module):
    """
    Residual connection using geodesic interpolation.
    
    Instead of x + f(x), we do geodesic(x, f(x), α)
    """
    
    def __init__(self, module: dtn.Module, alpha: float = 0.5):
        super().__init__()
        self.module = module
        self.alpha = alpha
    
    def forward(self, x: ManifoldTensor) -> ManifoldTensor:
        fx = self.module(x)
        return x.geodesic(fx, self.alpha)


# ============================================================================
# High-Level Modules
# ============================================================================

class ManifoldTransformerBlock(dtn.Module):
    """
    Transformer block with geometric attention and manifold residuals.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        manifold: Manifold,
        ff_dim: int = None,
        dropout: float = 0.1
    ):
        super().__init__()
        ff_dim = ff_dim or 4 * embed_dim
        
        self.attn = GeometricAttention(embed_dim, num_heads, manifold, dropout)
        self.norm1 = ManifoldBatchNorm(manifold)
        
        self.ff = dtn.Sequential(
            GeodesicLinear(manifold, Euclidean(ff_dim)),
            dtn.GELU(),
            GeodesicLinear(Euclidean(ff_dim), manifold)
        )
        self.norm2 = ManifoldBatchNorm(manifold)
        
        self.residual1 = ManifoldResidual(dtn.Identity(), alpha=0.5)
        self.residual2 = ManifoldResidual(dtn.Identity(), alpha=0.5)
    
    def forward(self, x: ManifoldTensor) -> ManifoldTensor:
        # Self-attention with residual
        attn_out = self.attn(x, x, x.to_euclidean())
        x = self.residual1(x, self.manifold.from_euclidean(attn_out))
        x = self.norm1(x)
        
        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.residual2(x, ff_out)
        x = self.norm2(x)
        
        return x
```

---

## Part 6: Optimizers

```python
# ============================================================================
# Riemannian Optimizers (Native, Not Wrapped)
# ============================================================================

class RiemannianSGD(dtn.Optimizer):
    """
    Riemannian stochastic gradient descent.
    
    Update rule:
        x_{t+1} = exp_{x_t}(-lr * grad_t)
    
    The gradient is already in the tangent space (thanks to autograd),
    so we just exp in the negative gradient direction.
    """
    
    def __init__(self, params, lr: float = 0.01, momentum: float = 0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.velocities = {}
    
    def step(self):
        for param in self.params:
            grad = param.grad  # TangentTensor at param
            
            if self.momentum > 0:
                if param not in self.velocities:
                    self.velocities[param] = grad
                else:
                    # Transport old velocity to current point
                    old_vel = self.velocities[param]
                    old_point = self.velocities.base_point
                    transported = old_vel.transport_to(param)
                    
                    # Update velocity
                    self.velocities[param] = self.momentum * transported + grad
                
                update = self.velocities[param]
            else:
                update = grad
            
            # Riemannian update: exp in negative gradient direction
            new_param = param.exp(-self.lr * update)
            param.data = new_param.data


class RiemannianAdam(dtn.Optimizer):
    """
    Riemannian Adam optimizer.
    
    Maintains first and second moment estimates in the tangent space.
    Uses parallel transport to move moments between iterations.
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        amsgrad: bool = False
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.amsgrad = amsgrad
        
        self.state = {}
    
    def step(self):
        for param in self.params:
            if param not in self.state:
                self.state[param] = {
                    'm': param.manifold.zero_tangent(param),  # First moment
                    'v': 0.0,  # Second moment (scalar for diagonal)
                    't': 0,    # Step count
                    'prev_point': param.clone()
                }
            
            state = self.state[param]
            grad = param.grad  # TangentTensor at param
            
            state['t'] += 1
            
            # Transport previous moments to current point
            if state['t'] > 1:
                state['m'] = state['m'].transport_to(param)
            
            # Update moments
            state['m'] = self.beta1 * state['m'] + (1 - self.beta1) * grad
            state['v'] = self.beta2 * state['v'] + (1 - self.beta2) * grad.norm().pow(2)
            
            # Bias correction
            m_hat = state['m'] / (1 - self.beta1 ** state['t'])
            v_hat = state['v'] / (1 - self.beta2 ** state['t'])
            
            # Update
            update = m_hat / (v_hat.sqrt() + self.eps)
            new_param = param.exp(-self.lr * update)
            
            # Save for next iteration
            state['prev_point'] = param.clone()
            param.data = new_param.data


class NaturalGradientDescent(dtn.Optimizer):
    """
    Natural gradient descent using Fisher information / metric tensor.
    
    Update: x_{t+1} = exp_{x_t}(-lr * G^{-1}(x_t) * grad_t)
    
    Where G is the metric tensor (= Fisher information for statistical manifolds).
    """
    
    def __init__(self, params, lr: float = 0.01):
        super().__init__(params)
        self.lr = lr
    
    def step(self):
        for param in self.params:
            grad = param.grad
            
            # Get inverse metric
            G_inv = param.manifold.inverse_metric(param)
            
            # Natural gradient
            natural_grad = G_inv @ grad.data
            natural_grad_tangent = TangentTensor(natural_grad, param)
            
            # Update
            new_param = param.exp(-self.lr * natural_grad_tangent)
            param.data = new_param.data
```

---

## Part 7: Memory and Device Management

```python
# ============================================================================
# Storage Backend
# ============================================================================

class Storage:
    """
    Raw memory buffer backing a tensor.
    
    Handles:
    - Allocation on CPU/GPU/etc.
    - Reference counting
    - Memory pooling for efficiency
    """
    
    def __init__(self, size: int, device: Device, dtype: DType):
        self.size = size
        self.device = device
        self.dtype = dtype
        self.data_ptr = device.allocate(size * dtype.size)
        self.ref_count = 1
    
    def __del__(self):
        self.ref_count -= 1
        if self.ref_count == 0:
            self.device.free(self.data_ptr)


class Device:
    """Abstract device interface."""
    
    @abstractmethod
    def allocate(self, size: int) -> int:
        """Allocate memory, return pointer."""
        ...
    
    @abstractmethod
    def free(self, ptr: int):
        """Free memory."""
        ...
    
    @abstractmethod
    def copy(self, src: int, dst: int, size: int):
        """Copy memory."""
        ...
    
    @abstractmethod
    def synchronize(self):
        """Wait for all operations to complete."""
        ...


class CPUDevice(Device):
    """CPU backend using system allocator or memory pool."""
    ...


class CUDADevice(Device):
    """CUDA backend using cudaMalloc or memory pool."""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.stream = cuda.Stream()
        self.memory_pool = CUDAMemoryPool(device_id)
    
    def allocate(self, size: int) -> int:
        return self.memory_pool.allocate(size)
    
    def free(self, ptr: int):
        self.memory_pool.free(ptr)


# ============================================================================
# Tensor Core (underlying data structure)
# ============================================================================

class TensorCore:
    """
    Core tensor data structure.
    
    Contains:
    - Storage (raw data)
    - Shape and strides
    - Geometric metadata (manifold, base point, etc.)
    - Autograd metadata (grad_fn, etc.)
    """
    
    def __init__(
        self,
        storage: Storage,
        shape: Tuple[int, ...],
        strides: Tuple[int, ...],
        offset: int = 0,
        manifold: Optional[Manifold] = None,
        base_point: Optional['TensorCore'] = None,  # For tangent vectors
        requires_grad: bool = False
    ):
        self.storage = storage
        self.shape = shape
        self.strides = strides
        self.offset = offset
        
        # Geometric metadata
        self.manifold = manifold
        self.base_point = base_point
        self.geometric_type = self._infer_geometric_type()
        
        # Autograd
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None
    
    def _infer_geometric_type(self) -> GeometricType:
        if self.manifold is None:
            return GeometricType.EUCLIDEAN
        elif self.base_point is not None:
            return GeometricType.TANGENT
        else:
            return GeometricType.MANIFOLD_POINT
```

---

## Part 8: Example Usage

```python
import davistensor as dt
import davistensor.nn as dtn

# ============================================================================
# Basic Usage
# ============================================================================

# Create points on manifolds
x = dt.randn(100, 64, manifold=dt.Hyperbolic(64))
y = dt.randn(100, 64, manifold=dt.Hyperbolic(64))

# Geometric operations are natural
d = x.distance(y)              # Geodesic distance
v = x.log(y)                   # Tangent vector from x to y
z = x.exp(v)                   # Same as y (up to numerical precision)
m = x.geodesic(y, 0.5)         # Midpoint

# Arithmetic is geometric
w = x + v                      # exp(x, v) - move x in direction v
v2 = y - x                     # log(x, y) - tangent from x to y

# Gradients are tangent vectors
x.requires_grad = True
loss = x.distance(y).mean()
loss.backward()
print(x.grad)                  # TangentTensor at x, not ambient vector!


# ============================================================================
# Neural Network
# ============================================================================

class HyperbolicClassifier(dtn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int):
        super().__init__()
        
        self.manifold = dt.Hyperbolic(hidden_dim)
        
        # Euclidean → Hyperbolic
        self.input_proj = dtn.GeodesicLinear(
            dt.Euclidean(input_dim),
            self.manifold
        )
        
        # Hyperbolic → Hyperbolic (2 layers)
        self.hidden = dtn.Sequential(
            dtn.GeodesicLinear(self.manifold, self.manifold),
            dtn.ManifoldBatchNorm(self.manifold),
            dtn.GeodesicLinear(self.manifold, self.manifold),
            dtn.ManifoldBatchNorm(self.manifold),
        )
        
        # Class prototypes on manifold
        self.prototypes = dtn.ManifoldParameter(
            dt.randn(n_classes, manifold=self.manifold)
        )
    
    def forward(self, x: dt.Tensor) -> dt.Tensor:
        # Project to manifold
        h = self.input_proj(x)  # ManifoldTensor
        
        # Process on manifold
        h = self.hidden(h)      # Still ManifoldTensor
        
        # Distance to prototypes
        # Negative distance as logits
        logits = -dt.pairwise_distance(h, self.prototypes)
        
        return logits  # Regular tensor for cross-entropy


# Training
model = HyperbolicClassifier(784, 64, 10)
optimizer = dt.optim.RiemannianAdam(model.parameters(), lr=0.001)

for x, y in dataloader:
    optimizer.zero_grad()
    logits = model(x)
    loss = dt.nn.functional.cross_entropy(logits, y)
    loss.backward()  # Geometric autograd!
    optimizer.step() # Riemannian update!


# ============================================================================
# Product Manifold Example
# ============================================================================

# Knowledge graph with hierarchy (Hyperbolic) + features (Euclidean)
manifold = dt.ProductManifold([
    dt.Hyperbolic(32),   # For hierarchical relations
    dt.Euclidean(32)     # For attribute relations
])

entities = dt.randn(10000, manifold=manifold)

# Access components
hyp_part = entities.component(0)  # Hyperbolic component
euc_part = entities.component(1)  # Euclidean component

# Operations respect product structure
d = entities[0].distance(entities[1])  # sqrt(d_hyp² + d_euc²)


# ============================================================================
# Learned Manifold (DavisManifold)
# ============================================================================

# Metric that adapts to data
manifold = dt.LearnedManifold(
    dim=64,
    metric_network=dtn.Sequential(
        dtn.Linear(64, 128),
        dtn.ReLU(),
        dtn.Linear(128, 64 * 65 // 2)  # Lower triangular elements
    )
)

# The metric is learned jointly with the model
x = dt.randn(100, 64, manifold=manifold, requires_grad=True)
y = dt.randn(100, 64, manifold=manifold)

# Distance uses learned metric
d = x.distance(y)
d.sum().backward()  # Gradients flow through metric network!
```

---

## Part 9: Implementation Roadmap

### Phase 1: Core Foundation (Months 1-3)

```
Week 1-2: Memory and Storage
- [ ] Storage class with reference counting
- [ ] CPU allocator
- [ ] Basic TensorCore structure

Week 3-4: Raw Tensor Operations
- [ ] Element-wise ops (add, mul, exp, log, etc.)
- [ ] Reductions (sum, mean, max)
- [ ] Indexing and slicing
- [ ] Broadcasting

Week 5-6: Autograd Engine (Euclidean)
- [ ] Tape-based recording
- [ ] Backward pass
- [ ] Gradient accumulation
- [ ] Basic test suite

Week 7-8: Type System
- [ ] ManifoldTensor wrapper
- [ ] TangentTensor wrapper
- [ ] Scalar wrapper
- [ ] Type checking infrastructure

Week 9-12: Basic Manifolds
- [ ] Euclidean (trivial, for testing)
- [ ] Sphere (exp, log, distance, parallel transport)
- [ ] Hyperbolic (Poincaré ball model)
- [ ] Manifold test suite
```

### Phase 2: Geometric Autograd (Months 4-6)

```
Week 13-16: Riemannian Autograd
- [ ] Tangent-space gradients
- [ ] Automatic projection to tangent space
- [ ] ExpBackward, LogBackward, DistanceBackward
- [ ] Parallel transport in backward pass

Week 17-20: Optimizers
- [ ] RiemannianSGD
- [ ] RiemannianAdam
- [ ] Natural gradient descent

Week 21-24: Testing and Validation
- [ ] Numerical gradient checking
- [ ] Comparison with geoopt/geotorch
- [ ] Performance benchmarks
```

### Phase 3: GPU Backend (Months 7-9)

```
Week 25-30: CUDA Kernels
- [ ] Basic tensor ops on GPU
- [ ] Hyperbolic exp/log/distance kernels
- [ ] Sphere kernels
- [ ] Parallel transport kernels

Week 31-36: Optimization
- [ ] Memory pooling
- [ ] Kernel fusion
- [ ] Batch operations
- [ ] Mixed precision support
```

### Phase 4: Neural Network Layers (Months 10-12)

```
Week 37-42: Core Layers
- [ ] GeodesicLinear
- [ ] ManifoldEmbedding
- [ ] FrechetMeanPool
- [ ] GeometricAttention
- [ ] ManifoldBatchNorm

Week 43-48: High-Level Modules
- [ ] ManifoldTransformer
- [ ] Graph neural network layers
- [ ] Serialization (save/load)
- [ ] Pretrained models
```

### Phase 5: Advanced Features (Months 13-18)

```
- [ ] SPD manifold with efficient eigendecomposition
- [ ] Product manifolds
- [ ] Learned manifolds (DavisManifold)
- [ ] Compiler optimizations (ManifoldFusion, etc.)
- [ ] JIT compilation
- [ ] Distributed training
```

---

## Part 10: Performance Targets

### Comparison with PyTorch + GeoOpt

| Operation | PyTorch + GeoOpt | DavisTensor Target | Notes |
|-----------|------------------|-------------------|-------|
| Hyperbolic exp (1M points) | 5ms | 3ms | Fused kernel |
| Hyperbolic distance (1M pairs) | 8ms | 4ms | Avoid intermediate log |
| Parallel transport (1M vectors) | 12ms | 6ms | Specialized kernel |
| Fréchet mean (1K points) | 50ms | 20ms | Better convergence |
| GeodesicLinear forward | 2ms | 1ms | Fused exp+linear+log |
| Riemannian backward pass | 1.5x Euclidean | 1.2x Euclidean | Less overhead |

### Memory Efficiency

| Metric | PyTorch + GeoOpt | DavisTensor Target |
|--------|------------------|-------------------|
| Tangent vector allocation | Always materialize | Eliminate when possible |
| Metric tensor storage | Explicit matrix | Lazy evaluation |
| Gradient storage | Ambient space | Tangent space (smaller) |

---

## Appendix: Why This Matters

### For Researchers

- **Express geometric ideas directly**: Code matches math
- **Correct by construction**: Type system prevents geometric errors
- **Faster iteration**: Less time debugging projection issues

### For Practitioners

- **Better embeddings**: Hierarchical data → Hyperbolic
- **Faster training**: Geometry-aware optimization converges faster
- **New architectures**: Geometric attention, manifold transformers

### For the Field

- **Foundation for geometric AI**: Like how PyTorch enabled deep learning
- **Reproducibility**: Standard geometric primitives
- **Education**: Learn Riemannian geometry through code

---

## Appendix: Comparison with Existing Libraries

| Feature | PyTorch | GeoOpt | Geomstats | **DavisTensor** |
|---------|---------|--------|-----------|-----------------|
| Geometry built-in | ❌ | Wrapper | Wrapper | **Native** |
| Type-safe tangent vectors | ❌ | ❌ | ❌ | **✓** |
| Automatic parallel transport | ❌ | ❌ | ❌ | **✓** |
| Geometry-aware autograd | ❌ | Partial | Partial | **Full** |
| Compiler optimizations | General | ❌ | ❌ | **Geometric** |
| Learned manifolds | ❌ | ❌ | ❌ | **✓** |

---

*DavisTensor: Where geometry is not a constraint to enforce, but the natural language of computation.*
