"""
DavisTensor Neural Network Layers Specification
================================================

Geometry-aware neural network layers.

Key insight: Standard neural network operations (linear, attention, pooling)
have natural Riemannian generalizations.

- Linear: maps through tangent space
- Attention: uses geodesic distance instead of dot product
- Pooling: Fréchet mean instead of arithmetic mean
- Normalization: Riemannian centering and scaling

IMPLEMENTATION SPEC
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np

# Type hints for forward references
from ..core.storage import TensorCore
from ..manifolds.base import Manifold, Euclidean


# =============================================================================
# Base Classes
# =============================================================================

class Parameter:
    """
    Learnable parameter.
    
    Like torch.nn.Parameter - a tensor that should be included
    in module.parameters() for optimization.
    """
    
    def __init__(
        self, 
        data: Union[np.ndarray, TensorCore],
        requires_grad: bool = True
    ):
        from ..core.storage import tensor
        
        if isinstance(data, TensorCore):
            self._data = data
        else:
            self._data = tensor(data)
        
        self._data.requires_grad = requires_grad
    
    @property
    def data(self) -> TensorCore:
        return self._data
    
    @data.setter
    def data(self, value):
        if isinstance(value, np.ndarray):
            self._data.storage._data[:] = value.flatten()
        else:
            self._data = value
    
    @property
    def grad(self) -> Optional[TensorCore]:
        return self._data.grad
    
    def numpy(self) -> np.ndarray:
        return self._data.numpy()
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self._data.shape
    
    def __repr__(self) -> str:
        return f"Parameter({self.shape})"


class ManifoldParameter(Parameter):
    """
    Learnable parameter constrained to a manifold.
    
    After each optimization step, project back to manifold.
    """
    
    def __init__(
        self,
        data: Union[np.ndarray, TensorCore],
        manifold: Manifold,
        requires_grad: bool = True
    ):
        super().__init__(data, requires_grad)
        self._manifold = manifold
        
        # Project to manifold on creation
        self._data.storage._data[:] = manifold.project_point(self._data.numpy()).flatten()
        self._data.manifold = manifold
    
    @property
    def manifold(self) -> Manifold:
        return self._manifold
    
    def project(self):
        """Project parameter back to manifold (call after optimizer step)."""
        self._data.storage._data[:] = self._manifold.project_point(self._data.numpy()).flatten()
    
    def __repr__(self) -> str:
        return f"ManifoldParameter({self.shape}, manifold={self._manifold})"


class Module(ABC):
    """
    Base class for all neural network modules.
    
    Like torch.nn.Module:
    - Contains parameters
    - Defines forward pass
    - Tracks submodules
    """
    
    def __init__(self):
        self._parameters: Dict[str, Parameter] = {}
        self._modules: Dict[str, 'Module'] = {}
        self._training: bool = True
    
    def __setattr__(self, name: str, value: Any):
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        super().__setattr__(name, value)
    
    def parameters(self, recurse: bool = True) -> List[Parameter]:
        """Return all parameters."""
        params = list(self._parameters.values())
        if recurse:
            for module in self._modules.values():
                params.extend(module.parameters(recurse=True))
        return params
    
    def manifold_parameters(self, recurse: bool = True) -> List[ManifoldParameter]:
        """Return only manifold-constrained parameters."""
        params = [p for p in self._parameters.values() if isinstance(p, ManifoldParameter)]
        if recurse:
            for module in self._modules.values():
                params.extend(module.manifold_parameters(recurse=True))
        return params
    
    def train(self, mode: bool = True) -> 'Module':
        """Set training mode."""
        self._training = mode
        for module in self._modules.values():
            module.train(mode)
        return self
    
    def eval(self) -> 'Module':
        """Set evaluation mode."""
        return self.train(False)
    
    @property
    def training(self) -> bool:
        return self._training
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass - must be implemented by subclasses."""
        ...
    
    def __call__(self, *args, **kwargs):
        """Call forward()."""
        return self.forward(*args, **kwargs)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# =============================================================================
# Linear Layers
# =============================================================================

class Linear(Module):
    """
    Standard Euclidean linear layer: y = Wx + b
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Xavier initialization
        std = np.sqrt(2.0 / (in_features + out_features))
        self.weight = Parameter(np.random.randn(out_features, in_features) * std)
        
        if bias:
            self.bias = Parameter(np.zeros(out_features))
        else:
            self.bias = None
    
    def forward(self, x: TensorCore) -> TensorCore:
        """
        Args:
            x: (..., in_features)
        
        Returns:
            (..., out_features)
        """
        from ..core.storage import tensor
        
        y = x.numpy() @ self.weight.numpy().T
        if self.bias is not None:
            y = y + self.bias.numpy()
        
        return tensor(y, requires_grad=x.requires_grad)
    
    def __repr__(self) -> str:
        return f"Linear({self.in_features}, {self.out_features}, bias={self.bias is not None})"


class GeodesicLinear(Module):
    """
    Linear layer between manifolds via tangent space.
    
    Maps M₁ → M₂:
    1. Log map: x → log_o(x) ∈ T_o M₁  (to tangent space at origin)
    2. Linear: W @ v + b  (in tangent spaces)
    3. Exp map: exp_o(Wv + b) ∈ M₂
    
    This is the natural generalization of linear layers to manifolds.
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
        
        in_dim = in_manifold.ambient_dim
        out_dim = out_manifold.ambient_dim
        
        # Weight matrix
        std = np.sqrt(2.0 / (in_dim + out_dim))
        self.weight = Parameter(np.random.randn(out_dim, in_dim) * std)
        
        # Bias in tangent space at origin of output manifold
        if bias:
            self.bias = Parameter(np.zeros(out_dim))
        else:
            self.bias = None
    
    def forward(self, x: TensorCore) -> TensorCore:
        """
        Args:
            x: Point on in_manifold, shape (..., in_ambient_dim)
        
        Returns:
            Point on out_manifold, shape (..., out_ambient_dim)
        """
        from ..core.storage import tensor
        
        x_np = x.numpy()
        batch_shape = x_np.shape[:-1]
        
        # Origin of input manifold
        in_origin = self.in_manifold.origin()
        
        # Log map: point → tangent vector at origin
        # For batched: need to handle each point
        if len(batch_shape) == 0:
            v_in = self.in_manifold.log(in_origin, x_np)
        else:
            # Flatten batch, apply, unflatten
            flat_x = x_np.reshape(-1, x_np.shape[-1])
            v_in = np.stack([self.in_manifold.log(in_origin, xi) for xi in flat_x])
            v_in = v_in.reshape(*batch_shape, -1)
        
        # Linear transform in tangent space
        v_out = v_in @ self.weight.numpy().T
        if self.bias is not None:
            v_out = v_out + self.bias.numpy()
        
        # Exp map: tangent vector → point on output manifold
        out_origin = self.out_manifold.origin()
        if len(batch_shape) == 0:
            y = self.out_manifold.exp(out_origin, v_out)
        else:
            flat_v = v_out.reshape(-1, v_out.shape[-1])
            y = np.stack([self.out_manifold.exp(out_origin, vi) for vi in flat_v])
            y = y.reshape(*batch_shape, -1)
        
        result = tensor(y, requires_grad=x.requires_grad)
        result.manifold = self.out_manifold
        return result
    
    def __repr__(self) -> str:
        return f"GeodesicLinear({self.in_manifold} → {self.out_manifold})"


class ManifoldMLR(Module):
    """
    Multinomial Logistic Regression on manifold.
    
    For classification using manifold embeddings.
    Scores class k as: -d(x, p_k)² where p_k is class prototype.
    """
    
    def __init__(self, manifold: Manifold, n_classes: int):
        super().__init__()
        
        self.manifold = manifold
        self.n_classes = n_classes
        
        # Class prototypes on manifold
        prototype_data = manifold.random_point(n_classes) * 0.01
        self.prototypes = ManifoldParameter(prototype_data, manifold)
    
    def forward(self, x: TensorCore) -> TensorCore:
        """
        Args:
            x: Points on manifold, shape (B, ambient_dim)
        
        Returns:
            Logits, shape (B, n_classes)
        """
        from ..core.storage import tensor
        
        x_np = x.numpy()
        prototypes_np = self.prototypes.numpy()
        
        # Compute negative squared distances to each prototype
        B = x_np.shape[0]
        logits = np.zeros((B, self.n_classes))
        
        for k in range(self.n_classes):
            for b in range(B):
                d = self.manifold.distance(x_np[b], prototypes_np[k])
                logits[b, k] = -d ** 2
        
        return tensor(logits, requires_grad=x.requires_grad)
    
    def __repr__(self) -> str:
        return f"ManifoldMLR({self.manifold}, n_classes={self.n_classes})"


# =============================================================================
# Embedding Layers
# =============================================================================

class Embedding(Module):
    """
    Standard Euclidean embedding table.
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Initialize with normal distribution
        self.weight = Parameter(np.random.randn(num_embeddings, embedding_dim) * 0.01)
    
    def forward(self, indices: np.ndarray) -> TensorCore:
        """
        Args:
            indices: Integer indices, any shape
        
        Returns:
            Embeddings, shape (*indices.shape, embedding_dim)
        """
        from ..core.storage import tensor
        
        embeddings = self.weight.numpy()[indices]
        return tensor(embeddings)
    
    def __repr__(self) -> str:
        return f"Embedding({self.num_embeddings}, {self.embedding_dim})"


class ManifoldEmbedding(Module):
    """
    Embedding table where embeddings live on a manifold.
    
    Unlike standard Embedding which returns Euclidean vectors,
    this returns points on a Riemannian manifold.
    """
    
    def __init__(
        self,
        num_embeddings: int,
        manifold: Manifold,
        scale: float = 0.01
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.manifold = manifold
        
        # Initialize as random points on manifold
        initial = manifold.random_point(num_embeddings) * scale
        self.weight = ManifoldParameter(initial, manifold)
    
    def forward(self, indices: np.ndarray) -> TensorCore:
        """
        Args:
            indices: Integer indices, any shape
        
        Returns:
            Manifold points, shape (*indices.shape, ambient_dim)
        """
        from ..core.storage import tensor
        
        embeddings = self.weight.numpy()[indices]
        result = tensor(embeddings)
        result.manifold = self.manifold
        return result
    
    def __repr__(self) -> str:
        return f"ManifoldEmbedding({self.num_embeddings}, {self.manifold})"


# =============================================================================
# Pooling Layers
# =============================================================================

class MeanPool(Module):
    """
    Standard arithmetic mean pooling.
    """
    
    def __init__(self, dim: int = -2):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: TensorCore) -> TensorCore:
        from ..core.storage import tensor
        return tensor(x.numpy().mean(axis=self.dim))


class FrechetMeanPool(Module):
    """
    Pooling via Fréchet mean on a manifold.
    
    Instead of arithmetic mean, computes the Riemannian center of mass:
        μ = argmin_p Σ_i w_i d(p, x_i)²
    
    This is the natural generalization of mean pooling to manifolds.
    """
    
    def __init__(self, manifold: Manifold, dim: int = -2, n_iters: int = 10):
        super().__init__()
        self.manifold = manifold
        self.dim = dim
        self.n_iters = n_iters
    
    def forward(
        self, 
        x: TensorCore, 
        weights: Optional[np.ndarray] = None
    ) -> TensorCore:
        """
        Args:
            x: Points to pool, shape (..., N, ambient_dim) where N is pool dim
            weights: Optional weights, shape (..., N)
        
        Returns:
            Pooled point, shape (..., ambient_dim)
        """
        from ..core.storage import tensor
        
        x_np = x.numpy()
        
        # Move pooling dim to position -2 if needed
        if self.dim != -2:
            x_np = np.moveaxis(x_np, self.dim, -2)
        
        batch_shape = x_np.shape[:-2]
        N = x_np.shape[-2]
        
        if weights is None:
            weights = np.ones((*batch_shape, N)) / N
        else:
            weights = weights / weights.sum(axis=-1, keepdims=True)
        
        # Compute Fréchet mean via gradient descent
        def compute_mean(points, w):
            """Compute single Fréchet mean."""
            # Initialize with first point
            mean = points[0].copy()
            
            for _ in range(self.n_iters):
                # Weighted sum of log maps
                tangent_sum = np.zeros_like(mean)
                for i in range(len(points)):
                    v = self.manifold.log(mean, points[i])
                    tangent_sum += w[i] * v
                
                # Check convergence
                if self.manifold.norm(mean, tangent_sum) < 1e-6:
                    break
                
                # Update
                mean = self.manifold.exp(mean, tangent_sum)
            
            return mean
        
        # Handle batching
        if len(batch_shape) == 0:
            result = compute_mean(x_np, weights)
        else:
            flat_x = x_np.reshape(-1, N, x_np.shape[-1])
            flat_w = weights.reshape(-1, N)
            
            means = []
            for b in range(flat_x.shape[0]):
                means.append(compute_mean(flat_x[b], flat_w[b]))
            
            result = np.stack(means).reshape(*batch_shape, -1)
        
        out = tensor(result)
        out.manifold = self.manifold
        return out
    
    def __repr__(self) -> str:
        return f"FrechetMeanPool({self.manifold}, dim={self.dim})"


# =============================================================================
# Attention Layers
# =============================================================================

class GeometricAttention(Module):
    """
    Attention using geodesic distances instead of dot products.
    
    Standard attention: score = q·k / √d
    Geometric attention: score = -d(q, k)² / τ
    
    Where d is geodesic distance and τ is temperature.
    """
    
    def __init__(
        self,
        manifold: Manifold,
        temperature: float = 1.0
    ):
        super().__init__()
        self.manifold = manifold
        self.temperature = temperature
    
    def forward(
        self,
        query: TensorCore,    # (B, N, D)
        key: TensorCore,      # (B, M, D)
        value: TensorCore     # (B, M, V)
    ) -> TensorCore:
        """
        Compute geometric attention.
        
        Returns:
            Weighted values, shape (B, N, V)
        """
        from ..core.storage import tensor
        
        q = query.numpy()
        k = key.numpy()
        v = value.numpy()
        
        B, N, D = q.shape
        M = k.shape[1]
        V = v.shape[-1]
        
        # Compute pairwise distances
        scores = np.zeros((B, N, M))
        for b in range(B):
            for i in range(N):
                for j in range(M):
                    d = self.manifold.distance(q[b, i], k[b, j])
                    scores[b, i, j] = -d ** 2 / self.temperature
        
        # Softmax
        scores_exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = scores_exp / scores_exp.sum(axis=-1, keepdims=True)
        
        # Weighted sum of values
        output = attn @ v  # (B, N, V)
        
        return tensor(output)
    
    def __repr__(self) -> str:
        return f"GeometricAttention({self.manifold}, τ={self.temperature})"


# =============================================================================
# Normalization Layers
# =============================================================================

class ManifoldBatchNorm(Module):
    """
    Batch normalization on a manifold.
    
    1. Center: compute Fréchet mean, map to tangent space
    2. Scale: normalize variance in tangent space
    3. Shift/scale: learnable parameters in tangent space
    4. Map back to manifold
    """
    
    def __init__(
        self,
        manifold: Manifold,
        momentum: float = 0.1,
        eps: float = 1e-5
    ):
        super().__init__()
        self.manifold = manifold
        self.momentum = momentum
        self.eps = eps
        
        # Running statistics
        self._running_mean = manifold.origin()
        
        # Learnable scale and bias (in tangent space)
        self.scale = Parameter(np.ones(manifold.ambient_dim))
        self.bias = Parameter(np.zeros(manifold.ambient_dim))
    
    def forward(self, x: TensorCore) -> TensorCore:
        """
        Args:
            x: Points on manifold, shape (B, ambient_dim)
        
        Returns:
            Normalized points, shape (B, ambient_dim)
        """
        from ..core.storage import tensor
        
        x_np = x.numpy()
        B = x_np.shape[0]
        
        if self._training:
            # Compute batch Fréchet mean
            weights = np.ones(B) / B
            mean = self._running_mean.copy()
            for _ in range(5):
                tangent_sum = np.zeros_like(mean)
                for i in range(B):
                    tangent_sum += self.manifold.log(mean, x_np[i]) / B
                mean = self.manifold.exp(mean, tangent_sum)
            
            # Update running mean (geodesic moving average)
            self._running_mean = self.manifold.geodesic(
                self._running_mean, mean, self.momentum
            )
        else:
            mean = self._running_mean
        
        # Map to tangent space at mean
        tangents = np.stack([self.manifold.log(mean, x_np[i]) for i in range(B)])
        
        # Normalize in tangent space
        if self._training:
            std = tangents.std(axis=0) + self.eps
        else:
            std = 1.0  # Use fixed scale at test time
        
        normalized = tangents / std
        
        # Apply learnable scale and bias
        transformed = normalized * self.scale.numpy() + self.bias.numpy()
        
        # Map back to manifold
        output = np.stack([self.manifold.exp(mean, transformed[i]) for i in range(B)])
        
        result = tensor(output)
        result.manifold = self.manifold
        return result
    
    def __repr__(self) -> str:
        return f"ManifoldBatchNorm({self.manifold})"


# =============================================================================
# Activation Functions
# =============================================================================

class ReLU(Module):
    """Standard ReLU activation."""
    
    def forward(self, x: TensorCore) -> TensorCore:
        from ..core.storage import tensor
        return tensor(np.maximum(x.numpy(), 0))


class TangentReLU(Module):
    """
    ReLU in tangent space.
    
    For manifold points:
    1. Map to tangent space at origin
    2. Apply ReLU
    3. Map back to manifold
    """
    
    def __init__(self, manifold: Manifold):
        super().__init__()
        self.manifold = manifold
    
    def forward(self, x: TensorCore) -> TensorCore:
        from ..core.storage import tensor
        
        x_np = x.numpy()
        origin = self.manifold.origin()
        
        # To tangent space
        if x_np.ndim == 1:
            v = self.manifold.log(origin, x_np)
            v_relu = np.maximum(v, 0)
            y = self.manifold.exp(origin, v_relu)
        else:
            # Batched
            vs = np.stack([self.manifold.log(origin, xi) for xi in x_np])
            vs_relu = np.maximum(vs, 0)
            y = np.stack([self.manifold.exp(origin, vi) for vi in vs_relu])
        
        result = tensor(y)
        result.manifold = self.manifold
        return result
    
    def __repr__(self) -> str:
        return f"TangentReLU({self.manifold})"


# =============================================================================
# Sequential Container
# =============================================================================

class Sequential(Module):
    """
    Sequential container for layers.
    """
    
    def __init__(self, *layers: Module):
        super().__init__()
        for i, layer in enumerate(layers):
            self._modules[str(i)] = layer
    
    def forward(self, x):
        for layer in self._modules.values():
            x = layer(x)
        return x
    
    def __repr__(self) -> str:
        lines = [f"Sequential("]
        for name, module in self._modules.items():
            lines.append(f"  ({name}): {module}")
        lines.append(")")
        return "\n".join(lines)


# =============================================================================
# Test Function
# =============================================================================

def test_layers():
    """Test neural network layers."""
    print("=" * 60)
    print("Testing DavisTensor Neural Network Layers")
    print("=" * 60)
    
    from ..core.storage import tensor, randn
    from ..manifolds.base import Euclidean
    from ..manifolds.hyperbolic import Hyperbolic
    from ..manifolds.sphere import Sphere
    
    # Test 1: Linear layer
    print("\n1. Linear layer")
    linear = Linear(10, 5)
    x = randn(32, 10)
    y = linear(x)
    print(f"   Input: {x.shape}")
    print(f"   Output: {y.shape}")
    print(f"   Parameters: {len(list(linear.parameters()))}")
    assert y.shape == (32, 5)
    print("   ✅ PASS")
    
    # Test 2: GeodesicLinear
    print("\n2. GeodesicLinear (Hyperbolic → Euclidean)")
    H = Hyperbolic(8)
    E = Euclidean(4)
    geo_linear = GeodesicLinear(H, E)
    
    x = tensor(H.random_point(16))
    x.manifold = H
    y = geo_linear(x)
    
    print(f"   Input: {x.shape} on {H}")
    print(f"   Output: {y.shape} on {y.manifold}")
    assert y.shape == (16, 4)
    print("   ✅ PASS")
    
    # Test 3: ManifoldEmbedding
    print("\n3. ManifoldEmbedding")
    emb = ManifoldEmbedding(1000, Hyperbolic(32))
    indices = np.array([0, 5, 10, 15])
    embeddings = emb(indices)
    
    print(f"   Indices: {indices.shape}")
    print(f"   Embeddings: {embeddings.shape}")
    print(f"   On manifold: {embeddings.manifold}")
    assert embeddings.shape == (4, 32)
    print("   ✅ PASS")
    
    # Test 4: FrechetMeanPool
    print("\n4. FrechetMeanPool")
    S = Sphere(2)
    pool = FrechetMeanPool(S)
    
    # Create points on sphere
    points = tensor(S.random_point(5))
    points.manifold = S
    
    mean = pool(points)
    print(f"   Input: {points.shape} on {S}")
    print(f"   Output: {mean.shape}")
    print(f"   ||mean|| = {np.linalg.norm(mean.numpy()):.6f} (should be 1)")
    assert np.allclose(np.linalg.norm(mean.numpy()), 1.0, atol=1e-5)
    print("   ✅ PASS")
    
    # Test 5: ManifoldMLR
    print("\n5. ManifoldMLR (Classification)")
    mlr = ManifoldMLR(Hyperbolic(16), n_classes=5)
    x = tensor(Hyperbolic(16).random_point(32))
    logits = mlr(x)
    
    print(f"   Input: {x.shape}")
    print(f"   Logits: {logits.shape}")
    assert logits.shape == (32, 5)
    print("   ✅ PASS")
    
    # Test 6: GeometricAttention
    print("\n6. GeometricAttention")
    attn = GeometricAttention(Euclidean(8))
    
    q = randn(2, 4, 8)  # (B, N, D)
    k = randn(2, 6, 8)  # (B, M, D)
    v = randn(2, 6, 16) # (B, M, V)
    
    output = attn(q, k, v)
    print(f"   Query: {q.shape}")
    print(f"   Key: {k.shape}")
    print(f"   Value: {v.shape}")
    print(f"   Output: {output.shape}")
    assert output.shape == (2, 4, 16)
    print("   ✅ PASS")
    
    # Test 7: Sequential
    print("\n7. Sequential container")
    model = Sequential(
        Linear(10, 20),
        ReLU(),
        Linear(20, 5)
    )
    
    x = randn(8, 10)
    y = model(x)
    
    print(f"   Model:\n{model}")
    print(f"   Input: {x.shape}")
    print(f"   Output: {y.shape}")
    print(f"   Total parameters: {sum(p.numpy().size for p in model.parameters())}")
    assert y.shape == (8, 5)
    print("   ✅ PASS")
    
    # Test 8: Parameter projection
    print("\n8. ManifoldParameter projection")
    H = Hyperbolic(8)
    param = ManifoldParameter(np.random.randn(10, 8), H)
    
    # Check on manifold
    for i in range(10):
        assert H.check_point(param.numpy()[i])
    
    print("   All embeddings on manifold after initialization")
    print("   ✅ PASS")
    
    print("\n" + "=" * 60)
    print("All layer tests passed! ✅")
    print("=" * 60)


if __name__ == "__main__":
    test_layers()
