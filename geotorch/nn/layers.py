"""
GeoTorch Neural Network Layers
==============================

Building blocks for deep learning on Riemannian manifolds.

Layers:
- ManifoldLinear: Linear map with output on a manifold
- GeodesicEmbedding: Embedding table on a manifold (like nn.Embedding)
- GeometricAttention: Attention using geodesic distances
- MultiHeadGeometricAttention: Multi-head version with projections
- FrechetMean: Differentiable mean on manifolds
- GeodesicLayer: Maps between different manifolds

These enable:
- Hyperbolic neural networks
- Spherical neural networks  
- Geometric deep learning
- Constrained representation learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
import math

from ..manifolds import Sphere, Hyperbolic, Euclidean
from .parameter import ManifoldParameter


# =============================================================================
# MANIFOLD LINEAR
# =============================================================================

class ManifoldLinear(nn.Module):
    """
    Linear layer with output projected onto a manifold.
    
    Computes: manifold.project(Wx + b) or manifold.exp(origin, Wx + b)
    
    This is the fundamental building block for manifold neural networks.
    Unlike standard nn.Linear which outputs to R^n, this outputs to M.
    
    Args:
        in_features: Size of input features
        out_features: Size of output features (ambient dimension of manifold)
        manifold: Target manifold (Sphere, Hyperbolic, etc.)
        bias: Include bias term (default: True)
        method: 'project' or 'exp' (default: 'project')
            - 'project': Apply Wx+b then project to manifold
            - 'exp': Treat Wx+b as tangent vector at origin, apply exp map
    
    Example:
        >>> from geotorch import Sphere
        >>> from geotorch.nn import ManifoldLinear
        >>> layer = ManifoldLinear(128, 64, Sphere(64))
        >>> x = torch.randn(32, 128)  # batch of 32
        >>> y = layer(x)  # shape (32, 64), each row on S^63
        >>> assert torch.allclose(y.norm(dim=-1), torch.ones(32))
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        manifold,
        bias: bool = True,
        method: str = 'project'
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.manifold = manifold
        self.method = method
        
        # Standard linear weights
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Origin point for exp map method
        if method == 'exp':
            origin = self._get_origin()
            self.register_buffer('origin', origin)
        
        self.reset_parameters()
    
    def _get_origin(self) -> Tensor:
        """Get canonical origin point for the manifold."""
        # Get dimension from manifold
        if hasattr(self.manifold, 'n'):
            n = self.manifold.n
        elif hasattr(self.manifold, '_ambient_dim'):
            n = self.manifold._ambient_dim
        else:
            n = self.out_features
        
        # Sphere: north pole [1, 0, 0, ...]
        if isinstance(self.manifold, Sphere):
            origin = torch.zeros(n)
            origin[0] = 1.0
            return origin
        
        # Hyperbolic (Poincaré): origin [0, 0, ...]
        elif isinstance(self.manifold, Hyperbolic):
            return torch.zeros(n)
        
        # Default
        else:
            return torch.zeros(n)
    
    def reset_parameters(self):
        """Initialize weights with Kaiming uniform, scaled for manifold."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        # Scale down for manifold stability
        with torch.no_grad():
            self.weight.mul_(0.1)
        
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            with torch.no_grad():
                self.bias.mul_(0.1)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor, shape (..., in_features)
        
        Returns:
            Output on manifold, shape (..., out_features)
        """
        # Linear transformation
        y = F.linear(x, self.weight, self.bias)
        
        # Map to manifold
        if self.method == 'project':
            return self.manifold.project(y)
        
        elif self.method == 'exp':
            # Treat y as tangent vector at origin
            # Expand origin to match batch dimensions
            origin = self.origin.expand_as(y)
            
            # Project y to tangent space at origin
            y_tangent = self.manifold.project_tangent(origin, y)
            
            # Exponential map
            return self.manifold.exp(origin, y_tangent)
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'manifold={self.manifold.__class__.__name__}, method={self.method}')


# =============================================================================
# GEODESIC EMBEDDING
# =============================================================================

class GeodesicEmbedding(nn.Module):
    """
    Embedding table on a Riemannian manifold.
    
    Like nn.Embedding, but each embedding vector lives on a manifold.
    Essential for hyperbolic embeddings of hierarchies, spherical word vectors, etc.
    
    Args:
        num_embeddings: Size of vocabulary / number of embeddings
        embedding_dim: Dimension of each embedding (ambient dimension)
        manifold: The manifold embeddings live on
        sparse: Use sparse gradients (default: False)
        scale: Initial scale of embeddings (default: 0.001)
    
    Example:
        >>> from geotorch import Hyperbolic
        >>> from geotorch.nn import GeodesicEmbedding
        >>> emb = GeodesicEmbedding(10000, 64, Hyperbolic(64))
        >>> indices = torch.tensor([1, 5, 3, 7])
        >>> vectors = emb(indices)  # shape (4, 64), all inside Poincaré ball
    
    Applications:
        - Poincaré embeddings for hierarchies (Nickel & Kiela 2017)
        - Spherical word embeddings
        - Knowledge graph embeddings
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        manifold,
        sparse: bool = False,
        scale: float = 0.001,
        _weight: Optional[Tensor] = None
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.manifold = manifold
        self.sparse = sparse
        self.scale = scale
        
        if _weight is None:
            weight = self._init_weights()
        else:
            weight = _weight
        
        # Use ManifoldParameter for proper gradient handling
        self.weight = ManifoldParameter(weight, manifold)
    
    def _init_weights(self) -> Tensor:
        """Initialize embeddings on the manifold."""
        # Initialize small random vectors
        weight = torch.randn(self.num_embeddings, self.embedding_dim) * self.scale
        
        # Project onto manifold
        weight = self.manifold.project(weight)
        
        return weight
    
    def forward(self, indices: Tensor) -> Tensor:
        """
        Look up embeddings.
        
        Args:
            indices: Integer tensor of indices, shape (*)
        
        Returns:
            Embeddings on manifold, shape (*, embedding_dim)
        """
        return F.embedding(indices, self.weight, sparse=self.sparse)
    
    def extra_repr(self) -> str:
        return (f'num_embeddings={self.num_embeddings}, '
                f'embedding_dim={self.embedding_dim}, '
                f'manifold={self.manifold.__class__.__name__}')


# =============================================================================
# FRÉCHET MEAN
# =============================================================================

class FrechetMean(nn.Module):
    """
    Differentiable Fréchet mean on a Riemannian manifold.
    
    The Fréchet mean is the generalization of the arithmetic mean to manifolds:
        μ = argmin_p Σ_i w_i * d(p, x_i)²
    
    It's the point that minimizes weighted sum of squared geodesic distances.
    
    This implementation uses iterative gradient descent which is differentiable
    and works for any manifold with exp/log maps.
    
    Args:
        manifold: The Riemannian manifold
        n_iters: Number of gradient descent iterations (default: 10)
        lr: Learning rate for internal optimization (default: 0.5)
    
    Example:
        >>> from geotorch import Sphere
        >>> from geotorch.nn import FrechetMean
        >>> frechet = FrechetMean(Sphere(64))
        >>> points = Sphere(64).random_point(100)
        >>> mean = frechet(points)  # shape (64,), on sphere
    
    Applications:
        - Aggregating node features in geometric GNNs
        - Pooling in hyperbolic neural networks
        - Prototype computation for few-shot learning
    """
    
    def __init__(
        self,
        manifold,
        n_iters: int = 10,
        lr: float = 0.5
    ):
        super().__init__()
        self.manifold = manifold
        self.n_iters = n_iters
        self.lr = lr
    
    def forward(
        self,
        points: Tensor,
        weights: Optional[Tensor] = None
    ) -> Tensor:
        """
        Compute weighted Fréchet mean.
        
        Args:
            points: Points on manifold, shape (N, D) or (B, N, D)
            weights: Optional weights, shape (N,) or (B, N). Default: uniform.
        
        Returns:
            Fréchet mean, shape (D,) or (B, D)
        """
        # Handle batched vs unbatched
        if points.dim() == 2:
            return self._frechet_mean_single(points, weights)
        elif points.dim() == 3:
            # Batched: (B, N, D)
            B = points.shape[0]
            if weights is None:
                weights_list = [None] * B
            else:
                weights_list = [weights[i] for i in range(B)]
            means = [self._frechet_mean_single(points[i], weights_list[i]) for i in range(B)]
            return torch.stack(means)
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {points.dim()}D")
    
    def _frechet_mean_single(
        self,
        points: Tensor,
        weights: Optional[Tensor] = None
    ) -> Tensor:
        """Compute Fréchet mean for single batch."""
        N, D = points.shape
        
        # Default: uniform weights
        if weights is None:
            weights = torch.ones(N, device=points.device, dtype=points.dtype) / N
        else:
            weights = weights / weights.sum()  # Normalize
        
        # Initialize at weighted Euclidean mean, projected to manifold
        mean = self.manifold.project((weights.unsqueeze(-1) * points).sum(dim=0))
        
        # Iterative refinement via Riemannian gradient descent
        for _ in range(self.n_iters):
            # Compute weighted sum of log maps (Riemannian gradient)
            # grad = Σ_i w_i * log_mean(x_i)
            logs = torch.stack([self.manifold.log(mean, p) for p in points])
            grad = (weights.unsqueeze(-1) * logs).sum(dim=0)
            
            # Update: mean = exp_mean(lr * grad)
            mean = self.manifold.exp(mean, self.lr * grad)
        
        return mean


# =============================================================================
# GEOMETRIC ATTENTION
# =============================================================================

class GeometricAttention(nn.Module):
    """
    Attention mechanism using geodesic distances.
    
    Instead of dot-product attention (q·k), uses:
        attention(q, k) = softmax(-d(q, k)² / temperature)
    
    Where d is the geodesic distance on the manifold.
    
    This respects the geometry: nearby points (small geodesic distance)
    get high attention, regardless of Euclidean arrangement.
    
    Args:
        manifold: The Riemannian manifold
        temperature: Softmax temperature (default: 1.0)
        hard: Use hard attention (argmax) instead of soft (default: False)
    
    Example:
        >>> from geotorch import Hyperbolic
        >>> from geotorch.nn import GeometricAttention
        >>> H = Hyperbolic(64)
        >>> attn = GeometricAttention(H)
        >>> queries = H.project(torch.randn(32, 8, 64))   # (batch, n_queries, dim)
        >>> keys = H.project(torch.randn(32, 16, 64))     # (batch, n_keys, dim)
        >>> values = torch.randn(32, 16, 128)             # (batch, n_keys, value_dim)
        >>> output, weights = attn(queries, keys, values)  # (32, 8, 128)
    
    Applications:
        - Hyperbolic attention for hierarchical data
        - Spherical attention for directional data
        - Geometric graph attention networks
    """
    
    def __init__(
        self,
        manifold,
        temperature: float = 1.0,
        hard: bool = False
    ):
        super().__init__()
        self.manifold = manifold
        self.temperature = temperature
        self.hard = hard
    
    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute geometric attention.
        
        Args:
            queries: Query points on manifold, shape (B, N_q, D)
            keys: Key points on manifold, shape (B, N_k, D)
            values: Value vectors (any space), shape (B, N_k, V)
            mask: Optional attention mask, shape (B, N_q, N_k)
        
        Returns:
            output: Attended values, shape (B, N_q, V)
            attention_weights: Attention distribution, shape (B, N_q, N_k)
        """
        B, N_q, D = queries.shape
        _, N_k, _ = keys.shape
        
        # Compute pairwise geodesic distances
        # Shape: (B, N_q, N_k)
        distances = self._pairwise_distances(queries, keys)
        
        # Convert to attention scores (negative squared distance)
        scores = -distances.pow(2) / self.temperature
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Softmax to get attention weights
        if self.hard:
            # Hard attention: one-hot at minimum distance
            indices = distances.argmin(dim=-1)
            attention_weights = F.one_hot(indices, N_k).float()
        else:
            attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.bmm(attention_weights, values)
        
        return output, attention_weights
    
    def _pairwise_distances(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Compute pairwise geodesic distances.
        
        Args:
            x: shape (B, N, D)
            y: shape (B, M, D)
        
        Returns:
            distances: shape (B, N, M)
        """
        B, N, D = x.shape
        _, M, _ = y.shape
        
        # Expand for broadcasting: (B, N, 1, D) and (B, 1, M, D)
        x_exp = x.unsqueeze(2).expand(B, N, M, D)
        y_exp = y.unsqueeze(1).expand(B, N, M, D)
        
        # Flatten for manifold.distance
        x_flat = x_exp.reshape(-1, D)
        y_flat = y_exp.reshape(-1, D)
        
        # Compute distances
        dist_flat = self.manifold.distance(x_flat, y_flat)
        
        # Reshape back
        return dist_flat.reshape(B, N, M)


# =============================================================================
# MULTI-HEAD GEOMETRIC ATTENTION
# =============================================================================

class MultiHeadGeometricAttention(nn.Module):
    """
    Multi-head attention with geodesic distance scoring.
    
    Each head operates on a different manifold subspace, allowing the model
    to attend to different geometric aspects of the data.
    
    Args:
        embed_dim: Total embedding dimension
        num_heads: Number of attention heads
        manifold: The Riemannian manifold (for each head)
        temperature: Softmax temperature (default: 1.0)
        dropout: Dropout probability (default: 0.0)
    
    Example:
        >>> from geotorch import Sphere
        >>> from geotorch.nn import MultiHeadGeometricAttention
        >>> mha = MultiHeadGeometricAttention(256, 8, Sphere(32))
        >>> x = torch.randn(32, 100, 256)  # (batch, seq_len, embed_dim)
        >>> output, weights = mha(x, x, x)
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        manifold,
        temperature: float = 1.0,
        dropout: float = 0.0
    ):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.manifold = manifold
        self.temperature = temperature
        
        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Per-head geometric attention
        self.geo_attn = GeometricAttention(manifold, temperature)
    
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Multi-head geometric attention forward pass.
        
        Args:
            query: shape (B, N_q, embed_dim)
            key: shape (B, N_k, embed_dim)
            value: shape (B, N_k, embed_dim)
            mask: Optional mask, shape (B, N_q, N_k)
        
        Returns:
            output: shape (B, N_q, embed_dim)
            attention_weights: shape (B, num_heads, N_q, N_k)
        """
        B, N_q, _ = query.shape
        _, N_k, _ = key.shape
        
        # Project
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape to (B * num_heads, N, head_dim)
        q = q.view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        q = q.reshape(B * self.num_heads, N_q, self.head_dim)
        k = k.reshape(B * self.num_heads, N_k, self.head_dim)
        v = v.reshape(B * self.num_heads, N_k, self.head_dim)
        
        # Project to manifold
        q = self.manifold.project(q)
        k = self.manifold.project(k)
        
        # Geometric attention
        if mask is not None:
            mask = mask.unsqueeze(1).expand(B, self.num_heads, N_q, N_k)
            mask = mask.reshape(B * self.num_heads, N_q, N_k)
        
        output, attn_weights = self.geo_attn(q, k, v, mask)
        
        # Reshape back
        output = output.view(B, self.num_heads, N_q, self.head_dim)
        output = output.transpose(1, 2).reshape(B, N_q, self.embed_dim)
        attn_weights = attn_weights.view(B, self.num_heads, N_q, N_k)
        
        # Output projection
        output = self.out_proj(output)
        output = self.dropout(output)
        
        return output, attn_weights


# =============================================================================
# GEODESIC LAYER (MAPS BETWEEN MANIFOLDS)
# =============================================================================

class GeodesicLayer(nn.Module):
    """
    Layer that maps between two manifolds via tangent space.
    
    Architecture:
        1. Log map: M1 → T_p(M1) (manifold to tangent space)
        2. Linear: T_p(M1) → T_q(M2) (tangent to tangent)
        3. Exp map: T_q(M2) → M2 (tangent space to manifold)
    
    This is the proper way to do "linear" operations between manifolds.
    
    Args:
        in_manifold: Source manifold
        out_manifold: Target manifold
        in_dim: Input dimension
        out_dim: Output dimension
    
    Example:
        >>> from geotorch import Sphere, Hyperbolic
        >>> from geotorch.nn import GeodesicLayer
        >>> layer = GeodesicLayer(Sphere(64), Hyperbolic(32), 64, 32)
        >>> x = Sphere(64).random_point(100)  # 100 points on S^63
        >>> y = layer(x)  # 100 points in H^31
    """
    
    def __init__(
        self,
        in_manifold,
        out_manifold,
        in_dim: int,
        out_dim: int
    ):
        super().__init__()
        
        self.in_manifold = in_manifold
        self.out_manifold = out_manifold
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Linear map between tangent spaces
        self.linear = nn.Linear(in_dim, out_dim)
        
        # Learnable base points
        self.register_buffer('in_origin', self._get_origin(in_manifold, in_dim))
        self.register_buffer('out_origin', self._get_origin(out_manifold, out_dim))
    
    def _get_origin(self, manifold, dim: int) -> Tensor:
        """Get canonical origin for a manifold."""
        if hasattr(manifold, 'random_point'):
            return manifold.random_point()
        else:
            return manifold.project(torch.zeros(dim))
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Map from in_manifold to out_manifold.
        
        Args:
            x: Points on in_manifold, shape (*, in_dim)
        
        Returns:
            Points on out_manifold, shape (*, out_dim)
        """
        # Log map: M1 → tangent space
        in_origin = self.in_origin.expand_as(x)
        v = self.in_manifold.log(in_origin, x)
        
        # Linear transformation in tangent space
        v_transformed = self.linear(v)
        
        # Exp map: tangent space → M2
        batch_shape = x.shape[:-1]
        out_origin = self.out_origin.expand(*batch_shape, -1)
        v_tangent = self.out_manifold.project_tangent(out_origin, v_transformed)
        
        return self.out_manifold.exp(out_origin, v_tangent)


# =============================================================================
# CONVENIENCE: HYPERBOLIC LAYERS
# =============================================================================

class HyperbolicLinear(ManifoldLinear):
    """Convenience wrapper for ManifoldLinear with Hyperbolic manifold."""
    
    def __init__(self, in_features: int, out_features: int, **kwargs):
        super().__init__(in_features, out_features, Hyperbolic(out_features), **kwargs)


class HyperbolicEmbedding(GeodesicEmbedding):
    """Convenience wrapper for GeodesicEmbedding with Hyperbolic manifold."""
    
    def __init__(self, num_embeddings: int, embedding_dim: int, **kwargs):
        super().__init__(num_embeddings, embedding_dim, Hyperbolic(embedding_dim), **kwargs)


class SphericalLinear(ManifoldLinear):
    """Convenience wrapper for ManifoldLinear with Sphere manifold."""
    
    def __init__(self, in_features: int, out_features: int, **kwargs):
        super().__init__(in_features, out_features, Sphere(out_features), **kwargs)


class SphericalEmbedding(GeodesicEmbedding):
    """Convenience wrapper for GeodesicEmbedding with Sphere manifold."""
    
    def __init__(self, num_embeddings: int, embedding_dim: int, **kwargs):
        super().__init__(num_embeddings, embedding_dim, Sphere(embedding_dim), **kwargs)
