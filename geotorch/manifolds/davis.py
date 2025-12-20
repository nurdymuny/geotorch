"""
DavisManifold - Manifold with Learned Riemannian Metric.

A novel contribution from the Davis-Wilson framework: instead of using a fixed
metric (Euclidean, Hyperbolic, etc.), the metric tensor G(x) is learned from data.

Key insight from Yang-Mills:
    "The metric should reflect the distinguishability structure of the data.
    Points that are semantically different should be far apart.
    The curvature encodes this distinguishability."

The metric is parameterized as:
    G(x) = L(x) L(x)^T + εI

where L(x) is the output of a neural network. This ensures G(x) is always
symmetric positive definite.

Applications:
- Adaptive embedding spaces
- Metric learning from similarity labels
- Few-shot learning with learned geometry
- Data-dependent similarity measures
"""

import math
from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class DavisManifold(nn.Module):
    """
    Manifold with learned Riemannian metric.
    
    Instead of a fixed metric (Euclidean, Hyperbolic, etc.), the metric tensor
    G(x) is learned from data. This allows the geometry to adapt to the
    intrinsic structure of the data.
    
    The metric is parameterized as:
        G(x) = L(x) L(x)^T + εI
    
    where L(x) is the output of a neural network. This ensures G(x) is always
    symmetric positive definite.
    
    Args:
        dim: Dimension of the manifold
        hidden_dim: Hidden dimension of metric network
        n_layers: Number of layers in metric network
        min_eigenvalue: Minimum eigenvalue of G(x) for numerical stability
        diagonal_only: If True, learn only diagonal metric (computationally cheaper)
    
    Example:
        >>> manifold = DavisManifold(64, hidden_dim=128)
        >>> x = torch.randn(32, 64)
        >>> 
        >>> # Metric adapts to x
        >>> G = manifold.metric_tensor(x)  # (32, 64, 64)
        >>> 
        >>> # All manifold operations use learned metric
        >>> dist = manifold.distance(x[0], x[1])
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        min_eigenvalue: float = 0.01,
        diagonal_only: bool = False
    ):
        super().__init__()
        
        self.dim = dim
        self.n = dim  # Compatibility with other manifolds
        self.min_eigenvalue = min_eigenvalue
        self.diagonal_only = diagonal_only
        
        # Neural network that outputs metric components
        layers = []
        in_dim = dim
        for _ in range(n_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU()
            ])
            in_dim = hidden_dim
        
        if diagonal_only:
            # Output just diagonal elements
            n_components = dim
        else:
            # Output dim*(dim+1)/2 components for lower triangular L
            n_components = dim * (dim + 1) // 2
        
        layers.append(nn.Linear(in_dim, n_components))
        self.metric_net = nn.Sequential(*layers)
    
    def metric_tensor(self, x: Tensor) -> Tensor:
        """
        Compute metric tensor G(x) at point x.
        
        Args:
            x: Points, shape (..., dim)
        
        Returns:
            G: Metric tensors, shape (..., dim, dim)
        """
        if self.diagonal_only:
            # Diagonal metric: G = diag(σ(f(x))²) + εI
            diag_components = self.metric_net(x)  # (..., dim)
            diag_values = F.softplus(diag_components) + self.min_eigenvalue
            return torch.diag_embed(diag_values)
        else:
            # Full metric: G = L L^T + εI
            L_components = self.metric_net(x)  # (..., n_components)
            L = self._components_to_lower_triangular(L_components)
            G = L @ L.transpose(-2, -1)
            G = G + self.min_eigenvalue * torch.eye(self.dim, device=x.device)
            return G
    
    def _components_to_lower_triangular(self, components: Tensor) -> Tensor:
        """Convert flat components to lower triangular matrix."""
        batch_shape = components.shape[:-1]
        device = components.device
        
        L = torch.zeros(*batch_shape, self.dim, self.dim, device=device)
        
        # Fill lower triangular
        idx = 0
        for i in range(self.dim):
            for j in range(i + 1):
                L[..., i, j] = components[..., idx]
                idx += 1
        
        # Ensure positive diagonal for stability
        diag_indices = torch.arange(self.dim, device=device)
        L[..., diag_indices, diag_indices] = (
            F.softplus(L[..., diag_indices, diag_indices]) + 0.1
        )
        
        return L
    
    def inverse_metric_tensor(self, x: Tensor) -> Tensor:
        """Compute inverse metric G^{-1}(x)."""
        G = self.metric_tensor(x)
        return torch.linalg.inv(G)
    
    def inner_product(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        """
        Compute ⟨u, v⟩_x = u^T G(x) v
        
        Args:
            x: Base point
            u, v: Tangent vectors
        
        Returns:
            Inner product value(s)
        """
        G = self.metric_tensor(x)
        return torch.einsum('...i,...ij,...j->...', u, G, v)
    
    def norm(self, x: Tensor, v: Tensor) -> Tensor:
        """Norm of tangent vector: ||v||_x = sqrt(⟨v, v⟩_x)"""
        return self.inner_product(x, v, v).clamp(min=1e-8).sqrt()
    
    def distance(
        self,
        x: Tensor,
        y: Tensor,
        n_steps: int = 10
    ) -> Tensor:
        """
        Approximate geodesic distance using numerical integration.
        
        Integrates arc length along straight line (approximate geodesic).
        For short distances, this is accurate. For longer distances,
        would need geodesic shooting.
        
        Args:
            x: Start points, shape (..., dim)
            y: End points, shape (..., dim)
            n_steps: Integration steps
        
        Returns:
            Distances, shape (...)
        """
        # Integrate along straight line path
        total_length = torch.zeros(x.shape[:-1], device=x.device)
        
        for i in range(n_steps):
            t = i / n_steps
            
            # Point along path
            p = x + t * (y - x)
            
            # Velocity (tangent vector)
            v = (y - x) / n_steps
            
            # Length element: ds = ||v||_p dt
            ds = self.norm(p, v)
            total_length = total_length + ds
        
        return total_length
    
    def exp(
        self,
        x: Tensor,
        v: Tensor,
        n_steps: int = 10
    ) -> Tensor:
        """
        Approximate exponential map via Euler integration of geodesic equation.
        
        Solves: γ''(t) + Γ^k_ij γ'^i γ'^j = 0
        with γ(0) = x, γ'(0) = v
        
        Args:
            x: Base point, shape (..., dim)
            v: Tangent vector, shape (..., dim)
            n_steps: Integration steps
        
        Returns:
            exp_x(v), shape (..., dim)
        """
        dt = 1.0 / n_steps
        
        # Initialize position and velocity
        pos = x.clone()
        vel = v.clone()
        
        for _ in range(n_steps):
            # Get Christoffel symbols at current position
            Gamma = self.christoffel_symbols(pos)
            
            # Geodesic acceleration: a^k = -Γ^k_ij v^i v^j
            accel = -torch.einsum('...kij,...i,...j->...k', Gamma, vel, vel)
            
            # Euler update
            pos = pos + dt * vel
            vel = vel + dt * accel
        
        return pos
    
    def log(
        self,
        x: Tensor,
        y: Tensor,
        n_iters: int = 10,
        lr: float = 0.1
    ) -> Tensor:
        """
        Approximate logarithm map via optimization.
        
        Find v such that exp_x(v) ≈ y.
        
        Args:
            x: Base point
            y: Target point
            n_iters: Optimization iterations
            lr: Learning rate
        
        Returns:
            log_x(y), shape (..., dim)
        """
        # Initialize with straight-line direction
        v = y - x
        
        for _ in range(n_iters):
            # Compute exp_x(v)
            y_pred = self.exp(x, v, n_steps=5)
            
            # Gradient: direction to move v
            error = y - y_pred
            
            # Update v
            v = v + lr * error
        
        return v
    
    def christoffel_symbols(self, x: Tensor) -> Tensor:
        """
        Compute Christoffel symbols Γ^k_ij from learned metric.
        
        Γ^k_ij = (1/2) g^{kl} (∂_i g_{jl} + ∂_j g_{il} - ∂_l g_{ij})
        
        Uses finite differences for efficiency.
        
        Args:
            x: Points, shape (..., dim)
        
        Returns:
            Christoffel symbols, shape (..., dim, dim, dim)
        """
        batch_shape = x.shape[:-1]
        device = x.device
        eps = 1e-4
        
        # Compute metric and inverse at x
        G = self.metric_tensor(x)
        G_inv = torch.linalg.inv(G)
        
        # Compute metric derivatives via finite differences
        dG = torch.zeros(*batch_shape, self.dim, self.dim, self.dim, device=device)
        
        for l in range(self.dim):
            # Perturb in direction l
            x_plus = x.clone()
            x_plus[..., l] = x_plus[..., l] + eps
            
            x_minus = x.clone()
            x_minus[..., l] = x_minus[..., l] - eps
            
            G_plus = self.metric_tensor(x_plus)
            G_minus = self.metric_tensor(x_minus)
            
            # ∂_l G_ij
            dG[..., l, :, :] = (G_plus - G_minus) / (2 * eps)
        
        # Christoffel symbols: Γ^k_ij = (1/2) g^{kl} (∂_i g_{jl} + ∂_j g_{il} - ∂_l g_{ij})
        Gamma = torch.zeros(*batch_shape, self.dim, self.dim, self.dim, device=device)
        
        for k in range(self.dim):
            for i in range(self.dim):
                for j in range(self.dim):
                    for l in range(self.dim):
                        Gamma[..., k, i, j] = Gamma[..., k, i, j] + 0.5 * G_inv[..., k, l] * (
                            dG[..., i, j, l] + dG[..., j, i, l] - dG[..., l, i, j]
                        )
        
        return Gamma
    
    def project(self, x: Tensor) -> Tensor:
        """Identity projection (ambient space is Euclidean)."""
        return x
    
    def project_tangent(self, x: Tensor, v: Tensor) -> Tensor:
        """Identity projection (tangent space is full R^n)."""
        return v
    
    def random_point(self, *batch_shape) -> Tensor:
        """Random point (standard normal)."""
        return torch.randn(*batch_shape, self.dim)
    
    # ==========================================================================
    # Regularization
    # ==========================================================================
    
    def metric_regularization(
        self,
        x: Tensor,
        target_det: float = 1.0,
        det_weight: float = 1.0,
        smooth_weight: float = 0.1
    ) -> Tensor:
        """
        Regularization loss to keep metric well-behaved.
        
        - Determinant regularization: keep det(G) ≈ target_det
        - Smoothness regularization: metric shouldn't change too fast
        
        Args:
            x: Sample points
            target_det: Target determinant (1 = volume-preserving)
            det_weight: Weight for determinant loss
            smooth_weight: Weight for smoothness loss
        
        Returns:
            Regularization loss
        """
        G = self.metric_tensor(x)
        
        # Determinant regularization
        det_G = torch.linalg.det(G)
        det_loss = ((det_G.log() - math.log(target_det)) ** 2).mean()
        
        # Smoothness: penalize metric variation
        if x.shape[0] > 1:
            # Compare metric at nearby points
            G_rolled = torch.roll(G, shifts=1, dims=0)
            smooth_loss = ((G - G_rolled) ** 2).mean()
        else:
            smooth_loss = torch.tensor(0.0, device=x.device)
        
        return det_weight * det_loss + smooth_weight * smooth_loss


# =============================================================================
# Metric Learning with DavisManifold
# =============================================================================

class DavisMetricLearner(nn.Module):
    """
    Learn a DavisManifold metric from similarity labels.
    
    Given pairs (x_i, x_j) with similarity labels, learn metric such that:
    - d(x_i, x_j) small when similar
    - d(x_i, x_j) large when dissimilar
    
    This is deep metric learning with full Riemannian structure!
    
    Args:
        dim: Dimension of the embedding space
        hidden_dim: Hidden dimension for metric network
        n_layers: Number of layers in metric network
    
    Example:
        >>> learner = DavisMetricLearner(dim=64)
        >>> 
        >>> # Training loop
        >>> for x1, x2, similar in data:
        ...     loss = learner.contrastive_loss(x1, x2, similar)
        ...     loss.backward()
        ...     optimizer.step()
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2
    ):
        super().__init__()
        self.manifold = DavisManifold(dim, hidden_dim, n_layers)
    
    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """Compute distances using learned metric."""
        return self.manifold.distance(x1, x2)
    
    def contrastive_loss(
        self,
        x1: Tensor,
        x2: Tensor,
        labels: Tensor,
        margin: float = 1.0
    ) -> Tensor:
        """
        Contrastive loss for metric learning.
        
        Args:
            x1, x2: Point pairs, shape (B, dim)
            labels: 1 = similar, 0 = dissimilar
            margin: Margin for negative pairs
        
        Returns:
            Loss value
        """
        distances = self.forward(x1, x2)
        
        # Similar pairs: minimize distance
        pos_loss = labels * distances.pow(2)
        
        # Dissimilar pairs: push beyond margin
        neg_loss = (1 - labels) * F.relu(margin - distances).pow(2)
        
        return (pos_loss + neg_loss).mean()
    
    def triplet_loss(
        self,
        anchor: Tensor,
        positive: Tensor,
        negative: Tensor,
        margin: float = 1.0
    ) -> Tensor:
        """
        Triplet loss: d(anchor, positive) < d(anchor, negative) - margin
        
        Args:
            anchor: Anchor points
            positive: Similar points
            negative: Dissimilar points
            margin: Margin between positive and negative
        
        Returns:
            Loss value
        """
        d_pos = self.forward(anchor, positive)
        d_neg = self.forward(anchor, negative)
        
        return F.relu(d_pos - d_neg + margin).mean()
