"""
DavisTensor NN - Linear Layers
===============================

Standard and geometric linear transformations.
"""

from __future__ import annotations
import numpy as np

from .module import Module, Parameter, ManifoldParameter


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
    
    def forward(self, x):
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
        in_manifold,
        out_manifold,
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
    
    def forward(self, x):
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
    
    def __init__(self, manifold, n_classes: int):
        super().__init__()
        
        self.manifold = manifold
        self.n_classes = n_classes
        
        # Class prototypes on manifold
        prototype_data = manifold.random_point(n_classes) * 0.01
        self.prototypes = ManifoldParameter(prototype_data, manifold)
    
    def forward(self, x):
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
