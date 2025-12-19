"""Manifold-aware tensor classes."""

import torch
from torch import Tensor
from typing import Optional
from .manifold import Manifold


class ManifoldTensor(torch.Tensor):
    """Tensor that lives on a manifold.
    
    This class wraps a PyTorch tensor and associates it with a manifold,
    enabling geometric operations like exponential map, logarithmic map,
    and geodesic distance.
    
    Attributes:
        manifold: The manifold on which this tensor lives
    """
    
    @staticmethod
    def __new__(cls, data: Tensor, manifold: Manifold):
        """Create a new ManifoldTensor.
        
        Args:
            data: Tensor data
            manifold: Associated manifold
        
        Returns:
            ManifoldTensor instance
        """
        # Create tensor subclass
        instance = torch.Tensor._make_subclass(cls, data)
        instance.manifold = manifold
        return instance
    
    def project_(self) -> 'ManifoldTensor':
        """Project onto manifold in-place.
        
        Returns:
            Self (for chaining)
        """
        projected = self.manifold.project(self)
        self.data.copy_(projected)
        return self
    
    def exp(self, v: Tensor) -> 'ManifoldTensor':
        """Move along geodesic with velocity v.
        
        Args:
            v: Tangent vector at this point
        
        Returns:
            New point on manifold
        """
        result = self.manifold.exp(self, v)
        return ManifoldTensor(result, self.manifold)
    
    def log(self, q: 'ManifoldTensor') -> 'TangentTensor':
        """Tangent vector pointing toward q.
        
        Args:
            q: Target point on manifold
        
        Returns:
            Tangent vector at self pointing toward q
        """
        if not isinstance(q, ManifoldTensor):
            raise TypeError("q must be a ManifoldTensor")
        if q.manifold != self.manifold:
            raise ValueError("q must be on the same manifold")
        
        v = self.manifold.log(self, q)
        return TangentTensor(v, self.manifold, self)
    
    def distance(self, q: 'ManifoldTensor') -> Tensor:
        """Geodesic distance to q.
        
        Args:
            q: Target point on manifold
        
        Returns:
            Geodesic distance
        """
        if not isinstance(q, ManifoldTensor):
            raise TypeError("q must be a ManifoldTensor")
        if q.manifold != self.manifold:
            raise ValueError("q must be on the same manifold")
        
        return self.manifold.distance(self, q)
    
    def geodesic_to(self, q: 'ManifoldTensor', t: float) -> 'ManifoldTensor':
        """Interpolate toward q along geodesic.
        
        Args:
            q: Target point
            t: Parameter in [0, 1] (0 = self, 1 = q)
        
        Returns:
            Point along geodesic
        """
        if not isinstance(q, ManifoldTensor):
            raise TypeError("q must be a ManifoldTensor")
        if q.manifold != self.manifold:
            raise ValueError("q must be on the same manifold")
        
        result = self.manifold.geodesic(self, q, t)
        return ManifoldTensor(result, self.manifold)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ManifoldTensor({super().__repr__()}, manifold={self.manifold.__class__.__name__})"


class TangentTensor(torch.Tensor):
    """Tangent vector with base point reference.
    
    This class represents a vector in the tangent space T_pM at a base point p.
    It stores both the vector and its base point for geometric operations.
    
    Attributes:
        manifold: The manifold
        base_point: The point at which this is a tangent vector
    """
    
    @staticmethod
    def __new__(cls, data: Tensor, manifold: Manifold, base_point: Tensor):
        """Create a new TangentTensor.
        
        Args:
            data: Tangent vector data
            manifold: Associated manifold
            base_point: Base point on manifold
        
        Returns:
            TangentTensor instance
        """
        instance = torch.Tensor._make_subclass(cls, data)
        instance.manifold = manifold
        instance.base_point = base_point
        return instance
    
    def exp(self) -> ManifoldTensor:
        """Apply exponential map to get point on manifold.
        
        Returns:
            Point on manifold reached by following geodesic
        """
        result = self.manifold.exp(self.base_point, self)
        return ManifoldTensor(result, self.manifold)
    
    def norm(self) -> Tensor:
        """Riemannian norm at base point.
        
        Returns:
            Norm of this tangent vector
        """
        return self.manifold.norm(self.base_point, self)
    
    def project_(self) -> 'TangentTensor':
        """Project onto tangent space in-place.
        
        Returns:
            Self (for chaining)
        """
        projected = self.manifold.project_tangent(self.base_point, self)
        self.data.copy_(projected)
        return self
    
    def parallel_transport_to(self, q: Tensor) -> 'TangentTensor':
        """Parallel transport to another point.
        
        Args:
            q: Destination point on manifold
        
        Returns:
            Tangent vector at q
        """
        result = self.manifold.parallel_transport(self, self.base_point, q)
        return TangentTensor(result, self.manifold, q)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"TangentTensor({super().__repr__()}, at={self.base_point})"
