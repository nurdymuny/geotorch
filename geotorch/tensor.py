"""Tensor wrappers for manifold-valued data."""

import torch
from torch import Tensor
from typing import Optional
from .manifold import Manifold


class ManifoldTensor(torch.Tensor):
    """
    Tensor that lives on a manifold.
    
    Wraps a torch.Tensor with an associated manifold, enabling
    geometric operations like exp, log, and geodesic distance.
    
    Example:
        >>> S = Sphere(64)
        >>> p = ManifoldTensor(S.random_point(), manifold=S)
        >>> v = S.random_tangent(p)
        >>> q = p.exp(v)  # Move along geodesic
        >>> print(p.distance(q))  # Geodesic distance
    """
    
    @staticmethod
    def __new__(cls, data, manifold: Manifold, **kwargs):
        """
        Create a new ManifoldTensor.
        
        Args:
            data: Tensor data (must be on the manifold)
            manifold: Associated Riemannian manifold
            **kwargs: Additional arguments for torch.Tensor
        """
        # Create tensor as subclass of torch.Tensor
        if isinstance(data, torch.Tensor):
            tensor = torch.Tensor._make_subclass(cls, data)
        else:
            tensor = torch.Tensor._make_subclass(cls, torch.as_tensor(data, **kwargs))
        
        # Store manifold as an attribute
        tensor.manifold = manifold
        return tensor
    
    def __repr__(self):
        return f"ManifoldTensor({super().__repr__()}, manifold={self.manifold.__class__.__name__})"
    
    def __add__(self, other):
        """Addition with TangentTensor or other tensors."""
        if isinstance(other, (TangentTensor, ManifoldTensor)):
            # Return underlying tensor addition (loses manifold structure)
            return torch.Tensor.__add__(self, other)
        return torch.Tensor.__add__(self, other)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __mul__(self, other):
        """Scalar multiplication."""
        result = torch.Tensor.__mul__(self, other)
        # For scalar multiplication, try to preserve manifold structure
        if isinstance(other, (int, float)):
            return result  # But this may not be on manifold anymore
        return result
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        Handle torch function calls to preserve ManifoldTensor type.
        """
        if kwargs is None:
            kwargs = {}
        
        # For most operations, fall back to standard Tensor behavior
        # This loses the manifold structure, which is intentional for operations
        # that don't preserve the manifold (like addition, multiplication, etc.)
        ret = super().__torch_function__(func, types, args, kwargs)
        return ret
    
    def project_(self) -> 'ManifoldTensor':
        """
        Project onto manifold in-place.
        
        Returns:
            Self after projection
        """
        projected = self.manifold.project(self)
        self.data.copy_(projected)
        return self
    
    def exp(self, v: Tensor) -> 'ManifoldTensor':
        """
        Move along geodesic with velocity v.
        
        Args:
            v: Tangent vector at this point
        
        Returns:
            New point on manifold after exponential map
        """
        # Convert to plain torch.Tensor to avoid type issues in manifold methods
        p_data = torch.Tensor(self)
        v_data = torch.Tensor(v) if isinstance(v, torch.Tensor) else v
        result = self.manifold.exp(p_data, v_data)
        return ManifoldTensor(result, manifold=self.manifold)
    
    def log(self, q: 'ManifoldTensor') -> Tensor:
        """
        Tangent vector pointing toward q.
        
        Args:
            q: Target point on manifold
        
        Returns:
            Tangent vector v such that self.exp(v) = q
        """
        if not isinstance(q, ManifoldTensor):
            raise TypeError("Argument must be a ManifoldTensor")
        if q.manifold != self.manifold:
            raise ValueError("Points must be on the same manifold")
        
        # Convert to plain torch.Tensor
        p_data = torch.Tensor(self)
        q_data = torch.Tensor(q)
        return self.manifold.log(p_data, q_data)
    
    def distance(self, q: 'ManifoldTensor') -> Tensor:
        """
        Geodesic distance to q.
        
        Args:
            q: Target point on manifold
        
        Returns:
            Geodesic distance as a scalar tensor
        """
        if not isinstance(q, ManifoldTensor):
            raise TypeError("Argument must be a ManifoldTensor")
        if q.manifold != self.manifold:
            raise ValueError("Points must be on the same manifold")
        
        # Convert to plain torch.Tensor
        p_data = torch.Tensor(self)
        q_data = torch.Tensor(q)
        return self.manifold.distance(p_data, q_data)
    
    def geodesic_to(self, q: 'ManifoldTensor', t: float) -> 'ManifoldTensor':
        """
        Interpolate toward q along geodesic.
        
        Args:
            q: Target point on manifold
            t: Parameter in [0, 1], where 0 returns self and 1 returns q
        
        Returns:
            Point along geodesic at parameter t
        """
        if not isinstance(q, ManifoldTensor):
            raise TypeError("Argument must be a ManifoldTensor")
        if q.manifold != self.manifold:
            raise ValueError("Points must be on the same manifold")
        
        result = self.manifold.geodesic(self, q, t)
        return ManifoldTensor(result, manifold=self.manifold)


class TangentTensor(torch.Tensor):
    """
    Tangent vector with base point reference.
    
    A tangent vector v ∈ T_pM needs to know its base point p
    for operations like parallel transport.
    """
    
    @staticmethod
    def __new__(cls, data, base_point: Tensor, manifold: Manifold, **kwargs):
        """
        Create a new TangentTensor.
        
        Args:
            data: Tensor data (must be in tangent space at base_point)
            base_point: Base point on manifold where this tangent vector lives
            manifold: Associated Riemannian manifold
            **kwargs: Additional arguments for torch.Tensor
        """
        # Create tensor as subclass of torch.Tensor
        if isinstance(data, torch.Tensor):
            tensor = torch.Tensor._make_subclass(cls, data)
        else:
            tensor = torch.Tensor._make_subclass(cls, torch.as_tensor(data, **kwargs))
        
        # Store base point and manifold as attributes
        tensor.base_point = base_point
        tensor.manifold = manifold
        return tensor
    
    def __repr__(self):
        return (f"TangentTensor({super().__repr__()}, "
                f"manifold={self.manifold.__class__.__name__})")
    
    def __add__(self, other):
        """Addition with ManifoldTensor or other TangentTensor."""
        if isinstance(other, ManifoldTensor):
            # Return underlying tensor addition (loses tangent structure)
            return torch.Tensor.__add__(self, other)
        elif isinstance(other, TangentTensor):
            result = torch.Tensor.__add__(self, other)
            return TangentTensor(result, base_point=self.base_point, manifold=self.manifold)
        return torch.Tensor.__add__(self, other)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __mul__(self, other):
        """Scalar multiplication or interaction with ManifoldTensor."""
        if isinstance(other, (int, float)):
            result = torch.Tensor.__mul__(self, other)
            return TangentTensor(result, base_point=self.base_point, manifold=self.manifold)
        elif isinstance(other, ManifoldTensor):
            # Return underlying tensor multiplication
            return torch.Tensor.__mul__(self, other)
        elif isinstance(other, torch.Tensor) and not isinstance(other, (ManifoldTensor, TangentTensor)):
            result = torch.Tensor.__mul__(self, other)
            return result
        return torch.Tensor.__mul__(self, other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        Handle torch function calls to preserve TangentTensor type.
        """
        if kwargs is None:
            kwargs = {}
        
        # For most operations, fall back to standard Tensor behavior
        ret = super().__torch_function__(func, types, args, kwargs)
        return ret
    
    def parallel_transport(self, q: Tensor) -> 'TangentTensor':
        """
        Parallel transport to tangent space at q.
        
        Args:
            q: Destination point on manifold
        
        Returns:
            Transported tangent vector at q
        """
        transported = self.manifold.parallel_transport(self, self.base_point, q)
        return TangentTensor(transported, base_point=q, manifold=self.manifold)
    
    def norm(self) -> Tensor:
        """
        Riemannian norm of this tangent vector.
        
        Returns:
            ||self||_p where p is the base point
        """
        return self.manifold.norm(self.base_point, self)
