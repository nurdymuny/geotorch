"""Tensor wrappers for manifold-valued data."""

import torch
from torch import Tensor
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
        return f"ManifoldTensor({torch.Tensor.__repr__(self)}, manifold={self.manifold.__class__.__name__})"
    
    def __add__(self, other):
        """Addition with TangentTensor or other tensors."""
        # Use torch.Tensor methods directly to avoid __torch_function__ recursion
        if isinstance(other, (TangentTensor, ManifoldTensor)):
            return torch.Tensor.__add__(self, other)
        return torch.Tensor.__add__(self, other)
    
    def __radd__(self, other):
        """Reverse addition."""
        if isinstance(other, (TangentTensor, ManifoldTensor)):
            return torch.Tensor.__add__(other, self)
        return torch.Tensor.__radd__(self, other)
    
    def __mul__(self, other):
        """Multiplication with scalars or other tensors."""
        # Use torch.Tensor methods directly to avoid __torch_function__ recursion
        if isinstance(other, (TangentTensor, ManifoldTensor)):
            return torch.Tensor.__mul__(self, other)
        return torch.Tensor.__mul__(self, other)
    
    def __rmul__(self, other):
        """Reverse multiplication."""
        if isinstance(other, (TangentTensor, ManifoldTensor)):
            return torch.Tensor.__mul__(other, self)
        return torch.Tensor.__rmul__(self, other)
    
    def __truediv__(self, other):
        """Division by scalars or tensors."""
        # Use torch.Tensor methods directly to avoid __torch_function__ recursion
        if isinstance(other, (TangentTensor, ManifoldTensor)):
            return torch.Tensor.__truediv__(self, other)
        return torch.Tensor.__truediv__(self, other)
    
    def __rtruediv__(self, other):
        """Reverse division."""
        if isinstance(other, (TangentTensor, ManifoldTensor)):
            return torch.Tensor.__truediv__(other, self)
        return torch.Tensor.__rtruediv__(self, other)
    
    def __sub__(self, other):
        """Subtraction."""
        # Use torch.Tensor methods directly to avoid __torch_function__ recursion
        if isinstance(other, (TangentTensor, ManifoldTensor)):
            return torch.Tensor.__sub__(self, other)
        return torch.Tensor.__sub__(self, other)
    
    def __rsub__(self, other):
        """Reverse subtraction."""
        if isinstance(other, (TangentTensor, ManifoldTensor)):
            return torch.Tensor.__sub__(other, self)
        return torch.Tensor.__rsub__(self, other)
    
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
        # Create plain tensor view to pass to manifold methods
        # Use .detach() to get a plain tensor without triggering __torch_function__
        p_data = self.detach()
        projected = self.manifold.project(p_data)
        # Use torch.Tensor methods to avoid triggering __torch_function__
        torch.Tensor.copy_(self, projected)
        return self
    
    def exp(self, v: Tensor) -> 'ManifoldTensor':
        """
        Move along geodesic with velocity v.
        
        Args:
            v: Tangent vector at this point
        
        Returns:
            New point on manifold after exponential map
        """
        # Use .detach() to get plain tensor views
        p_data = self.detach()
        v_data = v.detach() if isinstance(v, (ManifoldTensor, TangentTensor)) else v
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
        
        # Use .detach() to get plain tensor views
        p_data = self.detach()
        q_data = q.detach()
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
        
        # Use .detach() to get plain tensor views
        p_data = self.detach()
        q_data = q.detach()
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
        
        # Use .detach() to get plain tensor views
        p_data = self.detach()
        q_data = q.detach()
        result = self.manifold.geodesic(p_data, q_data, t)
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
        return (f"TangentTensor({torch.Tensor.__repr__(self)}, "
                f"manifold={self.manifold.__class__.__name__})")
    
    def __add__(self, other):
        """Addition with ManifoldTensor or other tensors."""
        # Use torch.Tensor methods directly to avoid __torch_function__ recursion
        if isinstance(other, (ManifoldTensor, TangentTensor)):
            return torch.Tensor.__add__(self, other)
        return torch.Tensor.__add__(self, other)
    
    def __radd__(self, other):
        """Reverse addition."""
        if isinstance(other, (ManifoldTensor, TangentTensor)):
            return torch.Tensor.__add__(other, self)
        return torch.Tensor.__radd__(self, other)
    
    def __mul__(self, other):
        """Multiplication with scalars or other tensors."""
        # Use torch.Tensor methods directly to avoid __torch_function__ recursion
        if isinstance(other, (ManifoldTensor, TangentTensor)):
            return torch.Tensor.__mul__(self, other)
        return torch.Tensor.__mul__(self, other)
    
    def __rmul__(self, other):
        """Reverse multiplication."""
        if isinstance(other, (ManifoldTensor, TangentTensor)):
            return torch.Tensor.__mul__(other, self)
        return torch.Tensor.__rmul__(self, other)
    
    def __truediv__(self, other):
        """Division by scalars or tensors."""
        # Use torch.Tensor methods directly to avoid __torch_function__ recursion
        if isinstance(other, (ManifoldTensor, TangentTensor)):
            return torch.Tensor.__truediv__(self, other)
        return torch.Tensor.__truediv__(self, other)
    
    def __rtruediv__(self, other):
        """Reverse division."""
        if isinstance(other, (ManifoldTensor, TangentTensor)):
            return torch.Tensor.__truediv__(other, self)
        return torch.Tensor.__rtruediv__(self, other)
    
    def __sub__(self, other):
        """Subtraction."""
        # Use torch.Tensor methods directly to avoid __torch_function__ recursion
        if isinstance(other, (ManifoldTensor, TangentTensor)):
            return torch.Tensor.__sub__(self, other)
        return torch.Tensor.__sub__(self, other)
    
    def __rsub__(self, other):
        """Reverse subtraction."""
        if isinstance(other, (ManifoldTensor, TangentTensor)):
            return torch.Tensor.__sub__(other, self)
        return torch.Tensor.__rsub__(self, other)
    
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
