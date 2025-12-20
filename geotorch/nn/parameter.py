"""ManifoldParameter: nn.Parameter constrained to a Riemannian manifold."""

import torch
from torch import nn
from torch import Tensor
from typing import Optional


class ManifoldParameter(nn.Parameter):
    """
    Neural network parameter constrained to a Riemannian manifold.
    
    This class extends torch.nn.Parameter with manifold awareness, automatically
    projecting gradients to the tangent space during backpropagation. When used
    with Riemannian optimizers (RiemannianSGD, RiemannianAdam), parameter updates
    are performed via geodesic flows using the exponential map.
    
    Args:
        data: Initial parameter value (will be projected onto manifold if needed)
        manifold: The Riemannian manifold on which this parameter lives
        requires_grad: Whether this parameter requires gradients (default: True)
    
    Example:
        >>> from geotorch import Sphere
        >>> from geotorch.nn import ManifoldParameter
        >>> 
        >>> manifold = Sphere(64)
        >>> param = ManifoldParameter(manifold.random_point(), manifold)
        >>> 
        >>> # Use in a model
        >>> class MyModel(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.weight = ManifoldParameter(manifold.random_point(), manifold)
        ...     
        ...     def forward(self, x):
        ...         return x @ self.weight
    
    Notes:
        - Gradients are automatically projected to tangent space via a registered hook
        - Compatible with all standard PyTorch training utilities
        - Works seamlessly with both Riemannian and standard optimizers
        - Preserves all functionality of nn.Parameter (state_dict, etc.)
    """
    
    def __new__(cls, data: Tensor, manifold, requires_grad: bool = True):
        """Create a new ManifoldParameter instance."""
        # Ensure data is on the manifold
        data = torch.as_tensor(data)
        data_projected = manifold.project(data)
        
        # Create the parameter using the parent constructor
        instance = super().__new__(cls, data_projected, requires_grad=requires_grad)
        
        # Store the manifold (cannot use __init__ for nn.Parameter subclasses)
        # We attach it as a non-persistent attribute
        instance.manifold = manifold
        
        # Register gradient projection hook if gradients are required
        if requires_grad:
            # Create a closure to capture the manifold
            def grad_projection_hook(grad):
                """Project gradient to tangent space during backward pass."""
                if grad is None:
                    return None
                # Project to tangent space at current parameter value
                return manifold.project_tangent(instance.data, grad)
            
            instance.register_hook(grad_projection_hook)
        
        return instance
    
    def __repr__(self):
        """String representation of ManifoldParameter."""
        return (
            f"ManifoldParameter containing:\n"
            f"{self.data}\n"
            f"Manifold: {self.manifold.__class__.__name__}"
        )
    
    def __reduce_ex__(self, proto):
        """
        Custom serialization to preserve manifold information.
        
        This ensures that the manifold is properly saved and restored
        when using torch.save() and torch.load().
        """
        # Get the base Parameter's reduction
        base_reduce = super().__reduce_ex__(proto)
        
        # Return a tuple that includes manifold reconstruction
        return (
            _rebuild_manifold_parameter,
            (base_reduce, self.manifold, self.requires_grad),
        )


def _rebuild_manifold_parameter(base_reduce, manifold, requires_grad):
    """Helper function to rebuild ManifoldParameter from serialized state."""
    # First rebuild the base Parameter
    base_rebuild_fn = base_reduce[0]
    base_args = base_reduce[1]
    param_data = base_rebuild_fn(*base_args)
    
    # Create a new ManifoldParameter with the reconstructed data
    return ManifoldParameter(param_data, manifold, requires_grad)
