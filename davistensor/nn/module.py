"""
DavisTensor NN - Base Module and Parameters
============================================

Base classes for neural network modules and parameters.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np


class Parameter:
    """
    Learnable parameter.
    
    Like torch.nn.Parameter - a tensor that should be included
    in module.parameters() for optimization.
    """
    
    def __init__(
        self, 
        data: Union[np.ndarray, 'TensorCore'],
        requires_grad: bool = True
    ):
        from ..core.storage import tensor
        
        if hasattr(data, 'numpy'):
            self._data = data
        else:
            self._data = tensor(data)
        
        self._data.requires_grad = requires_grad
    
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, value):
        if isinstance(value, np.ndarray):
            self._data._storage._data[:] = value.flatten()
        else:
            self._data = value
    
    @property
    def grad(self):
        return self._data.grad
    
    def numpy(self) -> np.ndarray:
        return self._data.numpy()
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self._data.shape
    
    def zero_grad(self):
        """Reset gradient to None."""
        self._data.grad = None
    
    def __repr__(self) -> str:
        return f"Parameter({self.shape})"


class ManifoldParameter(Parameter):
    """
    Learnable parameter constrained to a manifold.
    
    After each optimization step, project back to manifold.
    """
    
    def __init__(
        self,
        data: Union[np.ndarray, 'TensorCore'],
        manifold,
        requires_grad: bool = True
    ):
        super().__init__(data, requires_grad)
        self._manifold = manifold
        
        # Project to manifold on creation
        projected = manifold.project_point(self._data.numpy())
        self._data.storage._data[:] = projected.flatten()
        self._data.manifold = manifold
    
    @property
    def manifold(self):
        return self._manifold
    
    def project(self):
        """Project parameter back to manifold (call after optimizer step)."""
        projected = self._manifold.project_point(self._data.numpy())
        self._data.storage._data[:] = projected.flatten()
    
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
        if name.startswith('_'):
            super().__setattr__(name, value)
        elif isinstance(value, Parameter):
            if not hasattr(self, '_parameters'):
                super().__setattr__('_parameters', {})
            self._parameters[name] = value
            super().__setattr__(name, value)
        elif isinstance(value, Module):
            if not hasattr(self, '_modules'):
                super().__setattr__('_modules', {})
            self._modules[name] = value
            super().__setattr__(name, value)
        else:
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
    
    def zero_grad(self):
        """Reset all parameter gradients to None."""
        for param in self.parameters():
            param.zero_grad()
    
    def project_manifold_parameters(self):
        """Project all manifold parameters back to their manifolds."""
        for param in self.manifold_parameters():
            param.project()
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass - must be implemented by subclasses."""
        ...
    
    def __call__(self, *args, **kwargs):
        """Call forward()."""
        return self.forward(*args, **kwargs)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
