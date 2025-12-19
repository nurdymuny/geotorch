"""
DavisTensor Autograd - Gradient Functions
==========================================

Base class and basic backward functions for automatic differentiation.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import numpy as np


@dataclass
class SavedContext:
    """
    Saved tensors and metadata for backward pass.
    """
    tensors: Dict[str, np.ndarray] = field(default_factory=dict)
    manifolds: Dict[str, Any] = field(default_factory=dict)
    scalars: Dict[str, Any] = field(default_factory=dict)
    
    def save_for_backward(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                self.tensors[k] = v
            elif hasattr(v, 'exp'):  # Duck type check for Manifold
                self.manifolds[k] = v
            else:
                self.scalars[k] = v


class GradFn(ABC):
    """
    Base class for backward functions.
    
    Each operation defines how gradients flow backward.
    """
    
    @abstractmethod
    def apply(self, grad_output: np.ndarray) -> Tuple[Optional[np.ndarray], ...]:
        """
        Compute gradients w.r.t. inputs given gradient of output.
        
        Parameters
        ----------
        grad_output : np.ndarray
            Gradient of loss w.r.t. this operation's output
        
        Returns
        -------
        Tuple of gradients for each input (None if input doesn't require grad)
        """
        ...
    
    @property
    @abstractmethod
    def inputs(self) -> Tuple:
        """Input tensors to this operation."""
        ...


class AddBackward(GradFn):
    """Backward for addition: z = x + y"""
    
    def __init__(self, x, y):
        self._inputs = (x, y)
        self.ctx = SavedContext()
    
    @property
    def inputs(self) -> Tuple:
        return self._inputs
    
    def apply(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # ∂L/∂x = ∂L/∂z, ∂L/∂y = ∂L/∂z
        return grad_output, grad_output


class MulBackward(GradFn):
    """Backward for multiplication: z = x * y"""
    
    def __init__(self, x, y):
        self._inputs = (x, y)
        self.ctx = SavedContext()
        self.ctx.save_for_backward(x=x.numpy(), y=y.numpy())
    
    @property
    def inputs(self) -> Tuple:
        return self._inputs
    
    def apply(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = self.ctx.tensors['x']
        y = self.ctx.tensors['y']
        # ∂L/∂x = ∂L/∂z * y, ∂L/∂y = ∂L/∂z * x
        return grad_output * y, grad_output * x


class MatMulBackward(GradFn):
    """Backward for matrix multiplication: z = x @ y"""
    
    def __init__(self, x, y):
        self._inputs = (x, y)
        self.ctx = SavedContext()
        self.ctx.save_for_backward(x=x.numpy(), y=y.numpy())
    
    @property
    def inputs(self) -> Tuple:
        return self._inputs
    
    def apply(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = self.ctx.tensors['x']
        y = self.ctx.tensors['y']
        # z = x @ y
        # ∂L/∂x = ∂L/∂z @ y^T
        # ∂L/∂y = x^T @ ∂L/∂z
        grad_x = grad_output @ np.swapaxes(y, -2, -1)
        grad_y = np.swapaxes(x, -2, -1) @ grad_output
        return grad_x, grad_y


class SumBackward(GradFn):
    """Backward for sum: z = sum(x)"""
    
    def __init__(self, x, axis: Optional[int] = None):
        self._inputs = (x,)
        self.ctx = SavedContext()
        self.ctx.save_for_backward(shape=x.shape, axis=axis)
    
    @property
    def inputs(self) -> Tuple:
        return self._inputs
    
    def apply(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        shape = self.ctx.scalars['shape']
        # Gradient broadcasts back to input shape
        return (np.ones(shape) * grad_output,)


class ExpBackward(GradFn):
    """Backward for exp: z = exp(x)"""
    
    def __init__(self, x):
        self._inputs = (x,)
        self.ctx = SavedContext()
        self.ctx.save_for_backward(exp_x=np.exp(x.numpy()))
    
    @property
    def inputs(self) -> Tuple:
        return self._inputs
    
    def apply(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        exp_x = self.ctx.tensors['exp_x']
        # ∂exp(x)/∂x = exp(x)
        return (grad_output * exp_x,)


class LogBackward(GradFn):
    """Backward for log: z = log(x)"""
    
    def __init__(self, x):
        self._inputs = (x,)
        self.ctx = SavedContext()
        self.ctx.save_for_backward(x=x.numpy())
    
    @property
    def inputs(self) -> Tuple:
        return self._inputs
    
    def apply(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        x = self.ctx.tensors['x']
        # ∂log(x)/∂x = 1/x
        return (grad_output / x,)


class NegBackward(GradFn):
    """Backward for negation: z = -x"""
    
    def __init__(self, x):
        self._inputs = (x,)
        self.ctx = SavedContext()
    
    @property
    def inputs(self) -> Tuple:
        return self._inputs
    
    def apply(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        return (-grad_output,)


class DivBackward(GradFn):
    """Backward for division: z = x / y"""
    
    def __init__(self, x, y):
        self._inputs = (x, y)
        self.ctx = SavedContext()
        self.ctx.save_for_backward(x=x.numpy(), y=y.numpy())
    
    @property
    def inputs(self) -> Tuple:
        return self._inputs
    
    def apply(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = self.ctx.tensors['x']
        y = self.ctx.tensors['y']
        # z = x / y
        # ∂L/∂x = ∂L/∂z / y
        # ∂L/∂y = -∂L/∂z * x / y^2
        grad_x = grad_output / y
        grad_y = -grad_output * x / (y ** 2)
        return grad_x, grad_y


class PowBackward(GradFn):
    """Backward for power: z = x ** n"""
    
    def __init__(self, x, power: float):
        self._inputs = (x,)
        self.ctx = SavedContext()
        self.ctx.save_for_backward(x=x.numpy(), power=power)
    
    @property
    def inputs(self) -> Tuple:
        return self._inputs
    
    def apply(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        x = self.ctx.tensors['x']
        n = self.ctx.scalars['power']
        # ∂(x^n)/∂x = n * x^(n-1)
        return (grad_output * n * np.power(x, n - 1),)


class SqrtBackward(GradFn):
    """Backward for sqrt: z = sqrt(x)"""
    
    def __init__(self, x):
        self._inputs = (x,)
        self.ctx = SavedContext()
        self.ctx.save_for_backward(sqrt_x=np.sqrt(x.numpy()))
    
    @property
    def inputs(self) -> Tuple:
        return self._inputs
    
    def apply(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        sqrt_x = self.ctx.tensors['sqrt_x']
        # ∂sqrt(x)/∂x = 1/(2*sqrt(x))
        return (grad_output / (2 * sqrt_x + 1e-10),)


class MeanBackward(GradFn):
    """Backward for mean: z = mean(x)"""
    
    def __init__(self, x, axis: Optional[int] = None):
        self._inputs = (x,)
        self.ctx = SavedContext()
        self.ctx.save_for_backward(shape=x.shape, axis=axis)
    
    @property
    def inputs(self) -> Tuple:
        return self._inputs
    
    def apply(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        shape = self.ctx.scalars['shape']
        axis = self.ctx.scalars['axis']
        
        if axis is None:
            n = np.prod(shape)
        else:
            n = shape[axis]
        
        # Gradient is 1/n broadcast to input shape
        return (np.ones(shape) * grad_output / n,)
