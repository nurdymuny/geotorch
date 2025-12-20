"""Riemannian Stochastic Gradient Descent optimizer."""

import torch
from torch.optim import Optimizer
from typing import List, Optional, Callable, Iterable, Union
from ..nn import ManifoldParameter


class RiemannianSGD(Optimizer):
    """
    Riemannian Stochastic Gradient Descent with momentum and geodesic updates.
    
    Performs optimization on Riemannian manifolds using the exponential map
    for parameter updates. For standard Euclidean parameters, falls back to
    standard SGD behavior.
    
    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float): Learning rate (required).
        momentum (float, optional): Momentum factor (default: 0). When non-zero,
            uses parallel transport to move momentum vectors between tangent spaces.
        dampening (float, optional): Dampening for momentum (default: 0).
        weight_decay (float, optional): Weight decay (L2 penalty) (default: 0).
            Applied in tangent space before exponential map update.
        nesterov (bool, optional): Enables Nesterov momentum (default: False).
        grad_clip (float, optional): Maximum norm for gradient clipping in
            tangent space (default: None). If specified, gradients are clipped
            before update.
        stabilize (bool, optional): Apply periodic manifold projection to
            counteract numerical drift (default: True).
    
    Example:
        >>> from geotorch import Sphere
        >>> from geotorch.nn import ManifoldParameter
        >>> from geotorch.optim import RiemannianSGD
        >>> 
        >>> manifold = Sphere(64)
        >>> param = ManifoldParameter(manifold.random_point(), manifold)
        >>> optimizer = RiemannianSGD([param], lr=0.01, momentum=0.9)
        >>> 
        >>> optimizer.zero_grad()
        >>> loss = compute_loss(param)
        >>> loss.backward()
        >>> optimizer.step()
    
    Notes:
        - For ManifoldParameter instances, gradients are automatically projected
          to tangent space and updates use the manifold's exponential map.
        - For standard nn.Parameter instances, behaves identically to torch.optim.SGD.
        - Momentum vectors are parallel transported when moving to new points on
          the manifold, preserving their geometric meaning.
    """
    
    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], Iterable[dict]],
        lr: float,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
        grad_clip: Optional[float] = None,
        stabilize: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if grad_clip is not None and grad_clip <= 0.0:
            raise ValueError(f"Invalid grad_clip value: {grad_clip}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            grad_clip=grad_clip,
            stabilize=stabilize,
        )
        super(RiemannianSGD, self).__init__(params, defaults)
        
        self._step_count = 0
    
    def __setstate__(self, state):
        """Restore optimizer state."""
        super(RiemannianSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('grad_clip', None)
            group.setdefault('stabilize', True)
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Performs a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss.
        
        Returns:
            The loss value if closure is provided, None otherwise.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            grad_clip = group['grad_clip']
            lr = group['lr']
            stabilize = group['stabilize']
            
            for param in group['params']:
                if param.grad is None:
                    continue
                
                grad = param.grad.data
                state = self.state[param]
                
                # Check if this is a manifold parameter
                is_manifold = isinstance(param, ManifoldParameter)
                
                if is_manifold:
                    # Project gradient to tangent space
                    # (Note: gradient should already be projected by the hook,
                    # but we ensure it here for safety)
                    manifold = param.manifold
                    grad = manifold.project_tangent(param.data, grad)
                
                # Apply weight decay in tangent space
                if weight_decay != 0:
                    grad = grad.add(param.data, alpha=weight_decay)
                
                # Gradient clipping in tangent space
                if grad_clip is not None:
                    grad_norm = grad.norm()
                    if grad_norm > grad_clip:
                        grad = grad * (grad_clip / grad_norm)
                
                # Momentum
                if momentum != 0:
                    if 'momentum_buffer' not in state:
                        buf = state['momentum_buffer'] = torch.zeros_like(grad)
                    else:
                        buf = state['momentum_buffer']
                        
                        # Parallel transport momentum if on manifold
                        if is_manifold and 'prev_point' in state:
                            prev_point = state['prev_point']
                            buf = manifold.parallel_transport(buf, prev_point, param.data)
                            state['momentum_buffer'] = buf
                    
                    buf.mul_(momentum).add_(grad, alpha=1 - dampening)
                    
                    if nesterov:
                        grad = grad.add(buf, alpha=momentum)
                    else:
                        grad = buf
                
                # Store current point for next momentum transport
                if is_manifold and momentum != 0:
                    state['prev_point'] = param.data.clone()
                
                # Update parameter
                if is_manifold:
                    # Geodesic update via exponential map
                    param.data = manifold.exp(param.data, -lr * grad)
                    
                    # Periodic stabilization (project back to manifold)
                    if stabilize and self._step_count % 10 == 0:
                        param.data = manifold.project(param.data)
                else:
                    # Standard Euclidean update
                    param.data.add_(grad, alpha=-lr)
        
        self._step_count += 1
        return loss
