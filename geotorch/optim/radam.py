"""Riemannian Adam optimizer."""

import math
import torch
from torch.optim import Optimizer
from typing import Tuple, Optional, Callable, Iterable, Union
from ..nn import ManifoldParameter


class RiemannianAdam(Optimizer):
    """
    Riemannian Adam optimizer with adaptive learning rates and parallel transport.
    
    Implements Adam optimization on Riemannian manifolds by maintaining first and
    second moment estimates in tangent spaces and using parallel transport to move
    these estimates between tangent spaces as parameters evolve.
    
    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): Learning rate (default: 1e-3).
        betas (Tuple[float, float], optional): Coefficients for computing running
            averages of gradient and its square (default: (0.9, 0.999)).
        eps (float, optional): Term added to denominator for numerical stability
            (default: 1e-8).
        weight_decay (float, optional): Weight decay coefficient (default: 0).
            Applied as Riemannian regularization in tangent space.
        amsgrad (bool, optional): Whether to use AMSGrad variant (default: False).
        stabilize (bool, optional): Apply periodic manifold projection (default: True).
        stabilize_period (int, optional): Number of steps between stabilization
            projections (default: 10).
    
    Example:
        >>> from geotorch import Sphere
        >>> from geotorch.nn import ManifoldParameter
        >>> from geotorch.optim import RiemannianAdam
        >>> 
        >>> manifold = Sphere(64)
        >>> param = ManifoldParameter(manifold.random_point(), manifold)
        >>> optimizer = RiemannianAdam([param], lr=1e-3, betas=(0.9, 0.999))
        >>> 
        >>> for epoch in range(num_epochs):
        >>>     optimizer.zero_grad()
        >>>     loss = model(data)
        >>>     loss.backward()
        >>>     optimizer.step()
    
    Notes:
        - First and second moment estimates (m_t, v_t) are stored in tangent space
        - Moments are parallel transported to new tangent space after each update
        - Bias correction is applied as in standard Adam
        - For standard parameters, behaves identically to torch.optim.Adam
    """
    
    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], Iterable[dict]],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        stabilize: bool = True,
        stabilize_period: int = 10,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if stabilize_period <= 0:
            raise ValueError(f"Invalid stabilize_period: {stabilize_period}")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            stabilize=stabilize,
            stabilize_period=stabilize_period,
        )
        super(RiemannianAdam, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        """Restore optimizer state."""
        super(RiemannianAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('stabilize', True)
            group.setdefault('stabilize_period', 10)
    
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
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            weight_decay = group['weight_decay']
            amsgrad = group['amsgrad']
            stabilize = group['stabilize']
            stabilize_period = group['stabilize_period']
            
            for param in group['params']:
                if param.grad is None:
                    continue
                
                grad = param.grad.data
                
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
                
                # State initialization
                state = self.state[param]
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(grad)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(grad)
                else:
                    # Parallel transport moments if on manifold
                    if is_manifold and 'prev_point' in state:
                        prev = state['prev_point']
                        # Transport first moment
                        state['exp_avg'] = manifold.parallel_transport(
                            state['exp_avg'], prev, param.data
                        )
                        # For second moment, we use a simpler approach:
                        # Element-wise absolute value after transport to ensure non-negativity
                        # TODO: Investigate more sophisticated second moment transport methods
                        # See: Becigneul & Ganea (2019) "Riemannian Adaptive Optimization Methods"
                        # for potential improvements using vector transport or retraction-based methods
                        state['exp_avg_sq'] = torch.abs(manifold.parallel_transport(
                            state['exp_avg_sq'], prev, param.data
                        ))
                        if amsgrad:
                            # Transport max second moment similarly
                            state['max_exp_avg_sq'] = torch.abs(manifold.parallel_transport(
                                state['max_exp_avg_sq'], prev, param.data
                            ))
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                
                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(eps)
                else:
                    denom = exp_avg_sq.sqrt().add_(eps)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                
                # Compute update direction
                direction = -step_size * (exp_avg / denom)
                
                # Store current point for next transport
                if is_manifold:
                    state['prev_point'] = param.data.clone()
                
                # Apply update
                if is_manifold:
                    # Geodesic update via exponential map
                    param.data = manifold.exp(param.data, direction)
                    
                    # Periodic stabilization (project back to manifold)
                    if stabilize and state['step'] % stabilize_period == 0:
                        param.data = manifold.project(param.data)
                else:
                    # Standard Euclidean update
                    param.data.add_(direction)
        
        return loss
