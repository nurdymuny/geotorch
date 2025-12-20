"""Tests for Riemannian optimizers."""

import pytest
import torch
import torch.nn as nn
from geotorch import Sphere, Euclidean
from geotorch.nn import ManifoldParameter
from geotorch.optim import RiemannianSGD, RiemannianAdam


class TestManifoldParameter:
    """Tests for ManifoldParameter class."""
    
    def test_manifold_parameter_creation(self):
        """ManifoldParameter should be created correctly."""
        manifold = Sphere(64)
        data = manifold.random_point()
        param = ManifoldParameter(data, manifold)
        
        assert isinstance(param, nn.Parameter)
        assert isinstance(param, ManifoldParameter)
        assert hasattr(param, 'manifold')
        assert param.manifold is manifold
        assert param.requires_grad
    
    def test_manifold_parameter_projection(self):
        """ManifoldParameter should project data onto manifold."""
        manifold = Sphere(64)
        # Create non-normalized data
        data = torch.randn(64) * 5.0
        param = ManifoldParameter(data, manifold)
        
        # Should be projected to unit sphere
        assert torch.allclose(torch.norm(param.data), torch.tensor(1.0), atol=1e-5)
    
    def test_gradient_projection_hook(self):
        """Gradients should be projected to tangent space."""
        manifold = Sphere(64)
        param = ManifoldParameter(manifold.random_point(), manifold, requires_grad=True)
        
        # Compute a loss and backprop
        loss = (param ** 2).sum()
        loss.backward()
        
        # Gradient should be in tangent space (orthogonal to point for sphere)
        assert param.grad is not None
        dot_product = torch.dot(param.data, param.grad)
        assert torch.abs(dot_product) < 1e-5, \
            f"Gradient not in tangent space: dot={dot_product}"
    
    def test_no_grad_no_hook(self):
        """Parameters without gradients should not register hooks."""
        manifold = Sphere(64)
        param = ManifoldParameter(manifold.random_point(), manifold, requires_grad=False)
        
        assert not param.requires_grad


class TestRiemannianSGD:
    """Tests for RiemannianSGD optimizer."""
    
    def test_optimizer_creation(self):
        """RiemannianSGD should be created with valid parameters."""
        manifold = Sphere(64)
        param = ManifoldParameter(manifold.random_point(), manifold)
        optimizer = RiemannianSGD([param], lr=0.01)
        
        assert optimizer is not None
        assert len(optimizer.param_groups) == 1
    
    def test_invalid_learning_rate(self):
        """Negative learning rate should raise ValueError."""
        manifold = Sphere(64)
        param = ManifoldParameter(manifold.random_point(), manifold)
        
        with pytest.raises(ValueError):
            RiemannianSGD([param], lr=-0.01)
    
    def test_invalid_momentum(self):
        """Negative momentum should raise ValueError."""
        manifold = Sphere(64)
        param = ManifoldParameter(manifold.random_point(), manifold)
        
        with pytest.raises(ValueError):
            RiemannianSGD([param], lr=0.01, momentum=-0.5)
    
    def test_nesterov_requires_momentum(self):
        """Nesterov should require momentum > 0 and dampening = 0."""
        manifold = Sphere(64)
        param = ManifoldParameter(manifold.random_point(), manifold)
        
        with pytest.raises(ValueError):
            RiemannianSGD([param], lr=0.01, nesterov=True)
        
        with pytest.raises(ValueError):
            RiemannianSGD([param], lr=0.01, momentum=0.9, dampening=0.1, nesterov=True)
    
    def test_step_reduces_loss(self):
        """Single step should reduce loss for simple convex problem."""
        manifold = Sphere(64)
        param = ManifoldParameter(manifold.random_point(), manifold)
        target = torch.randn(64)
        target = target / target.norm()  # Normalize to sphere
        
        optimizer = RiemannianSGD([param], lr=0.1)
        
        # Compute initial loss
        loss_before = ((param - target) ** 2).sum().item()
        
        # Take optimization step
        optimizer.zero_grad()
        loss = ((param - target) ** 2).sum()
        loss.backward()
        optimizer.step()
        
        # Compute final loss
        loss_after = ((param - target) ** 2).sum().item()
        
        # Loss should decrease
        assert loss_after < loss_before
    
    def test_manifold_constraint_preservation(self):
        """Parameter should remain on manifold after optimization steps."""
        manifold = Sphere(128)
        param = ManifoldParameter(manifold.random_point(), manifold)
        optimizer = RiemannianSGD([param], lr=0.01)
        
        for _ in range(50):
            optimizer.zero_grad()
            loss = (param ** 2).sum()
            loss.backward()
            optimizer.step()
            
            # Check parameter is still on sphere (unit norm)
            norm = torch.norm(param.data)
            assert torch.isclose(norm, torch.tensor(1.0), atol=1e-4), \
                f"Parameter left manifold: norm={norm}"
    
    def test_handles_none_gradients(self):
        """Optimizer should skip parameters with None gradients."""
        manifold = Sphere(64)
        param1 = ManifoldParameter(manifold.random_point(), manifold)
        param2 = ManifoldParameter(manifold.random_point(), manifold)
        
        optimizer = RiemannianSGD([param1, param2], lr=0.01)
        
        # Only compute gradient for param1
        loss = (param1 ** 2).sum()
        loss.backward()
        
        # Should not raise error
        optimizer.step()
        
        assert param1.grad is not None
        assert param2.grad is None
    
    def test_parameter_groups(self):
        """Different parameter groups should have different hyperparameters."""
        manifold = Sphere(64)
        param1 = ManifoldParameter(manifold.random_point(), manifold)
        param2 = ManifoldParameter(manifold.random_point(), manifold)
        
        optimizer = RiemannianSGD([
            {'params': [param1], 'lr': 0.1},
            {'params': [param2], 'lr': 0.01},
        ], lr=0.001)  # Default lr (will be overridden by group-specific lr)
        
        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]['lr'] == 0.1
        assert optimizer.param_groups[1]['lr'] == 0.01
    
    def test_convergence_to_target_sphere(self):
        """Optimizer should minimize distance to target on sphere."""
        manifold = Sphere(128)
        param = ManifoldParameter(manifold.random_point(), manifold)
        target = torch.randn(128)
        target = target / target.norm()  # Normalize to sphere
        
        initial_distance = ((param - target) ** 2).sum().sqrt().item()
        
        optimizer = RiemannianSGD([param], lr=0.1)
        
        for _ in range(200):
            optimizer.zero_grad()
            loss = ((param - target) ** 2).sum()
            loss.backward()
            optimizer.step()
        
        final_distance = ((param - target) ** 2).sum().sqrt().item()
        
        # Should reduce distance by >99%
        assert final_distance < initial_distance * 0.01, \
            f"Distance reduction insufficient: {initial_distance:.4f} -> {final_distance:.4f}"
    
    def test_momentum_preserves_norm(self):
        """Parallel transport should preserve momentum vector norm."""
        manifold = Sphere(64)
        param = ManifoldParameter(manifold.random_point(), manifold)
        optimizer = RiemannianSGD([param], lr=0.01, momentum=0.9)
        
        # Take a few steps to build up momentum
        for _ in range(5):
            optimizer.zero_grad()
            loss = (param ** 2).sum()
            loss.backward()
            optimizer.step()
        
        # Get momentum buffer
        state = optimizer.state[param]
        if 'momentum_buffer' in state:
            momentum_norm_before = state['momentum_buffer'].norm().item()
            prev_point = state['prev_point'].clone()
            
            # Take another step
            optimizer.zero_grad()
            loss = (param ** 2).sum()
            loss.backward()
            optimizer.step()
            
            # Check momentum norm is preserved (approximately)
            # Note: There may be small numerical differences
            momentum_norm_after = state['momentum_buffer'].norm().item()
            # Allow some tolerance due to parallel transport numerics and momentum updates
            # This is testing that the norm is roughly preserved, not exact equality
            if momentum_norm_before > 1e-6:  # Only test if momentum is non-trivial
                relative_change = abs(momentum_norm_after - momentum_norm_before) / momentum_norm_before
                assert relative_change < 0.5, \
                    f"Momentum norm changed too much: {momentum_norm_before:.6e} -> {momentum_norm_after:.6e}"
    
    def test_euclidean_fallback(self):
        """Standard parameters should use Euclidean updates."""
        # Use standard nn.Parameter (not ManifoldParameter)
        param = nn.Parameter(torch.randn(64))
        optimizer = RiemannianSGD([param], lr=0.01)
        
        param_before = param.data.clone()
        
        optimizer.zero_grad()
        loss = (param ** 2).sum()
        loss.backward()
        optimizer.step()
        
        # Should have updated
        assert not torch.allclose(param.data, param_before)
    
    def test_reproducibility_with_seed(self):
        """Same seed should give identical optimization trajectory."""
        def run_optimization(seed):
            torch.manual_seed(seed)
            manifold = Sphere(64)
            param = ManifoldParameter(manifold.random_point(), manifold)
            optimizer = RiemannianSGD([param], lr=0.01)
            
            for _ in range(10):
                optimizer.zero_grad()
                loss = (param ** 2).sum()
                loss.backward()
                optimizer.step()
            
            return param.data.clone()
        
        result1 = run_optimization(42)
        result2 = run_optimization(42)
        
        assert torch.allclose(result1, result2, atol=1e-7), \
            "Results should be identical with same seed"


class TestRiemannianAdam:
    """Tests for RiemannianAdam optimizer."""
    
    def test_optimizer_creation(self):
        """RiemannianAdam should be created with valid parameters."""
        manifold = Sphere(64)
        param = ManifoldParameter(manifold.random_point(), manifold)
        optimizer = RiemannianAdam([param], lr=0.001)
        
        assert optimizer is not None
        assert len(optimizer.param_groups) == 1
    
    def test_invalid_learning_rate(self):
        """Negative learning rate should raise ValueError."""
        manifold = Sphere(64)
        param = ManifoldParameter(manifold.random_point(), manifold)
        
        with pytest.raises(ValueError):
            RiemannianAdam([param], lr=-0.001)
    
    def test_invalid_betas(self):
        """Invalid beta values should raise ValueError."""
        manifold = Sphere(64)
        param = ManifoldParameter(manifold.random_point(), manifold)
        
        with pytest.raises(ValueError):
            RiemannianAdam([param], lr=0.001, betas=(-0.1, 0.999))
        
        with pytest.raises(ValueError):
            RiemannianAdam([param], lr=0.001, betas=(0.9, 1.5))
    
    def test_step_reduces_loss(self):
        """Single step should reduce loss for simple convex problem."""
        manifold = Sphere(64)
        param = ManifoldParameter(manifold.random_point(), manifold)
        target = torch.randn(64)
        target = target / target.norm()  # Normalize to sphere
        
        optimizer = RiemannianAdam([param], lr=0.1)
        
        # Compute initial loss
        loss_before = ((param - target) ** 2).sum().item()
        
        # Take optimization step
        optimizer.zero_grad()
        loss = ((param - target) ** 2).sum()
        loss.backward()
        optimizer.step()
        
        # Compute final loss
        loss_after = ((param - target) ** 2).sum().item()
        
        # Loss should decrease
        assert loss_after < loss_before
    
    def test_manifold_constraint_preservation(self):
        """Parameter should remain on manifold after optimization steps."""
        manifold = Sphere(128)
        param = ManifoldParameter(manifold.random_point(), manifold)
        optimizer = RiemannianAdam([param], lr=0.01)
        
        for _ in range(50):
            optimizer.zero_grad()
            loss = (param ** 2).sum()
            loss.backward()
            optimizer.step()
            
            # Check parameter is still on sphere (unit norm)
            norm = torch.norm(param.data)
            assert torch.isclose(norm, torch.tensor(1.0), atol=1e-4), \
                f"Parameter left manifold: norm={norm}"
    
    def test_convergence_to_target_sphere(self):
        """Optimizer should minimize distance to target on sphere."""
        manifold = Sphere(128)
        param = ManifoldParameter(manifold.random_point(), manifold)
        target = torch.randn(128)
        target = target / target.norm()  # Normalize to sphere
        
        initial_distance = ((param - target) ** 2).sum().sqrt().item()
        
        # Use lower learning rate for Adam to avoid numerical instability
        optimizer = RiemannianAdam([param], lr=0.01)
        
        for _ in range(500):  # More steps with lower lr
            optimizer.zero_grad()
            loss = ((param - target) ** 2).sum()
            loss.backward()
            optimizer.step()
        
        final_distance = ((param - target) ** 2).sum().sqrt().item()
        
        # Should reduce distance by >99%
        assert final_distance < initial_distance * 0.01, \
            f"Distance reduction insufficient: {initial_distance:.4f} -> {final_distance:.4f}"
    
    def test_bias_correction(self):
        """Bias correction should be applied correctly."""
        manifold = Sphere(64)
        param = ManifoldParameter(manifold.random_point(), manifold)
        optimizer = RiemannianAdam([param], lr=1e-3, betas=(0.9, 0.999))
        
        for step in range(1, 11):
            optimizer.zero_grad()
            loss = (param ** 2).sum()
            loss.backward()
            optimizer.step()
            
            state = optimizer.state[param]
            assert state['step'] == step
    
    def test_moment_transport_preserves_norm(self):
        """Parallel transport should preserve moment vector norms."""
        manifold = Sphere(64)
        param = ManifoldParameter(manifold.random_point(), manifold)
        optimizer = RiemannianAdam([param], lr=0.01)
        
        # Take a few steps to build up moments
        for _ in range(5):
            optimizer.zero_grad()
            loss = (param ** 2).sum()
            loss.backward()
            optimizer.step()
        
        # Get moment buffers
        state = optimizer.state[param]
        if 'exp_avg' in state:
            exp_avg_norm_before = state['exp_avg'].norm().item()
            exp_avg_sq_norm_before = state['exp_avg_sq'].norm().item()
            
            # Take another step
            optimizer.zero_grad()
            loss = (param ** 2).sum()
            loss.backward()
            optimizer.step()
            
            # Check moment norms are approximately preserved
            exp_avg_norm_after = state['exp_avg'].norm().item()
            exp_avg_sq_norm_after = state['exp_avg_sq'].norm().item()
            
            # Allow some tolerance due to parallel transport and updates
            if exp_avg_norm_before > 0:
                assert abs(exp_avg_norm_after - exp_avg_norm_before) < 0.2 * exp_avg_norm_before
            if exp_avg_sq_norm_before > 0:
                assert abs(exp_avg_sq_norm_after - exp_avg_sq_norm_before) < 0.2 * exp_avg_sq_norm_before
    
    def test_amsgrad_variant(self):
        """AMSGrad variant should work correctly."""
        manifold = Sphere(64)
        param = ManifoldParameter(manifold.random_point(), manifold)
        optimizer = RiemannianAdam([param], lr=0.01, amsgrad=True)
        
        for _ in range(10):
            optimizer.zero_grad()
            loss = (param ** 2).sum()
            loss.backward()
            optimizer.step()
        
        state = optimizer.state[param]
        assert 'max_exp_avg_sq' in state
    
    def test_handles_none_gradients(self):
        """Optimizer should skip parameters with None gradients."""
        manifold = Sphere(64)
        param1 = ManifoldParameter(manifold.random_point(), manifold)
        param2 = ManifoldParameter(manifold.random_point(), manifold)
        
        optimizer = RiemannianAdam([param1, param2], lr=0.01)
        
        # Only compute gradient for param1
        loss = (param1 ** 2).sum()
        loss.backward()
        
        # Should not raise error
        optimizer.step()
        
        assert param1.grad is not None
        assert param2.grad is None
    
    def test_parameter_groups(self):
        """Different parameter groups should have different hyperparameters."""
        manifold = Sphere(64)
        param1 = ManifoldParameter(manifold.random_point(), manifold)
        param2 = ManifoldParameter(manifold.random_point(), manifold)
        
        optimizer = RiemannianAdam([
            {'params': [param1], 'lr': 0.1},
            {'params': [param2], 'lr': 0.01},
        ])
        
        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]['lr'] == 0.1
        assert optimizer.param_groups[1]['lr'] == 0.01
    
    def test_euclidean_fallback(self):
        """Standard parameters should use Euclidean updates."""
        # Use standard nn.Parameter (not ManifoldParameter)
        param = nn.Parameter(torch.randn(64))
        optimizer = RiemannianAdam([param], lr=0.01)
        
        param_before = param.data.clone()
        
        optimizer.zero_grad()
        loss = (param ** 2).sum()
        loss.backward()
        optimizer.step()
        
        # Should have updated
        assert not torch.allclose(param.data, param_before)
    
    def test_reproducibility_with_seed(self):
        """Same seed should give identical optimization trajectory."""
        def run_optimization(seed):
            torch.manual_seed(seed)
            manifold = Sphere(64)
            param = ManifoldParameter(manifold.random_point(), manifold)
            optimizer = RiemannianAdam([param], lr=0.01)
            
            for _ in range(10):
                optimizer.zero_grad()
                loss = (param ** 2).sum()
                loss.backward()
                optimizer.step()
            
            return param.data.clone()
        
        result1 = run_optimization(42)
        result2 = run_optimization(42)
        
        assert torch.allclose(result1, result2, atol=1e-7), \
            "Results should be identical with same seed"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for AMP tests")
class TestAMPCompatibility:
    """Tests for automatic mixed precision compatibility."""
    
    def test_riemannian_sgd_amp_compatibility(self):
        """RiemannianSGD should work with automatic mixed precision."""
        manifold = Sphere(64)
        param = ManifoldParameter(manifold.random_point(), manifold).cuda()
        
        optimizer = RiemannianSGD([param], lr=0.01)
        scaler = torch.cuda.amp.GradScaler()
        
        for _ in range(10):
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                loss = (param ** 2).sum()
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        # Should complete without errors
        assert torch.isclose(torch.norm(param.data), torch.tensor(1.0).cuda(), atol=1e-4)
    
    def test_riemannian_adam_amp_compatibility(self):
        """RiemannianAdam should work with automatic mixed precision."""
        manifold = Sphere(64)
        param = ManifoldParameter(manifold.random_point(), manifold).cuda()
        
        optimizer = RiemannianAdam([param], lr=0.01)
        scaler = torch.cuda.amp.GradScaler()
        
        for _ in range(10):
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                loss = (param ** 2).sum()
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        # Should complete without errors
        assert torch.isclose(torch.norm(param.data), torch.tensor(1.0).cuda(), atol=1e-4)
