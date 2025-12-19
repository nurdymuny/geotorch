"""Tests for ManifoldTensor and TangentTensor."""

import pytest
import torch
from geotorch import Sphere, ManifoldTensor, TangentTensor


class TestManifoldTensor:
    """Test ManifoldTensor functionality."""
    
    def test_create_manifold_tensor(self):
        """Can create ManifoldTensor."""
        S = Sphere(3)
        data = S.random_point()
        mt = ManifoldTensor(data, S)
        
        assert isinstance(mt, ManifoldTensor)
        assert mt.manifold == S
    
    def test_project_inplace(self):
        """project_() works in-place."""
        S = Sphere(3)
        data = torch.randn(3)  # Not on sphere
        mt = ManifoldTensor(data, S)
        
        mt.project_()
        norm = torch.linalg.norm(mt)
        assert torch.allclose(norm, torch.tensor(1.0), atol=1e-6)
    
    def test_exp_returns_manifold_tensor(self):
        """exp() returns ManifoldTensor."""
        S = Sphere(3)
        p = ManifoldTensor(S.random_point(), S)
        v = S.random_tangent(p)
        
        q = p.exp(v)
        assert isinstance(q, ManifoldTensor)
        assert q.manifold == S
    
    def test_log_returns_tangent_tensor(self):
        """log() returns TangentTensor."""
        S = Sphere(3)
        p = ManifoldTensor(S.random_point(), S)
        q = ManifoldTensor(S.random_point(), S)
        
        v = p.log(q)
        assert isinstance(v, TangentTensor)
        assert v.manifold == S
    
    def test_distance_computation(self):
        """distance() computes geodesic distance."""
        S = Sphere(3)
        p = ManifoldTensor(S.random_point(), S)
        q = ManifoldTensor(S.random_point(), S)
        
        d = p.distance(q)
        assert d >= 0
        assert d <= torch.pi  # Max distance on sphere
    
    def test_geodesic_interpolation(self):
        """geodesic_to() interpolates along geodesic."""
        S = Sphere(3)
        p = ManifoldTensor(S.random_point(), S)
        q = ManifoldTensor(S.random_point(), S)
        
        mid = p.geodesic_to(q, 0.5)
        assert isinstance(mid, ManifoldTensor)
        
        # Check endpoints
        start = p.geodesic_to(q, 0.0)
        end = p.geodesic_to(q, 1.0)
        assert torch.allclose(start, p, atol=1e-5)
        assert torch.allclose(end, q, atol=1e-5)


class TestTangentTensor:
    """Test TangentTensor functionality."""
    
    def test_create_tangent_tensor(self):
        """Can create TangentTensor."""
        S = Sphere(3)
        p = S.random_point()
        v = S.random_tangent(p)
        
        tt = TangentTensor(v, S, p)
        assert isinstance(tt, TangentTensor)
        assert tt.manifold == S
        assert torch.allclose(tt.base_point, p)
    
    def test_tangent_exp_returns_manifold_tensor(self):
        """exp() on TangentTensor returns ManifoldTensor."""
        S = Sphere(3)
        p = S.random_point()
        v = S.random_tangent(p)
        
        tt = TangentTensor(v, S, p)
        q = tt.exp()
        
        assert isinstance(q, ManifoldTensor)
        assert q.manifold == S
    
    def test_tangent_norm(self):
        """norm() computes Riemannian norm."""
        S = Sphere(3)
        p = S.random_point()
        v = S.random_tangent(p)
        
        tt = TangentTensor(v, S, p)
        norm = tt.norm()
        
        assert norm >= 0
    
    def test_tangent_project_inplace(self):
        """project_() works in-place."""
        S = Sphere(3)
        p = S.random_point()
        v = torch.randn_like(p)  # Not necessarily tangent
        
        tt = TangentTensor(v, S, p)
        tt.project_()
        
        # Should be tangent (orthogonal to p)
        dot = torch.sum(tt * p)
        assert torch.allclose(dot, torch.tensor(0.0), atol=1e-6)
    
    def test_parallel_transport(self):
        """parallel_transport_to() transports vector."""
        S = Sphere(3)
        p = S.random_point()
        q = S.random_point()
        v = S.random_tangent(p)
        
        tt = TangentTensor(v, S, p)
        tt_q = tt.parallel_transport_to(q)
        
        assert isinstance(tt_q, TangentTensor)
        assert torch.allclose(tt_q.base_point, q)
        
        # Norm should be preserved
        norm_p = tt.norm()
        norm_q = tt_q.norm()
        assert torch.allclose(norm_p, norm_q, atol=1e-4)
