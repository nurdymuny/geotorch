"""Tests for manifold implementations."""

import pytest
import torch
from geotorch import Sphere, Hyperbolic, Euclidean


class TestManifoldProperties:
    """Property tests that should hold for all manifolds."""
    
    def test_exp_at_zero_is_identity(self, manifold):
        """exp_p(0) = p"""
        p = manifold.random_point()
        v = torch.zeros_like(p)
        result = manifold.exp(p, v)
        assert torch.allclose(result, p, atol=1e-5)
    
    def test_log_at_same_point_is_zero(self, manifold):
        """log_p(p) = 0"""
        p = manifold.random_point()
        v = manifold.log(p, p)
        assert torch.allclose(v, torch.zeros_like(v), atol=1e-5)
    
    def test_exp_log_inverse(self, manifold):
        """exp_p(log_p(q)) = q for q in domain of log_p"""
        p = manifold.random_point()
        # Use small tangent to ensure q is in domain
        v = 0.5 * manifold.random_tangent(p)
        q = manifold.exp(p, v)
        
        v_recovered = manifold.log(p, q)
        q_recovered = manifold.exp(p, v_recovered)
        # Use 1e-3 tolerance for hyperbolic numerical precision
        assert torch.allclose(q_recovered, q, atol=1e-3)
    
    def test_log_exp_inverse(self, manifold):
        """log_p(exp_p(v)) = v for small v"""
        p = manifold.random_point()
        v = 0.1 * manifold.random_tangent(p)
        q = manifold.exp(p, v)
        v_recovered = manifold.log(p, q)
        assert torch.allclose(v_recovered, v, atol=1e-4)
    
    def test_distance_symmetry(self, manifold):
        """d(p, q) = d(q, p)"""
        p = manifold.random_point()
        q = manifold.random_point()
        d_pq = manifold.distance(p, q)
        d_qp = manifold.distance(q, p)
        assert torch.allclose(d_pq, d_qp, atol=1e-6)
    
    def test_distance_non_negative(self, manifold):
        """d(p, q) >= 0"""
        p = manifold.random_point()
        q = manifold.random_point()
        assert manifold.distance(p, q) >= 0
    
    def test_distance_zero_iff_same_point(self, manifold):
        """d(p, p) = 0"""
        p = manifold.random_point()
        assert torch.allclose(manifold.distance(p, p), torch.tensor(0.0), atol=1e-6)
    
    def test_distance_equals_log_norm(self, manifold):
        """d(p, q) = ||log_p(q)||"""
        p = manifold.random_point()
        v = 0.5 * manifold.random_tangent(p)
        q = manifold.exp(p, v)
        
        dist = manifold.distance(p, q)
        log_norm = manifold.norm(p, manifold.log(p, q))
        # Use 1e-3 tolerance for hyperbolic space due to numerical precision
        assert torch.allclose(dist, log_norm, atol=1e-3)
    
    def test_parallel_transport_preserves_norm(self, manifold):
        """||PT(v)||_q = ||v||_p using Riemannian norm"""
        p = manifold.random_point()
        v_transport = 0.3 * manifold.random_tangent(p)
        q = manifold.exp(p, v_transport)
        
        v = manifold.random_tangent(p)
        v_transported = manifold.parallel_transport(v, p, q)
        
        norm_v = manifold.norm(p, v)
        norm_vt = manifold.norm(q, v_transported)
        assert torch.allclose(norm_v, norm_vt, atol=1e-4)
    
    def test_projection_is_idempotent(self, manifold):
        """project(project(x)) = project(x)"""
        x = torch.randn_like(manifold.random_point())  # Ambient space
        p1 = manifold.project(x)
        p2 = manifold.project(p1)
        assert torch.allclose(p1, p2, atol=1e-6)
    
    def test_random_point_on_manifold(self, manifold):
        """Random points satisfy manifold constraints"""
        p = manifold.random_point()
        p_projected = manifold.project(p)
        assert torch.allclose(p, p_projected, atol=1e-6)
    
    def test_tangent_is_orthogonal_to_normal(self, manifold):
        """Projected tangent is in tangent space"""
        p = manifold.random_point()
        v = manifold.random_tangent(p)
        v_proj = manifold.project_tangent(p, v)
        assert torch.allclose(v, v_proj, atol=1e-6)


class TestEuclideanSpecific:
    """Euclidean-specific tests"""
    
    def test_exp_is_addition(self):
        E = Euclidean(64)
        p = E.random_point()
        v = E.random_tangent(p)
        assert torch.allclose(E.exp(p, v), p + v, atol=1e-6)
    
    def test_log_is_subtraction(self):
        E = Euclidean(64)
        p = E.random_point()
        q = E.random_point()
        assert torch.allclose(E.log(p, q), q - p, atol=1e-6)
    
    def test_parallel_transport_is_identity(self):
        E = Euclidean(64)
        p = E.random_point()
        q = E.random_point()
        v = E.random_tangent(p)
        assert torch.allclose(E.parallel_transport(v, p, q), v, atol=1e-6)
    
    def test_intrinsic_dimension(self):
        E = Euclidean(64)
        assert E.dim == 64


class TestSphereSpecific:
    """Sphere-specific tests"""
    
    def test_points_have_unit_norm(self):
        S = Sphere(64)
        p = S.random_point()
        assert torch.allclose(torch.norm(p), torch.tensor(1.0), atol=1e-6)
    
    def test_tangent_orthogonal_to_point(self):
        S = Sphere(64)
        p = S.random_point()
        v = S.random_tangent(p)
        assert torch.allclose(torch.dot(p, v), torch.tensor(0.0), atol=1e-6)
    
    def test_antipodal_not_in_domain(self):
        S = Sphere(64)
        p = S.random_point()
        q = -p  # Antipodal point
        assert not S.in_domain(p, q).item()
    
    def test_intrinsic_dimension(self):
        S = Sphere(64)
        assert S.dim == 63  # S^{n-1} has dimension n-1
    
    def test_exp_preserves_norm(self):
        S = Sphere(64)
        p = S.random_point()
        v = S.random_tangent(p)
        q = S.exp(p, v)
        assert torch.allclose(torch.norm(q), torch.tensor(1.0), atol=1e-6)
    
    def test_projection_normalizes(self):
        S = Sphere(64)
        x = torch.randn(64)
        p = S.project(x)
        assert torch.allclose(torch.norm(p), torch.tensor(1.0), atol=1e-6)


class TestHyperbolicSpecific:
    """Hyperbolic-specific tests"""
    
    def test_poincare_points_inside_ball(self):
        H = Hyperbolic(64, model='poincare')
        p = H.random_point()
        assert torch.norm(p) < 1.0
    
    def test_always_in_domain(self):
        """Hyperbolic space has no cut locus"""
        H = Hyperbolic(64, model='poincare')
        p = H.random_point()
        q = H.random_point()
        assert H.in_domain(p, q).item()
    
    def test_intrinsic_dimension(self):
        H = Hyperbolic(64)
        assert H.dim == 63  # H^{n-1} has dimension n-1
    
    def test_exp_preserves_ball(self):
        H = Hyperbolic(64, model='poincare')
        p = H.random_point()
        v = 0.5 * H.random_tangent(p)
        q = H.exp(p, v)
        assert torch.norm(q) < 1.0
    
    def test_projection_inside_ball(self):
        H = Hyperbolic(64, model='poincare')
        x = torch.randn(64) * 2  # Outside ball
        p = H.project(x)
        assert torch.norm(p) < 1.0
    
    def test_invalid_model_raises(self):
        with pytest.raises(ValueError):
            Hyperbolic(64, model='invalid')
    
    def test_positive_curvature_raises(self):
        with pytest.raises(ValueError):
            Hyperbolic(64, curvature=1.0)


class TestBatchOperations:
    """Test batch operations on manifolds."""
    
    def test_batch_exp(self, manifold):
        """Test exponential map with batched inputs"""
        batch_size = 10
        p = manifold.random_point(batch_size)
        v = torch.randn_like(p)
        v = torch.stack([manifold.project_tangent(p[i], v[i]) for i in range(batch_size)])
        
        # Should work with batched inputs
        q = manifold.exp(p, v)
        assert q.shape == p.shape
    
    def test_batch_log(self, manifold):
        """Test logarithmic map with batched inputs"""
        batch_size = 10
        p = manifold.random_point(batch_size)
        q = manifold.random_point(batch_size)
        
        v = manifold.log(p, q)
        assert v.shape == p.shape
    
    def test_batch_distance(self, manifold):
        """Test distance with batched inputs"""
        batch_size = 10
        p = manifold.random_point(batch_size)
        q = manifold.random_point(batch_size)
        
        d = manifold.distance(p, q)
        assert d.shape == (batch_size,)
