"""Property-based tests for manifold implementations."""

import pytest
import torch
from geotorch import Sphere, Hyperbolic, Euclidean


@pytest.mark.parametrize("manifold", [
    Euclidean(64),
    Sphere(64),
    Hyperbolic(64, model='poincare'),
])
class TestManifoldProperties:
    """Test that all manifolds satisfy required geometric properties."""
    
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
        """exp_p(log_p(q)) = q for q in domain"""
        p = manifold.random_point()
        q = manifold.random_point()
        
        # Skip if q is at cut locus of p
        if not manifold.in_domain(p, q):
            pytest.skip("q is at cut locus of p")
        
        v = manifold.log(p, q)
        q_recovered = manifold.exp(p, v)
        assert torch.allclose(q_recovered, q, atol=1e-4)
    
    def test_log_exp_inverse(self, manifold):
        """log_p(exp_p(v)) = v for small v"""
        p = manifold.random_point()
        v = 0.1 * manifold.random_tangent(p)  # Small tangent
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
    
    def test_distance_equals_log_norm(self, manifold):
        """d(p, q) = ||log_p(q)||"""
        p = manifold.random_point()
        q = manifold.random_point()
        
        # Skip if q is at cut locus of p
        if not manifold.in_domain(p, q):
            pytest.skip("q is at cut locus of p")
        
        dist = manifold.distance(p, q)
        v = manifold.log(p, q)
        log_norm = manifold.norm(p, v)
        assert torch.allclose(dist, log_norm, atol=1e-5)
    
    def test_parallel_transport_preserves_norm(self, manifold):
        """||PT(v)||_q = ||v||_p using Riemannian norm"""
        p = manifold.random_point()
        q = manifold.random_point()
        v = manifold.random_tangent(p)
        
        v_transported = manifold.parallel_transport(v, p, q)
        
        # Compute norms using Riemannian metric
        norm_v = manifold.norm(p, v)
        norm_vt = manifold.norm(q, v_transported)
        
        assert torch.allclose(norm_v, norm_vt, atol=1e-4)
    
    def test_projection_is_idempotent(self, manifold):
        """project(project(x)) = project(x)"""
        x = torch.randn(manifold.dim + 10)  # Ambient space
        p1 = manifold.project(x)
        p2 = manifold.project(p1)
        assert torch.allclose(p1, p2, atol=1e-6)
    
    def test_tangent_projection_is_tangent(self, manifold):
        """Projected vector is in tangent space"""
        p = manifold.random_point()
        v = torch.randn_like(p)
        v_proj = manifold.project_tangent(p, v)
        
        # For sphere: tangent iff v·p = 0
        if isinstance(manifold, Sphere):
            dot = torch.sum(v_proj * p, dim=-1)
            assert torch.allclose(dot, torch.tensor(0.0), atol=1e-6)
    
    def test_geodesic_endpoints(self, manifold):
        """geodesic(p, q, 0) = p and geodesic(p, q, 1) = q"""
        p = manifold.random_point()
        q = manifold.random_point()
        
        # Skip if q is at cut locus of p
        if not manifold.in_domain(p, q):
            pytest.skip("q is at cut locus of p")
        
        start = manifold.geodesic(p, q, 0.0)
        end = manifold.geodesic(p, q, 1.0)
        
        assert torch.allclose(start, p, atol=1e-5)
        assert torch.allclose(end, q, atol=1e-5)


class TestSphereSpecific:
    """Sphere-specific tests."""
    
    def test_sphere_points_have_unit_norm(self):
        """All points on sphere have unit norm."""
        S = Sphere(64)
        p = S.random_point()
        norm = torch.linalg.norm(p)
        assert torch.allclose(norm, torch.tensor(1.0), atol=1e-6)
    
    def test_sphere_dimension(self):
        """Sphere(n) creates S^{n-1}."""
        S = Sphere(64)
        assert S.dim == 63
    
    def test_sphere_tangent_orthogonal(self):
        """Tangent vectors are orthogonal to base point."""
        S = Sphere(64)
        p = S.random_point()
        v = S.random_tangent(p)
        dot = torch.sum(p * v, dim=-1)
        assert torch.allclose(dot, torch.tensor(0.0), atol=1e-6)


class TestHyperbolicSpecific:
    """Hyperbolic-specific tests."""
    
    def test_hyperbolic_poincare_in_ball(self):
        """Points in Poincaré ball have norm < 1."""
        H = Hyperbolic(64, model='poincare')
        p = H.random_point()
        norm = torch.linalg.norm(p)
        assert norm < 1.0
    
    def test_hyperbolic_dimension(self):
        """Hyperbolic(n) creates H^{n-1}."""
        H = Hyperbolic(64, model='poincare')
        assert H.dim == 63
    
    def test_hyperbolic_models(self):
        """Both Poincaré and hyperboloid models work."""
        H_poincare = Hyperbolic(64, model='poincare')
        H_hyperboloid = Hyperbolic(64, model='hyperboloid')
        
        assert H_poincare.dim == 63
        assert H_hyperboloid.dim == 63


class TestEuclideanSpecific:
    """Euclidean-specific tests."""
    
    def test_euclidean_dimension(self):
        """Euclidean(n) has dimension n."""
        E = Euclidean(64)
        assert E.dim == 64
    
    def test_euclidean_operations_are_linear(self):
        """Verify that operations are linear as expected."""
        E = Euclidean(64)
        p = E.random_point()
        q = E.random_point()
        v = E.log(p, q)
        
        # In Euclidean space, log is just subtraction
        assert torch.allclose(v, q - p)
        
        # And exp is just addition
        recovered = E.exp(p, v)
        assert torch.allclose(recovered, q)
