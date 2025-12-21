"""Tests for DavisTensor Phase 1 foundation."""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import davistensor as dt


class TestImports:
    """Test that all required imports work."""
    
    def test_import_davistensor(self):
        """Test basic import."""
        assert dt.__version__ == "0.1.0"
    
    def test_import_classes(self):
        """Test that main classes are available."""
        assert hasattr(dt, 'ManifoldTensor')
        assert hasattr(dt, 'TangentTensor')
        assert hasattr(dt, 'Scalar')
    
    def test_import_manifolds(self):
        """Test that manifold classes are available."""
        assert hasattr(dt, 'Manifold')
        assert hasattr(dt, 'Euclidean')
    
    def test_import_factory_functions(self):
        """Test that factory functions are available."""
        assert hasattr(dt, 'randn')
        assert hasattr(dt, 'origin')
        assert hasattr(dt, 'tangent_randn')
        assert hasattr(dt, 'tangent_zeros')


class TestEuclideanManifold:
    """Test Euclidean manifold implementation."""
    
    def test_create_euclidean(self):
        """Test creating Euclidean manifold."""
        E = dt.Euclidean(10)
        assert E.dim == 10
    
    def test_euclidean_exp(self):
        """Test exponential map (should be addition)."""
        E = dt.Euclidean(3)
        p = E.origin()
        v_data = np.array([1.0, 2.0, 3.0])
        v = dt.TangentTensor(v_data, base_point=dt.ManifoldTensor(p, E), manifold=E)
        
        q = E.exp(p, v._core)
        q_data = q.numpy()
        
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(q_data, expected, rtol=1e-5)
    
    def test_euclidean_log(self):
        """Test logarithmic map (should be subtraction)."""
        E = dt.Euclidean(3)
        p_data = np.array([1.0, 2.0, 3.0])
        q_data = np.array([4.0, 5.0, 6.0])
        
        from davistensor.core.storage import _create_tensor, GeometricType
        p = _create_tensor(p_data, manifold=E, geometric_type=GeometricType.MANIFOLD_POINT)
        q = _create_tensor(q_data, manifold=E, geometric_type=GeometricType.MANIFOLD_POINT)
        
        v = E.log(p, q)
        v_data = v.numpy()
        
        expected = np.array([3.0, 3.0, 3.0])
        np.testing.assert_allclose(v_data, expected, rtol=1e-5)
    
    def test_euclidean_distance(self):
        """Test geodesic distance (should be Euclidean norm)."""
        E = dt.Euclidean(3)
        p_data = np.array([0.0, 0.0, 0.0])
        q_data = np.array([3.0, 4.0, 0.0])
        
        from davistensor.core.storage import _create_tensor, GeometricType
        p = _create_tensor(p_data, manifold=E, geometric_type=GeometricType.MANIFOLD_POINT)
        q = _create_tensor(q_data, manifold=E, geometric_type=GeometricType.MANIFOLD_POINT)
        
        d = E.distance(p, q)
        d_val = d.numpy()
        
        expected = 5.0  # sqrt(9 + 16)
        np.testing.assert_allclose(d_val, expected, rtol=1e-5)


class TestManifoldTensor:
    """Test ManifoldTensor wrapper class."""
    
    def test_create_manifold_tensor(self):
        """Test creating a ManifoldTensor."""
        E = dt.Euclidean(5)
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        x = dt.ManifoldTensor(data, manifold=E)
        
        assert x.manifold == E
        assert x.shape == (5,)
        np.testing.assert_allclose(x.numpy(), data)
    
    def test_manifold_tensor_exp(self):
        """Test exponential map through ManifoldTensor."""
        E = dt.Euclidean(3)
        x = dt.ManifoldTensor(np.array([1.0, 2.0, 3.0]), manifold=E)
        v = dt.TangentTensor(np.array([0.5, 0.5, 0.5]), base_point=x, manifold=E)
        
        y = x.exp(v)
        
        expected = np.array([1.5, 2.5, 3.5])
        np.testing.assert_allclose(y.numpy(), expected, rtol=1e-5)
    
    def test_manifold_tensor_log(self):
        """Test logarithmic map through ManifoldTensor."""
        E = dt.Euclidean(3)
        x = dt.ManifoldTensor(np.array([1.0, 2.0, 3.0]), manifold=E)
        y = dt.ManifoldTensor(np.array([4.0, 5.0, 6.0]), manifold=E)
        
        v = x.log(y)
        
        assert isinstance(v, dt.TangentTensor)
        expected = np.array([3.0, 3.0, 3.0])
        np.testing.assert_allclose(v.numpy(), expected, rtol=1e-5)
    
    def test_manifold_tensor_distance(self):
        """Test distance through ManifoldTensor."""
        E = dt.Euclidean(3)
        x = dt.ManifoldTensor(np.array([0.0, 0.0, 0.0]), manifold=E)
        y = dt.ManifoldTensor(np.array([3.0, 4.0, 0.0]), manifold=E)
        
        d = x.distance(y)
        
        assert isinstance(d, dt.Scalar)
        assert abs(d.item() - 5.0) < 1e-5
    
    def test_manifold_tensor_add_tangent(self):
        """Test x + v = exp map."""
        E = dt.Euclidean(3)
        x = dt.ManifoldTensor(np.array([1.0, 2.0, 3.0]), manifold=E)
        v = dt.TangentTensor(np.array([0.5, 0.5, 0.5]), base_point=x, manifold=E)
        
        y = x + v
        
        assert isinstance(y, dt.ManifoldTensor)
        expected = np.array([1.5, 2.5, 3.5])
        np.testing.assert_allclose(y.numpy(), expected, rtol=1e-5)
    
    def test_manifold_tensor_sub(self):
        """Test x - y = log map."""
        E = dt.Euclidean(3)
        x = dt.ManifoldTensor(np.array([4.0, 5.0, 6.0]), manifold=E)
        y = dt.ManifoldTensor(np.array([1.0, 2.0, 3.0]), manifold=E)
        
        v = x - y
        
        assert isinstance(v, dt.TangentTensor)
        expected = np.array([3.0, 3.0, 3.0])
        np.testing.assert_allclose(v.numpy(), expected, rtol=1e-5)


class TestTangentTensor:
    """Test TangentTensor wrapper class."""
    
    def test_create_tangent_tensor(self):
        """Test creating a TangentTensor."""
        E = dt.Euclidean(3)
        x = dt.ManifoldTensor(np.array([1.0, 2.0, 3.0]), manifold=E)
        v = dt.TangentTensor(np.array([0.5, 0.5, 0.5]), base_point=x, manifold=E)
        
        assert v.manifold == E
        assert isinstance(v.base_point, dt.ManifoldTensor)
        np.testing.assert_allclose(v.numpy(), np.array([0.5, 0.5, 0.5]))
    
    def test_tangent_tensor_add_same_point(self):
        """Test adding tangent vectors at the same point."""
        E = dt.Euclidean(3)
        x = dt.ManifoldTensor(np.array([1.0, 2.0, 3.0]), manifold=E)
        v1 = dt.TangentTensor(np.array([1.0, 0.0, 0.0]), base_point=x, manifold=E)
        v2 = dt.TangentTensor(np.array([0.0, 1.0, 0.0]), base_point=x, manifold=E)
        
        v3 = v1 + v2
        
        assert isinstance(v3, dt.TangentTensor)
        expected = np.array([1.0, 1.0, 0.0])
        np.testing.assert_allclose(v3.numpy(), expected, rtol=1e-5)
    
    def test_tangent_tensor_add_different_points_raises(self):
        """Test that adding tangent vectors at different points raises TypeError."""
        E = dt.Euclidean(3)
        x = dt.ManifoldTensor(np.array([1.0, 2.0, 3.0]), manifold=E)
        y = dt.ManifoldTensor(np.array([4.0, 5.0, 6.0]), manifold=E)
        
        vx = dt.TangentTensor(np.array([1.0, 0.0, 0.0]), base_point=x, manifold=E)
        vy = dt.TangentTensor(np.array([0.0, 1.0, 0.0]), base_point=y, manifold=E)
        
        with pytest.raises(TypeError, match="Cannot add tangent vectors at different points"):
            vx + vy
    
    def test_tangent_tensor_scalar_mul(self):
        """Test scalar multiplication of tangent vectors."""
        E = dt.Euclidean(3)
        x = dt.ManifoldTensor(np.array([1.0, 2.0, 3.0]), manifold=E)
        v = dt.TangentTensor(np.array([1.0, 2.0, 3.0]), base_point=x, manifold=E)
        
        v2 = v * 2.0
        v3 = 3.0 * v
        
        np.testing.assert_allclose(v2.numpy(), np.array([2.0, 4.0, 6.0]), rtol=1e-5)
        np.testing.assert_allclose(v3.numpy(), np.array([3.0, 6.0, 9.0]), rtol=1e-5)
    
    def test_tangent_tensor_neg(self):
        """Test negation of tangent vectors."""
        E = dt.Euclidean(3)
        x = dt.ManifoldTensor(np.array([1.0, 2.0, 3.0]), manifold=E)
        v = dt.TangentTensor(np.array([1.0, 2.0, 3.0]), base_point=x, manifold=E)
        
        v_neg = -v
        
        np.testing.assert_allclose(v_neg.numpy(), np.array([-1.0, -2.0, -3.0]), rtol=1e-5)


class TestScalar:
    """Test Scalar wrapper class."""
    
    def test_create_scalar(self):
        """Test creating a Scalar."""
        s = dt.Scalar(5.0)
        assert s.item() == 5.0
    
    def test_scalar_arithmetic(self):
        """Test scalar arithmetic operations."""
        s1 = dt.Scalar(3.0)
        s2 = dt.Scalar(2.0)
        
        s_add = s1 + s2
        s_mul = s1 * s2
        s_div = s1 / s2
        
        assert s_add.item() == 5.0
        assert s_mul.item() == 6.0
        assert s_div.item() == 1.5


class TestFactoryFunctions:
    """Test factory functions."""
    
    def test_randn(self):
        """Test dt.randn factory function."""
        E = dt.Euclidean(10)
        x = dt.randn(5, manifold=E)
        
        assert isinstance(x, dt.ManifoldTensor)
        assert x.manifold == E
        assert x.shape == (5, 10)
    
    def test_origin(self):
        """Test dt.origin factory function."""
        E = dt.Euclidean(10)
        x = dt.origin(manifold=E)
        
        assert isinstance(x, dt.ManifoldTensor)
        assert x.manifold == E
        assert x.shape == (10,)
        np.testing.assert_allclose(x.numpy(), np.zeros(10))
    
    def test_tangent_randn(self):
        """Test dt.tangent_randn factory function."""
        E = dt.Euclidean(10)
        x = dt.randn(manifold=E)
        v = dt.tangent_randn(x)
        
        assert isinstance(v, dt.TangentTensor)
        assert v.manifold == E
        assert v.shape == (10,)
    
    def test_tangent_zeros(self):
        """Test dt.tangent_zeros factory function."""
        E = dt.Euclidean(10)
        x = dt.randn(manifold=E)
        v = dt.tangent_zeros(x)
        
        assert isinstance(v, dt.TangentTensor)
        assert v.manifold == E
        np.testing.assert_allclose(v.numpy(), np.zeros(10))


class TestAcceptanceCriteria:
    """Test all acceptance criteria from the problem statement."""
    
    def test_import_davistensor(self):
        """AC: import davistensor as dt works."""
        import davistensor as dt
        assert dt is not None
    
    def test_randn_creates_manifold_tensor(self):
        """AC: dt.randn(manifold=dt.Euclidean(10)) creates a ManifoldTensor."""
        x = dt.randn(manifold=dt.Euclidean(10))
        assert isinstance(x, dt.ManifoldTensor)
        assert x.manifold.dim == 10
    
    def test_adding_tangent_at_different_points_raises(self):
        """AC: Adding tangent vectors at different points raises TypeError."""
        E = dt.Euclidean(10)
        x = dt.randn(manifold=E)
        y = dt.randn(manifold=E)
        
        vx = dt.tangent_randn(x)
        vy = dt.tangent_randn(y)
        
        with pytest.raises(TypeError):
            vx + vy
    
    def test_point_plus_tangent_is_exp(self):
        """AC: x + v performs exponential map."""
        E = dt.Euclidean(3)
        x = dt.ManifoldTensor(np.array([1.0, 2.0, 3.0]), manifold=E)
        v = dt.TangentTensor(np.array([0.5, 0.5, 0.5]), base_point=x, manifold=E)
        
        y = x + v
        
        # For Euclidean: exp(x, v) = x + v
        expected = np.array([1.5, 2.5, 3.5])
        np.testing.assert_allclose(y.numpy(), expected, rtol=1e-5)
    
    def test_point_minus_point_is_log(self):
        """AC: x - y performs logarithm map."""
        E = dt.Euclidean(3)
        x = dt.ManifoldTensor(np.array([4.0, 5.0, 6.0]), manifold=E)
        y = dt.ManifoldTensor(np.array([1.0, 2.0, 3.0]), manifold=E)
        
        v = x - y
        
        # For Euclidean: log_y(x) = x - y
        expected = np.array([3.0, 3.0, 3.0])
        np.testing.assert_allclose(v.numpy(), expected, rtol=1e-5)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
