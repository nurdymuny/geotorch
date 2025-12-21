"""Comprehensive tests for DavisTensor Phase 1."""

import pytest
import numpy as np
import davistensor as dt


class TestCoreStorage:
    """Tests for core storage layer."""
    
    def test_device_creation(self):
        """Test Device creation."""
        device = dt.Device('cpu')
        assert device.device_type == 'cpu'
        assert str(device) == 'cpu'
        
        # CUDA should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            dt.Device('cuda')
    
    def test_dtype_enum(self):
        """Test DType enum."""
        assert dt.DType.FLOAT32.numpy_dtype == np.float32
        assert dt.DType.FLOAT64.numpy_dtype == np.float64
        assert dt.DType.INT32.numpy_dtype == np.int32
        assert dt.DType.INT64.numpy_dtype == np.int64
    
    def test_geometric_type_enum(self):
        """Test GeometricType enum."""
        assert dt.GeometricType.SCALAR.value == 'scalar'
        assert dt.GeometricType.EUCLIDEAN.value == 'euclidean'
        assert dt.GeometricType.MANIFOLD_POINT.value == 'manifold_point'
        assert dt.GeometricType.TANGENT.value == 'tangent'
    
    def test_storage_creation(self):
        """Test Storage creation."""
        data = np.array([1.0, 2.0, 3.0])
        storage = dt.Storage(data)
        assert storage.device.device_type == 'cpu'
        assert storage.ref_count == 1
        np.testing.assert_array_equal(storage.data, data)
    
    def test_storage_clone(self):
        """Test Storage cloning."""
        data = np.array([1.0, 2.0, 3.0])
        storage = dt.Storage(data)
        cloned = storage.clone()
        
        # Should be different objects
        assert cloned is not storage
        # Data should be equal but different arrays
        np.testing.assert_array_equal(cloned.data, storage.data)
        assert cloned.data is not storage.data
    
    def test_tensorcore_creation(self):
        """Test TensorCore creation."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        storage = dt.Storage(data)
        
        core = dt.TensorCore(
            storage=storage,
            shape=(2, 2),
            dtype=dt.DType.FLOAT64,
        )
        
        assert core.shape == (2, 2)
        assert core.ndim == 2
        assert core.size == 4
        assert core.dtype == dt.DType.FLOAT64
        assert core.geometric_type == dt.GeometricType.EUCLIDEAN
    
    def test_tensorcore_indexing(self):
        """Test TensorCore indexing."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        storage = dt.Storage(data)
        core = dt.TensorCore(storage=storage, shape=(2, 2), dtype=dt.DType.FLOAT64)
        
        # Test indexing
        row = core[0]
        assert isinstance(row, dt.TensorCore)
        np.testing.assert_array_equal(row.numpy(), np.array([1.0, 2.0]))
        
        # Test scalar indexing
        val = core[0, 0]
        assert val == 1.0
    
    def test_tensorcore_reshape(self):
        """Test TensorCore reshaping."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        storage = dt.Storage(data)
        core = dt.TensorCore(storage=storage, shape=(2, 2), dtype=dt.DType.FLOAT64)
        
        reshaped = core.reshape(4)
        assert reshaped.shape == (4,)
        np.testing.assert_array_equal(reshaped.numpy(), np.array([1.0, 2.0, 3.0, 4.0]))
    
    def test_tensorcore_clone(self):
        """Test TensorCore cloning."""
        data = np.array([1.0, 2.0, 3.0])
        storage = dt.Storage(data)
        core = dt.TensorCore(storage=storage, shape=(3,), dtype=dt.DType.FLOAT64)
        
        cloned = core.clone()
        assert cloned is not core
        np.testing.assert_array_equal(cloned.numpy(), core.numpy())
        assert cloned.storage is not core.storage


class TestFactoryFunctions:
    """Tests for factory functions."""
    
    def test_zeros(self):
        """Test zeros factory."""
        x = dt.zeros(2, 3)
        assert x.shape == (2, 3)
        np.testing.assert_array_equal(x.numpy(), np.zeros((2, 3)))
    
    def test_ones(self):
        """Test ones factory."""
        x = dt.ones(2, 3)
        assert x.shape == (2, 3)
        np.testing.assert_array_equal(x.numpy(), np.ones((2, 3)))
    
    def test_randn(self):
        """Test randn factory."""
        x = dt.core_randn(2, 3)
        assert x.shape == (2, 3)
        # Check it's not all zeros (statistically very unlikely)
        assert not np.allclose(x.numpy(), 0.0)
    
    def test_rand(self):
        """Test rand factory."""
        x = dt.rand(2, 3)
        assert x.shape == (2, 3)
        # Check values are in [0, 1)
        assert np.all(x.numpy() >= 0.0)
        assert np.all(x.numpy() < 1.0)
    
    def test_tensor(self):
        """Test tensor factory."""
        data = [[1.0, 2.0], [3.0, 4.0]]
        x = dt.tensor(data)
        assert x.shape == (2, 2)
        np.testing.assert_array_equal(x.numpy(), np.array(data))
    
    def test_from_numpy(self):
        """Test from_numpy factory."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        x = dt.from_numpy(arr)
        assert x.shape == (2, 2)
        np.testing.assert_array_equal(x.numpy(), arr)


class TestEuclideanManifold:
    """Tests for Euclidean manifold."""
    
    def test_creation(self):
        """Test Euclidean manifold creation."""
        E = dt.Euclidean(3)
        assert E.dim == 3
        assert E.ambient_dim == 3
        assert E.curvature_type == 'constant'
    
    def test_random_point(self):
        """Test random point generation."""
        E = dt.Euclidean(3)
        x = E.random_point()
        assert x.shape == (3,)
        
        # Batch generation
        x_batch = E.random_point(5)
        assert x_batch.shape == (5, 3)
    
    def test_origin(self):
        """Test origin point."""
        E = dt.Euclidean(3)
        x = E.origin()
        np.testing.assert_array_equal(x, np.zeros(3))
    
    def test_check_point(self):
        """Test point validation."""
        E = dt.Euclidean(3)
        x = np.array([1.0, 2.0, 3.0])
        assert E.check_point(x)
        
        # Wrong dimension
        y = np.array([1.0, 2.0])
        assert not E.check_point(y)
    
    def test_project_point(self):
        """Test point projection (identity for Euclidean)."""
        E = dt.Euclidean(3)
        x = np.array([1.0, 2.0, 3.0])
        y = E.project_point(x)
        np.testing.assert_array_equal(y, x)
    
    def test_check_tangent(self):
        """Test tangent vector validation."""
        E = dt.Euclidean(3)
        x = np.array([1.0, 2.0, 3.0])
        v = np.array([0.1, 0.2, 0.3])
        assert E.check_tangent(x, v)
    
    def test_project_tangent(self):
        """Test tangent projection (identity for Euclidean)."""
        E = dt.Euclidean(3)
        x = np.array([1.0, 2.0, 3.0])
        v = np.array([0.1, 0.2, 0.3])
        u = E.project_tangent(x, v)
        np.testing.assert_array_equal(u, v)
    
    def test_random_tangent(self):
        """Test random tangent generation."""
        E = dt.Euclidean(3)
        x = np.array([1.0, 2.0, 3.0])
        v = E.random_tangent(x)
        assert v.shape == x.shape
    
    def test_zero_tangent(self):
        """Test zero tangent."""
        E = dt.Euclidean(3)
        x = np.array([1.0, 2.0, 3.0])
        v = E.zero_tangent(x)
        np.testing.assert_array_equal(v, np.zeros(3))
    
    def test_metric(self):
        """Test metric tensor (identity for Euclidean)."""
        E = dt.Euclidean(3)
        x = np.array([1.0, 2.0, 3.0])
        g = E.metric(x)
        np.testing.assert_array_equal(g, np.eye(3))
    
    def test_inner(self):
        """Test inner product."""
        E = dt.Euclidean(3)
        x = np.array([1.0, 2.0, 3.0])
        u = np.array([1.0, 0.0, 0.0])
        v = np.array([0.0, 1.0, 0.0])
        
        # Orthogonal vectors
        result = E.inner(x, u, v)
        assert result == 0.0
        
        # Same vector
        result = E.inner(x, u, u)
        assert result == 1.0
    
    def test_norm(self):
        """Test norm."""
        E = dt.Euclidean(3)
        x = np.array([1.0, 2.0, 3.0])
        v = np.array([3.0, 4.0, 0.0])
        
        result = E.norm(x, v)
        assert np.isclose(result, 5.0)
    
    def test_exp(self):
        """Test exponential map (addition for Euclidean)."""
        E = dt.Euclidean(3)
        x = np.array([1.0, 2.0, 3.0])
        v = np.array([0.1, 0.2, 0.3])
        
        y = E.exp(x, v)
        np.testing.assert_array_almost_equal(y, x + v)
    
    def test_log(self):
        """Test logarithm map (subtraction for Euclidean)."""
        E = dt.Euclidean(3)
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.5, 2.5, 3.5])
        
        v = E.log(x, y)
        np.testing.assert_array_almost_equal(v, y - x)
    
    def test_distance(self):
        """Test geodesic distance."""
        E = dt.Euclidean(3)
        x = np.array([0.0, 0.0, 0.0])
        y = np.array([3.0, 4.0, 0.0])
        
        d = E.distance(x, y)
        assert np.isclose(d, 5.0)
    
    def test_geodesic(self):
        """Test geodesic interpolation."""
        E = dt.Euclidean(3)
        x = np.array([0.0, 0.0, 0.0])
        y = np.array([1.0, 1.0, 1.0])
        
        # Midpoint
        m = E.geodesic(x, y, 0.5)
        np.testing.assert_array_almost_equal(m, np.array([0.5, 0.5, 0.5]))
        
        # At x
        p = E.geodesic(x, y, 0.0)
        np.testing.assert_array_almost_equal(p, x)
        
        # At y
        q = E.geodesic(x, y, 1.0)
        np.testing.assert_array_almost_equal(q, y)
    
    def test_parallel_transport(self):
        """Test parallel transport (identity for Euclidean)."""
        E = dt.Euclidean(3)
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 3.0, 4.0])
        v = np.array([0.1, 0.2, 0.3])
        
        u = E.parallel_transport(x, y, v)
        np.testing.assert_array_equal(u, v)
    
    def test_sectional_curvature(self):
        """Test sectional curvature (zero for Euclidean)."""
        E = dt.Euclidean(3)
        x = np.array([1.0, 2.0, 3.0])
        u = np.array([1.0, 0.0, 0.0])
        v = np.array([0.0, 1.0, 0.0])
        
        K = E.sectional_curvature(x, u, v)
        assert K == 0.0
    
    def test_scalar_curvature(self):
        """Test scalar curvature (zero for Euclidean)."""
        E = dt.Euclidean(3)
        x = np.array([1.0, 2.0, 3.0])
        
        R = E.scalar_curvature(x)
        assert R == 0.0


class TestScalar:
    """Tests for Scalar type."""
    
    def test_creation(self):
        """Test Scalar creation."""
        s = dt.Scalar(3.14)
        assert s.item() == 3.14
    
    def test_arithmetic(self):
        """Test arithmetic operations."""
        a = dt.Scalar(2.0)
        b = dt.Scalar(3.0)
        
        # Addition
        c = a + b
        assert c.item() == 5.0
        
        # Subtraction
        c = b - a
        assert c.item() == 1.0
        
        # Multiplication
        c = a * b
        assert c.item() == 6.0
        
        # Division
        c = b / a
        assert c.item() == 1.5
        
        # Power
        c = a ** b
        assert c.item() == 8.0
        
        # Negation
        c = -a
        assert c.item() == -2.0
    
    def test_arithmetic_with_floats(self):
        """Test arithmetic with Python floats."""
        a = dt.Scalar(2.0)
        
        c = a + 3.0
        assert c.item() == 5.0
        
        c = 3.0 + a
        assert c.item() == 5.0
        
        c = a * 2.0
        assert c.item() == 4.0
    
    def test_sqrt(self):
        """Test square root."""
        a = dt.Scalar(9.0)
        b = a.sqrt()
        assert b.item() == 3.0


class TestManifoldTensor:
    """Tests for ManifoldTensor type."""
    
    def test_creation(self):
        """Test ManifoldTensor creation."""
        E = dt.Euclidean(3)
        x = dt.randn(manifold=E)
        
        assert isinstance(x, dt.ManifoldTensor)
        assert x.manifold is E
        assert x.shape == (3,)
    
    def test_origin(self):
        """Test origin factory."""
        E = dt.Euclidean(3)
        x = dt.origin(E)
        
        np.testing.assert_array_equal(x.numpy(), np.zeros(3))
    
    def test_clone(self):
        """Test cloning."""
        E = dt.Euclidean(3)
        x = dt.randn(manifold=E)
        y = x.clone()
        
        assert y is not x
        np.testing.assert_array_equal(y.numpy(), x.numpy())
    
    def test_exp(self):
        """Test exponential map."""
        E = dt.Euclidean(3)
        x = dt.randn(manifold=E)
        v = x.random_tangent()
        
        y = x.exp(v)
        assert isinstance(y, dt.ManifoldTensor)
        
        # For Euclidean, exp(x, v) = x + v
        expected = x.numpy() + v.numpy()
        np.testing.assert_array_almost_equal(y.numpy(), expected)
    
    def test_log(self):
        """Test logarithm map."""
        E = dt.Euclidean(3)
        x = dt.randn(manifold=E)
        y = dt.randn(manifold=E)
        
        v = x.log(y)
        assert isinstance(v, dt.TangentTensor)
        assert v.base_point is x
        
        # For Euclidean, log(x, y) = y - x
        expected = y.numpy() - x.numpy()
        np.testing.assert_array_almost_equal(v.numpy(), expected)
    
    def test_distance(self):
        """Test distance."""
        E = dt.Euclidean(3)
        x = dt.origin(E)
        
        # Create point at distance 5
        data = np.array([3.0, 4.0, 0.0])
        from davistensor.core.storage import Storage
        storage = Storage(data, dt.Device('cpu'))
        core = dt.TensorCore(storage=storage, shape=(3,), dtype=dt.DType.FLOAT64)
        y = dt.ManifoldTensor(core, E)
        
        d = x.distance(y)
        assert isinstance(d, dt.Scalar)
        assert np.isclose(d.item(), 5.0)
    
    def test_geodesic(self):
        """Test geodesic interpolation."""
        E = dt.Euclidean(3)
        x = dt.origin(E)
        
        data = np.array([1.0, 1.0, 1.0])
        from davistensor.core.storage import Storage
        storage = Storage(data, dt.Device('cpu'))
        core = dt.TensorCore(storage=storage, shape=(3,), dtype=dt.DType.FLOAT64)
        y = dt.ManifoldTensor(core, E)
        
        # Midpoint
        m = x.geodesic(y, 0.5)
        expected = np.array([0.5, 0.5, 0.5])
        np.testing.assert_array_almost_equal(m.numpy(), expected)
    
    def test_random_tangent(self):
        """Test random tangent generation."""
        E = dt.Euclidean(3)
        x = dt.randn(manifold=E)
        v = x.random_tangent()
        
        assert isinstance(v, dt.TangentTensor)
        assert v.base_point is x
        assert v.shape == x.shape
    
    def test_zero_tangent(self):
        """Test zero tangent generation."""
        E = dt.Euclidean(3)
        x = dt.randn(manifold=E)
        v = x.zero_tangent()
        
        assert isinstance(v, dt.TangentTensor)
        np.testing.assert_array_equal(v.numpy(), np.zeros(3))
    
    def test_geometric_addition(self):
        """Test point + tangent = exp."""
        E = dt.Euclidean(3)
        x = dt.randn(manifold=E)
        v = x.random_tangent()
        
        y1 = x + v
        y2 = x.exp(v)
        
        np.testing.assert_array_almost_equal(y1.numpy(), y2.numpy())
    
    def test_geometric_subtraction(self):
        """Test point - point = log."""
        E = dt.Euclidean(3)
        x = dt.randn(manifold=E)
        y = dt.randn(manifold=E)
        
        v1 = y - x
        v2 = x.log(y)
        
        np.testing.assert_array_almost_equal(v1.numpy(), v2.numpy())


class TestTangentTensor:
    """Tests for TangentTensor type."""
    
    def test_creation(self):
        """Test TangentTensor creation."""
        E = dt.Euclidean(3)
        x = dt.randn(manifold=E)
        v = dt.tangent_randn(x)
        
        assert isinstance(v, dt.TangentTensor)
        assert v.base_point is x
        assert v.manifold is E
    
    def test_zeros(self):
        """Test zero tangent vector."""
        E = dt.Euclidean(3)
        x = dt.randn(manifold=E)
        v = dt.tangent_zeros(x)
        
        np.testing.assert_array_equal(v.numpy(), np.zeros(3))
    
    def test_addition_same_base(self):
        """Test adding tangent vectors at same point."""
        E = dt.Euclidean(3)
        x = dt.randn(manifold=E)
        v1 = x.random_tangent()
        v2 = x.random_tangent()
        
        v3 = v1 + v2
        assert isinstance(v3, dt.TangentTensor)
        
        expected = v1.numpy() + v2.numpy()
        np.testing.assert_array_almost_equal(v3.numpy(), expected)
    
    def test_addition_different_base_raises(self):
        """Test that adding tangent vectors at different points raises TypeError."""
        E = dt.Euclidean(3)
        x = dt.randn(manifold=E)
        y = dt.randn(manifold=E)
        
        vx = x.random_tangent()
        vy = y.random_tangent()
        
        with pytest.raises(TypeError, match="Cannot add tangent vectors at different points"):
            _ = vx + vy
    
    def test_scalar_multiplication(self):
        """Test scalar multiplication."""
        E = dt.Euclidean(3)
        x = dt.randn(manifold=E)
        v = x.random_tangent()
        
        u = 2.0 * v
        expected = 2.0 * v.numpy()
        np.testing.assert_array_almost_equal(u.numpy(), expected)
        
        # Test right multiplication
        u = v * 2.0
        np.testing.assert_array_almost_equal(u.numpy(), expected)
    
    def test_division(self):
        """Test scalar division."""
        E = dt.Euclidean(3)
        x = dt.randn(manifold=E)
        v = x.random_tangent()
        
        u = v / 2.0
        expected = v.numpy() / 2.0
        np.testing.assert_array_almost_equal(u.numpy(), expected)
    
    def test_negation(self):
        """Test negation."""
        E = dt.Euclidean(3)
        x = dt.randn(manifold=E)
        v = x.random_tangent()
        
        u = -v
        expected = -v.numpy()
        np.testing.assert_array_almost_equal(u.numpy(), expected)
    
    def test_subtraction(self):
        """Test subtraction."""
        E = dt.Euclidean(3)
        x = dt.randn(manifold=E)
        v1 = x.random_tangent()
        v2 = x.random_tangent()
        
        v3 = v1 - v2
        expected = v1.numpy() - v2.numpy()
        np.testing.assert_array_almost_equal(v3.numpy(), expected)
    
    def test_inner(self):
        """Test inner product."""
        E = dt.Euclidean(3)
        x = dt.origin(E)
        
        # Create specific tangent vectors
        data1 = np.array([1.0, 0.0, 0.0])
        data2 = np.array([0.0, 1.0, 0.0])
        
        from davistensor.core.storage import Storage
        storage1 = Storage(data1, dt.Device('cpu'))
        core1 = dt.TensorCore(storage=storage1, shape=(3,), dtype=dt.DType.FLOAT64)
        v1 = dt.TangentTensor(core1, x, E)
        
        storage2 = Storage(data2, dt.Device('cpu'))
        core2 = dt.TensorCore(storage=storage2, shape=(3,), dtype=dt.DType.FLOAT64)
        v2 = dt.TangentTensor(core2, x, E)
        
        # Orthogonal vectors
        result = v1.inner(v2)
        assert isinstance(result, dt.Scalar)
        assert np.isclose(result.item(), 0.0)
        
        # Same vector
        result = v1.inner(v1)
        assert np.isclose(result.item(), 1.0)
    
    def test_norm(self):
        """Test norm."""
        E = dt.Euclidean(3)
        x = dt.origin(E)
        
        data = np.array([3.0, 4.0, 0.0])
        from davistensor.core.storage import Storage
        storage = Storage(data, dt.Device('cpu'))
        core = dt.TensorCore(storage=storage, shape=(3,), dtype=dt.DType.FLOAT64)
        v = dt.TangentTensor(core, x, E)
        
        n = v.norm()
        assert isinstance(n, dt.Scalar)
        assert np.isclose(n.item(), 5.0)
    
    def test_normalize(self):
        """Test normalization."""
        E = dt.Euclidean(3)
        x = dt.origin(E)
        
        data = np.array([3.0, 4.0, 0.0])
        from davistensor.core.storage import Storage
        storage = Storage(data, dt.Device('cpu'))
        core = dt.TensorCore(storage=storage, shape=(3,), dtype=dt.DType.FLOAT64)
        v = dt.TangentTensor(core, x, E)
        
        u = v.normalize()
        assert np.isclose(u.norm().item(), 1.0)
        
        expected = data / 5.0
        np.testing.assert_array_almost_equal(u.numpy(), expected)
    
    def test_transport_to(self):
        """Test parallel transport."""
        E = dt.Euclidean(3)
        x = dt.randn(manifold=E)
        y = dt.randn(manifold=E)
        v = x.random_tangent()
        
        u = v.transport_to(y)
        assert isinstance(u, dt.TangentTensor)
        assert u.base_point is y
        
        # For Euclidean, parallel transport is identity
        np.testing.assert_array_almost_equal(u.numpy(), v.numpy())


class TestIntegration:
    """Integration tests using the API as a user would."""
    
    def test_basic_workflow(self):
        """Test basic geometric workflow."""
        # Create manifold
        E = dt.Euclidean(3)
        
        # Create points
        x = dt.randn(manifold=E)
        y = dt.randn(manifold=E)
        
        # Compute tangent vector from x to y
        v = x.log(y)
        
        # Move from x along v
        z = x.exp(v)
        
        # z should be close to y
        np.testing.assert_array_almost_equal(z.numpy(), y.numpy(), decimal=10)
    
    def test_geodesic_properties(self):
        """Test geodesic properties."""
        E = dt.Euclidean(3)
        x = dt.randn(manifold=E)
        y = dt.randn(manifold=E)
        
        # Geodesic from x to y at t=0.5
        m = x.geodesic(y, 0.5)
        
        # Distance from x to m should be half distance from x to y
        d_xy = x.distance(y).item()
        d_xm = x.distance(m).item()
        
        assert np.isclose(d_xm, d_xy / 2.0)
    
    def test_tangent_transport_workflow(self):
        """Test workflow with tangent vector transport."""
        E = dt.Euclidean(3)
        
        # Create two points
        x = dt.randn(manifold=E)
        y = dt.randn(manifold=E)
        
        # Create tangent vector at x
        vx = x.random_tangent()
        
        # Transport to y
        vy = vx.transport_to(y)
        
        # Now we can add vy to other tangent vectors at y
        wy = y.random_tangent()
        zy = vy + wy
        
        assert isinstance(zy, dt.TangentTensor)
        assert zy.base_point is y
    
    def test_arithmetic_operations(self):
        """Test geometric arithmetic operations."""
        E = dt.Euclidean(3)
        
        x = dt.randn(manifold=E)
        y = dt.randn(manifold=E)
        
        # y - x gives tangent at x
        v = y - x
        assert isinstance(v, dt.TangentTensor)
        assert v.base_point is x
        
        # x + v should give y
        z = x + v
        np.testing.assert_array_almost_equal(z.numpy(), y.numpy(), decimal=10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
