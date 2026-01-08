"""Tests for ManifoldTensor and TangentTensor."""

import pytest
import torch
from geotorch import Sphere, Euclidean, ManifoldTensor, TangentTensor


class TestManifoldTensor:
    """Tests for ManifoldTensor class."""
    
    def test_creation(self):
        """Test ManifoldTensor creation"""
        S = Sphere(64)
        p = S.random_point()
        mt = ManifoldTensor(p, manifold=S)
        
        assert isinstance(mt, torch.Tensor)
        assert isinstance(mt, ManifoldTensor)
        assert mt.manifold == S
        assert mt.shape == p.shape
    
    def test_project_inplace(self):
        """Test in-place projection"""
        S = Sphere(64)
        x = torch.randn(64)
        mt = ManifoldTensor(x, manifold=S)
        
        # Before projection, might not be on manifold
        mt.project_()
        
        # After projection, should be on sphere
        assert torch.allclose(torch.norm(mt), torch.tensor(1.0), atol=1e-6)
    
    def test_exp(self):
        """Test exponential map through ManifoldTensor"""
        S = Sphere(64)
        p = S.random_point()
        mt = ManifoldTensor(p, manifold=S)
        
        v = S.random_tangent(p)
        qt = mt.exp(v)
        
        assert isinstance(qt, ManifoldTensor)
        assert qt.manifold == S
        assert torch.allclose(torch.norm(qt), torch.tensor(1.0), atol=1e-6)
    
    def test_log(self):
        """Test logarithmic map through ManifoldTensor"""
        S = Sphere(64)
        p = S.random_point()
        q = S.random_point()
        
        mt_p = ManifoldTensor(p, manifold=S)
        mt_q = ManifoldTensor(q, manifold=S)
        
        v = mt_p.log(mt_q)
        
        assert isinstance(v, torch.Tensor)
        assert v.shape == p.shape
        
        # Check it's a tangent vector (orthogonal to p for sphere)
        assert torch.allclose(torch.dot(p, v), torch.tensor(0.0), atol=1e-5)
    
    def test_distance(self):
        """Test distance computation"""
        S = Sphere(64)
        p = S.random_point()
        q = S.random_point()
        
        mt_p = ManifoldTensor(p, manifold=S)
        mt_q = ManifoldTensor(q, manifold=S)
        
        dist = mt_p.distance(mt_q)
        
        assert isinstance(dist, torch.Tensor)
        assert dist.ndim == 0  # Scalar
        assert dist >= 0
        
        # Should match manifold distance
        assert torch.allclose(dist, S.distance(p, q), atol=1e-6)
    
    def test_geodesic_to(self):
        """Test geodesic interpolation"""
        S = Sphere(64)
        p = S.random_point()
        q = S.random_point()
        
        mt_p = ManifoldTensor(p, manifold=S)
        mt_q = ManifoldTensor(q, manifold=S)
        
        # Test endpoints
        mt_0 = mt_p.geodesic_to(mt_q, 0.0)
        mt_1 = mt_p.geodesic_to(mt_q, 1.0)
        
        assert torch.allclose(mt_0, p, atol=1e-6)
        assert torch.allclose(mt_1, q, atol=1e-6)
        
        # Test midpoint
        mt_mid = mt_p.geodesic_to(mt_q, 0.5)
        assert isinstance(mt_mid, ManifoldTensor)
        assert torch.allclose(torch.norm(mt_mid), torch.tensor(1.0), atol=1e-6)
    
    def test_different_manifold_error(self):
        """Test that operations between different manifolds raise errors"""
        S = Sphere(64)
        E = Euclidean(64)
        
        p_sphere = ManifoldTensor(S.random_point(), manifold=S)
        p_euclidean = ManifoldTensor(E.random_point(), manifold=E)
        
        with pytest.raises(ValueError):
            p_sphere.log(p_euclidean)
        
        with pytest.raises(ValueError):
            p_sphere.distance(p_euclidean)
    
    def test_repr(self):
        """Test string representation"""
        S = Sphere(64)
        p = S.random_point()
        mt = ManifoldTensor(p, manifold=S)
        
        repr_str = repr(mt)
        assert "ManifoldTensor" in repr_str
        assert "Sphere" in repr_str


class TestTangentTensor:
    """Tests for TangentTensor class."""
    
    def test_creation(self):
        """Test TangentTensor creation"""
        S = Sphere(64)
        p = S.random_point()
        v = S.random_tangent(p)
        
        tt = TangentTensor(v, base_point=p, manifold=S)
        
        assert isinstance(tt, torch.Tensor)
        assert isinstance(tt, TangentTensor)
        assert tt.manifold == S
        assert torch.allclose(tt.base_point, p, atol=1e-6)
        assert tt.shape == v.shape
    
    def test_norm(self):
        """Test norm computation"""
        S = Sphere(64)
        p = S.random_point()
        v = S.random_tangent(p)
        
        tt = TangentTensor(v, base_point=p, manifold=S)
        norm = tt.norm()
        
        assert isinstance(norm, torch.Tensor)
        assert norm >= 0
        
        # Should match manifold norm
        assert torch.allclose(norm, S.norm(p, v), atol=1e-6)
    
    def test_parallel_transport(self):
        """Test parallel transport"""
        S = Sphere(64)
        p = S.random_point()
        v_tangent = 0.3 * S.random_tangent(p)
        q = S.exp(p, v_tangent)
        
        v = S.random_tangent(p)
        tt = TangentTensor(v, base_point=p, manifold=S)
        
        tt_transported = tt.parallel_transport(q)
        
        assert isinstance(tt_transported, TangentTensor)
        assert tt_transported.manifold == S
        assert torch.allclose(tt_transported.base_point, q, atol=1e-6)
        
        # Should preserve norm
        assert torch.allclose(tt.norm(), tt_transported.norm(), atol=1e-4)
    
    def test_repr(self):
        """Test string representation"""
        S = Sphere(64)
        p = S.random_point()
        v = S.random_tangent(p)
        
        tt = TangentTensor(v, base_point=p, manifold=S)
        repr_str = repr(tt)
        assert "TangentTensor" in repr_str
        assert "Sphere" in repr_str


class TestTensorIntegration:
    """Test integration between ManifoldTensor and TangentTensor."""
    
    def test_exp_with_tangent_tensor(self):
        """Test exp with TangentTensor"""
        E = Euclidean(64)
        p = E.random_point()
        v = E.random_tangent(p)
        
        mt = ManifoldTensor(p, manifold=E)
        tt = TangentTensor(v, base_point=p, manifold=E)
        
        # Should work with regular tensor
        q1 = mt.exp(v)
        # Also with TangentTensor
        q2 = mt.exp(tt)
        
        assert torch.allclose(q1, q2, atol=1e-6)
    
    def test_workflow(self):
        """Test complete geometric workflow"""
        S = Sphere(64)
        
        # Create points
        p = ManifoldTensor(S.random_point(), manifold=S)
        q = ManifoldTensor(S.random_point(), manifold=S)
        
        # Compute tangent vector
        v = p.log(q)
        tt = TangentTensor(v, base_point=p, manifold=S)
        
        # Move along geodesic
        q_recovered = p.exp(tt)
        
        # Should recover q
        assert torch.allclose(q_recovered, q, atol=1e-4)
        
        # Transport tangent vector
        w = S.random_tangent(p)
        wt = TangentTensor(w, base_point=p, manifold=S)
        wt_transported = wt.parallel_transport(q)
        
        # Transported vector should be at q
        assert torch.allclose(wt_transported.base_point, q, atol=1e-6)


class TestMetadataPreservation:
    """Tests for metadata preservation through torch operations."""
    
    def test_clone_preserves_manifold(self):
        """Test that clone preserves ManifoldTensor metadata."""
        S = Sphere(64)
        p = ManifoldTensor(S.random_point(), manifold=S)
        
        p_cloned = p.clone()
        
        assert isinstance(p_cloned, ManifoldTensor)
        assert p_cloned.manifold == S
    
    def test_detach_preserves_manifold(self):
        """Test that detach preserves ManifoldTensor metadata."""
        S = Sphere(64)
        p = ManifoldTensor(S.random_point(), manifold=S)
        p.requires_grad_(True)
        
        p_detached = p.detach()
        
        assert isinstance(p_detached, ManifoldTensor)
        assert p_detached.manifold == S
        assert not p_detached.requires_grad
    
    def test_to_preserves_manifold(self):
        """Test that .to() preserves ManifoldTensor metadata."""
        S = Sphere(64)
        p = ManifoldTensor(S.random_point(), manifold=S)
        
        p_double = p.to(torch.float64)
        
        assert isinstance(p_double, ManifoldTensor)
        assert p_double.manifold == S
        assert p_double.dtype == torch.float64
    
    def test_indexing_preserves_manifold(self):
        """Test that indexing preserves ManifoldTensor metadata."""
        S = Sphere(64)
        # Create batch of points
        batch = torch.randn(10, 64)
        batch = batch / batch.norm(dim=-1, keepdim=True)  # Normalize to sphere
        mt = ManifoldTensor(batch, manifold=S)
        
        # Index single element
        single = mt[0]
        assert isinstance(single, ManifoldTensor)
        assert single.manifold == S
        
        # Slice subset
        subset = mt[2:5]
        assert isinstance(subset, ManifoldTensor)
        assert subset.manifold == S
        assert subset.shape == (3, 64)
    
    def test_tangent_clone_preserves_metadata(self):
        """Test that clone preserves TangentTensor metadata."""
        S = Sphere(64)
        p = S.random_point()
        v = S.random_tangent(p)
        tt = TangentTensor(v, base_point=p, manifold=S)
        
        tt_cloned = tt.clone()
        
        assert isinstance(tt_cloned, TangentTensor)
        assert tt_cloned.manifold == S
        assert torch.allclose(tt_cloned.base_point, p, atol=1e-6)
    
    def test_tangent_to_migrates_base_point(self):
        """Test that .to() migrates base_point to same dtype."""
        S = Sphere(64)
        p = S.random_point()
        v = S.random_tangent(p)
        tt = TangentTensor(v, base_point=p, manifold=S)
        
        tt_double = tt.to(torch.float64)
        
        assert isinstance(tt_double, TangentTensor)
        assert tt_double.manifold == S
        assert tt_double.dtype == torch.float64
        # Base point should also be migrated
        assert tt_double.base_point.dtype == torch.float64
    
    def test_tangent_to_with_manifold_tensor_base(self):
        """Test that .to() preserves ManifoldTensor metadata on base_point."""
        S = Sphere(64)
        p = ManifoldTensor(S.random_point(), manifold=S)
        v = S.random_tangent(p)
        tt = TangentTensor(v, base_point=p, manifold=S)
        
        tt_double = tt.to(torch.float64)
        
        assert isinstance(tt_double, TangentTensor)
        assert tt_double.dtype == torch.float64
        # Base point should be migrated AND preserve manifold
        assert tt_double.base_point.dtype == torch.float64
        assert isinstance(tt_double.base_point, ManifoldTensor)
        assert tt_double.base_point.manifold == S
    
    def test_tangent_indexing_preserves_metadata(self):
        """Test that indexing preserves TangentTensor metadata."""
        S = Sphere(64)
        p = S.random_point()
        # Create batch of tangent vectors
        batch = torch.randn(10, 64)
        # Project to tangent space (orthogonal to p)
        batch = batch - torch.outer(batch @ p, p)
        tt = TangentTensor(batch, base_point=p, manifold=S)
        
        # Index single element
        single = tt[0]
        assert isinstance(single, TangentTensor)
        assert single.manifold == S
        assert torch.allclose(single.base_point, p, atol=1e-6)
        
        # Slice subset
        subset = tt[2:5]
        assert isinstance(subset, TangentTensor)
        assert subset.manifold == S
        assert subset.shape == (3, 64)
    
    def test_unbind_preserves_manifold_tensor(self):
        """Test that torch.unbind preserves ManifoldTensor metadata on each element."""
        S = Sphere(64)
        # Create batch of points
        batch = torch.randn(5, 64)
        batch = batch / batch.norm(dim=-1, keepdim=True)
        mt = ManifoldTensor(batch, manifold=S)
        
        # Unbind along batch dimension
        elements = torch.unbind(mt, dim=0)
        
        assert isinstance(elements, tuple)
        assert len(elements) == 5
        for elem in elements:
            assert isinstance(elem, ManifoldTensor), f"Expected ManifoldTensor, got {type(elem)}"
            assert elem.manifold == S
            assert elem.shape == (64,)
    
    def test_chunk_preserves_tangent_tensor(self):
        """Test that torch.chunk preserves TangentTensor metadata on each chunk."""
        S = Sphere(64)
        p = S.random_point()
        # Create batch of tangent vectors
        batch = torch.randn(6, 64)
        batch = batch - torch.outer(batch @ p, p)  # Project to tangent space
        tt = TangentTensor(batch, base_point=p, manifold=S)
        
        # Chunk into 3 parts
        chunks = torch.chunk(tt, chunks=3, dim=0)
        
        assert isinstance(chunks, tuple)
        assert len(chunks) == 3
        for chunk in chunks:
            assert isinstance(chunk, TangentTensor), f"Expected TangentTensor, got {type(chunk)}"
            assert chunk.manifold == S
            assert torch.allclose(chunk.base_point, p, atol=1e-6)
            assert chunk.shape == (2, 64)
    
    def test_tangent_detach_also_detaches_base_point(self):
        """Test that detach() on TangentTensor also detaches base_point."""
        S = Sphere(64)
        p = S.random_point()
        p.requires_grad_(True)
        v = S.random_tangent(p.detach())  # Use detached for tangent generation
        tt = TangentTensor(v, base_point=p, manifold=S)
        tt.requires_grad_(True)
        
        # Base point has grad before detach
        assert tt.base_point.requires_grad
        
        tt_detached = tt.detach()
        
        assert isinstance(tt_detached, TangentTensor)
        assert not tt_detached.requires_grad
        # Base point should also be detached
        assert not tt_detached.base_point.requires_grad

