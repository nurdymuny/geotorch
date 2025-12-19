"""Tests for GeoTorch neural network layers."""

import pytest
import torch
from geotorch import Sphere, Hyperbolic
from geotorch.nn import (
    ManifoldParameter,
    ManifoldLinear,
    GeodesicEmbedding,
    FrechetMean,
    GeometricAttention,
    MultiHeadGeometricAttention,
    GeodesicLayer,
    HyperbolicLinear,
    HyperbolicEmbedding,
    SphericalLinear,
    SphericalEmbedding,
)


class TestManifoldLinear:
    """Tests for ManifoldLinear layer."""
    
    def test_sphere_projection(self):
        """Output should be on unit sphere."""
        sphere = Sphere(64)
        layer = ManifoldLinear(128, 64, sphere)
        
        x = torch.randn(32, 128)
        y = layer(x)
        
        assert y.shape == (32, 64)
        norms = y.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(32), atol=1e-5)
    
    def test_hyperbolic_projection(self):
        """Output should be inside Poincaré ball."""
        hyp = Hyperbolic(32)
        layer = ManifoldLinear(64, 32, hyp)
        
        x = torch.randn(16, 64)
        y = layer(x)
        
        assert y.shape == (16, 32)
        norms = y.norm(dim=-1)
        assert (norms < 1.0).all()
    
    def test_exp_method(self):
        """Test exponential map method."""
        sphere = Sphere(32)
        layer = ManifoldLinear(64, 32, sphere, method='exp')
        
        x = torch.randn(8, 64)
        y = layer(x)
        
        assert y.shape == (8, 32)
        norms = y.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(8), atol=1e-5)
    
    def test_gradient_flow(self):
        """Gradients should flow through the layer."""
        sphere = Sphere(32)
        layer = ManifoldLinear(64, 32, sphere)
        
        x = torch.randn(8, 64, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None
        assert layer.weight.grad is not None


class TestGeodesicEmbedding:
    """Tests for GeodesicEmbedding layer."""
    
    def test_hyperbolic_embedding(self):
        """Embeddings should be inside Poincaré ball."""
        hyp = Hyperbolic(32)
        emb = GeodesicEmbedding(1000, 32, hyp)
        
        indices = torch.randint(0, 1000, (64,))
        vectors = emb(indices)
        
        assert vectors.shape == (64, 32)
        norms = vectors.norm(dim=-1)
        assert (norms < 1.0).all()
    
    def test_sphere_embedding(self):
        """Embeddings should be on unit sphere."""
        sphere = Sphere(64)
        emb = GeodesicEmbedding(500, 64, sphere)
        
        indices = torch.tensor([0, 10, 100, 499])
        vectors = emb(indices)
        
        assert vectors.shape == (4, 64)
        norms = vectors.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)
    
    def test_gradient_flow(self):
        """Gradients should flow to embedding weights."""
        hyp = Hyperbolic(16)
        emb = GeodesicEmbedding(100, 16, hyp)
        
        indices = torch.tensor([0, 5, 10])
        vectors = emb(indices)
        loss = vectors.sum()
        loss.backward()
        
        assert emb.weight.grad is not None


class TestFrechetMean:
    """Tests for FrechetMean layer."""
    
    def test_sphere_mean(self):
        """Mean should be on sphere."""
        sphere = Sphere(64)
        frechet = FrechetMean(sphere, n_iters=5)
        
        # Generate points on sphere
        points = sphere.random_point(100)
        mean = frechet(points)
        
        assert mean.shape == (64,)
        assert torch.allclose(mean.norm(), torch.tensor(1.0), atol=1e-4)
    
    def test_weighted_mean(self):
        """Weighted mean should work."""
        sphere = Sphere(32)
        frechet = FrechetMean(sphere, n_iters=5)
        
        points = sphere.random_point(50)
        weights = torch.rand(50)
        mean = frechet(points, weights)
        
        assert mean.shape == (32,)
        assert torch.allclose(mean.norm(), torch.tensor(1.0), atol=1e-4)
    
    def test_batched_mean(self):
        """Batched mean should work."""
        sphere = Sphere(16)
        frechet = FrechetMean(sphere, n_iters=3)
        
        # (B=4, N=20, D=16)
        points = sphere.random_point(4, 20)
        means = frechet(points)
        
        assert means.shape == (4, 16)
        norms = means.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-4)


class TestGeometricAttention:
    """Tests for GeometricAttention layer."""
    
    def test_attention_weights_sum_to_one(self):
        """Attention weights should sum to 1."""
        sphere = Sphere(64)
        attn = GeometricAttention(sphere, temperature=0.1)
        
        queries = sphere.random_point(8, 10)   # (batch=8, n_queries=10)
        keys = sphere.random_point(8, 20)      # (batch=8, n_keys=20)
        values = torch.randn(8, 20, 128)
        
        output, weights = attn(queries, keys, values)
        
        assert output.shape == (8, 10, 128)
        assert weights.shape == (8, 10, 20)
        assert torch.allclose(weights.sum(-1), torch.ones(8, 10), atol=1e-5)
    
    def test_hyperbolic_attention(self):
        """Should work with hyperbolic manifold."""
        hyp = Hyperbolic(32)
        attn = GeometricAttention(hyp, temperature=1.0)
        
        queries = hyp.random_point(4, 5)
        keys = hyp.random_point(4, 8)
        values = torch.randn(4, 8, 64)
        
        output, weights = attn(queries, keys, values)
        
        assert output.shape == (4, 5, 64)
        assert weights.shape == (4, 5, 8)
    
    def test_hard_attention(self):
        """Hard attention should produce one-hot weights."""
        sphere = Sphere(16)
        attn = GeometricAttention(sphere, hard=True)
        
        queries = sphere.random_point(2, 3)
        keys = sphere.random_point(2, 5)
        values = torch.randn(2, 5, 8)
        
        output, weights = attn(queries, keys, values)
        
        # Each row should be one-hot
        assert weights.shape == (2, 3, 5)
        assert torch.allclose(weights.sum(-1), torch.ones(2, 3))
        assert (weights.max(dim=-1).values == 1.0).all()


class TestMultiHeadGeometricAttention:
    """Tests for MultiHeadGeometricAttention layer."""
    
    def test_output_shape(self):
        """Output should have correct shape."""
        mha = MultiHeadGeometricAttention(256, 8, Sphere(32))
        x = torch.randn(4, 50, 256)
        
        output, weights = mha(x, x, x)
        
        assert output.shape == (4, 50, 256)
        assert weights.shape == (4, 8, 50, 50)
    
    def test_gradient_flow(self):
        """Gradients should flow through all projections."""
        mha = MultiHeadGeometricAttention(128, 4, Sphere(32))
        x = torch.randn(2, 10, 128, requires_grad=True)
        
        output, _ = mha(x, x, x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert mha.q_proj.weight.grad is not None
        assert mha.out_proj.weight.grad is not None


class TestGeodesicLayer:
    """Tests for GeodesicLayer (manifold-to-manifold mapping)."""
    
    def test_sphere_to_hyperbolic(self):
        """Should map from sphere to hyperbolic space."""
        sphere = Sphere(64)
        hyp = Hyperbolic(32)
        layer = GeodesicLayer(sphere, hyp, 64, 32)
        
        x = sphere.random_point(100)
        y = layer(x)
        
        assert y.shape == (100, 32)
        norms = y.norm(dim=-1)
        assert (norms < 1.0).all()  # Inside Poincaré ball
    
    def test_hyperbolic_to_sphere(self):
        """Should map from hyperbolic to sphere."""
        hyp = Hyperbolic(32)
        sphere = Sphere(64)
        layer = GeodesicLayer(hyp, sphere, 32, 64)
        
        x = hyp.random_point(50)
        y = layer(x)
        
        assert y.shape == (50, 64)
        norms = y.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(50), atol=1e-5)


class TestConvenienceWrappers:
    """Tests for convenience wrapper classes."""
    
    def test_hyperbolic_linear(self):
        """HyperbolicLinear should work."""
        layer = HyperbolicLinear(128, 64)
        x = torch.randn(32, 128)
        y = layer(x)
        
        assert y.shape == (32, 64)
        assert (y.norm(dim=-1) < 1.0).all()
    
    def test_spherical_linear(self):
        """SphericalLinear should work."""
        layer = SphericalLinear(128, 64)
        x = torch.randn(32, 128)
        y = layer(x)
        
        assert y.shape == (32, 64)
        assert torch.allclose(y.norm(dim=-1), torch.ones(32), atol=1e-5)
    
    def test_hyperbolic_embedding(self):
        """HyperbolicEmbedding should work."""
        emb = HyperbolicEmbedding(1000, 64)
        indices = torch.randint(0, 1000, (16,))
        vectors = emb(indices)
        
        assert vectors.shape == (16, 64)
        assert (vectors.norm(dim=-1) < 1.0).all()
    
    def test_spherical_embedding(self):
        """SphericalEmbedding should work."""
        emb = SphericalEmbedding(500, 32)
        indices = torch.randint(0, 500, (8,))
        vectors = emb(indices)
        
        assert vectors.shape == (8, 32)
        assert torch.allclose(vectors.norm(dim=-1), torch.ones(8), atol=1e-5)


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_simple_network(self):
        """Build and run a simple manifold network."""
        import torch.nn as nn
        
        class ManifoldNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(128, 64)
                self.manifold_out = ManifoldLinear(64, 32, Sphere(32))
            
            def forward(self, x):
                x = torch.relu(self.linear1(x))
                return self.manifold_out(x)
        
        model = ManifoldNet()
        x = torch.randn(16, 128)
        y = model(x)
        
        assert y.shape == (16, 32)
        assert torch.allclose(y.norm(dim=-1), torch.ones(16), atol=1e-5)
    
    def test_embedding_with_frechet_pooling(self):
        """Embedding lookup followed by Fréchet mean pooling."""
        sphere = Sphere(32)
        emb = GeodesicEmbedding(100, 32, sphere)
        pool = FrechetMean(sphere, n_iters=3)
        
        # Look up 10 embeddings and pool them
        indices = torch.randint(0, 100, (10,))
        vectors = emb(indices)
        pooled = pool(vectors)
        
        assert pooled.shape == (32,)
        assert torch.allclose(pooled.norm(), torch.tensor(1.0), atol=1e-4)
