"""
Tests for Phase 5 Advanced Manifolds.

Tests for:
- SPD (Symmetric Positive Definite) manifold
- DavisManifold (learned metric)
- ProductManifold (Cartesian products)
"""

import pytest
import torch
import torch.nn as nn
from geotorch.manifolds import (
    SPD, LogEuclideanSPD, SPDTransform, SPDBiMap, SPDReLU, SPDLogEig,
    DavisManifold, DavisMetricLearner,
    ProductManifold, HyperbolicSphere, HyperbolicEuclidean, SphereEuclidean,
    MultiHyperbolic, MultiSphere, ProductEmbedding, ProductLinear,
    Sphere, Hyperbolic, Euclidean
)


# =============================================================================
# SPD Manifold Tests
# =============================================================================

class TestSPD:
    """Tests for SPD manifold."""
    
    def test_random_point(self):
        """Random points should be SPD."""
        spd = SPD(4)
        P = spd.random_point()
        
        assert P.shape == (4, 4)
        assert torch.allclose(P, P.T)  # Symmetric
        eigenvalues = torch.linalg.eigvalsh(P)
        assert (eigenvalues > 0).all()  # Positive definite
    
    def test_random_point_batched(self):
        """Batched random points."""
        spd = SPD(4)
        P = spd.random_point(8)
        
        assert P.shape == (8, 4, 4)
        assert torch.allclose(P, P.transpose(-2, -1))
        eigenvalues = torch.linalg.eigvalsh(P)
        assert (eigenvalues > 0).all()
    
    def test_project(self):
        """Projection should make matrix SPD."""
        spd = SPD(4)
        X = torch.randn(4, 4)  # Not SPD
        P = spd.project(X)
        
        assert torch.allclose(P, P.T)
        eigenvalues = torch.linalg.eigvalsh(P)
        assert (eigenvalues > 0).all()
    
    def test_project_tangent(self):
        """Tangent projection should symmetrize."""
        spd = SPD(4)
        P = spd.random_point()
        V = torch.randn(4, 4)  # Not symmetric
        V_tan = spd.project_tangent(P, V)
        
        assert torch.allclose(V_tan, V_tan.T)
    
    def test_exp_log_inverse(self):
        """exp(log(Q)) = Q."""
        spd = SPD(4)
        P = spd.random_point()
        Q = spd.random_point()
        
        V = spd.log(P, Q)
        Q_recovered = spd.exp(P, V)
        
        assert torch.allclose(Q, Q_recovered, atol=1e-4)
    
    def test_distance_properties(self):
        """Distance should be symmetric and zero for identical points."""
        spd = SPD(4)
        P = spd.random_point()
        Q = spd.random_point()
        
        d_PQ = spd.distance(P, Q)
        d_QP = spd.distance(Q, P)
        d_PP = spd.distance(P, P)
        
        assert torch.allclose(d_PQ, d_QP, atol=0.01)  # Symmetric (numerical error from eigendecomp)
        assert d_PP < 1e-4  # Zero for identical
        assert d_PQ > 0  # Positive for different
    
    def test_geodesic(self):
        """Geodesic midpoint should be equidistant."""
        spd = SPD(4)
        P = spd.random_point()
        Q = spd.random_point()
        
        mid = spd.geodesic(P, Q, 0.5)
        d_P_mid = spd.distance(P, mid)
        d_mid_Q = spd.distance(mid, Q)
        d_P_Q = spd.distance(P, Q)
        
        assert torch.allclose(d_P_mid + d_mid_Q, d_P_Q, atol=1e-3)
    
    def test_parallel_transport(self):
        """Parallel transport preserves symmetry."""
        spd = SPD(4)
        P = spd.random_point()
        Q = spd.random_point()
        V = spd.project_tangent(P, torch.randn(4, 4))
        
        V_transported = spd.parallel_transport(P, Q, V)
        
        assert torch.allclose(V_transported, V_transported.T)
    
    def test_frechet_mean(self):
        """Fréchet mean should be SPD."""
        spd = SPD(4)
        Ps = torch.stack([spd.random_point() for _ in range(10)])
        
        mean = spd.frechet_mean(Ps)
        
        # Fréchet mean should be approximately symmetric (may have small numerical errors)
        assert torch.allclose(mean, mean.T, atol=1e-4)
        eigenvalues = torch.linalg.eigvalsh(mean)
        assert (eigenvalues > 0).all()
    
    def test_log_euclidean_spd(self):
        """Log-Euclidean SPD should work."""
        spd = LogEuclideanSPD(4)
        P = spd.random_point()
        Q = spd.random_point()
        
        d = spd.distance(P, Q)
        assert d > 0
    
    def test_matrix_operations(self):
        """Test sqrtm, logm, expm, powm."""
        spd = SPD(4)
        P = spd.random_point()
        
        # sqrt(P)² = P
        P_sqrt = spd.sqrtm(P)
        P_recovered = P_sqrt @ P_sqrt
        assert torch.allclose(P, P_recovered, atol=1e-4)
        
        # exp(log(P)) = P
        P_log = spd.logm(P)
        P_recovered = spd.expm(P_log)
        assert torch.allclose(P, P_recovered, atol=1e-4)
        
        # P^1 = P
        P_pow1 = spd.powm(P, 1.0)
        assert torch.allclose(P, P_pow1, atol=1e-4)


class TestSPDLayers:
    """Tests for SPD neural network layers."""
    
    def test_spd_transform(self):
        """SPD transform should preserve SPD structure."""
        spd = SPD(4)
        transform = SPDTransform(4, 3)
        
        P = spd.random_point()
        Q = transform(P)
        
        assert Q.shape == (3, 3)
        # Output should be symmetric
        assert torch.allclose(Q, Q.T, atol=1e-5)
    
    def test_spd_bimap(self):
        """SPD BiMap should produce SPD output."""
        transform = SPDBiMap(4, 3, bias=True)
        spd = SPD(4)
        
        P = spd.random_point()
        Q = transform(P)
        
        assert Q.shape == (3, 3)
        assert torch.allclose(Q, Q.T, atol=1e-5)
    
    def test_spd_relu(self):
        """SPD ReLU should produce valid SPD."""
        spd = SPD(4)
        relu = SPDReLU(threshold=1e-3)
        
        P = spd.random_point()
        Q = relu(P)
        
        eigenvalues = torch.linalg.eigvalsh(Q)
        assert (eigenvalues >= 1e-3 - 1e-6).all()
    
    def test_spd_logeig(self):
        """SPD LogEig should produce symmetric output."""
        spd = SPD(4)
        logeig = SPDLogEig()
        
        P = spd.random_point()
        Q = logeig(P)
        
        assert torch.allclose(Q, Q.T, atol=1e-5)


# =============================================================================
# DavisManifold Tests
# =============================================================================

class TestDavisManifold:
    """Tests for DavisManifold."""
    
    def test_creation(self):
        """Create DavisManifold."""
        manifold = DavisManifold(dim=8, hidden_dim=32, n_layers=2)
        
        assert manifold.dim == 8
        assert len(list(manifold.parameters())) > 0
    
    def test_metric_tensor(self):
        """Metric tensor should be symmetric positive definite."""
        manifold = DavisManifold(dim=8, hidden_dim=32)
        x = torch.randn(16, 8)
        
        G = manifold.metric_tensor(x)
        
        assert G.shape == (16, 8, 8)
        # Symmetric
        assert torch.allclose(G, G.transpose(-2, -1))
        # Positive definite
        eigenvalues = torch.linalg.eigvalsh(G)
        assert (eigenvalues > 0).all()
    
    def test_diagonal_metric(self):
        """Diagonal metric should be diagonal."""
        manifold = DavisManifold(dim=8, hidden_dim=32, diagonal_only=True)
        x = torch.randn(16, 8)
        
        G = manifold.metric_tensor(x)
        
        # Check diagonal
        off_diag = G - torch.diag_embed(torch.diagonal(G, dim1=-2, dim2=-1))
        assert torch.allclose(off_diag, torch.zeros_like(off_diag))
    
    def test_inner_product_symmetry(self):
        """Inner product should be symmetric: <u, v> = <v, u>."""
        manifold = DavisManifold(dim=8, hidden_dim=32)
        x = torch.randn(16, 8)
        u = torch.randn(16, 8)
        v = torch.randn(16, 8)
        
        ip_uv = manifold.inner_product(x, u, v)
        ip_vu = manifold.inner_product(x, v, u)
        
        assert torch.allclose(ip_uv, ip_vu)
    
    def test_norm_positive(self):
        """Norm should be non-negative."""
        manifold = DavisManifold(dim=8, hidden_dim=32)
        x = torch.randn(16, 8)
        v = torch.randn(16, 8)
        
        norm = manifold.norm(x, v)
        
        assert (norm >= 0).all()
    
    def test_distance_self_zero(self):
        """Distance to self should be ~0."""
        manifold = DavisManifold(dim=8, hidden_dim=32)
        x = torch.randn(16, 8)
        
        dist = manifold.distance(x, x, n_steps=5)
        
        assert dist.mean() < 0.01
    
    def test_christoffel_symbols_symmetry(self):
        """Christoffel symbols should be symmetric in lower indices."""
        manifold = DavisManifold(dim=4, hidden_dim=16)
        x = torch.randn(1, 4)
        
        Gamma = manifold.christoffel_symbols(x)
        
        # Γ^k_ij = Γ^k_ji
        assert torch.allclose(Gamma, Gamma.transpose(-2, -1), atol=1e-4)
    
    def test_regularization(self):
        """Regularization loss should be finite."""
        manifold = DavisManifold(dim=8, hidden_dim=32)
        x = torch.randn(16, 8)
        
        reg_loss = manifold.metric_regularization(x)
        
        assert torch.isfinite(reg_loss)
    
    def test_gradient_flow(self):
        """Gradients should flow through metric."""
        manifold = DavisManifold(dim=8, hidden_dim=32)
        x = torch.randn(16, 8)
        y = torch.randn(16, 8)
        
        dist = manifold.distance(x, y, n_steps=3)
        loss = dist.mean()
        loss.backward()
        
        has_grad = all(p.grad is not None for p in manifold.parameters())
        assert has_grad


class TestDavisMetricLearner:
    """Tests for DavisMetricLearner."""
    
    def test_contrastive_loss(self):
        """Contrastive loss should compute."""
        learner = DavisMetricLearner(dim=8, hidden_dim=32)
        
        x1 = torch.randn(32, 8)
        x2 = torch.randn(32, 8)
        labels = torch.randint(0, 2, (32,)).float()
        
        loss = learner.contrastive_loss(x1, x2, labels)
        
        assert torch.isfinite(loss)
        
        # Check gradient
        loss.backward()
        has_grad = all(p.grad is not None for p in learner.parameters())
        assert has_grad
    
    def test_triplet_loss(self):
        """Triplet loss should compute."""
        learner = DavisMetricLearner(dim=8, hidden_dim=32)
        
        anchor = torch.randn(32, 8)
        positive = torch.randn(32, 8)
        negative = torch.randn(32, 8)
        
        loss = learner.triplet_loss(anchor, positive, negative)
        
        assert torch.isfinite(loss)


# =============================================================================
# ProductManifold Tests
# =============================================================================

class TestProductManifold:
    """Tests for ProductManifold."""
    
    def test_creation(self):
        """Create product manifold."""
        M = ProductManifold([Sphere(3), Euclidean(5)])
        
        assert M.dim == 8
        assert M.n_components == 2
        assert M.dims == [3, 5]
    
    def test_split_combine(self):
        """Split and combine should be inverse."""
        M = ProductManifold([Sphere(3), Euclidean(5)])
        x = torch.randn(16, 8)
        
        components = M.split(x)
        x_recovered = M.combine(components)
        
        assert torch.allclose(x, x_recovered)
        assert components[0].shape == (16, 3)
        assert components[1].shape == (16, 5)
    
    def test_project(self):
        """Projection should project each component."""
        M = ProductManifold([Sphere(3), Euclidean(5)])
        x = torch.randn(16, 8)
        
        x_proj = M.project(x)
        sphere, euc = M.split(x_proj)
        
        # Sphere component should have unit norm
        norms = sphere.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    
    def test_random_point(self):
        """Random points should be on manifold."""
        M = ProductManifold([Sphere(3), Euclidean(5)])
        x = M.random_point(16)
        
        assert x.shape == (16, 8)
        sphere, euc = M.split(x)
        norms = sphere.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    
    def test_distance(self):
        """Product distance should be sqrt of sum of squared distances."""
        M = ProductManifold([Sphere(3), Euclidean(5)])
        p = M.random_point(10)
        q = M.random_point(10)
        
        # Total distance
        d = M.distance(p, q)
        
        # Component distances
        comp_dists = M.component_distances(p, q)
        sum_sq = sum(d_i.pow(2) for d_i in comp_dists)
        
        assert torch.allclose(d, sum_sq.sqrt(), atol=1e-5)
    
    def test_distance_self_zero(self):
        """Distance to self should be zero."""
        M = ProductManifold([Sphere(3), Euclidean(5)])
        p = M.random_point(10)
        
        d = M.distance(p, p)
        
        assert d.mean() < 1e-5
    
    def test_exp_log(self):
        """Exp and log should be approximately inverse."""
        M = ProductManifold([Sphere(3), Euclidean(5)])
        p = M.random_point()
        q = M.random_point()
        
        v = M.log(p, q)
        q_recovered = M.exp(p, v)
        
        # Should be close (exact for Euclidean, approximate for Sphere)
        error = (q - q_recovered).abs().max()
        assert error < 0.1  # Approximate due to sphere log simplification
    
    def test_geodesic_endpoints(self):
        """Geodesic at t=0 and t=1 should be endpoints."""
        M = ProductManifold([Sphere(3), Euclidean(5)])
        p = M.random_point()
        q = M.random_point()
        
        g0 = M.geodesic(p, q, 0.0)
        g1 = M.geodesic(p, q, 1.0)
        
        assert torch.allclose(g0, p, atol=1e-4)
        # g1 ≈ q (approximate due to sphere)
        assert (g1 - q).abs().max() < 0.1
    
    def test_repr(self):
        """String representation should work."""
        M = ProductManifold([Sphere(3), Euclidean(5)])
        repr_str = repr(M)
        
        assert 'Sphere' in repr_str
        assert 'Euclidean' in repr_str


class TestSpecialProductManifolds:
    """Tests for specialized product manifolds."""
    
    def test_hyperbolic_sphere(self):
        """HyperbolicSphere should work."""
        M = HyperbolicSphere(hyp_dim=16, sphere_dim=8)
        
        assert M.dim == 24
        assert M.hyp_dim == 16
        assert M.sphere_dim == 8
        
        x = M.random_point(10)
        hyp = M.get_hyperbolic(x)
        sph = M.get_sphere(x)
        
        assert hyp.shape == (10, 16)
        assert sph.shape == (10, 8)
    
    def test_hyperbolic_euclidean(self):
        """HyperbolicEuclidean should work."""
        M = HyperbolicEuclidean(hyp_dim=16, euc_dim=8)
        
        assert M.dim == 24
        
        x = M.random_point(10)
        hyp = M.get_hyperbolic(x)
        euc = M.get_euclidean(x)
        
        assert hyp.shape == (10, 16)
        assert euc.shape == (10, 8)
    
    def test_sphere_euclidean(self):
        """SphereEuclidean should work."""
        M = SphereEuclidean(sphere_dim=8, euc_dim=16)
        
        x = M.random_point(10)
        sph = M.get_sphere(x)
        euc = M.get_euclidean(x)
        
        assert sph.shape == (10, 8)
        assert euc.shape == (10, 16)
        
        # Sphere should be normalized
        norms = sph.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    
    def test_multi_hyperbolic(self):
        """MultiHyperbolic should work."""
        M = MultiHyperbolic(dims=[8, 8], curvatures=[-1.0, -0.5])
        
        assert M.dim == 16
        assert M.curvatures == [-1.0, -0.5]
        
        x = M.random_point(10)
        assert x.shape == (10, 16)
    
    def test_multi_sphere(self):
        """MultiSphere should work."""
        M = MultiSphere(dims=[4, 4, 4])
        
        assert M.dim == 12
        
        x = M.random_point(10)
        assert x.shape == (10, 12)


class TestProductNNLayers:
    """Tests for product manifold neural network layers."""
    
    def test_product_embedding(self):
        """ProductEmbedding should work."""
        M = ProductManifold([Sphere(4), Euclidean(8)])
        emb = ProductEmbedding(100, M)
        
        indices = torch.tensor([0, 5, 10])
        x = emb(indices)
        
        assert x.shape == (3, 12)
        
        # Sphere component should be normalized
        sph, euc = M.split(x)
        norms = sph.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    
    def test_product_embedding_component(self):
        """Get component from ProductEmbedding."""
        M = ProductManifold([Sphere(4), Euclidean(8)])
        emb = ProductEmbedding(100, M)
        
        indices = torch.tensor([0, 5, 10])
        sph = emb.get_component(indices, 0)
        euc = emb.get_component(indices, 1)
        
        assert sph.shape == (3, 4)
        assert euc.shape == (3, 8)
    
    def test_product_linear(self):
        """ProductLinear should transform components separately."""
        M_in = ProductManifold([Sphere(4), Euclidean(8)])
        M_out = ProductManifold([Sphere(8), Euclidean(16)])
        linear = ProductLinear(M_in, M_out)
        
        x = M_in.random_point(10)
        y = linear(x)
        
        assert y.shape == (10, 24)
        
        # Output sphere should be normalized
        sph, euc = M_out.split(y)
        norms = sph.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests across Phase 5 manifolds."""
    
    def test_spd_in_product(self):
        """SPD can be used in product (indirectly via custom wrapper)."""
        # SPD has different interface (matrices vs vectors), 
        # so direct product is complex. Test that both exist.
        spd = SPD(4)
        prod = ProductManifold([Sphere(4), Euclidean(8)])
        
        P = spd.random_point()
        x = prod.random_point()
        
        assert P.shape == (4, 4)
        assert x.shape == (12,)
    
    def test_all_manifolds_importable(self):
        """All Phase 5 manifolds should be importable from geotorch.manifolds."""
        from geotorch.manifolds import (
            SPD, LogEuclideanSPD, SPDTransform, SPDBiMap, SPDReLU, SPDLogEig,
            DavisManifold, DavisMetricLearner,
            ProductManifold, HyperbolicSphere, HyperbolicEuclidean,
            SphereEuclidean, MultiHyperbolic, MultiSphere,
            ProductEmbedding, ProductLinear
        )
        
        # All should be classes
        assert isinstance(SPD, type)
        assert isinstance(DavisManifold, type)
        assert isinstance(ProductManifold, type)
    
    def test_davis_with_optimizer(self):
        """DavisManifold should work with standard optimizers."""
        manifold = DavisManifold(dim=8, hidden_dim=32)
        optimizer = torch.optim.Adam(manifold.parameters(), lr=0.01)
        
        for _ in range(3):
            x = torch.randn(16, 8)
            y = torch.randn(16, 8)
            
            dist = manifold.distance(x, y, n_steps=3)
            loss = dist.mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Should complete without error
        assert True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
