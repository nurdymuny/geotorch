"""
ProductManifold - Cartesian Product of Riemannian Manifolds.

The product manifold M = M₁ × M₂ × ... × M_k consists of tuples
(p₁, p₂, ..., p_k) where p_i ∈ M_i.

The metric is the sum of component metrics (orthogonal product):
    ⟨(u₁, u₂), (v₁, v₂)⟩ = ⟨u₁, v₁⟩_{M₁} + ⟨u₂, v₂⟩_{M₂}

All operations are componentwise:
    exp_{(p,q)}(u, v) = (exp_p(u), exp_q(v))
    log_{(p,q)}(x, y) = (log_p(x), log_q(y))
    d((p,q), (x,y)) = sqrt(d(p,x)² + d(q,y)²)

Applications:
- Complex embeddings (hierarchical + directional: Hyperbolic × Sphere)
- Multi-scale representations (multiple Hyperbolic spaces)
- Factorized geometry (different aspects on different manifolds)
"""

from typing import List, Tuple
import torch
from torch import Tensor
import torch.nn as nn


class ProductManifold:
    """
    Product of multiple Riemannian manifolds.
    
    The product manifold M = M₁ × M₂ × ... × M_k consists of tuples
    (p₁, p₂, ..., p_k) where p_i ∈ M_i.
    
    Points are represented as concatenated vectors: [p₁ | p₂ | ... | p_k].
    
    Args:
        manifolds: List of component manifolds
    
    Example:
        >>> from geotorch.manifolds import Hyperbolic, Sphere
        >>> 
        >>> # Hierarchical embeddings: depth (Hyperbolic) + features (Sphere)
        >>> M = ProductManifold([
        ...     Hyperbolic(16),   # Hierarchy structure
        ...     Sphere(48)        # Directional attributes
        ... ])
        >>> 
        >>> x = M.random_point(32)  # Batch of 32 points
        >>> x_hyp, x_sphere = M.split(x)  # Split into components
        >>> 
        >>> # Distance combines both geometries
        >>> d = M.distance(x[0], x[1])
    """
    
    def __init__(self, manifolds: List):
        self.manifolds = manifolds
        self.n_components = len(manifolds)
        
        # Compute dimensions and split points
        # Use ambient dimension (_ambient_dim or n) for vector concatenation
        self.dims = []
        for m in manifolds:
            if hasattr(m, '_ambient_dim'):
                self.dims.append(m._ambient_dim)
            elif hasattr(m, 'n'):
                self.dims.append(m.n)
            elif hasattr(m, 'dim'):
                self.dims.append(m.dim)
            else:
                raise ValueError(f"Manifold {m} has no dimension attribute")
        
        self.dim = sum(self.dims)
        self.n = self.dim  # Alias for compatibility
        
        # Split indices
        self.split_indices: List[Tuple[int, int]] = []
        idx = 0
        for d in self.dims:
            self.split_indices.append((idx, idx + d))
            idx += d
    
    def split(self, x: Tensor) -> List[Tensor]:
        """
        Split concatenated point into components.
        
        Args:
            x: Product point, shape (..., total_dim)
        
        Returns:
            List of component tensors
        """
        return [x[..., start:end] for start, end in self.split_indices]
    
    def combine(self, components: List[Tensor]) -> Tensor:
        """
        Combine component points into product point.
        
        Args:
            components: List of component tensors
        
        Returns:
            Concatenated tensor, shape (..., total_dim)
        """
        return torch.cat(components, dim=-1)
    
    def project(self, x: Tensor) -> Tensor:
        """
        Project each component onto its manifold.
        
        Args:
            x: Point to project
        
        Returns:
            Projected point
        """
        components = self.split(x)
        projected = [
            m.project(c) for m, c in zip(self.manifolds, components)
        ]
        return self.combine(projected)
    
    def project_tangent(self, p: Tensor, v: Tensor) -> Tensor:
        """
        Project tangent vector componentwise.
        
        Args:
            p: Base point
            v: Vector to project
        
        Returns:
            Projected tangent vector
        """
        p_split = self.split(p)
        v_split = self.split(v)
        
        projected = [
            m.project_tangent(p_i, v_i)
            for m, p_i, v_i in zip(self.manifolds, p_split, v_split)
        ]
        return self.combine(projected)
    
    def exp(self, p: Tensor, v: Tensor) -> Tensor:
        """
        Exponential map (componentwise).
        
        exp_{(p₁,p₂)}(v₁, v₂) = (exp_{p₁}(v₁), exp_{p₂}(v₂))
        
        Args:
            p: Base point on product manifold
            v: Tangent vector
        
        Returns:
            Point on manifold
        """
        p_split = self.split(p)
        v_split = self.split(v)
        
        results = [
            m.exp(p_i, v_i)
            for m, p_i, v_i in zip(self.manifolds, p_split, v_split)
        ]
        return self.combine(results)
    
    def log(self, p: Tensor, q: Tensor) -> Tensor:
        """
        Logarithm map (componentwise).
        
        log_{(p₁,p₂)}(q₁, q₂) = (log_{p₁}(q₁), log_{p₂}(q₂))
        
        Args:
            p: Base point
            q: Target point
        
        Returns:
            Tangent vector
        """
        p_split = self.split(p)
        q_split = self.split(q)
        
        results = [
            m.log(p_i, q_i)
            for m, p_i, q_i in zip(self.manifolds, p_split, q_split)
        ]
        return self.combine(results)
    
    def distance(self, p: Tensor, q: Tensor) -> Tensor:
        """
        Product distance: sqrt(d₁² + d₂² + ... + d_k²)
        
        Args:
            p, q: Points on product manifold
        
        Returns:
            Distance value(s)
        """
        p_split = self.split(p)
        q_split = self.split(q)
        
        squared_dists = [
            m.distance(p_i, q_i) ** 2
            for m, p_i, q_i in zip(self.manifolds, p_split, q_split)
        ]
        
        return torch.sqrt(sum(squared_dists))
    
    def component_distances(self, p: Tensor, q: Tensor) -> List[Tensor]:
        """
        Get individual distances for each component.
        
        Args:
            p, q: Points on product manifold
        
        Returns:
            List of component distances
        """
        p_split = self.split(p)
        q_split = self.split(q)
        
        return [
            m.distance(p_i, q_i)
            for m, p_i, q_i in zip(self.manifolds, p_split, q_split)
        ]
    
    def inner_product(self, p: Tensor, u: Tensor, v: Tensor) -> Tensor:
        """
        Sum of component inner products.
        
        Args:
            p: Base point
            u, v: Tangent vectors
        
        Returns:
            Inner product value
        """
        p_split = self.split(p)
        u_split = self.split(u)
        v_split = self.split(v)
        
        products = []
        for m, p_i, u_i, v_i in zip(self.manifolds, p_split, u_split, v_split):
            if hasattr(m, 'inner_product'):
                products.append(m.inner_product(p_i, u_i, v_i))
            else:
                # Default Euclidean inner product
                products.append((u_i * v_i).sum(dim=-1))
        
        return sum(products)
    
    def norm(self, p: Tensor, v: Tensor) -> Tensor:
        """Norm of tangent vector in product metric."""
        return self.inner_product(p, v, v).sqrt()
    
    def random_point(self, *batch_shape) -> Tensor:
        """
        Random point from product distribution.
        
        Each component is sampled from its manifold's distribution.
        
        Args:
            *batch_shape: Optional batch dimensions
        
        Returns:
            Random point(s)
        """
        components = []
        for m in self.manifolds:
            if hasattr(m, 'random_point'):
                components.append(m.random_point(*batch_shape))
            else:
                # Default: standard normal
                dim = m.dim if hasattr(m, 'dim') else m.n
                components.append(torch.randn(*batch_shape, dim))
        return self.combine(components)
    
    def parallel_transport(self, p: Tensor, q: Tensor, v: Tensor) -> Tensor:
        """
        Parallel transport (componentwise).
        
        Args:
            p: Source point
            q: Target point
            v: Tangent vector at p
        
        Returns:
            Transported tangent vector at q
        """
        p_split = self.split(p)
        q_split = self.split(q)
        v_split = self.split(v)
        
        results = []
        for m, p_i, q_i, v_i in zip(self.manifolds, p_split, q_split, v_split):
            if hasattr(m, 'parallel_transport'):
                results.append(m.parallel_transport(p_i, q_i, v_i))
            else:
                # Default: identity transport
                results.append(v_i)
        
        return self.combine(results)
    
    def retract(self, p: Tensor, v: Tensor) -> Tensor:
        """
        Retraction (componentwise).
        
        Args:
            p: Base point
            v: Tangent vector
        
        Returns:
            Point on manifold
        """
        p_split = self.split(p)
        v_split = self.split(v)
        
        results = []
        for m, p_i, v_i in zip(self.manifolds, p_split, v_split):
            if hasattr(m, 'retract'):
                results.append(m.retract(p_i, v_i))
            else:
                # Fall back to exp
                results.append(m.exp(p_i, v_i))
        
        return self.combine(results)
    
    def geodesic(self, p: Tensor, q: Tensor, t: float) -> Tensor:
        """
        Point on geodesic at time t ∈ [0, 1].
        
        Args:
            p: Start point
            q: End point
            t: Time parameter
        
        Returns:
            Point on geodesic
        """
        v = self.log(p, q)
        return self.exp(p, t * v)
    
    def __repr__(self) -> str:
        manifold_names = [m.__class__.__name__ for m in self.manifolds]
        dims = [str(d) for d in self.dims]
        components = [f"{name}({dim})" for name, dim in zip(manifold_names, dims)]
        return f"ProductManifold({' x '.join(components)})"


# =============================================================================
# Common Product Manifolds
# =============================================================================

class HyperbolicSphere(ProductManifold):
    """
    Hyperbolic × Sphere product manifold.
    
    Useful for embeddings with:
    - Hierarchical structure (Hyperbolic component)
    - Directional/categorical attributes (Sphere component)
    
    Example use case: Knowledge graph entities with types
    
    Args:
        hyp_dim: Dimension of hyperbolic space
        sphere_dim: Dimension of sphere
        curvature: Curvature of hyperbolic space (negative)
    """
    
    def __init__(self, hyp_dim: int, sphere_dim: int, curvature: float = -1.0):
        # Import here to avoid circular imports
        from geotorch.manifolds import Hyperbolic, Sphere
        
        self.hyp_dim = hyp_dim
        self.sphere_dim = sphere_dim
        
        super().__init__([
            Hyperbolic(hyp_dim, curvature=curvature),
            Sphere(sphere_dim)
        ])
    
    def get_hyperbolic(self, x: Tensor) -> Tensor:
        """Extract hyperbolic component."""
        return self.split(x)[0]
    
    def get_sphere(self, x: Tensor) -> Tensor:
        """Extract sphere component."""
        return self.split(x)[1]


class HyperbolicEuclidean(ProductManifold):
    """
    Hyperbolic × Euclidean product manifold.
    
    Useful for embeddings with:
    - Hierarchical structure (Hyperbolic component)
    - Continuous attributes (Euclidean component)
    
    Example use case: Products with category hierarchy + numerical features
    
    Args:
        hyp_dim: Dimension of hyperbolic space
        euc_dim: Dimension of Euclidean space
        curvature: Curvature of hyperbolic space (negative)
    """
    
    def __init__(self, hyp_dim: int, euc_dim: int, curvature: float = -1.0):
        from geotorch.manifolds import Hyperbolic, Euclidean
        
        self.hyp_dim = hyp_dim
        self.euc_dim = euc_dim
        
        super().__init__([
            Hyperbolic(hyp_dim, curvature=curvature),
            Euclidean(euc_dim)
        ])
    
    def get_hyperbolic(self, x: Tensor) -> Tensor:
        """Extract hyperbolic component."""
        return self.split(x)[0]
    
    def get_euclidean(self, x: Tensor) -> Tensor:
        """Extract Euclidean component."""
        return self.split(x)[1]


class SphereEuclidean(ProductManifold):
    """
    Sphere × Euclidean: Direction + Features
    
    Args:
        sphere_dim: Dimension of sphere
        euc_dim: Dimension of Euclidean space
    """
    
    def __init__(self, sphere_dim: int, euc_dim: int):
        from geotorch.manifolds import Sphere, Euclidean
        
        self.sphere_dim = sphere_dim
        self.euc_dim = euc_dim
        
        super().__init__([
            Sphere(sphere_dim),
            Euclidean(euc_dim)
        ])
    
    def get_sphere(self, x: Tensor) -> Tensor:
        """Extract sphere component."""
        return self.split(x)[0]
    
    def get_euclidean(self, x: Tensor) -> Tensor:
        """Extract Euclidean component."""
        return self.split(x)[1]


class MultiHyperbolic(ProductManifold):
    """
    Product of multiple hyperbolic spaces with different curvatures.
    
    Each component can capture hierarchies at different scales.
    
    Args:
        dims: Dimensions of each hyperbolic space
        curvatures: Curvatures of each space (all negative)
    """
    
    def __init__(self, dims: List[int], curvatures: List[float]):
        from geotorch.manifolds import Hyperbolic
        
        assert len(dims) == len(curvatures), "dims and curvatures must have same length"
        
        manifolds = [
            Hyperbolic(d, curvature=c)
            for d, c in zip(dims, curvatures)
        ]
        super().__init__(manifolds)
        
        self.curvatures = curvatures


class MultiSphere(ProductManifold):
    """
    Product of multiple spheres (torus-like structure for multiple angles).
    
    Args:
        dims: Dimensions of each sphere
    """
    
    def __init__(self, dims: List[int]):
        from geotorch.manifolds import Sphere
        
        manifolds = [Sphere(d) for d in dims]
        super().__init__(manifolds)


# =============================================================================
# Neural Network Integration
# =============================================================================

class ProductEmbedding(nn.Module):
    """
    Embedding table on product manifold.
    
    Each embedding has multiple components on different manifolds.
    
    Args:
        num_embeddings: Number of embeddings
        manifold: ProductManifold instance
        scale: Initialization scale
    
    Example:
        >>> M = HyperbolicSphere(16, 32)
        >>> embeddings = ProductEmbedding(10000, M)
        >>> emb = embeddings(torch.tensor([0, 5, 10]))  # (3, 48)
    """
    
    def __init__(
        self,
        num_embeddings: int,
        manifold: ProductManifold,
        scale: float = 0.01
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.manifold = manifold
        
        # Initialize on product manifold
        embeddings = manifold.random_point(num_embeddings) * scale
        embeddings = manifold.project(embeddings)
        self.embeddings = nn.Parameter(embeddings)
    
    def forward(self, indices: Tensor) -> Tensor:
        """
        Look up embeddings.
        
        Args:
            indices: Embedding indices
        
        Returns:
            Projected embeddings
        """
        return self.manifold.project(self.embeddings[indices])
    
    def get_component(self, indices: Tensor, component: int) -> Tensor:
        """
        Get specific component of embeddings.
        
        Args:
            indices: Embedding indices
            component: Component index
        
        Returns:
            Component embeddings
        """
        emb = self.forward(indices)
        return self.manifold.split(emb)[component]


class ProductLinear(nn.Module):
    """
    Linear layer respecting product structure.
    
    Applies separate transformations to each component.
    
    Args:
        manifold: Input ProductManifold
        out_manifold: Output ProductManifold
    """
    
    def __init__(
        self,
        manifold: ProductManifold,
        out_manifold: ProductManifold
    ):
        super().__init__()
        
        self.in_manifold = manifold
        self.out_manifold = out_manifold
        
        # Separate linear for each component
        self.linears = nn.ModuleList([
            nn.Linear(d_in, d_out)
            for d_in, d_out in zip(manifold.dims, out_manifold.dims)
        ])
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply componentwise linear transformations.
        
        Args:
            x: Input on in_manifold
        
        Returns:
            Output on out_manifold (projected)
        """
        components = self.in_manifold.split(x)
        
        transformed = [
            linear(comp)
            for linear, comp in zip(self.linears, components)
        ]
        
        combined = self.out_manifold.combine(transformed)
        return self.out_manifold.project(combined)
