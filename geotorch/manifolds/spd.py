"""
SPD (Symmetric Positive Definite) Manifold.

The space of symmetric positive definite matrices with the affine-invariant
Riemannian metric. Fundamental for:
- Covariance matrices (brain connectivity, financial correlations)
- Diffusion tensors (medical imaging)
- Kernel matrices (machine learning)

Mathematical background:
- SPD(n) = {P ∈ ℝⁿˣⁿ : P = Pᵀ, P ≻ 0}
- Tangent space at P: T_P SPD = Sym(n) (all symmetric matrices)
- Affine-invariant metric: ⟨U, V⟩_P = tr(P⁻¹ U P⁻¹ V)
- Geodesic: γ(t) = P^{1/2} (P^{-1/2} Q P^{-1/2})^t P^{1/2}
- Distance: d(P, Q) = ||log(P^{-1/2} Q P^{-1/2})||_F
"""

from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn


class SPD:
    """
    Manifold of Symmetric Positive Definite matrices.
    
    Uses the affine-invariant Riemannian metric by default, which is invariant
    under congruence transformations: d(A P A^T, A Q A^T) = d(P, Q).
    
    Args:
        n: Size of matrices (n×n)
        metric: Type of Riemannian metric:
            - 'affine': Affine-invariant metric (default, geometrically natural)
            - 'log_euclidean': Log-Euclidean metric (computationally simpler)
        eps: Small constant for numerical stability
    
    Example:
        >>> spd = SPD(4)
        >>> P = spd.random_point()  # Random 4×4 SPD matrix
        >>> Q = spd.random_point()
        >>> d = spd.distance(P, Q)  # Riemannian distance
        >>> V = spd.log(P, Q)       # Tangent vector from P to Q
        >>> Q_recovered = spd.exp(P, V)  # Recover Q
    """
    
    def __init__(self, n: int, metric: str = 'affine', eps: float = 1e-6):
        self.n = n
        self.dim = n * (n + 1) // 2  # Degrees of freedom
        self.metric = metric
        self.eps = eps
    
    def random_point(self, *batch_shape) -> Tensor:
        """
        Generate random SPD matrix via A @ A.T + εI.
        
        Args:
            *batch_shape: Optional batch dimensions
        
        Returns:
            Random SPD matrix(ces), shape (*batch_shape, n, n)
        """
        shape = (*batch_shape, self.n, self.n) if batch_shape else (self.n, self.n)
        A = torch.randn(shape)
        P = A @ A.transpose(-2, -1)
        # Add small diagonal for numerical stability
        P = P + self.eps * torch.eye(self.n, device=A.device)
        return P
    
    def project(self, X: Tensor) -> Tensor:
        """
        Project matrix to nearest SPD matrix.
        
        Uses symmetric part and eigenvalue thresholding.
        
        Args:
            X: Matrix to project, shape (..., n, n)
        
        Returns:
            Projected SPD matrix, shape (..., n, n)
        """
        # Symmetrize
        X_sym = 0.5 * (X + X.transpose(-2, -1))
        
        # Eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(X_sym)
        
        # Threshold eigenvalues to be positive
        eigenvalues = eigenvalues.clamp(min=self.eps)
        
        # Reconstruct
        return eigenvectors @ torch.diag_embed(eigenvalues) @ eigenvectors.transpose(-2, -1)
    
    def project_tangent(self, P: Tensor, V: Tensor) -> Tensor:
        """
        Project to tangent space at P.
        
        Tangent space is all symmetric matrices, so we symmetrize.
        
        Args:
            P: Base point (unused, tangent space is Sym(n) everywhere)
            V: Vector to project
        
        Returns:
            Symmetric matrix (tangent vector)
        """
        return 0.5 * (V + V.transpose(-2, -1))
    
    def inner_product(self, P: Tensor, U: Tensor, V: Tensor) -> Tensor:
        """
        Riemannian inner product: ⟨U, V⟩_P = tr(P⁻¹ U P⁻¹ V)
        
        Args:
            P: Base point on manifold
            U, V: Tangent vectors at P
        
        Returns:
            Inner product value(s)
        """
        P_inv = torch.linalg.inv(P)
        return torch.einsum('...ij,...jk,...kl,...li->...', P_inv, U, P_inv, V)
    
    def norm(self, P: Tensor, V: Tensor) -> Tensor:
        """Riemannian norm of tangent vector."""
        return self.inner_product(P, V, V).clamp(min=0).sqrt()
    
    def exp(self, P: Tensor, V: Tensor) -> Tensor:
        """
        Exponential map: exp_P(V).
        
        exp_P(V) = P^{1/2} exp(P^{-1/2} V P^{-1/2}) P^{1/2}
        
        Args:
            P: Base point (SPD matrix)
            V: Tangent vector (symmetric matrix)
        
        Returns:
            Point on manifold
        """
        if self.metric == 'affine':
            P_sqrt = self.sqrtm(P)
            P_invsqrt = self.invsqrtm(P)
            
            # Transform V to identity
            V_0 = P_invsqrt @ V @ P_invsqrt
            
            # Exponential at identity is matrix exponential
            exp_V_0 = self.expm(V_0)
            
            # Transform back
            return P_sqrt @ exp_V_0 @ P_sqrt
        
        elif self.metric == 'log_euclidean':
            return self.expm(self.logm(P) + V)
        
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def log(self, P: Tensor, Q: Tensor) -> Tensor:
        """
        Logarithm map: log_P(Q).
        
        log_P(Q) = P^{1/2} log(P^{-1/2} Q P^{-1/2}) P^{1/2}
        
        Args:
            P: Base point
            Q: Target point
        
        Returns:
            Tangent vector at P pointing toward Q
        """
        if self.metric == 'affine':
            P_sqrt = self.sqrtm(P)
            P_invsqrt = self.invsqrtm(P)
            
            # Transform Q to tangent space at identity
            Q_0 = P_invsqrt @ Q @ P_invsqrt
            
            # Log at identity
            log_Q_0 = self.logm(Q_0)
            
            # Transform back
            return P_sqrt @ log_Q_0 @ P_sqrt
        
        elif self.metric == 'log_euclidean':
            return self.logm(Q) - self.logm(P)
        
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def distance(self, P: Tensor, Q: Tensor) -> Tensor:
        """
        Geodesic distance: d(P, Q) = ||log(P^{-1/2} Q P^{-1/2})||_F
        
        Args:
            P, Q: SPD matrices
        
        Returns:
            Distance value(s)
        """
        if self.metric == 'affine':
            P_invsqrt = self.invsqrtm(P)
            M = P_invsqrt @ Q @ P_invsqrt
            log_M = self.logm(M)
            return torch.linalg.norm(log_M, ord='fro', dim=(-2, -1))
        
        elif self.metric == 'log_euclidean':
            diff = self.logm(P) - self.logm(Q)
            return torch.linalg.norm(diff, ord='fro', dim=(-2, -1))
        
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def parallel_transport(self, P: Tensor, Q: Tensor, V: Tensor) -> Tensor:
        """
        Parallel transport from T_P to T_Q.
        
        Γ_{P→Q}(V) = E V E^T where E = (Q P^{-1})^{1/2}
        
        Args:
            P: Source point
            Q: Target point
            V: Tangent vector at P
        
        Returns:
            Transported tangent vector at Q
        """
        E = self.sqrtm(Q @ torch.linalg.inv(P))
        return E @ V @ E.transpose(-2, -1)
    
    def geodesic(self, P: Tensor, Q: Tensor, t: float) -> Tensor:
        """
        Point on geodesic from P to Q at time t ∈ [0, 1].
        
        γ(t) = P^{1/2} (P^{-1/2} Q P^{-1/2})^t P^{1/2}
        
        Args:
            P: Start point
            Q: End point
            t: Time parameter (0 = P, 1 = Q)
        
        Returns:
            Point on geodesic
        """
        P_sqrt = self.sqrtm(P)
        P_invsqrt = self.invsqrtm(P)
        
        M = P_invsqrt @ Q @ P_invsqrt
        M_t = self.powm(M, t)
        
        return P_sqrt @ M_t @ P_sqrt
    
    # ==========================================================================
    # Matrix operations via eigendecomposition
    # ==========================================================================
    
    def sqrtm(self, P: Tensor) -> Tensor:
        """Matrix square root via eigendecomposition."""
        eigenvalues, eigenvectors = torch.linalg.eigh(P)
        eigenvalues = eigenvalues.clamp(min=self.eps)
        sqrt_eigenvalues = eigenvalues.sqrt()
        return eigenvectors @ torch.diag_embed(sqrt_eigenvalues) @ eigenvectors.transpose(-2, -1)
    
    def invsqrtm(self, P: Tensor) -> Tensor:
        """Inverse matrix square root."""
        eigenvalues, eigenvectors = torch.linalg.eigh(P)
        eigenvalues = eigenvalues.clamp(min=self.eps)
        invsqrt_eigenvalues = 1.0 / eigenvalues.sqrt()
        return eigenvectors @ torch.diag_embed(invsqrt_eigenvalues) @ eigenvectors.transpose(-2, -1)
    
    def logm(self, P: Tensor) -> Tensor:
        """Matrix logarithm via eigendecomposition."""
        eigenvalues, eigenvectors = torch.linalg.eigh(P)
        eigenvalues = eigenvalues.clamp(min=self.eps)
        log_eigenvalues = eigenvalues.log()
        return eigenvectors @ torch.diag_embed(log_eigenvalues) @ eigenvectors.transpose(-2, -1)
    
    def expm(self, V: Tensor) -> Tensor:
        """Matrix exponential via eigendecomposition (for symmetric V)."""
        eigenvalues, eigenvectors = torch.linalg.eigh(V)
        exp_eigenvalues = eigenvalues.exp()
        return eigenvectors @ torch.diag_embed(exp_eigenvalues) @ eigenvectors.transpose(-2, -1)
    
    def powm(self, P: Tensor, t: float) -> Tensor:
        """Matrix power P^t via eigendecomposition."""
        eigenvalues, eigenvectors = torch.linalg.eigh(P)
        eigenvalues = eigenvalues.clamp(min=self.eps)
        pow_eigenvalues = eigenvalues.pow(t)
        return eigenvectors @ torch.diag_embed(pow_eigenvalues) @ eigenvectors.transpose(-2, -1)
    
    # ==========================================================================
    # Fréchet mean
    # ==========================================================================
    
    def frechet_mean(
        self,
        Ps: Tensor,
        weights: Optional[Tensor] = None,
        n_iters: int = 10
    ) -> Tensor:
        """
        Fréchet mean of SPD matrices.
        
        μ = argmin_M Σ_i w_i d(M, P_i)²
        
        Uses iterative algorithm (Karcher mean).
        
        Args:
            Ps: SPD matrices, shape (N, n, n)
            weights: Optional weights, shape (N,)
            n_iters: Number of iterations
        
        Returns:
            Fréchet mean, shape (n, n)
        """
        N = Ps.shape[0]
        if weights is None:
            weights = torch.ones(N, device=Ps.device) / N
        
        # Initialize with arithmetic mean (projected)
        mean = self.project(Ps.mean(dim=0))
        
        for _ in range(n_iters):
            # Compute mean of log maps
            logs = torch.stack([self.log(mean, P) for P in Ps])
            weighted_log = (weights.view(N, 1, 1) * logs).sum(dim=0)
            
            # Update mean
            mean = self.exp(mean, weighted_log)
        
        return mean
    
    def __repr__(self) -> str:
        return f"SPD({self.n}, metric='{self.metric}')"


class LogEuclideanSPD(SPD):
    """
    SPD manifold with Log-Euclidean metric.
    
    Treats SPD as vector space after log transform:
        d(P, Q) = ||log(P) - log(Q)||_F
    
    Computationally simpler than affine-invariant but less geometrically natural.
    """
    
    def __init__(self, n: int):
        super().__init__(n, metric='log_euclidean')


# =============================================================================
# SPD Neural Network Layers
# =============================================================================

class SPDTransform(nn.Module):
    """
    Learnable transformation on SPD manifold.
    
    Applies congruence transformation: P → W P W^T
    This preserves SPD structure.
    
    Args:
        n_in: Input matrix size
        n_out: Output matrix size
    """
    
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.W = nn.Parameter(torch.eye(n_out, n_in))
    
    def forward(self, P: Tensor) -> Tensor:
        return self.W @ P @ self.W.T


class SPDBiMap(nn.Module):
    """
    Bilinear mapping on SPD: P → W P W^T + b
    
    Args:
        n_in: Input matrix size
        n_out: Output matrix size
        bias: Whether to include bias term
    """
    
    def __init__(self, n_in: int, n_out: int, bias: bool = True):
        super().__init__()
        
        self.W = nn.Parameter(torch.randn(n_out, n_in) * 0.1)
        if bias:
            # Bias is also SPD (via bb^T)
            self.bias = nn.Parameter(torch.eye(n_out) * 0.01)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, P: Tensor) -> Tensor:
        out = self.W @ P @ self.W.T
        if self.bias is not None:
            out = out + self.bias @ self.bias.T
        return out


class SPDReLU(nn.Module):
    """
    ReLU-like activation for SPD matrices.
    
    Applies threshold to eigenvalues to ensure SPD output.
    
    Args:
        threshold: Minimum eigenvalue threshold
    """
    
    def __init__(self, threshold: float = 1e-4):
        super().__init__()
        self.threshold = threshold
    
    def forward(self, P: Tensor) -> Tensor:
        eigenvalues, eigenvectors = torch.linalg.eigh(P)
        eigenvalues = eigenvalues.clamp(min=self.threshold)
        return eigenvectors @ torch.diag_embed(eigenvalues) @ eigenvectors.transpose(-2, -1)


class SPDLogEig(nn.Module):
    """
    Log-eigenvalue layer: maps SPD to symmetric matrix via log of eigenvalues.
    
    Useful as final layer before Euclidean classifier.
    
    Args:
        eps: Small constant for numerical stability
    """
    
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, P: Tensor) -> Tensor:
        eigenvalues, eigenvectors = torch.linalg.eigh(P)
        log_eigenvalues = (eigenvalues.clamp(min=self.eps)).log()
        return eigenvectors @ torch.diag_embed(log_eigenvalues) @ eigenvectors.transpose(-2, -1)
