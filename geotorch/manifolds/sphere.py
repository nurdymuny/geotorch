import torch
from .manifold import Manifold


class Sphere(Manifold):
    """
    Sphere manifold: unit-norm vectors in R^n.
    
    The sphere S^{n-1} = {x ∈ R^n : ||x|| = 1} embedded in n-dimensional space.
    """
    
    def __init__(self, size):
        """
        Initialize sphere manifold.
        
        Args:
            size: Tuple specifying tensor shape. The manifold constraint
                  applies to the last dimension (vectors are normalized).
        """
        super().__init__(size)
        self.name = f"Sphere{size}"
    
    def projx(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project point onto sphere by normalizing to unit length.
        
        Args:
            x: Point in ambient space R^n
        
        Returns:
            Normalized point on sphere S^{n-1}
        """
        return x / x.norm(dim=-1, keepdim=True)
    
    def proju(self, p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Project vector onto tangent space at point p.
        
        The tangent space T_pS at p is the orthogonal complement of p.
        We project by removing the component parallel to p: v - (v·p)p
        
        Args:
            p: Point on sphere
            v: Vector in ambient space
        
        Returns:
            Tangent vector in T_pS
        """
        # Compute dot product v·p along last dimension
        dot = (v * p).sum(dim=-1, keepdim=True)
        # Remove radial component
        return v - dot * p
    
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project ambient space point onto manifold.
        
        Projects by normalizing to unit length.
        
        Args:
            x: Point in ambient space
        
        Returns:
            Closest point on manifold (normalized)
        """
        return self.projx(x)
    
    def project_tangent(self, p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Project ambient vector onto tangent space at p.
        
        The tangent space at p is orthogonal to p, so we subtract the
        radial component: v - (v·p)p
        
        Args:
            p: Point on manifold
            v: Vector in ambient space
        
        Returns:
            Component of v in T_pM
        """
        return self.proju(p, v)
