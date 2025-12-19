"""
Binning Methods for O(1) Retrieval
==================================

Multiple strategies for mapping manifold points to discrete bins.
Adapted from geodesic_storage.py spatial hashing + Morton encoding.
"""

import torch
from torch import Tensor
from typing import Optional, Tuple
import math


def morton_encode_3d(x: Tensor, y: Tensor, z: Tensor, bits: int = 10) -> Tensor:
    """
    Morton Z-order encoding for 3D coordinates.
    
    Interleaves bits to preserve spatial locality:
    nearby points → nearby addresses
    
    From geodesic_storage.py AddressMapper.
    
    Args:
        x, y, z: Integer coordinates, shape (...)
        bits: Bits per dimension (max 21 for 64-bit output)
    
    Returns:
        Morton codes, shape (...)
    """
    def spread_bits(v: Tensor) -> Tensor:
        """Spread bits for interleaving."""
        v = v.long()
        # Spread bits apart: 0b111 -> 0b001001001
        v = (v | (v << 16)) & 0x030000FF
        v = (v | (v << 8)) & 0x0300F00F
        v = (v | (v << 4)) & 0x030C30C3
        v = (v | (v << 2)) & 0x09249249
        return v
    
    # Clamp to valid range
    max_val = (1 << bits) - 1
    x = x.clamp(0, max_val)
    y = y.clamp(0, max_val)
    z = z.clamp(0, max_val)
    
    # Interleave
    return spread_bits(x) | (spread_bits(y) << 1) | (spread_bits(z) << 2)


def morton_encode(coords: Tensor, n_dims: int = 3, bits: int = 10) -> Tensor:
    """
    Morton encoding for arbitrary dimensions.
    
    Args:
        coords: Coordinates, shape (..., D) in [0, 2^bits)
        n_dims: Number of dimensions to use
        bits: Bits per dimension
    
    Returns:
        Morton codes, shape (...)
    """
    if n_dims == 3:
        return morton_encode_3d(
            coords[..., 0].long(),
            coords[..., 1].long(),
            coords[..., 2].long(),
            bits
        )
    
    # General case: simple interleaving
    result = torch.zeros(coords.shape[:-1], dtype=torch.long, device=coords.device)
    for d in range(min(n_dims, coords.shape[-1])):
        result = result | (coords[..., d].long() << (d * bits))
    return result


def spatial_binning(
    x: Tensor,
    n_bins: int = 1024,
    grid_dims: int = 3,
    coord_range: Tuple[float, float] = (-1.0, 1.0)
) -> Tensor:
    """
    Spatial grid-based binning (from geodesic_storage.py _hash_point).
    
    Uses first `grid_dims` coordinates to create a spatial hash.
    Works well for Sphere (S^n) and Euclidean manifolds.
    
    Args:
        x: Points, shape (..., D)
        n_bins: Total number of bins
        grid_dims: Number of dimensions to use for grid
        coord_range: Expected coordinate range
    
    Returns:
        Bin indices, shape (...)
    """
    # Bins per dimension (cube root for 3D grid)
    bins_per_dim = max(2, int(round(n_bins ** (1.0 / grid_dims))))
    
    # Normalize coordinates to [0, bins_per_dim)
    lo, hi = coord_range
    coords = x[..., :grid_dims]
    normalized = (coords - lo) / (hi - lo + 1e-8)  # [0, 1]
    grid_coords = (normalized * bins_per_dim).long()
    grid_coords = grid_coords.clamp(0, bins_per_dim - 1)
    
    # Use Morton encoding for locality preservation
    if grid_dims == 3:
        bins = morton_encode_3d(
            grid_coords[..., 0],
            grid_coords[..., 1],
            grid_coords[..., 2],
            bits=int(math.log2(bins_per_dim)) + 1
        )
    else:
        # Simple linear combination
        bins = torch.zeros(x.shape[:-1], dtype=torch.long, device=x.device)
        multiplier = 1
        for d in range(grid_dims):
            bins = bins + grid_coords[..., d] * multiplier
            multiplier *= bins_per_dim
    
    return bins % n_bins


def curvature_binning(
    x: Tensor,
    manifold,
    n_bins: int = 1024
) -> Tensor:
    """
    Curvature-based binning adapted per manifold type.
    
    Uses intrinsic geometric properties for binning:
    - Hyperbolic: Distance from origin (hierarchy depth) + spatial
    - Sphere: Spatial grid on unit sphere
    - Euclidean: Spatial grid
    
    Args:
        x: Points on manifold, shape (..., D)
        manifold: GeoTorch manifold instance
        n_bins: Number of bins
    
    Returns:
        Bin indices, shape (...)
    """
    manifold_name = manifold.__class__.__name__
    
    if manifold_name == 'Hyperbolic':
        # Poincaré ball: depth (norm) is the key geometric property
        # Combine with spatial hash for angular distribution
        
        norm = x.norm(dim=-1)  # [0, 1) for Poincaré ball
        
        # Allocate some bins to depth, rest to spatial direction
        depth_bins = max(4, int(n_bins ** 0.25))  # e.g., 4 for 256 bins
        spatial_bins = n_bins // depth_bins
        
        # Depth bin from norm
        depth_idx = (norm * depth_bins).long().clamp(0, depth_bins - 1)
        
        # Spatial bin from coordinates (use raw coords for spatial locality)
        # Using more dimensions gives better distribution
        spatial_idx = spatial_binning(x, spatial_bins, grid_dims=min(6, x.shape[-1]))
        
        return (depth_idx * spatial_bins + spatial_idx) % n_bins
    
    elif manifold_name == 'Sphere':
        # S^n: points are unit vectors, use spatial hash on coordinates
        return spatial_binning(x, n_bins, grid_dims=min(6, x.shape[-1]))
    
    else:
        # Euclidean or unknown: spatial binning
        return spatial_binning(x, n_bins)


def lsh_binning(
    x: Tensor,
    planes: Tensor,
    n_bins: int = 1024
) -> Tensor:
    """
    Locality-Sensitive Hashing with random hyperplanes.
    
    Projects points onto random directions and uses sign pattern
    as the hash. Similar points likely have same hash.
    
    Args:
        x: Points, shape (..., D)
        planes: Random hyperplanes, shape (n_hashes, D)
        n_bins: Number of bins (should be 2^n_hashes)
    
    Returns:
        Bin indices, shape (...)
    """
    n_hashes = planes.shape[0]
    
    # Which side of each hyperplane?
    # Shape: (..., n_hashes)
    projections = x @ planes.T
    signs = (projections > 0).long()
    
    # Convert binary pattern to integer
    powers = (2 ** torch.arange(n_hashes, device=x.device)).long()
    bin_idx = (signs * powers).sum(-1)
    
    return bin_idx % n_bins


class LSHPlanes:
    """
    Manages random hyperplanes for LSH binning.
    
    Generates and caches random planes for consistent hashing.
    """
    
    def __init__(self, dim: int, n_hashes: int = 10, seed: Optional[int] = None):
        """
        Args:
            dim: Dimension of points
            n_hashes: Number of hash functions (bits)
            seed: Random seed for reproducibility
        """
        self.dim = dim
        self.n_hashes = n_hashes
        
        if seed is not None:
            torch.manual_seed(seed)
        
        # Random hyperplanes through origin
        self.planes = torch.randn(n_hashes, dim)
        self.planes = self.planes / self.planes.norm(dim=-1, keepdim=True)
    
    def to(self, device):
        """Move planes to device."""
        self.planes = self.planes.to(device)
        return self
    
    def __call__(self, x: Tensor, n_bins: int = 1024) -> Tensor:
        """Compute LSH bins for points x."""
        planes = self.planes.to(x.device)
        return lsh_binning(x, planes, n_bins)
