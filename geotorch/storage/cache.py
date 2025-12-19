"""
DavisCache: Topological Cache Map Γ
===================================

The foundation of O(1) retrieval from the Davis-Wilson framework.

Maps manifold points to discrete bins based on geometric properties.
Key property: Γ(x) ≠ Γ(y) ⟹ d(x,y) ≥ κ

Adapted from geodesic_storage.py spatial hashing.
"""

import torch
from torch import Tensor
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import math

from .binning import curvature_binning, spatial_binning, lsh_binning, LSHPlanes


class SpatialHash:
    """
    Spatial hash table for O(1) cell lookup.
    
    From geodesic_storage.py GeodesicStorage.spatial_hash pattern.
    Maps grid cells to lists of item indices.
    """
    
    def __init__(self, n_cells: int = 1024):
        """
        Args:
            n_cells: Number of hash cells
        """
        self.n_cells = n_cells
        self.cells: Dict[int, List[int]] = defaultdict(list)
        self._item_to_cell: Dict[int, int] = {}
    
    def add(self, item_id: int, cell_idx: int):
        """Add item to a cell."""
        self.cells[cell_idx].append(item_id)
        self._item_to_cell[item_id] = cell_idx
    
    def remove(self, item_id: int) -> bool:
        """Remove item from its cell."""
        if item_id not in self._item_to_cell:
            return False
        
        cell_idx = self._item_to_cell[item_id]
        if item_id in self.cells[cell_idx]:
            self.cells[cell_idx].remove(item_id)
        del self._item_to_cell[item_id]
        return True
    
    def get_cell(self, cell_idx: int) -> List[int]:
        """Get all items in a cell. O(1)."""
        return self.cells.get(cell_idx, [])
    
    def get_item_cell(self, item_id: int) -> Optional[int]:
        """Get which cell an item is in."""
        return self._item_to_cell.get(item_id)
    
    def get_nearby_cells(self, cell_idx: int, radius: int = 1) -> List[int]:
        """
        Get neighboring cells (for expanded search).
        
        For grid-based cells, returns adjacent cell indices.
        """
        neighbors = [cell_idx]
        
        # Simple approach: ±radius around cell index
        # More sophisticated: decode cell, get grid neighbors, re-encode
        for offset in range(1, radius + 1):
            neighbors.append((cell_idx + offset) % self.n_cells)
            neighbors.append((cell_idx - offset) % self.n_cells)
        
        return neighbors
    
    def clear(self):
        """Clear all cells."""
        self.cells.clear()
        self._item_to_cell.clear()
    
    def stats(self) -> Dict:
        """Statistics about the hash table."""
        sizes = [len(items) for items in self.cells.values()]
        return {
            'n_cells': self.n_cells,
            'n_nonempty': len(self.cells),
            'total_items': len(self._item_to_cell),
            'avg_cell_size': sum(sizes) / max(len(sizes), 1),
            'max_cell_size': max(sizes) if sizes else 0,
            'min_cell_size': min(sizes) if sizes else 0,
        }


class DavisCache:
    """
    Topological cache map Γ: M → {0, ..., K-1}
    
    Maps manifold points to discrete bins based on geometric properties.
    The foundation of O(1) retrieval from Davis-Wilson framework.
    
    Key insight from Yang-Mills proof:
        Γ(x) ≠ Γ(y) ⟹ d(x,y) ≥ κ
        "Different bins means geometrically distinguishable"
    
    Methods:
        - 'curvature': Use manifold-specific geometric properties
        - 'spatial': Grid-based spatial hashing (from geodesic_storage.py)
        - 'lsh': Locality-sensitive hashing
    
    Example:
        >>> cache = DavisCache(Hyperbolic(64), n_bins=1024)
        >>> x = manifold.random_point()
        >>> bin_idx = cache(x)  # O(1) bin assignment
    """
    
    def __init__(
        self,
        manifold,
        n_bins: int = 1024,
        method: str = 'curvature',
        kappa: float = 0.1,
        dim: Optional[int] = None
    ):
        """
        Args:
            manifold: GeoTorch manifold (Hyperbolic, Sphere, Euclidean)
            n_bins: Number of cache bins (K)
            method: Binning method ('curvature', 'spatial', 'lsh')
            kappa: Minimum distinguishability threshold
            dim: Dimension (inferred from manifold if not provided)
        """
        self.manifold = manifold
        self.n_bins = n_bins
        self.method = method
        self.kappa = kappa
        self.dim = dim or getattr(manifold, 'n', 64)
        
        # For LSH method
        if method == 'lsh':
            n_hashes = max(1, int(math.log2(n_bins)))
            self.lsh = LSHPlanes(self.dim, n_hashes)
        else:
            self.lsh = None
        
        # Spatial hash for storing items per bin
        self.spatial_hash = SpatialHash(n_bins)
    
    def __call__(self, x: Tensor) -> Tensor:
        """
        Compute cache label Γ(x).
        
        Args:
            x: Points on manifold, shape (..., D)
        
        Returns:
            Bin indices, shape (...)
        """
        if self.method == 'curvature':
            return curvature_binning(x, self.manifold, self.n_bins)
        elif self.method == 'spatial':
            return spatial_binning(x, self.n_bins)
        elif self.method == 'lsh':
            return self.lsh(x, self.n_bins)
        elif self.method == 'hybrid':
            # Hybrid: combine curvature and spatial binning
            curv_bins = curvature_binning(x, self.manifold, int(self.n_bins ** 0.5))
            spat_bins = spatial_binning(x, int(self.n_bins ** 0.5))
            # Combine: curv_bin * sqrt(n_bins) + spat_bin
            return curv_bins * int(self.n_bins ** 0.5) + spat_bins
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def compute_bin(self, x: Tensor) -> Tensor:
        """Alias for __call__."""
        return self(x)
    
    def add_to_bin(self, item_id: int, bin_idx: int):
        """Register an item in a bin."""
        self.spatial_hash.add(item_id, bin_idx)
    
    def remove_from_bin(self, item_id: int) -> bool:
        """Remove item from its bin."""
        return self.spatial_hash.remove(item_id)
    
    def get_bin_items(self, bin_idx: int) -> List[int]:
        """Get all item IDs in a bin. O(1)."""
        return self.spatial_hash.get_cell(bin_idx)
    
    def get_nearby_bin_items(self, bin_idx: int, radius: int = 1) -> List[int]:
        """Get items in this bin and neighboring bins."""
        cells = self.spatial_hash.get_nearby_cells(bin_idx, radius)
        items = []
        for cell in cells:
            items.extend(self.spatial_hash.get_cell(cell))
        return items
    
    def neighboring_bins(self, bin_idx: int, radius: int = 1) -> List[int]:
        """Get neighboring bin indices."""
        return self.spatial_hash.get_nearby_cells(bin_idx, radius)
    
    def curvature_signature(self, x: Tensor) -> Tensor:
        """
        Compute curvature-based signature for points.
        
        Returns a continuous signature that can be quantized to bins.
        Useful for debugging or visualization.
        """
        manifold_name = self.manifold.__class__.__name__
        
        if manifold_name == 'Hyperbolic':
            # Norm = depth, angle = direction
            norm = x.norm(dim=-1)
            if x.shape[-1] >= 2:
                angle = torch.atan2(x[..., 1], x[..., 0])
            else:
                angle = torch.zeros_like(norm)
            return torch.stack([norm, angle], dim=-1)
        
        elif manifold_name == 'Sphere':
            # First few coordinates as latitude-like values
            return x[..., :3].clone()
        
        else:
            return x[..., :3].clone()
    
    def stats(self) -> Dict:
        """Statistics about cache usage."""
        return self.spatial_hash.stats()
    
    def clear(self):
        """Clear all cached items."""
        self.spatial_hash.clear()
