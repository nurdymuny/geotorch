"""
GeoStorage: O(1) Retrieval for Manifold Embeddings
===================================================

Storage backend using topological binning for constant-time similarity search.

Based on geodesic_storage.py GeodesicStorage patterns:
- Spatial hashing for O(1) cell lookup
- Morton encoding for address locality
- Read/write with manifold positions
"""

import torch
from torch import Tensor
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import time

from .cache import DavisCache


@dataclass
class StorageItem:
    """
    Item stored in GeoStorage.
    
    Like geodesic_storage.py StoredData but using tensors.
    """
    id: int
    embedding: Tensor
    bin_idx: int
    metadata: Optional[Dict[str, Any]] = None
    
    # For LRU eviction
    access_time: float = field(default_factory=time.time)


class GeoStorage:
    """
    O(1) retrieval storage for manifold embeddings.
    
    Uses DavisCache for topological binning, enabling constant-time
    similarity search. Adapted from geodesic_storage.py.
    
    Key insight:
        Traditional: Find THE nearest neighbor → O(N) or O(log N)
        GeoStorage: Find A geometrically similar neighbor → O(1)
    
    For most ML applications (retrieval, attention), this is sufficient.
    
    Example:
        >>> storage = GeoStorage(Hyperbolic(64), dim=64, n_bins=1024)
        >>> 
        >>> # Store embeddings
        >>> for emb in embeddings:
        ...     storage.add(emb, metadata={'text': ...})
        >>> 
        >>> # O(1) retrieval
        >>> results = storage.query(query_emb, k=10)
    
    Complexity:
        - add(): O(1) amortized
        - query(): O(1) bin lookup + O(bin_size) local search
        - With proper bin sizing, effective O(1)
    """
    
    def __init__(
        self,
        manifold,
        dim: int,
        n_bins: int = 1024,
        cache_method: str = 'curvature',
        max_items: Optional[int] = None
    ):
        """
        Args:
            manifold: GeoTorch manifold (Hyperbolic, Sphere, Euclidean)
            dim: Embedding dimension
            n_bins: Number of cache bins
            cache_method: Binning method ('curvature', 'spatial', 'lsh')
            max_items: Maximum items (enables LRU eviction)
        """
        self.manifold = manifold
        self.dim = dim
        self.n_bins = n_bins
        self.max_items = max_items
        
        # The cache map
        self.cache = DavisCache(manifold, n_bins, cache_method, dim=dim)
        
        # Storage: bin_idx -> list of items
        self.bins: Dict[int, List[StorageItem]] = defaultdict(list)
        
        # ID -> item mapping for O(1) access by ID
        self.items: Dict[int, StorageItem] = {}
        
        # Auto-incrementing ID
        self._next_id = 0
        
        # Statistics (like geodesic_storage.py stats)
        self._stats = {
            'writes': 0,
            'reads': 0,
            'read_times': [],
        }
    
    def add(
        self,
        embedding: Tensor,
        metadata: Optional[Dict] = None,
        id: Optional[int] = None
    ) -> int:
        """
        Add embedding to storage.
        
        Args:
            embedding: Point on manifold, shape (D,)
            metadata: Optional metadata dict
            id: Optional ID (auto-generated if None)
        
        Returns:
            Item ID
        """
        if id is None:
            id = self._next_id
            self._next_id += 1
        
        # Compute bin
        bin_idx = self.cache(embedding.unsqueeze(0)).item()
        
        # Create item
        item = StorageItem(
            id=id,
            embedding=embedding.detach().clone(),
            bin_idx=bin_idx,
            metadata=metadata,
            access_time=time.time()
        )
        
        # Store
        self.bins[bin_idx].append(item)
        self.items[id] = item
        self.cache.add_to_bin(id, bin_idx)
        
        self._stats['writes'] += 1
        
        # LRU eviction if needed
        if self.max_items and len(self.items) > self.max_items:
            self._evict_lru()
        
        return id
    
    def add_batch(
        self,
        embeddings: Tensor,
        metadata_list: Optional[List[Dict]] = None
    ) -> List[int]:
        """
        Add batch of embeddings.
        
        Args:
            embeddings: shape (N, D)
            metadata_list: Optional list of metadata dicts
        
        Returns:
            List of IDs
        """
        N = embeddings.shape[0]
        if metadata_list is None:
            metadata_list = [None] * N
        
        ids = []
        for i in range(N):
            id = self.add(embeddings[i], metadata_list[i])
            ids.append(id)
        
        return ids
    
    def query(
        self,
        query: Tensor,
        k: int = 10,
        expand_bins: int = 0
    ) -> List[Tuple[int, float, Optional[Dict]]]:
        """
        Query for similar embeddings. O(1) + O(bin_size).
        
        Adapted from geodesic_storage.py read_related pattern.
        
        Args:
            query: Query point on manifold, shape (D,)
            k: Number of results
            expand_bins: Search neighboring bins too (0 = exact bin only)
        
        Returns:
            List of (id, distance, metadata) tuples, sorted by distance
        """
        start_time = time.time()
        
        # O(1): Get query bin
        query_bin = self.cache(query.unsqueeze(0)).item()
        
        # O(1): Get candidate bins
        if expand_bins > 0:
            candidate_bins = self.cache.neighboring_bins(query_bin, expand_bins)
        else:
            candidate_bins = [query_bin]
        
        # O(1): Collect candidates from bins
        candidates = []
        for bin_idx in candidate_bins:
            candidates.extend(self.bins.get(bin_idx, []))
        
        if not candidates:
            self._stats['reads'] += 1
            self._stats['read_times'].append(time.time() - start_time)
            return []
        
        # O(bin_size): Compute distances
        candidate_embeddings = torch.stack([c.embedding for c in candidates])
        query_expanded = query.unsqueeze(0).expand(len(candidates), -1)
        
        distances = self.manifold.distance(query_expanded, candidate_embeddings)
        
        # O(k log k): Sort and return top-k
        sorted_indices = distances.argsort()[:k]
        
        results = []
        for idx in sorted_indices:
            item = candidates[idx]
            item.access_time = time.time()  # Update for LRU
            results.append((item.id, distances[idx].item(), item.metadata))
        
        self._stats['reads'] += 1
        self._stats['read_times'].append(time.time() - start_time)
        
        return results
    
    def batch_query(
        self,
        queries: Tensor,
        k: int = 10
    ) -> Tuple[Tensor, Tensor]:
        """
        Batch query.
        
        Args:
            queries: shape (B, D)
            k: results per query
        
        Returns:
            indices: shape (B, k) - IDs of nearest items
            distances: shape (B, k) - distances to items
        """
        B = queries.shape[0]
        all_indices = []
        all_distances = []
        
        for i in range(B):
            results = self.query(queries[i], k=k)
            
            # Pad if needed
            ids = [r[0] for r in results]
            dists = [r[1] for r in results]
            
            while len(ids) < k:
                ids.append(-1)
                dists.append(float('inf'))
            
            all_indices.append(ids[:k])
            all_distances.append(dists[:k])
        
        return (
            torch.tensor(all_indices, dtype=torch.long),
            torch.tensor(all_distances)
        )
    
    def get(self, id: int) -> Optional[Tuple[Tensor, Optional[Dict]]]:
        """Get item by ID. O(1)."""
        item = self.items.get(id)
        if item is None:
            return None
        item.access_time = time.time()
        return (item.embedding, item.metadata)
    
    def update(self, id: int, embedding: Tensor):
        """
        Update embedding (may change bin).
        
        Args:
            id: Item ID
            embedding: New embedding
        """
        item = self.items.get(id)
        if item is None:
            raise KeyError(f"Item {id} not found")
        
        old_bin = item.bin_idx
        
        # Remove from old bin
        self.bins[old_bin] = [x for x in self.bins[old_bin] if x.id != id]
        self.cache.remove_from_bin(id)
        
        # Compute new bin
        new_bin = self.cache(embedding.unsqueeze(0)).item()
        
        # Update item
        item.embedding = embedding.detach().clone()
        item.bin_idx = new_bin
        item.access_time = time.time()
        
        # Add to new bin
        self.bins[new_bin].append(item)
        self.cache.add_to_bin(id, new_bin)
    
    def delete(self, id: int) -> bool:
        """Delete item by ID."""
        item = self.items.get(id)
        if item is None:
            return False
        
        # Remove from bin
        self.bins[item.bin_idx] = [x for x in self.bins[item.bin_idx] if x.id != id]
        self.cache.remove_from_bin(id)
        
        # Remove from items dict
        del self.items[id]
        return True
    
    def _evict_lru(self, n_evict: int = None):
        """
        Evict least recently used items.
        
        From geodesic_storage.py DavisCache._evict_lru.
        """
        if n_evict is None:
            n_evict = max(1, len(self.items) // 10)  # 10%
        
        # Sort by access time
        sorted_items = sorted(self.items.values(), key=lambda x: x.access_time)
        
        # Delete oldest
        for item in sorted_items[:n_evict]:
            self.delete(item.id)
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __contains__(self, id: int) -> bool:
        return id in self.items
    
    def stats(self) -> Dict:
        """Storage statistics."""
        bin_sizes = [len(items) for items in self.bins.values()]
        
        read_times = self._stats['read_times']
        import numpy as np
        
        return {
            'total_items': len(self.items),
            'n_bins': self.n_bins,
            'n_nonempty_bins': len(self.bins),
            'avg_bin_size': sum(bin_sizes) / max(len(bin_sizes), 1),
            'max_bin_size': max(bin_sizes) if bin_sizes else 0,
            'min_bin_size': min(bin_sizes) if bin_sizes else 0,
            'total_writes': self._stats['writes'],
            'total_reads': self._stats['reads'],
            'mean_read_time': np.mean(read_times) if read_times else 0,
            'std_read_time': np.std(read_times) if read_times else 0,
            'p50_read_time': np.percentile(read_times, 50) if read_times else 0,
            'p95_read_time': np.percentile(read_times, 95) if read_times else 0,
            'p99_read_time': np.percentile(read_times, 99) if read_times else 0,
        }
    
    def clear(self):
        """Clear all stored items."""
        self.bins.clear()
        self.items.clear()
        self.cache.clear()
        self._next_id = 0
        self._stats = {'writes': 0, 'reads': 0, 'read_times': []}
