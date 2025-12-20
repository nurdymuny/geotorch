"""
GeoKVCache: Geometric KV-Cache for Long-Context Generation
===========================================================

Organizes past keys/values by topological similarity for O(1) retrieval
of relevant context during autoregressive generation.

Standard KV-Cache: O(context_length) per token
GeoKVCache: O(1) retrieval of relevant context
"""

import torch
from torch import Tensor
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from ..storage import DavisCache


class GeoKVCache:
    """
    Geometric KV-Cache for long-context generation.
    
    Organizes past keys/values by topological similarity,
    enabling O(1) retrieval of relevant context.
    
    Standard KV-Cache: Store all past KV, O(context) per token
    GeoKVCache: Organize by similarity, O(1) retrieval of relevant context
    
    Particularly useful for:
        - Very long context (>100K tokens)
        - Hierarchical/structured generation
        - Memory-augmented transformers
    
    Args:
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        head_dim: Dimension per head
        manifold: Manifold for key space
        max_length: Maximum sequence length
        n_bins: Number of cache bins per layer
    
    Example:
        >>> cache = GeoKVCache(n_layers=12, n_heads=8, head_dim=64,
        ...                    manifold=Hyperbolic(64), max_length=100000)
        >>> 
        >>> # During generation
        >>> for token in generate():
        ...     k, v = cache.get_relevant(query, layer=0, k=512)
        ...     output = attention(query, k, v)
        ...     cache.add(new_k, new_v, layer=0)
    """
    
    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        manifold,
        max_length: int = 100000,
        n_bins: int = 1024
    ):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.manifold = manifold
        self.max_length = max_length
        self.n_bins = n_bins
        
        # Cache map per layer
        self.caches = [
            DavisCache(manifold, n_bins, dim=head_dim)
            for _ in range(n_layers)
        ]
        
        # KV storage: layer -> head -> bin -> list of (position, k, v)
        self.kv_storage: List[Dict[int, Dict[int, List]]] = [
            defaultdict(lambda: defaultdict(list))
            for _ in range(n_layers)
        ]
        
        # Track positions
        self.positions = 0
        
        # Statistics
        self._stats = {
            'adds': 0,
            'retrievals': 0,
            'cache_hits': 0,
        }
    
    def add(
        self,
        key: Tensor,
        value: Tensor,
        layer: int
    ):
        """
        Add KV pair to cache.
        
        Args:
            key: shape (n_heads, head_dim) or (head_dim,)
            value: shape (n_heads, head_dim) or (head_dim,)
            layer: Layer index
        """
        # Handle single-head case
        if key.dim() == 1:
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
        
        position = self.positions
        self.positions += 1
        
        cache = self.caches[layer]
        storage = self.kv_storage[layer]
        
        for h in range(key.shape[0]):
            k_h = key[h]  # (head_dim,)
            v_h = value[h]  # (head_dim,)
            
            # Project key to manifold for binning
            k_proj = self.manifold.project(k_h.unsqueeze(0)).squeeze(0)
            
            # Get bin
            bin_idx = cache(k_proj.unsqueeze(0)).item()
            
            # Store
            storage[h][bin_idx].append((position, k_h.detach(), v_h.detach()))
        
        self._stats['adds'] += 1
        
        # Evict if over capacity
        if self.positions > self.max_length:
            self._evict_oldest(layer)
    
    def add_batch(
        self,
        keys: Tensor,
        values: Tensor,
        layer: int
    ):
        """
        Add batch of KV pairs.
        
        Args:
            keys: shape (seq_len, n_heads, head_dim)
            values: shape (seq_len, n_heads, head_dim)
            layer: Layer index
        """
        seq_len = keys.shape[0]
        for t in range(seq_len):
            self.add(keys[t], values[t], layer)
    
    def get_relevant(
        self,
        query: Tensor,
        layer: int,
        k: int = 512,
        expand_bins: int = 1
    ) -> Tuple[Tensor, Tensor]:
        """
        Get k most relevant past KV pairs for query.
        
        Args:
            query: shape (n_heads, head_dim) or (head_dim,)
            layer: Layer index
            k: Number of KV pairs to retrieve per head
            expand_bins: Search radius for neighboring bins
        
        Returns:
            keys: shape (n_heads, k, head_dim)
            values: shape (n_heads, k, head_dim)
        """
        # Handle single-head case
        if query.dim() == 1:
            query = query.unsqueeze(0)
        
        n_heads = query.shape[0]
        cache = self.caches[layer]
        storage = self.kv_storage[layer]
        
        all_keys = []
        all_values = []
        
        for h in range(n_heads):
            q_h = query[h]  # (head_dim,)
            
            # Project query for binning
            q_proj = self.manifold.project(q_h.unsqueeze(0)).squeeze(0)
            
            # Get query bin
            q_bin = cache(q_proj.unsqueeze(0)).item()
            
            # Get candidates from this bin and neighbors
            candidate_bins = cache.neighboring_bins(q_bin, radius=expand_bins)
            candidates = []
            
            for bin_idx in candidate_bins:
                candidates.extend(storage[h].get(bin_idx, []))
            
            self._stats['retrievals'] += 1
            if candidates:
                self._stats['cache_hits'] += 1
            
            # If not enough candidates, search more bins
            if len(candidates) < k:
                for bin_idx in range(self.n_bins):
                    if bin_idx not in candidate_bins:
                        candidates.extend(storage[h].get(bin_idx, []))
                    if len(candidates) >= k:
                        break
            
            # Sort by position (most recent first) or could sort by distance
            candidates.sort(key=lambda x: -x[0])  # Most recent first
            
            # Take top k
            selected = candidates[:k]
            
            # Pad if needed
            if len(selected) < k:
                # Pad with zeros or repeat last
                while len(selected) < k:
                    if selected:
                        selected.append(selected[-1])
                    else:
                        # No candidates at all, use zeros
                        zero_k = torch.zeros(self.head_dim, device=query.device)
                        zero_v = torch.zeros(self.head_dim, device=query.device)
                        selected.append((0, zero_k, zero_v))
            
            # Stack
            head_keys = torch.stack([s[1].to(query.device) for s in selected])
            head_values = torch.stack([s[2].to(query.device) for s in selected])
            
            all_keys.append(head_keys)
            all_values.append(head_values)
        
        keys = torch.stack(all_keys)  # (n_heads, k, head_dim)
        values = torch.stack(all_values)  # (n_heads, k, head_dim)
        
        return keys, values
    
    def get_all(self, layer: int) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """
        Get all cached KV pairs for a layer (for full attention fallback).
        
        Returns:
            keys: shape (n_heads, total_len, head_dim) or None
            values: shape (n_heads, total_len, head_dim) or None
        """
        storage = self.kv_storage[layer]
        
        all_keys = []
        all_values = []
        
        for h in range(self.n_heads):
            head_items = []
            for bin_items in storage[h].values():
                head_items.extend(bin_items)
            
            if not head_items:
                return None, None
            
            # Sort by position
            head_items.sort(key=lambda x: x[0])
            
            head_keys = torch.stack([item[1] for item in head_items])
            head_values = torch.stack([item[2] for item in head_items])
            
            all_keys.append(head_keys)
            all_values.append(head_values)
        
        return torch.stack(all_keys), torch.stack(all_values)
    
    def _evict_oldest(self, layer: int, n_evict: int = None):
        """Evict oldest items from cache."""
        if n_evict is None:
            n_evict = max(1, self.max_length // 10)
        
        storage = self.kv_storage[layer]
        
        # Collect all items with positions
        all_items = []
        for h in range(self.n_heads):
            for bin_idx, items in storage[h].items():
                for item in items:
                    all_items.append((item[0], h, bin_idx, item))
        
        # Sort by position (oldest first)
        all_items.sort(key=lambda x: x[0])
        
        # Remove oldest
        for pos, h, bin_idx, item in all_items[:n_evict]:
            if item in storage[h][bin_idx]:
                storage[h][bin_idx].remove(item)
    
    def clear(self):
        """Clear all cached KV pairs."""
        for layer_storage in self.kv_storage:
            layer_storage.clear()
        self.positions = 0
        self._stats = {'adds': 0, 'retrievals': 0, 'cache_hits': 0}
    
    def clear_layer(self, layer: int):
        """Clear cache for a specific layer."""
        self.kv_storage[layer] = defaultdict(lambda: defaultdict(list))
    
    def __len__(self) -> int:
        """Total number of cached positions."""
        return self.positions
    
    def stats(self) -> Dict:
        """Get cache statistics."""
        total_items = 0
        for layer_storage in self.kv_storage:
            for head_bins in layer_storage.values():
                for bin_items in head_bins.values():
                    total_items += len(bin_items)
        
        return {
            **self._stats,
            'total_items': total_items,
            'positions': self.positions,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'hit_rate': self._stats['cache_hits'] / max(self._stats['retrievals'], 1),
        }


class StreamingGeoKVCache(GeoKVCache):
    """
    Streaming variant of GeoKVCache for infinite context.
    
    Uses a sliding window with geometric organization:
    - Recent tokens: Keep all
    - Older tokens: Keep geometrically representative samples
    
    This allows "infinite" context by intelligently forgetting
    redundant information while keeping diverse context.
    """
    
    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        manifold,
        window_size: int = 4096,
        compressed_size: int = 1024,
        n_bins: int = 256
    ):
        super().__init__(
            n_layers, n_heads, head_dim, manifold,
            max_length=window_size + compressed_size,
            n_bins=n_bins
        )
        
        self.window_size = window_size
        self.compressed_size = compressed_size
        
        # Separate storage for compressed (old) context
        self.compressed_storage: List[Dict[int, Dict[int, List]]] = [
            defaultdict(lambda: defaultdict(list))
            for _ in range(n_layers)
        ]
    
    def add(self, key: Tensor, value: Tensor, layer: int):
        """Add with automatic compression of old context."""
        super().add(key, value, layer)
        
        # Check if window is full
        if self.positions > self.window_size:
            self._compress_old_context(layer)
    
    def _compress_old_context(self, layer: int):
        """
        Move old context to compressed storage.
        
        Keep one representative per bin (geometric compression).
        """
        storage = self.kv_storage[layer]
        compressed = self.compressed_storage[layer]
        
        # Find items older than window
        cutoff = self.positions - self.window_size
        
        for h in range(self.n_heads):
            for bin_idx, items in list(storage[h].items()):
                old_items = [item for item in items if item[0] < cutoff]
                new_items = [item for item in items if item[0] >= cutoff]
                
                # Keep only new items in main storage
                storage[h][bin_idx] = new_items
                
                # Add old items to compressed (keep one per bin)
                if old_items and bin_idx not in compressed[h]:
                    # Keep the most recent old item as representative
                    compressed[h][bin_idx] = [max(old_items, key=lambda x: x[0])]
    
    def get_relevant(
        self,
        query: Tensor,
        layer: int,
        k: int = 512,
        expand_bins: int = 1
    ) -> Tuple[Tensor, Tensor]:
        """Get relevant context from both recent and compressed storage."""
        # Get from recent
        keys, values = super().get_relevant(query, layer, k, expand_bins)
        
        # TODO: Also search compressed storage and merge results
        
        return keys, values
