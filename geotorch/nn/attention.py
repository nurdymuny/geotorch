"""
GeoCachedAttention: O(1) Key Retrieval Attention
=================================================

Attention mechanism with topological caching for O(N) complexity
instead of O(N²) standard attention.

Uses DavisCache to bin keys, then only attends to keys in the same
bin as each query.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple

from ..storage import DavisCache, GeoStorage


class GeoCachedAttention(nn.Module):
    """
    Attention with O(1) key retrieval via topological caching.
    
    Instead of O(N²) all-pairs attention, retrieves relevant keys
    from the same cache bin as each query.
    
    Standard attention: O(N²) - compute scores for all query-key pairs
    GeoCachedAttention: O(N) - O(1) retrieval per query
    
    How it works:
        1. Keys are binned using DavisCache
        2. For each query, find its bin
        3. Compute attention only over keys in same/nearby bins
    
    Args:
        embed_dim: Embedding dimension
        n_heads: Number of attention heads
        manifold: Manifold for geometric operations
        n_bins: Number of cache bins
        candidates_per_query: Max candidates to consider per query
        dropout: Dropout probability
    
    Example:
        >>> attn = GeoCachedAttention(256, 8, Hyperbolic(32), n_bins=512)
        >>> x = torch.randn(B, N, 256)  # N can be very large!
        >>> out = attn(x, x, x)  # O(N) instead of O(N²)
    """
    
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        manifold,
        n_bins: int = 512,
        candidates_per_query: int = 64,
        dropout: float = 0.1,
        cache_method: str = 'curvature'
    ):
        super().__init__()
        
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.candidates_per_query = candidates_per_query
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.manifold = manifold
        self.n_bins = n_bins
        self.cache_method = cache_method
        
        # Per-head caches (created lazily)
        self._caches: Optional[list] = None
        
        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def _init_caches(self, device):
        """Initialize per-head caches."""
        if self._caches is None:
            self._caches = [
                DavisCache(self.manifold, self.n_bins, self.cache_method, dim=self.head_dim)
                for _ in range(self.n_heads)
            ]
    
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        use_cache: bool = True,
        attn_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass with optional caching.
        
        Args:
            query: (B, N_q, D)
            key: (B, N_k, D)
            value: (B, N_k, D)
            use_cache: Whether to use geometric caching
            attn_mask: Optional attention mask
        
        Returns:
            output: (B, N_q, D)
        """
        B, N_q, D = query.shape
        _, N_k, _ = key.shape
        
        # Project
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head: (B, N, H, D_h) -> (B, H, N, D_h)
        q = q.view(B, N_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N_k, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N_k, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Project to manifold for geometric operations
        q = self.manifold.project(q)
        k = self.manifold.project(k)
        
        # Decide whether to use caching based on sequence length
        if use_cache and N_k > self.candidates_per_query * 2:
            self._init_caches(query.device)
            out = self._cached_attention(q, k, v, attn_mask)
        else:
            out = self._standard_attention(q, k, v, attn_mask)
        
        # Reshape back: (B, H, N_q, D_h) -> (B, N_q, D)
        out = out.transpose(1, 2).reshape(B, N_q, D)
        out = self.out_proj(out)
        
        return out
    
    def _standard_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Standard O(N²) attention."""
        # (B, H, N_q, D) @ (B, H, D, N_k) -> (B, H, N_q, N_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # (B, H, N_q, N_k) @ (B, H, N_k, D) -> (B, H, N_q, D)
        return torch.matmul(attn, v)
    
    def _cached_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Cached attention: O(N) via topological binning.
        
        For each query, only attend to keys in the same/nearby bins.
        """
        B, H, N_q, D = q.shape
        _, _, N_k, _ = k.shape
        
        outputs = []
        
        for b in range(B):
            batch_outputs = []
            
            for h in range(H):
                cache = self._caches[h]
                
                # Get bin assignments for all keys in this batch/head
                k_h = k[b, h]  # (N_k, D)
                v_h = v[b, h]  # (N_k, D)
                q_h = q[b, h]  # (N_q, D)
                
                k_bins = cache(k_h)  # (N_k,)
                q_bins = cache(q_h)  # (N_q,)
                
                # Build bin -> key indices mapping
                bin_to_keys = {}
                for idx in range(N_k):
                    bin_idx = k_bins[idx].item()
                    if bin_idx not in bin_to_keys:
                        bin_to_keys[bin_idx] = []
                    bin_to_keys[bin_idx].append(idx)
                
                # For each query, attend only to keys in same bin
                head_outputs = []
                
                for qi in range(N_q):
                    q_bin = q_bins[qi].item()
                    
                    # Get candidate key indices (same bin + neighbors)
                    candidate_bins = cache.neighboring_bins(q_bin, radius=1)
                    candidate_indices = []
                    for cb in candidate_bins:
                        candidate_indices.extend(bin_to_keys.get(cb, []))
                    
                    if not candidate_indices:
                        # Fallback: use all keys
                        candidate_indices = list(range(N_k))
                    
                    # Limit candidates
                    if len(candidate_indices) > self.candidates_per_query:
                        candidate_indices = candidate_indices[:self.candidates_per_query]
                    
                    # Get candidate keys and values
                    cand_k = k_h[candidate_indices]  # (n_cand, D)
                    cand_v = v_h[candidate_indices]  # (n_cand, D)
                    
                    # Compute attention for this query
                    q_i = q_h[qi:qi+1]  # (1, D)
                    scores = torch.matmul(q_i, cand_k.T) * self.scale  # (1, n_cand)
                    
                    # Apply mask if provided
                    if attn_mask is not None:
                        # Would need to index mask appropriately
                        pass
                    
                    attn_weights = F.softmax(scores, dim=-1)
                    attn_weights = self.dropout(attn_weights)
                    
                    out_i = torch.matmul(attn_weights, cand_v)  # (1, D)
                    head_outputs.append(out_i)
                
                head_out = torch.cat(head_outputs, dim=0)  # (N_q, D)
                batch_outputs.append(head_out)
            
            batch_out = torch.stack(batch_outputs, dim=0)  # (H, N_q, D)
            outputs.append(batch_out)
        
        return torch.stack(outputs, dim=0)  # (B, H, N_q, D)
    
    def clear_cache(self):
        """Clear internal caches."""
        self._caches = None


class FastGeoCachedAttention(nn.Module):
    """
    Faster GeoCachedAttention using vectorized operations.
    
    Instead of per-query loops, bins queries and processes each bin
    as a batch operation.
    
    More efficient for GPU but same O(N) complexity.
    """
    
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        manifold,
        n_bins: int = 512,
        max_bin_size: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert embed_dim % n_heads == 0
        
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.max_bin_size = max_bin_size
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.manifold = manifold
        self.cache = DavisCache(manifold, n_bins, dim=self.head_dim)
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        use_geo: bool = True
    ) -> Tensor:
        """Forward pass."""
        B, N_q, D = query.shape
        _, N_k, _ = key.shape
        
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        q = q.view(B, N_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N_k, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N_k, self.n_heads, self.head_dim).transpose(1, 2)
        
        if use_geo and N_k > self.max_bin_size:
            out = self._geo_attention(q, k, v)
        else:
            out = self._standard_attention(q, k, v)
        
        out = out.transpose(1, 2).reshape(B, N_q, D)
        return self.out_proj(out)
    
    def _standard_attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Standard attention."""
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, v)
    
    def _geo_attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """
        Geometric attention with bin-level batching.
        
        Process all queries in a bin together for better GPU utilization.
        """
        B, H, N_q, D = q.shape
        _, _, N_k, _ = k.shape
        
        # Project to manifold
        q_proj = self.manifold.project(q)
        k_proj = self.manifold.project(k)
        
        # Flatten for binning: (B*H, N, D)
        q_flat = q_proj.reshape(B * H, N_q, D)
        k_flat = k_proj.reshape(B * H, N_k, D)
        v_flat = v.reshape(B * H, N_k, D)
        
        outputs = []
        
        for bh in range(B * H):
            q_bh = q_flat[bh]  # (N_q, D)
            k_bh = k_flat[bh]  # (N_k, D)
            v_bh = v_flat[bh]  # (N_k, D)
            
            # Compute bins
            q_bins = self.cache(q_bh)  # (N_q,)
            k_bins = self.cache(k_bh)  # (N_k,)
            
            # Get unique query bins
            unique_q_bins = q_bins.unique()
            
            # Output tensor
            out_bh = torch.zeros_like(q_bh)
            
            for qb in unique_q_bins:
                # Queries in this bin
                q_mask = (q_bins == qb)
                q_in_bin = q_bh[q_mask]  # (n_q_bin, D)
                
                # Keys in this bin (or neighboring bins)
                neighbor_bins = self.cache.neighboring_bins(qb.item(), radius=1)
                k_mask = torch.zeros(N_k, dtype=torch.bool, device=k_bh.device)
                for nb in neighbor_bins:
                    k_mask = k_mask | (k_bins == nb)
                
                k_in_bin = k_bh[k_mask]  # (n_k_bin, D)
                v_in_bin = v_bh[k_mask]  # (n_k_bin, D)
                
                if k_in_bin.shape[0] == 0:
                    # Fallback to all keys
                    k_in_bin = k_bh
                    v_in_bin = v_bh
                
                # Limit keys if too many
                if k_in_bin.shape[0] > self.max_bin_size:
                    k_in_bin = k_in_bin[:self.max_bin_size]
                    v_in_bin = v_in_bin[:self.max_bin_size]
                
                # Attention within bin
                scores = torch.matmul(q_in_bin, k_in_bin.T) * self.scale
                attn = F.softmax(scores, dim=-1)
                attn = self.dropout(attn)
                out_bin = torch.matmul(attn, v_in_bin)
                
                out_bh[q_mask] = out_bin
            
            outputs.append(out_bh)
        
        out = torch.stack(outputs, dim=0)  # (B*H, N_q, D)
        return out.view(B, H, N_q, D)
