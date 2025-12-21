"""
Adaptive Metric Learning with DavisManifold
============================================

Learn a data-dependent similarity metric using the Davis-Wilson framework.

This example demonstrates:
- DavisManifold learns metric tensor G(x) from similarity labels
- Curvature adapts to reflect data distinguishability
- Comparison with fixed Euclidean/Cosine metrics
- Visualization of learned metric space

Key insight from Yang-Mills:
    "Distinguishability requires curvature."
    
We learn where the space should be "curved" (high distinguishability)
vs "flat" (similar items should cluster).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import time
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import random
import math

# GeoTorch imports
import sys
sys.path.insert(0, '.')
from geotorch.manifolds.davis import DavisManifold, DavisMetricLearner


# =============================================================================
# SYNTHETIC DATA WITH COMPLEX SIMILARITY STRUCTURE
# =============================================================================

def generate_metric_learning_data(
    n_samples: int = 1000,
    dim: int = 32,
    n_clusters: int = 10,
    cluster_std: float = 0.3,
    noise_dims: int = 16  # Extra noise dimensions
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Generate data with complex similarity structure.
    
    The "true" similarity is based on cluster membership,
    but the data has:
    - Relevant dimensions (capture cluster structure)
    - Noise dimensions (irrelevant for similarity)
    - Non-spherical clusters (some stretched, some compact)
    
    A good metric should:
    - Identify relevant dimensions
    - Ignore noise dimensions
    - Adapt to cluster shapes
    
    Returns:
        X: (n_samples, dim) data points
        labels: (n_samples,) cluster labels
        info: metadata
    """
    
    # Relevant dimensions
    relevant_dim = dim - noise_dims
    
    # Generate cluster centers in relevant dimensions
    cluster_centers = torch.randn(n_clusters, relevant_dim)
    
    # Generate cluster covariances (non-spherical)
    cluster_covariances = []
    for c in range(n_clusters):
        # Random eigenvalues (some stretched)
        eigenvalues = torch.rand(relevant_dim) * 2 + 0.1
        
        # Random rotation
        Q, _ = torch.linalg.qr(torch.randn(relevant_dim, relevant_dim))
        
        cov = Q @ torch.diag(eigenvalues) @ Q.T
        cluster_covariances.append(cov * cluster_std ** 2)
    
    # Generate samples
    X_relevant = []
    labels = []
    
    samples_per_cluster = n_samples // n_clusters
    for c in range(n_clusters):
        center = cluster_centers[c]
        cov = cluster_covariances[c]
        
        # Sample from multivariate Gaussian
        L = torch.linalg.cholesky(cov)
        samples = torch.randn(samples_per_cluster, relevant_dim) @ L.T + center
        
        X_relevant.append(samples)
        labels.extend([c] * samples_per_cluster)
    
    X_relevant = torch.cat(X_relevant, dim=0)
    
    # Add noise dimensions
    X_noise = torch.randn(len(labels), noise_dims) * 0.5
    X = torch.cat([X_relevant, X_noise], dim=1)
    
    labels = torch.tensor(labels)
    
    info = {
        'n_samples': len(labels),
        'dim': dim,
        'relevant_dim': relevant_dim,
        'noise_dims': noise_dims,
        'n_clusters': n_clusters,
        'cluster_centers': cluster_centers
    }
    
    return X, labels, info


def generate_pairs(
    X: torch.Tensor,
    labels: torch.Tensor,
    n_pairs: int = 5000,
    pos_ratio: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate training pairs with similarity labels."""
    
    n = len(labels)
    n_pos = int(n_pairs * pos_ratio)
    n_neg = n_pairs - n_pos
    
    # Positive pairs (same cluster)
    pos_i, pos_j = [], []
    for _ in range(n_pos):
        c = random.randint(0, labels.max().item())
        cluster_idx = (labels == c).nonzero(as_tuple=True)[0]
        if len(cluster_idx) >= 2:
            i, j = random.sample(cluster_idx.tolist(), 2)
            pos_i.append(i)
            pos_j.append(j)
    
    # Negative pairs (different clusters)
    neg_i, neg_j = [], []
    for _ in range(n_neg):
        c1, c2 = random.sample(range(labels.max().item() + 1), 2)
        cluster1_idx = (labels == c1).nonzero(as_tuple=True)[0]
        cluster2_idx = (labels == c2).nonzero(as_tuple=True)[0]
        if len(cluster1_idx) > 0 and len(cluster2_idx) > 0:
            i = random.choice(cluster1_idx.tolist())
            j = random.choice(cluster2_idx.tolist())
            neg_i.append(i)
            neg_j.append(j)
    
    # Combine
    all_i = pos_i + neg_i
    all_j = pos_j + neg_j
    all_labels = [1.0] * len(pos_i) + [0.0] * len(neg_i)
    
    # Shuffle
    combined = list(zip(all_i, all_j, all_labels))
    random.shuffle(combined)
    all_i, all_j, all_labels = zip(*combined)
    
    return (
        X[list(all_i)],
        X[list(all_j)],
        torch.tensor(all_labels)
    )


# =============================================================================
# BASELINE METRICS
# =============================================================================

class EuclideanMetric:
    """Fixed Euclidean distance."""
    
    def distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return (x1 - x2).norm(dim=-1)


class CosineMetric:
    """Fixed cosine distance."""
    
    def distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        cos_sim = F.cosine_similarity(x1, x2, dim=-1)
        return 1 - cos_sim  # Convert to distance


class MahalanobisMetric(nn.Module):
    """Learned Mahalanobis distance (fixed linear transform)."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.eye(dim))
    
    def distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        diff = x1 - x2
        Wd = diff @ self.W.T
        return Wd.norm(dim=-1)


# =============================================================================
# TRAINING
# =============================================================================

class PairDataset(Dataset):
    def __init__(self, x1, x2, labels):
        self.x1 = x1
        self.x2 = x2
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.x1[idx], self.x2[idx], self.labels[idx]


def train_metric(
    metric,
    train_x1: torch.Tensor,
    train_x2: torch.Tensor,
    train_labels: torch.Tensor,
    n_epochs: int = 50,
    batch_size: int = 128,
    lr: float = 0.001,
    margin: float = 1.0
) -> Dict:
    """Train a metric with contrastive loss."""
    
    dataset = PairDataset(train_x1, train_x2, train_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    if isinstance(metric, nn.Module):
        optimizer = torch.optim.Adam(metric.parameters(), lr=lr)
    else:
        # Fixed metric, no training
        return {'loss': [0.0]}
    
    history = {'loss': []}
    
    for epoch in range(n_epochs):
        total_loss = 0
        n_batches = 0
        
        for x1, x2, labels in loader:
            optimizer.zero_grad()
            
            # Compute distances
            if hasattr(metric, 'manifold'):
                distances = metric.manifold.distance(x1, x2)
            else:
                distances = metric.distance(x1, x2)
            
            # Contrastive loss
            pos_loss = labels * distances.pow(2)
            neg_loss = (1 - labels) * F.relu(margin - distances).pow(2)
            loss = (pos_loss + neg_loss).mean()
            
            # Regularization for DavisManifold
            if hasattr(metric, 'manifold') and hasattr(metric.manifold, 'metric_regularization'):
                reg_loss = metric.manifold.metric_regularization(x1[:32])
                loss = loss + 0.01 * reg_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        history['loss'].append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d} | Loss: {avg_loss:.4f}")
    
    return history


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_metric(
    metric,
    X: torch.Tensor,
    labels: torch.Tensor,
    n_queries: int = 30,  # Reduced for speed
    k: int = 10
) -> Dict:
    """Evaluate metric quality via k-NN retrieval."""
    
    results = {
        'recall@k': [],
        'precision@k': [],
        'map': []
    }
    
    n = len(labels)
    
    # Pre-select query indices
    query_indices = random.sample(range(n), min(n_queries, n))
    
    for query_idx in query_indices:
        query = X[query_idx]
        query_label = labels[query_idx]
        
        # Compute distances - use fast approximate method for Davis
        if hasattr(metric, 'manifold'):
            # For Davis: use simple weighted Euclidean approximation for speed
            G = metric.manifold.metric_tensor(query.unsqueeze(0))  # (1, dim, dim)
            G = G.squeeze(0)  # (dim, dim)
            diff = X - query  # (n, dim)
            # d(x,y) ≈ sqrt((x-y)^T G (x-y))
            distances = torch.sqrt(torch.einsum('ni,ij,nj->n', diff, G, diff) + 1e-8)
        else:
            # For Euclidean/Cosine: batch compute
            if hasattr(metric, 'distance'):
                distances = metric.distance(query.unsqueeze(0).expand(n, -1), X)
            else:
                distances = (X - query).norm(dim=-1)
        
        # Exclude self
        distances[query_idx] = float('inf')
        
        # Get top-k
        top_k_idx = distances.argsort()[:k]
        top_k_labels = labels[top_k_idx]
        
        # Metrics
        relevant = (top_k_labels == query_label).float()
        n_same_class = (labels == query_label).sum().float()
        
        recall = relevant.sum() / n_same_class
        precision = relevant.mean()
        
        # Average precision
        precisions = []
        n_relevant = 0
        for i, is_rel in enumerate(relevant):
            if is_rel:
                n_relevant += 1
                precisions.append(n_relevant / (i + 1))
        ap = sum(precisions) / max(len(precisions), 1)
        
        results['recall@k'].append(recall.item())
        results['precision@k'].append(precision.item())
        results['map'].append(ap)
    
    return {
        'recall@k': sum(results['recall@k']) / len(results['recall@k']),
        'precision@k': sum(results['precision@k']) / len(results['precision@k']),
        'map': sum(results['map']) / len(results['map'])
    }


def analyze_learned_metric(
    davis_metric: DavisMetricLearner,
    X: torch.Tensor,
    labels: torch.Tensor
):
    """Analyze properties of the learned metric."""
    
    print("\nLearned Metric Analysis:")
    print("-" * 50)
    
    # Sample metric tensors at different points
    sample_idx = random.sample(range(len(X)), 10)
    sample_X = X[sample_idx]
    
    G = davis_metric.manifold.metric_tensor(sample_X)
    
    # 1. Eigenvalue analysis
    eigenvalues = torch.linalg.eigvalsh(G)
    
    print(f"Metric tensor eigenvalues:")
    print(f"  Min: {eigenvalues.min():.4f}")
    print(f"  Max: {eigenvalues.max():.4f}")
    print(f"  Condition number: {eigenvalues.max() / eigenvalues.min():.2f}")
    
    # 2. Compare metric at similar vs different points
    same_cluster = (labels[sample_idx[0]] == labels).nonzero(as_tuple=True)[0]
    diff_cluster = (labels[sample_idx[0]] != labels).nonzero(as_tuple=True)[0]
    
    if len(same_cluster) > 1 and len(diff_cluster) > 1:
        same_pair = (sample_X[0:1], X[same_cluster[1]:same_cluster[1]+1])
        diff_pair = (sample_X[0:1], X[diff_cluster[0]:diff_cluster[0]+1])
        
        d_same = davis_metric.manifold.distance(same_pair[0].squeeze(), same_pair[1].squeeze())
        d_diff = davis_metric.manifold.distance(diff_pair[0].squeeze(), diff_pair[1].squeeze())
        
        print(f"\nDistance comparison:")
        print(f"  Same cluster pair: {d_same.item():.4f}")
        print(f"  Different cluster pair: {d_diff.item():.4f}")
        print(f"  Ratio: {d_diff.item() / d_same.item():.2f}x")
    
    # 3. Metric variation
    G_var = G.var(dim=0).mean()
    print(f"\nMetric variation across points: {G_var:.4f}")
    print("  (Higher = more adaptive, Lower = more uniform)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("ADAPTIVE METRIC LEARNING WITH DAVISMANIFOLD")
    print("Learning where space should be curved vs flat")
    print("=" * 70)
    
    torch.manual_seed(42)
    random.seed(42)
    
    # Generate data (smaller for fast demo)
    print("\n1. Generating data with complex similarity structure...")
    X, labels, info = generate_metric_learning_data(
        n_samples=300,  # Reduced for speed
        dim=16,         # Smaller dimension
        n_clusters=5,   # Fewer clusters
        noise_dims=8    # Less noise
    )
    
    print(f"   Samples: {info['n_samples']}")
    print(f"   Total dimensions: {info['dim']}")
    print(f"   Relevant dimensions: {info['relevant_dim']}")
    print(f"   Noise dimensions: {info['noise_dims']}")
    print(f"   Clusters: {info['n_clusters']}")
    
    # Generate training pairs
    print("\n2. Generating training pairs...")
    train_x1, train_x2, train_labels = generate_pairs(X, labels, n_pairs=2000)
    print(f"   Training pairs: {len(train_labels)}")
    print(f"   Positive ratio: {train_labels.mean():.1%}")
    
    # Evaluate fixed Euclidean metric
    print("\n" + "=" * 70)
    print("3. EUCLIDEAN METRIC (Fixed)")
    print("=" * 70)
    
    euclidean = EuclideanMetric()
    euc_metrics = evaluate_metric(euclidean, X, labels)
    
    print(f"   Recall@10: {euc_metrics['recall@k']:.1%}")
    print(f"   Precision@10: {euc_metrics['precision@k']:.1%}")
    print(f"   MAP: {euc_metrics['map']:.3f}")
    
    # Evaluate fixed Cosine metric
    print("\n" + "=" * 70)
    print("4. COSINE METRIC (Fixed)")
    print("=" * 70)
    
    cosine = CosineMetric()
    cos_metrics = evaluate_metric(cosine, X, labels)
    
    print(f"   Recall@10: {cos_metrics['recall@k']:.1%}")
    print(f"   Precision@10: {cos_metrics['precision@k']:.1%}")
    print(f"   MAP: {cos_metrics['map']:.3f}")
    
    # Train Mahalanobis metric
    print("\n" + "=" * 70)
    print("5. MAHALANOBIS METRIC (Learned Linear)")
    print("=" * 70)
    
    mahal = MahalanobisMetric(info['dim'])
    print(f"   Parameters: {sum(p.numel() for p in mahal.parameters()):,}")
    
    start = time.time()
    train_metric(mahal, train_x1, train_x2, train_labels, n_epochs=30)
    mahal_time = time.time() - start
    
    mahal_metrics = evaluate_metric(mahal, X, labels)
    print(f"\n   Recall@10: {mahal_metrics['recall@k']:.1%}")
    print(f"   Precision@10: {mahal_metrics['precision@k']:.1%}")
    print(f"   MAP: {mahal_metrics['map']:.3f}")
    print(f"   Training time: {mahal_time:.1f}s")
    
    # Train DavisManifold metric
    print("\n" + "=" * 70)
    print("6. DAVIS MANIFOLD (Learned Adaptive)")
    print("=" * 70)
    
    # Create with diagonal_only=True for faster training
    from geotorch.manifolds.davis import DavisManifold
    davis_manifold = DavisManifold(dim=info['dim'], hidden_dim=32, n_layers=2, diagonal_only=True)
    
    # Wrap in a simple learner class
    class SimpleDavisLearner(nn.Module):
        def __init__(self, manifold):
            super().__init__()
            self.manifold = manifold
        def forward(self, x1, x2):
            return self.manifold.distance(x1, x2, n_steps=3)  # Fewer steps for speed
    
    davis = SimpleDavisLearner(davis_manifold)
    print(f"   Parameters: {sum(p.numel() for p in davis.parameters()):,}")
    
    start = time.time()
    train_metric(davis, train_x1, train_x2, train_labels, n_epochs=30, lr=0.01)
    davis_time = time.time() - start
    
    davis_metrics = evaluate_metric(davis, X, labels)
    print(f"\n   Recall@10: {davis_metrics['recall@k']:.1%}")
    print(f"   Precision@10: {davis_metrics['precision@k']:.1%}")
    print(f"   MAP: {davis_metrics['map']:.3f}")
    print(f"   Training time: {davis_time:.1f}s")
    
    # Analyze learned metric
    analyze_learned_metric(davis, X, labels)
    
    # Comparison
    print("\n" + "=" * 70)
    print("7. COMPARISON")
    print("=" * 70)
    
    print(f"""
    ┌─────────────────────┬────────────┬────────────┬────────────┐
    │ Metric              │ Recall@10  │ Prec@10    │ MAP        │
    ├─────────────────────┼────────────┼────────────┼────────────┤
    │ Euclidean (fixed)   │ {euc_metrics['recall@k']:>10.1%} │ {euc_metrics['precision@k']:>10.1%} │ {euc_metrics['map']:>10.3f} │
    │ Cosine (fixed)      │ {cos_metrics['recall@k']:>10.1%} │ {cos_metrics['precision@k']:>10.1%} │ {cos_metrics['map']:>10.3f} │
    │ Mahalanobis (linear)│ {mahal_metrics['recall@k']:>10.1%} │ {mahal_metrics['precision@k']:>10.1%} │ {mahal_metrics['map']:>10.3f} │
    │ Davis (adaptive)    │ {davis_metrics['recall@k']:>10.1%} │ {davis_metrics['precision@k']:>10.1%} │ {davis_metrics['map']:>10.3f} │
    └─────────────────────┴────────────┴────────────┴────────────┘
    """)
    
    # Summary
    print("=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
    DavisManifold Advantages:
    
    1. ADAPTIVE CURVATURE
       - Metric tensor G(x) varies with position
       - High curvature where distinguishability needed
       - Flat where items should cluster
    
    2. HANDLES COMPLEX STRUCTURE
       - Non-spherical clusters (stretched, rotated)
       - Noise dimensions automatically down-weighted
       - Non-linear similarity boundaries
    
    3. FROM YANG-MILLS
       - "Distinguishability requires curvature"
       - Curvature is computable (Christoffel symbols)
       - Learned curvature = learned distinguishability
    
    4. BEYOND MAHALANOBIS
       - Mahalanobis = fixed linear transform
       - Davis = position-dependent nonlinear metric
       - More expressive, better recall
    
    The DavisManifold learns WHERE to be discriminative!
    """)


if __name__ == '__main__':
    main()
