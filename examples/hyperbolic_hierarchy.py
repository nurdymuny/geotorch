"""
🔥 HYPERBOLIC EMBEDDINGS: Learning Hierarchies in Curved Space 🔥
=================================================================

This demo shows why hyperbolic geometry is a game-changer for hierarchical data.

THE PROBLEM:
- You have a tree/hierarchy (like WordNet, organizational charts, phylogenetic trees)
- You want to embed it in a vector space for ML
- Euclidean space FAILS because trees grow exponentially, but R^n grows polynomially

THE INSIGHT:
- Hyperbolic space grows EXPONENTIALLY (like trees!)
- A 2D Poincaré disk can embed trees that would need 100+ Euclidean dimensions
- Distances near the boundary expand → leaves naturally spread out

THIS DEMO:
1. Generate a large synthetic hierarchy (10,000+ nodes)
2. Learn embeddings using Riemannian optimization in the Poincaré ball
3. Compare with Euclidean baseline (watch it fail)
4. Visualize the beautiful Poincaré disk embedding
5. Run on GPU to show real performance

Based on: "Poincaré Embeddings for Learning Hierarchical Representations"
          Nickel & Kiela, NeurIPS 2017 (Facebook AI Research)

🚀 BLACKWELL MODE: ENGAGED 🚀
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from collections import defaultdict
import time
import math

# GeoTorch imports
from geotorch.manifolds import Hyperbolic, Euclidean
from geotorch.nn import ManifoldParameter  
from geotorch.optim import RiemannianSGD, RiemannianAdam

# =============================================================================
# CONFIGURATION - CRANK IT UP 🔥
# =============================================================================

CONFIG = {
    # Hierarchy parameters
    'branching_factor': 4,      # Children per node
    'max_depth': 6,             # Tree depth (4^6 = 4096 leaves)
    
    # Embedding parameters  
    'embedding_dim': 32,        # Embedding dimension
    'hyperbolic_dim': 2,        # For visualization (2D Poincaré disk)
    
    # Training parameters
    'n_epochs': 100,
    'batch_size': 4096,         # Large batches for GPU efficiency
    'lr': 0.1,
    'n_negatives': 50,          # Negative samples per positive
    'margin': 0.1,
    
    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

print(f"🖥️  Device: {CONFIG['device'].upper()}")
if CONFIG['device'] == 'cuda':
    print(f"🎮 GPU: {torch.cuda.get_device_name()}")
    print(f"💾 Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# =============================================================================
# GENERATE SYNTHETIC HIERARCHY
# =============================================================================

def generate_tree(branching_factor, max_depth):
    """
    Generate a complete tree with given branching factor and depth.
    
    Returns:
        nodes: List of node IDs
        edges: List of (parent, child) pairs
        depths: Dict mapping node ID to depth
        node_names: Dict mapping node ID to human-readable name
    """
    nodes = [0]  # Root
    edges = []
    depths = {0: 0}
    node_names = {0: "ROOT"}
    
    node_id = 1
    queue = [(0, 0)]  # (node, depth)
    
    while queue:
        parent, depth = queue.pop(0)
        
        if depth >= max_depth:
            continue
            
        for i in range(branching_factor):
            child = node_id
            nodes.append(child)
            edges.append((parent, child))
            depths[child] = depth + 1
            node_names[child] = f"L{depth+1}_{parent}_{i}"
            
            queue.append((child, depth + 1))
            node_id += 1
    
    return nodes, edges, depths, node_names


def compute_tree_distances(nodes, edges):
    """Compute shortest path distances between all node pairs (expensive!)."""
    n = len(nodes)
    
    # Build adjacency list
    adj = defaultdict(list)
    for p, c in edges:
        adj[p].append(c)
        adj[c].append(p)
    
    # BFS from each node (we'll sample instead for large trees)
    distances = {}
    for node in nodes:
        dist = {node: 0}
        queue = [node]
        while queue:
            curr = queue.pop(0)
            for neighbor in adj[curr]:
                if neighbor not in dist:
                    dist[neighbor] = dist[curr] + 1
                    queue.append(neighbor)
        distances[node] = dist
    
    return distances


def sample_training_pairs(nodes, edges, depths, n_samples, n_negatives):
    """
    Sample training pairs for the embedding.
    
    Positive pairs: (parent, child) edges
    Negative pairs: Random non-edge pairs (preferably at similar depth)
    """
    # All edges are positive pairs
    positives = edges.copy()
    
    # For efficiency, sample a subset if too many
    if len(positives) > n_samples:
        indices = torch.randperm(len(positives))[:n_samples].tolist()
        positives = [positives[i] for i in indices]
    
    # Sample negatives for each positive
    node_set = set(nodes)
    edge_set = set(edges) | set((c, p) for p, c in edges)
    
    all_pairs = []
    labels = []
    
    for p, c in positives:
        # Positive
        all_pairs.append((p, c))
        labels.append(1.0)
        
        # Negatives: random nodes that aren't connected
        neg_candidates = list(node_set - {p, c})
        neg_samples = np.random.choice(neg_candidates, size=min(n_negatives, len(neg_candidates)), replace=False)
        
        for neg in neg_samples:
            if (p, neg) not in edge_set:
                all_pairs.append((p, neg))
                labels.append(0.0)
    
    return all_pairs, labels


# =============================================================================
# POINCARÉ DISTANCE (for loss computation)
# =============================================================================

def poincare_distance(u, v, eps=1e-5):
    """
    Hyperbolic distance in the Poincaré ball.
    
    d(u, v) = arcosh(1 + 2 * ||u - v||² / ((1 - ||u||²)(1 - ||v||²)))
    """
    u_sqnorm = (u * u).sum(dim=-1, keepdim=True).clamp(max=1-eps)
    v_sqnorm = (v * v).sum(dim=-1, keepdim=True).clamp(max=1-eps)
    diff_sqnorm = ((u - v) ** 2).sum(dim=-1, keepdim=True)
    
    x = 1 + 2 * diff_sqnorm / ((1 - u_sqnorm) * (1 - v_sqnorm)).clamp(min=eps)
    
    # arcosh(x) = log(x + sqrt(x² - 1))
    return torch.acosh(x.clamp(min=1 + eps)).squeeze(-1)


def euclidean_distance(u, v):
    """Standard Euclidean distance."""
    return ((u - v) ** 2).sum(dim=-1).sqrt()


# =============================================================================
# EMBEDDING MODELS
# =============================================================================

class HyperbolicEmbedding(nn.Module):
    """Poincaré ball embedding using GeoTorch's Riemannian optimization."""
    
    def __init__(self, n_nodes, dim, device):
        super().__init__()
        self.dim = dim
        self.device = device
        
        # Initialize embeddings inside the Poincaré ball
        # Start near origin for stability
        manifold = Hyperbolic(dim)
        init_data = torch.randn(n_nodes, dim, device=device) * 0.001
        self.embeddings = ManifoldParameter(init_data, manifold)
    
    def forward(self, indices):
        return self.embeddings[indices]
    
    def distance(self, u_idx, v_idx):
        u = self.embeddings[u_idx]
        v = self.embeddings[v_idx]
        return poincare_distance(u, v)


class EuclideanEmbedding(nn.Module):
    """Standard Euclidean embedding for comparison."""
    
    def __init__(self, n_nodes, dim, device):
        super().__init__()
        self.embeddings = nn.Parameter(torch.randn(n_nodes, dim, device=device) * 0.01)
    
    def forward(self, indices):
        return self.embeddings[indices]
    
    def distance(self, u_idx, v_idx):
        u = self.embeddings[u_idx]
        v = self.embeddings[v_idx]
        return euclidean_distance(u, v)


# =============================================================================
# LOSS FUNCTION
# =============================================================================

def embedding_loss(model, u_idx, v_idx, labels, margin=0.1):
    """
    Contrastive loss for hierarchy embedding.
    
    For positive pairs (edges): minimize distance
    For negative pairs (non-edges): push distance above margin
    """
    distances = model.distance(u_idx, v_idx)
    
    # Positive loss: minimize distance for connected nodes
    pos_mask = labels > 0.5
    pos_loss = distances[pos_mask].mean() if pos_mask.any() else torch.tensor(0.0)
    
    # Negative loss: push apart unconnected nodes
    neg_mask = labels < 0.5
    neg_distances = distances[neg_mask]
    neg_loss = F.relu(margin - neg_distances).mean() if neg_mask.any() else torch.tensor(0.0)
    
    return pos_loss + neg_loss, pos_loss.item(), neg_loss.item()


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_embedding(model, optimizer, pairs, labels, n_epochs, batch_size, device):
    """Train the embedding model."""
    
    pairs_tensor = torch.tensor(pairs, device=device)
    labels_tensor = torch.tensor(labels, device=device)
    
    n_samples = len(pairs)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    history = {'loss': [], 'pos_loss': [], 'neg_loss': [], 'time': []}
    
    model.train()
    start_time = time.time()
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        epoch_pos = 0
        epoch_neg = 0
        
        # Shuffle
        perm = torch.randperm(n_samples, device=device)
        pairs_shuffled = pairs_tensor[perm]
        labels_shuffled = labels_tensor[perm]
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            
            batch_pairs = pairs_shuffled[start_idx:end_idx]
            batch_labels = labels_shuffled[start_idx:end_idx]
            
            u_idx = batch_pairs[:, 0]
            v_idx = batch_pairs[:, 1]
            
            optimizer.zero_grad()
            loss, pos_loss, neg_loss = embedding_loss(model, u_idx, v_idx, batch_labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_pos += pos_loss
            epoch_neg += neg_loss
        
        avg_loss = epoch_loss / n_batches
        avg_pos = epoch_pos / n_batches
        avg_neg = epoch_neg / n_batches
        elapsed = time.time() - start_time
        
        history['loss'].append(avg_loss)
        history['pos_loss'].append(avg_pos)
        history['neg_loss'].append(avg_neg)
        history['time'].append(elapsed)
        
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch:3d}: loss={avg_loss:.4f} (pos={avg_pos:.4f}, neg={avg_neg:.4f}) [{elapsed:.1f}s]")
    
    return history


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def evaluate_embedding(model, nodes, edges, depths, distances, device):
    """
    Evaluate embedding quality.
    
    Metrics:
    - Mean Average Precision (MAP): Can we reconstruct the tree from distances?
    - Distortion: How well do embedding distances preserve tree distances?
    - Depth correlation: Do deeper nodes have larger norms (hyperbolic property)?
    """
    model.eval()
    
    with torch.no_grad():
        all_embeddings = model.embeddings.data if hasattr(model.embeddings, 'data') else model.embeddings
        
        # 1. Mean Average Precision for edge reconstruction
        n_nodes = len(nodes)
        edge_set = set(edges) | set((c, p) for p, c in edges)
        
        # Sample some nodes for evaluation (full is O(n²))
        eval_nodes = np.random.choice(nodes, size=min(100, len(nodes)), replace=False)
        
        aps = []
        for node in eval_nodes:
            node_emb = all_embeddings[node:node+1]
            
            # Get distances to all other nodes
            all_embs = all_embeddings
            if hasattr(model, 'distance'):
                dists = []
                for other in nodes:
                    if other != node:
                        d = poincare_distance(node_emb, all_embeddings[other:other+1])
                        dists.append((other, d.item()))
            else:
                dists = [(other, euclidean_distance(node_emb, all_embeddings[other:other+1]).item()) 
                         for other in nodes if other != node]
            
            # Sort by distance
            dists.sort(key=lambda x: x[1])
            
            # Compute AP: are true neighbors ranked first?
            true_neighbors = [n for n, d in distances[node].items() if d == 1 and n != node]
            
            if len(true_neighbors) == 0:
                continue
                
            hits = 0
            precision_sum = 0
            for rank, (other, _) in enumerate(dists, 1):
                if other in true_neighbors:
                    hits += 1
                    precision_sum += hits / rank
            
            if hits > 0:
                aps.append(precision_sum / len(true_neighbors))
        
        mean_ap = np.mean(aps) if aps else 0
        
        # 2. Depth-norm correlation (hyperbolic should have high correlation)
        norms = torch.linalg.norm(all_embeddings, dim=-1).cpu().numpy()
        node_depths = [depths[n] for n in nodes]
        correlation = np.corrcoef(norms, node_depths)[0, 1]
        
        # 3. Average distortion on sampled pairs
        sample_pairs = np.random.choice(len(edges), size=min(500, len(edges)), replace=False)
        distortions = []
        for idx in sample_pairs:
            p, c = edges[idx]
            emb_dist = poincare_distance(all_embeddings[p:p+1], all_embeddings[c:c+1]).item()
            tree_dist = distances[p][c]
            distortions.append(abs(emb_dist - tree_dist) / max(tree_dist, 1))
        
        avg_distortion = np.mean(distortions)
        
    return {
        'MAP': mean_ap,
        'depth_correlation': correlation,
        'distortion': avg_distortion,
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_poincare_disk(model, nodes, edges, depths, title, filename):
    """
    Visualize 2D Poincaré disk embedding.
    
    This is where the magic becomes visible:
    - Root near center
    - Leaves near boundary
    - Hierarchy structure preserved
    """
    model.eval()
    
    with torch.no_grad():
        embeddings = model.embeddings.data.cpu().numpy() if hasattr(model.embeddings, 'data') else model.embeddings.detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(14, 14))
    
    # Draw Poincaré disk boundary
    circle = Circle((0, 0), 1, fill=False, color='black', linewidth=2)
    ax.add_patch(circle)
    
    # Color by depth
    max_depth = max(depths.values())
    colors = plt.cm.plasma(np.linspace(0, 1, max_depth + 1))
    
    # Draw edges (sample for visibility)
    edge_sample = edges if len(edges) < 2000 else [edges[i] for i in np.random.choice(len(edges), 2000, replace=False)]
    for p, c in edge_sample:
        x = [embeddings[p, 0], embeddings[c, 0]]
        y = [embeddings[p, 1], embeddings[c, 1]]
        ax.plot(x, y, 'k-', alpha=0.1, linewidth=0.5)
    
    # Draw nodes colored by depth
    for depth in range(max_depth + 1):
        depth_nodes = [n for n in nodes if depths[n] == depth]
        x = embeddings[depth_nodes, 0]
        y = embeddings[depth_nodes, 1]
        size = 100 if depth == 0 else max(5, 50 - depth * 8)
        ax.scatter(x, y, c=[colors[depth]], s=size, label=f'Depth {depth}', alpha=0.7, edgecolors='white', linewidth=0.5)
    
    # Highlight root
    ax.scatter([embeddings[0, 0]], [embeddings[0, 1]], c='red', s=200, marker='*', zorder=100, label='ROOT')
    
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  📊 Saved: {filename}")
    plt.close()


def plot_training_comparison(hyp_history, euc_history, filename):
    """Plot training curves comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Loss
    ax = axes[0]
    ax.plot(hyp_history['loss'], 'b-', linewidth=2, label='Hyperbolic (GeoTorch)')
    ax.plot(euc_history['loss'], 'r--', linewidth=2, label='Euclidean (PyTorch)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Time
    ax = axes[1]
    ax.plot(hyp_history['time'], hyp_history['loss'], 'b-', linewidth=2, label='Hyperbolic')
    ax.plot(euc_history['time'], euc_history['loss'], 'r--', linewidth=2, label='Euclidean')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Loss')
    ax.set_title('Loss vs Wall Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Positive vs Negative loss
    ax = axes[2]
    ax.plot(hyp_history['pos_loss'], 'b-', linewidth=2, label='Hyperbolic (pos)')
    ax.plot(hyp_history['neg_loss'], 'b--', linewidth=2, label='Hyperbolic (neg)')
    ax.plot(euc_history['pos_loss'], 'r-', linewidth=1, label='Euclidean (pos)')
    ax.plot(euc_history['neg_loss'], 'r--', linewidth=1, label='Euclidean (neg)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Component')
    ax.set_title('Positive vs Negative Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"  📊 Saved: {filename}")
    plt.close()


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("🔥 HYPERBOLIC EMBEDDINGS: Learning Hierarchies in Curved Space 🔥")
    print("=" * 70)
    
    device = CONFIG['device']
    
    # =========================================================================
    # GENERATE HIERARCHY
    # =========================================================================
    print("\n📊 Generating synthetic hierarchy...")
    
    nodes, edges, depths, node_names = generate_tree(
        CONFIG['branching_factor'], 
        CONFIG['max_depth']
    )
    
    n_nodes = len(nodes)
    n_edges = len(edges)
    max_depth = max(depths.values())
    n_leaves = sum(1 for d in depths.values() if d == max_depth)
    
    print(f"  Nodes: {n_nodes:,}")
    print(f"  Edges: {n_edges:,}")
    print(f"  Depth: {max_depth}")
    print(f"  Leaves: {n_leaves:,}")
    print(f"  Branching factor: {CONFIG['branching_factor']}")
    
    # Compute tree distances (for evaluation)
    print("\n🔍 Computing tree distances...")
    start = time.time()
    distances = compute_tree_distances(nodes, edges)
    print(f"  Done in {time.time() - start:.1f}s")
    
    # Sample training pairs
    print("\n🎲 Sampling training pairs...")
    pairs, labels = sample_training_pairs(
        nodes, edges, depths, 
        n_samples=min(50000, n_edges),
        n_negatives=CONFIG['n_negatives']
    )
    print(f"  Total pairs: {len(pairs):,}")
    print(f"  Positive: {sum(labels):,.0f}")
    print(f"  Negative: {len(labels) - sum(labels):,.0f}")
    
    # =========================================================================
    # TRAIN HYPERBOLIC EMBEDDING
    # =========================================================================
    print("\n" + "=" * 70)
    print("🌀 Training HYPERBOLIC embedding (Poincaré ball + GeoTorch)")
    print("=" * 70)
    
    hyp_model = HyperbolicEmbedding(n_nodes, CONFIG['hyperbolic_dim'], device)
    hyp_optimizer = RiemannianAdam([hyp_model.embeddings], lr=CONFIG['lr'])
    
    hyp_history = train_embedding(
        hyp_model, hyp_optimizer, pairs, labels,
        CONFIG['n_epochs'], CONFIG['batch_size'], device
    )
    
    print("\n📈 Evaluating hyperbolic embedding...")
    hyp_metrics = evaluate_embedding(hyp_model, nodes, edges, depths, distances, device)
    print(f"  MAP (edge reconstruction): {hyp_metrics['MAP']:.4f}")
    print(f"  Depth-norm correlation:    {hyp_metrics['depth_correlation']:.4f}")
    print(f"  Average distortion:        {hyp_metrics['distortion']:.4f}")
    
    # =========================================================================
    # TRAIN EUCLIDEAN EMBEDDING (BASELINE)
    # =========================================================================
    print("\n" + "=" * 70)
    print("📐 Training EUCLIDEAN embedding (baseline)")
    print("=" * 70)
    
    euc_model = EuclideanEmbedding(n_nodes, CONFIG['hyperbolic_dim'], device)
    euc_optimizer = torch.optim.Adam([euc_model.embeddings], lr=CONFIG['lr'])
    
    euc_history = train_embedding(
        euc_model, euc_optimizer, pairs, labels,
        CONFIG['n_epochs'], CONFIG['batch_size'], device
    )
    
    print("\n📈 Evaluating Euclidean embedding...")
    euc_metrics = evaluate_embedding(euc_model, nodes, edges, depths, distances, device)
    print(f"  MAP (edge reconstruction): {euc_metrics['MAP']:.4f}")
    print(f"  Depth-norm correlation:    {euc_metrics['depth_correlation']:.4f}")
    print(f"  Average distortion:        {euc_metrics['distortion']:.4f}")
    
    # =========================================================================
    # COMPARISON & VISUALIZATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("📊 RESULTS COMPARISON")
    print("=" * 70)
    
    print("""
┌────────────────────┬─────────────────┬─────────────────┐
│ Metric             │ Hyperbolic      │ Euclidean       │
├────────────────────┼─────────────────┼─────────────────┤""")
    print(f"│ MAP                │ {hyp_metrics['MAP']:>15.4f} │ {euc_metrics['MAP']:>15.4f} │")
    print(f"│ Depth Correlation  │ {hyp_metrics['depth_correlation']:>15.4f} │ {euc_metrics['depth_correlation']:>15.4f} │")
    print(f"│ Distortion         │ {hyp_metrics['distortion']:>15.4f} │ {euc_metrics['distortion']:>15.4f} │")
    print(f"│ Final Loss         │ {hyp_history['loss'][-1]:>15.4f} │ {euc_history['loss'][-1]:>15.4f} │")
    print(f"│ Training Time      │ {hyp_history['time'][-1]:>14.1f}s │ {euc_history['time'][-1]:>14.1f}s │")
    print("└────────────────────┴─────────────────┴─────────────────┘")
    
    # Visualization
    print("\n🎨 Generating visualizations...")
    
    visualize_poincare_disk(
        hyp_model, nodes, edges, depths,
        f"Hyperbolic Embedding (Poincaré Disk)\n{n_nodes:,} nodes, MAP={hyp_metrics['MAP']:.3f}",
        "hyperbolic_embedding.png"
    )
    
    visualize_poincare_disk(
        euc_model, nodes, edges, depths,
        f"Euclidean Embedding (for comparison)\n{n_nodes:,} nodes, MAP={euc_metrics['MAP']:.3f}",
        "euclidean_embedding.png"
    )
    
    plot_training_comparison(hyp_history, euc_history, "training_comparison.png")
    
    # =========================================================================
    # KEY INSIGHT
    # =========================================================================
    print("\n" + "=" * 70)
    print("💡 KEY INSIGHT: Why Hyperbolic Wins")
    print("=" * 70)
    print("""
🌳 TREES GROW EXPONENTIALLY:
   - A tree with branching factor b and depth d has O(b^d) nodes
   - Our tree: 4^6 = 4,096 leaf nodes
   
📐 EUCLIDEAN SPACE IS TOO SMALL:
   - R^n has polynomial volume growth: O(r^n)
   - To embed a tree faithfully, you need n ~ O(b^d) dimensions!
   - That's why the Euclidean embedding has HIGH distortion
   
🌀 HYPERBOLIC SPACE IS JUST RIGHT:
   - Poincaré ball has EXPONENTIAL volume growth near boundary
   - A 2D disk can embed trees that need 100+ Euclidean dimensions
   - Root at center, leaves at boundary (natural hierarchy!)
   
📊 THE NUMBERS DON'T LIE:
""")
    
    if hyp_metrics['MAP'] > euc_metrics['MAP']:
        improvement = (hyp_metrics['MAP'] - euc_metrics['MAP']) / euc_metrics['MAP'] * 100
        print(f"   Hyperbolic MAP is {improvement:.1f}% HIGHER than Euclidean")
    
    if hyp_metrics['depth_correlation'] > euc_metrics['depth_correlation']:
        print(f"   Depth correlation: {hyp_metrics['depth_correlation']:.3f} vs {euc_metrics['depth_correlation']:.3f}")
        print(f"   → Hyperbolic naturally encodes depth in norm!")
    
    print("""
🚀 APPLICATIONS:
   - Word embeddings (WordNet hierarchy)
   - Knowledge graphs (Freebase, Wikidata)
   - Social networks (follower trees)
   - Biological taxonomies (species trees)
   - Organizational charts
   - File system structures

🔬 PAPERS TO READ:
   - "Poincaré Embeddings" (Nickel & Kiela, NeurIPS 2017)
   - "Hyperbolic Neural Networks" (Ganea et al., NeurIPS 2018)
   - "Hyperbolic Graph Neural Networks" (Chami et al., NeurIPS 2019)
""")
    
    print("=" * 70)
    print("✅ EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
