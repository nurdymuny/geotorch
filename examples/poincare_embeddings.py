"""
Poincaré Word Embeddings
========================

Learn hierarchical word relationships using hyperbolic geometry.

This example demonstrates:
- GeodesicEmbedding for vocabulary
- Hyperbolic distance-based loss
- RiemannianAdam optimizer
- Visualization of learned hierarchy

Based on: "Poincaré Embeddings for Learning Hierarchical Representations"
          (Nickel & Kiela, NeurIPS 2017)

The key insight: words form taxonomies (dog IS-A animal IS-A thing).
Hyperbolic space naturally represents this exponential branching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random
import math
from typing import List, Tuple, Dict, Set
from collections import defaultdict

# GeoTorch imports
import sys
sys.path.insert(0, '.')
from geotorch.manifolds import Hyperbolic
from geotorch.nn import ManifoldParameter
from geotorch.optim import RiemannianAdam


# =============================================================================
# WORDNET-STYLE HIERARCHY
# =============================================================================

def create_wordnet_sample() -> Tuple[List[str], List[Tuple[int, int]], Dict[str, int]]:
    """
    Create a sample WordNet-style hierarchy.
    
    Structure:
        entity
        ├── living_thing
        │   ├── animal
        │   │   ├── mammal
        │   │   │   ├── dog, cat, horse, cow, elephant
        │   │   │   └── whale, dolphin
        │   │   ├── bird
        │   │   │   └── eagle, sparrow, penguin
        │   │   └── fish
        │   │       └── salmon, tuna, shark
        │   └── plant
        │       ├── tree
        │       │   └── oak, pine, maple
        │       └── flower
        │           └── rose, tulip, daisy
        ├── object
        │   ├── vehicle
        │   │   ├── car, truck, bus
        │   │   └── airplane, helicopter
        │   ├── furniture
        │   │   └── chair, table, bed
        │   └── tool
        │       └── hammer, screwdriver, wrench
        └── abstract
            ├── concept
            │   └── idea, theory, belief
            └── emotion
                └── happiness, sadness, anger
    """
    
    hierarchy = {
        'entity': ['living_thing', 'object', 'abstract'],
        'living_thing': ['animal', 'plant'],
        'animal': ['mammal', 'bird', 'fish'],
        'mammal': ['dog', 'cat', 'horse', 'cow', 'elephant', 'whale', 'dolphin'],
        'bird': ['eagle', 'sparrow', 'penguin'],
        'fish': ['salmon', 'tuna', 'shark'],
        'plant': ['tree', 'flower'],
        'tree': ['oak', 'pine', 'maple'],
        'flower': ['rose', 'tulip', 'daisy'],
        'object': ['vehicle', 'furniture', 'tool'],
        'vehicle': ['car', 'truck', 'bus', 'airplane', 'helicopter'],
        'furniture': ['chair', 'table', 'bed'],
        'tool': ['hammer', 'screwdriver', 'wrench'],
        'abstract': ['concept', 'emotion'],
        'concept': ['idea', 'theory', 'belief'],
        'emotion': ['happiness', 'sadness', 'anger'],
    }
    
    # Collect all words
    all_words = set()
    for parent, children in hierarchy.items():
        all_words.add(parent)
        all_words.update(children)
    
    words = sorted(list(all_words))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Create edges (parent, child) pairs
    edges = []
    for parent, children in hierarchy.items():
        for child in children:
            edges.append((word_to_idx[parent], word_to_idx[child]))
    
    return words, edges, word_to_idx


def compute_depths(edges: List[Tuple[int, int]], n_words: int) -> Dict[int, int]:
    """Compute depth of each node in the hierarchy."""
    # Build adjacency
    children = defaultdict(list)
    parents = defaultdict(list)
    for p, c in edges:
        children[p].append(c)
        parents[c].append(p)
    
    # Find root (no parents)
    all_children = set(c for _, c in edges)
    all_parents = set(p for p, _ in edges)
    roots = all_parents - all_children
    
    # BFS to compute depths
    depths = {}
    queue = [(r, 0) for r in roots]
    
    while queue:
        node, depth = queue.pop(0)
        if node in depths:
            continue
        depths[node] = depth
        for child in children[node]:
            queue.append((child, depth + 1))
    
    # Nodes not in hierarchy get depth 0
    for i in range(n_words):
        if i not in depths:
            depths[i] = 0
    
    return depths


# =============================================================================
# DATASET
# =============================================================================

class HierarchyDataset(Dataset):
    """
    Dataset for training Poincaré embeddings.
    
    Samples:
    - Positive: (parent, child) pairs from hierarchy
    - Negative: random pairs (not in hierarchy)
    
    Pre-generates all samples for reproducibility.
    """
    
    def __init__(
        self,
        edges: List[Tuple[int, int]],
        n_words: int,
        n_negatives: int = 10
    ):
        self.edges = edges
        self.n_words = n_words
        self.n_negatives = n_negatives
        
        # Create set for fast lookup
        self.edge_set = set(edges)
        # Also add reverse edges (child, parent) for symmetry
        self.edge_set.update((c, p) for p, c in edges)
        
        # Pre-generate all samples
        self.samples = []
        for parent, child in edges:
            # Positive sample
            self.samples.append(([parent, child], 1.0))
            
            # Negative samples
            for _ in range(n_negatives):
                neg = random.randint(0, n_words - 1)
                # Ensure neg is different from parent and not an edge
                attempts = 0
                while (neg == parent or (parent, neg) in self.edge_set) and attempts < 100:
                    neg = random.randint(0, n_words - 1)
                    attempts += 1
                self.samples.append(([parent, neg], 0.0))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        pairs, label = self.samples[idx]
        return torch.tensor(pairs), torch.tensor(label)


# =============================================================================
# POINCARÉ EMBEDDING MODEL
# =============================================================================

class PoincareEmbedding(nn.Module):
    """
    Poincaré embeddings using GeoTorch.
    
    Each word is embedded on the Poincaré ball (hyperbolic space).
    The loss pushes related words closer (in geodesic distance)
    while pushing unrelated words apart.
    """
    
    def __init__(
        self,
        n_words: int,
        embed_dim: int = 32,
        curvature: float = -1.0
    ):
        super().__init__()
        
        self.n_words = n_words
        self.embed_dim = embed_dim
        self.manifold = Hyperbolic(embed_dim, curvature=curvature)
        
        # Initialize embeddings near origin (will spread out during training)
        embeddings = torch.randn(n_words, embed_dim) * 0.01
        embeddings = self.manifold.project(embeddings)
        self.embeddings = ManifoldParameter(embeddings, self.manifold)
    
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Get embeddings for indices."""
        emb = self.embeddings[indices]
        return self.manifold.project(emb)
    
    def distance(self, idx1: torch.Tensor, idx2: torch.Tensor) -> torch.Tensor:
        """Compute hyperbolic distance between pairs."""
        emb1 = self.forward(idx1)
        emb2 = self.forward(idx2)
        return self.manifold.distance(emb1, emb2)
    
    def loss(
        self,
        pairs: torch.Tensor,
        labels: torch.Tensor,
        margin: float = 0.1
    ) -> torch.Tensor:
        """
        Contrastive loss based on hyperbolic distance.
        
        For positive pairs: minimize distance
        For negative pairs: maximize distance (push beyond margin)
        """
        idx1, idx2 = pairs[:, 0], pairs[:, 1]
        distances = self.distance(idx1, idx2)
        
        # Clamp distances to avoid NaN in gradients
        distances = distances.clamp(min=1e-6)
        
        # Positive loss: distance should be small
        pos_loss = labels * distances
        
        # Negative loss: distance should be large (beyond margin)
        neg_loss = (1 - labels) * F.relu(margin - distances)
        
        return (pos_loss + neg_loss).mean()


# =============================================================================
# TRAINING
# =============================================================================

def train_poincare_embeddings(
    model: PoincareEmbedding,
    dataset: HierarchyDataset,
    n_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 0.01
):
    """Train Poincaré embeddings with Riemannian optimizer."""
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Riemannian optimizer respects manifold geometry
    # ManifoldParameter is automatically detected
    optimizer = RiemannianAdam([model.embeddings], lr=lr)
    
    print("\nTraining Poincaré embeddings...")
    print("-" * 50)
    
    for epoch in range(n_epochs):
        total_loss = 0
        n_batches = 0
        
        for pairs, labels in dataloader:
            optimizer.zero_grad()
            loss = model.loss(pairs, labels)
            
            # Check for NaN and skip batch
            if torch.isnan(loss):
                continue
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / max(n_batches, 1)
            
            # Compute norm statistics (depth proxy)
            norms = model.embeddings.data.norm(dim=-1)
            print(f"  Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | "
                  f"Norm: {norms.mean():.3f} ± {norms.std():.3f}")
    
    return model


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_embeddings(
    model: PoincareEmbedding,
    words: List[str],
    edges: List[Tuple[int, int]],
    word_to_idx: Dict[str, int]
):
    """Evaluate learned embeddings."""
    
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    # 1. Depth correlation
    print("\n1. Depth vs. Distance from Origin")
    print("-" * 40)
    
    depths = compute_depths(edges, len(words))
    norms = model.embeddings.data.norm(dim=-1).detach()
    
    # Group by depth
    depth_norms = defaultdict(list)
    for i, word in enumerate(words):
        depth_norms[depths[i]].append(norms[i].item())
    
    print("  Depth | Mean Norm | Words")
    print("  ------|-----------|-------")
    for d in sorted(depth_norms.keys()):
        mean_norm = sum(depth_norms[d]) / len(depth_norms[d])
        sample_words = [w for w in words if depths[word_to_idx[w]] == d][:3]
        print(f"    {d}   |   {mean_norm:.3f}   | {', '.join(sample_words)}")
    
    # 2. Nearest neighbors
    print("\n2. Nearest Neighbors (Hyperbolic Distance)")
    print("-" * 40)
    
    test_words = ['dog', 'car', 'oak', 'happiness', 'animal']
    
    for word in test_words:
        if word not in word_to_idx:
            continue
        
        idx = word_to_idx[word]
        emb = model.embeddings.data[idx:idx+1]
        
        # Compute distances to all words
        all_embs = model.embeddings.data
        distances = model.manifold.distance(
            emb.expand(len(words), -1),
            all_embs
        )
        
        # Get nearest (excluding self)
        distances[idx] = float('inf')
        nearest_idx = distances.argsort()[:5]
        nearest = [(words[i], distances[i].item()) for i in nearest_idx]
        
        print(f"  {word:12s} → {', '.join(f'{w}({d:.2f})' for w, d in nearest)}")
    
    # 3. Hierarchy preservation
    print("\n3. Hierarchy Preservation (Edge Ranking)")
    print("-" * 40)
    
    ranks = []
    for parent, child in edges:
        parent_emb = model.embeddings.data[parent:parent+1]
        child_emb = model.embeddings.data[child:child+1]
        
        # Distance to actual child
        true_dist = model.manifold.distance(parent_emb, child_emb).item()
        
        # Distances to all words
        all_dists = model.manifold.distance(
            parent_emb.expand(len(words), -1),
            model.embeddings.data
        )
        
        # Rank of true child
        rank = (all_dists < true_dist).sum().item() + 1
        ranks.append(rank)
    
    mean_rank = sum(ranks) / len(ranks)
    mrr = sum(1/r for r in ranks) / len(ranks)
    hits_at_10 = sum(1 for r in ranks if r <= 10) / len(ranks)
    
    print(f"  Mean Rank: {mean_rank:.1f}")
    print(f"  MRR: {mrr:.3f}")
    print(f"  Hits@10: {hits_at_10:.1%}")
    
    # 4. Visualization data
    print("\n4. Embedding Visualization (Poincaré Disk)")
    print("-" * 40)
    
    print("  Sample embeddings (x, y for 2D projection):")
    for word in ['entity', 'animal', 'dog', 'cat', 'car', 'oak']:
        if word not in word_to_idx:
            continue
        idx = word_to_idx[word]
        emb = model.embeddings.data[idx]
        x, y = emb[0].item(), emb[1].item()
        norm = emb.norm().item()
        print(f"    {word:12s}: ({x:+.3f}, {y:+.3f}), norm={norm:.3f}, depth={depths[idx]}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("POINCARÉ WORD EMBEDDINGS")
    print("Learning hierarchical structure with hyperbolic geometry")
    print("=" * 60)
    
    # Create hierarchy
    words, edges, word_to_idx = create_wordnet_sample()
    print(f"\nHierarchy: {len(words)} words, {len(edges)} edges")
    
    # Create dataset
    dataset = HierarchyDataset(edges, len(words), n_negatives=10)
    print(f"Dataset: {len(dataset)} samples (with negatives)")
    
    # Create model
    model = PoincareEmbedding(len(words), embed_dim=32, curvature=-1.0)
    print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train
    model = train_poincare_embeddings(
        model, dataset,
        n_epochs=100,
        batch_size=64,
        lr=0.01
    )
    
    # Evaluate
    evaluate_embeddings(model, words, edges, word_to_idx)
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT")
    print("=" * 60)
    print("""
    Hyperbolic space naturally represents hierarchies because:
    
    1. EXPONENTIAL VOLUME GROWTH
       - Trees have O(b^d) nodes at depth d
       - Hyperbolic space has exponential volume near boundary
       - Perfect match!
    
    2. ROOT AT CENTER, LEAVES AT EDGE
       - Abstract concepts (entity, animal) near origin
       - Concrete instances (dog, cat) near boundary
       - Distance from origin ≈ depth in hierarchy
    
    3. GEODESIC DISTANCE = SEMANTIC DISTANCE
       - dog↔cat: small distance (siblings)
       - dog↔car: large distance (different subtrees)
       - Natural clustering by category
    
    This is what flat Euclidean space CANNOT do with the same dimensions.
    """)


if __name__ == '__main__':
    main()
