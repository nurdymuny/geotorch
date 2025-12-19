#!/usr/bin/env python
"""
DavisTensor: Hyperbolic Hierarchy Embeddings
=============================================

Learn hierarchical relationships using hyperbolic geometry with DavisTensor.

This demo shows:
1. Using ManifoldEmbedding for hierarchical data
2. Hyperbolic distance-based loss
3. Training with geometry-aware gradients
4. Visualizing embeddings in the Poincaré disk

Key insight: Trees grow exponentially, and so does hyperbolic space!
A 2D Poincaré disk can embed trees that would need 100+ Euclidean dimensions.

Based on: "Poincaré Embeddings for Learning Hierarchical Representations"
          Nickel & Kiela, NeurIPS 2017
"""

import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parent_dir)

import numpy as np
import davistensor as dt
from davistensor import Hyperbolic, Euclidean
from davistensor.nn import ManifoldEmbedding
from davistensor.core.storage import tensor

# For visualization
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# GENERATE HIERARCHY
# =============================================================================

def create_sample_hierarchy():
    """
    Create a sample WordNet-style hierarchy.
    
    Returns dict mapping parent -> children, list of all words, 
    and list of (parent_idx, child_idx) edges.
    """
    hierarchy = {
        'entity': ['living_thing', 'object', 'abstract'],
        'living_thing': ['animal', 'plant'],
        'animal': ['mammal', 'bird', 'fish'],
        'mammal': ['dog', 'cat', 'elephant'],
        'bird': ['eagle', 'sparrow'],
        'fish': ['salmon', 'shark'],
        'plant': ['tree', 'flower'],
        'tree': ['oak', 'pine'],
        'flower': ['rose', 'tulip'],
        'object': ['vehicle', 'furniture'],
        'vehicle': ['car', 'truck'],
        'furniture': ['chair', 'table'],
        'abstract': ['concept', 'emotion'],
        'concept': ['idea', 'theory'],
        'emotion': ['happiness', 'sadness'],
    }
    
    # Collect all words
    all_words = set()
    for parent, children in hierarchy.items():
        all_words.add(parent)
        all_words.update(children)
    
    # Create word->index mapping
    word2idx = {w: i for i, w in enumerate(sorted(all_words))}
    idx2word = {i: w for w, i in word2idx.items()}
    
    # Create edges as (parent_idx, child_idx)
    edges = []
    for parent, children in hierarchy.items():
        parent_idx = word2idx[parent]
        for child in children:
            child_idx = word2idx[child]
            edges.append((parent_idx, child_idx))
    
    # Compute depths for each word
    depths = {'entity': 0}
    queue = ['entity']
    while queue:
        parent = queue.pop(0)
        if parent in hierarchy:
            for child in hierarchy[parent]:
                depths[child] = depths[parent] + 1
                queue.append(child)
    
    return word2idx, idx2word, edges, depths


# =============================================================================
# TRAINING
# =============================================================================

def train_embeddings(word2idx, edges, dim=2, n_epochs=100, lr=0.5, n_negatives=5):
    """
    Train Poincaré embeddings for the hierarchy.
    
    Loss: margin-based ranking loss with hyperbolic distance
    
    L = Σ max(0, margin + d(parent, child) - d(parent, negative))
    """
    n_words = len(word2idx)
    H = Hyperbolic(dim)
    
    # Create embedding table - points in Poincaré ball
    embeddings = H.random_point(n_words) * 0.01  # Start near origin
    
    margin = 0.5
    
    print(f"\nTraining {n_words} embeddings in {dim}D Poincaré ball")
    print(f"Epochs: {n_epochs}, Learning rate: {lr}")
    print("-" * 50)
    
    for epoch in range(n_epochs):
        total_loss = 0.0
        
        # Shuffle edges
        np.random.shuffle(edges)
        
        for parent_idx, child_idx in edges:
            parent_emb = embeddings[parent_idx]
            child_emb = embeddings[child_idx]
            
            # Positive distance (should be small)
            pos_dist = H.distance(parent_emb, child_emb)
            
            # Sample negative (random non-child)
            neg_losses = []
            for _ in range(n_negatives):
                neg_idx = np.random.randint(n_words)
                while neg_idx == child_idx:
                    neg_idx = np.random.randint(n_words)
                
                neg_emb = embeddings[neg_idx]
                neg_dist = H.distance(parent_emb, neg_emb)
                
                # Margin loss
                loss = max(0, margin + pos_dist - neg_dist)
                neg_losses.append(loss)
            
            batch_loss = np.mean(neg_losses)
            total_loss += batch_loss
            
            if batch_loss > 0:
                # Compute gradients manually (simplified Riemannian gradient)
                # grad_d/dx = -log_x(y) / d(x,y) (normalized direction)
                
                # Move parent toward child
                v_to_child = H.log(parent_emb, child_emb)
                v_norm = np.linalg.norm(v_to_child)
                if v_norm > 1e-6:
                    step = lr * batch_loss * v_to_child / v_norm * 0.1
                    new_parent = H.exp(parent_emb, step)
                    embeddings[parent_idx] = H.project_point(new_parent)
                
                # Move child toward parent
                v_to_parent = H.log(child_emb, parent_emb)
                v_norm = np.linalg.norm(v_to_parent)
                if v_norm > 1e-6:
                    step = lr * batch_loss * v_to_parent / v_norm * 0.1
                    new_child = H.exp(child_emb, step)
                    embeddings[child_idx] = H.project_point(new_child)
        
        avg_loss = total_loss / len(edges)
        
        if epoch % 20 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch:3d}: Loss = {avg_loss:.4f}")
    
    return embeddings


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_embeddings(embeddings, word2idx, idx2word, edges, depths):
    """Evaluate the learned embeddings."""
    H = Hyperbolic(embeddings.shape[1])
    
    print("\n" + "=" * 50)
    print("EVALUATION")
    print("=" * 50)
    
    # Check that parents are closer to root (origin) than children
    root_idx = word2idx['entity']
    root_emb = embeddings[root_idx]
    
    correct_order = 0
    for parent_idx, child_idx in edges:
        parent_dist = H.distance(root_emb, embeddings[parent_idx])
        child_dist = H.distance(root_emb, embeddings[child_idx])
        if parent_dist < child_dist:
            correct_order += 1
    
    print(f"\nHierarchy preservation: {correct_order}/{len(edges)} "
          f"({100*correct_order/len(edges):.1f}%)")
    print("(Parents should be closer to root than children)")
    
    # Show distances by depth
    print("\nAverage distance from root by depth:")
    depth_dists = {}
    for word, depth in depths.items():
        idx = word2idx[word]
        dist = H.distance(root_emb, embeddings[idx])
        if depth not in depth_dists:
            depth_dists[depth] = []
        depth_dists[depth].append(dist)
    
    for depth in sorted(depth_dists.keys()):
        avg_dist = np.mean(depth_dists[depth])
        print(f"  Depth {depth}: {avg_dist:.4f}")
    
    # Show some example distances
    print("\nExample distances:")
    examples = [
        ('dog', 'cat'),      # Siblings
        ('dog', 'mammal'),   # Child-parent
        ('dog', 'entity'),   # Child-root
        ('dog', 'car'),      # Unrelated
    ]
    
    for w1, w2 in examples:
        if w1 in word2idx and w2 in word2idx:
            d = H.distance(embeddings[word2idx[w1]], embeddings[word2idx[w2]])
            print(f"  d({w1}, {w2}) = {d:.4f}")


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_embeddings(embeddings, word2idx, idx2word, depths, edges):
    """Visualize 2D Poincaré disk embeddings."""
    if not HAS_MATPLOTLIB:
        print("\nSkipping visualization (matplotlib not installed)")
        return
    
    if embeddings.shape[1] != 2:
        print("\nSkipping visualization (only 2D supported)")
        return
    
    print("\nGenerating Poincaré disk visualization...")
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Draw unit circle (boundary)
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
    ax.add_patch(circle)
    
    # Color by depth
    max_depth = max(depths.values())
    colors = plt.cm.viridis(np.linspace(0, 1, max_depth + 1))
    
    # Draw edges
    for parent_idx, child_idx in edges:
        p1 = embeddings[parent_idx]
        p2 = embeddings[child_idx]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                'gray', alpha=0.3, linewidth=0.5)
    
    # Draw points
    for word, idx in word2idx.items():
        x, y = embeddings[idx]
        depth = depths.get(word, 0)
        ax.scatter(x, y, c=[colors[depth]], s=100, zorder=5)
        ax.annotate(word, (x, y), fontsize=8, ha='center', va='bottom')
    
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.set_title('Poincaré Disk Embeddings\n(Root near center, leaves near boundary)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('poincare_hierarchy.png', dpi=150)
    print("Saved to poincare_hierarchy.png")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("DAVISTENSOR: Hyperbolic Hierarchy Embeddings")
    print("=" * 60)
    
    # Create hierarchy
    word2idx, idx2word, edges, depths = create_sample_hierarchy()
    print(f"\nHierarchy: {len(word2idx)} words, {len(edges)} edges")
    print(f"Depth range: 0-{max(depths.values())}")
    
    # Train embeddings
    embeddings = train_embeddings(
        word2idx, edges,
        dim=2,  # 2D for visualization
        n_epochs=100,
        lr=0.3,
        n_negatives=10
    )
    
    # Evaluate
    evaluate_embeddings(embeddings, word2idx, idx2word, edges, depths)
    
    # Visualize
    visualize_embeddings(embeddings, word2idx, idx2word, depths, edges)
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
