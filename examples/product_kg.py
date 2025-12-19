"""
Multi-Aspect Knowledge Graph Embeddings with ProductManifold
=============================================================

Entities have multiple aspects that require different geometries:
- Hierarchical structure (IS-A, PART-OF) → Hyperbolic
- Semantic similarity (SIMILAR-TO) → Sphere
- Numerical attributes (HAS-PROPERTY) → Euclidean

ProductManifold combines these into a unified embedding space.

This example demonstrates:
- Hyperbolic × Sphere × Euclidean product space
- Different relations use different components
- Joint training with multi-relation loss
- Comparison with single-geometry baselines
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
from geotorch.manifolds.product import ProductManifold
from geotorch.manifolds import Hyperbolic, Sphere, Euclidean


# =============================================================================
# MULTI-ASPECT KNOWLEDGE GRAPH
# =============================================================================

def create_multi_aspect_kg(
    n_entities: int = 500
) -> Tuple[List[str], Dict[str, int], List[Tuple[int, str, int]], Dict]:
    """
    Create a knowledge graph with multiple relation types requiring
    different geometries.
    
    Relation types:
    - HIERARCHICAL: is_a, part_of, located_in → needs Hyperbolic
    - SEMANTIC: similar_to, related_to, synonym → needs Sphere
    - ATTRIBUTE: has_age, has_size, has_value → needs Euclidean
    
    Returns:
        entities: List of entity names
        entity_to_idx: Name to index mapping
        triples: List of (head, relation, tail) tuples
        info: Metadata including relation types
    """
    
    # Entity hierarchy (for hierarchical relations)
    hierarchy = {
        'entity': ['living_thing', 'object', 'place', 'concept'],
        'living_thing': ['animal', 'plant', 'person'],
        'animal': ['mammal', 'bird', 'fish', 'insect'],
        'mammal': ['dog', 'cat', 'horse', 'elephant', 'whale', 'lion', 'tiger'],
        'bird': ['eagle', 'sparrow', 'penguin', 'parrot'],
        'fish': ['salmon', 'tuna', 'shark'],
        'plant': ['tree', 'flower', 'grass'],
        'tree': ['oak', 'pine', 'maple', 'palm'],
        'flower': ['rose', 'tulip', 'daisy'],
        'person': ['scientist', 'artist', 'athlete', 'doctor'],
        'object': ['vehicle', 'tool', 'furniture', 'food'],
        'vehicle': ['car', 'truck', 'airplane', 'boat', 'bicycle'],
        'tool': ['hammer', 'screwdriver', 'saw'],
        'furniture': ['chair', 'table', 'bed', 'sofa'],
        'food': ['fruit', 'vegetable', 'meat'],
        'fruit': ['apple', 'banana', 'orange'],
        'place': ['country', 'city', 'building'],
        'country': ['usa', 'france', 'japan', 'brazil', 'india'],
        'city': ['paris', 'tokyo', 'new_york', 'london', 'mumbai'],
        'concept': ['science', 'art', 'sport', 'emotion'],
    }
    
    # Semantic groups (for semantic relations)
    semantic_groups = {
        'pets': ['dog', 'cat', 'parrot'],
        'wild_animals': ['lion', 'tiger', 'elephant', 'eagle'],
        'aquatic': ['whale', 'shark', 'salmon', 'tuna', 'penguin', 'boat'],
        'transportation': ['car', 'truck', 'airplane', 'boat', 'bicycle'],
        'european_cities': ['paris', 'london'],
        'asian_cities': ['tokyo', 'mumbai'],
        'professionals': ['scientist', 'artist', 'athlete', 'doctor'],
        'nature': ['tree', 'flower', 'grass', 'oak', 'pine'],
    }
    
    # Numerical attributes (for Euclidean relations)
    # (entity, attribute_value) - values normalized to [-1, 1]
    attributes = {
        'elephant': {'size': 0.9, 'speed': 0.3, 'lifespan': 0.8},
        'dog': {'size': 0.3, 'speed': 0.6, 'lifespan': 0.3},
        'cat': {'size': 0.2, 'speed': 0.5, 'lifespan': 0.3},
        'whale': {'size': 1.0, 'speed': 0.4, 'lifespan': 0.9},
        'eagle': {'size': 0.2, 'speed': 0.9, 'lifespan': 0.4},
        'sparrow': {'size': 0.05, 'speed': 0.4, 'lifespan': 0.1},
        'car': {'size': 0.4, 'speed': 0.6, 'price': 0.4},
        'airplane': {'size': 0.8, 'speed': 1.0, 'price': 0.9},
        'bicycle': {'size': 0.15, 'speed': 0.2, 'price': 0.1},
    }
    
    # Collect all entities
    all_entities = set()
    for parent, children in hierarchy.items():
        all_entities.add(parent)
        all_entities.update(children)
    for group in semantic_groups.values():
        all_entities.update(group)
    for entity in attributes.keys():
        all_entities.add(entity)
    
    entities = sorted(list(all_entities))
    
    # Pad to desired size
    while len(entities) < n_entities:
        entities.append(f"entity_{len(entities)}")
    
    entity_to_idx = {e: i for i, e in enumerate(entities)}
    
    # Generate triples
    triples = []
    
    # Hierarchical relations
    for parent, children in hierarchy.items():
        if parent in entity_to_idx:
            for child in children:
                if child in entity_to_idx:
                    triples.append((entity_to_idx[child], 'is_a', entity_to_idx[parent]))
    
    # Semantic relations
    for group_name, members in semantic_groups.items():
        for i, m1 in enumerate(members):
            for m2 in members[i+1:]:
                if m1 in entity_to_idx and m2 in entity_to_idx:
                    triples.append((entity_to_idx[m1], 'similar_to', entity_to_idx[m2]))
                    triples.append((entity_to_idx[m2], 'similar_to', entity_to_idx[m1]))
    
    # Attribute relations (encoded as triples with special attribute entities)
    for entity, attrs in attributes.items():
        if entity in entity_to_idx:
            for attr_name, value in attrs.items():
                # Create attribute entity if needed
                attr_entity = f"attr_{attr_name}_{int(value*10)}"
                if attr_entity not in entity_to_idx:
                    entity_to_idx[attr_entity] = len(entity_to_idx)
                    entities.append(attr_entity)
                
                triples.append((entity_to_idx[entity], 'has_property', entity_to_idx[attr_entity]))
    
    # Relation type mapping
    relation_types = {
        'is_a': 'hierarchical',
        'part_of': 'hierarchical',
        'located_in': 'hierarchical',
        'similar_to': 'semantic',
        'related_to': 'semantic',
        'synonym': 'semantic',
        'has_property': 'attribute',
        'has_age': 'attribute',
        'has_size': 'attribute',
    }
    
    info = {
        'n_entities': len(entities),
        'n_relations': len(set(t[1] for t in triples)),
        'n_triples': len(triples),
        'relation_types': relation_types,
        'hierarchy': hierarchy,
        'semantic_groups': semantic_groups,
        'attributes': attributes
    }
    
    return entities, entity_to_idx, triples, info


# =============================================================================
# EMBEDDING MODELS
# =============================================================================

class ProductKGE(nn.Module):
    """
    Knowledge Graph Embeddings on Product Manifold.
    
    Embedding space: Hyperbolic × Sphere × Euclidean
    
    Different relations use different components:
    - Hierarchical → Hyperbolic component
    - Semantic → Sphere component  
    - Attribute → Euclidean component
    """
    
    def __init__(
        self,
        n_entities: int,
        hyp_dim: int = 16,
        sphere_dim: int = 16,
        euc_dim: int = 16,
        curvature: float = -1.0
    ):
        super().__init__()
        
        self.hyp_dim = hyp_dim
        self.sphere_dim = sphere_dim
        self.euc_dim = euc_dim
        self.total_dim = hyp_dim + sphere_dim + euc_dim
        
        # Create component manifolds
        self.hyperbolic = Hyperbolic(hyp_dim, curvature=curvature)
        self.sphere = Sphere(sphere_dim)
        self.euclidean = Euclidean(euc_dim)
        
        # Product manifold
        self.manifold = ProductManifold([
            self.hyperbolic,
            self.sphere,
            self.euclidean
        ])
        
        # Entity embeddings
        embeddings = self.manifold.random_point(n_entities) * 0.01
        embeddings = self.manifold.project(embeddings)
        self.entity_embeddings = nn.Parameter(embeddings)
        
        # Relation-specific transformations
        self.relation_transforms = nn.ModuleDict({
            'hierarchical': nn.Linear(hyp_dim, hyp_dim),
            'semantic': nn.Linear(sphere_dim, sphere_dim),
            'attribute': nn.Linear(euc_dim, euc_dim)
        })
    
    def get_component(self, embeddings: torch.Tensor, component: str) -> torch.Tensor:
        """Extract specific component from product embedding."""
        if component == 'hyperbolic':
            return embeddings[..., :self.hyp_dim]
        elif component == 'sphere':
            return embeddings[..., self.hyp_dim:self.hyp_dim+self.sphere_dim]
        else:  # euclidean
            return embeddings[..., self.hyp_dim+self.sphere_dim:]
    
    def forward(
        self,
        heads: torch.Tensor,
        relations: List[str],
        tails: torch.Tensor
    ) -> torch.Tensor:
        """
        Score triples using component-specific distances.
        """
        # Get embeddings
        head_emb = self.manifold.project(self.entity_embeddings[heads])
        tail_emb = self.manifold.project(self.entity_embeddings[tails])
        
        scores = []
        
        for i, rel in enumerate(relations):
            h = head_emb[i]
            t = tail_emb[i]
            
            # Determine relation type
            if rel in ['is_a', 'part_of', 'located_in']:
                rel_type = 'hierarchical'
                component = 'hyperbolic'
                manifold = self.hyperbolic
            elif rel in ['similar_to', 'related_to', 'synonym']:
                rel_type = 'semantic'
                component = 'sphere'
                manifold = self.sphere
            else:
                rel_type = 'attribute'
                component = 'euclidean'
                manifold = self.euclidean
            
            # Get component embeddings
            h_comp = self.get_component(h, component)
            t_comp = self.get_component(t, component)
            
            # Apply relation-specific transform
            h_transformed = self.relation_transforms[rel_type](h_comp)
            
            # Project back to manifold
            if component == 'hyperbolic':
                h_transformed = manifold.project(h_transformed)
            elif component == 'sphere':
                h_transformed = h_transformed / h_transformed.norm().clamp(min=1e-7)
            
            # Compute distance
            if component == 'euclidean':
                dist = (h_transformed - t_comp).norm()
            else:
                dist = manifold.distance(h_transformed, t_comp)
            
            scores.append(-dist)
        
        return torch.stack(scores)
    
    def loss(
        self,
        pos_heads: torch.Tensor,
        pos_relations: List[str],
        pos_tails: torch.Tensor,
        neg_tails: torch.Tensor,
        margin: float = 1.0
    ) -> torch.Tensor:
        """Margin-based ranking loss."""
        pos_scores = self.forward(pos_heads, pos_relations, pos_tails)
        neg_scores = self.forward(pos_heads, pos_relations, neg_tails)
        
        loss = F.relu(margin - pos_scores + neg_scores)
        return loss.mean()


class SingleManifoldKGE(nn.Module):
    """
    Baseline: Single manifold for all relations.
    """
    
    def __init__(
        self,
        n_entities: int,
        dim: int = 48,
        manifold_type: str = 'euclidean'
    ):
        super().__init__()
        
        self.dim = dim
        self.manifold_type = manifold_type
        
        if manifold_type == 'euclidean':
            self.manifold = Euclidean(dim)
        elif manifold_type == 'hyperbolic':
            self.manifold = Hyperbolic(dim, curvature=-1.0)
        else:
            self.manifold = Sphere(dim)
        
        # Entity embeddings
        embeddings = self.manifold.random_point(n_entities) * 0.01
        embeddings = self.manifold.project(embeddings)
        self.entity_embeddings = nn.Parameter(embeddings)
        
        # Single relation transform
        self.relation_transform = nn.Linear(dim, dim)
    
    def forward(
        self,
        heads: torch.Tensor,
        relations: List[str],
        tails: torch.Tensor
    ) -> torch.Tensor:
        head_emb = self.manifold.project(self.entity_embeddings[heads])
        tail_emb = self.manifold.project(self.entity_embeddings[tails])
        
        # Apply transform
        head_transformed = self.relation_transform(head_emb)
        head_transformed = self.manifold.project(head_transformed)
        
        # Compute distances
        if self.manifold_type == 'euclidean':
            dists = (head_transformed - tail_emb).norm(dim=-1)
        else:
            dists = self.manifold.distance(head_transformed, tail_emb)
        
        return -dists
    
    def loss(
        self,
        pos_heads: torch.Tensor,
        pos_relations: List[str],
        pos_tails: torch.Tensor,
        neg_tails: torch.Tensor,
        margin: float = 1.0
    ) -> torch.Tensor:
        pos_scores = self.forward(pos_heads, pos_relations, pos_tails)
        neg_scores = self.forward(pos_heads, pos_relations, neg_tails)
        
        loss = F.relu(margin - pos_scores + neg_scores)
        return loss.mean()


# =============================================================================
# TRAINING
# =============================================================================

class KGDataset(Dataset):
    def __init__(self, triples: List[Tuple[int, str, int]], n_entities: int):
        self.triples = triples
        self.n_entities = n_entities
        self.triple_set = set((h, r, t) for h, r, t in triples)
    
    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, idx):
        h, r, t = self.triples[idx]
        
        # Negative sampling
        while True:
            neg_t = random.randint(0, self.n_entities - 1)
            if (h, r, neg_t) not in self.triple_set:
                break
        
        return h, r, t, neg_t


def train_kge(
    model: nn.Module,
    triples: List[Tuple[int, str, int]],
    n_entities: int,
    n_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 0.01
) -> Dict:
    """Train KGE model."""
    
    dataset = KGDataset(triples, n_entities)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                        collate_fn=lambda x: list(zip(*x)))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = {'loss': []}
    
    for epoch in range(n_epochs):
        total_loss = 0
        n_batches = 0
        
        for heads, relations, tails, neg_tails in loader:
            heads = torch.tensor(heads)
            tails = torch.tensor(tails)
            neg_tails = torch.tensor(neg_tails)
            relations = list(relations)
            
            optimizer.zero_grad()
            
            loss = model.loss(heads, relations, tails, neg_tails)
            loss.backward()
            optimizer.step()
            
            # Project back to manifold
            with torch.no_grad():
                if hasattr(model, 'manifold'):
                    model.entity_embeddings.data = model.manifold.project(
                        model.entity_embeddings.data
                    )
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        history['loss'].append(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1:3d} | Loss: {avg_loss:.4f}")
    
    return history


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_kge(
    model: nn.Module,
    triples: List[Tuple[int, str, int]],
    entities: List[str],
    n_entities: int
) -> Dict:
    """Evaluate link prediction."""
    
    model.eval()
    
    results_by_type = defaultdict(lambda: {'ranks': [], 'hits1': 0, 'hits10': 0})
    
    # Categorize relations
    hierarchical = ['is_a', 'part_of', 'located_in']
    semantic = ['similar_to', 'related_to', 'synonym']
    
    for h, r, t in triples[:200]:  # Sample for speed
        # Score all entities as tails
        heads = torch.tensor([h] * n_entities)
        relations = [r] * n_entities
        tails = torch.arange(n_entities)
        
        with torch.no_grad():
            scores = model.forward(heads, relations, tails)
        
        # Rank of true tail
        true_score = scores[t].item()
        rank = (scores > true_score).sum().item() + 1
        
        # Categorize
        if r in hierarchical:
            rel_type = 'hierarchical'
        elif r in semantic:
            rel_type = 'semantic'
        else:
            rel_type = 'attribute'
        
        results_by_type[rel_type]['ranks'].append(rank)
        if rank <= 1:
            results_by_type[rel_type]['hits1'] += 1
        if rank <= 10:
            results_by_type[rel_type]['hits10'] += 1
        
        results_by_type['all']['ranks'].append(rank)
        if rank <= 1:
            results_by_type['all']['hits1'] += 1
        if rank <= 10:
            results_by_type['all']['hits10'] += 1
    
    # Compute metrics
    metrics = {}
    for rel_type, data in results_by_type.items():
        n = len(data['ranks'])
        if n > 0:
            metrics[rel_type] = {
                'MR': sum(data['ranks']) / n,
                'MRR': sum(1/r for r in data['ranks']) / n,
                'Hits@1': data['hits1'] / n,
                'Hits@10': data['hits10'] / n
            }
    
    return metrics


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("MULTI-ASPECT KNOWLEDGE GRAPH WITH PRODUCTMANIFOLD")
    print("Different relations need different geometries")
    print("=" * 70)
    
    torch.manual_seed(42)
    random.seed(42)
    
    # Create knowledge graph
    print("\n1. Creating multi-aspect knowledge graph...")
    entities, entity_to_idx, triples, info = create_multi_aspect_kg(n_entities=500)
    
    print(f"   Entities: {info['n_entities']}")
    print(f"   Triples: {info['n_triples']}")
    
    # Count by relation type
    rel_counts = defaultdict(int)
    for _, r, _ in triples:
        rel_counts[r] += 1
    print(f"   Relations: {dict(rel_counts)}")
    
    # Train Euclidean baseline
    print("\n" + "=" * 70)
    print("2. EUCLIDEAN KGE (Single Geometry)")
    print("=" * 70)
    
    euc_model = SingleManifoldKGE(len(entities), dim=48, manifold_type='euclidean')
    print(f"   Parameters: {sum(p.numel() for p in euc_model.parameters()):,}")
    
    start = time.time()
    train_kge(euc_model, triples, len(entities), n_epochs=100)
    euc_time = time.time() - start
    
    euc_metrics = evaluate_kge(euc_model, triples, entities, len(entities))
    print(f"\n   Overall - MR: {euc_metrics['all']['MR']:.1f}, Hits@10: {euc_metrics['all']['Hits@10']:.1%}")
    if 'hierarchical' in euc_metrics:
        print(f"   Hierarchical - MR: {euc_metrics['hierarchical']['MR']:.1f}, Hits@10: {euc_metrics['hierarchical']['Hits@10']:.1%}")
    if 'semantic' in euc_metrics:
        print(f"   Semantic - MR: {euc_metrics['semantic']['MR']:.1f}, Hits@10: {euc_metrics['semantic']['Hits@10']:.1%}")
    
    # Train Hyperbolic baseline
    print("\n" + "=" * 70)
    print("3. HYPERBOLIC KGE (Single Geometry)")
    print("=" * 70)
    
    hyp_model = SingleManifoldKGE(len(entities), dim=48, manifold_type='hyperbolic')
    print(f"   Parameters: {sum(p.numel() for p in hyp_model.parameters()):,}")
    
    start = time.time()
    train_kge(hyp_model, triples, len(entities), n_epochs=100)
    hyp_time = time.time() - start
    
    hyp_metrics = evaluate_kge(hyp_model, triples, entities, len(entities))
    print(f"\n   Overall - MR: {hyp_metrics['all']['MR']:.1f}, Hits@10: {hyp_metrics['all']['Hits@10']:.1%}")
    if 'hierarchical' in hyp_metrics:
        print(f"   Hierarchical - MR: {hyp_metrics['hierarchical']['MR']:.1f}, Hits@10: {hyp_metrics['hierarchical']['Hits@10']:.1%}")
    if 'semantic' in hyp_metrics:
        print(f"   Semantic - MR: {hyp_metrics['semantic']['MR']:.1f}, Hits@10: {hyp_metrics['semantic']['Hits@10']:.1%}")
    
    # Train Product manifold model
    print("\n" + "=" * 70)
    print("4. PRODUCT KGE (Hyperbolic × Sphere × Euclidean)")
    print("=" * 70)
    
    prod_model = ProductKGE(len(entities), hyp_dim=16, sphere_dim=16, euc_dim=16)
    print(f"   Parameters: {sum(p.numel() for p in prod_model.parameters()):,}")
    print(f"   Total dimension: {prod_model.total_dim}")
    
    start = time.time()
    train_kge(prod_model, triples, len(entities), n_epochs=100)
    prod_time = time.time() - start
    
    prod_metrics = evaluate_kge(prod_model, triples, entities, len(entities))
    print(f"\n   Overall - MR: {prod_metrics['all']['MR']:.1f}, Hits@10: {prod_metrics['all']['Hits@10']:.1%}")
    if 'hierarchical' in prod_metrics:
        print(f"   Hierarchical - MR: {prod_metrics['hierarchical']['MR']:.1f}, Hits@10: {prod_metrics['hierarchical']['Hits@10']:.1%}")
    if 'semantic' in prod_metrics:
        print(f"   Semantic - MR: {prod_metrics['semantic']['MR']:.1f}, Hits@10: {prod_metrics['semantic']['Hits@10']:.1%}")
    
    # Comparison
    print("\n" + "=" * 70)
    print("5. COMPARISON BY RELATION TYPE")
    print("=" * 70)
    
    print(f"""
    +---------------------+--------------------------------------------+
    |                     |         Hits@10 by Relation Type           |
    | Model               +------------+------------+-----------------+
    |                     | Hierarchical| Semantic   | Overall        |
    +---------------------+------------+------------+-----------------+
    | Euclidean           | {euc_metrics.get('hierarchical', {}).get('Hits@10', 0):>10.1%} | {euc_metrics.get('semantic', {}).get('Hits@10', 0):>10.1%} | {euc_metrics['all']['Hits@10']:>13.1%} |
    | Hyperbolic          | {hyp_metrics.get('hierarchical', {}).get('Hits@10', 0):>10.1%} | {hyp_metrics.get('semantic', {}).get('Hits@10', 0):>10.1%} | {hyp_metrics['all']['Hits@10']:>13.1%} |
    | Product (HxSxE)     | {prod_metrics.get('hierarchical', {}).get('Hits@10', 0):>10.1%} | {prod_metrics.get('semantic', {}).get('Hits@10', 0):>10.1%} | {prod_metrics['all']['Hits@10']:>13.1%} |
    +---------------------+------------+------------+-----------------+
    """)
    
    # Embedding analysis
    print("=" * 70)
    print("6. EMBEDDING ANALYSIS")
    print("=" * 70)
    
    with torch.no_grad():
        emb = prod_model.entity_embeddings.data
        hyp_part = prod_model.get_component(emb, 'hyperbolic')
        sphere_part = prod_model.get_component(emb, 'sphere')
        euc_part = prod_model.get_component(emb, 'euclidean')
        
        print("\nComponent norms by entity type:")
        
        # Hierarchical entities (check hyperbolic norms)
        root_idx = entity_to_idx.get('entity', 0)
        leaf_indices = [entity_to_idx.get(e, 0) for e in ['dog', 'cat', 'oak', 'car']]
        
        root_hyp_norm = hyp_part[root_idx].norm().item()
        leaf_hyp_norms = [hyp_part[i].norm().item() for i in leaf_indices]
        
        print(f"  Hyperbolic norm (root 'entity'): {root_hyp_norm:.4f}")
        print(f"  Hyperbolic norm (leaves): {sum(leaf_hyp_norms)/len(leaf_hyp_norms):.4f}")
        print("  → Leaves should have higher norm (deeper in hierarchy)")
    
    # Summary
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
    ProductManifold Advantages for Multi-Aspect KGs:
    
    1. RIGHT GEOMETRY FOR EACH RELATION
       - IS-A, PART-OF → Hyperbolic (tree structure)
       - SIMILAR-TO → Sphere (direction captures similarity)
       - HAS-PROPERTY → Euclidean (numerical attributes)
    
    2. PARAMETER EFFICIENCY
       - Product of 16+16+16=48 dims
       - Each component is specialized
       - Better than 48-dim single geometry
    
    3. INTERPRETABLE COMPONENTS
       - Hyperbolic: hierarchy depth = norm
       - Sphere: direction = semantic category
       - Euclidean: continuous attribute values
    
    4. MIXED-CURVATURE ADVANTAGE
       - Gu et al. (ICLR 2019): "Learning Mixed-Curvature Representations"
       - Real-world KGs have multiple structure types
       - Product space captures all of them
    
    GeoTorch makes this easy:
        manifold = ProductManifold([
            Hyperbolic(16),
            Sphere(16),
            Euclidean(16)
        ])
    """)


if __name__ == '__main__':
    main()
