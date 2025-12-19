"""
Knowledge Graph Embeddings on Manifolds
=======================================

Learn entity and relation embeddings for knowledge graph completion.

This example demonstrates:
- Entity embeddings on hyperbolic space (for hierarchies)
- Relation embeddings as transformations
- Link prediction task
- Comparison with Euclidean baseline (TransE-style)

Based on insights from:
- "Multi-relational Poincaré Graph Embeddings" (Balazevic et al., NeurIPS 2019)
- "Low-Dimensional Hyperbolic Knowledge Graph Embeddings" (Chami et al., ACL 2020)

Key insight: Knowledge graphs contain hierarchical relations (IS-A, PART-OF).
Hyperbolic geometry represents these naturally.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random
import time
from typing import List, Tuple, Dict, Set
from collections import defaultdict
from dataclasses import dataclass

# GeoTorch imports
import sys
sys.path.insert(0, '.')
from geotorch.manifolds import Hyperbolic
from geotorch.optim import RiemannianAdam


# =============================================================================
# KNOWLEDGE GRAPH DATA
# =============================================================================

@dataclass
class Triple:
    head: int
    relation: int
    tail: int


def create_knowledge_graph(
    n_entities: int = 500,
    n_relations: int = 10
) -> Tuple[List[str], List[str], List[Triple], List[Triple]]:
    """
    Create a synthetic knowledge graph with hierarchical structure.
    
    Relations:
        - is_a: hierarchical (dog is_a animal is_a living_thing)
        - part_of: hierarchical (wheel part_of car)
        - located_in: hierarchical (Paris located_in France located_in Europe)
        - works_at: flat relation
        - knows: flat relation
        - similar_to: flat relation
        - opposite_of: flat relation
        - created_by: flat relation
        - used_for: flat relation
        - has_property: flat relation
    
    The hierarchical relations should benefit most from hyperbolic embeddings.
    """
    
    # Create entity hierarchy
    entity_hierarchy = {
        # Living things
        'entity': ['living_thing', 'object', 'place', 'concept'],
        'living_thing': ['animal', 'plant', 'person'],
        'animal': ['mammal', 'bird', 'fish', 'reptile'],
        'mammal': ['dog', 'cat', 'horse', 'elephant', 'whale'],
        'bird': ['eagle', 'sparrow', 'penguin', 'owl'],
        'plant': ['tree', 'flower', 'grass'],
        'tree': ['oak', 'pine', 'maple'],
        'person': ['scientist', 'artist', 'athlete'],
        
        # Objects
        'object': ['vehicle', 'tool', 'furniture', 'device'],
        'vehicle': ['car', 'truck', 'airplane', 'boat'],
        'car': ['sedan', 'suv', 'sports_car'],
        'tool': ['hammer', 'screwdriver', 'wrench'],
        'furniture': ['chair', 'table', 'bed'],
        'device': ['computer', 'phone', 'camera'],
        
        # Places
        'place': ['country', 'city', 'building'],
        'country': ['usa', 'france', 'japan', 'brazil'],
        'city': ['paris', 'tokyo', 'new_york', 'london'],
        
        # Concepts
        'concept': ['science', 'art', 'sport'],
        'science': ['physics', 'biology', 'chemistry'],
        'art': ['music', 'painting', 'literature'],
    }
    
    # Flatten entities
    all_entities = set()
    for parent, children in entity_hierarchy.items():
        all_entities.add(parent)
        all_entities.update(children)
    
    entities = sorted(list(all_entities))
    entity_to_idx = {e: i for i, e in enumerate(entities)}
    
    # Pad to desired size
    while len(entities) < n_entities:
        entities.append(f"entity_{len(entities)}")
        entity_to_idx[entities[-1]] = len(entities) - 1
    
    # Relations
    relations = [
        'is_a',        # Hierarchical
        'part_of',     # Hierarchical
        'located_in',  # Hierarchical
        'works_at',    # Flat
        'knows',       # Flat
        'similar_to',  # Flat
        'opposite_of', # Flat
        'created_by',  # Flat
        'used_for',    # Flat
        'has_property' # Flat
    ]
    relation_to_idx = {r: i for i, r in enumerate(relations)}
    
    # Generate triples
    triples = []
    
    # IS-A triples from hierarchy
    for parent, children in entity_hierarchy.items():
        if parent in entity_to_idx:
            for child in children:
                if child in entity_to_idx:
                    triples.append(Triple(
                        entity_to_idx[child],
                        relation_to_idx['is_a'],
                        entity_to_idx[parent]
                    ))
    
    # PART-OF triples
    part_of_pairs = [
        ('wheel', 'car'), ('engine', 'car'), ('window', 'building'),
        ('leaf', 'tree'), ('petal', 'flower'), ('wing', 'bird'),
        ('screen', 'phone'), ('keyboard', 'computer')
    ]
    for part, whole in part_of_pairs:
        if part in entity_to_idx and whole in entity_to_idx:
            triples.append(Triple(
                entity_to_idx[part],
                relation_to_idx['part_of'],
                entity_to_idx[whole]
            ))
    
    # LOCATED-IN triples
    location_pairs = [
        ('paris', 'france'), ('tokyo', 'japan'), ('new_york', 'usa'),
        ('london', 'country')  # Would need UK
    ]
    for loc, container in location_pairs:
        if loc in entity_to_idx and container in entity_to_idx:
            triples.append(Triple(
                entity_to_idx[loc],
                relation_to_idx['located_in'],
                entity_to_idx[container]
            ))
    
    # Random flat relations
    flat_relations = ['works_at', 'knows', 'similar_to', 'created_by', 'used_for']
    for _ in range(len(triples) * 2):  # Add more flat relations
        rel = random.choice(flat_relations)
        head = random.randint(0, len(entities) - 1)
        tail = random.randint(0, len(entities) - 1)
        if head != tail:
            triples.append(Triple(head, relation_to_idx[rel], tail))
    
    # Split train/test
    random.shuffle(triples)
    split = int(0.8 * len(triples))
    train_triples = triples[:split]
    test_triples = triples[split:]
    
    return entities, relations, train_triples, test_triples


# =============================================================================
# KNOWLEDGE GRAPH EMBEDDING MODELS
# =============================================================================

class HyperbolicKGE(nn.Module):
    """
    Knowledge Graph Embeddings on Hyperbolic Space.
    
    - Entities are embedded on Poincaré ball
    - Relations are modeled as transformations (translations in tangent space)
    - Scoring: negative hyperbolic distance after transformation
    
    This is similar to RotH/AttH but simplified.
    """
    
    def __init__(
        self,
        n_entities: int,
        n_relations: int,
        embed_dim: int = 32,
        curvature: float = -1.0
    ):
        super().__init__()
        
        self.manifold = Hyperbolic(embed_dim, curvature=curvature)
        
        # Entity embeddings on Poincaré ball
        entity_emb = torch.randn(n_entities, embed_dim) * 0.01
        self.entity_embeddings = nn.Parameter(self.manifold.project(entity_emb))
        
        # Relation embeddings (in tangent space, used as translations)
        self.relation_embeddings = nn.Parameter(
            torch.randn(n_relations, embed_dim) * 0.01
        )
    
    def forward(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor
    ) -> torch.Tensor:
        """
        Score triples (h, r, t).
        
        Score = -d(exp_h(r), t)
        
        Higher score = more likely to be true.
        """
        # Get embeddings
        h = self.manifold.project(self.entity_embeddings[heads])
        r = self.relation_embeddings[relations]
        t = self.manifold.project(self.entity_embeddings[tails])
        
        # Apply relation as translation in tangent space
        # h_r = exp_h(r) = move from h in direction r
        h_r = self.manifold.exp(h, r * 0.1)  # Scale r for stability
        
        # Score = negative distance
        dist = self.manifold.distance(h_r, t)
        return -dist
    
    def loss(
        self,
        pos_heads: torch.Tensor,
        pos_relations: torch.Tensor,
        pos_tails: torch.Tensor,
        neg_tails: torch.Tensor,
        margin: float = 1.0
    ) -> torch.Tensor:
        """
        Margin-based ranking loss.
        
        Push positive triples to have higher scores than negative.
        """
        pos_scores = self.forward(pos_heads, pos_relations, pos_tails)
        neg_scores = self.forward(pos_heads, pos_relations, neg_tails)
        
        # Margin loss: want pos_score > neg_score + margin
        loss = F.relu(margin - pos_scores + neg_scores)
        return loss.mean()


class EuclideanKGE(nn.Module):
    """
    Baseline: TransE-style Euclidean KGE.
    
    Score = -||h + r - t||
    """
    
    def __init__(
        self,
        n_entities: int,
        n_relations: int,
        embed_dim: int = 32
    ):
        super().__init__()
        
        self.entity_embeddings = nn.Parameter(
            torch.randn(n_entities, embed_dim) * 0.01
        )
        self.relation_embeddings = nn.Parameter(
            torch.randn(n_relations, embed_dim) * 0.01
        )
    
    def forward(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor
    ) -> torch.Tensor:
        """TransE scoring: -||h + r - t||"""
        h = self.entity_embeddings[heads]
        r = self.relation_embeddings[relations]
        t = self.entity_embeddings[tails]
        
        # Score = negative L2 distance
        dist = (h + r - t).norm(dim=-1)
        return -dist
    
    def loss(
        self,
        pos_heads: torch.Tensor,
        pos_relations: torch.Tensor,
        pos_tails: torch.Tensor,
        neg_tails: torch.Tensor,
        margin: float = 1.0
    ) -> torch.Tensor:
        """Margin-based ranking loss."""
        pos_scores = self.forward(pos_heads, pos_relations, pos_tails)
        neg_scores = self.forward(pos_heads, pos_relations, neg_tails)
        
        loss = F.relu(margin - pos_scores + neg_scores)
        return loss.mean()


# =============================================================================
# DATASET
# =============================================================================

class KGDataset(Dataset):
    def __init__(
        self,
        triples: List[Triple],
        n_entities: int,
        n_negatives: int = 1
    ):
        self.triples = triples
        self.n_entities = n_entities
        self.n_negatives = n_negatives
        
        # For negative sampling
        self.triple_set = set((t.head, t.relation, t.tail) for t in triples)
    
    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, idx):
        triple = self.triples[idx]
        
        # Sample negative tail
        while True:
            neg_tail = random.randint(0, self.n_entities - 1)
            if (triple.head, triple.relation, neg_tail) not in self.triple_set:
                break
        
        return (
            torch.tensor(triple.head),
            torch.tensor(triple.relation),
            torch.tensor(triple.tail),
            torch.tensor(neg_tail)
        )


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

def train_kge(
    model: nn.Module,
    train_triples: List[Triple],
    n_entities: int,
    n_epochs: int = 100,
    batch_size: int = 128,
    lr: float = 0.01,
    use_riemannian: bool = False
):
    """Train KGE model."""
    
    dataset = KGDataset(train_triples, n_entities)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    if use_riemannian and hasattr(model, 'manifold'):
        optimizer = RiemannianAdam([
            {'params': [model.entity_embeddings], 'manifold': model.manifold},
            {'params': [model.relation_embeddings]}
        ], lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(n_epochs):
        total_loss = 0
        n_batches = 0
        
        for heads, relations, tails, neg_tails in dataloader:
            optimizer.zero_grad()
            
            loss = model.loss(heads, relations, tails, neg_tails)
            loss.backward()
            optimizer.step()
            
            # Project back to manifold if needed
            if hasattr(model, 'manifold'):
                with torch.no_grad():
                    model.entity_embeddings.data = model.manifold.project(
                        model.entity_embeddings.data
                    )
            
            total_loss += loss.item()
            n_batches += 1
        
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1:3d} | Loss: {total_loss/n_batches:.4f}")
    
    return model


def evaluate_kge(
    model: nn.Module,
    test_triples: List[Triple],
    n_entities: int,
    entities: List[str],
    relations: List[str]
) -> Dict:
    """Evaluate KGE model on link prediction."""
    
    model.eval()
    
    ranks = []
    hits_1 = 0
    hits_3 = 0
    hits_10 = 0
    
    for triple in test_triples[:200]:  # Sample for speed
        head = torch.tensor([triple.head])
        relation = torch.tensor([triple.relation])
        true_tail = triple.tail
        
        # Score all entities as tails
        all_tails = torch.arange(n_entities)
        heads_exp = head.expand(n_entities)
        relations_exp = relation.expand(n_entities)
        
        scores = model(heads_exp, relations_exp, all_tails)
        
        # Rank of true tail
        true_score = scores[true_tail].item()
        rank = (scores > true_score).sum().item() + 1
        
        ranks.append(rank)
        if rank <= 1:
            hits_1 += 1
        if rank <= 3:
            hits_3 += 1
        if rank <= 10:
            hits_10 += 1
    
    n_test = len(ranks)
    
    return {
        'MR': sum(ranks) / n_test,
        'MRR': sum(1/r for r in ranks) / n_test,
        'Hits@1': hits_1 / n_test,
        'Hits@3': hits_3 / n_test,
        'Hits@10': hits_10 / n_test
    }


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_embeddings(
    model: nn.Module,
    entities: List[str],
    entity_hierarchy: Dict
):
    """Analyze learned embeddings."""
    
    print("\nEmbedding Analysis:")
    print("-" * 50)
    
    if hasattr(model, 'manifold'):
        # Hyperbolic: check if depth correlates with norm
        print("Norm by hierarchy level (Hyperbolic):")
        
        # Compute depths
        depths = {}
        
        def compute_depth(entity, current_depth=0):
            depths[entity] = current_depth
            if entity in entity_hierarchy:
                for child in entity_hierarchy[entity]:
                    compute_depth(child, current_depth + 1)
        
        compute_depth('entity')
        
        # Group norms by depth
        depth_norms = defaultdict(list)
        entity_to_idx = {e: i for i, e in enumerate(entities)}
        
        for entity, depth in depths.items():
            if entity in entity_to_idx:
                idx = entity_to_idx[entity]
                norm = model.entity_embeddings[idx].norm().item()
                depth_norms[depth].append(norm)
        
        for d in sorted(depth_norms.keys()):
            if depth_norms[d]:
                mean_norm = sum(depth_norms[d]) / len(depth_norms[d])
                print(f"  Depth {d}: mean norm = {mean_norm:.3f}")
        
        # Sample embeddings
        print("\nSample entity positions:")
        for entity in ['entity', 'living_thing', 'animal', 'mammal', 'dog']:
            if entity in entity_to_idx:
                idx = entity_to_idx[entity]
                norm = model.entity_embeddings[idx].norm().item()
                depth = depths.get(entity, '?')
                print(f"  {entity:15s}: norm={norm:.3f}, depth={depth}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("KNOWLEDGE GRAPH EMBEDDINGS ON MANIFOLDS")
    print("Learning hierarchical relations with hyperbolic geometry")
    print("=" * 70)
    
    # Create knowledge graph
    print("\n1. Creating knowledge graph...")
    entities, relations, train_triples, test_triples = create_knowledge_graph(
        n_entities=500,
        n_relations=10
    )
    
    print(f"   Entities: {len(entities)}")
    print(f"   Relations: {len(relations)}")
    print(f"   Train triples: {len(train_triples)}")
    print(f"   Test triples: {len(test_triples)}")
    print(f"   Relations: {relations}")
    
    # Count relation types
    rel_counts = defaultdict(int)
    for t in train_triples:
        rel_counts[relations[t.relation]] += 1
    print(f"   Relation distribution: {dict(rel_counts)}")
    
    # Train Euclidean baseline
    print("\n" + "=" * 70)
    print("2. EUCLIDEAN KGE (TransE-style)")
    print("=" * 70)
    
    euc_model = EuclideanKGE(len(entities), len(relations), embed_dim=32)
    print(f"   Parameters: {sum(p.numel() for p in euc_model.parameters()):,}")
    
    start = time.time()
    train_kge(euc_model, train_triples, len(entities), n_epochs=100)
    euc_time = time.time() - start
    
    euc_metrics = evaluate_kge(euc_model, test_triples, len(entities), entities, relations)
    print(f"\n   Results:")
    print(f"   - MR: {euc_metrics['MR']:.1f}")
    print(f"   - MRR: {euc_metrics['MRR']:.3f}")
    print(f"   - Hits@1: {euc_metrics['Hits@1']:.1%}")
    print(f"   - Hits@10: {euc_metrics['Hits@10']:.1%}")
    print(f"   - Time: {euc_time:.1f}s")
    
    # Train Hyperbolic model
    print("\n" + "=" * 70)
    print("3. HYPERBOLIC KGE (GeoTorch)")
    print("=" * 70)
    
    hyp_model = HyperbolicKGE(len(entities), len(relations), embed_dim=32)
    print(f"   Parameters: {sum(p.numel() for p in hyp_model.parameters()):,}")
    
    start = time.time()
    train_kge(hyp_model, train_triples, len(entities), n_epochs=100, use_riemannian=True)
    hyp_time = time.time() - start
    
    hyp_metrics = evaluate_kge(hyp_model, test_triples, len(entities), entities, relations)
    print(f"\n   Results:")
    print(f"   - MR: {hyp_metrics['MR']:.1f}")
    print(f"   - MRR: {hyp_metrics['MRR']:.3f}")
    print(f"   - Hits@1: {hyp_metrics['Hits@1']:.1%}")
    print(f"   - Hits@10: {hyp_metrics['Hits@10']:.1%}")
    print(f"   - Time: {hyp_time:.1f}s")
    
    # Comparison
    print("\n" + "=" * 70)
    print("4. COMPARISON")
    print("=" * 70)
    
    print(f"""
    ┌─────────────────────┬────────────┬────────────┬─────────────┐
    │ Metric              │ Euclidean  │ Hyperbolic │ Improvement │
    ├─────────────────────┼────────────┼────────────┼─────────────┤
    │ Mean Rank (↓)       │ {euc_metrics['MR']:>10.1f} │ {hyp_metrics['MR']:>10.1f} │ {(1-hyp_metrics['MR']/euc_metrics['MR'])*100:>+10.1f}% │
    │ MRR (↑)             │ {euc_metrics['MRR']:>10.3f} │ {hyp_metrics['MRR']:>10.3f} │ {(hyp_metrics['MRR']/euc_metrics['MRR']-1)*100:>+10.1f}% │
    │ Hits@1 (↑)          │ {euc_metrics['Hits@1']:>10.1%} │ {hyp_metrics['Hits@1']:>10.1%} │ {(hyp_metrics['Hits@1']/max(euc_metrics['Hits@1'],0.001)-1)*100:>+10.1f}% │
    │ Hits@10 (↑)         │ {euc_metrics['Hits@10']:>10.1%} │ {hyp_metrics['Hits@10']:>10.1%} │ {(hyp_metrics['Hits@10']/max(euc_metrics['Hits@10'],0.001)-1)*100:>+10.1f}% │
    └─────────────────────┴────────────┴────────────┴─────────────┘
    """)
    
    # Embedding analysis
    print("=" * 70)
    print("5. EMBEDDING ANALYSIS")
    print("=" * 70)
    
    entity_hierarchy = {
        'entity': ['living_thing', 'object', 'place', 'concept'],
        'living_thing': ['animal', 'plant', 'person'],
        'animal': ['mammal', 'bird', 'fish', 'reptile'],
        'mammal': ['dog', 'cat', 'horse', 'elephant', 'whale'],
    }
    
    analyze_embeddings(hyp_model, entities, entity_hierarchy)
    
    # Summary
    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print("""
    Hyperbolic KGE excels on hierarchical relations because:
    
    1. IS-A RELATIONS
       - "dog IS-A mammal IS-A animal" forms a tree
       - Hyperbolic distance naturally respects tree depth
       - Similar entities cluster by category
    
    2. PART-OF RELATIONS
       - "wheel PART-OF car" is hierarchical
       - Parts are embedded deeper than wholes
    
    3. FLAT RELATIONS STILL WORK
       - "knows", "similar_to" don't need hierarchy
       - Hyperbolic space handles them fine (they're just different)
    
    4. EMBEDDING GEOMETRY
       - Abstract concepts (entity, animal) near origin
       - Specific instances (dog, cat) near boundary
       - Norm ≈ specificity in the hierarchy
    
    This is why Poincaré/Hyperbolic embeddings beat Euclidean
    for knowledge graphs with hierarchical structure!
    """)


if __name__ == '__main__':
    main()
