"""
Brain Connectivity Classification with SPD Manifold
====================================================

Classify brain states (healthy vs. diseased) from fMRI connectivity matrices.

This example demonstrates:
- SPD matrices as natural representation for correlation/covariance
- Riemannian operations preserve positive-definiteness
- Fréchet mean for prototype-based classification
- SPD neural network layers

Real-world applications:
- Alzheimer's disease detection
- Autism spectrum disorder classification
- Mental state decoding (rest, task, sleep)
- Brain-computer interfaces
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import time
from typing import List, Tuple, Dict
from collections import defaultdict
import random
import math

# GeoTorch imports
import sys
sys.path.insert(0, '.')
from geotorch.manifolds.spd import SPD, SPDTransform, SPDBiMap, SPDReLU, SPDLogEig


# =============================================================================
# SYNTHETIC BRAIN DATA
# =============================================================================

def generate_brain_connectivity_data(
    n_subjects: int = 200,
    n_regions: int = 20,  # Brain regions (ROIs)
    n_timepoints: int = 100,  # fMRI timepoints per subject
    disease_effect: float = 0.3
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Generate synthetic fMRI connectivity matrices.
    
    Healthy brains have:
    - Strong within-network connectivity
    - Weak between-network connectivity
    
    Diseased brains have:
    - Disrupted network structure
    - Altered connectivity patterns
    
    Returns:
        connectivity: (n_subjects, n_regions, n_regions) SPD matrices
        labels: (n_subjects,) 0=healthy, 1=diseased
        info: metadata
    """
    
    # Define brain networks (groups of regions)
    # e.g., Default Mode, Visual, Motor, Attention
    n_networks = 4
    regions_per_network = n_regions // n_networks
    
    # Network membership
    network_assignments = []
    for net in range(n_networks):
        network_assignments.extend([net] * regions_per_network)
    
    # Pad if needed
    while len(network_assignments) < n_regions:
        network_assignments.append(n_networks - 1)
    
    connectivity_matrices = []
    labels = []
    
    for subject in range(n_subjects):
        is_diseased = subject >= n_subjects // 2
        labels.append(1 if is_diseased else 0)
        
        # Generate fMRI time series
        # Each region has a time series, correlated within networks
        timeseries = torch.zeros(n_regions, n_timepoints)
        
        # Generate network-level signals
        network_signals = torch.randn(n_networks, n_timepoints)
        
        for region in range(n_regions):
            network = network_assignments[region]
            
            # Signal = network signal + individual noise
            network_weight = 0.7 if not is_diseased else 0.7 - disease_effect * random.random()
            noise_weight = 1 - network_weight
            
            timeseries[region] = (
                network_weight * network_signals[network] +
                noise_weight * torch.randn(n_timepoints)
            )
            
            # Add disease-specific disruption
            if is_diseased:
                # Cross-network leakage
                other_network = (network + 1) % n_networks
                timeseries[region] += disease_effect * 0.5 * network_signals[other_network]
        
        # Compute correlation matrix
        timeseries = timeseries - timeseries.mean(dim=1, keepdim=True)
        timeseries = timeseries / (timeseries.std(dim=1, keepdim=True) + 1e-6)
        
        corr = timeseries @ timeseries.T / n_timepoints
        
        # Regularize to ensure SPD
        corr = corr + 0.1 * torch.eye(n_regions)
        
        connectivity_matrices.append(corr)
    
    connectivity = torch.stack(connectivity_matrices)
    labels = torch.tensor(labels)
    
    info = {
        'n_subjects': n_subjects,
        'n_regions': n_regions,
        'n_networks': n_networks,
        'network_assignments': network_assignments
    }
    
    return connectivity, labels, info


# =============================================================================
# MODELS
# =============================================================================

class EuclideanClassifier(nn.Module):
    """
    Baseline: Treat connectivity matrix as flat vector.
    
    Ignores SPD structure entirely.
    """
    
    def __init__(self, n_regions: int, hidden_dim: int = 64):
        super().__init__()
        
        # Flatten upper triangular (avoid redundancy)
        n_features = n_regions * (n_regions + 1) // 2
        
        self.classifier = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 2)
        )
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: (B, n_regions, n_regions) connectivity matrices
        """
        B, n, _ = X.shape
        
        # Extract upper triangular as features
        indices = torch.triu_indices(n, n)
        features = X[:, indices[0], indices[1]]
        
        return self.classifier(features)


class SPDClassifier(nn.Module):
    """
    Riemannian classifier respecting SPD geometry.
    
    Architecture:
    1. SPD BiMap layers (learnable congruence transforms)
    2. SPD ReLU (eigenvalue thresholding)
    3. Log-eigenvalue layer (map to tangent space)
    4. Linear classifier
    """
    
    def __init__(self, n_regions: int, hidden_dim: int = 10):
        super().__init__()
        
        self.spd = SPD(n_regions)
        
        # SPD layers
        self.bimap1 = SPDBiMap(n_regions, hidden_dim)
        self.relu1 = SPDReLU(threshold=1e-4)
        
        self.bimap2 = SPDBiMap(hidden_dim, hidden_dim // 2)
        self.relu2 = SPDReLU(threshold=1e-4)
        
        # Map to tangent space
        self.log_eig = SPDLogEig()
        
        # Classifier on flattened log-eigenvalue features
        n_features = (hidden_dim // 2) * ((hidden_dim // 2) + 1) // 2
        self.classifier = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: (B, n_regions, n_regions) SPD connectivity matrices
        """
        # Ensure SPD
        X = self.spd.project(X)
        
        # SPD transformations
        h = self.bimap1(X)
        h = self.relu1(h)
        
        h = self.bimap2(h)
        h = self.relu2(h)
        
        # Map to tangent space at identity
        h_log = self.log_eig(h)
        
        # Flatten and classify
        B, n, _ = h_log.shape
        indices = torch.triu_indices(n, n)
        features = h_log[:, indices[0], indices[1]]
        
        return self.classifier(features)


class FrechetMeanClassifier(nn.Module):
    """
    Prototype-based classifier using Fréchet means.
    
    Compute class prototypes as Riemannian means,
    classify by nearest prototype (geodesic distance).
    """
    
    def __init__(self, n_regions: int, n_classes: int = 2):
        super().__init__()
        
        self.spd = SPD(n_regions)
        self.n_classes = n_classes
        self.n_regions = n_regions
        
        # Learnable prototypes (initialized later)
        self.prototypes = nn.ParameterList([
            nn.Parameter(torch.eye(n_regions))
            for _ in range(n_classes)
        ])
    
    def compute_prototypes(self, X: torch.Tensor, y: torch.Tensor):
        """Compute class prototypes as Fréchet means."""
        with torch.no_grad():
            for c in range(self.n_classes):
                mask = (y == c)
                if mask.sum() > 0:
                    class_matrices = X[mask]
                    # Simple initialization: arithmetic mean projected
                    mean = self.spd.project(class_matrices.mean(dim=0))
                    self.prototypes[c].data = mean
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Classify by geodesic distance to prototypes.
        
        Returns negative distances as logits (smaller distance = higher score).
        """
        B = X.shape[0]
        
        # Ensure SPD
        X = self.spd.project(X)
        
        # Compute distance to each prototype
        distances = []
        for c in range(self.n_classes):
            prototype = self.spd.project(self.prototypes[c])
            # Compute distance for each sample
            d = torch.stack([
                self.spd.distance(X[i], prototype)
                for i in range(B)
            ])
            distances.append(d)
        
        distances = torch.stack(distances, dim=1)  # (B, n_classes)
        
        # Return negative distance as logits
        return -distances


# =============================================================================
# TRAINING
# =============================================================================

class ConnectivityDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_model(
    model: nn.Module,
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    val_X: torch.Tensor,
    val_y: torch.Tensor,
    n_epochs: int = 50,
    batch_size: int = 16,
    lr: float = 0.001
) -> Dict:
    """Train classification model."""
    
    dataset = ConnectivityDataset(train_X, train_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == y_batch).sum().item()
            total += y_batch.shape[0]
        
        train_acc = correct / total
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(val_X)
            val_pred = val_logits.argmax(dim=1)
            val_acc = (val_pred == val_y).float().mean().item()
        
        history['train_loss'].append(total_loss / len(loader))
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d} | Loss: {total_loss/len(loader):.4f} | "
                  f"Train Acc: {train_acc:.1%} | Val Acc: {val_acc:.1%}")
    
    return history


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_connectivity_geometry(
    X: torch.Tensor,
    y: torch.Tensor,
    info: Dict
):
    """Analyze the geometry of connectivity matrices."""
    
    spd = SPD(info['n_regions'])
    
    print("\nConnectivity Geometry Analysis:")
    print("-" * 50)
    
    # 1. Compute class means
    healthy = X[y == 0]
    diseased = X[y == 1]
    
    healthy_mean = spd.project(healthy.mean(dim=0))
    diseased_mean = spd.project(diseased.mean(dim=0))
    
    # 2. Between-class distance
    between_dist = spd.distance(healthy_mean, diseased_mean)
    print(f"Distance between class means: {between_dist:.4f}")
    
    # 3. Within-class variance
    healthy_dists = torch.stack([spd.distance(h, healthy_mean) for h in healthy[:20]])
    diseased_dists = torch.stack([spd.distance(d, diseased_mean) for d in diseased[:20]])
    
    print(f"Within-class std (healthy): {healthy_dists.std():.4f}")
    print(f"Within-class std (diseased): {diseased_dists.std():.4f}")
    
    # 4. Fisher-like discriminability
    discriminability = between_dist / (healthy_dists.std() + diseased_dists.std())
    print(f"Discriminability ratio: {discriminability:.4f}")
    
    # 5. Eigenvalue analysis
    healthy_eigs = torch.linalg.eigvalsh(healthy_mean)
    diseased_eigs = torch.linalg.eigvalsh(diseased_mean)
    
    print(f"\nEigenvalue spectrum (healthy): [{healthy_eigs.min():.3f}, {healthy_eigs.max():.3f}]")
    print(f"Eigenvalue spectrum (diseased): [{diseased_eigs.min():.3f}, {diseased_eigs.max():.3f}]")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("BRAIN CONNECTIVITY CLASSIFICATION WITH SPD MANIFOLD")
    print("fMRI connectivity matrices as points on Riemannian manifold")
    print("=" * 70)
    
    torch.manual_seed(42)
    random.seed(42)
    
    # Generate data
    print("\n1. Generating synthetic brain connectivity data...")
    X, y, info = generate_brain_connectivity_data(
        n_subjects=200,
        n_regions=20,
        disease_effect=0.3
    )
    
    print(f"   Subjects: {info['n_subjects']} ({(y==0).sum()} healthy, {(y==1).sum()} diseased)")
    print(f"   Brain regions: {info['n_regions']}")
    print(f"   Networks: {info['n_networks']}")
    
    # Verify SPD property
    eigenvalues = torch.linalg.eigvalsh(X)
    print(f"   Min eigenvalue: {eigenvalues.min():.4f} (should be > 0)")
    
    # Split data
    n_train = int(0.7 * len(y))
    n_val = int(0.15 * len(y))
    
    indices = torch.randperm(len(y))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    
    train_X, train_y = X[train_idx], y[train_idx]
    val_X, val_y = X[val_idx], y[val_idx]
    test_X, test_y = X[test_idx], y[test_idx]
    
    print(f"\n   Train: {len(train_y)} | Val: {len(val_y)} | Test: {len(test_y)}")
    
    # Analyze geometry
    analyze_connectivity_geometry(X, y, info)
    
    # Train Euclidean baseline
    print("\n" + "=" * 70)
    print("2. EUCLIDEAN CLASSIFIER (Baseline)")
    print("=" * 70)
    
    euc_model = EuclideanClassifier(info['n_regions'], hidden_dim=64)
    print(f"   Parameters: {sum(p.numel() for p in euc_model.parameters()):,}")
    
    start = time.time()
    euc_history = train_model(euc_model, train_X, train_y, val_X, val_y, n_epochs=50)
    euc_time = time.time() - start
    
    # Test evaluation
    euc_model.eval()
    with torch.no_grad():
        test_logits = euc_model(test_X)
        euc_test_acc = (test_logits.argmax(1) == test_y).float().mean().item()
    
    print(f"\n   Test Accuracy: {euc_test_acc:.1%}")
    print(f"   Training Time: {euc_time:.1f}s")
    
    # Train SPD classifier
    print("\n" + "=" * 70)
    print("3. SPD CLASSIFIER (Riemannian)")
    print("=" * 70)
    
    spd_model = SPDClassifier(info['n_regions'], hidden_dim=10)
    print(f"   Parameters: {sum(p.numel() for p in spd_model.parameters()):,}")
    
    start = time.time()
    spd_history = train_model(spd_model, train_X, train_y, val_X, val_y, n_epochs=50)
    spd_time = time.time() - start
    
    spd_model.eval()
    with torch.no_grad():
        test_logits = spd_model(test_X)
        spd_test_acc = (test_logits.argmax(1) == test_y).float().mean().item()
    
    print(f"\n   Test Accuracy: {spd_test_acc:.1%}")
    print(f"   Training Time: {spd_time:.1f}s")
    
    # Train Fréchet mean classifier
    print("\n" + "=" * 70)
    print("4. FRÉCHET MEAN CLASSIFIER (Prototype-based)")
    print("=" * 70)
    
    frechet_model = FrechetMeanClassifier(info['n_regions'])
    print(f"   Parameters: {sum(p.numel() for p in frechet_model.parameters()):,}")
    
    # Initialize prototypes
    frechet_model.compute_prototypes(train_X, train_y)
    
    start = time.time()
    frechet_history = train_model(frechet_model, train_X, train_y, val_X, val_y, 
                                   n_epochs=30, lr=0.01)
    frechet_time = time.time() - start
    
    frechet_model.eval()
    with torch.no_grad():
        test_logits = frechet_model(test_X)
        frechet_test_acc = (test_logits.argmax(1) == test_y).float().mean().item()
    
    print(f"\n   Test Accuracy: {frechet_test_acc:.1%}")
    print(f"   Training Time: {frechet_time:.1f}s")
    
    # Comparison
    print("\n" + "=" * 70)
    print("5. COMPARISON")
    print("=" * 70)
    
    print(f"""
    ┌─────────────────────┬────────────┬────────────┬────────────┐
    │ Model               │ Test Acc   │ Parameters │ Time (s)   │
    ├─────────────────────┼────────────┼────────────┼────────────┤
    │ Euclidean (flat)    │ {euc_test_acc:>10.1%} │ {sum(p.numel() for p in euc_model.parameters()):>10,} │ {euc_time:>10.1f} │
    │ SPD (Riemannian)    │ {spd_test_acc:>10.1%} │ {sum(p.numel() for p in spd_model.parameters()):>10,} │ {spd_time:>10.1f} │
    │ Fréchet Mean        │ {frechet_test_acc:>10.1%} │ {sum(p.numel() for p in frechet_model.parameters()):>10,} │ {frechet_time:>10.1f} │
    └─────────────────────┴────────────┴────────────┴────────────┘
    """)
    
    # Summary
    print("=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
    SPD Manifold Advantages for Brain Connectivity:
    
    1. NATURAL REPRESENTATION
       - Correlation matrices ARE symmetric positive definite
       - SPD manifold operations preserve this structure
       - Euclidean operations can break positive-definiteness
    
    2. RIEMANNIAN OPERATIONS
       - Geodesic distance captures matrix similarity properly
       - Fréchet mean gives principled averaging
       - Log-map enables tangent space analysis
    
    3. INTERPRETABILITY
       - Eigenvalue spectra have neuroscience meaning
       - Network structure preserved through transformations
       - Prototypes are valid connectivity patterns
    
    4. EFFICIENCY
       - Fewer parameters needed (structure does work)
       - Geometry provides regularization
       - Works well with small datasets (typical in neuroimaging)
    
    Real-world applications:
    - Alzheimer's detection from resting-state fMRI
    - Autism classification from functional connectivity
    - Mental state decoding (attention, memory, emotion)
    - Brain-computer interfaces
    """)


if __name__ == '__main__':
    main()
