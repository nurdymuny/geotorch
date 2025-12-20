# GeoTorch: Manifold-Native Deep Learning Framework

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

GeoTorch is a Riemannian deep learning framework that extends PyTorch with native support for manifold-valued parameters, geodesic optimization, and geometric operations.

## Installation

### From Source

```bash
git clone https://github.com/nurdymuny/geotorch.git
cd geotorch
pip install -e .
```

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- NumPy ≥ 1.24

### Development Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import torch
from geotorch import Sphere, ManifoldTensor

# Create a sphere manifold
S = Sphere(64)  # S^63 (63-dimensional sphere in R^64)

# Generate random points on the sphere
p = S.random_point()
q = S.random_point()

# Compute geodesic distance
distance = S.distance(p, q)

# Logarithmic map: tangent vector from p to q
v = S.log(p, q)

# Exponential map: move along geodesic
new_point = S.exp(p, v)

# Use ManifoldTensor for convenient operations
p_tensor = ManifoldTensor(p, S)
q_tensor = ManifoldTensor(q, S)
distance = p_tensor.distance(q_tensor)
```

## Supported Manifolds

- **Euclidean**: Flat space ℝⁿ
- **Sphere**: Unit sphere S^{n-1} ⊂ ℝⁿ
- **Hyperbolic**: Hyperbolic space H^{n-1} (Poincaré ball and hyperboloid models)

## Testing

Run the test suite:

```bash
pytest tests/
```

## Documentation

For detailed documentation, see [spec.md](spec.md).

## License

Apache License 2.0. See LICENSE file for details.

## Citation

If you use GeoTorch in your research, please cite:

```bibtex
@software{geotorch2025,
  author = {Davis, Bee Rosa},
  title = {GeoTorch: Manifold-Native Deep Learning Framework},
  year = {2025},
  url = {https://github.com/nurdymuny/geotorch},
  note = {Riemannian deep learning with geodesic optimization and O(1) retrieval}
}
```