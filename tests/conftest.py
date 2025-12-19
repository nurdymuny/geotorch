"""Pytest configuration and fixtures."""

import pytest
import torch
from geotorch import Euclidean, Sphere, Hyperbolic


@pytest.fixture
def euclidean():
    """Fixture for Euclidean manifold."""
    return Euclidean(64)


@pytest.fixture
def sphere():
    """Fixture for Sphere manifold."""
    return Sphere(64)


@pytest.fixture
def hyperbolic():
    """Fixture for Hyperbolic manifold."""
    return Hyperbolic(64, model='poincare')


@pytest.fixture(params=[
    Euclidean(64),
    Sphere(64),
    Hyperbolic(64, model='poincare')
])
def manifold(request):
    """Fixture that parametrizes over all manifolds."""
    return request.param


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    return 42
