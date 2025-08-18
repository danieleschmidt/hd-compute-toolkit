"""
Sample data fixtures for testing HD-Compute-Toolkit functionality.

This module provides reusable test fixtures including sample hypervectors,
datasets, and configuration objects for consistent testing across all test suites.
"""

import numpy as np
import pytest
import torch
from typing import Dict, List, Tuple, Any

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


class HDCTestFixtures:
    """Container for HD-Compute-Toolkit test fixtures."""
    
    # Standard dimensions for testing
    DIMS = [1000, 4096, 10000, 16000]
    SMALL_DIM = 1000  # For quick tests
    DEFAULT_DIM = 10000  # Standard dimension
    LARGE_DIM = 16000   # For performance tests
    
    # Batch sizes for testing
    BATCH_SIZES = [1, 8, 32, 128]
    SMALL_BATCH = 8
    DEFAULT_BATCH = 32
    
    # Test data sizes
    NUM_SAMPLES = {
        'tiny': 10,
        'small': 100,
        'medium': 1000,
        'large': 10000
    }


@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Global test configuration."""
    return {
        'default_dim': HDCTestFixtures.DEFAULT_DIM,
        'default_batch': HDCTestFixtures.DEFAULT_BATCH,
        'seed': 42,
        'tolerance': 1e-6,
        'device': 'cpu',  # Override in GPU tests
    }


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_binary_hv(test_config: Dict[str, Any]) -> np.ndarray:
    """Generate a sample binary hypervector."""
    dim = test_config['default_dim']
    rng = np.random.default_rng(test_config['seed'])
    return (rng.random(dim) > 0.5).astype(np.int8)


@pytest.fixture
def sample_binary_hvs(test_config: Dict[str, Any]) -> np.ndarray:
    """Generate a batch of sample binary hypervectors."""
    dim = test_config['default_dim']
    batch = test_config['default_batch']
    rng = np.random.default_rng(test_config['seed'])
    return (rng.random((batch, dim)) > 0.5).astype(np.int8)


@pytest.fixture
def sample_torch_hv(test_config: Dict[str, Any]) -> torch.Tensor:
    """Generate a sample binary hypervector as PyTorch tensor."""
    dim = test_config['default_dim']
    torch.manual_seed(test_config['seed'])
    return torch.randint(0, 2, (dim,), dtype=torch.int8)


@pytest.fixture
def sample_torch_hvs(test_config: Dict[str, Any]) -> torch.Tensor:
    """Generate a batch of sample binary hypervectors as PyTorch tensors."""
    dim = test_config['default_dim']
    batch = test_config['default_batch']
    torch.manual_seed(test_config['seed'])
    return torch.randint(0, 2, (batch, dim), dtype=torch.int8)


@pytest.fixture
def sample_jax_hv(test_config: Dict[str, Any]):
    """Generate a sample binary hypervector as JAX array."""
    if not JAX_AVAILABLE:
        pytest.skip("JAX not available")
    
    dim = test_config['default_dim']
    key = jax.random.PRNGKey(test_config['seed'])
    return jax.random.bernoulli(key, 0.5, (dim,)).astype(jnp.int8)


@pytest.fixture
def sample_jax_hvs(test_config: Dict[str, Any]):
    """Generate a batch of sample binary hypervectors as JAX arrays."""
    if not JAX_AVAILABLE:
        pytest.skip("JAX not available")
    
    dim = test_config['default_dim']
    batch = test_config['default_batch']
    key = jax.random.PRNGKey(test_config['seed'])
    return jax.random.bernoulli(key, 0.5, (batch, dim)).astype(jnp.int8)


@pytest.fixture
def sample_item_memory_data() -> Dict[str, List]:
    """Sample data for item memory testing."""
    return {
        'symbols': ['cat', 'dog', 'bird', 'fish', 'tree', 'car', 'house'],
        'sequences': [
            ['cat', 'dog'],
            ['bird', 'fish', 'tree'],
            ['car', 'house', 'cat'],
            ['dog', 'bird', 'car', 'fish']
        ],
        'labels': [0, 1, 0, 1]
    }


@pytest.fixture
def sample_speech_data() -> Dict[str, Any]:
    """Sample data for speech command testing."""
    return {
        'commands': ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off'],
        'features': np.random.randn(100, 13),  # MFCC features
        'labels': np.random.randint(0, 8, 100),
        'sample_rate': 16000,
        'duration': 1.0  # seconds
    }


@pytest.fixture
def performance_test_sizes() -> List[Tuple[int, int]]:
    """Dimension and batch size combinations for performance testing."""
    return [
        (1000, 1),
        (1000, 100),
        (10000, 1),
        (10000, 100),
        (16000, 1),
        (16000, 100),
    ]


@pytest.fixture
def stress_test_config() -> Dict[str, Any]:
    """Configuration for stress testing."""
    return {
        'max_dimension': 32000,
        'max_batch_size': 1000,
        'max_memory_mb': 4096,  # 4GB memory limit
        'timeout_seconds': 300,   # 5 minute timeout
    }


@pytest.fixture
def mock_dataset() -> Dict[str, Any]:
    """Mock dataset for testing applications."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 13  # MFCC features
    n_classes = 10
    
    return {
        'train': {
            'features': np.random.randn(n_samples, n_features),
            'labels': np.random.randint(0, n_classes, n_samples)
        },
        'val': {
            'features': np.random.randn(n_samples // 5, n_features),
            'labels': np.random.randint(0, n_classes, n_samples // 5)
        },
        'test': {
            'features': np.random.randn(n_samples // 5, n_features),
            'labels': np.random.randint(0, n_classes, n_samples // 5)
        },
        'metadata': {
            'n_samples': n_samples,
            'n_features': n_features,
            'n_classes': n_classes,
            'class_names': [f'class_{i}' for i in range(n_classes)]
        }
    }


@pytest.fixture
def gpu_test_config() -> Dict[str, Any]:
    """Configuration for GPU tests."""
    return {
        'devices': ['cuda:0'] if torch.cuda.is_available() else [],
        'memory_fraction': 0.8,
        'sync_operations': True,  # For testing synchronization
    }


@pytest.fixture
def benchmark_config() -> Dict[str, Any]:
    """Configuration for benchmark tests."""
    return {
        'warmup_rounds': 3,
        'timing_rounds': 10,
        'min_time': 0.001,  # Minimum timing resolution
        'max_time': 60.0,   # Maximum time per benchmark
        'memory_tracking': True,
    }


# Utility functions for test data generation
def generate_similarity_pairs(
    hvs: np.ndarray, 
    similarity_threshold: float = 0.8
) -> List[Tuple[int, int, float]]:
    """Generate pairs of hypervectors with known similarity scores."""
    pairs = []
    n_hvs = len(hvs)
    
    for i in range(n_hvs):
        for j in range(i + 1, n_hvs):
            # Compute Hamming similarity
            similarity = 1.0 - np.mean(hvs[i] != hvs[j])
            if similarity >= similarity_threshold:
                pairs.append((i, j, similarity))
    
    return pairs


def create_test_hierarchy() -> Dict[str, Any]:
    """Create a hierarchical structure for testing nested operations."""
    return {
        'animals': {
            'mammals': ['cat', 'dog', 'whale'],
            'birds': ['eagle', 'sparrow', 'penguin'],
            'fish': ['salmon', 'shark', 'goldfish']
        },
        'objects': {
            'vehicles': ['car', 'bike', 'plane'],
            'furniture': ['chair', 'table', 'bed'],
            'tools': ['hammer', 'saw', 'drill']
        }
    }