# Developer Guide

This guide covers the development workflow, architecture patterns, and best practices for contributing to HD-Compute-Toolkit.

## Development Environment Setup

### Prerequisites

- Python 3.8+ with virtual environment support
- Git with pre-commit hooks
- CUDA Toolkit (for GPU development)
- Docker (for containerized testing)

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/hd-compute-toolkit
cd hd-compute-toolkit

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Setup pre-commit hooks
pre-commit install

# Verify installation
pytest
```

### Development Tools

The project uses several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting and style checking
- **mypy**: Static type checking
- **pytest**: Testing framework
- **pre-commit**: Git hook automation

## Architecture Overview

### Module Structure

```
hd_compute/
├── core/           # Abstract base classes and interfaces
├── torch/          # PyTorch backend implementation
├── jax/            # JAX backend implementation  
├── kernels/        # Hardware acceleration kernels
├── memory/         # Memory structure implementations
├── applications/   # Domain-specific applications
└── utils/          # Utility functions and helpers
```

### Core Design Patterns

#### Abstract Base Class Pattern

All backends implement common interfaces defined in `hd_compute.core`:

```python
from abc import ABC, abstractmethod
from typing import Union, List
import numpy as np

class HDCBackend(ABC):
    """Abstract base class for HDC backends"""
    
    @abstractmethod
    def random_hv(self) -> 'Hypervector':
        """Generate random hypervector"""
        pass
    
    @abstractmethod
    def bundle(self, hvs: List['Hypervector']) -> 'Hypervector':
        """Bundle multiple hypervectors"""
        pass
    
    @abstractmethod
    def bind(self, hv1: 'Hypervector', hv2: 'Hypervector') -> 'Hypervector':
        """Bind two hypervectors"""
        pass
```

#### Backend Registration Pattern

Backends are registered dynamically for automatic discovery:

```python
from hd_compute.core.registry import register_backend

@register_backend('torch')
class HDComputeTorch(HDCBackend):
    """PyTorch implementation"""
    pass

@register_backend('jax')  
class HDComputeJAX(HDCBackend):
    """JAX implementation"""
    pass
```

## Implementing New Features

### Adding Core Operations

1. **Define the interface** in `hd_compute/core/interfaces.py`
2. **Implement in PyTorch** backend (`hd_compute/torch/`)
3. **Implement in JAX** backend (`hd_compute/jax/`)
4. **Add comprehensive tests** for both backends
5. **Update documentation** with examples

Example: Adding a new similarity metric

```python
# Step 1: Define interface
class SimilarityMixin(ABC):
    @abstractmethod
    def jaccard_similarity(self, hv1: 'Hypervector', hv2: 'Hypervector') -> float:
        """Compute Jaccard similarity between binary hypervectors"""
        pass

# Step 2: PyTorch implementation  
def jaccard_similarity(self, hv1: torch.Tensor, hv2: torch.Tensor) -> float:
    intersection = torch.logical_and(hv1, hv2).sum()
    union = torch.logical_or(hv1, hv2).sum()
    return (intersection / union).item()

# Step 3: JAX implementation
def jaccard_similarity(self, hv1: jnp.ndarray, hv2: jnp.ndarray) -> float:
    intersection = jnp.logical_and(hv1, hv2).sum()
    union = jnp.logical_or(hv1, hv2).sum()
    return float(intersection / union)

# Step 4: Tests
def test_jaccard_similarity():
    hdc = HDCompute(dim=1000)
    hv1 = hdc.random_hv()
    hv2 = hdc.random_hv()
    
    similarity = hdc.jaccard_similarity(hv1, hv2)
    assert 0.0 <= similarity <= 1.0
    
    # Test edge cases
    assert hdc.jaccard_similarity(hv1, hv1) == 1.0
```

### Adding Hardware Acceleration

Hardware kernels follow a plugin architecture:

```python
# Step 1: Define kernel interface
from hd_compute.kernels.base import HardwareKernel

class CustomKernel(HardwareKernel):
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self._initialize_hardware()
    
    def bundle_batch(self, hvs: List[np.ndarray]) -> np.ndarray:
        """Accelerated batch bundling"""
        # Hardware-specific implementation
        pass
    
    def _initialize_hardware(self):
        """Initialize hardware resources"""
        pass

# Step 2: Register kernel
from hd_compute.kernels.registry import register_kernel

register_kernel('custom', CustomKernel)

# Step 3: Use in main API
hdc = HDCompute(dim=10000, accelerator='custom')
```

### Adding Applications

Applications demonstrate domain-specific HDC usage:

```python
# File: hd_compute/applications/example_app.py

from hd_compute.core.application import HDCApplication
from typing import List, Tuple, Any

class ExampleApplication(HDCApplication):
    """Example HDC application for [domain]"""
    
    def __init__(self, dim: int = 10000, **kwargs):
        super().__init__(dim=dim, **kwargs)
        self._setup_domain_specific_components()
    
    def train(self, data: List[Tuple[Any, Any]]) -> None:
        """Train the HDC model on domain data"""
        for features, label in data:
            encoded_features = self._encode_features(features)
            self._update_memory(encoded_features, label)
    
    def predict(self, features: Any) -> Any:
        """Make prediction on new data"""
        encoded = self._encode_features(features) 
        return self._memory_lookup(encoded)
    
    def _encode_features(self, features: Any) -> 'Hypervector':
        """Domain-specific feature encoding"""
        # Implementation depends on domain
        pass
```

## Testing Guidelines

### Test Structure

```
tests/
├── unit/              # Unit tests for individual components
├── integration/       # Integration tests across components  
├── performance/       # Performance benchmarks
├── hardware/          # Hardware-specific tests
└── applications/      # Application-level tests
```

### Writing Tests

#### Unit Tests

Focus on individual function behavior:

```python
import pytest
from hd_compute import HDCompute

class TestBundling:
    @pytest.fixture
    def hdc(self):
        return HDCompute(dim=1000, seed=42)
    
    def test_bundle_preserves_similarity(self, hdc):
        """Bundled vectors should be similar to components"""
        hv1 = hdc.random_hv()
        hv2 = hdc.random_hv()
        bundled = hdc.bundle([hv1, hv2])
        
        sim1 = hdc.cosine_similarity(bundled, hv1)
        sim2 = hdc.cosine_similarity(bundled, hv2)
        
        assert sim1 > 0.5  # Should be similar
        assert sim2 > 0.5
    
    def test_bundle_empty_list_raises(self, hdc):
        """Bundling empty list should raise ValueError"""
        with pytest.raises(ValueError):
            hdc.bundle([])
```

#### Integration Tests

Test component interactions:

```python
def test_end_to_end_classification():
    """Test complete classification pipeline"""
    hdc = HDCompute(dim=5000)
    
    # Generate synthetic data
    train_data = generate_classification_data(n_samples=100)
    test_data = generate_classification_data(n_samples=20)
    
    # Train classifier
    classifier = hdc.create_classifier()
    classifier.fit(train_data)
    
    # Test accuracy
    predictions = classifier.predict([x for x, y in test_data])
    actual = [y for x, y in test_data]
    
    accuracy = sum(p == a for p, a in zip(predictions, actual)) / len(actual)
    assert accuracy > 0.8  # Expect reasonable performance
```

#### Performance Tests

Benchmark critical operations:

```python
import time
import pytest

@pytest.mark.performance
def test_bundling_performance():
    """Bundle operation should complete within time limits"""
    hdc = HDCompute(dim=10000, device='cuda')
    hvs = [hdc.random_hv() for _ in range(1000)]
    
    start_time = time.time()
    result = hdc.bundle(hvs)
    end_time = time.time()
    
    elapsed = end_time - start_time
    assert elapsed < 0.1  # Should complete in <100ms
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest -m performance  # Performance tests only

# Run with coverage
pytest --cov=hd_compute --cov-report=html

# Run tests for specific backend
pytest -k "torch"
pytest -k "jax"
```

## Code Quality Standards

### Type Hints

All public APIs must include type hints:

```python
from typing import List, Optional, Union, Tuple
import numpy as np

def bundle(
    self, 
    hvs: List['Hypervector'], 
    weights: Optional[np.ndarray] = None
) -> 'Hypervector':
    """Bundle hypervectors with optional weights"""
    pass
```

### Documentation

Use Google-style docstrings:

```python
def cosine_similarity(self, hv1: 'Hypervector', hv2: 'Hypervector') -> float:
    """Compute cosine similarity between two hypervectors.
    
    Args:
        hv1: First hypervector for comparison
        hv2: Second hypervector for comparison
        
    Returns:
        Similarity score between -1 and 1, where 1 indicates
        identical vectors and -1 indicates opposite vectors.
        
    Raises:
        ValueError: If hypervectors have different dimensions
        
    Example:
        >>> hdc = HDCompute(dim=1000)
        >>> hv1 = hdc.random_hv()
        >>> hv2 = hdc.random_hv()
        >>> similarity = hdc.cosine_similarity(hv1, hv2)
        >>> print(f"Similarity: {similarity:.3f}")
    """
    if hv1.shape != hv2.shape:
        raise ValueError("Hypervectors must have same dimensions")
    # Implementation...
```

### Error Handling

Use descriptive error messages and appropriate exception types:

```python
def bind(self, hv1: 'Hypervector', hv2: 'Hypervector') -> 'Hypervector':
    if hv1.shape != hv2.shape:
        raise ValueError(
            f"Cannot bind hypervectors with different dimensions: "
            f"{hv1.shape} vs {hv2.shape}"
        )
    
    if not self._is_binary(hv1) or not self._is_binary(hv2):
        raise TypeError(
            "Bind operation requires binary hypervectors"
        )
```

## Performance Optimization

### Profiling

Use built-in profiling tools to identify bottlenecks:

```python
# For PyTorch backend
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
) as prof:
    result = hdc.bundle(large_hypervector_list)

print(prof.key_averages().table(sort_by="cuda_time_total"))

# For JAX backend  
with jax.profiler.trace("/tmp/jax_trace"):
    result = hdc_jax.bundle(large_hypervector_list)
```

### Memory Management

#### Memory Pooling

```python
class MemoryPool:
    """Reusable memory pool for hypervectors"""
    
    def __init__(self, dim: int, pool_size: int = 1000):
        self.dim = dim
        self.pool = [self._allocate() for _ in range(pool_size)]
        self.available = list(range(pool_size))
    
    def get(self) -> int:
        if not self.available:
            raise RuntimeError("Memory pool exhausted")
        return self.available.pop()
    
    def release(self, idx: int):
        self.available.append(idx)
        self.pool[idx].zero_()  # Clear for reuse
```

#### Batch Operations

Optimize for batch processing:

```python
def bundle_batch(self, hvs_batch: List[List['Hypervector']]) -> List['Hypervector']:
    """Bundle multiple groups of hypervectors efficiently"""
    # Stack all hypervectors for vectorized operations
    stacked = torch.stack([torch.stack(hvs) for hvs in hvs_batch])
    
    # Vectorized bundling across batches
    bundled = stacked.sum(dim=1) > (stacked.shape[1] // 2)
    
    return [bundled[i] for i in range(bundled.shape[0])]
```

## Debugging Tips

### Common Issues

1. **Dimension Mismatches**: Always check hypervector dimensions
2. **Device Mismatches**: Ensure all tensors are on same device
3. **Memory Leaks**: Use context managers for resource cleanup
4. **Precision Issues**: Be aware of floating-point vs binary operations

### Debugging Tools

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# PyTorch debugging
torch.autograd.set_detect_anomaly(True)

# JAX debugging  
from jax.config import config
config.update("jax_debug_nans", True)
```

## Release Process

### Version Bumping

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md` with release notes
3. Run full test suite across all backends
4. Build and test packages locally
5. Create release tag and push to GitHub

### Continuous Integration

The CI pipeline runs:
- Tests across Python 3.8-3.11
- Tests on CPU and GPU environments  
- Code quality checks (black, flake8, mypy)
- Documentation building
- Performance regression tests

## Contributing Guidelines

### Pull Request Process

1. Fork the repository and create feature branch
2. Implement changes following code quality standards
3. Add comprehensive tests for new functionality
4. Update documentation and examples
5. Ensure CI passes all checks
6. Submit PR with detailed description

### Community Guidelines

- Be respectful and inclusive in all interactions
- Provide constructive feedback in code reviews
- Help newcomers get started with the project
- Share knowledge through documentation and examples

For more details, see [CONTRIBUTING.md](../../CONTRIBUTING.md).