# Testing Guide

Comprehensive testing guide for HD-Compute-Toolkit development and validation.

## Test Structure

```
tests/
├── unit/              # Unit tests for individual components
├── integration/       # Integration tests for component interactions
├── e2e/              # End-to-end tests for complete workflows
├── performance/      # Performance benchmarks and regression tests
├── hardware/         # Hardware-specific tests (GPU, FPGA, Vulkan)
├── fixtures/         # Test data and mock objects
└── conftest.py       # PyTest configuration and shared fixtures
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m benchmark     # Performance benchmarks
pytest -m gpu          # GPU-specific tests
pytest -m slow         # Long-running tests

# Exclude specific categories
pytest -m "not slow"    # Skip slow tests
pytest -m "not gpu"     # Skip GPU tests
```

### Coverage Reports

```bash
# Generate HTML coverage report
pytest --cov=hd_compute --cov-report=html

# View coverage in terminal
pytest --cov=hd_compute --cov-report=term-missing

# Generate XML coverage for CI
pytest --cov=hd_compute --cov-report=xml
```

### Parallel Testing

```bash
# Run tests in parallel (requires pytest-xdist)
pytest -n auto          # Auto-detect CPU cores
pytest -n 4            # Use 4 parallel workers
```

## Test Categories

### Unit Tests (`tests/unit/`)

Test individual functions and methods in isolation.

```python
import pytest
from hd_compute.core import HDCompute

class TestHDCompute:
    def test_random_hv_generation(self):
        hdc = HDCompute(dim=1000)
        hv = hdc.random_hv()
        assert hv.shape == (1000,)
        assert hv.dtype == torch.bool
```

### Integration Tests (`tests/integration/`)

Test component interactions and system behavior.

```python
def test_pytorch_jax_compatibility():
    # Test that PyTorch and JAX backends produce compatible results
    torch_hdc = HDComputeTorch(dim=1000)
    jax_hdc = HDComputeJAX(dim=1000, key=jax.random.PRNGKey(42))
    
    # Compare operations
    assert torch.allclose(
        torch_result,
        torch.from_numpy(jax_result)
    )
```

### End-to-End Tests (`tests/e2e/`)

Test complete workflows from input to output.

```python
def test_speech_command_pipeline():
    model = SpeechCommandHDC(dim=16000, num_classes=35)
    
    # Load test audio
    audio = load_test_audio("yes.wav")
    
    # Train on small dataset
    model.train(mini_dataset, epochs=1)
    
    # Test inference
    prediction = model.predict(audio)
    assert prediction in model.class_names
```

### Performance Tests (`tests/performance/`)

Benchmark operations and detect performance regressions.

```python
@pytest.mark.benchmark
def test_bundle_performance(benchmark):
    hdc = HDCompute(dim=16000, device='cuda')
    hypervectors = [hdc.random_hv() for _ in range(1000)]
    
    result = benchmark(hdc.bundle, hypervectors)
    assert result.shape == (16000,)
```

### Hardware Tests (`tests/hardware/`)

Test hardware-specific acceleration.

```python
@pytest.mark.gpu
def test_gpu_acceleration():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    hdc_cpu = HDCompute(dim=16000, device='cpu')
    hdc_gpu = HDCompute(dim=16000, device='cuda')
    
    # Verify GPU speedup
    assert gpu_time < cpu_time * 0.5  # At least 2x speedup
```

## Test Configuration

### PyTest Markers

Configure test markers in `pytest.ini`:

```ini
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    gpu: marks tests that require GPU
    integration: marks tests as integration tests
    benchmark: marks tests as benchmarks
    hardware: marks tests that require special hardware
    unit: marks tests as unit tests
```

### Fixtures (`conftest.py`)

Common test setup and teardown:

```python
@pytest.fixture
def hdc_config():
    return {
        'dim': 1000,
        'device': 'cpu',
        'dtype': torch.bool
    }

@pytest.fixture
def sample_hypervectors(hdc_config):
    hdc = HDCompute(**hdc_config)
    return [hdc.random_hv() for _ in range(10)]
```

## Continuous Integration

### GitHub Actions Workflow

```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
    
    - name: Run tests
      run: |
        pytest -m "not slow and not gpu" --cov=hd_compute
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## Mock and Test Data

### Creating Test Fixtures

```python
# tests/fixtures/data.py
import torch
import numpy as np

def create_test_audio():
    """Generate synthetic audio for testing."""
    return torch.randn(16000)  # 1 second at 16kHz

def create_test_hypervectors(dim=1000, count=10):
    """Generate test hypervectors."""
    return torch.randint(0, 2, (count, dim), dtype=torch.bool)
```

### Mocking External Dependencies

```python
from unittest.mock import patch, MagicMock

@patch('hd_compute.hardware.cuda.is_available')
def test_fallback_to_cpu(mock_cuda):
    mock_cuda.return_value = False
    
    hdc = HDCompute(dim=1000, device='auto')
    assert hdc.device == 'cpu'
```

## Performance Testing

### Benchmark Configuration

```python
# pytest-benchmark configuration in pyproject.toml
[tool.pytest.ini_options]
addopts = "--benchmark-only --benchmark-sort=mean"

[tool.benchmark]
min_rounds = 5
max_time = 10.0
```

### Performance Regression Detection

```bash
# Store baseline performance
pytest --benchmark-save=baseline

# Compare against baseline
pytest --benchmark-compare=baseline
```

## Memory and Resource Testing

### Memory Leak Detection

```python
import tracemalloc

def test_memory_usage():
    tracemalloc.start()
    
    # Run memory-intensive operations
    large_computation()
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    assert peak < 1024 * 1024 * 100  # Less than 100MB
```

### Resource Cleanup

```python
@pytest.fixture(autouse=True)
def cleanup_resources():
    yield
    
    # Clean up CUDA memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Clean up temporary files
    cleanup_temp_files()
```

## Test Best Practices

### Writing Good Tests

1. **Arrange-Act-Assert**: Structure tests clearly
2. **Single Responsibility**: One assertion per test when possible
3. **Descriptive Names**: Test names should describe the scenario
4. **Fast by Default**: Make slow tests explicit with markers
5. **Deterministic**: Use fixed seeds for reproducible tests

### Test Data Management

```python
# Use parametrized tests for multiple scenarios
@pytest.mark.parametrize("dim,device", [
    (1000, 'cpu'),
    (10000, 'cpu'),
    pytest.param(16000, 'cuda', marks=pytest.mark.gpu)
])
def test_various_configurations(dim, device):
    hdc = HDCompute(dim=dim, device=device)
    result = hdc.random_hv()
    assert result.shape == (dim,)
```

## Debugging Failed Tests

### Verbose Output

```bash
# Increase verbosity
pytest -v -s

# Show local variables on failure
pytest --tb=long --showlocals

# Enter debugger on failure
pytest --pdb
```

### Test Isolation

```bash
# Run specific test
pytest tests/unit/test_hdc_operations.py::test_bundle

# Run tests matching pattern
pytest -k "test_bundle"

# Stop after first failure
pytest -x
```

## Quality Gates

### Coverage Requirements

- Minimum 80% line coverage
- Critical paths require 95% coverage
- Hardware-specific code exempted from coverage requirements

### Performance Requirements

- Core operations must complete within defined time limits
- Memory usage must not exceed 2x theoretical minimum
- GPU acceleration must provide minimum 2x speedup

### Integration Requirements

- All integration tests must pass
- Cross-framework compatibility verified
- Hardware acceleration validated where available