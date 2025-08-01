"""Shared pytest configuration and fixtures for HD-Compute-Toolkit tests."""

import pytest
import numpy as np
import torch
import tempfile
import os
from pathlib import Path
from typing import Generator, Dict, Any, List
from unittest.mock import Mock

# Import the main module (when it exists)
try:
    from hd_compute.core import HDComputeBase
except ImportError:
    HDComputeBase = None


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance benchmark tests")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU hardware")
    config.addinivalue_line("markers", "fpga: Tests requiring FPGA hardware") 
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "memory: Memory usage tests")


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location."""
    for item in items:
        # Mark tests based on directory structure
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
            
        # Mark GPU tests based on test name
        if "gpu" in item.name.lower() or "cuda" in item.name.lower():
            item.add_marker(pytest.mark.gpu)
            
        # Mark FPGA tests
        if "fpga" in item.name.lower():
            item.add_marker(pytest.mark.fpga)
            
        # Mark slow tests
        if "benchmark" in item.name.lower() or "stress" in item.name.lower():
            item.add_marker(pytest.mark.slow)


# Hardware availability fixtures
@pytest.fixture(scope="session")
def cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


@pytest.fixture(scope="session") 
def gpu_device() -> str:
    """Get the appropriate GPU device string."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def fpga_available() -> bool:
    """Check if FPGA hardware is available."""
    try:
        import pynq
        return True
    except ImportError:
        return False


# Test data fixtures
@pytest.fixture(scope="session")
def test_dimensions() -> List[int]:
    """Standard test dimensions for hypervectors."""
    return [100, 1000, 10000]


@pytest.fixture(scope="session")
def large_test_dimensions() -> List[int]:
    """Large dimensions for performance testing."""
    return [16000, 32000]


@pytest.fixture
def random_seed() -> int:
    """Fixed random seed for reproducible tests."""
    return 42


@pytest.fixture
def numpy_rng(random_seed: int) -> np.random.Generator:
    """NumPy random number generator with fixed seed."""
    return np.random.default_rng(random_seed)


@pytest.fixture
def torch_rng(random_seed: int) -> torch.Generator:
    """PyTorch random number generator with fixed seed."""
    generator = torch.Generator()
    generator.manual_seed(random_seed)
    return generator


# Hypervector test data
@pytest.fixture
def binary_hypervector_1d(numpy_rng: np.random.Generator) -> np.ndarray:
    """Generate a 1D binary hypervector for testing."""
    return numpy_rng.choice([0, 1], size=1000).astype(np.int8)


@pytest.fixture
def binary_hypervector_batch(numpy_rng: np.random.Generator) -> np.ndarray:
    """Generate a batch of binary hypervectors for testing."""
    return numpy_rng.choice([0, 1], size=(32, 1000)).astype(np.int8)


@pytest.fixture 
def bipolar_hypervector_1d(numpy_rng: np.random.Generator) -> np.ndarray:
    """Generate a 1D bipolar hypervector for testing."""
    return numpy_rng.choice([-1, 1], size=1000).astype(np.int8)


@pytest.fixture
def sparse_hypervector(numpy_rng: np.random.Generator) -> np.ndarray:
    """Generate a sparse hypervector with controlled sparsity."""
    hv = np.zeros(1000, dtype=np.int8)
    num_ones = int(1000 * 0.1)  # 10% sparsity
    indices = numpy_rng.choice(1000, size=num_ones, replace=False)
    hv[indices] = 1
    return hv


# Temporary file fixtures
@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def temp_file() -> Generator[Path, None, None]:
    """Create a temporary file for testing."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        temp_path = Path(tmp_file.name)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


# Mock fixtures for external dependencies
@pytest.fixture
def mock_cuda_device() -> Mock:
    """Mock CUDA device for testing without hardware."""
    mock_device = Mock()
    mock_device.type = "cuda"
    mock_device.index = 0
    return mock_device


@pytest.fixture
def mock_fpga_overlay() -> Mock:
    """Mock FPGA overlay for testing without hardware."""
    mock_overlay = Mock()
    mock_overlay.is_loaded.return_value = True
    mock_overlay.download.return_value = None
    return mock_overlay


# Benchmark fixtures
@pytest.fixture
def benchmark_config() -> Dict[str, Any]:
    """Configuration for benchmark tests."""
    return {
        "warmup_iterations": 5,
        "measurement_iterations": 10,
        "timeout_seconds": 30,
        "dimensions": [1000, 10000],
        "batch_sizes": [1, 32, 128],
        "devices": ["cpu"],
    }


@pytest.fixture
def performance_thresholds() -> Dict[str, float]:
    """Performance thresholds for benchmark validation."""
    return {
        "random_hv_latency_ms": 10.0,
        "bundle_latency_ms": 5.0,
        "bind_latency_ms": 5.0,
        "similarity_latency_ms": 2.0,
        "memory_usage_mb": 100.0,
    }


# Test environment fixtures
@pytest.fixture(scope="session")
def test_env_info() -> Dict[str, Any]:
    """Collect test environment information."""
    info = {
        "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    try:
        import jax
        info["jax_version"] = jax.__version__
        info["jax_backend"] = jax.default_backend()
    except ImportError:
        info["jax_version"] = None
        info["jax_backend"] = None
        
    try:
        import pynq
        info["pynq_available"] = True
        info["pynq_version"] = pynq.__version__
    except ImportError:
        info["pynq_available"] = False
        info["pynq_version"] = None
    
    return info


# Parameterized test fixtures
@pytest.fixture(params=["cpu", "cuda"])
def device_under_test(request, cuda_available: bool) -> str:
    """Parameterized device fixture for cross-device testing."""
    device = request.param
    if device == "cuda" and not cuda_available:
        pytest.skip("CUDA not available")
    return device


@pytest.fixture(params=[100, 1000, 10000])
def dimension_under_test(request) -> int:
    """Parameterized dimension fixture for testing different sizes."""
    return request.param


@pytest.fixture(params=["binary", "bipolar"])
def hypervector_type(request) -> str:
    """Parameterized hypervector type for testing different encodings."""
    return request.param


# Hypothesis integration
try:
    from hypothesis import strategies as st
    
    @pytest.fixture
    def hypothesis_dimension_strategy():
        """Hypothesis strategy for generating test dimensions."""
        return st.integers(min_value=64, max_value=10000)
    
    @pytest.fixture
    def hypothesis_batch_size_strategy():
        """Hypothesis strategy for generating batch sizes."""
        return st.integers(min_value=1, max_value=128)
        
except ImportError:
    # Hypothesis not available, provide dummy fixtures
    @pytest.fixture
    def hypothesis_dimension_strategy():
        return None
        
    @pytest.fixture
    def hypothesis_batch_size_strategy():
        return None


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_cuda_cache():
    """Automatically cleanup CUDA cache after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(autouse=True)
def reset_random_seeds(random_seed: int):
    """Reset random seeds before each test for reproducibility."""
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    yield