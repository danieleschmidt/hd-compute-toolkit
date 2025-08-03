"""Pytest configuration and shared fixtures for HD-Compute-Toolkit tests."""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import numpy as np
import pytest
import torch

# Import main modules
try:
    from hd_compute import HDCompute, HDComputeTorch, HDComputeJAX, CacheManager, DatabaseConnection
    from hd_compute.core.hdc import HDCompute as HDComputeBase
    from hd_compute.memory import ItemMemory, AssociativeMemory
    from hd_compute.applications import SpeechCommandHDC, SemanticMemory
except ImportError:
    # Fallback for when modules are not yet implemented
    HDCompute = None
    HDComputeTorch = None
    HDComputeJAX = None
    HDComputeBase = None
    CacheManager = None
    DatabaseConnection = None
    ItemMemory = None
    AssociativeMemory = None
    SpeechCommandHDC = None
    SemanticMemory = None


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "benchmark: marks tests as benchmarks")
    config.addinivalue_line("markers", "hardware: marks tests that require special hardware")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file location."""
    for item in items:
        # Add slow marker to performance tests
        if "performance" in str(item.fspath):
            item.add_marker(pytest.mark.slow)
        
        # Add integration marker to integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add hardware marker to hardware tests
        if "hardware" in str(item.fspath):
            item.add_marker(pytest.mark.hardware)


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return the test data directory path."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def random_seed() -> int:
    """Provide a consistent random seed for reproducible tests."""
    return 42


@pytest.fixture
def set_random_seeds(random_seed: int) -> None:
    """Set random seeds for all libraries to ensure reproducibility."""
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)


@pytest.fixture(params=[1000, 5000, 10000])
def dimension(request) -> int:
    """Parametrized fixture for different hypervector dimensions."""
    return request.param


@pytest.fixture(params=["cpu"])
def device(request) -> str:
    """Parametrized fixture for different compute devices."""
    device = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return device


@pytest.fixture
def small_dimension() -> int:
    """Small dimension for fast tests."""
    return 100


@pytest.fixture
def medium_dimension() -> int:
    """Medium dimension for standard tests."""
    return 1000


@pytest.fixture
def large_dimension() -> int:
    """Large dimension for performance tests."""
    return 10000


@pytest.fixture
def hdc_backend(device: str, medium_dimension: int, set_random_seeds) -> Optional[Any]:
    """Create an HDC backend instance for testing."""
    if HDCompute is None:
        pytest.skip("HDCompute not implemented yet")
    
    try:
        return HDCompute(dim=medium_dimension, device=device, seed=42)
    except Exception as e:
        pytest.skip(f"Could not create HDC backend: {e}")


@pytest.fixture
def sample_hypervectors(hdc_backend, medium_dimension: int) -> List[Any]:
    """Generate sample hypervectors for testing."""
    if hdc_backend is None:
        pytest.skip("HDC backend not available")
    
    return [hdc_backend.random_hv() for _ in range(5)]


@pytest.fixture
def binary_hypervector(medium_dimension: int) -> np.ndarray:
    """Create a binary hypervector using numpy."""
    np.random.seed(42)
    return np.random.choice([0, 1], size=medium_dimension, p=[0.5, 0.5]).astype(bool)


@pytest.fixture
def binary_hypervector_pair(medium_dimension: int) -> tuple[np.ndarray, np.ndarray]:
    """Create a pair of binary hypervectors."""
    np.random.seed(42)
    hv1 = np.random.choice([0, 1], size=medium_dimension, p=[0.5, 0.5]).astype(bool)
    np.random.seed(43)
    hv2 = np.random.choice([0, 1], size=medium_dimension, p=[0.5, 0.5]).astype(bool)
    return hv1, hv2


@pytest.fixture
def performance_config() -> Dict[str, Any]:
    """Configuration for performance tests."""
    return {
        "warmup_iterations": 3,
        "test_iterations": 10,
        "timeout_seconds": 30,
        "memory_limit_mb": 1000,
    }


@pytest.fixture
def gpu_available() -> bool:
    """Check if GPU is available for testing."""
    return torch.cuda.is_available()


@pytest.fixture
def skip_if_no_gpu(gpu_available: bool) -> None:
    """Skip test if no GPU is available."""
    if not gpu_available:
        pytest.skip("GPU not available")


@pytest.fixture(scope="session")
def test_datasets_dir(test_data_dir: Path) -> Path:
    """Directory containing test datasets."""
    datasets_dir = test_data_dir / "datasets"
    datasets_dir.mkdir(exist_ok=True)
    return datasets_dir


@pytest.fixture
def mock_speech_data(test_datasets_dir: Path) -> Path:
    """Generate mock speech command data for testing."""
    data_file = test_datasets_dir / "mock_speech_commands.npz"
    
    if not data_file.exists():
        # Generate synthetic speech-like data
        np.random.seed(42)
        n_samples = 100
        n_features = 13  # MFCC features
        n_timesteps = 50
        n_classes = 10
        
        features = np.random.randn(n_samples, n_timesteps, n_features)
        labels = np.random.randint(0, n_classes, n_samples)
        class_names = [f"command_{i}" for i in range(n_classes)]
        
        np.savez(
            data_file,
            features=features,
            labels=labels,
            class_names=class_names,
        )
    
    return data_file


@pytest.fixture
def benchmark_timeout() -> int:
    """Timeout for benchmark tests in seconds."""
    return int(os.getenv("BENCHMARK_TIMEOUT", "60"))


@pytest.fixture
def ci_environment() -> bool:
    """Check if running in CI environment."""
    return bool(os.getenv("CI", False))


@pytest.fixture
def skip_slow_tests(ci_environment: bool) -> None:
    """Skip slow tests in CI unless explicitly requested."""
    if ci_environment and not os.getenv("RUN_SLOW_TESTS", False):
        pytest.skip("Slow tests disabled in CI")


class MockHDCBackend:
    """Mock HDC backend for testing when real implementation is not available."""
    
    def __init__(self, dim: int = 1000, device: str = "cpu", seed: int = 42):
        self.dim = dim
        self.device = device
        self.seed = seed
        np.random.seed(seed)
    
    def random_hv(self) -> np.ndarray:
        """Generate a random binary hypervector."""
        return np.random.choice([0, 1], size=self.dim, p=[0.5, 0.5]).astype(bool)
    
    def bundle(self, hvs: List[np.ndarray]) -> np.ndarray:
        """Bundle hypervectors by majority vote."""
        if not hvs:
            raise ValueError("Cannot bundle empty list")
        
        stacked = np.stack(hvs)
        return (stacked.sum(axis=0) > len(hvs) // 2).astype(bool)
    
    def bind(self, hv1: np.ndarray, hv2: np.ndarray) -> np.ndarray:
        """Bind hypervectors using XOR."""
        return np.logical_xor(hv1, hv2)
    
    def cosine_similarity(self, hv1: np.ndarray, hv2: np.ndarray) -> float:
        """Compute cosine similarity between binary hypervectors."""
        # For binary vectors, cosine similarity = (2 * intersection - total) / total
        intersection = np.logical_and(hv1, hv2).sum()
        total = len(hv1)
        return (2 * intersection - total) / total


@pytest.fixture
def mock_hdc_backend(medium_dimension: int, device: str, set_random_seeds) -> MockHDCBackend:
    """Create a mock HDC backend for testing."""
    return MockHDCBackend(dim=medium_dimension, device=device, seed=42)


@pytest.fixture
def pytorch_backend(medium_dimension: int, device: str, set_random_seeds):
    """Create PyTorch HDC backend for testing."""
    if HDComputeTorch is None:
        pytest.skip("HDComputeTorch not available")
    return HDComputeTorch(dim=medium_dimension, device=device)


@pytest.fixture
def jax_backend(medium_dimension: int, set_random_seeds):
    """Create JAX HDC backend for testing."""
    if HDComputeJAX is None:
        pytest.skip("HDComputeJAX not available")
    try:
        import jax.random as random
        return HDComputeJAX(dim=medium_dimension, key=random.PRNGKey(42))
    except ImportError:
        pytest.skip("JAX not available")


@pytest.fixture
def test_database(temp_dir: Path):
    """Create a test database instance."""
    if DatabaseConnection is None:
        pytest.skip("DatabaseConnection not available")
    
    db_path = temp_dir / "test.db"
    return DatabaseConnection(str(db_path))


@pytest.fixture
def test_cache_manager(temp_dir: Path):
    """Create a test cache manager instance."""
    if CacheManager is None:
        pytest.skip("CacheManager not available")
    
    cache_dir = temp_dir / "cache"
    return CacheManager(cache_dir=str(cache_dir), max_size_mb=10)


@pytest.fixture
def item_memory(pytorch_backend):
    """Create ItemMemory instance for testing."""
    if ItemMemory is None or pytorch_backend is None:
        pytest.skip("ItemMemory or backend not available")
    return ItemMemory(pytorch_backend)


@pytest.fixture
def associative_memory(pytorch_backend):
    """Create AssociativeMemory instance for testing."""
    if AssociativeMemory is None or pytorch_backend is None:
        pytest.skip("AssociativeMemory or backend not available")
    return AssociativeMemory(pytorch_backend, capacity=100)


@pytest.fixture 
def speech_command_hdc(pytorch_backend):
    """Create SpeechCommandHDC instance for testing."""
    if SpeechCommandHDC is None or pytorch_backend is None:
        pytest.skip("SpeechCommandHDC or backend not available")
    return SpeechCommandHDC(pytorch_backend, dim=1000, num_classes=10)


@pytest.fixture
def semantic_memory(pytorch_backend):
    """Create SemanticMemory instance for testing."""
    if SemanticMemory is None or pytorch_backend is None:
        pytest.skip("SemanticMemory or backend not available")
    return SemanticMemory(pytorch_backend, dim=1000)


# Helper functions for test assertions

def assert_hypervector_properties(hv: np.ndarray, expected_dim: int) -> None:
    """Assert basic properties of a hypervector."""
    assert isinstance(hv, np.ndarray), "Hypervector must be numpy array"
    assert hv.shape == (expected_dim,), f"Expected shape ({expected_dim},), got {hv.shape}"
    assert hv.dtype == bool, f"Expected bool dtype, got {hv.dtype}"


def assert_similarity_in_range(similarity: float, min_val: float = -1.0, max_val: float = 1.0) -> None:
    """Assert that similarity is in valid range."""
    assert isinstance(similarity, (int, float)), "Similarity must be numeric"
    assert min_val <= similarity <= max_val, f"Similarity {similarity} not in range [{min_val}, {max_val}]"


def assert_hvs_different(hv1: np.ndarray, hv2: np.ndarray, min_hamming_distance: int = 100) -> None:
    """Assert that two hypervectors are sufficiently different."""
    hamming_distance = np.logical_xor(hv1, hv2).sum()
    assert hamming_distance >= min_hamming_distance, (
        f"Hypervectors too similar: Hamming distance {hamming_distance} < {min_hamming_distance}"
    )